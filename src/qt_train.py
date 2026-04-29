"""Train screen — build a PatchCore memory bank from captured good images."""
from __future__ import annotations

import statistics
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QFont, QImage, QPixmap
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .heatmap import render_patch_grid
from .imageio import bgr_to_tensor
from .patchcore import INPUT_SIZE, MemoryBank, PatchFeatureExtractor, coreset_subsample


class TrainerWorker(QObject):
    progress = pyqtSignal(int, str)
    log = pyqtSignal(str)
    done = pyqtSignal(bool, str, str)  # success, message, bank_path
    viz_frame = pyqtSignal(object, float, float)  # bgr_ndarray, score_min, score_max

    def __init__(
        self,
        extractor: PatchFeatureExtractor,
        device,
        data_dir: str,
        out_path: str,
        ratio: float,
        batch: int,
    ):
        super().__init__()
        self.extractor = extractor
        self.device = device
        self.data_dir = data_dir
        self.out_path = out_path
        self.ratio = ratio
        self.batch = batch
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def _emit_viz(self, img_file: Path, score_map: np.ndarray) -> None:
        img = cv2.imread(str(img_file))
        if img is None:
            return
        h, w = img.shape[:2]
        s = min(h, w)
        crop = img[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
        crop = cv2.resize(crop, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
        grid = render_patch_grid(score_map, INPUT_SIZE, vmax=None)
        blended = cv2.addWeighted(crop, 0.45, grid, 0.55, 0.0)
        self.viz_frame.emit(blended, float(score_map.min()), float(score_map.max()))

    def _load_batch(self, files: list[Path]) -> torch.Tensor | None:
        tensors = []
        for f in files:
            img = cv2.imread(str(f))
            if img is None:
                self.log.emit(f'skip unreadable: {f}')
                continue
            tensors.append(bgr_to_tensor(img, self.device))
        if not tensors:
            return None
        return torch.cat(tensors, dim=0)

    @pyqtSlot()
    def run(self) -> None:
        try:
            data_dir = Path(self.data_dir)
            files = sorted(list(data_dir.glob('*.jpg')) + list(data_dir.glob('*.png')))
            if not files:
                self.done.emit(False, f'no images in {data_dir}', '')
                return

            n = len(files)
            self.log.emit(f'loading {n} images from {data_dir}')

            all_feats: list[torch.Tensor] = []
            for i in range(0, n, self.batch):
                if self._stop:
                    self.done.emit(False, 'cancelled', '')
                    return
                batch_files = files[i:i + self.batch]
                x = self._load_batch(batch_files)
                if x is None:
                    continue
                flat, _ = self.extractor.embed(x)
                all_feats.append(flat.cpu())
                done = i + len(batch_files)
                self.progress.emit(int(done / n * 50), f'extract {done}/{n}')

            if not all_feats:
                self.done.emit(False, 'no usable images', '')
                return

            feats = torch.cat(all_feats, dim=0)
            self.log.emit(f'patch embeddings: {tuple(feats.shape)}')

            self.progress.emit(55, f'coreset subsample (ratio={self.ratio})...')
            feats_dev = feats.to(self.device)
            bank_t = coreset_subsample(feats_dev, ratio=self.ratio)
            self.log.emit(f'coreset: {tuple(bank_t.shape)}')

            mem = MemoryBank(bank_t, self.device)

            per_image_max: list[float] = []
            for i in range(0, n, self.batch):
                if self._stop:
                    self.done.emit(False, 'cancelled', '')
                    return
                batch_files = files[i:i + self.batch]
                x = self._load_batch(batch_files)
                if x is None:
                    continue
                flat, (B, H, W) = self.extractor.embed(x)
                scores = mem.score(flat).view(B, H, W)
                per_image_max.extend(scores.amax(dim=(1, 2)).cpu().tolist())
                done = i + len(batch_files)
                self.progress.emit(60 + int(done / n * 35), f'calibrate {done}/{n}')
                self._emit_viz(batch_files[-1], scores[-1].cpu().numpy())

            good_mean = statistics.fmean(per_image_max)
            good_max = max(per_image_max)
            default_thresh = good_max * 2.0
            self.log.emit(
                f'good_max={good_max:.3f}  good_mean={good_mean:.3f}  '
                f'threshold={default_thresh:.3f}'
            )

            out_path = Path(self.out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            mem.save(str(out_path), meta={
                'threshold': default_thresh,
                'good_max': good_max,
                'good_mean': good_mean,
                'n_train': n,
            })
            self.progress.emit(100, 'saved')
            self.log.emit(f'saved {out_path}')
            self.done.emit(True, f'saved {out_path}  (threshold {default_thresh:.3f})', str(out_path))
        except Exception as e:
            self.log.emit(traceback.format_exc())
            self.done.emit(False, str(e), '')


class TrainScreen(QWidget):
    status_message = pyqtSignal(str, int)
    error = pyqtSignal(str)
    bank_saved = pyqtSignal(str)

    def __init__(
        self,
        extractor: PatchFeatureExtractor,
        device,
        default_data: str = 'dataset/good',
        default_out: str = 'models/bank.pt',
    ):
        super().__init__()
        self.extractor = extractor
        self.device = device
        self.worker: TrainerWorker | None = None
        self.thread: QThread | None = None

        self._build_ui(default_data, default_out)

    def _build_ui(self, default_data: str, default_out: str) -> None:
        mono = QFont('Consolas')
        mono.setStyleHint(QFont.StyleHint.Monospace)
        mono.setPointSize(10)

        data_row = QHBoxLayout()
        self.data_edit = QLineEdit(default_data)
        data_row.addWidget(QLabel('Dataset:'))
        data_row.addWidget(self.data_edit, stretch=1)

        out_row = QHBoxLayout()
        self.out_edit = QLineEdit(default_out)
        out_row.addWidget(QLabel('Output:'))
        out_row.addWidget(self.out_edit, stretch=1)

        params_row = QHBoxLayout()
        self.ratio_spin = QDoubleSpinBox()
        self.ratio_spin.setRange(0.01, 1.0)
        self.ratio_spin.setSingleStep(0.05)
        self.ratio_spin.setDecimals(2)
        self.ratio_spin.setValue(0.10)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 64)
        self.batch_spin.setValue(8)
        params_row.addWidget(QLabel('Coreset ratio:'))
        params_row.addWidget(self.ratio_spin)
        params_row.addSpacing(20)
        params_row.addWidget(QLabel('Batch:'))
        params_row.addWidget(self.batch_spin)
        params_row.addStretch(1)

        btn_row = QHBoxLayout()
        self.train_btn = QPushButton('Train')
        self.train_btn.setMinimumHeight(40)
        f = self.train_btn.font()
        f.setBold(True)
        self.train_btn.setFont(f)
        self.train_btn.clicked.connect(self.on_train_clicked)
        self.cancel_btn = QPushButton('Cancel')
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.on_cancel_clicked)
        btn_row.addWidget(self.train_btn, stretch=1)
        btn_row.addWidget(self.cancel_btn)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFont(mono)
        self.log_view.setPlaceholderText('Training log will appear here.')

        # Viz panel — shows patch grid on calibration images (toggleable)
        VIZ_SIZE = 256
        self.viz_btn = QPushButton('Calibration viz: ON')
        self.viz_btn.setCheckable(True)
        self.viz_btn.setChecked(True)
        self.viz_btn.toggled.connect(self._on_viz_toggle)

        self.viz_label = QLabel()
        self.viz_label.setFixedSize(VIZ_SIZE, VIZ_SIZE)
        self.viz_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.viz_label.setStyleSheet(
            'background: #0e0e14; border: 1px solid #2a2a3a; color: #555;'
        )
        self.viz_label.setText('patch grid\nappears during\ncalibration')

        self.viz_score_label = QLabel()
        self.viz_score_label.setFont(mono)
        self.viz_score_label.setStyleSheet('color: #7070a8; font-size: 9pt;')
        self.viz_score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        viz_col = QVBoxLayout()
        viz_col.setSpacing(6)
        viz_col.addWidget(self.viz_btn)
        viz_col.addWidget(self.viz_label)
        viz_col.addWidget(self.viz_score_label)
        viz_col.addStretch(1)

        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(12)
        bottom_row.addWidget(self.log_view, stretch=1)
        bottom_row.addLayout(viz_col)

        root = QVBoxLayout()
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(10)
        root.addLayout(data_row)
        root.addLayout(out_row)
        root.addLayout(params_row)
        root.addLayout(btn_row)
        root.addWidget(self.progress)
        root.addLayout(bottom_row, stretch=1)
        self.setLayout(root)

    def on_train_clicked(self) -> None:
        if self.thread is not None and self.thread.isRunning():
            return
        data_dir = self.data_edit.text().strip()
        out_path = self.out_edit.text().strip()
        if not Path(data_dir).exists():
            self.error.emit(f'Dataset folder not found: {data_dir}')
            return

        self.log_view.clear()
        self.progress.setValue(0)
        self.progress.setFormat('%p%')

        self.worker = TrainerWorker(
            self.extractor, self.device,
            data_dir, out_path,
            float(self.ratio_spin.value()),
            int(self.batch_spin.value()),
        )
        self.thread = QThread(self)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.on_progress)
        self.worker.log.connect(self.on_log)
        self.worker.done.connect(self.on_done)
        self.worker.done.connect(self.thread.quit)
        self.worker.viz_frame.connect(self.on_viz_frame)
        self.thread.start()

        self.train_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.data_edit.setEnabled(False)
        self.out_edit.setEnabled(False)
        self.ratio_spin.setEnabled(False)
        self.batch_spin.setEnabled(False)

    def on_cancel_clicked(self) -> None:
        if self.worker is not None:
            self.worker.stop()
        self.cancel_btn.setEnabled(False)

    @pyqtSlot(int, str)
    def on_progress(self, pct: int, msg: str) -> None:
        self.progress.setValue(pct)
        self.progress.setFormat(f'{msg}  —  %p%')

    @pyqtSlot(str)
    def on_log(self, line: str) -> None:
        self.log_view.appendPlainText(line)

    @pyqtSlot(object, float, float)
    def on_viz_frame(self, bgr: np.ndarray, score_min: float, score_max: float) -> None:
        if not self.viz_btn.isChecked():
            return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg).scaled(
            self.viz_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
        self.viz_label.setPixmap(pix)
        self.viz_label.setText('')
        self.viz_score_label.setText(f'score  min {score_min:.3f}  max {score_max:.3f}')

    def _on_viz_toggle(self, checked: bool) -> None:
        self.viz_btn.setText(f'Calibration viz: {"ON" if checked else "OFF"}')
        if not checked:
            self.viz_label.clear()
            self.viz_label.setText('patch grid\nappears during\ncalibration')
            self.viz_score_label.setText('')

    @pyqtSlot(bool, str, str)
    def on_done(self, success: bool, msg: str, bank_path: str) -> None:
        self.train_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.data_edit.setEnabled(True)
        self.out_edit.setEnabled(True)
        self.ratio_spin.setEnabled(True)
        self.batch_spin.setEnabled(True)
        self.thread = None
        self.worker = None

        if success:
            self.status_message.emit(msg, 5000)
            self.bank_saved.emit(bank_path)
        else:
            self.progress.setFormat(f'failed: {msg}')
            self.status_message.emit(f'training failed: {msg}', 5000)
