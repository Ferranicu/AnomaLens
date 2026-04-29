"""Run screen — live PatchCore inference with heatmap overlay + score graph."""
from __future__ import annotations

import collections
import time
from pathlib import Path

import cv2
import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QFont, QImage, QPixmap
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .anomaly_store import AnomalyStore
from .heatmap import render_heatmap, render_patch_grid
from .imageio import bgr_to_tensor, center_square_view
from .patchcore import MemoryBank, PatchFeatureExtractor


HISTORY = 300
ANOMALY_COOLDOWN_S = 1.5


class InferenceWorker(QObject):
    frame_ready = pyqtSignal(QImage, float, float)
    fps_update = pyqtSignal(float)
    error = pyqtSignal(str)
    finished = pyqtSignal()
    anomaly_detected = pyqtSignal(object, object, float, float, float)
    # full_bgr, zoom_bgr, score, threshold, ts

    def __init__(
        self,
        extractor: PatchFeatureExtractor,
        bank: MemoryBank,
        device,
        camera_index: int,
        ema: float,
        blend: float,
        threshold: float,
    ):
        super().__init__()
        self.extractor = extractor
        self.bank = bank
        self.device = device
        self.camera_index = camera_index
        self.ema = float(ema)
        self.blend = float(blend)
        self.threshold = float(threshold)
        self._stop = False
        self._show_heat = True
        self._show_grid = False

    def stop(self) -> None:
        self._stop = True

    def set_show_heat(self, on: bool) -> None:
        self._show_heat = bool(on)

    def set_show_grid(self, on: bool) -> None:
        self._show_grid = bool(on)

    def set_threshold(self, t: float) -> None:
        self.threshold = float(t)

    @pyqtSlot()
    def run(self) -> None:
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            self.error.emit('Cannot open camera')
            self.finished.emit()
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        ema_score: float | None = None
        fps_acc_t = time.time()
        fps_frames = 0
        was_anom = False
        last_anom_t = 0.0

        try:
            while not self._stop:
                ok, frame = cap.read()
                if not ok:
                    continue

                crop, (cx0, cy0, cs) = center_square_view(frame)
                x = bgr_to_tensor(crop, self.device)
                flat, (B, H, W) = self.extractor.embed(x)
                scores = self.bank.score(flat).view(H, W)
                max_score = float(scores.max().item())
                score_map_np = scores.cpu().numpy()

                if ema_score is None:
                    ema_score = max_score
                else:
                    ema_score = self.ema * ema_score + (1.0 - self.ema) * max_score

                display = frame.copy()
                if self._show_heat:
                    heat = render_heatmap(score_map_np, cs, vmax=self.threshold)
                    local = display[cy0:cy0 + cs, cx0:cx0 + cs]
                    blended = cv2.addWeighted(local, 1.0 - self.blend, heat, self.blend, 0.0)
                    display[cy0:cy0 + cs, cx0:cx0 + cs] = blended
                if self._show_grid:
                    grid = render_patch_grid(score_map_np, cs, vmax=self.threshold)
                    local = display[cy0:cy0 + cs, cx0:cx0 + cs]
                    blended = cv2.addWeighted(local, 1.0 - self.blend, grid, self.blend, 0.0)
                    display[cy0:cy0 + cs, cx0:cx0 + cs] = blended
                cv2.rectangle(display, (cx0, cy0), (cx0 + cs, cy0 + cs), (255, 255, 255), 2)

                now = time.time()
                is_anom = ema_score > self.threshold
                if (
                    is_anom
                    and not was_anom
                    and (now - last_anom_t) > ANOMALY_COOLDOWN_S
                ):
                    zoom_bgr = self._extract_zoom(display, score_map_np, cx0, cy0, cs, H, W)
                    self.anomaly_detected.emit(
                        display.copy(), zoom_bgr,
                        float(ema_score), float(self.threshold), now,
                    )
                    last_anom_t = now
                was_anom = is_anom

                rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()

                self.frame_ready.emit(qimg, max_score, float(ema_score))

                fps_frames += 1
                if now - fps_acc_t >= 0.5:
                    self.fps_update.emit(fps_frames / (now - fps_acc_t))
                    fps_acc_t = now
                    fps_frames = 0
        finally:
            cap.release()
            self.finished.emit()

    @staticmethod
    def _extract_zoom(
        display: np.ndarray,
        score_map: np.ndarray,
        cx0: int, cy0: int, cs: int,
        gh: int, gw: int,
    ) -> np.ndarray:
        """Return a square crop of `display` centered on the score-map peak."""
        gy, gx = np.unravel_index(int(np.argmax(score_map)), score_map.shape)
        peak_x = int(cx0 + cs * (gx + 0.5) / gw)
        peak_y = int(cy0 + cs * (gy + 0.5) / gh)
        zs = max(72, cs // 3)
        half = zs // 2
        h, w = display.shape[:2]
        x0 = max(0, min(w - zs, peak_x - half))
        y0 = max(0, min(h - zs, peak_y - half))
        return display[y0:y0 + zs, x0:x0 + zs].copy()


class ScoreGraph(pg.PlotWidget):
    def __init__(self, threshold: float):
        super().__init__()
        self.setBackground('#1a1a24')
        self.setMouseEnabled(False, False)
        self.hideButtons()
        self.setMenuEnabled(False)
        self.showGrid(x=False, y=True, alpha=0.08)
        self.getAxis('left').setTextPen(pg.mkPen('#686888'))
        self.getAxis('bottom').setTextPen(pg.mkPen('#686888'))
        self.getAxis('left').setPen(pg.mkPen('#2e2e42'))
        self.getAxis('bottom').setPen(pg.mkPen('#2e2e42'))
        self.setLabel('left', 'score', color='#686888')
        self.setLabel('bottom', 'frame', color='#686888')

        self.raw_data: collections.deque[float] = collections.deque(maxlen=HISTORY)
        self.ema_data: collections.deque[float] = collections.deque(maxlen=HISTORY)

        self.raw_curve = self.plot(pen=pg.mkPen(color=(60, 80, 160), width=1))
        self.ema_curve = self.plot(pen=pg.mkPen(color=(100, 160, 255), width=2))
        self.thr_line = pg.InfiniteLine(
            angle=0,
            pos=threshold,
            pen=pg.mkPen(color=(220, 50, 50), width=2, style=Qt.PenStyle.DashLine),
        )
        self.addItem(self.thr_line)

        self.setYRange(0, max(threshold * 3.0, 2.0))
        self.enableAutoRange(axis='y')
        self.setLimits(yMin=0)

    def append(self, raw: float, ema: float) -> None:
        self.raw_data.append(raw)
        self.ema_data.append(ema)
        xs = np.arange(len(self.raw_data))
        self.raw_curve.setData(xs, np.fromiter(self.raw_data, dtype=float))
        self.ema_curve.setData(xs, np.fromiter(self.ema_data, dtype=float))

    def reset(self) -> None:
        self.raw_data.clear()
        self.ema_data.clear()
        self.raw_curve.setData([], [])
        self.ema_curve.setData([], [])

    def set_threshold(self, t: float) -> None:
        self.thr_line.setValue(t)


class RunScreen(QWidget):
    status_message = pyqtSignal(str, int)
    error = pyqtSignal(str)

    def __init__(
        self,
        extractor: PatchFeatureExtractor,
        device,
        store: AnomalyStore,
        default_bank: str = 'models/bank.pt',
        default_camera: int = 0,
        ema: float = 0.4,
        blend: float = 0.5,
    ):
        super().__init__()
        self.extractor = extractor
        self.device = device
        self.store = store
        self.ema = ema
        self.blend = blend

        self.bank: MemoryBank | None = None
        self.threshold = 1.0
        self.last_qimg: QImage | None = None
        self.worker: InferenceWorker | None = None
        self.thread: QThread | None = None

        self._build_ui(default_bank, default_camera)

        if Path(default_bank).exists():
            self._load_bank(default_bank)

    def _build_ui(self, default_bank: str, default_camera: int) -> None:
        self.video_label = QLabel('No bank loaded.\nTrain a bank or pick an existing one.')
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet('background:#101010; color:#888; font-size:14px;')

        mono = QFont('Consolas')
        mono.setStyleHint(QFont.StyleHint.Monospace)
        mono.setPointSize(11)

        bank_row = QHBoxLayout()
        self.bank_edit = QLineEdit(default_bank)
        self.load_btn = QPushButton('Load')
        self.load_btn.clicked.connect(self.on_load_clicked)
        bank_row.addWidget(QLabel('Bank:'))
        bank_row.addWidget(self.bank_edit, stretch=1)
        bank_row.addWidget(self.load_btn)

        cam_row = QHBoxLayout()
        self.cam_spin = QSpinBox()
        self.cam_spin.setRange(0, 9)
        self.cam_spin.setValue(default_camera)
        self.start_btn = QPushButton('Start')
        self.start_btn.clicked.connect(self.on_start_clicked)
        self.stop_btn = QPushButton('Stop')
        self.stop_btn.clicked.connect(self.stop_inference)
        self.stop_btn.setEnabled(False)
        cam_row.addWidget(QLabel('Camera:'))
        cam_row.addWidget(self.cam_spin)
        cam_row.addStretch(1)
        cam_row.addWidget(self.start_btn)
        cam_row.addWidget(self.stop_btn)

        self.score_label = QLabel('raw:  --\nema:  --')
        self.score_label.setFont(mono)
        self.score_label.setStyleSheet('color: #9090c0; background: #1e1e2a; padding: 6px 8px; border-radius: 4px;')

        self.threshold_label = QLabel(f'threshold: {self.threshold:.3f}')
        self.threshold_label.setFont(mono)
        self.threshold_label.setStyleSheet('color: #7070a8;')

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 5000)
        self.slider.setValue(int(self.threshold * 1000))
        self.slider.setSingleStep(10)
        self.slider.setPageStep(50)
        self.slider.valueChanged.connect(self.on_slider)

        self.graph = ScoreGraph(self.threshold)
        self.graph.setMinimumHeight(220)

        self.heat_btn = QPushButton('Heatmap: ON')
        self.heat_btn.setCheckable(True)
        self.heat_btn.setChecked(True)
        self.heat_btn.toggled.connect(self.on_heat_toggle)

        self.grid_btn = QPushButton('Patch grid: OFF')
        self.grid_btn.setCheckable(True)
        self.grid_btn.setChecked(False)
        self.grid_btn.toggled.connect(self.on_grid_toggle)

        self.snap_btn = QPushButton('Save snapshot')
        self.snap_btn.clicked.connect(self.on_snap)

        overlay_row = QHBoxLayout()
        overlay_row.addWidget(self.heat_btn, stretch=1)
        overlay_row.addWidget(self.grid_btn, stretch=1)

        side = QVBoxLayout()
        side.setContentsMargins(12, 12, 12, 12)
        side.setSpacing(10)
        side.addLayout(bank_row)
        side.addLayout(cam_row)
        side.addWidget(self.score_label)
        side.addWidget(self.threshold_label)
        side.addWidget(self.slider)
        side.addWidget(self.graph, stretch=1)
        side.addLayout(overlay_row)
        side.addWidget(self.snap_btn)

        side_widget = QWidget()
        side_widget.setLayout(side)
        side_widget.setFixedWidth(360)
        side_widget.setStyleSheet('background: #16161e; border-left: 1px solid #22222e;')

        root = QHBoxLayout()
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self.video_label, stretch=1)
        root.addWidget(side_widget)
        self.setLayout(root)

    def on_load_clicked(self) -> None:
        path = self.bank_edit.text().strip()
        if not Path(path).exists():
            self.error.emit(f'Bank not found: {path}')
            return
        self._load_bank(path)

    def _load_bank(self, path: str) -> None:
        try:
            bank, meta = MemoryBank.load(path, self.device)
        except Exception as e:
            self.error.emit(f'Failed to load bank: {e}')
            return
        self.bank = bank
        self.threshold = float(meta.get('threshold', 1.0))
        slider_max = max(int(self.threshold * 4 * 1000), 5000)
        self.slider.setRange(0, slider_max)
        self.slider.setValue(int(self.threshold * 1000))
        self.threshold_label.setText(f'threshold: {self.threshold:.3f}')
        self.graph.set_threshold(self.threshold)
        self.status_message.emit(
            f'bank loaded: {bank.features.shape[0]} patches   threshold {self.threshold:.3f}',
            5000,
        )
        if self.video_label.text():  # placeholder text still showing
            self.video_label.setText('Press Start to begin live inference.')

    def on_start_clicked(self) -> None:
        if self.bank is None:
            self.error.emit('Load a bank before starting inference.')
            return
        self.start_inference()

    def start_inference(self) -> None:
        if self.thread is not None and self.thread.isRunning():
            return
        if self.bank is None:
            return
        self.graph.reset()
        self.worker = InferenceWorker(
            self.extractor, self.bank, self.device,
            self.cam_spin.value(), self.ema, self.blend,
            self.threshold,
        )
        self.worker.set_show_heat(self.heat_btn.isChecked())
        self.worker.set_show_grid(self.grid_btn.isChecked())
        self.thread = QThread(self)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.fps_update.connect(self.on_fps)
        self.worker.error.connect(self._on_worker_error)
        self.worker.anomaly_detected.connect(self.on_anomaly)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self._on_finished)
        self.thread.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.load_btn.setEnabled(False)
        self.cam_spin.setEnabled(False)

    def stop_inference(self) -> None:
        if self.worker is not None:
            self.worker.stop()
        if self.thread is not None and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait(2000)
        self._on_finished()

    def _on_finished(self) -> None:
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.load_btn.setEnabled(True)
        self.cam_spin.setEnabled(True)
        self.thread = None
        self.worker = None

    @pyqtSlot(str)
    def _on_worker_error(self, msg: str) -> None:
        self.error.emit(msg)

    @pyqtSlot(QImage, float, float)
    def on_frame(self, qimg: QImage, raw: float, ema: float) -> None:
        self.last_qimg = qimg
        pix = QPixmap.fromImage(qimg).scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(pix)
        self.video_label.setText('')
        self.score_label.setText(f'raw:  {raw:6.3f}\nema:  {ema:6.3f}')
        self.graph.append(raw, ema)

    @pyqtSlot(float)
    def on_fps(self, fps: float) -> None:
        self.status_message.emit(f'fps: {fps:4.1f}', 2000)

    def on_slider(self, value: int) -> None:
        self.threshold = value / 1000.0
        self.threshold_label.setText(f'threshold: {self.threshold:.3f}')
        self.graph.set_threshold(self.threshold)
        if self.worker is not None:
            self.worker.set_threshold(self.threshold)

    @pyqtSlot(object, object, float, float, float)
    def on_anomaly(self, full_bgr, zoom_bgr, score: float, threshold: float, ts: float) -> None:
        try:
            self.store.add(full_bgr, zoom_bgr, score, threshold, ts)
        except Exception as e:
            self.error.emit(f'failed to log anomaly: {e}')
            return
        self.status_message.emit(f'anomaly logged  (score {score:.3f})', 2000)

    def on_heat_toggle(self, checked: bool) -> None:
        if self.worker is not None:
            self.worker.set_show_heat(checked)
        self.heat_btn.setText(f'Heatmap: {"ON" if checked else "OFF"}')

    def on_grid_toggle(self, checked: bool) -> None:
        if self.worker is not None:
            self.worker.set_show_grid(checked)
        self.grid_btn.setText(f'Patch grid: {"ON" if checked else "OFF"}')

    def on_snap(self) -> None:
        if self.last_qimg is None:
            return
        snap_dir = Path('snapshots')
        snap_dir.mkdir(parents=True, exist_ok=True)
        fname = snap_dir / f'qt_{int(time.time())}.jpg'
        self.last_qimg.save(str(fname), 'JPG', 92)
        self.status_message.emit(f'saved {fname}', 3000)
