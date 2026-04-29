"""Capture screen — webcam preview with crop box and Save button."""
from __future__ import annotations

import time
from pathlib import Path

import cv2
from PyQt6.QtCore import Qt, QObject, QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QFont, QImage, QKeySequence, QPixmap, QShortcut
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .duck_detector import detect_ducks, square_crop


class CaptureWorker(QObject):
    frame_ready = pyqtSignal(QImage, object)  # display_qimg, crop_bgr (np.ndarray)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, camera_index: int):
        super().__init__()
        self.camera_index = camera_index
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    @pyqtSlot()
    def run(self) -> None:
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            self.error.emit('Cannot open camera')
            self.finished.emit()
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        try:
            while not self._stop:
                ok, frame = cap.read()
                if not ok:
                    continue
                boxes = detect_ducks(frame)
                display = frame.copy()
                for box in boxes:
                    _, (cx0, cy0, cs) = square_crop(frame, box)
                    cv2.rectangle(display, (cx0, cy0), (cx0 + cs, cy0 + cs), (0, 255, 0), 2)
                if not boxes:
                    cv2.putText(display, 'no ducks detected', (20, 50),
                                cv2.FONT_HERSHEY_DUPLEX, 0.9, (80, 80, 200), 2, cv2.LINE_AA)
                n = len(boxes)
                cv2.putText(display, f'ducks: {n}', (20, display.shape[0] - 16),
                            cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 220, 0) if n else (80, 80, 200),
                            1, cv2.LINE_AA)
                rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
                self.frame_ready.emit(qimg, (frame.copy(), boxes))
        finally:
            cap.release()
            self.finished.emit()


class CaptureScreen(QWidget):
    status_message = pyqtSignal(str, int)
    error = pyqtSignal(str)

    def __init__(self, default_out: str = 'dataset/good', default_camera: int = 0):
        super().__init__()
        self.latest_data: tuple | None = None  # (frame_bgr, list[DuckBox])
        self.worker: CaptureWorker | None = None
        self.thread: QThread | None = None
        self.flash_until = 0.0

        self._build_ui(default_out, default_camera)
        self._refresh_count()

        QShortcut(QKeySequence(Qt.Key.Key_Space), self, activated=self.on_save)

    def _build_ui(self, default_out: str, default_camera: int) -> None:
        self.video_label = QLabel('Camera idle.')
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet('background:#101010; color:#888;')

        mono = QFont('Consolas')
        mono.setStyleHint(QFont.StyleHint.Monospace)
        mono.setPointSize(11)

        out_row = QHBoxLayout()
        self.out_edit = QLineEdit(default_out)
        self.out_edit.editingFinished.connect(self._refresh_count)
        out_row.addWidget(QLabel('Output:'))
        out_row.addWidget(self.out_edit, stretch=1)

        cam_row = QHBoxLayout()
        self.cam_spin = QSpinBox()
        self.cam_spin.setRange(0, 9)
        self.cam_spin.setValue(default_camera)
        self.cam_spin.valueChanged.connect(self._restart_camera)
        cam_row.addWidget(QLabel('Camera:'))
        cam_row.addWidget(self.cam_spin)
        cam_row.addStretch(1)

        target_row = QHBoxLayout()
        self.target_spin = QSpinBox()
        self.target_spin.setRange(1, 9999)
        self.target_spin.setValue(50)
        self.target_spin.valueChanged.connect(self._update_count_label)
        target_row.addWidget(QLabel('Target count:'))
        target_row.addWidget(self.target_spin)
        target_row.addStretch(1)

        self.count_label = QLabel('saved: 0 / 50')
        self.count_label.setFont(mono)

        self.save_btn = QPushButton('Save  (Space)')
        self.save_btn.setMinimumHeight(56)
        f = self.save_btn.font()
        f.setPointSize(14)
        f.setBold(True)
        self.save_btn.setFont(f)
        self.save_btn.clicked.connect(self.on_save)

        self.hint = QLabel(
            'Place 1–3 ducks on the table.\n'
            'Vary count, position, distance between shots.\n'
            'Each detected duck is saved separately.'
        )
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet('color: #686890; font-size: 12px;')

        side = QVBoxLayout()
        side.setContentsMargins(12, 12, 12, 12)
        side.setSpacing(10)
        side.addLayout(out_row)
        side.addLayout(cam_row)
        side.addLayout(target_row)
        side.addWidget(self.count_label)
        side.addWidget(self.save_btn)
        side.addWidget(self.hint)
        side.addStretch(1)

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

    def _existing_count(self) -> int:
        out = Path(self.out_edit.text().strip() or '.')
        if not out.exists():
            return 0
        return len(list(out.glob('*.jpg')))

    def _update_count_label(self) -> None:
        self.count_label.setText(f'saved: {self._existing_count()} / {self.target_spin.value()}')

    def _refresh_count(self) -> None:
        self._update_count_label()

    def start_camera(self) -> None:
        if self.thread is not None and self.thread.isRunning():
            return
        self.worker = CaptureWorker(self.cam_spin.value())
        self.thread = QThread(self)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.error.connect(self._on_worker_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self._on_finished)
        self.thread.start()

    def stop_camera(self) -> None:
        if self.worker is not None:
            self.worker.stop()
        if self.thread is not None and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait(2000)
        self._on_finished()

    def _on_finished(self) -> None:
        self.thread = None
        self.worker = None

    def _restart_camera(self) -> None:
        if self.thread is not None and self.thread.isRunning():
            self.stop_camera()
            QTimer.singleShot(150, self.start_camera)

    @pyqtSlot(str)
    def _on_worker_error(self, msg: str) -> None:
        self.error.emit(msg)

    @pyqtSlot(QImage, object)
    def on_frame(self, qimg: QImage, data) -> None:
        self.latest_data = data
        pix = QPixmap.fromImage(qimg).scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(pix)
        self.video_label.setText('')
        if time.time() < self.flash_until:
            self.video_label.setStyleSheet('background:#fff;')
        else:
            self.video_label.setStyleSheet('background:#101010;')

    def on_save(self) -> None:
        if self.latest_data is None:
            self.status_message.emit('No frame yet.', 2000)
            return
        frame, boxes = self.latest_data
        if not boxes:
            self.status_message.emit('No ducks detected — nothing saved.', 2000)
            return
        out = Path(self.out_edit.text().strip())
        out.mkdir(parents=True, exist_ok=True)
        base_idx = len(list(out.glob('*.jpg')))
        for i, box in enumerate(boxes):
            crop, _ = square_crop(frame, box)
            fname = out / f'good_{base_idx + i:04d}.jpg'
            cv2.imwrite(str(fname), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        self.flash_until = time.time() + 0.12
        self._update_count_label()
        n = len(boxes)
        self.status_message.emit(f'saved {n} duck crop{"s" if n > 1 else ""}', 1500)
