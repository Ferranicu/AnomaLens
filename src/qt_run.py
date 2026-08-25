"""Run screen — live PatchCore inference, fair-mode UI.

Right panel: anomaly-only live zoom cards, hidden when the frame is clean.
Controls: edge-reveal bottom drawer with slide/fade animation.
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import (
    QEasingCurve, QEvent, QRect, Qt, QObject, QParallelAnimationGroup, QPropertyAnimation,
    QThread, pyqtSignal, pyqtSlot,
)
from PyQt6.QtGui import QCursor, QFont, QImage, QPixmap
from PyQt6.QtWidgets import (
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .anomaly_store import AnomalyStore
from .duck_detector import detect_ducks, square_crop
from .heatmap import render_heatmap, render_patch_grid
from .imageio import bgr_to_tensor
from .patchcore import MemoryBank, PatchFeatureExtractor


ANOMALY_COOLDOWN_S = 1.5
DUCK_CARD_ZOOM = 224     # px — square zoom image per duck card
PANEL_WIDTH = 264        # right panel width
CONTROLS_EDGE_PX = 16
CONTROLS_AWAY_PX = 72
MAX_DUCK_SLOTS = 4
_WORKER_ZOOM = 256       # intermediate resolution from worker


class InferenceWorker(QObject):
    # main_qimg, anomaly_data list[(QImage, score)], raw, ema, has_duck
    frame_ready = pyqtSignal(QImage, object, float, float, bool)
    fps_update = pyqtSignal(float)
    error = pyqtSignal(str)
    finished = pyqtSignal()
    anomaly_detected = pyqtSignal(object, object, float, float, float)

    def __init__(self, extractor, bank, device, camera_index, ema, blend, threshold):
        super().__init__()
        self.extractor = extractor
        self.bank = bank
        self.device = device
        self.camera_index = camera_index
        self.ema = float(ema)
        self.blend = float(blend)
        self.threshold = float(threshold)
        self._stop = False
        self._show_heat = False
        self._show_grid = False

    def stop(self) -> None: self._stop = True
    def set_show_heat(self, on: bool) -> None: self._show_heat = bool(on)
    def set_show_grid(self, on: bool) -> None: self._show_grid = bool(on)
    def set_threshold(self, t: float) -> None: self.threshold = float(t)

    def _bgr_qimg(self, bgr: np.ndarray) -> QImage:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        return QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()

    def _peak_zoom(self, bgr: np.ndarray, crop_box: tuple[int, int, int], score_map: np.ndarray) -> np.ndarray:
        x0, y0, x1, y1 = self._peak_square(crop_box, score_map, scale=5)
        marked = bgr.copy()
        cv2.rectangle(marked, (x0, y0), (x1, y1), (35, 35, 255), 3)

        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2
        _cx0, _cy0, cs = crop_box
        H, W = marked.shape[:2]
        size = max(48, int(cs * 0.42))
        size = min(size, H, W)
        half = size // 2
        zx0 = max(0, min(W - size, cx - half))
        zy0 = max(0, min(H - size, cy - half))
        return marked[zy0:zy0 + size, zx0:zx0 + size].copy()

    def _peak_square(
        self,
        crop_box: tuple[int, int, int],
        score_map: np.ndarray,
        scale: int = 3,
    ) -> tuple[int, int, int, int]:
        cx0, cy0, cs = crop_box
        map_h, map_w = score_map.shape
        py, px = np.unravel_index(int(np.argmax(score_map)), score_map.shape)
        cell = max(8, int(round(cs / max(map_h, map_w, 1))))
        size = min(cs, max(cell * scale, 24))
        cx = cx0 + int((px + 0.5) * cs / max(map_w, 1))
        cy = cy0 + int((py + 0.5) * cs / max(map_h, 1))
        half = size // 2
        x0 = max(cx0, min(cx0 + cs - size, cx - half))
        y0 = max(cy0, min(cy0 + cs - size, cy - half))
        return x0, y0, x0 + size, y0 + size

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

                now = time.time()
                display = frame.copy()
                boxes = detect_ducks(frame)
                raw_score = 0.0
                anomaly_data: list[tuple[QImage, float, int]] = []
                is_anom = False

                if boxes:
                    duck_results = []
                    for box in boxes:
                        crop, (cx0, cy0, cs) = square_crop(frame, box)
                        x = bgr_to_tensor(crop, self.device)
                        flat, (B, H, W) = self.extractor.embed(x)
                        score_map = self.bank.score(flat).view(H, W).cpu().numpy()
                        duck_results.append((box, (cx0, cy0, cs), score_map))

                    raw_score = max(float(sm.max()) for _, _, sm in duck_results)
                    if ema_score is None:
                        ema_score = raw_score
                    else:
                        ema_score = self.ema * ema_score + (1.0 - self.ema) * raw_score
                    is_anom = ema_score > self.threshold

                    for _, (cx0, cy0, cs), score_map in duck_results:
                        if self._show_heat:
                            heat = render_heatmap(score_map, cs, vmax=self.threshold)
                            local = display[cy0:cy0 + cs, cx0:cx0 + cs]
                            display[cy0:cy0 + cs, cx0:cx0 + cs] = cv2.addWeighted(
                                local, 1.0 - self.blend, heat, self.blend, 0.0)
                        if self._show_grid:
                            grid = render_patch_grid(score_map, cs, vmax=self.threshold)
                            local = display[cy0:cy0 + cs, cx0:cx0 + cs]
                            display[cy0:cy0 + cs, cx0:cx0 + cs] = cv2.addWeighted(
                                local, 1.0 - self.blend, grid, self.blend, 0.0)
                        duck_max = float(score_map.max())
                        col = (50, 50, 255) if duck_max > self.threshold else (60, 180, 80)
                        cv2.rectangle(display, (cx0, cy0), (cx0 + cs, cy0 + cs), col, 3)
                        cv2.putText(display, f'{duck_max:.2f}', (cx0 + 4, cy0 + cs - 8),
                                    cv2.FONT_HERSHEY_DUPLEX, 0.55, col, 1, cv2.LINE_AA)
                        if duck_max > self.threshold:
                            peak_crop = self._peak_zoom(frame, (cx0, cy0, cs), score_map)
                            zoomed = cv2.resize(peak_crop, (_WORKER_ZOOM, _WORKER_ZOOM),
                                                interpolation=cv2.INTER_LINEAR)
                            anomaly_data.append((self._bgr_qimg(zoomed), duck_max, cx0 + cs // 2))

                    if is_anom and not was_anom and (now - last_anom_t) > ANOMALY_COOLDOWN_S:
                        worst = max(duck_results, key=lambda t: t[2].max())
                        _, worst_crop_box, worst_score_map = worst
                        zoom_bgr = self._peak_zoom(frame, worst_crop_box, worst_score_map)
                        self.anomaly_detected.emit(
                            display.copy(), zoom_bgr,
                            float(ema_score), float(self.threshold), now,
                        )
                        last_anom_t = now
                    was_anom = is_anom
                else:
                    cv2.putText(display, 'no ducks detected', (20, 50),
                                cv2.FONT_HERSHEY_DUPLEX, 0.9, (80, 80, 200), 2, cv2.LINE_AA)
                    was_anom = False

                ema_out = ema_score if ema_score is not None else 0.0
                anomaly_data.sort(key=lambda item: item[2])
                self.frame_ready.emit(
                    self._bgr_qimg(display),
                    [(qimg, score) for qimg, score, _center_x in anomaly_data],
                    raw_score,
                    ema_out,
                    bool(boxes),
                )

                fps_frames += 1
                if now - fps_acc_t >= 0.5:
                    self.fps_update.emit(fps_frames / (now - fps_acc_t))
                    fps_acc_t = now
                    fps_frames = 0
        finally:
            cap.release()
            self.finished.emit()


class DuckCard(QWidget):
    """Image-only card for a detected anomaly; always styled as an alert."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('DuckCard')
        self._build()

    def _build(self) -> None:
        self._zoom = QLabel()
        self._zoom.setFixedSize(DUCK_CARD_ZOOM, DUCK_CARD_ZOOM)
        self._zoom.setAlignment(Qt.AlignmentFlag.AlignCenter)

        lay = QVBoxLayout()
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(0)
        lay.addWidget(self._zoom, alignment=Qt.AlignmentFlag.AlignHCenter)
        self.setLayout(lay)
        self._apply_alert_style()

    def update_duck(self, qimg: QImage) -> None:
        pix = QPixmap.fromImage(qimg).scaled(
            self._zoom.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._zoom.setPixmap(pix)
        self._zoom.setText('')
        self.show()

    def clear(self) -> None:
        self.hide()

    def _apply_alert_style(self) -> None:
        self.setStyleSheet(
            'QWidget#DuckCard {'
            '  background:#171114; border:1px solid #6b2a30; border-radius:6px;'
            '}'
        )
        self._zoom.setStyleSheet(
            'background:#09070a; border:1px solid #3f171c; border-radius:4px;'
        )


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

        self._info_panel_visible = False
        self._controls_visible = False

        self._build_ui(default_bank, default_camera)
        self._setup_animation()
        self.setMouseTracking(True)

        if Path(default_bank).exists():
            self.load_bank(default_bank)

    # ── Build UI ─────────────────────────────────────────────────────────

    def _build_ui(self, default_bank: str, default_camera: int) -> None:
        self.video_label = QLabel('No bank loaded.\nTrain or load a bank, then press Start.')
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet('background:#06060c; color:#44445a; font-size:14px;')
        self.video_label.setMouseTracking(True)
        self.video_label.installEventFilter(self)

        self.info_panel = self._build_info_panel()

        root = QHBoxLayout()
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self.video_label, stretch=1)
        root.addWidget(self.info_panel)
        self.setLayout(root)

        self.controls_overlay = self._build_controls_overlay(default_bank, default_camera)
        self.controls_overlay.hide()

    def _build_info_panel(self) -> QWidget:
        self._duck_cards: list[DuckCard] = []
        cards_widget = QWidget()
        cards_widget.setStyleSheet('background:transparent;')
        cards_lay = QVBoxLayout(cards_widget)
        cards_lay.setContentsMargins(10, 10, 10, 10)
        cards_lay.setSpacing(10)
        for _ in range(MAX_DUCK_SLOTS):
            card = DuckCard()
            card.hide()
            cards_lay.addWidget(card)
            self._duck_cards.append(card)
        cards_lay.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidget(cards_widget)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(
            'QScrollArea { border:none; background:transparent; }'
            'QScrollBar:vertical { background:#0c0d13; width:5px; border-radius:2px; }'
            'QScrollBar::handle:vertical { background:#3a242a; border-radius:2px; }'
            'QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0; }'
        )

        panel_lay = QVBoxLayout()
        panel_lay.setContentsMargins(0, 0, 0, 0)
        panel_lay.setSpacing(0)
        panel_lay.addWidget(scroll, stretch=1)

        panel = QWidget()
        panel.setLayout(panel_lay)
        panel.setMinimumWidth(0)
        panel.setMaximumWidth(0)
        panel.setStyleSheet(
            'background:#0c0d13; border-left:1px solid #352027;'
        )
        panel.hide()
        return panel

    def _build_controls_overlay(self, default_bank: str, default_camera: int) -> QWidget:
        overlay = QWidget(self)
        overlay.setObjectName('ControlsOverlay')
        overlay.setStyleSheet(
            'QWidget#ControlsOverlay { background:#10111a; border-top:1px solid #303448; }'
            'QLabel { color:#8c90ad; background:transparent; }'
        )
        overlay.setMouseTracking(True)
        overlay.installEventFilter(self)

        mono = QFont('Consolas', 10)
        mono.setStyleHint(QFont.StyleHint.Monospace)

        bank_row = QHBoxLayout()
        self.bank_edit = QLineEdit(default_bank)
        self.load_btn = QPushButton('Load bank')
        self.load_btn.clicked.connect(self.on_load_clicked)
        bank_row.addWidget(QLabel('Bank:'))
        bank_row.addWidget(self.bank_edit, stretch=1)
        bank_row.addWidget(self.load_btn)

        cam_row = QHBoxLayout()
        self.cam_spin = QSpinBox()
        self.cam_spin.setRange(0, 9)
        self.cam_spin.setValue(default_camera)
        self.start_btn = QPushButton('▶  Start')
        self.start_btn.clicked.connect(self.on_start_clicked)
        self.stop_btn = QPushButton('■  Stop')
        self.stop_btn.clicked.connect(self.stop_inference)
        self.stop_btn.setEnabled(False)
        cam_row.addWidget(QLabel('Camera:'))
        cam_row.addWidget(self.cam_spin)
        cam_row.addStretch(1)
        cam_row.addWidget(self.start_btn)
        cam_row.addWidget(self.stop_btn)

        thr_row = QHBoxLayout()
        self.threshold_label = QLabel(f'threshold: {self.threshold:.3f}')
        self.threshold_label.setFont(mono)
        self.threshold_label.setFixedWidth(190)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 5000)
        self.slider.setValue(int(self.threshold * 1000))
        self.slider.setSingleStep(10)
        self.slider.setPageStep(50)
        self.slider.valueChanged.connect(self.on_slider)
        thr_row.addWidget(self.threshold_label)
        thr_row.addWidget(self.slider, stretch=1)

        btn_row = QHBoxLayout()
        self.heat_btn = QPushButton('Heatmap: OFF')
        self.heat_btn.setCheckable(True)
        self.heat_btn.setChecked(False)
        self.heat_btn.toggled.connect(self.on_heat_toggle)
        self.grid_btn = QPushButton('Patch grid: OFF')
        self.grid_btn.setCheckable(True)
        self.grid_btn.setChecked(False)
        self.grid_btn.toggled.connect(self.on_grid_toggle)
        self.snap_btn = QPushButton('Snapshot')
        self.snap_btn.clicked.connect(self.on_snap)
        btn_row.addWidget(self.heat_btn, stretch=1)
        btn_row.addWidget(self.grid_btn, stretch=1)
        btn_row.addWidget(self.snap_btn)

        lay = QVBoxLayout()
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(8)
        lay.addLayout(bank_row)
        lay.addLayout(cam_row)
        lay.addLayout(thr_row)
        lay.addLayout(btn_row)
        overlay.setLayout(lay)
        return overlay

    def _setup_animation(self) -> None:
        self._fx = QGraphicsOpacityEffect(self.controls_overlay)
        self._fx.setOpacity(0.0)
        self.controls_overlay.setGraphicsEffect(self._fx)

        self._controls_anim = QParallelAnimationGroup(self)
        self._controls_opacity_anim = QPropertyAnimation(self._fx, b'opacity', self)
        self._controls_geometry_anim = QPropertyAnimation(self.controls_overlay, b'geometry', self)
        for anim in (self._controls_opacity_anim, self._controls_geometry_anim):
            anim.setDuration(220)
            anim.setEasingCurve(QEasingCurve.Type.OutCubic)
            self._controls_anim.addAnimation(anim)
        self._controls_anim.finished.connect(self._on_controls_anim_done)

        self._panel_anim = QParallelAnimationGroup(self)
        self._panel_fx = QGraphicsOpacityEffect(self.info_panel)
        self._panel_fx.setOpacity(0.0)
        self.info_panel.setGraphicsEffect(self._panel_fx)
        self._panel_opacity_anim = QPropertyAnimation(self._panel_fx, b'opacity', self)
        self._panel_min_anim = QPropertyAnimation(self.info_panel, b'minimumWidth', self)
        self._panel_max_anim = QPropertyAnimation(self.info_panel, b'maximumWidth', self)
        for anim in (self._panel_opacity_anim, self._panel_min_anim, self._panel_max_anim):
            anim.setDuration(300)
            anim.setEasingCurve(QEasingCurve.Type.OutCubic)
            self._panel_anim.addAnimation(anim)
        self._panel_anim.finished.connect(self._on_panel_anim_done)

    # ── Overlay show / hide ──────────────────────────────────────────────

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._reposition_overlay()

    def _reposition_overlay(self) -> None:
        self.controls_overlay.setGeometry(self._controls_rect(self._controls_visible))
        self.controls_overlay.raise_()

    def _controls_rect(self, visible: bool) -> QRect:
        oh = self.controls_overlay.sizeHint().height()
        video_w = max(0, self.width() - self.info_panel.width())
        y = self.height() - oh if visible else self.height()
        return QRect(0, y, video_w, oh)

    def _set_info_panel_visible(self, visible: bool) -> None:
        if visible == self._info_panel_visible:
            return
        self._info_panel_visible = visible
        self._panel_anim.stop()
        start = max(self.info_panel.width(), 0)
        target = PANEL_WIDTH if visible else 0
        if visible:
            self.info_panel.show()
        for anim in (self._panel_min_anim, self._panel_max_anim):
            anim.setStartValue(start)
            anim.setEndValue(target)
        self._panel_opacity_anim.setStartValue(float(self._panel_fx.opacity()))
        self._panel_opacity_anim.setEndValue(1.0 if visible else 0.0)
        for anim in (self._panel_opacity_anim, self._panel_min_anim, self._panel_max_anim):
            anim.setDuration(300 if visible else 210)
        self._panel_anim.start()

    def _on_panel_anim_done(self) -> None:
        if not self._info_panel_visible:
            self.info_panel.hide()
        self._reposition_overlay()

    def eventFilter(self, obj, event) -> bool:
        if event.type() == QEvent.Type.MouseMove:
            if obj is self.video_label or obj is self.controls_overlay:
                self._update_controls_from_mouse()
        return super().eventFilter(obj, event)

    def _update_controls_from_mouse(self) -> None:
        pos = self.mapFromGlobal(QCursor.pos())
        oh = self.controls_overlay.sizeHint().height()
        if pos.y() >= self.height() - CONTROLS_EDGE_PX:
            self._set_controls_visible(True)
        elif self._controls_visible and pos.y() < self.height() - oh - CONTROLS_AWAY_PX:
            self._set_controls_visible(False)

    def _set_controls_visible(self, visible: bool) -> None:
        if visible == self._controls_visible:
            return
        self._controls_visible = visible
        self._controls_anim.stop()
        if visible:
            self.controls_overlay.show()
            self.controls_overlay.raise_()
        self._controls_geometry_anim.setStartValue(self.controls_overlay.geometry())
        self._controls_geometry_anim.setEndValue(self._controls_rect(visible))
        self._controls_opacity_anim.setStartValue(float(self._fx.opacity()))
        self._controls_opacity_anim.setEndValue(1.0 if visible else 0.0)
        for anim in (self._controls_opacity_anim, self._controls_geometry_anim):
            anim.setDuration(240 if visible else 180)
        self._controls_anim.start()

    def _on_controls_anim_done(self) -> None:
        if float(self._fx.opacity()) < 0.05:
            self.controls_overlay.hide()

    # ── Bank loading ─────────────────────────────────────────────────────

    def on_load_clicked(self) -> None:
        path = self.bank_edit.text().strip()
        if not Path(path).exists():
            self.error.emit(f'Bank not found: {path}')
            return
        self.load_bank(path)

    def load_bank(self, path: str) -> None:
        """Load a memory bank from disk and sync the threshold slider to its saved threshold."""
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
        self.status_message.emit(
            f'bank loaded: {bank.features.shape[0]} patches   threshold {self.threshold:.3f}',
            5000,
        )
        if self.video_label.text():
            self.video_label.setText('Press Start to begin live inference.')

    # ── Inference control ────────────────────────────────────────────────

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
        self.worker = InferenceWorker(
            self.extractor, self.bank, self.device,
            self.cam_spin.value(), self.ema, self.blend, self.threshold,
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
        self._set_info_panel_visible(False)
        self._set_controls_visible(False)
        self.thread = None
        self.worker = None

    @pyqtSlot(str)
    def _on_worker_error(self, msg: str) -> None:
        self.error.emit(msg)

    # ── Frame / FPS slots ────────────────────────────────────────────────

    @pyqtSlot(QImage, object, float, float, bool)
    def on_frame(
        self,
        main_qimg: QImage,
        anomaly_data: list[tuple[QImage, float]],
        raw: float,
        ema: float,
        has_duck: bool,
    ) -> None:
        self.last_qimg = main_qimg
        pix = QPixmap.fromImage(main_qimg).scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(pix)
        self.video_label.setText('')

        for i, card in enumerate(self._duck_cards):
            if i < len(anomaly_data):
                qimg, score = anomaly_data[i]
                card.update_duck(qimg)
            else:
                card.clear()

        has_anomaly = bool(anomaly_data)
        self._set_info_panel_visible(has_anomaly)

    @pyqtSlot(float)
    def on_fps(self, fps: float) -> None:
        self.status_message.emit(f'fps: {fps:4.1f}', 2000)

    # ── Controls ─────────────────────────────────────────────────────────

    def on_slider(self, value: int) -> None:
        self.threshold = value / 1000.0
        self.threshold_label.setText(f'threshold: {self.threshold:.3f}')
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
