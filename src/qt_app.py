"""PyQt6 desktop app — capture, train, and run anomaly detection in one window."""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .anomaly_store import AnomalyStore
from .patchcore import PatchFeatureExtractor, pick_device
from .qt_anomalies import AnomaliesScreen
from .qt_capture import CaptureScreen
from .qt_run import RunScreen
from .qt_train import TrainScreen


CAPTURE_IDX = 0
TRAIN_IDX = 1
RUN_IDX = 2
ANOMALIES_IDX = 3

_STYLE = """
QWidget {
    background: #1b1b22;
    color: #d8d8e8;
    font-family: 'Segoe UI', sans-serif;
    font-size: 13px;
}
QMainWindow, QStackedWidget { background: #1b1b22; }

QLineEdit, QSpinBox, QDoubleSpinBox, QPlainTextEdit {
    background: #24242e;
    border: 1px solid #36364a;
    border-radius: 4px;
    padding: 4px 8px;
    color: #d8d8e8;
    selection-background-color: #4d7cfe;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus { border-color: #4d7cfe; }
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
    background: #36364a; border: none; border-radius: 2px;
}

QPushButton {
    background: #2a2a38;
    color: #c8c8e0;
    border: 1px solid #3a3a50;
    border-radius: 5px;
    padding: 5px 16px;
    font-weight: 600;
}
QPushButton:hover { background: #34344a; border-color: #4d7cfe; color: #e8e8ff; }
QPushButton:pressed { background: #4d7cfe; border-color: #4d7cfe; color: #fff; }
QPushButton:checked { background: #3050aa; border-color: #4d7cfe; color: #aac4ff; }
QPushButton:disabled { background: #1e1e28; color: #484860; border-color: #28283a; }

QSlider::groove:horizontal {
    height: 4px; background: #36364a; border-radius: 2px;
}
QSlider::sub-page:horizontal { background: #4d7cfe; border-radius: 2px; }
QSlider::handle:horizontal {
    background: #7099ff; border: none;
    width: 14px; height: 14px; margin: -5px 0; border-radius: 7px;
}

QProgressBar {
    background: #24242e; border: 1px solid #36364a; border-radius: 4px;
    text-align: center; color: #d8d8e8; min-height: 18px;
}
QProgressBar::chunk { background: #4d7cfe; border-radius: 3px; }

QLabel { background: transparent; color: #d8d8e8; }

QScrollBar:vertical {
    background: #1b1b22; width: 7px; border-radius: 3px; margin: 0;
}
QScrollBar::handle:vertical {
    background: #3a3a52; border-radius: 3px; min-height: 20px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }

QSplitter::handle { background: #26262e; }

QListWidget {
    background: #1e1e28; border: 1px solid #2e2e40; border-radius: 4px;
}
QListWidget::item { padding: 6px 8px; border-bottom: 1px solid #282838; color: #c8c8de; }
QListWidget::item:selected { background: #253060; color: #d8e0ff; }
QListWidget::item:hover:!selected { background: #24243a; }

QStatusBar {
    background: #13131a; color: #60607a; font-size: 11px;
    border-top: 1px solid #22222e;
}
QMessageBox { background: #26262e; }
QMessageBox QLabel { color: #d8d8e8; }
"""

_NAV_BTN = (
    'QPushButton {'
    '  padding: 9px 12px 9px 14px;'
    '  font-size: 12px; font-weight: 600;'
    '  text-align: left; border: none; border-radius: 5px;'
    '  background: transparent; color: #6868888;'
    '}'
    'QPushButton:hover { background: #22222e; color: #c0c0de; }'
    'QPushButton:checked {'
    '  background: #1e2d58; color: #7099ff;'
    '  border-left: 3px solid #4d7cfe; padding-left: 11px;'
    '}'
)


class MainWindow(QMainWindow):
    def __init__(
        self,
        bank_path: str = 'models/bank.pt',
        camera_index: int = 0,
        threshold_override: float | None = None,
        ema: float = 0.4,
        blend: float = 0.5,
        data_dir: str = 'dataset/good',
    ):
        super().__init__()
        self.setWindowTitle('PatoInspector')
        self.resize(1280, 760)
        self.setStyleSheet(_STYLE)

        self.device = pick_device()
        self.extractor = PatchFeatureExtractor(self.device)
        self.store = AnomalyStore('anomalies')

        self.capture_screen = CaptureScreen(default_out=data_dir, default_camera=camera_index)
        self.train_screen = TrainScreen(
            self.extractor, self.device,
            default_data=data_dir, default_out=bank_path,
        )
        self.run_screen = RunScreen(
            self.extractor, self.device, self.store,
            default_bank=bank_path, default_camera=camera_index,
            ema=ema, blend=blend,
        )
        self.anomalies_screen = AnomaliesScreen(self.store)

        for screen in (
            self.capture_screen, self.train_screen,
            self.run_screen, self.anomalies_screen,
        ):
            screen.status_message.connect(self._on_status)
            screen.error.connect(self._on_error)
        self.train_screen.bank_saved.connect(self._on_bank_saved)

        self._build_ui()
        self.statusBar().showMessage(f'device: {self.device}   ready')
        self._switch_to(CAPTURE_IDX)

    def _build_ui(self) -> None:
        self.stack = QStackedWidget()
        self.stack.addWidget(self.capture_screen)
        self.stack.addWidget(self.train_screen)
        self.stack.addWidget(self.run_screen)
        self.stack.addWidget(self.anomalies_screen)

        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)

        self.capture_btn = self._make_nav_btn('Capture', CAPTURE_IDX)
        self.train_btn = self._make_nav_btn('Train', TRAIN_IDX)
        self.run_btn = self._make_nav_btn('Run', RUN_IDX)
        self.anomalies_btn = self._make_nav_btn('Anomalies', ANOMALIES_IDX)

        nav = QVBoxLayout()
        nav.setContentsMargins(8, 20, 8, 16)
        nav.setSpacing(2)
        for btn in (self.capture_btn, self.train_btn, self.run_btn, self.anomalies_btn):
            nav.addWidget(btn)
        nav.addStretch(1)

        wordmark = QLabel('PATO\nINSPECTOR')
        wordmark.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        wordmark.setStyleSheet(
            'color: #303050; font-size: 9px; font-weight: 800;'
            ' letter-spacing: 2px; background: transparent;'
        )
        nav.addWidget(wordmark)

        sidebar = QWidget()
        sidebar.setLayout(nav)
        sidebar.setFixedWidth(128)
        sidebar.setStyleSheet('background: #13131a; border-right: 1px solid #22222e;')

        root = QHBoxLayout()
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(sidebar)
        root.addWidget(self.stack, stretch=1)

        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)

    def _make_nav_btn(self, label: str, index: int) -> QPushButton:
        btn = QPushButton(label)
        btn.setCheckable(True)
        btn.setMinimumHeight(36)
        btn.setStyleSheet(_NAV_BTN)
        btn.clicked.connect(lambda _, i=index: self._switch_to(i))
        self.nav_group.addButton(btn, index)
        return btn

    def _switch_to(self, index: int) -> None:
        current = self.stack.currentIndex()
        if current == CAPTURE_IDX and current != index:
            self.capture_screen.stop_camera()
        elif current == RUN_IDX and current != index:
            self.run_screen.stop_inference()

        if index == CAPTURE_IDX:
            self.capture_screen.start_camera()
        elif index == ANOMALIES_IDX:
            self.anomalies_screen.refresh()

        self.stack.setCurrentIndex(index)
        btn = self.nav_group.button(index)
        if btn is not None:
            btn.setChecked(True)

    def _on_status(self, msg: str, timeout_ms: int) -> None:
        self.statusBar().showMessage(msg, timeout_ms)

    def _on_error(self, msg: str) -> None:
        QMessageBox.warning(self, 'PatoInspector', msg)

    def _on_bank_saved(self, path: str) -> None:
        self.run_screen.bank_edit.setText(path)
        self.run_screen._load_bank(path)

    def start(self) -> None:
        pass

    def closeEvent(self, event) -> None:
        self.capture_screen.stop_camera()
        self.run_screen.stop_inference()
        if self.train_screen.worker is not None:
            self.train_screen.worker.stop()
            if self.train_screen.thread is not None:
                self.train_screen.thread.quit()
                self.train_screen.thread.wait(2000)
        super().closeEvent(event)
