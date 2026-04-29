"""Anomalies screen — browse the events the Run screen has logged."""
from __future__ import annotations

import time

from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtGui import QFont, QIcon, QPixmap
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .anomaly_store import AnomalyEvent, AnomalyStore


class AnomaliesScreen(QWidget):
    status_message = pyqtSignal(str, int)
    error = pyqtSignal(str)

    def __init__(self, store: AnomalyStore):
        super().__init__()
        self.store = store
        self._events: list[AnomalyEvent] = []
        self._build_ui()
        self.refresh()

    def _build_ui(self) -> None:
        self.refresh_btn = QPushButton('Refresh')
        self.refresh_btn.clicked.connect(self.refresh)

        self.clear_btn = QPushButton('Clear all')
        self.clear_btn.clicked.connect(self.on_clear)

        self.count_label = QLabel('0 events')
        self.count_label.setStyleSheet('color: #686890;')

        toolbar_widget = QWidget()
        toolbar_widget.setStyleSheet('background: #16161e; border-bottom: 1px solid #22222e;')
        toolbar = QHBoxLayout(toolbar_widget)
        toolbar.setContentsMargins(12, 8, 12, 8)
        toolbar.addWidget(self.refresh_btn)
        toolbar.addWidget(self.clear_btn)
        toolbar.addSpacing(20)
        toolbar.addWidget(self.count_label)
        toolbar.addStretch(1)

        self.list_widget = QListWidget()
        self.list_widget.setIconSize(QSize(140, 105))
        self.list_widget.setUniformItemSizes(False)
        self.list_widget.currentItemChanged.connect(self._on_select)

        self.full_label = QLabel('Select an event.')
        self.full_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.full_label.setStyleSheet('background:#101010; color:#888;')
        self.full_label.setMinimumHeight(360)

        self.zoom_label = QLabel()
        self.zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.zoom_label.setStyleSheet('background:#101010;')
        self.zoom_label.setMinimumHeight(220)

        mono = QFont('Consolas')
        mono.setStyleHint(QFont.StyleHint.Monospace)
        mono.setPointSize(10)
        self.meta_label = QLabel('—')
        self.meta_label.setFont(mono)
        self.meta_label.setStyleSheet('padding: 8px; color: #9090b8; background: #1e1e2a; border-radius: 4px;')

        right = QVBoxLayout()
        right.setContentsMargins(8, 8, 8, 8)
        right.setSpacing(6)
        right.addWidget(QLabel('<b>Full frame</b>'))
        right.addWidget(self.full_label, stretch=2)
        right.addWidget(QLabel('<b>Anomaly zoom</b>'))
        right.addWidget(self.zoom_label, stretch=1)
        right.addWidget(self.meta_label)

        right_widget = QWidget()
        right_widget.setLayout(right)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.list_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([320, 800])

        root = QVBoxLayout()
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(toolbar_widget)
        root.addWidget(splitter, stretch=1)
        self.setLayout(root)

    def refresh(self) -> None:
        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        self._events = self.store.list_events(limit=200)
        for evt in self._events:
            ts_str = time.strftime('%H:%M:%S', time.localtime(evt.ts))
            date_str = time.strftime('%Y-%m-%d', time.localtime(evt.ts))
            item = QListWidgetItem(f'{date_str}  {ts_str}\nscore {evt.score:.3f}  thr {evt.threshold:.3f}')
            pix = QPixmap(str(evt.full_path))
            if not pix.isNull():
                item.setIcon(QIcon(pix.scaled(
                    140, 105,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )))
            self.list_widget.addItem(item)
        self.list_widget.blockSignals(False)
        self.count_label.setText(f'{len(self._events)} event{"" if len(self._events) == 1 else "s"}')
        if self._events:
            self.list_widget.setCurrentRow(0)
        else:
            self.full_label.clear()
            self.full_label.setText('No anomalies recorded yet.')
            self.zoom_label.clear()
            self.meta_label.setText('—')

    def on_clear(self) -> None:
        if not self._events:
            return
        ans = QMessageBox.question(
            self, 'PatoInspector',
            f'Delete all {len(self._events)} anomaly events from disk?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if ans != QMessageBox.StandardButton.Yes:
            return
        try:
            self.store.clear()
        except Exception as e:
            self.error.emit(f'failed to clear: {e}')
            return
        self.refresh()
        self.status_message.emit('cleared all anomalies', 3000)

    def _on_select(self, current: QListWidgetItem | None, _previous) -> None:
        if current is None:
            return
        row = self.list_widget.row(current)
        if row < 0 or row >= len(self._events):
            return
        evt = self._events[row]
        self._show_image(self.full_label, str(evt.full_path))
        self._show_image(self.zoom_label, str(evt.zoom_path))
        ts_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(evt.ts))
        self.meta_label.setText(
            f'time:       {ts_str}\n'
            f'score:      {evt.score:.3f}\n'
            f'threshold:  {evt.threshold:.3f}\n'
            f'full:       {evt.full_path.name}\n'
            f'zoom:       {evt.zoom_path.name}'
        )

    def _show_image(self, label: QLabel, path: str) -> None:
        pix = QPixmap(path)
        if pix.isNull():
            label.setText('image missing')
            return
        label.setPixmap(pix.scaled(
            label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))
