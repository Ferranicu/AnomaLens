"""
PyQt6 desktop app for live anomaly detection.

Same args as scripts/run.py — drop-in replacement with a polished UI:
threshold slider, rolling score graph, big OK/ANOMALY badge.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from PyQt6.QtWidgets import QApplication  # noqa: E402

from src.qt_app import MainWindow  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--bank', type=str, default='models/bank.pt')
    ap.add_argument('--camera', type=int, default=0)
    ap.add_argument('--threshold', type=float, default=None, help='override saved threshold')
    ap.add_argument('--blend', type=float, default=0.5, help='heatmap blend alpha')
    ap.add_argument('--ema', type=float, default=0.4, help='score EMA smoothing (0=raw, 1=frozen)')
    args = ap.parse_args()

    app = QApplication(sys.argv)
    win = MainWindow(args.bank, args.camera, args.threshold, args.ema, args.blend)
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
