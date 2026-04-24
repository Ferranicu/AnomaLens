"""
Capture good-duck samples from the webcam.

Controls:
  SPACE  snap frame
  c      toggle live crop overlay
  ESC/q  quit
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.imageio import center_square_view  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=str, default='dataset/good')
    ap.add_argument('--count', type=int, default=40, help='stop after this many saved frames')
    ap.add_argument('--camera', type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob('*.jpg'))
    idx = len(existing)
    print(f'writing to {out_dir} (starting at index {idx})')

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print('ERROR: cannot open camera', file=sys.stderr)
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    show_crop = True
    flash_until = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            display = frame.copy()
            _, (x0, y0, s) = center_square_view(frame)
            if show_crop:
                cv2.rectangle(display, (x0, y0), (x0 + s, y0 + s), (0, 255, 0), 2)

            hud = f'saved: {idx}/{args.count}   [SPACE] snap   [c] box   [q/ESC] quit'
            cv2.rectangle(display, (0, 0), (display.shape[1], 34), (0, 0, 0), -1)
            cv2.putText(display, hud, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            if time.time() < flash_until:
                overlay = display.copy()
                cv2.rectangle(overlay, (0, 0), (display.shape[1], display.shape[0]), (255, 255, 255), -1)
                display = cv2.addWeighted(overlay, 0.35, display, 0.65, 0)

            cv2.imshow('PatoInspector — capture', display)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            if key == ord('c'):
                show_crop = not show_crop
            if key == ord(' '):
                crop, _ = center_square_view(frame)
                fname = out_dir / f'good_{idx:04d}.jpg'
                cv2.imwrite(str(fname), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
                idx += 1
                flash_until = time.time() + 0.12
                print(f'saved {fname}')
                if idx >= args.count:
                    print('reached target count')
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
