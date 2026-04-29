"""
Live anomaly detection. Camera feed + heatmap overlay + OK/ANOMALY badge.

Controls:
  [   lower threshold (more sensitive)
  ]   raise threshold (less sensitive)
  h   toggle heatmap overlay
  s   save snapshot to snapshots/
  q   quit
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.heatmap import render_heatmap  # noqa: E402
from src.imageio import bgr_to_tensor, center_square_view  # noqa: E402
from src.patchcore import INPUT_SIZE, MemoryBank, PatchFeatureExtractor, pick_device  # noqa: E402


# Engisoft palette (BGR for OpenCV)
COL_OK    = (60, 180, 80)      # green
COL_BAD   = (50, 50, 255)      # red
COL_INK   = (20, 20, 20)
COL_RULE  = (232, 232, 232)
COL_BG    = (255, 255, 255)


def draw_panel(frame: np.ndarray, is_anom: bool, score: float, threshold: float,
               fps: float, show_heat: bool) -> None:
    """Draw the OK/ANOMALY badge + readouts on the frame."""
    H, W = frame.shape[:2]

    # Top bar with fps + score + threshold
    bar_h = 36
    cv2.rectangle(frame, (0, 0), (W, bar_h), COL_BG, -1)
    cv2.rectangle(frame, (0, bar_h - 1), (W, bar_h), COL_RULE, -1)
    text = f'score: {score:6.3f}   thr: {threshold:.3f}   fps: {fps:4.1f}   heat: {"on" if show_heat else "off"}'
    cv2.putText(frame, text, (12, 24), cv2.FONT_HERSHEY_DUPLEX, 0.55, COL_INK, 1, cv2.LINE_AA)

    # Big badge on the right side
    badge_w, badge_h = 240, 96
    bx0 = W - badge_w - 20
    by0 = 54
    col = COL_BAD if is_anom else COL_OK
    cv2.rectangle(frame, (bx0, by0), (bx0 + badge_w, by0 + badge_h), COL_BG, -1)
    cv2.rectangle(frame, (bx0, by0), (bx0 + badge_w, by0 + badge_h), COL_RULE, 1)
    cv2.rectangle(frame, (bx0, by0), (bx0 + 4, by0 + badge_h), col, -1)
    label = 'ANOMALY' if is_anom else 'OK'
    sub = 'defecto detectado' if is_anom else 'pato correcto'
    cv2.putText(frame, label, (bx0 + 18, by0 + 46), cv2.FONT_HERSHEY_DUPLEX, 1.25, col, 2, cv2.LINE_AA)
    cv2.putText(frame, sub, (bx0 + 18, by0 + 74), cv2.FONT_HERSHEY_DUPLEX, 0.52, COL_INK, 1, cv2.LINE_AA)

    # Footer hint
    hint = '[ / ] threshold    h: heatmap    s: snapshot    q: quit'
    cv2.rectangle(frame, (0, H - 28), (W, H), COL_BG, -1)
    cv2.rectangle(frame, (0, H - 28), (W, H - 27), COL_RULE, -1)
    cv2.putText(frame, hint, (12, H - 9), cv2.FONT_HERSHEY_DUPLEX, 0.5, COL_INK, 1, cv2.LINE_AA)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--bank', type=str, default='models/bank.pt')
    ap.add_argument('--camera', type=int, default=0)
    ap.add_argument('--threshold', type=float, default=None, help='override saved threshold')
    ap.add_argument('--blend', type=float, default=0.5, help='heatmap blend alpha')
    ap.add_argument('--ema', type=float, default=0.4, help='score EMA smoothing (0=raw, 1=frozen)')
    args = ap.parse_args()

    device = pick_device()
    print(f'device: {device}')

    extractor = PatchFeatureExtractor(device)
    bank, meta = MemoryBank.load(args.bank, device)
    threshold = args.threshold if args.threshold is not None else meta.get('threshold', 1.0)
    print(f'bank: {bank.features.shape[0]} patches | threshold: {threshold:.3f}')

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print('ERROR: cannot open camera', file=sys.stderr)
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    snap_dir = Path('snapshots')
    snap_dir.mkdir(parents=True, exist_ok=True)

    show_heat = True
    ema_score = None
    ema = float(args.ema)
    fps = 0.0
    last_t = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            crop, (cx0, cy0, cs) = center_square_view(frame)
            x = bgr_to_tensor(crop, device)
            flat, (B, H, W) = extractor.embed(x)
            scores = bank.score(flat).view(H, W)
            max_score = float(scores.max().item())
            score_map_np = scores.cpu().numpy()

            # EMA smoothing (reduces flicker)
            if ema_score is None:
                ema_score = max_score
            else:
                ema_score = ema * ema_score + (1.0 - ema) * max_score

            is_anom = ema_score > threshold

            # Overlay heatmap on the crop region
            display = frame.copy()
            if show_heat:
                heat = render_heatmap(score_map_np, cs, vmax=threshold)
                # Modulate heat alpha by score magnitude so "OK" frames stay calm
                local = display[cy0:cy0 + cs, cx0:cx0 + cs]
                blended = cv2.addWeighted(local, 1.0 - args.blend, heat, args.blend, 0.0)
                display[cy0:cy0 + cs, cx0:cx0 + cs] = blended

            # Crop bounding box
            box_col = COL_BAD if is_anom else COL_OK
            cv2.rectangle(display, (cx0, cy0), (cx0 + cs, cy0 + cs), box_col, 3)

            now = time.time()
            dt = now - last_t
            last_t = now
            inst_fps = 1.0 / max(dt, 1e-3)
            fps = 0.9 * fps + 0.1 * inst_fps if fps else inst_fps

            draw_panel(display, is_anom, ema_score, threshold, fps, show_heat)

            cv2.imshow('PatoInspector — live', display)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            if key == ord('['):
                threshold = max(0.0, threshold - 0.05)
                print(f'threshold -> {threshold:.3f}')
            if key == ord(']'):
                threshold += 0.05
                print(f'threshold -> {threshold:.3f}')
            if key == ord('h'):
                show_heat = not show_heat
            if key == ord('s'):
                fname = snap_dir / f'snap_{int(now)}.jpg'
                cv2.imwrite(str(fname), display, [cv2.IMWRITE_JPEG_QUALITY, 92])
                print(f'saved {fname}')
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
