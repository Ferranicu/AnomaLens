"""Shared heatmap rendering used by run.py and the Qt UI."""
from __future__ import annotations

import cv2
import numpy as np


def _norm_score_map(score_map: np.ndarray, vmax: float | None) -> np.ndarray:
    hi = vmax if (vmax is not None and vmax > 1e-6) else float(score_map.max())
    if hi < 1e-6:
        return np.zeros_like(score_map, dtype=np.uint8)
    return np.clip(score_map / hi * 255.0, 0, 255).astype(np.uint8)


def render_heatmap(score_map: np.ndarray, size: int, vmax: float | None = None) -> np.ndarray:
    """score_map: (H, W) float -> (size, size, 3) BGR smooth colormap (JET).

    vmax pins the "full red" anchor so colors reflect absolute score magnitude.
    Pass threshold (or good_max) from bank meta. Falls back to per-frame max
    only when vmax is None (e.g. no bank meta available).
    """
    cm = cv2.applyColorMap(_norm_score_map(score_map, vmax), cv2.COLORMAP_JET)
    return cv2.resize(cm, (size, size), interpolation=cv2.INTER_LINEAR)


def render_patch_grid(score_map: np.ndarray, size: int, vmax: float | None = None) -> np.ndarray:
    """score_map: (H, W) float -> (size, size, 3) BGR cell grid (JET, INTER_NEAREST).

    Each patch is a solid colored rectangle; thin dark grid lines separate cells.
    vmax meaning identical to render_heatmap — pass threshold for absolute anchoring.
    H×W cells (28×28 for standard PatchCore) are visible as distinct squares.
    """
    H, W = score_map.shape
    cm = cv2.applyColorMap(_norm_score_map(score_map, vmax), cv2.COLORMAP_JET)
    cm = cv2.resize(cm, (size, size), interpolation=cv2.INTER_NEAREST)
    for i in range(1, H):
        y = int(round(i * size / H))
        cv2.line(cm, (0, y), (size - 1, y), (20, 20, 20), 1)
    for i in range(1, W):
        x = int(round(i * size / W))
        cv2.line(cm, (x, 0), (x, size - 1), (20, 20, 20), 1)
    return cm
