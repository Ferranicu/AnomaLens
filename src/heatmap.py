"""Shared heatmap rendering used by run.py and the Qt UI."""
from __future__ import annotations

import cv2
import numpy as np


def render_heatmap(score_map: np.ndarray, size: int, vmax: float | None = None) -> np.ndarray:
    """score_map: (H, W) float -> (size, size, 3) BGR colormap (JET).

    vmax pins the "full red" anchor so colors reflect absolute score magnitude.
    Pass threshold (or good_max) from bank meta. Falls back to per-frame max
    only when vmax is None (e.g. no bank meta available).
    """
    lo = 0.0
    hi = vmax if (vmax is not None and vmax > 1e-6) else float(score_map.max())
    if hi < 1e-6:
        norm = np.zeros_like(score_map, dtype=np.uint8)
    else:
        norm = np.clip(score_map / hi * 255.0, 0, 255).astype(np.uint8)
    cm = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    cm = cv2.resize(cm, (size, size), interpolation=cv2.INTER_LINEAR)
    return cm
