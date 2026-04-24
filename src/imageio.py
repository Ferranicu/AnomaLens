"""Image preprocessing shared by capture / train / run."""
from __future__ import annotations

import cv2
import numpy as np
import torch

from .patchcore import INPUT_SIZE


def bgr_to_tensor(bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Center-crop to square, resize to INPUT_SIZE, convert BGR->RGB,
    scale to [0, 1], move to device. Returns (1, 3, H, W).
    """
    h, w = bgr.shape[:2]
    s = min(h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    crop = bgr[y0:y0 + s, x0:x0 + s]
    resized = cv2.resize(crop, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    arr = rgb.astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return t


def center_square_view(bgr: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Return the same center-crop we feed to the model, plus (x0, y0, size)."""
    h, w = bgr.shape[:2]
    s = min(h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    return bgr[y0:y0 + s, x0:x0 + s].copy(), (x0, y0, s)
