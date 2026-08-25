"""Duck detection via HSV color segmentation for Engisoft blue PVC ducks."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class DuckBox:
    x: int
    y: int
    w: int
    h: int
    area: int


@dataclass
class DetectorParams:
    """HSV thresholds and filter settings. Tunable at runtime."""
    h_lo: int = 95
    h_hi: int = 130
    s_lo: int = 80
    s_hi: int = 255
    v_lo: int = 40
    v_hi: int = 255
    min_area: int = 1500
    max_area_frac: float = 0.80
    aspect_min: float = 0.35
    aspect_max: float = 2.80
    morph_close: int = 5
    morph_open: int = 3
    pad_ratio: float = 0.15


_DEFAULT_PARAMS = DetectorParams()


def detect_ducks(
    bgr: np.ndarray,
    params: DetectorParams = _DEFAULT_PARAMS,
) -> list[DuckBox]:
    """HSV-based detection of Engisoft blue PVC ducks.

    Returns a list of bounding boxes sorted by area descending (largest duck first).
    Returns empty list if no ducks are found.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array([params.h_lo, params.s_lo, params.v_lo]),
        np.array([params.h_hi, params.s_hi, params.v_hi]),
    )
    if params.morph_close > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (params.morph_close, params.morph_close)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    if params.morph_open > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (params.morph_open, params.morph_open)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

    n, _labels, stats, _cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    frame_area = bgr.shape[0] * bgr.shape[1]
    boxes: list[DuckBox] = []
    for i in range(1, n):  # 0 = background
        x, y, w, h, area = (int(v) for v in stats[i])
        if area < params.min_area or area > frame_area * params.max_area_frac:
            continue
        aspect = w / max(h, 1)
        if not (params.aspect_min <= aspect <= params.aspect_max):
            continue
        boxes.append(DuckBox(x, y, w, h, area))
    boxes.sort(key=lambda b: b.area, reverse=True)
    return boxes


def square_crop(
    bgr: np.ndarray,
    box: DuckBox,
    pad_ratio: float = _DEFAULT_PARAMS.pad_ratio,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Square crop centered on a DuckBox with padding, clipped to frame bounds.

    Returns (crop_bgr, (x0, y0, size)) — frame coordinates so callers can
    re-project heatmaps back to display coordinates unchanged.
    """
    cx = box.x + box.w // 2
    cy = box.y + box.h // 2
    half = int(max(box.w, box.h) * (1.0 + pad_ratio) / 2)
    H, W = bgr.shape[:2]
    x0 = max(0, cx - half)
    y0 = max(0, cy - half)
    x1 = min(W, x0 + 2 * half)
    y1 = min(H, y0 + 2 * half)
    size = min(x1 - x0, y1 - y0)
    return bgr[y0:y0 + size, x0:x0 + size].copy(), (x0, y0, size)
