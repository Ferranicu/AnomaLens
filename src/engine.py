"""Shared PatchCore pipeline used by every frontend.

The OpenCV CLI (scripts/run.py, scripts/train.py) and the Qt app
(src/qt_run.py, src/qt_train.py) drive the same math:

* inference — per-duck preprocessing, patch-feature extraction,
  nearest-neighbour scoring, EMA smoothing and threshold decisions;
* training — batched feature extraction, coreset subsampling and
  threshold calibration.

Those pieces live here so the frontends only handle presentation
(windows, widgets, threads) and keep their own palettes.
"""
from __future__ import annotations

import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from .duck_detector import detect_ducks, square_crop
from .heatmap import render_heatmap, render_patch_grid
from .imageio import bgr_to_tensor
from .patchcore import MemoryBank, PatchFeatureExtractor


#: Default threshold stored in a bank's meta: 2x the worst training score.
THRESHOLD_FACTOR = 2.0
#: Default EMA factor for the frame's peak score (higher = smoother).
DEFAULT_EMA = 0.4
#: Default heatmap/grid blend alpha.
DEFAULT_BLEND = 0.5

#: Duck outline colours (BGR): green below threshold, red above.
COL_OK_BGR = (60, 180, 80)
COL_BAD_BGR = (50, 50, 255)


@dataclass(frozen=True)
class DuckScore:
    """Per-duck patch anomaly scores in frame coordinates."""

    crop_box: tuple[int, int, int]  # (x0, y0, size) of the square crop
    score_map: np.ndarray           # (h, w) nearest-neighbour patch scores

    @property
    def max_score(self) -> float:
        return float(self.score_map.max())


# ── Inference ─────────────────────────────────────────────────────────────


def detect_and_score(
    frame: np.ndarray,
    extractor: PatchFeatureExtractor,
    bank: MemoryBank,
) -> list[DuckScore]:
    """Detect ducks in a BGR frame and score every patch against the bank."""
    results: list[DuckScore] = []
    for box in detect_ducks(frame):
        crop, (x0, y0, size) = square_crop(frame, box)
        x = bgr_to_tensor(crop, extractor.device)
        flat, (_, map_h, map_w) = extractor.embed(x)
        score_map = bank.score(flat).view(map_h, map_w).cpu().numpy()
        results.append(DuckScore((x0, y0, size), score_map))
    return results


def update_ema(previous: float | None, raw: float, factor: float) -> float:
    """Smooth the frame's peak score; the first sample seeds the tracker."""
    if previous is None:
        return raw
    return factor * previous + (1.0 - factor) * raw


def is_anomaly(score: float, threshold: float) -> bool:
    """Strict-threshold decision shared by both frontends."""
    return score > threshold


def draw_duck_overlays(
    display: np.ndarray,
    results: Sequence[DuckScore],
    threshold: float,
    *,
    blend: float = DEFAULT_BLEND,
    show_heat: bool = False,
    show_grid: bool = False,
    ok_color: tuple[int, int, int] = COL_OK_BGR,
    bad_color: tuple[int, int, int] = COL_BAD_BGR,
) -> None:
    """Blend optional heatmap/grid overlays and outline each scored duck.

    Draws in place on ``display``. Colours are BGR tuples so each frontend can
    pass its own palette.
    """
    for result in results:
        x0, y0, size = result.crop_box
        if show_heat:
            heat = render_heatmap(result.score_map, size, vmax=threshold)
            _blend_region(display, heat, x0, y0, size, blend)
        if show_grid:
            grid = render_patch_grid(result.score_map, size, vmax=threshold)
            _blend_region(display, grid, x0, y0, size, blend)
        color = bad_color if is_anomaly(result.max_score, threshold) else ok_color
        cv2.rectangle(display, (x0, y0), (x0 + size, y0 + size), color, 3)
        cv2.putText(display, f'{result.max_score:.2f}', (x0 + 4, y0 + size - 8),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, color, 1, cv2.LINE_AA)


def _blend_region(
    display: np.ndarray,
    overlay: np.ndarray,
    x0: int,
    y0: int,
    size: int,
    blend: float,
) -> None:
    region = display[y0:y0 + size, x0:x0 + size]
    display[y0:y0 + size, x0:x0 + size] = cv2.addWeighted(
        region, 1.0 - blend, overlay, blend, 0.0)


# ── Training ──────────────────────────────────────────────────────────────

UnreadableCB = Callable[[Path], None]


def list_dataset_images(data_dir: str | Path) -> list[Path]:
    """All jpg/png captures in a dataset folder, sorted by name."""
    d = Path(data_dir)
    return sorted(list(d.glob('*.jpg')) + list(d.glob('*.png')))


def _load_batch(
    files: Sequence[Path],
    device: torch.device,
    on_unreadable: UnreadableCB | None,
) -> torch.Tensor | None:
    tensors = []
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            if on_unreadable is not None:
                on_unreadable(f)
            continue
        boxes = detect_ducks(img)
        crop = square_crop(img, boxes[0])[0] if boxes else img
        tensors.append(bgr_to_tensor(crop, device))
    if not tensors:
        return None
    return torch.cat(tensors, dim=0)


def iter_extracted_batches(
    files: Sequence[Path],
    extractor: PatchFeatureExtractor,
    device: torch.device,
    batch_size: int,
    on_unreadable: UnreadableCB | None = None,
):
    """Yield ``(files_attempted, flat_patches_cpu, batch_files)`` per batch.

    ``files_attempted`` counts files up to and including this batch, whether or
    not they were readable. Patches come back on CPU so callers can accumulate
    arbitrarily large datasets.
    """
    total = len(files)
    for i in range(0, total, batch_size):
        batch = files[i:i + batch_size]
        x = _load_batch(batch, device, on_unreadable)
        if x is None:
            continue
        flat, _shape = extractor.embed(x)
        yield min(i + batch_size, total), flat.cpu(), batch


def iter_calibration_batches(
    files: Sequence[Path],
    mem: MemoryBank,
    extractor: PatchFeatureExtractor,
    device: torch.device,
    batch_size: int,
    on_unreadable: UnreadableCB | None = None,
):
    """Yield ``(files_attempted, score_maps, batch_files)`` per batch.

    ``score_maps`` has shape (B, H, W): one patch-score map per image.
    """
    total = len(files)
    for i in range(0, total, batch_size):
        batch = files[i:i + batch_size]
        x = _load_batch(batch, device, on_unreadable)
        if x is None:
            continue
        flat, (B, map_h, map_w) = extractor.embed(x)
        scores = mem.score(flat).view(B, map_h, map_w)
        yield min(i + batch_size, total), scores, batch


def calibrate_threshold(per_image_max: Sequence[float]) -> tuple[float, float, float]:
    """Summarise training-set peak scores into (good_mean, good_max, default_threshold).

    Training images aren't held out, so the suggested threshold is a soft lower
    bound — users tune it at runtime with the [/] keys or the Qt slider.
    """
    good_mean = statistics.fmean(per_image_max)
    good_max = max(per_image_max)
    return good_mean, good_max, good_max * THRESHOLD_FACTOR


def bank_meta(n_train: int, good_mean: float, good_max: float) -> dict:
    """Meta dict saved alongside a bank; inference fronts read ``'threshold'``."""
    return {
        'threshold': good_max * THRESHOLD_FACTOR,
        'good_max': good_max,
        'good_mean': good_mean,
        'n_train': n_train,
    }
