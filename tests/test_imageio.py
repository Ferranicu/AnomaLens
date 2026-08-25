"""Offline tests for preprocessing math in imageio (center crop + tensor conversion)."""
import numpy as np
import pytest
import torch

from src.imageio import bgr_to_tensor, open_camera
from src.patchcore import INPUT_SIZE


def test_output_shape_dtype_and_range():
    img = np.full((300, 200, 3), 128, dtype=np.uint8)
    t = bgr_to_tensor(img, torch.device('cpu'))
    assert t.shape == (1, 3, INPUT_SIZE, INPUT_SIZE)
    assert t.dtype == torch.float32
    assert 0.0 <= float(t.min()) <= float(t.max()) <= 1.0
    # A uniform 128/255 frame stays uniform through crop + resize.
    assert float(t.mean()) == pytest.approx(128.0 / 255.0, abs=1e-4)


def test_center_crop_takes_the_middle_square():
    # 400 tall x 200 wide -> square side 200, vertical offset (400-200)//2 = 100.
    img = np.zeros((400, 200, 3), dtype=np.uint8)
    img[:200, :, 1] = 255   # top half: green in BGR -> G plane after conversion
    img[200:, :, 2] = 255   # bottom half: red in BGR -> R plane after conversion

    t = bgr_to_tensor(img, torch.device('cpu'))
    r = t[0, 0]  # R plane of the RGB tensor
    g = t[0, 1]  # G plane

    # The crop spans source rows 100..299: rows 100..199 green, 200..299 red.
    # Leave a small margin around the seam for resize interpolation.
    top_green = g[10:110].mean()
    bottom_red = r[114:214].mean()
    assert top_green > 0.95
    assert bottom_red > 0.95
    # Neither colour leaks into the opposite half of the crop.
    assert r[10:110].mean() < 0.05
    assert g[114:214].mean() < 0.05


def test_bgr_channels_are_reordered_to_rgb():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, :, 0] = 255  # BGR blue channel

    t = bgr_to_tensor(img, torch.device('cpu'))
    assert float(t[0, 2].mean()) > 0.99  # blue lands on the last RGB plane
    assert float(t[0, 0].mean()) < 0.01
    assert float(t[0, 1].mean()) < 0.01


def test_open_camera_returns_closed_capture_for_bogus_index():
    # No camera hardware required: an out-of-range index yields a closed capture.
    cap = open_camera(99)
    try:
        assert not cap.isOpened()
    finally:
        cap.release()
