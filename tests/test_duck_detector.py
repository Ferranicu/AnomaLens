"""Offline tests for the HSV duck detector on synthetic images."""
import cv2
import numpy as np

from src.duck_detector import detect_ducks, square_crop


def _blank() -> np.ndarray:
    # Uniform light-gray frame: low saturation, so nothing is detected by default.
    return np.full((480, 640, 3), 240, dtype=np.uint8)


def test_detects_blue_blob_region():
    img = _blank()
    cv2.rectangle(img, (280, 200), (360, 260), (255, 0, 0), -1)  # solid blue, BGR

    boxes = detect_ducks(img)
    assert len(boxes) == 1
    box = boxes[0]
    cx, cy = 320, 230
    assert box.x <= cx < box.x + box.w
    assert box.y <= cy < box.y + box.h
    assert box.area >= 1500


def test_multiple_blobs_sorted_by_area_descending():
    img = _blank()
    cv2.rectangle(img, (50, 50), (90, 90), (255, 0, 0), -1)      # 40x40 = 1600 px
    cv2.rectangle(img, (400, 300), (500, 380), (255, 0, 0), -1)  # 100x80 = 8000 px

    boxes = detect_ducks(img)
    assert len(boxes) == 2
    areas = [b.area for b in boxes]
    assert areas == sorted(areas, reverse=True)


def test_plain_images_yield_no_detection():
    assert detect_ducks(_blank()) == []
    assert detect_ducks(np.zeros((480, 640, 3), dtype=np.uint8)) == []


def test_square_crop_returns_square_inside_frame():
    img = _blank()
    cv2.rectangle(img, (20, 20), (70, 60), (255, 0, 0), -1)  # blob near the corner

    boxes = detect_ducks(img)
    assert boxes
    crop, (x0, y0, size) = square_crop(img, boxes[0])
    h, w = crop.shape[:2]
    assert (h, w) == (size, size)
    assert x0 >= 0 and y0 >= 0
    assert x0 + size <= img.shape[1]
    assert y0 + size <= img.shape[0]
    # The crop actually contains blue duck pixels.
    assert (crop[:, :, 0] > 200).any() and (crop[:, :, 2] < 50).any()
