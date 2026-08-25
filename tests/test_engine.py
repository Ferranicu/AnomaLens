"""Offline tests for the shared inference/training engine helpers."""
import numpy as np
import pytest

from src.engine import (
    COL_BAD_BGR,
    COL_OK_BGR,
    DuckScore,
    bank_meta,
    draw_duck_overlays,
    is_anomaly,
    update_ema,
)


def test_update_ema_seeds_on_first_sample():
    assert update_ema(None, 0.7, 0.4) == 0.7


def test_update_ema_blends_previous_and_raw():
    # factor * previous + (1 - factor) * raw, matching both frontends' original math.
    assert update_ema(1.0, 0.0, 0.4) == pytest.approx(0.4)
    assert update_ema(0.5, 1.5, 0.4) == pytest.approx(1.1)


def test_is_anomaly_is_strict():
    assert not is_anomaly(0.5, 0.5)   # exactly at threshold is still OK
    assert is_anomaly(0.51, 0.5)


def _result(score_map_value: float) -> DuckScore:
    score_map = np.full((4, 4), score_map_value, dtype=np.float32)
    return DuckScore((8, 8, 32), score_map)


def test_draw_duck_overlays_boxes_below_threshold_in_ok_color():
    display = np.zeros((64, 64, 3), dtype=np.uint8)

    draw_duck_overlays(display, [_result(0.1)], threshold=0.5, ok_color=COL_OK_BGR)

    assert (display[8, 8:40] == np.array(COL_OK_BGR)).all()


def test_draw_duck_overlays_boxes_above_threshold_in_bad_color():
    display = np.zeros((64, 64, 3), dtype=np.uint8)

    draw_duck_overlays(display, [_result(0.9)], threshold=0.5, bad_color=COL_BAD_BGR)

    assert (display[8, 8:40] == np.array(COL_BAD_BGR)).all()


def test_bank_meta_matches_calibration_contract():
    meta = bank_meta(n_train=37, good_mean=0.111, good_max=0.6)
    assert meta['threshold'] == pytest.approx(2.0 * 0.6)   # THRESHOLD_FACTOR * good_max
    assert meta['good_max'] == pytest.approx(0.6)
    assert meta['good_mean'] == pytest.approx(0.111)
    assert meta['n_train'] == 37
