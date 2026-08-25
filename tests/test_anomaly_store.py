"""Offline tests for the JSONL anomaly event store."""
import json

import numpy as np
import pytest

from src.anomaly_store import AnomalyStore


def _img() -> np.ndarray:
    return np.zeros((12, 16, 3), dtype=np.uint8)


def test_add_then_list_round_trip(tmp_path):
    store = AnomalyStore(tmp_path / 'anomalies')
    ts = 1700000000.0

    evt = store.add(_img(), _img(), score=1.5, threshold=0.75, ts=ts)

    assert evt.full_path.exists()
    assert evt.zoom_path.exists()
    events = store.list_events()
    assert len(events) == 1
    e = events[0]
    assert e.ts == pytest.approx(ts)
    assert e.score == pytest.approx(1.5)
    assert e.threshold == pytest.approx(0.75)
    assert e.full_path == evt.full_path
    assert e.zoom_path == evt.zoom_path
    assert f'score {evt.score:.3f}' in e.label()


def test_newest_first_ordering_and_limit(tmp_path):
    store = AnomalyStore(tmp_path / 'anomalies')
    store.add(_img(), _img(), 1.0, 0.5, ts=1.0)
    store.add(_img(), _img(), 2.0, 0.5, ts=2.5)

    events = store.list_events()
    assert [e.ts for e in events] == [pytest.approx(2.5), pytest.approx(1.0)]
    assert len(store.list_events(limit=1)) == 1


def test_corrupt_and_blank_lines_are_skipped(tmp_path):
    store = AnomalyStore(tmp_path / 'anomalies')
    store.add(_img(), _img(), 1.0, 0.5, ts=1.0)

    with store.index_path.open('a', encoding='utf-8') as fh:
        fh.write('{not valid json\n')
        fh.write('\n')
        fh.write('[]\n')

    events = store.list_events()
    assert len(events) == 1
    assert events[0].ts == pytest.approx(1.0)


def test_records_pointing_at_missing_jpegs_are_skipped(tmp_path):
    store = AnomalyStore(tmp_path / 'anomalies')
    dangling = {
        'ts': 2.0, 'score': 2.0, 'threshold': 1.0,
        'full': 'missing_full.jpg', 'zoom': 'missing_zoom.jpg',
    }
    with store.index_path.open('a', encoding='utf-8') as fh:
        fh.write(json.dumps(dangling) + '\n')

    assert store.list_events() == []


def test_missing_index_file_is_handled(tmp_path):
    store = AnomalyStore(tmp_path / 'fresh')  # dir created, no index yet
    assert store.list_events() == []


def test_clear_removes_events(tmp_path):
    store = AnomalyStore(tmp_path / 'anomalies')
    store.add(_img(), _img(), 1.0, 0.5, ts=1.0)

    store.clear()

    assert store.list_events() == []
    assert not list(store.root.glob('evt_*.jpg'))
