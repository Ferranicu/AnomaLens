"""Disk-backed log of detected anomaly events.

Layout under `root/`:
  index.jsonl           — one JSON record per line, newest at bottom
  evt_<ts>_full.jpg     — composited frame (heatmap overlay + crop box)
  evt_<ts>_zoom.jpg     — zoomed sub-region around the score-map peak
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class AnomalyEvent:
    ts: float
    score: float
    threshold: float
    full_path: Path
    zoom_path: Path

    def label(self) -> str:
        return f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.ts))}   score {self.score:.3f}'


class AnomalyStore:
    def __init__(self, root: str | Path = 'anomalies'):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / 'index.jsonl'

    def add(
        self,
        full_bgr: np.ndarray,
        zoom_bgr: np.ndarray,
        score: float,
        threshold: float,
        ts: float | None = None,
    ) -> AnomalyEvent:
        if ts is None:
            ts = time.time()
        stamp = f'{ts:.3f}'.replace('.', '_')
        full_path = self.root / f'evt_{stamp}_full.jpg'
        zoom_path = self.root / f'evt_{stamp}_zoom.jpg'
        cv2.imwrite(str(full_path), full_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(str(zoom_path), zoom_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        record = {
            'ts': ts,
            'score': float(score),
            'threshold': float(threshold),
            'full': full_path.name,
            'zoom': zoom_path.name,
        }
        with self.index_path.open('a', encoding='utf-8') as fh:
            fh.write(json.dumps(record) + '\n')
        return AnomalyEvent(ts, float(score), float(threshold), full_path, zoom_path)

    def list_events(self, limit: int | None = None) -> list[AnomalyEvent]:
        if not self.index_path.exists():
            return []
        events: list[AnomalyEvent] = []
        with self.index_path.open('r', encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(rec, dict):
                    continue
                try:
                    full_path = self.root / rec['full']
                    zoom_path = self.root / rec['zoom']
                    ts = float(rec['ts'])
                    score = float(rec['score'])
                    threshold = float(rec['threshold'])
                except (KeyError, TypeError):
                    continue
                if not full_path.exists() or not zoom_path.exists():
                    continue
                events.append(AnomalyEvent(
                    ts=ts,
                    score=score,
                    threshold=threshold,
                    full_path=full_path,
                    zoom_path=zoom_path,
                ))
        events.sort(key=lambda e: e.ts, reverse=True)
        if limit is not None:
            events = events[:limit]
        return events

    def clear(self) -> None:
        for p in self.root.glob('evt_*.jpg'):
            p.unlink(missing_ok=True)
        self.index_path.unlink(missing_ok=True)
