"""
Build a PatchCore memory bank from captured "good" images.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

from src.engine import (
    bank_meta,
    calibrate_threshold,
    iter_calibration_batches,
    iter_extracted_batches,
    list_dataset_images,
)
from src.patchcore import MemoryBank, PatchFeatureExtractor, coreset_subsample, pick_device


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, default='dataset/good')
    ap.add_argument('--out', type=str, default='models/bank.pt')
    ap.add_argument('--coreset-ratio', type=float, default=0.10)
    ap.add_argument('--batch', type=int, default=8)
    args = ap.parse_args()

    data_dir = Path(args.data)
    files = list_dataset_images(data_dir)
    if not files:
        print(f'ERROR: no images in {data_dir}', file=sys.stderr)
        sys.exit(1)
    print(f'loading {len(files)} images from {data_dir}')

    device = pick_device()
    print(f'device: {device}')

    extractor = PatchFeatureExtractor(device)

    all_feats: list[torch.Tensor] = []
    t0 = time.time()
    for done, flat, _batch in iter_extracted_batches(
        files, extractor, device, args.batch,
        on_unreadable=lambda f: print(f'skip unreadable: {f}'),
    ):
        all_feats.append(flat)
        print(f'  [{done}/{len(files)}] extracted ({flat.shape[0]} patches)')

    feats = torch.cat(all_feats, dim=0)
    print(f'total patch embeddings: {feats.shape} in {time.time() - t0:.1f}s')

    print(f'coreset subsample (ratio={args.coreset_ratio})...')
    t0 = time.time()
    # Move to GPU for speed if available
    bank = coreset_subsample(feats.to(device), ratio=args.coreset_ratio)
    print(f'coreset: {bank.shape} in {time.time() - t0:.1f}s')

    mem = MemoryBank(bank, device)

    # Score training images against the bank to pick a sensible default threshold.
    # (Training images aren't held out, so this is a soft lower bound — users can
    #  tune at runtime with [/] keys.)
    print('calibrating threshold on training set...')
    per_image_max: list[float] = []
    for _done, scores, _batch in iter_calibration_batches(
        files, mem, extractor, device, args.batch,
        on_unreadable=lambda f: print(f'skip unreadable: {f}'),
    ):
        per_image_max.extend(scores.amax(dim=(1, 2)).cpu().tolist())
    good_mean, good_max, default_thresh = calibrate_threshold(per_image_max)
    print(f'  good max-scores: mean={good_mean:.3f} max={good_max:.3f}')
    print(f'  suggested threshold: {default_thresh:.3f}')

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mem.save(str(out_path), meta=bank_meta(len(files), good_mean, good_max))
    print(f'saved {out_path}')


if __name__ == '__main__':
    main()
