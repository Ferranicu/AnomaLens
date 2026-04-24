"""
Build a PatchCore memory bank from captured "good" images.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.imageio import bgr_to_tensor  # noqa: E402
from src.patchcore import MemoryBank, PatchFeatureExtractor, coreset_subsample, pick_device  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, default='dataset/good')
    ap.add_argument('--out', type=str, default='models/bank.pt')
    ap.add_argument('--coreset-ratio', type=float, default=0.10)
    ap.add_argument('--batch', type=int, default=8)
    args = ap.parse_args()

    data_dir = Path(args.data)
    files = sorted(list(data_dir.glob('*.jpg')) + list(data_dir.glob('*.png')))
    if not files:
        print(f'ERROR: no images in {data_dir}', file=sys.stderr)
        sys.exit(1)
    print(f'loading {len(files)} images from {data_dir}')

    device = pick_device()
    print(f'device: {device}')

    extractor = PatchFeatureExtractor(device)

    all_feats: list[torch.Tensor] = []
    t0 = time.time()
    for i in range(0, len(files), args.batch):
        batch_files = files[i:i + args.batch]
        tensors = []
        for f in batch_files:
            img = cv2.imread(str(f))
            if img is None:
                print(f'skip unreadable: {f}')
                continue
            tensors.append(bgr_to_tensor(img, device))
        if not tensors:
            continue
        x = torch.cat(tensors, dim=0)
        flat, (B, H, W) = extractor.embed(x)
        all_feats.append(flat.cpu())
        print(f'  [{i + len(batch_files)}/{len(files)}] extracted ({flat.shape[0]} patches)')

    feats = torch.cat(all_feats, dim=0)
    print(f'total patch embeddings: {feats.shape} in {time.time() - t0:.1f}s')

    print(f'coreset subsample (ratio={args.coreset_ratio})...')
    t0 = time.time()
    # Move to GPU for speed if available
    feats_dev = feats.to(device)
    bank = coreset_subsample(feats_dev, ratio=args.coreset_ratio)
    print(f'coreset: {bank.shape} in {time.time() - t0:.1f}s')

    mem = MemoryBank(bank, device)

    # Score training images against the bank to pick a sensible default threshold.
    # (Training images aren't held out, so this is a soft lower bound — users can
    #  tune at runtime with [/] keys.)
    print('calibrating threshold on training set...')
    per_image_max: list[float] = []
    for i in range(0, len(files), args.batch):
        batch_files = files[i:i + args.batch]
        tensors = [bgr_to_tensor(cv2.imread(str(f)), device) for f in batch_files]
        x = torch.cat(tensors, dim=0)
        flat, (B, H, W) = extractor.embed(x)
        scores = mem.score(flat).view(B, H, W)
        per_image_max.extend(scores.amax(dim=(1, 2)).cpu().tolist())
    import statistics
    good_mean = statistics.fmean(per_image_max)
    good_max = max(per_image_max)
    # Default threshold: 2x the max good-image score. Users will tune live.
    default_thresh = good_max * 2.0
    print(f'  good max-scores: mean={good_mean:.3f} max={good_max:.3f}')
    print(f'  suggested threshold: {default_thresh:.3f}')

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mem.save(str(out_path), meta={
        'threshold': default_thresh,
        'good_max': good_max,
        'good_mean': good_mean,
        'n_train': len(files),
    })
    print(f'saved {out_path}')


if __name__ == '__main__':
    main()
