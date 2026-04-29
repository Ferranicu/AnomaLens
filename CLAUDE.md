# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this is

**PatoInspector** — Webcam anomaly-detection POC for an "advanced factories" fair demo. Cold-start anomaly detection over Engisoft blue PVC stress-relief ducks: train only on *good* samples, flag defects (e.g. flawed eye) at inference time. Sibling to `../PatoShooter` (gesture-based carnival game) but this one is the serious-vision-POC pitch to Abraham/Josep.

## Stack

Python 3.10+, PyTorch + torchvision, OpenCV, scikit-learn. No anomalib — we roll a minimal PatchCore ourselves so there's no Lightning/yaml-config wrangling.

Install:
```bash
python -m venv .venv && .venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

First run downloads WideResNet50 weights (~260 MB) into the torch hub cache.

## Three-script flow

1. `scripts/capture.py` — webcam UI to save *good* JPEGs (SPACE = snap, ESC = quit). Writes `dataset/good/*.jpg`.
2. `scripts/train.py` — extracts patch features for every good image, runs greedy k-center coreset subsampling, calibrates a default threshold from the training set's max-scores, saves `models/bank.pt`.
3. `scripts/run.py` — live webcam inference. JET heatmap overlay on the center-crop + OK/ANOMALY badge. Live threshold tuning with `[` / `]`, `h` toggles heatmap, `s` saves snapshot, `q` quits.

## Architecture

```
PatoInspector/
├── src/
│   ├── patchcore.py   # feature extractor, coreset, memory bank (core algo)
│   └── imageio.py     # center-square crop + BGR→tensor preprocessing
└── scripts/
    ├── capture.py     # webcam → dataset/good/
    ├── train.py       # dataset/good → models/bank.pt
    └── run.py         # models/bank.pt + webcam → live overlay
```

All shared preprocessing lives in `src/imageio.py` — **do not duplicate it** in scripts. `INPUT_SIZE = 224` is defined in `src/patchcore.py`; both train and run read it from there.

## PatchCore details

- Backbone: WideResNet50 (ImageNet V2 weights), frozen, eval mode.
- Features: concat of `layer2` (512ch, 28×28) + `layer3` (1024ch, 14×14 upsampled to 28×28) → 1536ch, then 3×3 avg-pool with stride 1 to aggregate neighborhoods. Per image: 784 patches × 1536 dims.
- Coreset: greedy farthest-point. Default ratio `0.10`. Implemented on GPU when available — cdist + running-min distance vector.
- Scoring: per-patch min L2 to the bank, reshape to 28×28, upsample for the heatmap overlay. Global image score = `max` of the patch scores.
- Threshold calibration: scores all training images against the final bank, saves `good_max`, `good_mean`, and a default threshold of `2 × good_max` into the bank's meta. `run.py` loads it as the starting threshold — users tune live.

## Conventions

- Images are BGR numpy until the last moment (`bgr_to_tensor` does BGR→RGB + `/255` + device move).
- **Center-square crop** is the model's input region. The HUD draws a bounding box around it so the operator knows what the model actually sees.
- Artifacts (`dataset/`, `models/`, `snapshots/`) are gitignored. Don't commit them.

## Gotchas

- `cv2.VideoCapture(..., cv2.CAP_DSHOW)` is Windows-specific. Strip the backend flag on macOS/Linux.
- `torch.load` without `weights_only=True` may warn on newer PyTorch — it's safe here (own-saved dict) but can be tightened later.
- `scikit-learn` is in `requirements.txt` but currently unused in runtime code — kept for future coreset alternatives (KMeans approx, etc.). Remove if it's still unused later.
- Windows line-ending warnings on git add are benign.

## When extending

- **New feature map**: change layers in `PatchFeatureExtractor.forward` — output spatial size must stay consistent or update downstream reshape.
- **New scoring**: only `MemoryBank.score` needs to change; `run.py` treats it as a black-box `(N,) → (N,)`.
- **Bigger UI**: keep the overlay code in `run.py` pure OpenCV primitives to avoid adding a GUI dependency.
- **GitHub**: private repo at `github.com/Ferranicu/PatoInspector`. Don't push without being asked.

## Running the demo on-site

1. Capture 30–50 good ducks at the fair's actual table lighting / distance / background.
2. Re-train on-site (takes seconds). Don't reuse a bank from a different environment.
3. Walk up to the threshold slowly with `[` / `]` so it triggers confidently on the flawed duck but not on good ones.
