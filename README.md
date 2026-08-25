# PatoInspector

Webcam anomaly detection with a hand-rolled [PatchCore](https://arxiv.org/abs/2106.08265)
implementation: train only on *good* samples (cold-start — no defect examples
needed), then flag defective ones live. Demo subject: spotting flawed eyes on
blue PVC ducks.

Two frontends share one engine (`src/engine.py`):

- **CLI** — plain OpenCV windows (`pato-capture`, `pato-train`, `pato-run`)
- **Desktop app** — PyQt6, with capture / train / run / anomaly-browser screens (`pato-app`)

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # macOS/Linux
pip install -e .
```

First run downloads WideResNet50 weights (~260 MB) into the torch hub cache.
CPU-only machines work; a GPU speeds up feature extraction and coreset.

On Windows, cameras open through OpenCV's DirectShow backend automatically
(faster startup than the default MSMF); other platforms use the default backend.

## Quickstart (CLI)

1. **Capture good samples** — SPACE snaps every detected duck as its own crop,
   ESC quits.
   ```bash
   pato-capture --out dataset/good --count 40
   ```

2. **Build the memory bank** — extract patch features, subsample with greedy
   k-center coreset, calibrate a default threshold from the training scores.
   ```bash
   pato-train --data dataset/good --out models/bank.pt --coreset-ratio 0.10 --batch 8
   ```

3. **Run live inference** — JET heatmap overlay + OK/ANOMALY badge per duck.
   ```bash
   pato-run --bank models/bank.pt --camera 0 --blend 0.5 --ema 0.4
   ```
   Keys: `[` / `]` lower/raise threshold, `h` heatmap on/off, `g` patch grid,
   `s` save snapshot to `snapshots/`, `q` quit.

The same commands also work as plain scripts: `python scripts/capture.py ...`,
`python scripts/train.py ...`, `python scripts/run.py ...`.

## Desktop app

```bash
pato-app --bank models/bank.pt --camera 0
```

Accepts the same flags as `pato-run` plus `--threshold` (override the saved
threshold), `--blend` and `--ema`. The sidebar has four screens:

- **Capture** — live preview with detection boxes; Space saves each detected duck.
- **Train** — pick dataset/output folder, coreset ratio and batch size; watch
  progress and live calibration patch grids.
- **Run** — live inference with a threshold slider; anomalous ducks pop up as
  zoomed cards in the side panel and get logged to `anomalies/`.
- **Anomalies** — browse or clear logged anomaly events.

<!-- Screenshots TODO:
     - docs/screenshots/app-run.png    desktop app, Run screen with anomaly cards
     - docs/screenshots/cli-run.png    CLI heatmap overlay + OK/ANOMALY badge
-->

## How it works

WideResNet50 `layer2`+`layer3` features are concatenated per patch,
neighborhood-aggregated, then subsampled by greedy farthest-point coreset
(default ratio 0.10) into a memory bank saved at `models/bank.pt`. At inference
each patch's nearest-neighbour distance forms a 28x28 anomaly map upsampled for
the overlay; the frame's peak score is EMA-smoothed and compared against the
threshold. Training stores `2x` the worst training-image score as the default
threshold — tune it live with `[` / `]` or the app's slider.

Artifacts (`dataset/`, `models/`, `snapshots/`, `anomalies/`) are gitignored
and created on demand.

## Tests

Offline, CPU-only, no pretrained downloads:

```bash
pip install pytest
pytest -q
```

CI runs the same suite on every push/PR to `main`
(`.github/workflows/ci.yml`, CPU-only PyTorch wheels).
