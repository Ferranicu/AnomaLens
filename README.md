# PatoInspector

Webcam anomaly detection POC for the factory fair. Trains on "good" Engisoft blue PVC ducks, flags ducks with defects (e.g. flawed eye) in real time.

Uses a minimal PatchCore implementation (WideResNet50 patch features + coreset memory bank + nearest-neighbor scoring). No anomaly samples needed for training — cold-start.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

First run downloads WideResNet50 weights (~260MB) to the torch hub cache.

## Flow

1. **Capture good samples** — point camera at good ducks, press SPACE to snap, ESC to finish.
   ```bash
   python scripts/capture.py --out dataset/good --count 40
   ```

2. **Build memory bank** — extract features, coreset-subsample, save.
   ```bash
   python scripts/train.py --data dataset/good --out models/bank.pt
   ```

3. **Run live inference** — heatmap overlay + OK/ANOMALY badge.
   ```bash
   python scripts/run.py --bank models/bank.pt
   ```
   Adjust the threshold on the fly with `[` / `]` keys. Press `s` to save a snapshot, `q` to quit.

## Tips for a good demo

- Keep the camera framing consistent between capture and inference (same distance / background / lighting).
- 30–50 good-duck captures from varied angles is usually plenty.
- If everything looks anomalous, loosen the threshold (`]`). If nothing triggers on the flawed duck, tighten it (`[`).
