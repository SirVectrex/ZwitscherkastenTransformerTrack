# ZwitscherkastenTransformerTrack

Transformer-based utilities and experiments for bird-song / chirp tracking and analysis.

Summary

- Purpose: preprocess and track short audio events (bird chirps) and experiment with Transformer-style sequence models.
- Status: initial development / experimental.

Notable changes in this repo

- Added requirements.txt with core Python dependencies.
- Example scripts provided under `scripts/` (preprocess, train, evaluate).
- Configs under `configs/` and model code under `models/` (placeholders / examples).
- Data directories expected at `data/raw` and `data/processed`.

Requirements

- Python 3.8+
- See requirements.txt for pinned dependency versions.

Quick start

1. Create a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # macOS / Linux
   .venv\Scripts\activate      # Windows
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

Usage examples

- Preprocess audio:

  ```bash
  python scripts/preprocess.py --input data/raw --output data/processed
  ```

- Train a model:

  ```bash
  python scripts/train.py --config configs/train.yaml
  ```

- Evaluate:

  ```bash
  python scripts/evaluate.py --model outputs/checkpoint.pt --data data/processed
  ```
