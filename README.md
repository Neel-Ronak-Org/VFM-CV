# Robust, Lightweight Foundation-Model Camouflaged Object Detection

A corruption-robustness benchmark for camouflaged object detection (COD) that compares specialist COD models with SAM-based ones, plus a lightweight robust SAM adapter that runs on a 4 GB laptop GPU.

> **Status: Phase 0 (setup and novelty check).** No results yet. Every number this project reports will come from `results/*.csv`.

- Project brief: [`CLAUDE.md`](CLAUDE.md)
- Plan and doability assessment: [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md)
- Novelty check: [`docs/NOVELTY_REPORT.md`](docs/NOVELTY_REPORT.md)
- Plain-English explanations: [`docs/EXPLAINER.md`](docs/EXPLAINER.md)
- Running log: [`PROGRESS.md`](PROGRESS.md)

## Setup

Python 3.10 or 3.11.

```bash
# with uv (recommended)
uv venv --python 3.11 .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
# install the CUDA build of PyTorch first, matching your driver, e.g.:
uv pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements.txt
python scripts/env_check.py          # records OS / GPU / VRAM
python -m pytest -q                  # unit tests (CPU only)
```

## Data

```bash
python scripts/download_data.py --data      # COD10K+CAMO train, CHAMELEON/CAMO/COD10K/NC4K test
python scripts/download_data.py --verify    # counts, pairing, mask checks -> results/data_check.csv
```
If Google Drive blocks a download, the script prints the link and the path to save the file to. The datasets are not redistributed here; see each dataset's licence.

## Evaluate

```bash
# clean test sets
python -m src.evaluate --models SINet-V2 --datasets CHAMELEON CAMO COD10K NC4K --corruptions clean
# full corruption grid (8 corruptions x 5 severities), resumable
python -m src.evaluate --models SINet-V2 --datasets COD10K --corruptions all
```
Corruptions are generated on the fly with fixed per-image seeds (`configs/corruptions.yaml`), so results are bit-reproducible.

## Repository layout

See §3 of `CLAUDE.md`. `data/` and `checkpoints/` are gitignored; `results/*.csv` are committed and are the only source of numbers for the tables, figures and paper.

## Licence

Code: MIT (added at release). The datasets and baseline weights keep their own licences.
