# Robust, Lightweight Foundation-Model Camouflaged Object Detection

A corruption-robustness benchmark for camouflaged object detection (COD) that compares specialist COD models with SAM-based ones, plus a lightweight robust SAM adapter that runs on a 4 GB laptop GPU.

> **Status:** baselines reproduced (Phase 2). Corruption benchmark running (Phase 3). Our adapter's code and training script plus the demo are ready (Phases 4 and 6). Every number this project reports comes from `results/*.csv`.

- Project brief: [`CLAUDE.md`](CLAUDE.md)
- Plan and doability assessment: [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md)
- Novelty check: [`docs/NOVELTY_REPORT.md`](docs/NOVELTY_REPORT.md)
- Plain-English explanations: [`docs/EXPLAINER.md`](docs/EXPLAINER.md)
- Running log: [`PROGRESS.md`](PROGRESS.md)

## Setup

Python 3.11 in a project-local virtual environment, managed with [uv](https://docs.astral.sh/uv/). uv downloads Python 3.11 for you; your system Python (e.g. 3.13) is left alone.

**Windows (PowerShell), the student laptop:**
```powershell
# one-time: install uv
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

cd VFM-CV
uv venv --python 3.11 .venv
.venv\Scripts\activate
uv pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install -r requirements.txt
python scripts/env_check.py          # must show "CUDA available: True" and torch 2.6.0+cu124
python -m pytest -q                  # unit tests
```
If PowerShell refuses to run `activate`, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

**Linux / WSL2:** same, with `source .venv/bin/activate`.

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

## Train our adapter

```bash
python -m src.train --config configs/train/ours_full.yaml --profile 40   # speed + VRAM check, saves nothing
python -m src.train --config configs/train/ours_full.yaml                # train (Ours-Clean / Ours-Aug: ours_clean.yaml / ours_aug.yaml)
python -m src.train --config configs/train/ours_full.yaml --resume       # continue after --max-hours, Ctrl+C or a crash
python -m src.evaluate --models Ours-Full --datasets CHAMELEON --corruptions clean
```
Needs `checkpoints/mobile_sam/mobile_sam.pt` (`python scripts/download_data.py --weights mobile_sam`). Checkpoints go to `checkpoints/ours/<run>/`, logs to `results/train/<run>/`. See `docs/EXPLAINER.md` §9 for the design and the VRAM ladder.

## Demo

```bash
python demo/app.py                    # SINet-V2 vs our best trained adapter; opens in the browser
python demo/app.py --device cpu       # without a GPU
```
Upload your own photo (drag & drop) or put photos in `demo/my_photos/`, then tick up to 6 models to compare side by side (baselines and every trained Ours variant; each loads the first time it is ticked). Presentation shot list: `docs/DEMO_SCRIPT.md`.

## Resolution test (diagnostic)

```bash
python scripts/resolution_test.py                    # Ours-Full / Ours-Clean fed 352/512/768-px views, CHAMELEON
python scripts/resolution_test.py --dataset COD10K
```

## Paper tables and figures

Every number in the paper's tables and figures is generated from `results/*.csv`, never typed by hand:
```bash
python scripts/summarize_benchmark.py   # results/summary_robustness.csv
python scripts/significance.py          # results/significance.csv (paired bootstrap CIs)
python scripts/make_tables.py           # paper/tables/*.tex (the paper \input{}s these)
python scripts/make_figures.py          # figures/fig_{architecture,severity,tradeoff,resolution,heldout}.pdf/.png
python scripts/make_qualitative.py      # figures/fig_teaser + fig_qualitative (runs the models: laptop, ~3 min)
```

## Repository layout

See §3 of `CLAUDE.md`. `data/` and `checkpoints/` are gitignored; `results/*.csv` are committed and are the only source of numbers for the tables, figures and paper.

## Licence

Code: MIT (added at release). The datasets and baseline weights keep their own licences.
