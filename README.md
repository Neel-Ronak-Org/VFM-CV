# How Robust Are Foundation-Model Camouflage Detectors?

Code, configurations and results for a corruption-robustness benchmark of camouflaged object detection (COD) models, and for a lightweight, prompt-free MobileSAM adapter that trains and runs on a 4 GB laptop GPU.

- **Benchmark:** specialist COD models (SINet-V2, PFNet, ZoomNet, FEDER) and SAM-based ones (SAM2-UNet, SAM-Adapter) on CHAMELEON, CAMO, COD10K and NC4K, clean and under 8 corruptions × 5 severities, with a held-out (seen / unseen) corruption split. Protocol: [`docs/BENCHMARK.md`](docs/BENCHMARK.md).
- **Model:** a frozen MobileSAM TinyViT encoder with bottleneck adapters, learned prompts and a trained SAM mask decoder, trained with corruption augmentation and a clean/corrupted consistency loss. Details: [`docs/MODEL.md`](docs/MODEL.md).
- **Results:** every number in the paper is computed from the CSV files in [`results/`](results/) by the scripts below; the paper's LaTeX tables are generated into [`results/tables/`](results/tables/).

## Setup

Python 3.11 and PyTorch 2.6 with CUDA 12.4, installed with [uv](https://docs.astral.sh/uv/) (pip works too).

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\activate
uv pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install -r requirements.txt
python scripts/env_check.py          # OS, Python, CUDA, GPU and VRAM
python -m pytest -q                  # unit tests (metrics, corruptions, datasets, models, reporting)
```

## Data and weights

```bash
python scripts/download_data.py --data      # COD10K+CAMO train, CHAMELEON/CAMO/COD10K/NC4K test
python scripts/download_data.py --verify    # counts, pairing and mask checks -> results/data_check.csv
python scripts/download_data.py --weights   # all required weights (or name some: --weights mobile_sam)
```
Sources are listed in `configs/sources.yaml`. If Google Drive blocks a download, the script prints the link and where to save the file. Datasets and third-party weights are not redistributed here; their licences and citations are in [`docs/DATASETS.md`](docs/DATASETS.md).

## Reproduce

**Baselines on clean images** (reproduction check against the published numbers in `configs/reported.yaml`):
```bash
python scripts/run_clean_baselines.py       # results/benchmark.csv, results/reproduction.csv
```

**Corruption benchmark** (resumable; corrupted images are generated on the fly with fixed per-image seeds):
```bash
python scripts/run_benchmark.py             # plan in configs/benchmark.yaml; --status shows progress
python -m src.evaluate --models SINet-V2 --datasets COD10K --corruptions all   # a single model
```

**Train our adapter** (needs `checkpoints/mobile_sam/mobile_sam.pt`):
```bash
python -m src.train --config configs/train/ours_full.yaml --profile 40   # speed and memory check only
python -m src.train --config configs/train/ours_full.yaml                # Ours (1024)
python -m src.train --config configs/train/ours_full_512.yaml            # Ours (512); seeds: _s1, _s2
python -m src.train --config configs/train/ours_clean.yaml               # ablations: ours_aug, ours_full_*
```
Each run writes its config, git hash, loss curve and timings to `results/train/<run>/`.

**Efficiency** (parameters, GFLOPs, latency, peak VRAM at batch size 1):
```bash
python -m src.profile --models SINet-V2 PFNet ZoomNet SAM2-UNet SAM-Adapter Ours-Full Ours-Full-512
```

**Tables and figures** (read only `results/*.csv`):
```bash
python scripts/summarize_benchmark.py   # results/summary_robustness.csv: mCorr, retention, mCE
python scripts/significance.py          # results/significance.csv: paired bootstrap 95% CIs
python scripts/make_tables.py           # results/tables/*.tex (LaTeX tables + numbers.tex macros)
python scripts/make_figures.py          # figures/fig_{architecture,severity,tradeoff,resolution,heldout}
python scripts/make_corruption_examples.py --from-preview --severity 4   # figures/fig_corruptions (paper Fig. 2)
python scripts/make_qualitative.py      # figures/fig_teaser, fig_qualitative (runs the models)
python scripts/resolution_test.py       # input-resolution analysis -> results/resolution_test_<ds>.csv
```

## Demo

```bash
python demo/app.py                    # Gradio app: upload a photo or pick a test image, add a corruption
python demo/app.py --device cpu       # without a GPU
```
Choose up to six models, a corruption and a severity; the masks are shown side by side as overlays with each model's latency. For test images, the ground truth and per-image scores can be shown too.

## Repository layout

```
configs/      corruption protocol, benchmark plan, data and weight sources, training configs
src/          datasets, corruptions, metrics, evaluation, training, profiling, model wrappers
src/models/   one wrapper per model; third_party/ holds the vendored upstream code (own licences)
scripts/      download, benchmark, summaries, significance tests, tables, figures, analysis
demo/         Gradio demo
results/      benchmark.csv, per-image scores, training logs, efficiency, significance, tables/
figures/      generated figures
docs/         benchmark protocol, model details, dataset licences and citations
tests/        unit tests
data/, checkpoints/   datasets and weights (not tracked)
```

## Licence

Our code is released under the MIT licence ([`LICENSE`](LICENSE)). Vendored code in `src/models/third_party/` keeps its original licence (each folder has its LICENSE). Datasets and pretrained weights are subject to their authors' terms (see `docs/DATASETS.md`).
