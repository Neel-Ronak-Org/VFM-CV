# Implementation Plan

This turns `CLAUDE.md` into concrete steps, says where each step runs, and flags what is risky.

## 0. Is the brief doable?

**Yes, with five adjustments.**

| # | Issue | Adjustment |
|---|---|---|
| A | **Where it runs.** Code is being written in a cloud container with **no GPU** and no access to Google Drive, Hugging Face or arXiv. All GPU work and downloads must happen on the student's laptop (RTX 3050, 4–6 GB). | Everything is written so it runs on CPU for unit tests, and on the laptop for real runs. Each phase ends with exact commands for the student to run and paste the outputs back (`results/*.csv` + logs are committed). |
| B | **SAM2-Adapter / SAM3-Adapter have no released COD weights**, and training them needs about 4×A100. | Mark them "not evaluated – no released weights". Substitute **SAM2-UNet** (released COD weights). Keep SAM-Adapter (released checkpoint). |
| C | **Method novelty overlaps** RobustCODLite (COD10K-C) and RobustSAM. | Lead with benchmark + held-out protocol; frame the adapter as a laptop-scale robust baseline (see `NOVELTY_REPORT.md`). |
| D | **Compute budget.** The variant list (3 main + 3 λ + LoRA + FPN + 2 extra seeds ≈ 10 runs) in about 24 GPU-hours means ≈ 2.4 h/run. At 1024 px with a clean+corrupted pair per sample this is too tight. | Profile first. Default plan: **512 px** training (rung 3 of the VRAM ladder) with **cached features not possible for corrupted views**, so instead: 20 epochs, and prioritise runs (main 3 → λ sweep → seeds → LoRA/FPN). Log the 512-vs-1024 accuracy cost on one run. |
| E | **Benchmark compute & disk.** 40 corruption conditions × (2,026 COD10K + 250 CAMO + 4,121 NC4K) ≈ 256k images per model. ViT-H SAM-Adapter at ~2–3 FPS ≈ 30 h for COD10K alone **[estimate, verify]**. Stored corrupted sets ≈ 30–60 GB. | (i) Full grid for COD10K-test for every model; CAMO + NC4K for fast models only (SINet-V2, ours, MobileSAM-based, SAM2-UNet if fast). (ii) Generate corruptions **on the fly with deterministic per-image seeds** (bit-identical to stored versions, verified by test) and only cache to disk if CPU becomes the bottleneck. This keeps "generate once, fixed seeds" reproducibility without 60 GB of files. |

Also: COD10K-C's §3 (supplied by the student) defines its own corruption parameters, so we re-implement them rather than using ImageNet-C. `imagecorruptions` is kept for an optional extended unseen set (NumPy-2 shim included).

## 1. Phase-by-phase plan

### Phase 0 – Novelty + setup *(this session)*
- [x] Environment check (container): no GPU; Python 3.11 venv; torch 2.x CPU.
- [x] Literature search → `docs/NOVELTY_REPORT.md`.
- [x] Repo skeleton, pinned requirements, `.gitignore`, configs.
- [x] Core CPU-testable modules: `src/corruptions.py`, `src/metrics.py`, `src/datasets.py`, `src/models/base.py`, `src/evaluate.py` (resumable harness), unit tests.
- [ ] **Student:** run `python scripts/env_check.py` on the laptop and paste output (records real VRAM, OS, CUDA).
- 🛑 **CHECKPOINT 0** — go / pivot decision.

### Phase 1 – Data *(laptop)*
- `scripts/download_data.py` uses `gdown` for: SINet-V2 train (COD10K-train + CAMO-train = 4,040), SINet-V2 test (CHAMELEON 76, CAMO 250, COD10K 2,026, NC4K 4,121).
- If gdown is rate-limited, the script prints manual download instructions and the target folders.
- `python scripts/download_data.py --verify` checks counts, image/mask name alignment, size match, and mask binarity. It writes `results/data_check.csv`.
- Licences/citations → `docs/DATASETS.md`.

### Phase 2 – Harness + baselines *(code here, runs on laptop)*
- Metrics: `src/metrics.py` wraps PySODMetrics (Sα, Eφ-mean, Fβw, MAE) + Dice and IoU at threshold 0.5. Unit-tested.
- Wrappers, one file each, with source/commit/weights header:
  1. `sinet_v2.py` (**required**) — vendored Res2Net-50 + SINet-V2 network code, ported to torch 2.x.
  2. `zoomnet.py`, `pfnet.py`, `feder.py`.
  3. `sam_adapter.py` (ViT-H if it fits in fp16; else log), `sam2_unet.py`.
  4. `oracle_box_sam.py` — zero-shot MobileSAM with GT-box prompt (labelled "oracle, not a fair competitor").
- `results/reproduction.csv` vs paper numbers (±0.015 Sα).
- 🛑 **CHECKPOINT 2.**

### Phase 3 – Corruption benchmark *(laptop)*
- Corruptions: the 8 COD10K-C types × 5 severities (re-implemented from the paper's stated parameters; rain is our approximation). Seen = {gaussian_noise, gaussian_blur, brightness, jpeg_compression}; unseen = {motion_blur, contrast, fog, rain}; optional ImageNet-C `extended_unseen`. Defined once in `configs/corruptions.yaml`.
- Per-image seed = hash(dataset, image name, corruption, severity) → identical output on any machine.
- `src/evaluate.py` resumable; appends to `results/benchmark.csv`.
- `scripts/summarize.py` → mCorr, retention (seen / unseen), optional mCE vs SINet-V2.

### Phase 4 – Our method *(code here, training on laptop)*
- `src/models/robust_adapter.py`: frozen MobileSAM TinyViT, bottleneck adapters (or LoRA) in each block, learned prompt token → SAM mask decoder (alt.: small FPN head).
- `src/train.py`: AMP, grad accumulation, grad checkpointing, structure loss (wBCE + wIoU), 50 % seen-corruption augmentation, consistency loss (MSE / symmetric KL, with/without stop-grad).
- First run: `python src/profile.py` + 1-epoch timing → choose rung of VRAM ladder and epoch count.
- Variants: Ours-Clean, Ours-Aug, Ours-Full; ablations; seeds if time allows.
- 🛑 **CHECKPOINT 4.**

### Phase 5 – Efficiency *(laptop)*: `src/profile.py` → params, GFLOPs (fvcore/thop), FPS (bs 1, fp16, warm-up), peak VRAM → `results/efficiency.csv`.

### Phase 6 – Demo: `demo/app.py` (Gradio), SINet-V2 vs Ours-Full, corruption dropdown + severity slider, overlays, latency, CPU fallback; `docs/DEMO_SCRIPT.md`.

### Phase 7 – Figures/tables: `scripts/make_tables.py`, `scripts/make_figures.py` read only `results/*.csv`; tables emitted as `.tex`.

### Phase 8 – Paper: IEEEtran conference, verified `refs.bib` (URL comment above each entry), `SUBMISSION_CHECKLIST.md`. Compiled here if `latexmk` is available, otherwise Overleaf.
- 🛑 **CHECKPOINT 8.**

### Phase 9 – Release: README, MIT licence, Ours-Full checkpoint (GitHub release), no datasets.

## 2. Division of labour

| Here (cloud, no GPU) | Student's laptop |
|---|---|
| All code, configs, tests, docs, paper text, figure/table scripts | `nvidia-smi`, downloads, baseline inference, corruption benchmark, training, profiling, demo |
| Reads committed `results/*.csv` to build tables/figures and paper | Commits `results/*.csv` + logs and pushes |

## 3. Rough laptop time budget (to be replaced by measured numbers)

| Item | Estimate |
|---|---|
| Data download + verify | 1–2 h (manual if Drive rate-limits) |
| Clean baseline runs, all 4 test sets | 2–4 h |
| Corruption grid, COD10K, fast models (≈4 models × 81k imgs) | 10–20 h |
| Corruption grid, COD10K, SAM-Adapter ViT-H | 20–40 h **(may need to subsample severities or skip)** |
| Training (≈10 runs) | ≈24 h |
| Total | ≈60–90 GPU-hours over several weeks |
