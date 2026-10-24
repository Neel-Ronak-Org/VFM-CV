# Progress Log

Read this first at the start of every session. Newest entry at the bottom.

---

## Session 1 — 2026-10-13 (Claude, cloud container)

### Environment
| | Cloud container (code development) | Student laptop (real runs) |
|---|---|---|
| OS | Linux 6.18 (container) | Windows 11 (10.0.26200), native (no WSL) |
| GPU / VRAM | **none** | RTX 3050 Laptop, **4.00 GiB** (4096 MiB), compute 8.6, bf16 OK, driver 592.82 |
| Python | 3.11.17 (uv venv `.venv`) | system 3.13.7; project venv to be 3.11 via uv |
| torch | 2.14.1 (CPU) | 2.6.0+cu124 |
| Network | PyPI + GitHub only; arXiv, Google Drive, Hugging Face **blocked** | full |

Consequence: data downloads, baseline inference, training and profiling must run on the laptop.

### Done
- Read the brief; wrote a doability assessment and phase plan → `docs/IMPLEMENTATION_PLAN.md`.
- Phase 0 literature search → `docs/NOVELTY_REPORT.md`. Verdict: **GO with reframing**. Benchmark (C1), held-out protocol (C3) and analysis (C4) are open. The method (C2) overlaps RobustCODLite (COD10K-C) and RobustSAM, so it should be framed as a laptop-scale baseline.
- Checked baseline weight availability from the official READMEs → `configs/sources.yaml`.
  - Available: SINet-V2, ZoomNet, PFNet, FEDER, SAM-Adapter (1 checkpoint), SAM2-UNet, MobileSAM, EfficientSAM.
  - **Not available:** SAM2-Adapter, SAM3-Adapter (no COD-trained weights; training needs ~4×A100).
- Repo skeleton, pinned `requirements.txt`, `.gitignore`, configs.
- `src/corruptions.py`: deterministic per-image seeding (SHA-256) + NumPy-2 compatibility shim for `imagecorruptions` (fog, gaussian_blur were broken).
- `src/metrics.py` (PySODMetrics + Dice/IoU), `src/datasets.py` (loader + verifier), `src/models/base.py` (common interface, fp16, resize-back), `src/evaluate.py` (resumable harness).
- `scripts/download_data.py` (gdown + manual fallback + verify), `scripts/env_check.py`.
- 23 unit tests passing (`pytest -q`).

### Next
1. 🛑 **CHECKPOINT 0**: student decides go / pivot on the novelty report.
2. Student: run `python scripts/env_check.py`, record VRAM above.
3. Student: read §3 of COD10K-C (arXiv:2606.02603). Record its 8 corruption names and severity parameters, then update `configs/corruptions.yaml` if needed (`defocus_blur` → `gaussian_blur`?).
4. Phase 1: `python scripts/download_data.py --data && python scripts/download_data.py --verify`; commit `results/data_check.csv`.
5. Phase 2: write the SINet-V2 wrapper (vendor Res2Net + network code), then the others.

### Open problems
- Exact COD10K-C corruption list/severities not verified (arXiv blocked here).
- SAM-Adapter checkpoint backbone size unknown until downloaded; ViT-H at fp16 on 4 GB may not fit.
- Benchmark compute for large models (ViT-H) may exceed budget; plan is COD10K-only for them.
- `torch==2.14.1` CUDA wheel name for the laptop to be confirmed when installing.

### Reproduce
```bash
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
python -m pytest -q
python -m src.evaluate --models debug-centre --datasets COD10K --corruptions clean --limit 5  # needs data
```

---

## Session 2 — 2026-10-13 (Claude, cloud container)

### Decisions from the student (Checkpoint 0)
- **GO.** SAM2-UNet replaces SAM2-/SAM3-Adapter (no released weights).
- AI-assisted research is permitted by the course/venue (keep the disclosure in the paper's acknowledgement).
- Laptop env recorded above (Windows 11 native, RTX 3050 4 GB, 56.7 GiB free disk).
- COD10K-C §3 text supplied → corruption protocol switched to COD10K-C definitions.

### Done
- `configs/corruptions.yaml` + `src/corruptions.py`: re-implemented the 8 COD10K-C corruptions with the paper's exact severity parameters (gaussian_noise σ, horizontal motion blur k, Gaussian blur radius, brightness/contrast factors, fog α, JPEG quality, rain). Rain has no numeric definition in the paper, so its parameters are **our approximation**. All implementation assumptions are listed in `docs/EXPLAINER.md` §2.
- Held-out split: seen = {gaussian_noise, gaussian_blur, brightness, jpeg_compression}; unseen = {motion_blur, contrast, fog, rain}. Optional ImageNet-C `extended_unseen` = {shot_noise, defocus_blur, zoom_blur, pixelate} for unseen noise.
- Parameter-level unit tests (noise σ, box-kernel motion blur, multiplicative brightness, fog formula, JPEG quality, monotonic severity, determinism).
- `scripts/preview_corruptions.py` → visual grid of all corruptions × severities.
- Phase 1 code: mask-folder fallbacks (`GT`, `GT_Object`, …), dataset statistics in `--verify` (resolution, object-area fraction), archive organiser tested on fake SINet-V2-style zips, zip path-traversal guard.
- `docs/DATASETS.md`: licences and citations (COD10K non-commercial verified in SINet README; CAMO/CHAMELEON/NC4K licences unverified or not stated).
- Pinned torch 2.6.0 / torchvision 0.21.0 to match the laptop; Windows setup in README. 42 tests pass on torch 2.6.0.
- Corruption cost on a 1024×768 image: 3–50 ms each (noise slowest), so on-the-fly generation is fine.

### Next (student, on the laptop) — Phase 1
```powershell
git pull
uv venv --python 3.11 .venv ; .venv\Scripts\activate
uv pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install -r requirements.txt
python -m pytest -q
python scripts/download_data.py --data
python scripts/download_data.py --verify
python scripts/preview_corruptions.py --image data/raw/TestDataset/COD10K/Imgs/<any>.jpg
git add results/data_check.csv ; git commit -m "Phase 1 data check" ; git push
```
Then: Phase 2 — SINet-V2 wrapper (Claude), weights via `python scripts/download_data.py --weights sinet_v2`.

### Open problems
- ~~Push from the cloud container failed (403)~~ — resolved: GitHub access fixed, branch pushed 2026-10-13.
- Exact archive layout of the SINet-V2 Drive packages is not verified (organiser handles common variants; check the `--verify` output).
- Rain is not comparable with COD10K-C (our approximation).

---

## Session 3 — 2026-10-13 (Claude, cloud container)

### From the student
- PR #1 and #2 merged; install works on the laptop.
- `results/data_check.csv`: **all counts correct** (train 4040, CHAMELEON 76, CAMO 250, COD10K 2026, NC4K 4121); no missing pairs, non-binary or empty masks. **2 CHAMELEON images differ in size from their masks** (known quirk).
- `figures/corruption_preview.jpg` on a real COD10K image looks as intended. **Phase 1 complete.**

### Done (Phase 2, part 1: SINet-V2)
- Vendored SINet-V2 network + Res2Net unchanged (`src/models/third_party/sinet_v2/`, Apache-2.0 LICENSE kept). Only the import path and a hard-coded pretrained path were edited.
- `src/models/sinet_v2.py`: replicates the official `MyTesting.py` (PIL resize to 352, last output, logits upsampled to full resolution, sigmoid, per-image min-max). fp32 by default.
- Harness: predictions are resized to the mask size when they differ (CHAMELEON quirk; standard protocol). `--precision auto|fp16|fp32`; precision and device are recorded per row.
- `--verify`: size mismatch is now a warning (lists file names), not a failure.
- Reproduction tooling: `configs/reported.yaml` (paper S-alpha with verification URLs), `scripts/eval_prediction_maps.py` (score released maps with our metrics), `scripts/make_reproduction.py` → `results/reproduction.csv`.
- `scripts/download_data.py --maps sinet_v2` fetches the official SINet-V2 prediction maps.
- 51 tests pass.

### Next (student, on the laptop)
```powershell
git checkout main ; git pull
python scripts/download_data.py --weights sinet_v2
python scripts/download_data.py --maps sinet_v2
python -m src.evaluate --models SINet-V2 --datasets CHAMELEON --corruptions clean     # ~1 min smoke test
python -m src.evaluate --models SINet-V2 --datasets CAMO COD10K NC4K --corruptions clean
python scripts/eval_prediction_maps.py --model SINet-V2 --pred-root data/released_maps/SINet-V2
python scripts/make_reproduction.py
git add results/ ; git commit -m "SINet-V2 clean reproduction" ; git push
```

### Next (Claude)
- Remaining Phase 2 wrappers: SAM2-UNet, ZoomNet, PFNet, SAM-Adapter, FEDER, oracle-box MobileSAM.

### Open problems
- SINet-V2 E-phi / F-beta-w / MAE published values not yet confirmed against the TPAMI paper itself (secondary sources disagree); only S-alpha is used for the reproduction criterion.

---

## Session 4 — 2026-10-13 (Claude, cloud container)

### From the student
- PR #3 merged. **SINet-V2 reproduces:** clean S-alpha CHAMELEON 0.8881 / CAMO 0.8204 / COD10K 0.8152 / NC4K 0.8470 vs reported 0.888 / 0.820 / 0.815 / 0.847 (all |Δ| ≤ 0.0004, tolerance 0.015). `results/reproduction.csv`. Checkpoint-2 gate passed for SINet-V2.
- Note: CHAMELEON latency in that run was 2.4 s/img (others 60–90 ms) — warm-up or a busy laptop; latency for the paper will come from `src/profile.py` (Phase 5), not from evaluation runs.

### Done (Phase 2, part 2: remaining baselines)
Every wrapper copies the official test script's preprocessing and post-processing; each wrapper's header records source commit, licence, weights URL and porting changes. All networks vendored under `src/models/third_party/` with their LICENSE.

| Model | Port | Weights check in container | Reported S-alpha in `configs/reported.yaml` |
|---|---|---|---|
| SAM2-UNet | Hiera-L trunk built directly (no hydra/sam2 pkg) | — (Drive) | 4 datasets (arXiv HTML) |
| ZoomNet | timm ResNet-50, pretrained=False | **strict load OK, 901/901 keys** | CHAMELEON/CAMO/COD10K (survey); NC4K unconfirmed |
| PFNet | backbone only loaded for training | — (project page) | CHAMELEON/CAMO/COD10K (1 secondary table) |
| FEDER | import paths only | — (Drive) | none confirmed → released maps used |
| SAM-Adapter | loss/training code removed; ViT-B/L/H auto-detected from checkpoint | — (Drive) | CHAMELEON/CAMO/COD10K (SAM2-Adapter paper table) |
| Oracle-Box MobileSAM | timm import path; registry factories removed | **strict load OK** | n/a (oracle) |

- Harness: `needs_gt` / `set_gt()` hook for oracle models only.
- Parameter counts match the papers (ZoomNet 32.38M, PFNet 46.50M); tests check each wrapper's contract and size. 58 tests pass.
- `scripts/run_clean_baselines.py`: one command for all clean runs + reproduction table; skips models whose weights are missing.
- `download_data.py`: Drive-folder downloads (SAM2-UNet weights + maps, FEDER maps), ZoomNet maps from GitHub release.
- `timm==1.0.30` added to requirements (resolves on windows/linux py3.11).
- EXPLAINER §7: baseline table, post-processing differences, training-data note, oracle rationale.

### Next (student, on the laptop)
```powershell
git checkout main ; git pull
uv pip install -r requirements.txt                     # adds timm
python scripts/download_data.py --weights zoomnet feder sam_adapter sam2_unet mobile_sam
#   PFNet: download PFNet.pth manually from https://mhaiyang.github.io/CVPR2021_PFNet/index.html -> checkpoints/pfnet/PFNet.pth
python scripts/download_data.py --maps zoomnet feder sinet_v2
python scripts/run_clean_baselines.py
python scripts/eval_prediction_maps.py --model ZoomNet --pred-root data/released_maps/ZoomNet
python scripts/eval_prediction_maps.py --model FEDER   --pred-root data/released_maps/FEDER
python scripts/eval_prediction_maps.py --model SINet-V2 --pred-root data/released_maps/SINet-V2
python scripts/make_reproduction.py
git add results/ ; git commit -m "Phase 2 clean baselines" ; git push
```
Then 🛑 **CHECKPOINT 2** (full reproduction table).

### Open problems
- PFNet weights only via the project page (manual download).
- SAM-Adapter checkpoint backbone unknown until downloaded (auto-detected); ViT-H fp16 at 1024 should fit in 4 GB but is unverified.
- SAM-Adapter's published numbers are computed at 1024x1024 with its own metric code → expect a small gap.
- FEDER published S-alpha not confirmed; relies on released-maps comparison.

---

## Session 5 — 2026-10-14 (Claude, cloud container)

### From the student
- PR #4 merged. Laptop **crashed** during the Phase 2 downloads. Most `.pth` files arrived. The FEDER prediction-maps folder failed repeatedly: gdown "Cannot retrieve the public link ... had many accesses" = Google Drive per-file rate limit (the folder is thousands of individual PNGs).

### Done (download robustness)
- Single-file downloads write to `<dest>.part` and are renamed only when complete → a crash can't leave a truncated checkpoint that later runs would skip.
- Drive folders are marked complete (`.download_complete`) only if every file arrived; partial folders are kept and resumed (`resume=True`, `retries=3`) instead of being treated as done.
- `--maps` with no names now fetches only single-archive sources (SINet-V2, ZoomNet). FEDER / SAM2-UNet map folders are `optional` (browser download recommended).
- `--check-weights`: loads every checkpoint that is present and flags missing or corrupt ones.
- `make_reproduction.py` records `released_maps_missing` so an incomplete map set isn't mistaken for a valid reference.
- Found and fixed: gdown 6.4.2 has no `remaining_ok` argument (it raises at the end listing failed files instead).
- 60 tests pass.

### Next (student)
See the recovery steps in the reply / below:
```powershell
git checkout main ; git pull
python scripts/download_data.py --check-weights
python scripts/download_data.py --weights          # re-fetches only what is missing
python scripts/download_data.py --maps             # SINet-V2 + ZoomNet maps (single zips)
python scripts/run_clean_baselines.py              # resumable
```

### Session 5b — weights status from the laptop (`--check-weights`)
- OK: SINet-V2, ZoomNet (901 tensors), MobileSAM, SAM2-UNet `SAM2UNet-COD.pth` (1192 tensors).
- **SAM-Adapter: CORRUPT** (UnpicklingError) → delete + re-download (browser if gdown fails again).
- Missing: FEDER (gdown), PFNet (manual, project page).
- **Mistake (mine):** the SAM2-UNet Drive-folder download pulled all 7 task checkpoints (~5.8 GB) — breaks the "ask before >5 GB" rule. Fixed: SAM2-UNet is now manual-only with `satisfied_by: SAM2UNet-COD.pth`, so the folder is never re-fetched; the six non-COD checkpoints can be deleted.
- `res2net50` (training only) and `efficient_sam_vitt` (Phase 4 fallback) marked optional: not fetched by default, not reported as missing.
- `--check-weights` now says *why* a file is corrupt (HTML page from Drive vs truncated).

---

## Session 6 — 2026-10-14 (Claude, cloud container)

### State (from results/ on main)
- Clean results: SINet-V2, ZoomNet, SAM2-UNet **reproduced** (all |ΔSα| ≤ 0.0005 where a published value is confirmed); Oracle-Box MobileSAM run (Sα 0.79–0.85, needs a visual sanity check).
- Not yet in results: PFNet, SAM-Adapter (re-downloaded; checkpoint + released `config.yaml` now in `checkpoints/sam_adapter/`), FEDER clean-only (weights link dead → released maps), released-maps scores for SINet-V2 / ZoomNet.

### Done
- `scripts/check_phase2.py`: one command that checks data counts, that every required checkpoint loads, the released SAM-Adapter `config.yaml` against our wrapper's architecture (incl. the encoder's hard-coded adapter settings), released-map completeness, a clean result for every model × dataset (FEDER accepted as clean-only), and the reproduction tolerances. Prints the remaining commands and writes `results/phase2_status.md`. Exit 0 only when Phase 2 is complete.
- 66 tests pass.

### Next (student)
```powershell
git checkout main ; git pull
python scripts/check_phase2.py      # follow its "Still to run" list, then run it again
git add results/ ; git commit -m "Phase 2 status" ; git push
```

### Parked idea (2026-10-15)
- Student's idea for a follow-up: biologically inspired modules (V1 front end, contour integration / border ownership, recurrent top-down refinement) on a frozen backbone, tested with this project's benchmark. Written up in `docs/FUTURE_WORK.md`. **Not part of the current plan**; revisit after Phase 9.
---

## Session 7 — 2026-10-16 (Claude, cloud container) — 🛑 CHECKPOINT 2

### Reproduction table (from `results/phase2_status.md`, laptop run)
Clean S-alpha, ours vs published (tolerance ±0.015):

| Model | CHAMELEON | CAMO | COD10K | NC4K | Verdict |
|---|---|---|---|---|---|
| SINet-V2 | 0.8881 / 0.888 | 0.8204 / 0.820 | 0.8152 / 0.815 | 0.8470 / 0.847 | ✅ |Δ| ≤ 0.0004 |
| PFNet | 0.8819 / 0.882 | 0.7825 / 0.782 | 0.7996 / 0.800 | 0.8287 / — | ✅ |Δ| ≤ 0.0005 |
| ZoomNet | 0.9019 / 0.902 | 0.8201 / 0.820 | 0.8385 / 0.838 | 0.8531 / — | ✅ |Δ| ≤ 0.0005; vs authors' maps ≤ 0.0003 |
| SAM2-UNet | 0.9139 / 0.914 | 0.8835 / 0.884 | 0.8799 / 0.880 | 0.9007 / 0.901 | ✅ |Δ| ≤ 0.0005 |
| SAM-Adapter (ViT-H) | 0.8959 / 0.896 | 0.8470 / 0.847 | 0.8844 / 0.883 | 0.8842 / — | ✅ |Δ| ≤ 0.0014 |
| FEDER (released maps, clean-only) | 0.8867 | 0.8021 | — | 0.8470 | no weights; no COD10K maps |
| Oracle-Box MobileSAM (analysis) | 0.7869 | 0.8270 | 0.8273 | 0.8503 | needs visual check |

- Data, weights and the released SAM-Adapter config all check out (config identical to our wrapper).
- **Decision (student):** FEDER COD10K skipped → FEDER is clean-only on CHAMELEON/CAMO/NC4K and is not in the corruption benchmark. Recorded as a known gap in `check_phase2.py`.

### Problem found: SAM-Adapter is ~16–18 s/image on the laptop (others ~0.06–0.23 s)
- At that speed its 40-condition grid would take ~355 h on COD10K alone.
- Likely cause (mine): fp32 ViT-H weights (2.5 GB) + autocast fp16 copies + explicit 4096×4096×16 attention maps exceed 4 GB → Windows pages GPU memory to system RAM.
- Fix: SAM-Adapter **low-memory mode** (default on CUDA): fp16 encoder weights, attention computed 4 heads at a time (exact; unit-tested), FFT branch kept fp32. CPU test: fp16-encoder output within 3e-4 of fp32.
- Needs on the laptop: profile speed/VRAM, then re-check clean CHAMELEON S-alpha (should stay ≈ 0.896).

### Also done
- `src/profile.py` (Phase 5 tool, needed now): params, GFLOPs/GMACs (torch FlopCounterMode), end-to-end latency/FPS, peak VRAM, oversubscription flag → `results/efficiency.csv`.
- `scripts/qualitative_grid.py`: image | GT (+ oracle box) | model maps, worst-N + random rows → `figures/qual_*.jpg` (oracle check now; Fig. 5 later).
- `check_phase2.py`: known gaps (FEDER COD10K) and SINet-V2 map scoring are informational; warns about models too slow for Phase 3.

### Next (student)
```powershell
git checkout main ; git pull
python -m src.profile --models SAM-Adapter --runs 5
python -m src.evaluate --models SAM-Adapter --datasets CHAMELEON --corruptions clean --out results/samadapter_lowmem_check.csv
python scripts/qualitative_grid.py --models Oracle-Box-MobileSAM SINet-V2 SAM2-UNet --dataset CHAMELEON
python -m src.profile --models SINet-V2 PFNet ZoomNet SAM2-UNet Oracle-Box-MobileSAM
python scripts/check_phase2.py
git add results/ figures/ ; git commit -m "Checkpoint 2 checks" ; git push
```

---

## Session 8 — 2026-10-16 (Claude, cloud container) — Checkpoint 2 closed, Phase 3 tooling

### From the student (laptop)
- `check_phase2.py`: **PHASE 2 COMPLETE**.
- SAM-Adapter low-memory mode: **3.84 s/img** (was 16–18 s), peak 2.48 GB / reserved 2.89 GB of 4 GB (no oversubscription); clean CHAMELEON Sα **0.8960** (was 0.8959): unchanged. (Its clean rows in benchmark.csv were made before low-memory mode; the check shows identical scores, so they are kept.)
- `results/efficiency.csv` (fixed 1024×768 input, end-to-end): SINet-V2 29 ms, PFNet 38 ms, ZoomNet 94 ms, SAM2-UNet 71 ms, Oracle MobileSAM 78 ms, SAM-Adapter 3841 ms.
- **Oracle check (`figures/qual_CHAMELEON_clean.jpg`): not a bug.** It fails when the box covers most of the image (fills the box or *inverts* the mask) or when the object is thin or fragmented; it is good on compact objects. Written up in EXPLAINER §7.

### Decision: Phase 3 plan (student delegated: "best for the paper")
- 5 fast models: full grid (8 corruptions × 5 severities + clean) on COD10K, CAMO, CHAMELEON, NC4K.
- SAM-Adapter: full grid on COD10K, CAMO, CHAMELEON; NC4K clean-only (corrupted NC4K is optional, ~140 h extra).
- Order: COD10K first (main table). Plan in `configs/benchmark.yaml`.

### Done
- **Per-image scores** (`results/per_image/<run>/<model>/<dataset>/<corruption>-<sev>.csv[.gz]`): written as computed (full precision), resume mid-condition, gzipped when the condition is complete (committed; for CIs and significance tests). Verified: dataset metrics = mean of per-image scores (exact).
- **Parallel pipeline** in `src/evaluate.py` (`--workers`, default 2): 1 loader thread (load + corrupt; single thread because ImageNet-C uses a global RNG) + metric threads overlap with GPU inference; results identical to sequential (tested). Measured CPU costs per large image: metrics ~155 ms (wFm 82), corrupt ≤ 57 ms, load ~19 ms. On interruption, already-predicted images are still saved.
- `--max-hours` in `src/evaluate.py`; `scripts/run_benchmark.py` (plan runner, `--status` with ETA from the latest code version's timings, `--max-hours`, `--stage`, `--include-optional`).
- `scripts/summarize_benchmark.py` → `results/summary_robustness.csv`: clean, mCorr, retention (Sm/Em/wFm/Dice/IoU), MAE ratio, mCE (vs SINet-V2) for all / seen / unseen / families / single corruptions; flags incomplete groups.
- 74 tests pass.

### ETA
`run_benchmark.py --status`: ~200 h at the old sequential speed (SAM-Adapter ~72 h of it); the parallel pipeline should cut the fast-model part substantially. Check real speed after the first COD10K conditions.

### Next (student)
```powershell
git checkout main ; git pull
python scripts/run_benchmark.py --status
python scripts/run_benchmark.py --max-hours 10       # repeat as often as convenient; resumes automatically
git add results/ ; git commit -m "benchmark progress" ; git push   # after each session
```

### Next (Claude), while the benchmark runs
- Phase 4: our robust adapter (MobileSAM TinyViT + adapters + learned prompt, training script, corruption augmentation, consistency loss), so training can start as soon as the GPU is free.

---

## Session 9 — 2026-10-16 (Claude, cloud container) — Phase 4 adapter + training script, Phase 6 demo

### Plan change (student)
Demo presentation on **Wednesday 21 Oct**. Order: finish the current `--max-hours 10` benchmark run(s), then **train our model**, run the **demo**, and resume benchmarking after Wednesday. Paper drafting after that.

### Done
- **`src/models/robust_adapter.py`**:
  - frozen MobileSAM TinyViT encoder plus zero-initialised bottleneck adapters in the 10 transformer blocks (LoRA / none as ablations);
  - learned prompt tokens, box-initialised, plus a learned dense prompt; SAM mask decoder (trainable);
  - 3.71 M trainable of 10.33 M total.
- **Checks against the real `mobile_sam.pt`, done in the container:**
  - the encoder at init equals MobileSAM's image encoder exactly (max diff 0.0);
  - our batched decoder equals SAM's decoder per image (≤ 9e-6);
  - box-initialised tokens equal SAM's box prompt embedding.
- Registered `Ours-Clean`, `Ours-Aug` and `Ours-Full` (checkpoints at `checkpoints/ours/<run>/final.pt`). An unfinished checkpoint is reported as `Ours-Full@ep<x>-unfinished`.
- **`src/losses.py`:** structure loss (window scaled with resolution, separable box filter); consistency loss (MSE / symmetric KL, optional stop-gradient).
- **`CODTrainSet` in `src/datasets.py`:**
  - corrupts at the original resolution, then applies the same crop/flip to all views and resizes;
  - seeded by (seed, epoch, index); no photometric augmentation.
- **`src/train.py`:**
  - YAML configs with `base:` inheritance and `--set key=value` overrides;
  - AMP, gradient checkpointing (non-reentrant), gradient accumulation, AdamW with warm-up and cosine schedule, clipping;
  - `--profile N` (speed/VRAM → `results/train/profile.csv`);
  - `time_budget_hours` (cuts epochs to fit after measuring speed);
  - `--max-hours`, mid-epoch resumable `last.pt` (every 15 min, each epoch, and on Ctrl+C), `final.pt`;
  - `results/train/<run>/{config.yaml, log.csv, summary.json}`;
  - refuses unseen corruptions; refuses to resume with a changed config.
- **Configs:** `configs/train/{base, ours_clean, ours_aug, ours_full}` plus ablations (λ 0.1 / 0.5, sym_kl, no stop-grad, LoRA, no adapter).
- **`demo/app.py` (Gradio 6.29.1):**
  - Detector tab: 12 seeded test images (or `demo/examples.txt`) plus `demo/my_photos`, upload, corruption dropdown marking seen/unseen, severity 0–5, side-by-side overlays with latency, GT toggle with per-image Sα/MAE, CPU fallback;
  - Benchmark tab reads `results/*.csv`;
  - smoke-tested headless with screenshots.
- `docs/EXPLAINER.md` §9–10, `docs/DEMO_SCRIPT.md`, README Train/Demo sections, gradio pinned.
- 94 tests pass.

### Not done / open
- No GPU timing yet: epoch time and VRAM must come from `--profile` on the laptop. My rough guess is 15–40 min per epoch for Ours-Full at 1024 px; unverified.
- FPN-head ablation and the EfficientSAM fallback are not implemented (only if needed).
- If a variant is retrained, delete its old rows from `results/benchmark.csv` (resume skips existing rows).

### Next (student), in order
```powershell
git checkout main ; git pull ; uv pip install -r requirements.txt   # adds gradio
# 1. pause the benchmark (let the current --max-hours run end, or Ctrl+C; it resumes later)
# 2. speed/VRAM check (~2 min); look at peak_vram_mb and epoch_minutes
python -m src.train --config configs/train/ours_full.yaml --profile 40
#    if peak_vram_mb > ~3600 or it is very slow: add  --set train.batch_size=1 train.grad_accum=8  (and re-profile)
# 3. train Ours-Full within a budget (the schedule is fitted to it); resumable
python -m src.train --config configs/train/ours_full.yaml --set train.time_budget_hours=10
python -m src.train --config configs/train/ours_full.yaml --resume      # if it was interrupted
# 4. quick check + demo
python -m src.evaluate --models Ours-Full --datasets CHAMELEON --corruptions clean
python demo/app.py
# 5. if time before Wednesday: Ours-Clean (same commands with ours_clean.yaml) for the "with vs without robustness training" comparison
git add results/ ; git commit -m "Ours-Full training" ; git push
```

---

## Session 10 — 2026-10-20 (Claude, cloud container) — Ours-Full trained; training-script fixes

### From the student (laptop)
- `--profile 40` (Ours-Full, 1024 px, batch 2 × accum 4, AMP, grad checkpointing): 0.68 s/batch, **peak 1426 MB** of 4 GB (no VRAM ladder needed), estimated 23 min/epoch.
- **Ours-Full trained** with `time_budget_hours=6`. The budget picked **15 epochs**, which completed in **4.05 h** (about 16 min/epoch: faster than the start-up estimate) on 2026-10-20. Result: `checkpoints/ours/ours_full/final.pt`.
- `--resume` afterwards found the run already complete and re-saved `final.pt` with the same weights. As a side effect it overwrote `results/train/ours_full/summary.json` with empty timings (sec_per_batch 0, peak 74 MB). The real per-step timings and VRAM are in `results/train/ours_full/log.csv`.

### Fixes
- Resuming a finished run now says "already finished" and writes nothing.
- `time_budget_hours` now sizes the run from the **real wall time of the first epoch**, not the first ~30 batches. Those under-estimated speed by about 30% here, so the 6 h budget gave a 4 h run. The cosine schedule for epoch 1 barely changes (warm-up is independent of the total).
- `--set` accepts numbers written like `1e-3`; YAML reads those as text.
- **Decision (student): train every variant for the planned 30 epochs.** `base.yaml` stays at `epochs: 30`. The 15-epoch run is kept, not overwritten: rename its folders to `ours_full_ep15` and it is registered as model `Ours-Full-15ep`, both as the demo fallback and for a 15-vs-30-epoch ablation. A finished cosine schedule can't simply be extended (its learning rate has already decayed to the minimum), so the 30-epoch Ours-Full is a fresh run.
- Demo picks the first *finished* Ours variant (Full, then Full-15ep, Aug, Clean), so a run still training never replaces a finished model on screen.
- 97 tests pass.

### Next (student)
```powershell
# keep the 15-epoch run under its own name (folders only; nothing is retrained)
Rename-Item checkpoints\ours\ours_full ours_full_ep15
Rename-Item results\train\ours_full ours_full_ep15
python -m src.evaluate --models Ours-Full-15ep --datasets CHAMELEON --corruptions clean
# 30-epoch runs (~8 h for Full, roughly half each for Clean / Aug); resume any of them with --resume
python -m src.train --config configs/train/ours_full.yaml
python -m src.train --config configs/train/ours_clean.yaml
python -m src.train --config configs/train/ours_aug.yaml
git add results/ ; git commit -m "Ours training" ; git push
```

---

## Session 11 — 2026-10-21 (Claude, cloud container) — first results for our adapter (CHAMELEON)

### From the student (laptop)
- Trained, all 30 epochs, peak 1.4 GB (Full) / 0.74 GB (Clean, Aug):
  - Ours-Full: 8.48 h
  - Ours-Clean: 4.11 h
  - Ours-Aug: 4.30 h
  - Ours-Full-15ep: 4.05 h (kept for the epochs ablation)
- Evaluated on CHAMELEON, clean + 8 corruptions × 5 severities.

### Results (CHAMELEON, n = 76; from results/benchmark.csv and per-image files)

| Model | Clean Sα | mCorr Sα | Retention | mCE (vs SINet-V2) |
|---|---|---|---|---|
| SAM2-UNet | 0.914 | 0.896 | 0.980 | 0.81 |
| SINet-V2 | 0.888 | 0.870 | 0.980 | 1.00 |
| ZoomNet | 0.902 | 0.867 | 0.961 | 1.03 |
| PFNet | 0.882 | 0.849 | 0.963 | 1.17 |
| Ours-Clean | 0.885 | 0.832 | 0.939 | 1.30 |
| Ours-Aug | 0.876 | 0.839 | 0.957 | 1.24 |
| Ours-Full (30 ep) | 0.883 | 0.847 | 0.959 | 1.18 |

Paired bootstrap on per-image mean Sα under corruption:
- **Full vs Clean:** +0.015 [+0.006, +0.024]. Seen corruptions +0.024 [+0.014, +0.034]; **unseen +0.007 [−0.002, +0.016], not significant**.
- **Full vs Aug:** +0.008 [+0.001, +0.016].
- **30 vs 15 epochs:** +0.020 [+0.010, +0.031].
- **Full vs SINet-V2:** −0.024 [−0.038, −0.010].
- **Full vs SAM2-UNet:** −0.049 [−0.066, −0.033].

**Reading (negative result, reported honestly):**
- Our robustness training helps our own model on the seen corruptions.
- The held-out (unseen) gain is not significant yet on CHAMELEON.
- Our model is less robust than the 352-px specialists and SAM2-UNet.
- Our gap is almost entirely on *local, high-frequency* corruptions: noise, blur, JPEG, rain. Brightness, contrast and fog are on par.

**Hypothesis:** input resolution. The baselines resize ~744×981 images down to 352 px, which averages away noise and shrinks blur. We feed 1024 px, so the model sees the corruption at full strength. SAM-Adapter, also 1024 px, has the lowest retention on COD10K (partial results), which fits. This needs a controlled test.

### Next
- After the demo: evaluate Ours-Clean, Ours-Aug and Ours-Full on COD10K (the main table; n = 2026 is far more reliable than 76).
- Resolution test: evaluate Ours-Full with the input downsampled to 352/512 at test time before the model; optionally train a 512-px variant.

### Same session, later: resolution test + demo model picker
- **Resolution test** (`scripts/resolution_test.py`, EXPLAINER §11):
  - `Ours-Full-in352/512/768` and `Ours-Clean-in352` are the same trained weights fed a low-resolution view (`low_res_view`: PIL bilinear to N×N, as the baselines' preprocessing, then back up).
  - The script evaluates them (resumable) and reports retention on local (noise, blur, JPEG, rain) vs global (brightness, contrast, fog) corruptions, with paired bootstrap CIs → `results/resolution_test_<dataset>.csv`.
  - Already in the existing numbers: Ours-Full retention is 0.995 on global corruptions (SINet-V2 1.000) vs 0.938 on local ones (SINet-V2 0.968).
- **Demo:**
  - "Models to compare" picker: up to 6 of SINet-V2, PFNet, ZoomNet, SAM2-UNet, SAM-Adapter and every trained Ours variant (including Ours-Full-in352), loaded on first use and cached; frees unselected models on CUDA out-of-memory.
  - Uploading a photo runs detection immediately.
  - Each model keeps a fixed colour.
  - Browser-tested with upload + ticking a model.
- 100 tests pass.

### Next (student)
```powershell
git checkout main ; git pull
python scripts/resolution_test.py                 # ~40 min on CHAMELEON (4 variants x 41 conditions)
git add results/ ; git commit -m "resolution test CHAMELEON" ; git push
```

---

## Session 13 — 2026-10-21 (Claude, cloud container) — resolution test result

### From the student
`scripts/resolution_test.py` on CHAMELEON (4 variants × 41 conditions), pushed as `results/resolution_test_CHAMELEON.csv`.

### Result (Ours-Full fed an N-px view; clean Sα / mCorr / retention / local / global / mCE)
- 1024: 0.883 / 0.847 / 0.959 / 0.938 / 0.995 / 1.18
- 768:  0.878 / 0.842 / 0.959 / 0.942 / 0.989 / 1.22
- 512:  0.868 / 0.832 / 0.958 / 0.942 / 0.984 / 1.31
- 352:  0.842 / 0.814 / 0.968 / 0.958 / 0.985 / 1.45
- Ours-Clean at 352: local retention 0.910 → 0.940.
- Reference, SINet-V2: 0.888 / 0.870 / 0.980 / 0.968 / 1.000 / 1.00.

**Reading:**
- **Partly supported.** At 352 px, local-corruption retention rises and closes about two thirds of the gap to SINet-V2.
- **But absolute accuracy falls everywhere** (local −0.023, 95 % CI [−0.036, −0.010]; global −0.049). The model was trained at 1024 px, so a downsampled-then-upsampled input is itself a shift for it. Retention rises only because the clean score falls more.
- **Methodological lesson:** report mCorr/mCE beside retention.
- **Proper test:** train at low resolution.

### Done
- `configs/train/ours_full_512.yaml` and model `Ours-Full-512` (run `ours_full_512`). `resolution_test.py` includes it in the report once trained. EXPLAINER §11 has the result table.

### Next (student)
```powershell
git checkout main ; git pull
python -m src.train --config configs/train/ours_full_512.yaml --profile 40       # check speed first
python -m src.train --config configs/train/ours_full_512.yaml                    # expected much faster than 1024 (estimate)
python -m src.evaluate --models Ours-Full-512 --datasets CHAMELEON --corruptions clean
python -m src.evaluate --models Ours-Full-512 --datasets CHAMELEON --corruptions all
python scripts/resolution_test.py --report-only
git add results/ ; git commit -m "Ours-Full-512" ; git push
```
Then: our variants on COD10K, `python -m src.profile --models Ours-Full`, finish SAM-Adapter.

---

## Session 14 — 2026-10-22 (Claude, cloud container) — COD10K results for our adapter: it beats the specialists

### From the student
- Ours-Clean, Ours-Aug and Ours-Full evaluated on COD10K (41 conditions × 2,026 images).
- `src.profile` run for Ours-Full: 10.33 M params, 83.2 GFLOPs, **45.6 ms (21.9 FPS)**, peak **181 MB**, fp16.

### Results (COD10K; Sα)

| Model | Clean | mCorr | Retention | mCE |
|---|---|---|---|---|
| SAM2-UNet | 0.880 | 0.866 | 0.984 | 0.66 |
| SAM-Adapter (23/40) | 0.884 | 0.849 | 0.960 | — |
| **Ours-Full** | **0.840** | **0.812** | 0.967 | **0.92** |
| Ours-Aug | 0.838 | 0.810 | 0.966 | 0.93 |
| ZoomNet | 0.839 | 0.803 | 0.957 | 0.96 |
| Ours-Clean | 0.842 | 0.799 | 0.948 | 0.98 |
| SINet-V2 | 0.815 | 0.795 | 0.976 | 1.00 |
| PFNet | 0.800 | 0.773 | 0.967 | 1.11 |

**Paired bootstrap on COD10K (mean Sα over the 40 corrupted conditions, 95 % CI):**
- Ours-Full beats every specialist:
    - SINet-V2: +0.017 [+0.013, +0.021]
    - ZoomNet: +0.009 [+0.005, +0.014]
    - PFNet: +0.039 [+0.035, +0.043]
- Ours-Full vs SAM2-UNet: −0.054 [−0.058, −0.050].
- **Robustness training transfers to unseen corruptions** (Ours-Full − Ours-Clean):
    - seen +0.021 [+0.019, +0.023]
    - **unseen +0.005 [+0.003, +0.007]**: motion blur +0.008, **rain +0.015** (a family never seen in training); contrast and fog ±0.001 (not significant; already at ceiling).
- Ours-Aug − Ours-Clean, unseen: +0.004 [+0.002, +0.005]. Ours-Full − Ours-Aug: all +0.002 [+0.001, +0.004]; unseen +0.0015 [+0.0001, +0.0030]. The consistency loss adds a little.

**Reading:**
- On the main test set, Ours-Full is the most accurate and most robust model below SAM2-UNet and SAM-Adapter.
- It is on the speed/accuracy Pareto front: SINet-V2 (28.6 ms) → Ours-Full (45.6 ms) → SAM2-UNet (71.3 ms). It dominates ZoomNet (2.1× faster, 3× fewer parameters, 2.3× less memory, more accurate under corruption) and PFNet.
- CHAMELEON (76 images) had ranked SINet-V2 above us. COD10K (2,026 images) is the decisive set, and SAM-based models gain most on it.
- Retention alone still ranks us below SINet-V2 (0.967 vs 0.976) despite a higher absolute score. Report mCorr/mCE.

### Next
- Ours-Full-512 (PR #18), CAMO and NC4K for our variants, finish SAM-Adapter, then tables, figures and the paper.

---

## Session 15 — 2026-10-23 (Claude, cloud container) — 512-px model, CAMO, NC4K clean

### From the student (Block A done; Block B, Ours-Full NC4K corrupted, running)
- **Ours-Full-512 trained:** 30 epochs, **2.58 h**, 0.152 s/batch, peak 392 MB.
- Profile: **23.1 ms (43 FPS), 22.1 GFLOPs, 87 MB peak**. SINet-V2: 28.6 ms, 24.3 GFLOPs, 134 MB.
- Evaluated on CHAMELEON and COD10K (41 conditions each). Ours-Clean, Ours-Aug and Ours-Full also on CAMO (41) and NC4K (clean).

### COD10K (Sα)

| Model | Params | ms | Clean | mCorr | Retention | Local ret. | Global ret. | mCE |
|---|---|---|---|---|---|---|---|---|
| Ours-Full (1024) | 10.3 M | 45.6 | 0.840 | 0.812 | 0.967 | 0.949 | 0.996 | 0.92 |
| **Ours-Full-512** | 10.3 M | **23.1** | 0.824 | 0.805 | **0.976** | **0.967** | 0.992 | 0.95 |
| ZoomNet | 32.4 M | 94.4 | 0.839 | 0.803 | 0.957 | 0.943 | 0.981 | 0.96 |
| SINet-V2 | 27.0 M | 28.6 | 0.815 | 0.795 | 0.976 | 0.963 | 0.997 | 1.00 |

**Paired bootstrap, COD10K, 95 % CI:**
- 512 − SINet-V2, corrupted: +0.010 [+0.006, +0.014]
- 512 − ZoomNet, corrupted: +0.002 [−0.002, +0.007] (not significant)
- 512 − Ours-Full (1024): clean −0.016 [−0.020, −0.012]; **local corruptions −0.000 [−0.003, +0.003]**; global −0.018 [−0.022, −0.014]

**Reading:**
- **The resolution hypothesis is confirmed once the model is trained at the lower resolution.** At 512 px, retention rises to SINet-V2's level (0.976; local 0.967 vs 0.963). The absolute score under noise, blur, JPEG and rain is unchanged, while clean (−0.016) and global-corruption scores drop. The test-time-only version (`Ours-Full-in512`) lost accuracy everywhere, so the model must learn at the resolution it is used at.
- **Ours-Full-512 beats SINet-V2 on every axis on COD10K:** clean 0.824 vs 0.815, corrupted 0.805 vs 0.795 (significant), equal retention, 1.2× faster, 1.5× less memory, 2.6× fewer parameters, 10 % fewer GFLOPs.
- **A trade-off pair for the paper:** 1024 px for the best accuracy; 512 px for SINet-V2-level robustness at twice the speed, trained in 2.6 h.

### Other datasets (Sα)
- **CHAMELEON:** Ours-Full-512 clean 0.851, mCorr 0.833, retention 0.978 (local 0.970 vs SINet-V2 0.968).
- **CAMO:**
    - Ours-Full: clean 0.816, mCorr 0.785, mCE 1.03.
    - vs SINet-V2 (0.820 / 0.791): −0.006 [−0.018, +0.007], not significant.
    - vs ZoomNet (0.820 / 0.771): +0.014 [+0.001, +0.028].
    - Robustness training +0.018; unseen +0.010 [+0.005, +0.016].
- **NC4K clean:** Ours-Full 0.859 vs SINet-V2 0.847, ZoomNet 0.853, PFNet 0.829 (SAM2-UNet 0.901, SAM-Adapter 0.884).

### Next
- Block B (NC4K corrupted, Ours-Full) is running.
- Optional: Ours-Full-512 on CAMO/NC4K. Then SAM-Adapter, then the table/figure scripts and the paper.

---

## Session 16 — 2026-10-23 (Claude, cloud container) — NC4K under corruption; experiments nearly complete

### From the student (Block B)
Ours-Full on NC4K, 40 corrupted conditions × 4,121 images.

### Result (NC4K, Sα)

| Model | Clean | mCorr | Retention | mCE |
|---|---|---|---|---|
| SAM2-UNet | 0.901 | 0.887 | 0.985 | 0.65 |
| **Ours-Full** | **0.859** | **0.836** | 0.973 | **0.95** |
| SINet-V2 | 0.847 | 0.826 | 0.975 | 1.00 |
| ZoomNet | 0.853 | 0.822 | 0.964 | 1.02 |
| PFNet | 0.829 | 0.801 | 0.967 | 1.15 |

**Paired bootstrap, corrupted, n = 4,121:**
- vs SINet-V2: +0.009 [+0.007, +0.012]
- vs ZoomNet: +0.014 [+0.011, +0.016]
- vs PFNet: +0.035 [+0.032, +0.037]
- vs SAM2-UNet: −0.052 [−0.054, −0.049]

**So, under corruption, Ours-Full beats every specialist significantly on the two large test sets (COD10K, NC4K).** It ties SINet-V2 on CAMO and trails it on CHAMELEON (the two small sets, 250 and 76 images).

### Status
- Our models: complete for the paper's main tables.
- Missing: SAM-Adapter corrupted conditions (COD10K 24/40, CAMO and CHAMELEON 0/40; about 31 h).
- Optional extras: ablations, NC4K for Clean/Aug, 512 px on CAMO/NC4K, seeds.

### Next
- Student: SAM-Adapter sessions (`python scripts/run_benchmark.py --max-hours 10`).
- Claude: Phase 7, `scripts/make_tables.py` + `scripts/make_figures.py`, which read only `results/*.csv`.

---

## Session 17 — 2026-10-24 (Claude, cloud container) — benchmark complete (Checkpoint 4)

### From the student
SAM-Adapter finished: COD10K, CAMO and CHAMELEON, 41 conditions each. NC4K stays clean-only by design (~110 h).

### SAM-Adapter results (Sα; retention; mCE)
- **COD10K:**
    - Clean 0.884 (highest of all models), mCorr 0.853, retention 0.964 (local 0.948, global 0.992), mCE 0.72.
    - −0.013 vs SAM2-UNet under corruption [−0.017, −0.009]; +0.041 vs Ours-Full [+0.037, +0.045].
- **CAMO:** clean 0.847, mCorr 0.811, retention 0.958, mCE 0.90. −0.049 vs SAM2-UNet; +0.027 vs Ours-Full.
- **CHAMELEON:** clean 0.896, mCorr 0.857, retention 0.956, mCE 1.11. Below SINet-V2 (0.870) and ZoomNet (0.867) under corruption; vs Ours-Full +0.010 [−0.006, +0.025], not significant.
- **COD10K Dice retention:** Gaussian noise 0.876 and Gaussian blur 0.892, the lowest of all models except ZoomNet's blur.

**Reading:**
- **Size is not robustness.** The 641 M SAM-Adapter has the best clean score on COD10K but drops behind SAM2-UNet under corruption. On CHAMELEON it falls behind the 27 M SINet-V2.
- **Independent support for the resolution finding.** The two 1024-px models have almost identical local-corruption retention on COD10K: SAM-Adapter 0.948, Ours-Full 0.949. The 352-px models are higher: SINet-V2 0.963, SAM2-UNet 0.978. Ours-Full-512 is at 0.967.
- **Our 10 M model vs the 641 M SAM-Adapter:** 0.04 lower on COD10K under corruption, at 84× lower latency and 62× fewer parameters.

### Status
- Phases 0–4 done. Phase 5 done for all main models. Phase 6 done.
- **Checkpoint 4:** decide the paper framing with the student. Then Phase 7 (tables/figures scripts), Phase 8 (paper) and Phase 9 (release).
