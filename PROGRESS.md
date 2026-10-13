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
