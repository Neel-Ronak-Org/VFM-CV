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
- Push from the cloud container failed (403: Claude GitHub App not installed / not connected for `Neel-Ronak-Org/VFM-CV`).
- Exact archive layout of the SINet-V2 Drive packages is not verified (organiser handles common variants; check the `--verify` output).
- Rain is not comparable with COD10K-C (our approximation).
