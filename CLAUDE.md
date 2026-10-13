# Project Brief: Robust, Lightweight Foundation-Model Camouflaged Object Detection

You are building a complete research project for a 4th-year AI engineering student. The project has three end products:

1. **A working camouflaged object detector (COD)** with an interactive demo.
2. **A corruption-robustness benchmark** comparing specialist COD models with SAM-based ones.
3. **A conference paper**, about 6 pages in IEEE format, for a regional IEEE conference.

Read this whole file before doing anything. Work through the phases in order. Stop at every **🛑 CHECKPOINT** and report to the user before continuing.

---

## 0. Ground rules (non-negotiable)

- **Never fabricate results.** Every number in the paper, README or demo must come from a results file (`results/*.csv`) produced by code in this repo. If an experiment fails, report that honestly.
- **Never invent citations.** Every reference in `paper/refs.bib` must be checked against arXiv, a DOI or the publisher page. Record the verification URL in a comment above each BibTeX entry. If you can't verify a reference, don't cite it.
- **Report negative results honestly.** If our method doesn't beat a baseline on some setting, say so in the paper. Reviewers respect honest limitations; fabricated wins get papers rejected or retracted.
- **Keep a progress log.** After every work session, append to `PROGRESS.md`: what was done, what's next, open problems, and the commands to reproduce. Read `PROGRESS.md` at the start of every session to resume.
- **Teach as you go.** The student must be able to explain and defend this work to their professor and conference reviewers. For each major component, add a short plain-English explanation to `docs/EXPLAINER.md`: what it is, why we chose it, and how it works.
- **Ask before anything expensive or irreversible.** That includes downloads over 5 GB, deleting data, or changing the research direction.
- **AI-use disclosure.** Remind the user to check their course's and the target conference's policy on AI-assisted research and writing. Leave a placeholder acknowledgement in the paper.

---

## 1. Hardware and environment constraints

- **GPU:** NVIDIA RTX 3050 Laptop, usually **4 GB of VRAM** (sometimes 6 GB). Run `nvidia-smi` first and record the real VRAM in `PROGRESS.md`.
- **Operating system:** detect it. On Windows, prefer WSL2 (Ubuntu) if it's available. If not, use native Windows and avoid MATLAB-only tooling.
- **Python environment:** use Python 3.10 or 3.11 with a recent PyTorch build that has CUDA, managed with `uv` or conda. Pin versions in `requirements.txt`.
- **Memory rules:**
  - Always use mixed precision (`torch.autocast`, fp16/bf16).
  - Use small batches (1–4) with gradient accumulation.
  - Use gradient checkpointing inside the frozen encoder when training adapters.
  - Run all large-model evaluation in inference mode, fp16, batch size 1.
- **Old baseline repos** often target Python 3.6–3.8 and PyTorch 1.x. **Don't create old environments.** Port each model's inference code to our environment through a wrapper (§5). If a model truly can't be ported, use its officially released prediction maps for the clean-image numbers, and log that it's missing from the corruption benchmark.

---

## 2. Research question and contribution

**Working title:** *How Robust Are Foundation-Model Camouflage Detectors? A Corruption Benchmark and a Lightweight Robust SAM Adapter*

**Background, to verify and cite in Phase 0:**
- COD10K (Fan et al., CVPR 2020; TPAMI 2022) is the standard COD dataset. The SINet and SINet-V2 baselines come from these papers.
- COD10K-C (Sayem, 2026, arXiv:2606.02603) benchmarked corruption robustness for SINet-v2, PFNet and ZoomNet, using 8 corruption types at 5 severities on the COD10K test set. They found blur hurts most. It included **no SAM-based models**, and its code was "to be released" at the time of planning.
- SAM-Adapter (ICCV 2023 workshop), SAM2-Adapter and SAM3-Adapter (arXiv:2511.19425) report state-of-the-art clean COD results. Their robustness to corruption on COD is untested.
- General SAM robustness studies (arXiv:2305.16220, arXiv:2306.07713) found SAM robust to most corruptions except blur.
- MobileSAM (arXiv:2306.14289) and EfficientSAM (arXiv:2312.00863) are lightweight SAM variants.

**Contributions:**
1. **Benchmark:** the first corruption-robustness evaluation of SAM-family COD models (SAM, SAM2 and SAM3 adapters, plus lightweight SAM variants) against specialist COD models, across COD10K, CAMO and NC4K.
2. **Method:** a lightweight, prompt-free SAM adapter (MobileSAM or EfficientSAM backbone, frozen) trained with corruption augmentation plus a clean/corrupted consistency loss. It's designed to train and run on a 4 GB laptop GPU.
3. **Held-out-corruption protocol:** train the augmentation on some corruption types, then test on **unseen** types. This shows real robustness, not just memorized augmentations, and is a key extra over COD10K-C.
4. **Analysis:** which corruption families break which model family, whether robustness shrinks with model size, and the accuracy, robustness and speed trade-off on consumer hardware.

---

## 3. Repository layout

```
cod-robust/
├── CLAUDE.md                 # this file
├── PROGRESS.md               # running log (you maintain)
├── README.md                 # setup, reproduce, demo instructions
├── requirements.txt
├── configs/                  # YAML configs for every experiment
├── data/                     # datasets (gitignored)
│   ├── raw/                  # original downloads
│   └── corrupted/            # generated corrupted test sets
├── checkpoints/              # downloaded and trained weights (gitignored)
├── src/
│   ├── datasets.py           # COD dataset loaders (image + mask)
│   ├── corruptions.py        # corruption generation, deterministic seeds
│   ├── metrics.py            # wraps PySODMetrics
│   ├── models/
│   │   ├── base.py           # common interface: predict(image) -> mask in [0,1]
│   │   ├── sinet_v2.py       # baseline wrappers…
│   │   ├── ...
│   │   └── robust_adapter.py # OUR model
│   ├── train.py
│   ├── evaluate.py           # runs any model over any (dataset, corruption, severity)
│   └── profile.py            # params, FLOPs, FPS, peak VRAM
├── scripts/                  # download, generate_corruptions, run_benchmark, make_figures, make_tables
├── results/                  # CSVs (source of truth for all numbers)
├── figures/                  # generated by scripts/make_figures.py
├── demo/app.py               # Gradio demo
├── docs/EXPLAINER.md
└── paper/                    # LaTeX (IEEEtran conference), refs.bib, figures/
```

---

## 4. Phases

### Phase 0: Novelty check and setup (do this first)

1. Run an environment check (`nvidia-smi`, Python, CUDA) and record it.
2. **Literature search for scooping.** Search arXiv, Google Scholar and Papers with Code for work since 2023 that combines camouflaged object detection with SAM or foundation models and with corruption or robustness or blur or noise. Also check:
   - whether COD10K-C has released code or data, or added SAM models;
   - any newer COD robustness benchmarks.
3. Write `docs/NOVELTY_REPORT.md`: the closest 5–10 papers, what each covers, and a clear verdict on whether our contribution is still open.
   - If it has been scooped, propose 2–3 pivots that reuse the same pipeline: for example, focus on unseen-corruption generalization, a test-time adaptation angle, or the efficiency-robustness trade-off.
4. Set up the repo skeleton and environment.

**🛑 CHECKPOINT 0:** Show the user the novelty report and wait for a go/pivot decision.

### Phase 1: Data

- **Training set (standard protocol):** COD10K-train (3,040 images) + CAMO-train (1,000 images) = **4,040 images**.
- **Test sets:** CHAMELEON (76), CAMO-test (250), COD10K-test (2,026), NC4K (4,121).
- **Sources:** the SINet-V2 repo (github.com/GewelsJI/SINet-V2) links ready-made train/test packages. NC4K comes from its authors' repo (Lv et al., CVPR 2021). Some links are on Google Drive (use `gdown`) or Baidu. **If a download is blocked, ask the user to download it manually** and give exact instructions.
- Write `scripts/download_data.py` and `src/datasets.py`. Verify image and mask counts, alignment, and that masks are binary.
- Log dataset licences and citations.

### Phase 2: Evaluation harness and baseline reproduction

1. **Metrics:** use PySODMetrics (`pip install pysodmetrics`, github.com/lartpang/PySODMetrics), the standard Python version of the COD evaluation toolbox. Report:
   - Sα (structure measure), Eφ mean, Fβw (weighted F-measure), MAE (all standard);
   - Dice and IoU, so results can be compared with COD10K-C.
2. **Common model interface (`src/models/base.py`):** `predict(img: np.ndarray RGB) -> np.ndarray float mask [H,W]`, resized back to the original resolution.
3. **Baselines.** Check that weights are available before writing each wrapper, and log anything unavailable:
   - Specialists: **SINet-V2** (required), **PFNet**, **ZoomNet**, **FEDER**.
   - Foundation-model based: **SAM-Adapter** (github.com/tianrun-chen/SAM-Adapter-PyTorch; ViT-B, plus L/H if they fit in fp16), **SAM2-Adapter** (the "SAM2-Adapter" branch of the same repo), **SAM3-Adapter** (arXiv:2511.19425) if weights are released and inference fits in VRAM.
   - Optional analysis: zero-shot SAM / MobileSAM prompted with the ground-truth box. This is an "oracle prompt" upper bound; label it clearly as not a fair competitor.
4. **Reproduction check:** run each baseline on the *clean* test sets. Compare against the numbers each paper reports (Sα within ±0.015 is acceptable). Record this in `results/reproduction.csv` and explain any gaps.

**🛑 CHECKPOINT 2:** Show the reproduction table. Do not continue if SINet-V2 can't be reproduced.

### Phase 3: Corruption benchmark

1. **Corruptions:** use the `imagecorruptions` package (the ImageNet-C implementation). First check §3 of the COD10K-C paper and **match their 8 corruption types and severity definitions if they're stated**, so results are comparable. Otherwise use these 8, covering the same families:
   - `gaussian_noise`, `shot_noise`
   - `motion_blur`, `defocus_blur`
   - `fog`, `brightness`, `contrast`
   - `jpeg_compression`

   Use all 5 severities.
2. **Generation:** generate the corrupted test sets once, with fixed seeds, into `data/corrupted/`. Do this for COD10K-test at minimum, and for CAMO-test and NC4K if disk allows. Check disk space first; JPEG at quality 95 is fine for storage, **except** for the jpeg_compression corruption itself, which must be saved as PNG.
3. **Held-out split** for contribution 3. Pick one corruption from each family as "seen", for example `gaussian_noise`, `defocus_blur`, `brightness`, `jpeg_compression`. The others are "unseen": `shot_noise`, `motion_blur`, `fog`, `contrast`. Document this choice and its reasoning.
4. **Run every available model** on every (dataset, corruption, severity) combination. Save per-image predictions only for a small qualitative subset; save metrics for everything to `results/benchmark.csv`.
5. **Summary metrics:**
   - **mCorr:** the mean of each metric over all corruptions × severities.
   - **Retention** = mCorr / clean score (higher is more robust). Report it separately for seen and unseen corruptions.
   - Optionally, an mCE-style error normalized to SINet-V2.

### Phase 4: Our method (Robust Lightweight SAM Adapter)

**Architecture (start simple):**
- **Encoder:** MobileSAM's TinyViT image encoder, frozen (download its official weights).
  - Fallback: EfficientSAM-Ti or EfficientSAM-S.
- **Adapters:** lightweight bottleneck adapters, or LoRA, inserted into the encoder blocks. Follow the SAM-Adapter design idea of adding task-specific information through adapters while keeping the backbone frozen.
- **Prompt-free decoding:** use SAM's mask decoder with a **learned prompt embedding**, no user clicks. Alternatively, a small FPN-style decoder head; try both if time allows.
- **Output:** a single-channel camouflage map, upsampled to the input size.

**Training:**
- **Loss:** weighted BCE + weighted IoU (the standard structure loss used by SINet-V2 and PraNet).
- **Corruption augmentation:** apply a random *seen* corruption at random severity to 50% of the samples.
- **Consistency loss:** for each image, pass both a clean and a corrupted view and add λ · (symmetric KL or MSE between the two sigmoid maps). Stop gradients on the clean view in one variant; ablate both variants.
- **Optimizer:** AdamW, cosine schedule, roughly 30–60 epochs. Measure the epoch time first and pick an epoch count that fits in about 24 GPU-hours total.

**VRAM ladder.** Profile first and step down only as far as needed:
1. 1024×1024 input, batch size 2, AMP, gradient checkpointing.
2. Reduce to batch size 1 with gradient accumulation of 8.
3. Reduce the input to 512 with interpolated position embeddings (log the accuracy cost).
4. Switch to the EfficientSAM-Ti backbone.

**Variants to train:**
- **Ours-Clean:** adapter only, no corruption augmentation, no consistency loss.
- **Ours-Aug:** + corruption augmentation.
- **Ours-Full:** + augmentation + consistency loss.
- Ablations: λ ∈ {0.1, 0.5, 1.0}; adapter vs LoRA; SAM decoder vs FPN head.
- If time allows, run 3 seeds for Ours-Full and report mean ± standard deviation.

**🛑 CHECKPOINT 4:** Show clean and corrupted results for all variants against the baselines, then discuss with the user how to frame the paper based on the *actual* results.

### Phase 5: Efficiency profiling

For every model that runs locally, record parameters, GFLOPs, FPS at batch size 1 in fp16 on the student's GPU, and peak VRAM. Save to `results/efficiency.csv`. This supports the "runs on a laptop" claim.

### Phase 6: Demo (the working detector)

Build `demo/app.py` with **Gradio**:
- Image upload, plus a gallery of about 12 bundled test images. Also include a folder where the student can drop their own photos.
- Corruption controls: a dropdown for corruption type and a slider for severity from 0 to 5.
- **Side-by-side output:** input | SINet-V2 mask | Ours-Full mask, as coloured overlays on the image. Show the latency of each model.
- Optional checkbox: show the ground truth when the image comes from a test set.
- Must run on the laptop GPU and fall back to CPU (slower) if no GPU is found.
- Document `python demo/app.py` in the README. Record a 60-second screen capture script or shot list in `docs/DEMO_SCRIPT.md` so the student can present it.

### Phase 7: Figures and tables (all generated by script)

`scripts/make_figures.py` and `scripts/make_tables.py` read **only** from `results/*.csv`. They produce:
- **Table 1:** clean results on 4 datasets (Sα, Eφ, Fβw, MAE).
- **Table 2:** mCorr and retention on COD10K-test, split into seen and unseen corruptions.
- **Table 3:** ablations.
- **Table 4:** efficiency.
- **Fig. 1:** teaser — one image under increasing blur, baseline vs ours.
- **Fig. 2:** architecture diagram (draw it with TikZ or matplotlib; keep it simple).
- **Fig. 3:** performance-vs-severity curves for each corruption family.
- **Fig. 4:** robustness (retention) vs FPS scatter plot.
- **Fig. 5:** qualitative grid, including failure cases.

Tables go out as LaTeX `.tex` files that the paper `\input`s, so the numbers can never be typed in by hand.

### Phase 8: Paper

- Use the IEEE conference template (`IEEEtran`, `conference` option), about 6 pages plus references.
- **Sections:**
  1. Abstract (write it last)
  2. Introduction: motivation (wildlife monitoring, search and rescue, inspection, where cameras blur and compress), the gap, our 3–4 contributions
  3. Related Work: COD; foundation models and SAM adapters; robustness benchmarks
  4. Benchmark: datasets, corruptions, held-out protocol, metrics
  5. Method
  6. Experiments: setup, main results, seen vs unseen corruptions, ablations, efficiency, qualitative results
  7. Limitations: synthetic corruptions ≠ real-world degradation; large models only evaluated, not retrained; a single GPU class
  8. Conclusion and future work
- **References:** expect roughly 25–35. Cite the original sources for every dataset, baseline, metric and tool. Every entry must be verified (§0).
- **Author details:** placeholders for the student's name, affiliation and email, plus an acknowledgement placeholder (advisor, AI-use disclosure as required by the venue).
- **Writing quality:** clear, plain academic English, no hype words. Every claim must link to a table or figure.
- Compile with `latexmk` if it's available. Otherwise tell the user to upload `paper/` to Overleaf.
- Also produce `paper/SUBMISSION_CHECKLIST.md`: page limit, anonymization if the venue requires it, figure resolution, reference format, and the user's own steps (choose the venue, read the call for papers, get advisor review).

**🛑 CHECKPOINT 8:** Hand over the PDF and checklist. Recommend the student read the whole paper and `docs/EXPLAINER.md` before submitting.

### Phase 9: Release

- Clean up the README (setup, data, reproduce each table with one command, demo).
- Release code, configs, the corruption generation script with seeds, and the trained Ours-Full checkpoint, so others can reproduce the benchmark.
- Add a licence (MIT for code). Respect each dataset's own licence and don't redistribute datasets.

---

## 5. Engineering conventions

- Every experiment is driven by a YAML config in `configs/` and logs its config, git hash, seed and timings next to its results.
- Use deterministic seeds for corruption generation and for data splits.
- Make evaluation resumable: skip any (model, dataset, corruption, severity) combination already in `results/benchmark.csv`.
- Use one wrapper file per baseline, with a short note at the top on its source repo, commit or release, weights URL and any porting changes.
- Write small unit tests for the metrics (compare against a known PySODMetrics example), the corruption determinism and the dataset loaders.
- Never commit datasets or large checkpoints. Use `.gitignore`.

## 6. Risk register (check when stuck)

| Risk | Mitigation |
|---|---|
| Scooped during the project | Post a preprint on arXiv as soon as Phase 4 results exist; pivot options are in the novelty report |
| A baseline's weights are unavailable or won't port | Use its released clean prediction maps, mark it "clean only", and keep going with at least SINet-V2 plus one SAM-Adapter |
| Out of memory on 4 GB | Follow the VRAM ladder in Phase 4; evaluate big models in fp16 at batch size 1; skip ViT-H if it still doesn't fit |
| Our method doesn't beat baselines on clean data | Expected and fine; the paper's claim is robustness per unit of compute, especially on unseen corruptions |
| Our method doesn't improve robustness | The benchmark (contributions 1, 3, 4) is still publishable; frame the method as a negative or ablation result honestly |
| Training is too slow | Fewer epochs, 512 input, or cache frozen-encoder features for clean-only variants |

---

## 7. First actions when you start

1. Read `PROGRESS.md` if it exists; otherwise create it.
2. Run Phase 0, steps 1–3.
3. Stop at **CHECKPOINT 0**.
