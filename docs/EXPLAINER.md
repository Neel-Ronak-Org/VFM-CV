# Plain-English Explainer

Read this before meeting your professor or answering reviewers. Each section answers: **what is it, why did we choose it, how does it work.** New sections are added as components are built.

---

## 1. Camouflaged object detection (COD)

**What:** Given a photo, output a map that is bright where a hidden object is (a stick insect, a flounder on sand) and dark elsewhere. It is binary segmentation where the object is designed by nature to look like the background.

**Why it's hard:** Normal object detectors rely on the object standing out by colour, edge or contrast. Camouflage removes exactly those cues, so the model must use subtle texture and shape differences.

**How it's evaluated:** On standard test sets (CHAMELEON, CAMO, COD10K, NC4K) we compare the predicted map with a hand-drawn ground-truth mask using the metrics in §3.

## 2. Corruption robustness and the held-out protocol

**What:** Real cameras blur, add sensor noise, compress to JPEG, and shoot through fog and rain. A *corruption benchmark* takes clean test images, applies these effects synthetically at 5 strengths (severities), and measures how much each model's score drops.

**Which corruptions, and why these:** We use the **same 8 corruptions and the same severity values as COD10K-C** (Sayem 2026, the only existing COD corruption benchmark), so our numbers for SINet-V2 can be compared with theirs and reviewers can't say we picked easy settings.

| Corruption | Severity 1 → 5 | What it simulates |
|---|---|---|
| Gaussian noise | σ = 8, 16, 24, 32, 40 (pixel units) | sensor noise in low light |
| Motion blur | horizontal kernel k = 3 … 11 px | camera shake / moving camera |
| Gaussian blur | radius 0.8 … 2.5 px | out-of-focus lens |
| Brightness | × 0.85 … 0.30 | under-exposure |
| Contrast | factor 0.85 … 0.30 | haze, cheap sensors |
| Fog | blend with white, α = 0.12 … 0.44 | fog / mist |
| JPEG | quality 70 … 20 | compression on drones / phones |
| Rain | streak density 0.2 % … 1 %, length 10 … 30 px | rain on the lens / in the scene |

**Assumptions we had to make** (COD10K-C gives the parameters but no code), to be stated in the paper:
1. Gaussian noise is added independently per pixel and per RGB channel, then clipped to [0, 255] and rounded.
2. Motion blur is a normalised 1×k box filter (OpenCV `filter2D`, reflected borders).
3. Gaussian blur "radius" is PIL's `GaussianBlur(radius)`. PIL's radius is the standard deviation.
4. Brightness and contrast use PIL `ImageEnhance` (brightness multiplies every pixel by the factor; contrast blends toward the image's mean grey level).
5. Fog is exactly `(1−α)·image + α·255`.
6. JPEG uses PIL with default chroma subsampling.
7. **Rain is our own design.** The paper only says "synthetic streaks at different density and length". Ours: random seed pixels (density), smeared along an 80° line (length), lightly blurred, and blended toward white at up to 60 % opacity. Rain results are therefore *not* directly comparable with COD10K-C; say so in the paper.

`python scripts/preview_corruptions.py` draws every corruption at every severity so you can see them.

**How we keep it reproducible:** Every corrupted image gets its own random seed, computed from the dataset name, image name, corruption and severity with SHA-256. The same image always gets the same noise or rain, on any computer and in any order. COD10K-C also "applies the corruptions at evaluation time in a fixed way", and we do the same instead of storing 80k+ files.

**Held-out (seen/unseen) protocol:** We split the 8 corruptions into 4 *seen* and 4 *unseen*. Our model only ever trains with the seen ones.

| Family | Seen (used in training augmentation) | Unseen (test only) |
|---|---|---|
| Noise | Gaussian noise | — (optional extra: ImageNet-C shot noise) |
| Blur | Gaussian blur | Motion blur |
| Illumination | Brightness | Contrast |
| Weather | — | Fog, Rain |
| Digital | JPEG | — |

Why this split: each unseen blur and illumination corruption has a *related* seen one (does training on Gaussian blur help with motion blur?), and weather is held out completely (does robustness transfer to a new kind of degradation?). Noise and JPEG have only one member each in COD10K-C, so they must be seen. To also test unseen noise, the optional `extended_unseen` set adds four ImageNet-C corruptions (shot noise, defocus blur, zoom blur, pixelate), clearly marked as outside the COD10K-C protocol. If the model improves on unseen corruptions, it learned real robustness, not just the augmentations. COD10K-C did not test this.

**Summary numbers:**
- **mCorr** = average score over all corruptions × severities.
- **Retention** = mCorr ÷ clean score. 0.90 means the model keeps 90 % of its clean performance under corruption. Higher is more robust. COD10K-C reports this for Dice.

## 3. Metrics

All come from PySODMetrics, which was checked against the official MATLAB toolbox, so our numbers are comparable with published papers.

| Metric | Plain meaning | Better |
|---|---|---|
| **Sα (S-measure)** | Does the predicted map have the right *structure*: object shape and region layout? Mixes region-level and object-level similarity. | higher |
| **Eφ (E-measure, mean)** | Does the map match the mask both pixel-by-pixel and globally? Averaged over all 256 binarisation thresholds. | higher |
| **Fβw (weighted F-measure)** | Precision/recall, but errors near the object count more, and the soft map is used directly instead of being thresholded. | higher |
| **MAE** | Average absolute pixel error between map and mask. | lower |
| **Dice / IoU** | Overlap between the thresholded (≥0.5) map and the mask. Added to compare with COD10K-C. | higher |

A perfect binary map gives Eφ slightly below 1.0 (≈0.997). That is expected: at threshold 0 every pixel counts as foreground.

## 4. One interface for every model

Every baseline gets a small wrapper with one method: `predict(rgb_image) → float map in [0,1] at the original resolution`. The evaluation code doesn't care whether the model is SINet-V2 or a SAM variant, which keeps the comparison fair: same images, same corruptions, same metrics code. Old baselines (written for PyTorch 1.x) are *ported* into these wrappers instead of running in old environments.

## 5. SINet-V2, our reference specialist

**What:** SINet-V2 (Fan et al., TPAMI 2022) is the standard COD baseline from the team that built COD10K. It copies how a predator hunts: first *search* for a rough location, then *identify* the exact shape.
- A Res2Net-50 backbone (a ResNet variant that sees several scales inside each block) extracts features.
- "Receptive field" blocks widen what each feature sees.
- A *neighbour connection decoder* fuses the three deepest feature levels into a coarse map (search).
- Three *group-reversal attention* stages refine it (identify). Each stage looks at what is *not yet* predicted (1 − sigmoid of the current map), which pushes the network to recover missed object parts and edges.

**Why it's our reference:** Every COD paper compares against it, COD10K-C used it, and its code and weights are public. If we can reproduce its published numbers, our harness is trustworthy.

**How we run it:** Exactly like the official test script: resize to 352×352, normalise, take the final output, upsample the raw scores to the original size, apply a sigmoid, then **min-max normalise each map to [0, 1]**. That last step is part of the official protocol, but be ready to discuss it. It stretches *any* prediction to full contrast, so a model that is unsure everywhere still outputs a confident-looking map. Under heavy corruption this can inflate false positives. The wrapper has a switch to turn it off for an ablation.

## 6. How we check that a ported baseline is correct

Old baselines were written for PyTorch 1.x. We copy their network code unchanged and run it in our environment, then check it two ways:
1. **Against the paper:** our clean Sα must be within ±0.015 of the published Sα (`configs/reported.yaml` stores each published value with the URL where we confirmed it).
2. **Against the authors' own prediction maps:** we score their released maps with *our* metrics code. If our predictions score the same as their maps, the port is correct, even if the paper's number came from an older evaluation toolbox.

`scripts/make_reproduction.py` puts both comparisons in `results/reproduction.csv`.

## 7. The baselines and why each is in the benchmark

| Model | Family | Backbone (params) | Input | Why it's here |
|---|---|---|---|---|
| SINet-V2 (TPAMI'22) | specialist | Res2Net-50 (27M) | 352 | the standard reference; in COD10K-C |
| PFNet (CVPR'21) | specialist | ResNet-50 (46.5M) | 416 | in COD10K-C |
| ZoomNet (CVPR'22) | specialist | ResNet-50, 3 input scales (32.4M) | 192/384/576 | in COD10K-C; multi-scale input may change its blur robustness |
| FEDER (CVPR'23) | specialist | ResNet-50 + learnable wavelets (44.1M) | 384 | a *frequency-based* method; blur and noise attack exactly the frequencies it relies on |
| SAM-Adapter (ICCVW'23) | SAM-based | SAM ViT (B/L/H, detected from checkpoint) + adapters | 1024 | the original "adapt SAM to COD" model |
| SAM2-UNet (2024) | SAM2-based | SAM2 Hiera-L, frozen + adapters (216M) | 352 | our SAM2 representative (SAM2-/SAM3-Adapter released no COD weights) |
| Oracle-Box MobileSAM | SAM, **oracle** | TinyViT (10M) | 1024 | an upper bound, *not* a competitor (see below) |

**Two ways a model can post-process its output, and why it matters for robustness.** SINet-V2, FEDER and SAM2-UNet min-max normalise every map (§5). ZoomNet, PFNet and SAM-Adapter output the raw sigmoid. We keep each model's *official* choice so the clean numbers reproduce, and we report which one each uses. Min-max can hide a model's uncertainty under heavy corruption, so the paper should mention it whenever a min-max model looks surprisingly robust.

**FEDER is "clean-only".** Its trained weights are no longer downloadable, so we can't run it on corrupted images. We score the authors' released prediction maps instead, and those exist only for CHAMELEON, CAMO and NC4K (no COD10K). FEDER therefore appears only in the clean-results table. The brief allows this fallback; the paper should say so in one sentence.

**SAM-Adapter on a 4 GB GPU.** ViT-H has 632M parameters. Run naively, its fp32 weights plus the attention maps of its four global-attention layers (4096×4096 per head, 16 heads) need more than 4 GB, and Windows silently spills into system RAM (~16 s per image). Our wrapper keeps the encoder weights in fp16 and computes attention a few heads at a time. The maths is identical, and a unit test checks it. That trick is called *attention chunking*, and it's a standard way to trade a little speed for a lot of memory.

**Training-data note:** most baselines train on COD10K-train + CAMO-train (4,040 images). FEDER's README says COD10K-train only (3,040 images). The paper's baseline table should say so.

**The oracle (zero-shot MobileSAM with the ground-truth box).** We give SAM a box drawn around the true object, so it only has to *outline* the object, not *find* it. No real detector has that box, which is why it is labelled "oracle, not a fair competitor". It answers two useful questions:
1. How much of COD's difficulty is *finding* the object rather than *segmenting* it? The gap between the oracle and the real models measures this.
2. Does corruption hurt SAM's *segmentation* even when the location is given? If the oracle stays robust while SAM-based COD models drop, the weakness lies in localisation.

**What the oracle showed (Checkpoint 2, `figures/qual_CHAMELEON_clean.jpg`).** The oracle segments compact, clearly bounded animals well (Sα 0.89–0.95). It fails in two specific situations:
- **The box covers most of the image.** SAM can't tell what the box refers to and fills the whole box, sometimes even *inverting* the mask (on one spider it marked the background as the object).
- **The object is thin or broken up** (legs, stick insects). SAM fills the box instead of outlining the object.

So *knowing where the object is* does not solve camouflage for a zero-shot SAM. Outlining thin parts and frame-filling objects stays hard, and that is exactly where the trained COD models do better. This is a useful finding for the paper, and it shows the oracle is a probe, not an upper bound in every case.

## 8. Running the corruption benchmark (Phase 3)

**What runs:** every model on 8 corruptions × 5 severities, plus clean images, on each test set. The plan is in `configs/benchmark.yaml`: the five fast models on all four datasets, SAM-Adapter on COD10K, CAMO and CHAMELEON. SAM-Adapter on NC4K is optional because it would add about 6 days of laptop time for little gain. COD10K comes first because it's the paper's main table, as in COD10K-C.

**Why per-image scores are saved:** every metric we report is an average over images. We checked that the mean of per-image scores equals the toolbox's dataset score exactly. Saving each image's scores therefore (1) lets a crash resume mid-condition and lose at most one image, and (2) gives us the data for confidence intervals and paired significance tests later. "Is model A's robustness really better than model B's, or is it noise?" is the first question a reviewer will ask.

**Why it's faster than a simple loop:** the GPU prediction is often the *smallest* part of the per-image cost. Loading, corrupting and especially computing the six metrics (weighted F-measure alone is ~80 ms on a large image) take longer. So these run in background threads while the GPU predicts the next image. A test checks the results are identical to the simple sequential loop.

**Summary numbers** come from `scripts/summarize_benchmark.py`:
- **mCorr:** mean score over the group of corruptions;
- **retention:** mCorr ÷ clean score;
- **mCE:** corruption error relative to SINet-V2, the ImageNet-C convention.

Each is reported for all corruptions, seen and unseen ones, each family, and each single corruption.

## 9. Our model: the Robust Lightweight SAM Adapter (Phase 4)

**In one sentence:** we take MobileSAM, a small version of Meta's Segment Anything Model, freeze its image encoder, add about 0.2 M trainable "adapter" parameters inside it, and teach its mask decoder to find camouflaged objects without any user clicks, while training on corrupted images so it stays reliable when the camera blurs, adds noise or compresses.

**The parts** (`src/models/robust_adapter.py`):

| Part | What it does | Trained? | Parameters |
|---|---|---|---|
| TinyViT image encoder (MobileSAM) | turns a 1024×1024 image into a 64×64 grid of 256-number feature vectors | **frozen** | 6.07 M |
| Bottleneck adapters, one per transformer block (10 blocks) | small two-layer networks added next to each block's MLP: `x + MLP(x) + Up(GELU(Down(LN(x))))` | yes | ≈0.2 M |
| Learned prompt tokens (4) + learned dense prompt | replace the clicks/boxes SAM normally needs | yes | ≈1 k |
| SAM mask decoder | a small two-way transformer: prompt tokens attend to image features and vice versa, then upsample to a mask | yes (starts from MobileSAM) | ≈3.5 M |

In total about 3.7 M of the 10.3 M parameters are trained (`count_params()`).

**Why freeze the encoder?** A foundation model has already learned general visual features from 11 M images. Retraining all of it on our 4,040 camouflage images would be slow on a 4 GB GPU and would risk forgetting those features. Adapters are the standard compromise (SAM-Adapter, AdaptFormer, LoRA): the big network stays fixed and small add-on layers steer it towards the new task.

**Why do adapters start at zero?** Each adapter's last layer is initialised to zero, so at the start of training the encoder gives *exactly* MobileSAM's original features. A unit test checks this to five decimal places. Training then only moves away from the pretrained model as far as the data requires.

**How "prompt-free" works.** SAM normally needs a click or a box to know which object to segment. We give it four learned vectors instead. Two start as SAM's own embedding of "a box around the whole image" and two start small and random. During training they learn to mean "the camouflaged object in this image". SAM's original decoder also repeats every image once per prompt, which would mix up the images in a batch. Our `decode()` does the same maths once per image, and a test checks it matches SAM's decoder.

**Training** (`src/train.py`, configs in `configs/train/`):
- **Loss:** weighted BCE + weighted IoU, the "structure loss" of PraNet and SINet-V2. Pixels near the object boundary count up to 6× more, because boundaries are where camouflage is hardest. The boundary window is scaled from PraNet's 31 px at 352 px input to our resolution.
- **Augmentation:** random crop (75–100% of each side) and horizontal flip only. We deliberately use **no colour jitter or noise**, because those overlap with our test corruptions and would blur the line between Ours-Clean and Ours-Aug.
- **Corruption augmentation (Ours-Aug):** 50% of training images get one of the four **seen** corruptions (Gaussian noise, Gaussian blur, brightness, JPEG) at a random severity 1–5. The script refuses to train on unseen corruptions, which protects the held-out protocol.
- **Consistency loss (Ours-Full):** every image gives a clean view and a corrupted view with the same crop and flip. The supervised loss is averaged over both views, so half the supervised views are corrupted, the same as Ours-Aug. We add λ × the mean squared difference between the two predicted maps. With *stop-gradient* on the clean view, the clean prediction acts as a fixed teacher: the model learns to make its corrupted-image answer match its clean-image answer, rather than making the clean answer worse.
- **Variants:** Ours-Clean (no corruption), Ours-Aug (+ augmentation), Ours-Full (+ consistency, λ = 1). Ablations: λ ∈ {0.1, 0.5}, symmetric KL instead of MSE, no stop-gradient, LoRA instead of bottleneck adapters, and no adapters (decoder only).
- **Optimiser:** AdamW, learning rate 2e-4 with 100 warm-up steps then cosine decay, effective batch 8 (2 × 4 accumulation), fp16 mixed precision, gradient clipping at 1.0.
- **Memory tricks:** the frozen convolutional first stage runs without gradients; gradient checkpointing recomputes each transformer block's activations during the backward pass instead of storing them.
- **VRAM ladder**, if `--profile` shows the GPU is nearly full or slow: (1) 1024 px, batch 2; (2) `--set train.batch_size=1 train.grad_accum=8`; (3) `--set data.img_size=512` (smaller image, lower accuracy expected; log the cost); (4) EfficientSAM backbone (not implemented unless needed).
- **Reproducibility:** the image order and every crop, flip and corruption are seeded by (seed, epoch, image index). A run stopped by `--max-hours`, Ctrl+C or a crash resumes with `--resume` on exactly the same batch sequence. Each run logs its config, git hash, loss curve and timings in `results/train/<run>/`.
- **Model selection:** we use the **last** epoch and never pick checkpoints by test-set score. A checkpoint from an unfinished run is labelled `Ours-Full@ep7.3-unfinished` in every output, so it can never be mistaken for the final model.

**How novel is it? (be ready for this question)** Each ingredient is known: frozen SAM with adapters (SAM-Adapter), prompt-free SAM decoding, corruption augmentation, and consistency losses (AugMix and others). What's new is the combination for camouflage on a 4 GB laptop GPU, together with the **held-out-corruption evaluation**: training on four corruption types and testing on four others. That tests whether robustness transfers rather than just memorising the augmentations. RobustSAM (CVPR 2024; consistency losses to make SAM robust) and RobustCODLite (COD10K-C's own lightweight model, with corruption augmentation and a consistency loss) are the closest prior work (see `NOVELTY_REPORT.md`). The benchmark is the main contribution and the method is the secondary one.

## 10. The demo (Phase 6)

`python demo/app.py` opens a web page (Gradio) in your browser:
- **Detector tab:**
  - choose one of 12 test images (picked from your local test sets with a fixed seed, or listed in `demo/examples.txt`), upload a photo, or put photos in `demo/my_photos/`;
  - choose a corruption and a severity from 0 to 5. The dropdown says which corruptions the adapter saw in training and which it never saw;
  - the page shows the corrupted input, then each model's prediction as a coloured overlay with its outline at probability 0.5, and each model's latency on your GPU;
  - for test images, "Show ground truth" adds the true mask and each model's S-measure and MAE *on that image*.
- **Benchmark results tab:** the clean results table, the robustness summary (while the benchmark is still running, `complete` says which groups are finished), progress per model, efficiency, and the latest training-log entry. All of it is read from `results/*.csv`, so the demo can never show a number the code didn't produce.

Default models: SINet-V2 plus our best trained variant (Full, then Aug, then Clean). Other models can be added with `--models`. Without a GPU it runs on the CPU, more slowly.
