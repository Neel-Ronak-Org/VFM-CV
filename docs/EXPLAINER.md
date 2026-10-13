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

*(Sections on the SAM family, adapters, the consistency loss and efficiency metrics are added in Phases 2–5.)*
