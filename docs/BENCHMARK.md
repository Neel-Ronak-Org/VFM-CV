# Corruption benchmark protocol

Implementation: `src/corruptions.py`, parameters: `configs/corruptions.yaml`, run plan: `configs/benchmark.yaml`.

## Corruptions and severities

We use the 8 corruption types and severity values of COD10K-C (Sayem, arXiv:2606.02603), the only earlier corruption benchmark for camouflaged object detection, so that our specialist-model numbers can be compared with theirs.

| Corruption | Severity 1 → 5 | Simulates |
|---|---|---|
| Gaussian noise | σ = 8, 16, 24, 32, 40 (pixel units) | low-light sensor noise |
| Motion blur | horizontal kernel k = 3 … 11 px | camera shake, moving camera |
| Gaussian blur | radius 0.8 … 2.5 px | out-of-focus lens |
| Brightness | × 0.85 … 0.30 | under-exposure |
| Contrast | factor 0.85 … 0.30 | haze, low-quality sensors |
| Fog | blend with white, α = 0.12 … 0.44 | fog, mist |
| JPEG | quality 70 … 20 | compression on drones and phones |
| Rain | streak density 0.2 % … 1 %, length 10 … 30 px | rain on the lens or in the scene |

`python scripts/preview_corruptions.py` draws one image under every corruption and severity (`figures/corruption_preview.jpg`).

## Implementation choices

COD10K-C gives the parameters but no code, so these details are ours:

1. Gaussian noise is added independently per pixel and per RGB channel, then clipped to [0, 255] and rounded.
2. Motion blur is a normalised 1×k box filter (OpenCV `filter2D`, reflected borders).
3. Gaussian blur uses PIL's `GaussianBlur(radius)`; PIL's radius is the standard deviation.
4. Brightness and contrast use PIL `ImageEnhance` (brightness multiplies every pixel by the factor; contrast blends towards the image's mean grey level).
5. Fog is exactly `(1−α)·image + α·255`.
6. JPEG uses PIL with default chroma subsampling.
7. **Rain is our own design**: random seed pixels (density), smeared along an 80° line (length), lightly blurred, and blended towards white at up to 60 % opacity. Rain results are therefore not directly comparable with COD10K-C.

## Reproducibility

Each corrupted image has its own random seed, derived with SHA-256 from the dataset name, image name, corruption and severity (`image_seed` in `src/corruptions.py`). The same image always receives the same noise or rain, on any machine and in any order, so corrupted test sets are generated on the fly rather than stored. Evaluation is resumable: `src/evaluate.py` skips every (model, dataset, corruption, severity) already in `results/benchmark.csv`.

## Held-out (seen / unseen) protocol

The 8 corruptions are split into 4 *seen* and 4 *unseen*. Our adapter is trained with seen corruptions only; `src/train.py` refuses a config that augments with an unseen one unless `augment.allow_unseen` is set explicitly.

| Family | Seen (training augmentation) | Unseen (test only) |
|---|---|---|
| Noise | Gaussian noise | — |
| Blur | Gaussian blur | Motion blur |
| Illumination | Brightness | Contrast |
| Weather | — | Fog, Rain |
| Digital | JPEG | — |

Each unseen blur and illumination corruption has a related seen one (does training on Gaussian blur help with motion blur?), and weather is held out completely (does robustness transfer to a new kind of degradation?). Noise and JPEG have only one member each in COD10K-C, so they have to be seen. An optional `extended_unseen` set in `configs/corruptions.yaml` adds four ImageNet-C corruptions (shot noise, defocus blur, zoom blur, pixelate), outside the COD10K-C protocol.

## Metrics and summaries

Per image: Sα, Eφ (mean), Fβw, MAE (PySODMetrics), and Dice and IoU at threshold 0.5 (`src/metrics.py`). Dataset scores are means of per-image scores; per-image scores are kept in `results/per_image/` for paired tests.

- **mCorr**: mean score over all corruptions × severities.
- **Retention** = mCorr / clean score (higher is more robust), reported overall and for seen, unseen, local (noise, blur, JPEG, rain) and global (brightness, contrast, fog) corruptions.
- **mCE**: corruption error normalised to SINet-V2.
- **Paired bootstrap** (`scripts/significance.py`): per-image differences between two models, averaged over the conditions of a group, resampled 5,000 times (seed 0); a difference is significant when its 95 % interval excludes zero.
