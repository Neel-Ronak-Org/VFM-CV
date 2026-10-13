# Plain-English Explainer

Read this before meeting your professor or answering reviewers. Each section answers: **what is it, why did we choose it, how does it work.** New sections are added as components are built.

---

## 1. Camouflaged object detection (COD)

**What:** Given a photo, output a map that is bright where a hidden object is (a stick insect, a flounder on sand) and dark elsewhere. It is binary segmentation where the object is designed by nature to look like the background.

**Why it's hard:** Normal object detectors rely on the object standing out by colour, edge or contrast. Camouflage removes exactly those cues, so the model must use subtle texture and shape differences.

**How it's evaluated:** On standard test sets (CHAMELEON, CAMO, COD10K, NC4K) we compare the predicted map with a hand-drawn ground-truth mask using the metrics in §3.

## 2. Corruption robustness and the held-out protocol

**What:** Real cameras blur, add sensor noise, compress to JPEG and shoot in fog. A *corruption benchmark* takes clean test images, applies these effects synthetically at 5 strengths (severities), and measures how much each model's score drops.

**Why `imagecorruptions`:** It is the standard ImageNet-C implementation (Hendrycks & Dietterich, ICLR 2019) used by most robustness papers, so our severities mean the same thing as everyone else's.

**How we keep it reproducible:** Every corrupted image gets its own random seed, computed from the dataset name, image name, corruption and severity with SHA-256. The same image always gets the same noise, on any computer and in any order. This lets us generate corruptions on the fly instead of storing tens of GB of files, and still be exactly reproducible.

**Held-out (seen/unseen) protocol:** We split the 8 corruptions into 4 *seen* and 4 *unseen*, one seen per family (noise, blur, photometric, digital). Our model only trains with seen ones. If it also improves on unseen ones (e.g. trained on defocus blur, tested on motion blur), it has learned real robustness, not just memorised the augmentation. COD10K-C did not test this.

**Summary numbers:**
- **mCorr** = average score over all corruptions × severities.
- **Retention** = mCorr ÷ clean score. 0.90 means the model keeps 90 % of its clean performance under corruption. Higher is more robust.

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
