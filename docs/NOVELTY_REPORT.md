# Novelty Report (Phase 0)

*Date of search: 2026-10-13. Searches were made through a web search engine. The container could not open arXiv directly (blocked by the network policy), so the details below come from abstracts, arXiv HTML snippets and search-engine summaries. Anything marked **[verify]** must be confirmed against the full PDF before it goes into the paper.*

## 1. What we searched

- "camouflaged object detection" + SAM / SAM2 / SAM3 / foundation model + corruption / robustness / blur / noise / JPEG / degraded / low-quality
- COD10K-C: code/data release status, whether SAM models were added
- Newer COD robustness benchmarks (2025–2026)
- SAM robustness under corruption (general segmentation)
- Lightweight SAM (MobileSAM / EfficientSAM) + adapters + COD
- Corruption augmentation + consistency loss for SAM adaptation; held-out / unseen corruption protocols

## 2. Closest prior work

| # | Paper | What it covers | Overlap with us |
|---|---|---|---|
| 1 | **COD10K-C** — A. H. Sayem, arXiv:2606.02603 (May/June 2026) | Corruption benchmark on COD10K-test: 8 corruption types × 5 severities (40 conditions, 81,040 pairs). Models: SINet-v2, PFNet, ZoomNet + their own **RobustCODLite** (EfficientNet-B0, corruption augmentation, high-frequency prior, *uncertainty-consistency loss*). Retention of clean Dice: RobustCODLite 92.3 %, SINet-v2 87.7 %, ZoomNet 84.8 %, PFNet 84.1 %. Blur (motion, Gaussian) hurts most. | **High** for the benchmark idea and for the method (aug + consistency on a lightweight model). **No SAM-family models. COD10K only.** No held-out-corruption protocol reported **[verify]**. Code "will be released"; no repo found as of today. |
| 2 | **RobustSAM** — W.-T. Chen et al., CVPR 2024 | Makes SAM robust to degraded images (blur, noise, low light, weather) with extra modules trained with **consistency losses against clean-image features of the original SAM**. Generic (prompted) segmentation, not COD. | **High for the method idea.** Our consistency loss must be positioned as "applied to a lightweight, prompt-free COD adapter", not as new. |
| 3 | **GaRA-SAM** — arXiv:2506.02882 (2025) | Gated-rank adaptation (LoRA-like) to robustify SAM; evaluates on *unseen* real-world degradation datasets. | Medium: adapter-for-robustness + unseen-degradation evaluation, but not COD and not synthetic held-out corruption families. |
| 4 | **On the Robustness of Segment Anything** — Huang et al., arXiv:2305.16220 | SAM robust to most corruptions except blur (SA-1B, KITTI). | Background; motivates our blur focus. Not COD. |
| 5 | **Robustness of SAM: Segment Anything Under Corruptions and Beyond** — Qiao et al., arXiv:2306.07713 | 15 corruptions × 5 severities; SAM robust except zoom blur. | Background. Not COD, no adapters. |
| 6 | **KRNet: Knowledge Rectification for COD from Low-Quality Data** — Guan et al., arXiv:2503.22180 (2025) | COD on low-quality (degraded/low-resolution) images; shows HitNet Sα drops 82.6 → 73.0 on low-quality COD10K. Leader–follower distillation. | Medium: robustness to degraded COD inputs, but one degradation type, no SAM-family benchmark, no corruption-family protocol **[verify degradation type]**. |
| 7 | **SAM-Adapter** (ICCVW 2023, arXiv:2304.09148), **SAM2-Adapter** (arXiv:2408.04579), **SAM3-Adapter** (arXiv:2511.19425) | State-of-the-art clean COD with adapters on frozen SAM/SAM2/SAM3. SAM3-Adapter claims "robustness" in a general sense; no corruption tests found. | Our baselines. Robustness under corruption is untested. |
| 8 | **SAM-DSA** — Liu et al., ICCV 2025 | Dual-stream (RGB + depth) adapters for SAM in COD; future work mentions lightweight SAM. | Low: clean only. |
| 9 | **CATS / token sparsification for efficient COD with VFMs** — arXiv:2604.16854 (2026) | Efficiency of foundation-model COD (−34 % GFLOPs). | Low: efficiency only, no robustness. Cite it in the efficiency discussion. |
| 10 | **LeanCOD** (PMC, 2026) | Edge-device COD; evaluated fog / rain / low-light on COD10K (small drops). | Low: small synthetic-weather test, no SAM. |

Also found, clean-only SAM-COD work: SAM2-UNet (arXiv:2408.08870), "Evaluating SAM2's role in COD" (arXiv:2407.21596), SAM-COD (weakly supervised, arXiv:2408.10760), CamoSAM2 (video).

## 3. Verdict

**Contribution 1 (SAM-family corruption benchmark on COD): still open.** Nothing found evaluates SAM / SAM2 / SAM3-based COD models under common corruptions. COD10K-C explicitly omits them, covers COD10K only, and has no released code or data.

**Contribution 3 (held-out-corruption protocol): open.** No COD work found trains on some corruption families and tests on others. GaRA-SAM does an unseen-*dataset* evaluation for generic SAM, which is close in spirit and must be cited.

**Contribution 4 (analysis: family × corruption, size vs robustness, speed trade-off on consumer GPU): open.**

**Contribution 2 (method): partially scooped and must be re-framed.**
- *Corruption augmentation + consistency loss on a lightweight COD model* is already in COD10K-C (RobustCODLite).
- *Consistency loss to make SAM robust* is already in RobustSAM.
- What is left for us: a **frozen lightweight SAM encoder (MobileSAM/EfficientSAM) with a prompt-free adapter**, trained on a laptop GPU, and evaluated on **unseen** corruptions against SAM-family and specialist baselines. This is an incremental engineering contribution, not a new algorithm. The paper should lead with the benchmark and the held-out protocol and present the adapter as a strong, cheap baseline.

**Overall: GO, with the reframing above.** Suggested title adjustment:
*"How Robust Are Foundation-Model Camouflage Detectors? A Corruption Benchmark with Held-Out Corruptions and a Laptop-Scale Robust Adapter"*

## 4. Feasibility findings that affect the plan

These came out of checking repos and weights during Phase 0:

1. **SAM2-Adapter and SAM3-Adapter: no COD-trained weights found.** The SAM2-Adapter branch README asks you to train from `sam2.1_hiera_large.pt`. The SAM3-Adapter branch only ships a code zip. Training them needs about 4×A100, which is impossible on a 4 GB GPU. **Proposal:** mark both as "not evaluated (no released weights)" and substitute **SAM2-UNet** (arXiv:2408.08870), whose README links COD weights and prediction maps on Google Drive. Its Hiera-L encoder at 352 px input should fit in 4 GB fp16 **[verify by profiling]**.
2. **SAM-Adapter:** one released checkpoint (Google Drive). It is most likely the ViT-H COD model **[verify after download]**. ViT-H encoder in fp16 is about 1.3 GB of weights, so batch-1 inference on 4 GB is plausible but tight.
3. **Specialists:** SINet-V2 (Drive), ZoomNet (GitHub release, `.pth`), PFNet (author project page), FEDER (Drive). All have released weights.
4. **Lightweight SAMs:** MobileSAM weights are in its GitHub repo (`weights/mobile_sam.pt`, about 40 MB) and on Drive. EfficientSAM-Ti/S weights are in its GitHub repo.
5. **COD10K-C exact corruption list:** the snippets confirm motion blur, **Gaussian blur**, fog and brightness. The other four names and the severity parameters were not visible **[verify from the PDF §3]**. Our default list uses `defocus_blur`; if COD10K-C uses `gaussian_blur`, we should switch for comparability. `imagecorruptions` supports both.
6. **`imagecorruptions` is broken on NumPy 2 / scikit-image ≥ 0.19** (`fog`, `gaussian_blur`). Fixed with a small compatibility shim in `src/corruptions.py`; no need for an old environment.

## 5. Pivot options (if contribution 1 gets scooped before submission)

1. **Unseen-corruption generalization as the main story.** Same pipeline, but the paper's headline is the held-out protocol: which training corruptions transfer to which unseen ones, for both specialist and SAM-family models.
2. **Test-time adaptation angle.** Keep the frozen benchmark and add a cheap TTA step (e.g. entropy minimisation or BN/LayerNorm-statistics adaptation on the adapter only) to see how much robustness can be recovered with no retraining.
3. **Efficiency–robustness trade-off on consumer hardware.** Lead with retention-per-GFLOP and retention-per-FPS on a 4 GB laptop GPU, comparing MobileSAM / EfficientSAM / SAM2-UNet / SINet-V2.

## 6. Sources (to re-verify when writing `refs.bib`)

- COD10K-C: https://arxiv.org/abs/2606.02603
- RobustSAM: https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_RobustSAM_Segment_Anything_Robustly_on_Degraded_Images_CVPR_2024_paper.pdf
- GaRA-SAM: https://arxiv.org/abs/2506.02882
- On the Robustness of Segment Anything: https://arxiv.org/abs/2305.16220
- Robustness of SAM under corruptions: https://arxiv.org/abs/2306.07713
- KRNet: https://arxiv.org/abs/2503.22180
- SAM-Adapter: https://arxiv.org/abs/2304.09148 ; ICCVW: https://openaccess.thecvf.com/content/ICCV2023W/VCL/papers/Chen_SAM-Adapter_Adapting_Segment_Anything_in_Underperformed_Scenes_ICCVW_2023_paper.pdf
- SAM2-Adapter: https://arxiv.org/abs/2408.04579
- SAM3-Adapter: https://arxiv.org/abs/2511.19425
- SAM-DSA: https://openaccess.thecvf.com/content/ICCV2025/papers/Liu_Improving_SAM_for_Camouflaged_Object_Detection_via_Dual_Stream_Adapters_ICCV_2025_paper.pdf
- Token sparsification for COD: https://arxiv.org/abs/2604.16854
- LeanCOD: https://pmc.ncbi.nlm.nih.gov/articles/PMC13417466/
- SAM2-UNet: https://arxiv.org/abs/2408.08870 ; code https://github.com/WZH0120/SAM2-UNet
- RobustMedSAM (related, medical): https://arxiv.org/abs/2604.09814
