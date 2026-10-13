# Datasets: sources, licences, citations

We **never redistribute** any dataset. Users download from the official links (`scripts/download_data.py`). Corrupted versions are generated on the fly from fixed seeds, so they don't need to be shared either.

| Dataset | Split used | # images | Licence / terms (as stated by the authors) | Verified at |
|---|---|---|---|---|
| **COD10K** (Fan et al., CVPR 2020; TPAMI 2022) | train 3,040 / test 2,026 | 5,066 | "Made available for non-commercial purposes only"; no commercial use of the data or anything derived from it. | https://github.com/DengPingFan/SINet (README §9 LICENSE), checked 2026-10-13 |
| **CAMO** (Le et al., CVIU 2019) | train 1,000 / test 250 | 1,250 | Secondary source reports CC BY-NC-SA 4.0 (non-commercial, share-alike). **[unverified: official project page not reachable from the dev container]** | https://sites.google.com/view/ltnghia/research/camo (to check) |
| **CHAMELEON** (Skurowski et al., 2018, unpublished manuscript) | test 76 | 76 | Images collected from the Internet via Google image search; **no licence stated** in sources found. Use for research evaluation only. | https://www.polsl.pl/rau6/datasets/ |
| **NC4K** (Lv et al., CVPR 2021) | test 4,121 | 4,121 | Publicly released by the authors; repo carries **no LICENSE file** and the README states no terms. Research use only (conservative). | https://github.com/JingZhang617/COD-Rank-Localize-and-Segment |

**Packaging:** the training set (COD10K-train + CAMO-train = 4,040) and the test sets come from the ready-made packages linked in the SINet-V2 README (https://github.com/GewelsJI/SINet-V2). The packages only repackage the original datasets, so the original licences above apply.

**Code licences that matter for release (Phase 9):**
- SINet-V2 repo has an Apache-2.0 `LICENSE` file, but its README says "free for research and education use only; any commercial usage should get formal permission first". Vendored SINet-V2 network code keeps its own header and licence notice; our MIT licence applies to our own code only.
- Vendored baseline code (each folder in `src/models/third_party/` keeps its LICENSE):
  SAM2-UNet Apache-2.0 · ZoomNet MIT · PFNet BSD-style (copyright notice retained) · FEDER MIT ·
  SAM-Adapter MIT (SAM parts Apache-2.0) · MobileSAM Apache-2.0.

**Consequence for the paper:** state that all datasets are used for non-commercial academic research under their authors' terms.

## Citations (BibTeX goes in `paper/refs.bib` in Phase 8, each entry verified)

- COD10K / SINet: D.-P. Fan, G.-P. Ji, G. Sun, M.-M. Cheng, J. Shen, L. Shao. *Camouflaged Object Detection.* CVPR 2020.
- COD10K / SINet-V2: D.-P. Fan, G.-P. Ji, M.-M. Cheng, L. Shao. *Concealed Object Detection.* IEEE TPAMI 2022.
- CAMO: T.-N. Le, T. V. Nguyen, Z. Nie, M.-T. Tran, A. Sugimoto. *Anabranch Network for Camouflaged Object Segmentation.* CVIU 2019.
- CHAMELEON: P. Skurowski, H. Abdulameer, J. Błaszczyk, T. Depta, A. Kornacki, P. Kozieł. *Animal Camouflage Analysis: Chameleon Database.* Unpublished manuscript, 2018.
- NC4K: Y. Lv, J. Zhang, Y. Dai, A. Li, B. Liu, N. Barnes, D.-P. Fan. *Simultaneously Localize, Segment and Rank the Camouflaged Objects.* CVPR 2021.
- COD10K-C (corruption definitions): A. H. Sayem. *COD10K-C: Benchmarking Robustness of Camouflaged Object Detection Under Natural Image Corruptions.* arXiv:2606.02603, 2026.

## Verification results

Filled in from `results/data_check.csv` after `python scripts/download_data.py --verify` runs on the laptop.
