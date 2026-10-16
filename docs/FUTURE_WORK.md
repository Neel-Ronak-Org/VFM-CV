# Future work (parked ideas)

Ideas to pick up **after** the current project (benchmark + adapter + paper) is finished. Nothing here changes the current plan.

## Biologically inspired modules on a frozen foundation model

**Question:** does a biologically inspired module, added to a *frozen* backbone (MobileSAM / SAM2), improve robustness to *unseen* corruptions in camouflaged object detection?

It reuses everything this project builds: frozen backbones, the evaluation harness, the COD10K-C-style corruption benchmark and the held-out (seen/unseen) protocol. Test each module as a controlled ablation: same backbone, same parameter budget, with vs without the module.

Three candidates, cheapest first:

1. **V1 front end** (VOneBlock-style: fixed Gabor filter bank, simple/complex cells, neural noise) before or alongside the frozen encoder. Tests whether the VOneNet robustness gain carries over to SAM-family models and camouflage.
2. **Contour-integration / border-ownership module** on the frozen features. Targets what camouflage attacks: broken contours and figure-ground ambiguity (V1/V2 association fields, border-ownership cells). The most original option and the hardest.
3. **Recurrent top-down refinement:** feed the decoder's output back to re-weight features over a few iterations (cortical feedback / predictive-coding style). Compare against SINet-V2's reverse attention, a crude feed-forward version.

Background to read and **verify before citing** (listed from memory, not yet checked):
Neocognitron (Fukushima 1980), HMAX (Riesenhuber & Poggio 1999), VOneNet (Dapello et al., NeurIPS 2020),
CORnet (Kubilius et al., 2019), PredNet (Lotter et al., 2017), border-ownership coding in V2 (Zhou, Friedman & von der Heydt, 2000).

Cautions: "inspired by" not "works like the brain"; rule out that gains come only from extra parameters or regularisation.

*Logged 2026-10-15 at the student's request.*
