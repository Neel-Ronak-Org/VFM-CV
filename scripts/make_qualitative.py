"""Image figures for the paper: teaser (Fig. 1) and qualitative grid (Fig. 5). Runs the models, so it needs
the datasets and weights (run it on the laptop GPU; about 2-3 minutes, SAM-Adapter takes most of it).

    python scripts/make_qualitative.py                 # both figures
    python scripts/make_qualitative.py --only teaser   # or: --only grid

Images are picked by fixed rules from the per-image results (results/per_image/...), never by hand:
  teaser: under motion blur (an UNSEEN corruption), the COD10K image whose per-image gain of Ours (512) over
          SINet-V2, averaged over severities 1-5, is at the --teaser-quantile of all images (default 0.9: a
          clear but not the single best example; say so in the caption). Columns: severity 0, 2, 4.
  grid:   one row per (corruption, severity) in GRID_ROWS: the image whose Ours (512) S-alpha is the MEDIAN
          (representative), then FAILURE_ROWS: the image where Ours (512) scores LOWEST (failure cases).
Writes figures/fig_teaser, fig_qualitative (5 rows, main paper), fig_qualitative_full (all rows, supplement)
and results/qualitative_selection.csv
(chosen images and every panel's S-alpha, so captions can quote them).
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.corruptions import corrupt_for_benchmark, load_protocol  # noqa: E402
from src.datasets import CODDataset  # noqa: E402
from src.metrics import image_scores  # noqa: E402
from src.models import build_model  # noqa: E402
from src.reporting import RESULTS, display  # noqa: E402

DATASET = "COD10K"
GRID_MODELS = ["SINet-V2", "ZoomNet", "SAM2-UNet", "SAM-Adapter", "Ours-Full", "Ours-Full-512"]
GRID_ROWS = [("gaussian_noise", 4), ("gaussian_blur", 4), ("motion_blur", 4), ("rain", 4), ("jpeg_compression", 5)]
FAILURE_ROWS = [("motion_blur", 5), ("rain", 5)]
# rows of the compact grid in the main paper (fig_qualitative); the full grid goes to the supplement
MAIN_ROWS = [("gaussian_noise", 4), ("motion_blur", 4), ("rain", 4), ("motion_blur", 5), ("rain", 5)]
TEASER_MODELS = ["SINet-V2", "Ours-Full-512"]
TEASER_CORRUPTION, TEASER_SEVERITIES = "motion_blur", [0, 2, 4]
PER_IMAGE = RESULTS / "per_image" / "benchmark"
NAMES = {"gaussian_noise": "Gaussian noise", "gaussian_blur": "Gaussian blur", "motion_blur": "motion blur",
         "rain": "rain", "jpeg_compression": "JPEG", "fog": "fog", "brightness": "brightness", "contrast": "contrast"}

plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
                     "font.size": 7, "savefig.dpi": 300})


def per_image(model: str, corruption: str, severity: int) -> pd.Series:
    f = PER_IMAGE / model / DATASET / f"{corruption}-{severity}.csv.gz"
    if not f.exists():
        raise FileNotFoundError(f"{f} missing: evaluate {model} on {DATASET} first")
    return pd.read_csv(f).set_index("name").Sm


def pick_teaser(quantile: float) -> str:
    gains = []
    for s in range(1, 6):
        a, b = per_image(TEASER_MODELS[1], TEASER_CORRUPTION, s), per_image(TEASER_MODELS[0], TEASER_CORRUPTION, s)
        gains.append((a - b).dropna())
    g = pd.concat(gains, axis=1).mean(axis=1).sort_values()
    return g.index[min(len(g) - 1, int(round(quantile * (len(g) - 1))))]


def pick_rows() -> list[tuple[str, int, str, str]]:
    rows = []
    for c, s in GRID_ROWS:
        sc = per_image("Ours-Full-512", c, s).sort_values()
        rows.append((c, s, sc.index[len(sc) // 2], "median"))
    for c, s in FAILURE_ROWS:
        sc = per_image("Ours-Full-512", c, s).sort_values()
        rows.append((c, s, sc.index[0], "failure"))
    return rows


def run_models(models: list[str], jobs: list[tuple[str, str, int]], ds: CODDataset, base_seed: int) -> dict:
    """jobs: (image name, corruption, severity). Loads one model at a time to stay within 4 GB."""
    by_name = {s.name: i for i, s in enumerate(ds.samples)}
    cache, preds = {}, {}
    for name, c, s in jobs:
        if (name, c, s) not in cache:
            _, img, gt = ds[by_name[name]]
            cache[(name, c, s)] = (corrupt_for_benchmark(img, DATASET, name, c, s, base_seed), gt)
    for m in models:
        model = build_model(m)
        for key, (img, gt) in cache.items():
            p = model.predict(img)
            if p.shape != gt.shape:
                p = cv2.resize(p, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
            preds[(m, *key)] = (p, image_scores(p, gt)["Sm"])
        del model
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
    return cache, preds


def score(ax, value, size=6):
    """Per-panel S-alpha, boxed so it stays readable on white or black predictions."""
    ax.text(0.03, 0.05, f"{value:.3f}", transform=ax.transAxes, color="white", fontsize=size,
            bbox=dict(facecolor="black", alpha=0.7, pad=1.0, linewidth=0))


def show(ax, arr, title=None, cmap=None):
    ax.imshow(arr, cmap=cmap, vmin=0, vmax=1 if cmap else None, interpolation="bilinear")
    ax.set_aspect("equal")
    ax.set_anchor("C")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.3)
    if title:
        ax.set_title(title, fontsize=6.5, pad=2)


def teaser(ds, base_seed, quantile, log):
    name = pick_teaser(quantile)
    jobs = [(name, TEASER_CORRUPTION if s else "clean", s) for s in TEASER_SEVERITIES]
    cache, preds = run_models(TEASER_MODELS, jobs, ds, base_seed)
    fig, axes = plt.subplots(len(TEASER_MODELS) + 1, len(jobs), figsize=(3.5, 3.4))
    for j, (_, c, s) in enumerate(jobs):
        img, gt = cache[(name, c, s)]
        show(axes[0, j], img, "clean" if s == 0 else f"{NAMES[c]}, severity {s}")
        for i, m in enumerate(TEASER_MODELS, start=1):
            p, sc = preds[(m, name, c, s)]
            show(axes[i, j], p, cmap="gray")
            score(axes[i, j], sc)
            log.append({"figure": "teaser", "image": name, "corruption": c, "severity": s, "model": m, "Sm": sc,
                        "rule": f"quantile {quantile} of Ours-512 minus SINet-V2 gain under {TEASER_CORRUPTION}"})
    for i, m in enumerate(TEASER_MODELS, start=1):
        axes[i, 0].set_ylabel(display(m), fontsize=7)
    axes[0, 0].set_ylabel("Input", fontsize=7)
    fig.tight_layout(pad=0.2, h_pad=0.3, w_pad=0.2)
    for ext in ("pdf", "png"):
        fig.savefig(REPO / "figures" / f"fig_teaser.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote figures/fig_teaser.pdf/.png (image {name})")


def grid(ds, base_seed, log):
    rows_all = pick_rows()
    for rows, out_name in ((rows_all, "fig_qualitative_full"),
                           ([r for r in rows_all if (r[0], r[1]) in MAIN_ROWS], "fig_qualitative")):
        _grid(ds, base_seed, log if out_name == "fig_qualitative_full" else [], rows, out_name)


def _grid(ds, base_seed, log, rows, out_name):
    jobs = [(name, c, s) for c, s, name, _ in rows]
    cache, preds = run_models(GRID_MODELS, jobs, ds, base_seed)
    ncol = 2 + len(GRID_MODELS)
    # equal-sized panels (4:3 cells); portrait images are centred inside their cell, so rows line up
    cell_w = 7.16 / ncol
    fig, axes = plt.subplots(len(rows), ncol, figsize=(7.16, cell_w * 0.75 * len(rows) + 0.25),
                             gridspec_kw=dict(wspace=0.04, hspace=0.06))
    for r, (c, s, name, kind) in enumerate(rows):
        img, gt = cache[(name, c, s)]
        show(axes[r, 0], img, "Input" if r == 0 else None)
        axes[r, 0].set_ylabel(f"{NAMES[c]} {s}" + ("\n(failure)" if kind == "failure" else ""), fontsize=6.5)
        show(axes[r, 1], gt.astype(np.float32), "GT" if r == 0 else None, cmap="gray")
        for k, m in enumerate(GRID_MODELS, start=2):
            p, sc = preds[(m, name, c, s)]
            show(axes[r, k], p, display(m) if r == 0 else None, cmap="gray")
            score(axes[r, k], sc, size=5.5)
            log.append({"figure": "grid", "image": name, "corruption": c, "severity": s, "model": m, "Sm": sc,
                        "rule": f"{kind} Ours-512 S-alpha for this condition"})
    fig.subplots_adjust(left=0.06, right=0.995, top=0.96, bottom=0.005)
    for ext in ("pdf", "png"):
        fig.savefig(REPO / "figures" / f"{out_name}.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote figures/{out_name}.pdf/.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["teaser", "grid"], default=None)
    ap.add_argument("--teaser-quantile", type=float, default=0.9)
    args = ap.parse_args()
    ds = CODDataset.from_config(DATASET)
    base_seed = load_protocol()["base_seed"]
    (REPO / "figures").mkdir(exist_ok=True)
    log: list[dict] = []
    if args.only in (None, "teaser"):
        teaser(ds, base_seed, args.teaser_quantile, log)
    if args.only in (None, "grid"):
        grid(ds, base_seed, log)
    out = RESULTS / "qualitative_selection.csv"
    pd.DataFrame(log).round(4).to_csv(out, index=False)
    print(f"wrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
