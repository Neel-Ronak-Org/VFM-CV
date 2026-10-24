"""Paired bootstrap tests between models (the claims in the paper's text and Table 5).

    python scripts/significance.py            # -> results/significance.csv

For each comparison (a, b, corruption group) and each dataset where BOTH models have per-image scores for
every condition of the group (results/per_image/benchmark/<model>/<dataset>/<corruption>-<sev>.csv.gz):
per image, average Sm over the group's corrupted conditions; take the paired difference a - b; report
its mean, the 95 % percentile bootstrap CI of the mean (5,000 resamples, seed 0) and the share of images
where a scores higher. "significant" = the CI excludes 0. Reads only results files.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.corruptions import load_protocol  # noqa: E402
from src.reporting import DATASETS, GLOBAL, LOCAL, RESULTS, paired_bootstrap, per_image_mean  # noqa: E402

# (a, b, group); a group is a protocol key, "local"/"global", or a single corruption name.
COMPARISONS = [
    *[("Ours-Full", b, "all") for b in ["SINet-V2", "ZoomNet", "PFNet", "SAM2-UNet", "SAM-Adapter"]],
    *[("Ours-Full-512", b, "all") for b in ["SINet-V2", "ZoomNet", "PFNet", "SAM2-UNet", "SAM-Adapter", "Ours-Full"]],
    ("Ours-Full-512", "Ours-Full", "local"), ("Ours-Full-512", "Ours-Full", "global"),
    ("SAM-Adapter", "SAM2-UNet", "all"),
    *[("Ours-Full", "Ours-Clean", g) for g in ["all", "seen", "unseen", "motion_blur", "contrast", "fog", "rain",
                                             "gaussian_noise", "gaussian_blur", "brightness", "jpeg_compression"]],
    *[("Ours-Aug", "Ours-Clean", g) for g in ["all", "seen", "unseen"]],
    *[("Ours-Full", "Ours-Aug", g) for g in ["all", "seen", "unseen"]],
]


def group_corruptions(group: str, protocol: dict) -> list[str]:
    if group == "all":
        return list(protocol["corruptions"])
    if group in ("seen", "unseen"):
        return list(protocol[group])
    if group == "local":
        return LOCAL
    if group == "global":
        return GLOBAL
    return [group]


def run(comparisons=COMPARISONS, datasets=DATASETS, root: Path | None = None, n_boot: int = 5000) -> pd.DataFrame:
    protocol = load_protocol()
    n_sev = len(protocol["severities"])
    rows = []
    for a, b, group in comparisons:
        corrs = group_corruptions(group, protocol)
        for ds in datasets:
            x, y = per_image_mean(a, ds, corrs, root), per_image_mean(b, ds, corrs, root)
            if x.empty or y.empty:
                continue
            # only complete groups: every image must have all conditions (checked via file count below)
            d = (x - y).dropna()
            if len(d) == 0:
                continue
            files_ok = all(_n_files(m, ds, corrs, root) == len(corrs) * n_sev for m in (a, b))
            if not files_ok:
                continue
            diff = d.to_numpy()
            lo, hi = paired_bootstrap(diff, n_boot=n_boot)
            rows.append({"a": a, "b": b, "dataset": ds, "group": group, "n_images": len(diff),
                         "mean_diff": float(diff.mean()), "ci_lo": lo, "ci_hi": hi,
                         "significant": bool(lo > 0 or hi < 0), "win_rate": float((diff > 0).mean()),
                         "n_boot": n_boot})
    return pd.DataFrame(rows)


def _n_files(model, ds, corrs, root) -> int:
    import glob

    root = root or RESULTS / "per_image" / "benchmark"
    return sum(len(glob.glob(str(root / model / ds / f"{c}-*.csv.gz"))) for c in corrs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args()
    df = run(n_boot=args.n_boot)
    out = RESULTS / "significance.csv"
    df.round(6).to_csv(out, index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(df[["a", "b", "dataset", "group", "n_images", "mean_diff", "ci_lo", "ci_hi", "significant"]]
              .round(4).to_string(index=False))
    print(f"\nwrote {out.relative_to(REPO)} ({len(df)} comparisons)")


if __name__ == "__main__":
    main()
