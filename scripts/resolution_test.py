"""Resolution test: is our adapter less robust because it sees corruptions at 1024 px?

Hypothesis (PROGRESS session 11): the 352-px baselines shrink each image ~2.8x, which averages away
noise / JPEG / rain and shrinks blur, while our model looks at ~full resolution. If so, giving OUR trained
weights a low-resolution view (reduce to N x N, then back up; src/models/robust_adapter.py:low_res_view)
should raise retention on the LOCAL corruptions and leave the GLOBAL ones (brightness, contrast, fog) alone,
at some cost in clean accuracy. No retraining.

    python scripts/resolution_test.py                    # evaluate (resumable) + report, CHAMELEON
    python scripts/resolution_test.py --dataset COD10K   # same on COD10K (~2.5 h per model)
    python scripts/resolution_test.py --report-only

Rows go to results/benchmark.csv like every other model (names Ours-Full-in352 etc.); the comparison table
is written to results/resolution_test_<dataset>.csv. Reads only results files for the report.
"""
from __future__ import annotations

import argparse
import glob
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.corruptions import load_protocol  # noqa: E402

TEST_MODELS = ["Ours-Full-in352", "Ours-Full-in512", "Ours-Full-in768", "Ours-Clean-in352"]
REFERENCE = ["Ours-Full", "Ours-Full-512", "Ours-Clean", "SINet-V2", "SAM2-UNet", "ZoomNet", "PFNet"]
# Local corruptions change pixels within a small neighbourhood, so downsampling weakens them;
# global ones change every pixel the same way and survive downsampling unchanged.
LOCAL = ["gaussian_noise", "gaussian_blur", "motion_blur", "jpeg_compression", "rain"]
GLOBAL = ["brightness", "contrast", "fog"]


def per_image_sm(model: str, dataset: str, corrs: list[str]) -> pd.Series:
    """Mean Sm per image over the given corruptions x severities (from the per-image logs)."""
    files = []
    for c in corrs:
        files += glob.glob(str(REPO / "results" / "per_image" / "benchmark" / model / dataset / f"{c}-*.csv*"))
    if not files:
        return pd.Series(dtype=float)
    return pd.concat([pd.read_csv(f, usecols=["name", "Sm"]) for f in files]).groupby("name").Sm.mean()


def paired(a: str, b: str, dataset: str, corrs: list[str], n_boot: int = 5000) -> str:
    x, y = per_image_sm(a, dataset, corrs), per_image_sm(b, dataset, corrs)
    d = (x - y).dropna().to_numpy()
    if len(d) == 0:
        return f"{a} vs {b}: no paired data"
    rng = np.random.default_rng(0)
    boots = rng.choice(d, (n_boot, len(d))).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    sig = "significant" if lo > 0 or hi < 0 else "not significant"
    return f"{a} - {b}: {d.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  ({sig}, n={len(d)})"


def report(dataset: str) -> pd.DataFrame:
    bench = pd.read_csv(REPO / "results" / "benchmark.csv").drop_duplicates(
        ["model", "dataset", "corruption", "severity"], keep="last")
    bench = bench[bench.dataset == dataset]
    protocol = load_protocol()
    groups = {"all": protocol["corruptions"], "local": LOCAL, "global": GLOBAL,
              "seen": protocol["seen"], "unseen": protocol["unseen"]}
    rows = []
    for m in REFERENCE[:3] + TEST_MODELS + REFERENCE[3:]:
        sub = bench[bench.model == m]
        clean = sub[sub.corruption == "clean"]
        if clean.empty:
            continue
        row = {"model": m, "Sm_clean": clean.Sm.iloc[-1]}
        for g, corrs in groups.items():
            cs = sub[sub.corruption.isin(corrs)]
            row[f"Sm_{g}"] = cs.Sm.mean() if len(cs) else np.nan
            row[f"ret_{g}"] = row[f"Sm_{g}"] / row["Sm_clean"]
            row[f"n_{g}"] = len(cs)
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        print(f"no results for {dataset} yet")
        return df
    out = REPO / "results" / f"resolution_test_{dataset}.csv"
    df.round(6).to_csv(out, index=False)
    show = ["model", "Sm_clean", "Sm_all", "ret_all", "ret_local", "ret_global", "ret_seen", "ret_unseen", "n_all"]
    with pd.option_context("display.width", 200):
        print(df[show].round(3).to_string(index=False))
    print(f"\nPaired per-image comparisons on {dataset} (mean Sm under corruption):")
    for a, b, corrs, lbl in [("Ours-Full-in352", "Ours-Full", LOCAL, "local"),
                             ("Ours-Full-in352", "Ours-Full", GLOBAL, "global"),
                             ("Ours-Full-in352", "SINet-V2", protocol["corruptions"], "all"),
                             ("Ours-Clean-in352", "Ours-Clean", LOCAL, "local"),
                             ("Ours-Full-512", "Ours-Full", protocol["corruptions"], "all")]:
        print(f"  [{lbl:6s}] {paired(a, b, dataset, corrs)}")
    print("\nHypothesis supported if: in352 raises ret_local clearly, ret_global barely moves, and Sm_clean drops "
          "(the resolution trade-off).")
    print(f"wrote {out.relative_to(REPO)}")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="CHAMELEON")
    ap.add_argument("--models", nargs="+", default=TEST_MODELS)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    if not args.report_only:
        for corr in ("clean", "all"):
            cmd = [sys.executable, "-m", "src.evaluate", "--models", *args.models, "--datasets", args.dataset,
                   "--corruptions", corr]
            print(">", " ".join(cmd))
            subprocess.run(cmd, cwd=REPO, check=True)
    report(args.dataset)


if __name__ == "__main__":
    main()
