"""Shared helpers for the paper's tables and figures (scripts/make_tables.py, make_figures.py, significance.py).

Everything here is computed from results files only: results/benchmark.csv (+ per-image scores),
results/released_maps.csv, results/efficiency.csv, results/train/<run>/summary.json.
Model metadata below (display names, groups, input resolution) is not a result; it is read off each
wrapper's header in src/models/ and only labels the rows.
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.corruptions import load_protocol

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
DATASETS = ["CHAMELEON", "CAMO", "COD10K", "NC4K"]
METRICS = ["Sm", "Em", "wFm", "MAE", "Dice", "IoU"]
LOWER_IS_BETTER = {"MAE"}
# Local corruptions change pixels relative to their neighbours; global ones change all pixels alike.
LOCAL = ["gaussian_noise", "gaussian_blur", "motion_blur", "jpeg_compression", "rain"]
GLOBAL = ["brightness", "contrast", "fog"]
REFERENCE = "SINet-V2"  # mCE normalisation (COD10K-C / ImageNet-C convention)

# key -> (display name, group, input resolution in px; from each wrapper's preprocessing)
MODELS = {
    "SINet-V2": ("SINet-V2", "Specialist", "352"),
    "PFNet": ("PFNet", "Specialist", "416"),
    "ZoomNet": ("ZoomNet", "Specialist", "384*"),
    "FEDER": ("FEDER", "Specialist", "384"),
    "SAM2-UNet": ("SAM2-UNet", "SAM-based", "352"),
    "SAM-Adapter": ("SAM-Adapter", "SAM-based", "1024"),
    "Ours-Full": ("Ours (1024)", "Ours", "1024"),
    "Ours-Full-512": ("Ours (512)", "Ours", "512"),
    "Oracle-Box-MobileSAM": ("MobileSAM + GT box", "Oracle", "1024"),
}
MAIN_MODELS = ["SINet-V2", "PFNet", "ZoomNet", "FEDER", "SAM2-UNet", "SAM-Adapter", "Ours-Full", "Ours-Full-512"]
# Runs of the same configuration with different seeds: reported as mean +- sample std over the seeds
# that are complete for the dataset in question.
SEED_GROUPS = {"Ours-Full-512": ["Ours-Full-512", "Ours-Full-512-s1", "Ours-Full-512-s2"]}


def display(model: str) -> str:
    return MODELS.get(model, (model,))[0]


def load_bench(path: Path | None = None) -> pd.DataFrame:
    b = pd.read_csv(path or RESULTS / "benchmark.csv")
    return b.drop_duplicates(["model", "dataset", "corruption", "severity"], keep="last")


def load_released(path: Path | None = None) -> pd.DataFrame:
    p = path or RESULTS / "released_maps.csv"
    return pd.read_csv(p) if Path(p).exists() else pd.DataFrame(columns=["model", "dataset", *METRICS])


def clean_scores(bench: pd.DataFrame, model: str, dataset: str, released: pd.DataFrame | None = None) -> dict | None:
    """Clean-image metrics for one model/dataset; FEDER-style clean-only models come from released maps."""
    r = bench[(bench.model == model) & (bench.dataset == dataset) & (bench.corruption == "clean")]
    if len(r):
        return {k: float(r.iloc[-1][k]) for k in METRICS}
    if released is not None:
        r = released[(released.model == model) & (released.dataset == dataset)]
        if len(r) and int(r.iloc[-1].get("n_missing", 0) or 0) == 0:
            return {k: float(r.iloc[-1][k]) for k in METRICS}
    return None


def _mce(bench, model, dataset, corrs) -> float | None:
    ces = []
    for c in corrs:
        m = bench[(bench.model == model) & (bench.dataset == dataset) & (bench.corruption == c)]
        r = bench[(bench.model == REFERENCE) & (bench.dataset == dataset) & (bench.corruption == c)]
        if m.empty or len(m) != len(r):
            return None
        ces.append((1 - m.Sm).sum() / (1 - r.Sm).sum())
    return float(np.mean(ces))


def robustness(bench: pd.DataFrame, model: str, dataset: str, protocol: dict | None = None) -> dict | None:
    """Clean score, mCorr, retention (Sm, Dice), seen/unseen/local/global retention and mCE for one
    model/dataset. None unless all 40 corrupted conditions and clean are present."""
    protocol = protocol or load_protocol()
    sub = bench[(bench.model == model) & (bench.dataset == dataset)]
    corr = sub[sub.corruption.isin(protocol["corruptions"])]
    clean = sub[sub.corruption == "clean"]
    if clean.empty or len(corr) != len(protocol["corruptions"]) * len(protocol["severities"]):
        return None
    c = clean.iloc[-1]
    out = {"clean_Sm": float(c.Sm), "mcorr_Sm": float(corr.Sm.mean()), "ret_Sm": float(corr.Sm.mean() / c.Sm),
           "clean_Dice": float(c.Dice), "ret_Dice": float(corr.Dice.mean() / c.Dice),
           "mae_ratio": float(corr.MAE.mean() / c.MAE)}
    for name, cs in (("seen", protocol["seen"]), ("unseen", protocol["unseen"]), ("local", LOCAL), ("global", GLOBAL)):
        g = corr[corr.corruption.isin(cs)]
        out[f"mcorr_{name}"] = float(g.Sm.mean())
        out[f"ret_{name}"] = float(g.Sm.mean() / c.Sm)
    for cname in protocol["corruptions"]:
        g = corr[corr.corruption == cname]
        out[f"mcorr_{cname}"] = float(g.Sm.mean())
        out[f"retDice_{cname}"] = float(g.Dice.mean() / c.Dice)
    out["mCE"] = _mce(bench, model, dataset, protocol["corruptions"])
    return out


def aggregate_seeds(fn, model: str, *args) -> tuple[dict | None, dict | None, int]:
    """Apply fn(member, *args) to every seed of `model` that has results; returns (mean, std, n).
    std is the sample standard deviation (ddof=1) and None when fewer than 2 seeds are available."""
    members = SEED_GROUPS.get(model, [model])
    vals = [v for v in (fn(m, *args) for m in members) if v is not None]
    if not vals:
        return None, None, 0
    keys = [k for k in vals[0] if all(v.get(k) is not None for v in vals)]
    mean = {k: float(np.mean([v[k] for v in vals])) for k in keys}
    std = {k: float(np.std([v[k] for v in vals], ddof=1)) for k in keys} if len(vals) > 1 else None
    return mean, std, len(vals)


def efficiency(path: Path | None = None) -> pd.DataFrame:
    p = path or RESULTS / "efficiency.csv"
    return pd.read_csv(p).drop_duplicates("model", keep="last").set_index("model") if Path(p).exists() else pd.DataFrame()


def training_hours(run: str) -> float | None:
    p = RESULTS / "train" / run / "summary.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return float(d["hours"]) if d.get("status") == "finished" else None


# ---------------------------------------------------------------------------------- per-image scores
def per_image_mean(model: str, dataset: str, corruptions: list[str], root: Path | None = None) -> pd.Series:
    """Mean Sm per image over the given corruptions x severities, from results/per_image."""
    root = root or RESULTS / "per_image" / "benchmark"
    files = []
    for c in corruptions:
        files += glob.glob(str(root / model / dataset / f"{c}-*.csv*"))
    if not files:
        return pd.Series(dtype=float)
    return pd.concat([pd.read_csv(f, usecols=["name", "Sm"]) for f in files]).groupby("name").Sm.mean()


def paired_bootstrap(diff: np.ndarray, n_boot: int = 5000, seed: int = 0, chunk: int = 500) -> tuple[float, float]:
    """95 % percentile CI of the mean of paired per-image differences."""
    rng = np.random.default_rng(seed)
    means = []
    for i in range(0, n_boot, chunk):
        idx = rng.integers(0, len(diff), (min(chunk, n_boot - i), len(diff)))
        means.append(diff[idx].mean(axis=1))
    lo, hi = np.percentile(np.concatenate(means), [2.5, 97.5])
    return float(lo), float(hi)


def available_per_image(model: str, dataset: str, root: Path | None = None) -> int:
    root = root or RESULTS / "per_image" / "benchmark"
    return len([f for f in glob.glob(str(root / model / dataset / "*.csv.gz")) if not os.path.basename(f).startswith("clean")])
