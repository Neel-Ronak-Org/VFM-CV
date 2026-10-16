"""Phase 3 summary: clean score, mean under corruption (mCorr) and retention, per model x dataset.

    python scripts/summarize_benchmark.py            # reads results/benchmark.csv -> results/summary_robustness.csv

For every (model, dataset) and every corruption GROUP it reports, per metric:
    clean      score on clean images
    mcorr      mean over the group's corruptions x severities
    retention  mcorr / clean            (Sm, Em, wFm, Dice, IoU: higher = more robust)
    mae_ratio  mcorr / clean for MAE    (lower = more robust; MAE grows under corruption)
Groups: all (8 COD10K-C corruptions), seen, unseen (held-out protocol), each family
(noise, blur, illumination, weather, digital) and each single corruption.
Also mCE_Sm: COD10K-C/ImageNet-C-style corruption error normalised to SINet-V2 on the same dataset,
CE_c = sum_s (1 - Sm_{c,s}) / sum_s (1 - Sm^{SINet-V2}_{c,s}), mCE = mean_c CE_c (lower = more robust;
SINet-V2 = 1 by definition). `complete` is False when some of the group's conditions are missing.
Reads only results/benchmark.csv and configs/corruptions.yaml.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.corruptions import load_protocol  # noqa: E402

HIGHER = ["Sm", "Em", "wFm", "Dice", "IoU"]
REFERENCE = "SINet-V2"


def groups(protocol: dict) -> dict[str, list[str]]:
    g = {"all": protocol["corruptions"], "seen": protocol["seen"], "unseen": protocol["unseen"]}
    g.update({f"family:{k}": v for k, v in protocol["families"].items()})
    g.update({c: [c] for c in protocol["corruptions"]})
    return g


def mce(bench: pd.DataFrame, model: str, dataset: str, corrs: list[str]) -> float | None:
    ces = []
    for c in corrs:
        m = bench[(bench.model == model) & (bench.dataset == dataset) & (bench.corruption == c)]
        r = bench[(bench.model == REFERENCE) & (bench.dataset == dataset) & (bench.corruption == c)]
        if m.empty or r.empty or len(m) != len(r):
            return None
        den = (1 - r.Sm).sum()
        if den <= 0:
            return None
        ces.append((1 - m.Sm).sum() / den)
    return float(np.mean(ces)) if ces else None


def summarize(bench: pd.DataFrame, protocol: dict) -> pd.DataFrame:
    bench = bench.drop_duplicates(["model", "dataset", "corruption", "severity"], keep="last")
    n_sev = len(protocol["severities"])
    out = []
    for (model, dataset), sub in bench.groupby(["model", "dataset"]):
        clean = sub[sub.corruption == "clean"]
        if clean.empty:
            continue
        clean = clean.iloc[-1]
        for gname, corrs in groups(protocol).items():
            cs = sub[sub.corruption.isin(corrs)]
            if cs.empty:
                continue
            row = {"model": model, "dataset": dataset, "group": gname, "n_conditions": len(cs),
                   "complete": len(cs) == len(corrs) * n_sev}
            for k in HIGHER:
                row[f"{k}_clean"] = clean[k]
                row[f"{k}_mcorr"] = cs[k].mean()
                row[f"{k}_retention"] = cs[k].mean() / clean[k] if clean[k] > 0 else np.nan
            row["MAE_clean"], row["MAE_mcorr"] = clean.MAE, cs.MAE.mean()
            row["MAE_ratio"] = cs.MAE.mean() / clean.MAE if clean.MAE > 0 else np.nan
            row["mCE_Sm"] = mce(bench, model, dataset, corrs) if row["complete"] else None
            out.append(row)
    return pd.DataFrame(out)


def main():
    bench = pd.read_csv(REPO / "results" / "benchmark.csv")
    df = summarize(bench, load_protocol())
    out = REPO / "results" / "summary_robustness.csv"
    df.round(6).to_csv(out, index=False)
    view = df[df.group.isin(["all", "seen", "unseen"])][
        ["model", "dataset", "group", "n_conditions", "complete", "Sm_clean", "Sm_mcorr", "Sm_retention",
         "Dice_retention", "mCE_Sm"]]
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(view.round(4).to_string(index=False) if len(view) else "no corrupted results yet")
    print(f"\nwrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
