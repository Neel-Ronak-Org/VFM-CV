"""Build results/reproduction.csv: our clean results vs paper-reported numbers vs released maps.

Reads only results/benchmark.csv (clean rows), results/released_maps.csv and configs/reported.yaml.
    python scripts/make_reproduction.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]


def build(bench_csv: Path, maps_csv: Path, reported_yaml: Path) -> pd.DataFrame:
    rep = yaml.safe_load(open(reported_yaml))
    tol = rep.pop("tolerance_Sm")
    bench = pd.read_csv(bench_csv)
    clean = bench[(bench.corruption == "clean") & (bench.severity == 0)]
    clean = clean.drop_duplicates(["model", "dataset"], keep="last")
    maps = pd.read_csv(maps_csv).drop_duplicates(["model", "dataset"], keep="last") if maps_csv.exists() else None
    rows = []
    for _, r in clean.iterrows():
        row = {"model": r.model, "dataset": r.dataset, "Sm_ours": r.Sm, "Em_ours": r.Em, "wFm_ours": r.wFm,
               "MAE_ours": r.MAE, "n_images": r.n_images, "precision": r.get("precision", "")}
        reported = rep.get(r.model, {}).get("Sm", {}).get(r.dataset)
        row["Sm_reported"] = reported
        row["delta_Sm_vs_reported"] = None if reported is None else round(r.Sm - reported, 4)
        row["within_tolerance"] = None if reported is None else abs(r.Sm - reported) <= tol
        if maps is not None:
            m = maps[(maps.model == r.model) & (maps.dataset == r.dataset)]
            if len(m):
                row["Sm_released_maps"] = float(m.Sm.iloc[0])
                row["delta_Sm_vs_released_maps"] = round(r.Sm - float(m.Sm.iloc[0]), 4)
        row["reported_source"] = "; ".join(rep.get(r.model, {}).get("verified_urls", []))
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    df = build(REPO / "results" / "benchmark.csv", REPO / "results" / "released_maps.csv",
               REPO / "configs" / "reported.yaml")
    if df.empty:
        sys.exit("no clean rows in results/benchmark.csv yet")
    out = REPO / "results" / "reproduction.csv"
    df.to_csv(out, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(df[[c for c in df.columns if c != "reported_source"]].to_string(index=False))
    print(f"\nwrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
