"""Phase 3: run the corruption benchmark plan in configs/benchmark.yaml (resumable).

    python scripts/run_benchmark.py --status              # what's done, what's left, rough ETA
    python scripts/run_benchmark.py                       # run everything still missing
    python scripts/run_benchmark.py --max-hours 8         # stop starting new work after 8 h (rerun to continue)
    python scripts/run_benchmark.py --stage 1             # only the first stage
    python scripts/run_benchmark.py --include-optional    # also SAM-Adapter on NC4K (~140 h)

Every (model, dataset, corruption, severity) result goes to results/benchmark.csv; per-image scores
go to results/per_image/ so even a crash in the middle of a condition loses at most one image.
At the end it writes results/summary_robustness.csv (scripts/summarize_benchmark.py).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src import evaluate  # noqa: E402
from src.corruptions import load_protocol  # noqa: E402
from src.datasets import load_data_config  # noqa: E402

PLAN = REPO / "configs" / "benchmark.yaml"
OVERHEAD_S = 0.3  # per image when only a profile exists: loading, corrupting, scoring (measured ~0.3 s)


def load_plan(include_optional: bool) -> tuple[dict, list[dict]]:
    plan = yaml.safe_load(open(PLAN))
    stages = list(plan["stages"]) + (list(plan.get("optional", [])) if include_optional else [])
    return plan, stages


def conditions(plan: dict) -> list[tuple[str, int]]:
    p = load_protocol()
    return evaluate.resolve_conditions(["clean", plan.get("corruptions", "all")], p["severities"], p)


def timing_rows() -> pd.DataFrame | None:
    """Every evaluate output in results/ (benchmark.csv and one-off checks) with timing columns."""
    frames = []
    for f in (REPO / "results").glob("*.csv"):
        try:
            df = pd.read_csv(f)
        except Exception:  # noqa: BLE001
            continue
        if {"model", "seconds", "n_images", "timestamp", "git"} <= set(df.columns):
            frames.append(df[["model", "dataset", "seconds", "n_images", "timestamp", "git"]])
    return pd.concat(frames, ignore_index=True) if frames else None


def sec_per_image(model: str, timings: pd.DataFrame | None, eff: pd.DataFrame | None) -> float | None:
    """Median wall-clock seconds per image over the model's runs made with the same code version
    (git commit) as its most recent run - older rows (e.g. SAM-Adapter before low-memory mode) are ignored."""
    if timings is not None:
        t = timings[(timings.model == model) & (timings.n_images > 0)].sort_values("timestamp")
        if len(t):
            t = t[t.git == t.git.iloc[-1]]
            return float((t.seconds / t.n_images).median())
    if eff is not None and (eff.model == model).any():
        return float(eff[eff.model == model].latency_ms_median.iloc[0]) / 1000 + OVERHEAD_S
    return None


def status(stages: list[dict], conds: list[tuple[str, int]]) -> float:
    bench_p, eff_p = REPO / "results" / "benchmark.csv", REPO / "results" / "efficiency.csv"
    timings = timing_rows()
    eff = pd.read_csv(eff_p) if eff_p.exists() else None
    done = evaluate.load_done(bench_p)
    sizes = {d: v["expected"] for d, v in load_data_config()["test"].items()}
    total_h = 0.0
    print(f"{'stage / model':40s} {'dataset':10s} {'done':>9s} {'left':>5s} {'ETA':>9s}")
    for st in stages:
        print(f"-- {st['name']}")
        for d in st["datasets"]:
            for m in st["models"]:
                n_done = sum((m, d, c, s) in done for c, s in conds)
                left = len(conds) - n_done
                spi = sec_per_image(m, timings, eff)
                eta_h = left * sizes[d] * spi / 3600 if spi else float("nan")
                total_h += 0 if eta_h != eta_h else eta_h
                eta = "done" if left == 0 else (f"{eta_h:6.1f} h" if spi else "  ?")
                print(f"   {m:37s} {d:10s} {n_done:4d}/{len(conds):<4d} {left:5d} {eta:>9s}")
    print(f"\nestimated time left: {total_h:.1f} h  (~{total_h / 24:.1f} days of continuous running; based on recent"
          " sequential runs, so the parallel pipeline should finish sooner)")
    return total_h


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument("--stage", type=int, default=None, help="run only this stage (1-based)")
    ap.add_argument("--include-optional", action="store_true")
    args = ap.parse_args()
    plan, stages = load_plan(args.include_optional)
    if args.stage is not None:
        stages = [stages[args.stage - 1]]
    conds = conditions(plan)
    if args.status:
        status(stages, conds)
        return
    t0 = time.time()
    for st in stages:
        for d in st["datasets"]:
            for m in st["models"]:
                left_h = None if args.max_hours is None else args.max_hours - (time.time() - t0) / 3600
                if left_h is not None and left_h <= 0:
                    print(f"[stop] --max-hours {args.max_hours} reached; rerun to continue")
                    break
                cli = ["--models", m, "--datasets", d, "--corruptions", "clean", plan.get("corruptions", "all"),
                       "--save-pred", str(plan.get("save_pred", {}).get(d, 0))]
                if left_h is not None:
                    cli += ["--max-hours", f"{left_h:.4f}"]
                evaluate.main(cli)
    subprocess.run([sys.executable, str(REPO / "scripts" / "summarize_benchmark.py")], check=False)
    status(stages, conds)


if __name__ == "__main__":
    main()
