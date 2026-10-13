"""Run any model over any (dataset, corruption, severity) combination.

Resumable: combinations already present in the output CSV are skipped, so the benchmark
can be stopped and restarted at any time. One row per (model, dataset, corruption,
severity); clean images are corruption="clean", severity=0.

Examples
--------
Clean evaluation of SINet-V2 on all four test sets:
    python -m src.evaluate --models SINet-V2 --datasets CHAMELEON CAMO COD10K NC4K --corruptions clean
Full corruption grid on COD10K:
    python -m src.evaluate --models SINet-V2 --datasets COD10K --corruptions all
Smoke test with no data or weights:
    python -m src.evaluate --models debug-centre --datasets COD10K --corruptions clean --limit 5
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.corruptions import corrupt_for_benchmark, load_protocol
from src.datasets import CODDataset, load_data_config
from src.metrics import METRIC_NAMES, CODMetrics
from src.models import build_model
from src.utils import REPO, git_hash, write_run_log

KEY = ("model", "dataset", "corruption", "severity")
FIELDS = list(KEY) + list(METRIC_NAMES) + ["n_images", "mean_latency_ms", "seconds", "git", "base_seed", "timestamp"]


def load_done(csv_path: Path) -> set[tuple]:
    if not csv_path.exists():
        return set()
    with open(csv_path) as f:
        return {(r["model"], r["dataset"], r["corruption"], int(r["severity"])) for r in csv.DictReader(f)}


def append_row(csv_path: Path, row: dict) -> None:
    new = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def resolve_conditions(spec: list[str], severities: list[int], protocol: dict) -> list[tuple[str, int]]:
    names: list[str] = []
    for s in spec:
        if s == "all":
            names += protocol["corruptions"]
        elif s in ("seen", "unseen"):
            names += protocol[s]
        else:
            names.append(s)
    conds = []
    for n in dict.fromkeys(names):  # dedupe, keep order
        conds += [("clean", 0)] if n == "clean" else [(n, sev) for sev in severities]
    return conds


def evaluate_condition(model, ds: CODDataset, corruption: str, severity: int, base_seed: int,
                       limit: int | None = None, save_dir: Path | None = None, save_names: set | None = None) -> dict:
    metrics = CODMetrics()
    latencies = []
    n = len(ds) if limit is None else min(limit, len(ds))
    for i in tqdm(range(n), desc=f"{model.name}|{ds.name}|{corruption}-{severity}", leave=False):
        name, img, gt = ds[i]
        img = corrupt_for_benchmark(img, ds.name, name, corruption, severity, base_seed)
        pred = model.predict(img)
        metrics.step(pred, gt)
        latencies.append(model.last_latency_ms or 0.0)
        if save_dir is not None and save_names and name in save_names:
            save_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_dir / f"{name}.png"), np.round(pred * 255).astype(np.uint8))
    res = metrics.results()
    # Skip the first few latencies (CUDA warm-up) when there are enough samples.
    lat = latencies[5:] if len(latencies) > 10 else latencies
    res.update(n_images=n, mean_latency_ms=float(np.mean(lat)))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--datasets", nargs="+", default=["COD10K"])
    ap.add_argument("--corruptions", nargs="+", default=["clean"],
                    help="'clean', 'all', 'seen', 'unseen' or corruption names")
    ap.add_argument("--severities", nargs="+", type=int, default=None)
    ap.add_argument("--out", default="results/benchmark.csv")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="only first N images (smoke tests; not resumable-safe)")
    ap.add_argument("--save-pred", type=int, default=0, help="save predictions for the first N images (qualitative subset)")
    args = ap.parse_args(argv)

    protocol = load_protocol()
    severities = args.severities or protocol["severities"]
    conds = resolve_conditions(args.corruptions, severities, protocol)
    out = Path(args.out) if Path(args.out).is_absolute() else REPO / args.out
    if args.limit:
        out = out.with_name(out.stem + "_smoke.csv")  # never mix partial runs into the real results
    done = load_done(out)
    data_cfg = load_data_config()
    t_start = time.time()

    for model_name in args.models:
        todo = [(d, c, s) for d in args.datasets for c, s in conds if (model_name, d, c, s) not in done]
        if not todo:
            print(f"[skip] {model_name}: all {len(conds) * len(args.datasets)} combinations already in {out}")
            continue
        model = build_model(model_name, device=args.device, fp16=not args.no_fp16)
        for d in args.datasets:
            ds = CODDataset.from_config(d, data_cfg, root=args.data_root)
            save_names = {s.name for s in ds.samples[: args.save_pred]}
            for c, s in [(c, s) for dd, c, s in todo if dd == d]:
                t0 = time.time()
                save_dir = REPO / "results" / "predictions" / model_name / d / f"{c}-{s}" if args.save_pred else None
                res = evaluate_condition(model, ds, c, s, protocol["base_seed"], args.limit, save_dir, save_names)
                row = {"model": model_name, "dataset": d, "corruption": c, "severity": s, **res,
                       "seconds": round(time.time() - t0, 1), "git": git_hash(),
                       "base_seed": protocol["base_seed"], "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
                append_row(out, {k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()})
                print(f"{model_name:>14s} {d:>9s} {c:>16s}-{s}  Sm={res['Sm']:.4f}  MAE={res['MAE']:.4f}  "
                      f"Dice={res['Dice']:.4f}  ({res['mean_latency_ms']:.1f} ms/img)")
    write_run_log("evaluate", {"args": vars(args), "conditions": conds, "out": str(out),
                               "seconds": round(time.time() - t_start, 1)}, log_dir=out.parent / "logs")


if __name__ == "__main__":
    main()
