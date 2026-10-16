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
import gzip
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.corruptions import corrupt_for_benchmark, load_protocol
from src.datasets import CODDataset, load_data_config
from src.metrics import METRIC_NAMES, aggregate, image_scores
from src.models import build_model
from src.utils import REPO, git_hash, write_run_log

KEY = ("model", "dataset", "corruption", "severity")
FIELDS = list(KEY) + list(METRIC_NAMES) + ["n_images", "mean_latency_ms", "precision", "device", "seconds", "git",
                                            "base_seed", "timestamp"]


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
        elif s == "extended":
            names += protocol.get("extended_unseen", [])
        else:
            names.append(s)
    conds = []
    for n in dict.fromkeys(names):  # dedupe, keep order
        conds += [("clean", 0)] if n == "clean" else [(n, sev) for sev in severities]
    return conds


PER_IMAGE_FIELDS = ["name", *METRIC_NAMES, "latency_ms"]
PREFETCH = 4  # images loaded + corrupted ahead of the GPU


def per_image_path(out: Path, model: str, dataset: str, corruption: str, severity: int) -> Path:
    return out.parent / "per_image" / out.stem / model / dataset / f"{corruption}-{severity}.csv"


def _read_rows(path: Path) -> list[dict]:
    if path.with_suffix(".csv.gz").exists():
        with gzip.open(path.with_suffix(".csv.gz"), "rt", newline="") as f:
            return list(csv.DictReader(f))
    if path.exists():
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        return [r for r in rows if r.get("latency_ms") not in (None, "")]  # drop a half-written last line
    return []


def evaluate_condition(model, ds: CODDataset, corruption: str, severity: int, base_seed: int,
                       limit: int | None = None, save_dir: Path | None = None, save_names: set | None = None,
                       per_image: Path | None = None, workers: int = 2) -> dict:
    """Score one (dataset, corruption, severity). With `per_image`, every image's scores are appended
    to that CSV as soon as they are computed, so an interrupted run resumes where it stopped; when the
    condition is complete the file is gzipped (results/per_image/... .csv.gz, kept for statistics).
    With workers > 0, loading/corrupting (1 thread) and metric computation (`workers` threads)
    overlap with GPU inference; results are recorded in image order, identical to workers=0."""
    n = len(ds) if limit is None else min(limit, len(ds))
    rows = _read_rows(per_image) if per_image is not None else []
    done = {r["name"] for r in rows}
    fh = writer = None
    if per_image is not None and len(done) < n:
        per_image.parent.mkdir(parents=True, exist_ok=True)
        if rows:  # rewrite cleanly (drops any half-written line), then continue appending
            with open(per_image, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=PER_IMAGE_FIELDS)
                w.writeheader()
                w.writerows({k: r[k] for k in PER_IMAGE_FIELDS} for r in rows)
        fh = open(per_image, "a", newline="")
        writer = csv.DictWriter(fh, fieldnames=PER_IMAGE_FIELDS)
        if not rows:
            writer.writeheader()
    def load(i):  # runs in ONE loader thread (ImageNet-C corruptions use a global RNG)
        name, img, gt = ds[i]
        return name, corrupt_for_benchmark(img, ds.name, name, corruption, severity, base_seed), gt

    def record(row):
        rows.append(row)
        if writer is not None:
            writer.writerow({k: (repr(float(v)) if isinstance(v, float) else v) for k, v in row.items()})  # full precision
            fh.flush()

    todo = [i for i in range(n) if ds.samples[i].name not in done]
    desc = f"{model.name}|{ds.name}|{corruption}-{severity}" + (f" (resume {len(done)}/{n})" if done else "")
    loader = ThreadPoolExecutor(1) if workers > 0 else None
    scorer = ThreadPoolExecutor(workers) if workers > 0 else None
    pending: deque = deque()          # (name, latency, future) in image order
    try:
        ahead = deque(loader.submit(load, i) for i in todo[:PREFETCH]) if loader else None
        nxt = PREFETCH
        for i in tqdm(todo, desc=desc, leave=False):
            if loader:
                name, img, gt = ahead.popleft().result()
                if nxt < len(todo):
                    ahead.append(loader.submit(load, todo[nxt]))
                    nxt += 1
            else:
                name, img, gt = load(i)
            if model.needs_gt:
                model.set_gt(gt)
            pred = model.predict(img)
            if pred.shape != gt.shape:
                # A few test images (2 in CHAMELEON) differ in size from their mask. Standard COD
                # evaluation scores at the mask's resolution, so resize the prediction to it.
                pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
            lat = model.last_latency_ms or 0.0
            if save_dir is not None and save_names and name in save_names:
                save_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(save_dir / f"{name}.png"), np.round(pred * 255).astype(np.uint8))
            if scorer:
                pending.append((name, lat, scorer.submit(image_scores, pred, gt)))
                while pending and (pending[0][2].done() or len(pending) > 2 * workers):
                    nm, lt, fut = pending.popleft()
                    record({"name": nm, **fut.result(), "latency_ms": lt})
            else:
                record({"name": name, **image_scores(pred, gt), "latency_ms": lat})
        while pending:
            nm, lt, fut = pending.popleft()
            record({"name": nm, **fut.result(), "latency_ms": lt})
    except BaseException:
        # Interrupted (crash, Ctrl+C): save the images already predicted before stopping.
        while pending:
            nm, lt, fut = pending.popleft()
            try:
                record({"name": nm, **fut.result(), "latency_ms": lt})
            except Exception:  # noqa: BLE001
                break
        raise
    finally:
        for ex in (loader, scorer):
            if ex is not None:
                ex.shutdown(wait=True, cancel_futures=True)
        if fh is not None:
            fh.close()
    if per_image is not None and per_image.exists() and len(rows) >= n:
        with open(per_image, "rb") as src, gzip.open(per_image.with_suffix(".csv.gz"), "wb") as dst:
            dst.write(src.read())
        per_image.unlink()
    res = aggregate(rows)
    # Skip the first few latencies (CUDA warm-up) when there are enough samples.
    lat = [float(r["latency_ms"]) for r in rows]
    lat = lat[5:] if len(lat) > 10 else lat
    res.update(n_images=len(rows), mean_latency_ms=float(np.mean(lat)))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--datasets", nargs="+", default=["COD10K"])
    ap.add_argument("--corruptions", nargs="+", default=["clean"],
                    help="'clean', 'all' (8 COD10K-C corruptions), 'seen', 'unseen', 'extended' or names")
    ap.add_argument("--severities", nargs="+", type=int, default=None)
    ap.add_argument("--out", default="results/benchmark.csv")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--precision", choices=["auto", "fp16", "fp32"], default="auto",
                    help="auto = each model's default (fp32 for small specialists, fp16 for large models)")
    ap.add_argument("--limit", type=int, default=None, help="only first N images (smoke tests; not resumable-safe)")
    ap.add_argument("--save-pred", type=int, default=0, help="save predictions for the first N images (qualitative subset)")
    ap.add_argument("--workers", type=int, default=2,
                    help="metric threads overlapping with GPU inference (0 = fully sequential)")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="do not START a new condition after this many hours (finished work is kept; rerun to continue)")
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
        fp16 = {"auto": None, "fp16": True, "fp32": False}[args.precision]
        model = build_model(model_name, device=args.device, fp16=fp16)
        for d in args.datasets:
            ds = CODDataset.from_config(d, data_cfg, root=args.data_root)
            save_names = {s.name for s in ds.samples[: args.save_pred]}
            for c, s in [(c, s) for dd, c, s in todo if dd == d]:
                if args.max_hours is not None and time.time() - t_start > args.max_hours * 3600:
                    print(f"[stop] --max-hours {args.max_hours} reached; rerun the same command to continue")
                    write_run_log("evaluate", {"args": vars(args), "stopped_early": True, "out": str(out),
                                               "seconds": round(time.time() - t_start, 1)}, log_dir=out.parent / "logs")
                    return
                t0 = time.time()
                save_dir = REPO / "results" / "predictions" / model_name / d / f"{c}-{s}" if args.save_pred else None
                res = evaluate_condition(model, ds, c, s, protocol["base_seed"], args.limit, save_dir, save_names,
                                         per_image=per_image_path(out, model_name, d, c, s), workers=args.workers)
                row = {"model": model_name, "dataset": d, "corruption": c, "severity": s, **res,
                       "precision": "fp16" if model.fp16 else "fp32", "device": str(model.device),
                       "seconds": round(time.time() - t0, 1), "git": git_hash(),
                       "base_seed": protocol["base_seed"], "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
                append_row(out, {k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()})
                print(f"{model_name:>14s} {d:>9s} {c:>16s}-{s}  Sm={res['Sm']:.4f}  MAE={res['MAE']:.4f}  "
                      f"Dice={res['Dice']:.4f}  ({res['mean_latency_ms']:.1f} ms/img)")
    write_run_log("evaluate", {"args": vars(args), "conditions": conds, "out": str(out),
                               "seconds": round(time.time() - t_start, 1)}, log_dir=out.parent / "logs")


if __name__ == "__main__":
    main()
