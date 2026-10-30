"""Score a folder of prediction maps (PNG) against the ground truth with our metrics code.

Used for (a) checking a ported baseline against its officially released maps and
(b) the "clean only" fallback for baselines that cannot be ported.

    python scripts/eval_prediction_maps.py --model SINet-V2 --pred-root data/released_maps/SINet-V2
Searches <pred-root> recursively for a folder per dataset (CHAMELEON, CAMO, COD10K, NC4K) and
matches maps to masks by file stem. Appends to results/released_maps.csv.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.datasets import CODDataset, load_data_config, read_mask  # noqa: E402
from src.metrics import METRIC_NAMES, CODMetrics  # noqa: E402
from src.utils import git_hash  # noqa: E402

FIELDS = ["model", "dataset", "source"] + list(METRIC_NAMES) + ["n_images", "n_missing", "pred_root", "git", "timestamp"]


def find_dataset_dir(root: Path, dataset: str) -> Path | None:
    """Pick the folder named like the dataset that holds the most PNGs."""
    cands = [d for d in [root, *root.rglob("*")] if d.is_dir() and d.name.upper() == dataset.upper()]
    cands = [(len(list(d.rglob("*.png"))), d) for d in cands]
    cands = [c for c in cands if c[0] > 0]
    return max(cands)[1] if cands else None


def score(pred_dir: Path, ds: CODDataset) -> tuple[dict, int]:
    preds = {p.stem: p for p in pred_dir.rglob("*.png")}
    m, missing = CODMetrics(), 0
    for s in ds.samples:
        p = preds.get(s.name)
        if p is None:
            missing += 1
            continue
        gt = read_mask(s.mask_path)
        pred = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
        m.step(pred.astype(np.float32) / 255.0, gt)
    return m.results(), missing


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--pred-root", required=True)
    ap.add_argument("--datasets", nargs="+", default=["CHAMELEON", "CAMO", "COD10K", "NC4K"])
    ap.add_argument("--source", default="released_maps")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out", default="results/released_maps.csv")
    args = ap.parse_args(argv)
    root = Path(args.pred_root)
    out = Path(args.out) if Path(args.out).is_absolute() else REPO / args.out
    cfg = load_data_config()
    for d in args.datasets:
        pred_dir = find_dataset_dir(root, d)
        if pred_dir is None:
            print(f"[missing] no {d} folder with PNGs under {root}")
            continue
        res, missing = score(pred_dir, CODDataset.from_config(d, cfg, root=args.data_root))
        n = len(CODDataset.from_config(d, cfg, root=args.data_root)) - missing
        row = {"model": args.model, "dataset": d, "source": args.source,
               **{k: round(v, 6) for k, v in res.items()}, "n_images": n, "n_missing": missing,
               "pred_root": str(pred_dir), "git": git_hash(), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
        new = not out.exists()
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)
        print(f"{args.model} {d}: Sm={res['Sm']:.4f} Em={res['Em']:.4f} wFm={res['wFm']:.4f} MAE={res['MAE']:.4f}"
              f" ({n} maps, {missing} missing)")


if __name__ == "__main__":
    main()
