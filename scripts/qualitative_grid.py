"""Side-by-side qualitative grid: image | ground truth | one column per model, with per-image S-alpha.

    python scripts/qualitative_grid.py --models Oracle-Box-MobileSAM SINet-V2 SAM2-UNet --dataset CHAMELEON
    python scripts/qualitative_grid.py --models SINet-V2 SAM2-UNet --dataset COD10K --corruption motion_blur --severity 5

Rows = the `--worst` images on which the FIRST model scores lowest (failure cases) plus `--random`
other images (fixed seed). For oracle models the GT box given to SAM is drawn on the GT tile.
Writes figures/qual_<dataset>_<corruption><severity>.jpg (commit it) and prints the per-image scores.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.corruptions import corrupt_for_benchmark, load_protocol  # noqa: E402
from src.datasets import CODDataset  # noqa: E402
from src.metrics import CODMetrics  # noqa: E402
from src.models import build_model  # noqa: E402
from src.models.oracle_box_sam import gt_box  # noqa: E402


def predict(model, img, gt):
    if model.needs_gt:
        model.set_gt(gt)
    p = model.predict(img)
    if p.shape != gt.shape:
        p = cv2.resize(p, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
    return p


def sm(pred, gt) -> float:
    m = CODMetrics()
    m.step(pred, gt)
    return m.results()["Sm"]


def tile(arr_rgb, h, label=None):
    w = int(round(arr_rgb.shape[1] * h / arr_rgb.shape[0]))
    t = cv2.resize(arr_rgb, (w, h), interpolation=cv2.INTER_AREA)
    if label:
        cv2.rectangle(t, (0, 0), (min(w, 9 * len(label) + 8), 22), (0, 0, 0), -1)
        cv2.putText(t, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return t


def gray(x):
    return cv2.cvtColor((np.clip(x, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--dataset", default="CHAMELEON")
    ap.add_argument("--corruption", default="clean")
    ap.add_argument("--severity", type=int, default=0)
    ap.add_argument("--worst", type=int, default=4)
    ap.add_argument("--random", type=int, default=4)
    ap.add_argument("--scan", type=int, default=0, help="score only the first N images when choosing rows (0 = all)")
    ap.add_argument("--row-height", type=int, default=180)
    args = ap.parse_args()

    ds = CODDataset.from_config(args.dataset)
    base_seed = load_protocol()["base_seed"]
    models = [build_model(m) for m in args.models]
    first = models[0]
    n_scan = len(ds) if args.scan <= 0 else min(args.scan, len(ds))

    print(f"scoring {first.name} on {n_scan} {args.dataset} images to pick rows ...")
    scores = []
    for i in range(n_scan):
        name, img, gt = ds[i]
        img = corrupt_for_benchmark(img, args.dataset, name, args.corruption, args.severity, base_seed)
        scores.append((sm(predict(first, img, gt), gt), i))
    scores.sort()
    worst = [i for _, i in scores[: args.worst]]
    rest = [i for _, i in scores[args.worst:]]
    rng = np.random.default_rng(0)
    pick = worst + sorted(rng.choice(rest, size=min(args.random, len(rest)), replace=False).tolist())

    rows, h = [], args.row_height
    for i in pick:
        name, img, gt = ds[i]
        img = corrupt_for_benchmark(img, args.dataset, name, args.corruption, args.severity, base_seed)
        gt_rgb = gray(gt.astype(np.float32))
        if any(m.needs_gt for m in models):
            b = gt_box(gt, gt.shape).astype(int)
            cv2.rectangle(gt_rgb, (b[0], b[1]), (b[2] - 1, b[3] - 1), (255, 0, 0), max(2, gt.shape[0] // 150))
        img_disp = cv2.resize(img, (gt.shape[1], gt.shape[0])) if img.shape[:2] != gt.shape else img
        tiles = [tile(img_disp, h, name[:24]), tile(gt_rgb, h, "GT" + (" + oracle box" if any(m.needs_gt for m in models) else ""))]
        line = [f"{name:40s}"]
        for m in models:
            p = predict(m, img, gt)
            s = sm(p, gt)
            tiles.append(tile(gray(p), h, f"{m.name[:18]} S={s:.3f}"))
            line.append(f"{m.name}={s:.3f}")
        print("  ".join(line) + ("   <- worst for " + first.name if i in worst else ""))
        rows.append(np.hstack(tiles))
    width = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 4), (0, width - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
    tag = "clean" if args.corruption == "clean" else f"{args.corruption}{args.severity}"
    out = REPO / "figures" / f"qual_{args.dataset}_{tag}.jpg"
    out.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(out), cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"wrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
