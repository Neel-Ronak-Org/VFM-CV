"""Render a grid of every corruption at every severity for one image (visual sanity check).

    python scripts/preview_corruptions.py                         # built-in sample image
    python scripts/preview_corruptions.py --image data/raw/TestDataset/COD10K/Imgs/<name>.jpg
    python scripts/preview_corruptions.py --extended              # also ImageNet-C extras
Writes figures/corruption_preview.jpg.
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
from src.datasets import read_rgb  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None)
    ap.add_argument("--extended", action="store_true")
    ap.add_argument("--tile", type=int, default=256, help="tile width in px")
    ap.add_argument("--out", default="figures/corruption_preview.jpg")
    args = ap.parse_args()

    if args.image:
        img = read_rgb(args.image)
    else:
        from skimage import data
        img = data.chelsea()
    h = int(round(img.shape[0] * args.tile / img.shape[1]))
    small = lambda x: cv2.resize(x, (args.tile, h), interpolation=cv2.INTER_AREA)  # noqa: E731

    p = load_protocol()
    names = p["corruptions"] + (p["extended_unseen"] if args.extended else [])
    label_w = 150
    rows = []
    header = np.full((28, label_w + args.tile * 6, 3), 255, np.uint8)
    for j, t in enumerate(["clean"] + [f"severity {s}" for s in p["severities"]]):
        cv2.putText(header, t, (label_w + j * args.tile + 8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    rows.append(header)
    for c in names:
        tiles = [small(img)] + [small(corrupt_for_benchmark(img, "preview", "img", c, s, p["base_seed"]))
                                for s in p["severities"]]
        label = np.full((h, label_w, 3), 255, np.uint8)
        tag = "seen" if c in p["seen"] else ("unseen" if c in p["unseen"] else "ext")
        cv2.putText(label, c, (6, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(label, f"({tag})", (6, h // 2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1)
        rows.append(np.hstack([label] + tiles))
    grid = np.vstack(rows)
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
