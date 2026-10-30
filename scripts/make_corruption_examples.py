"""Paper figure: one test image under each of the 8 corruptions at one severity (figures/fig_corruptions).

    python scripts/make_corruption_examples.py                      # teaser image (cropped), severity 4; needs the data
    python scripts/make_corruption_examples.py --severity 3 --crop 0.1 0.2 0.9 0.8
    python scripts/make_corruption_examples.py --image path/to/photo.jpg
    python scripts/make_corruption_examples.py --from-preview       # no data: tiles of figures/corruption_preview.jpg
    python scripts/make_corruption_examples.py --rows 3             # 3x3 grid at column width instead of one row

Corruptions use the benchmark's own code and per-image seeds (src/corruptions.py), so each panel is exactly what
the models were evaluated on. --crop (fractions x0 y0 x1 y1) zooms into the image so that noise, JPEG blocks and
rain streaks stay visible at column width. --from-preview builds a lower-resolution draft from the preview grid
written by scripts/preview_corruptions.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.corruptions import corrupt_for_benchmark, load_protocol  # noqa: E402

DEFAULT_IMAGE = "COD10K-CAM-3-Flying-61-Katydid-4063"  # the teaser image (Fig. 1); portrait, so by default
DEFAULT_CROP = (0.0, 0.5, 1.0, 0.95)                    # crop to a landscape region around the lower katydid
NAMES = {"gaussian_noise": "Gaussian noise", "gaussian_blur": "Gaussian blur", "motion_blur": "Motion blur",
         "brightness": "Brightness", "contrast": "Contrast", "fog": "Fog", "jpeg_compression": "JPEG", "rain": "Rain"}
COL1, COL2 = 3.5, 7.16  # IEEE column / page width (inches)

plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "font.size": 7,
                     "savefig.dpi": 400})


def from_dataset(name: str, dataset: str, severity: int, crop, protocol) -> list[tuple[str, np.ndarray]]:
    from src.datasets import CODDataset
    ds = CODDataset.from_config(dataset)
    idx = next((i for i, s in enumerate(ds.samples) if s.name == name), None)
    if idx is None:
        raise SystemExit(f"{name} not found in {dataset}")
    _, img, _ = ds[idx]
    panels = [("Clean", img)] + [(c, corrupt_for_benchmark(img, dataset, name, c, severity, protocol["base_seed"]))
                                 for c in protocol["seen"] + protocol["unseen"]]
    return [(c, cropped(x, crop)) for c, x in panels]


def from_file(path: str, severity: int, crop, protocol) -> list[tuple[str, np.ndarray]]:
    img = np.asarray(Image.open(path).convert("RGB"))
    panels = [("Clean", img)] + [(c, corrupt_for_benchmark(img, "photo", Path(path).stem, c, severity,
                                                          protocol["base_seed"]))
                                 for c in protocol["seen"] + protocol["unseen"]]
    return [(c, cropped(x, crop)) for c, x in panels]


def from_preview(severity: int, protocol) -> list[tuple[str, np.ndarray]]:
    """Cut tiles out of figures/corruption_preview.jpg (label column 150 px, header 28 px, 6 columns)."""
    grid = np.asarray(Image.open(REPO / "figures" / "corruption_preview.jpg").convert("RGB"))
    rows = protocol["corruptions"]
    tw = (grid.shape[1] - 150) // 6
    th = (grid.shape[0] - 28) // len(rows)
    tile = lambda r, col: grid[28 + r * th + 1: 28 + (r + 1) * th - 1, 150 + col * tw + 1: 150 + (col + 1) * tw - 1]  # noqa: E731
    panels = [("Clean", tile(0, 0))]
    for c in protocol["seen"] + protocol["unseen"]:
        panels.append((c, tile(rows.index(c), severity)))
    return panels


def cropped(img: np.ndarray, crop) -> np.ndarray:
    if not crop:
        return img
    h, w = img.shape[:2]
    x0, y0, x1, y1 = crop
    return img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", default=None, help="a COD10K test image name, or a path to any photo")
    ap.add_argument("--dataset", default="COD10K")
    ap.add_argument("--severity", type=int, default=4)
    ap.add_argument("--crop", type=float, nargs=4, default=None, metavar=("X0", "Y0", "X1", "Y1"))
    ap.add_argument("--from-preview", action="store_true")
    ap.add_argument("--rows", type=int, default=1, help="1: one full-width row (default, paper); 3: 3x3 at column width")
    args = ap.parse_args()
    protocol = load_protocol()
    if args.from_preview:
        panels = from_preview(args.severity, protocol)
    elif args.image and Path(args.image).is_file():
        panels = from_file(args.image, args.severity, args.crop, protocol)
    else:
        crop = args.crop if (args.crop or args.image) else DEFAULT_CROP
        panels = from_dataset(args.image or DEFAULT_IMAGE, args.dataset, args.severity, crop, protocol)

    h, w = panels[0][1].shape[:2]
    rows = args.rows
    cols = -(-len(panels) // rows)
    width = COL2 if rows == 1 else COL1
    tile_w = width / cols
    fig, axes = plt.subplots(rows, cols, figsize=(width, rows * (tile_w * h / w + (0.32 if rows == 1 else 0.2))),
                             squeeze=False)
    for ax, (c, img) in zip(axes.flat, panels):
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        tag = "" if c == "Clean" else ("seen" if c in protocol["seen"] else "unseen")
        title = NAMES.get(c, c) + (f"\n({tag})" if rows == 1 and tag else f" ({tag})" if tag else ("\n" if rows == 1 else ""))
        ax.set_title(title, fontsize=6.5 if rows == 1 else 7, pad=2, linespacing=1.0)
        for s in ax.spines.values():
            s.set_linewidth(0.4)
    for ax in list(axes.flat)[len(panels):]:
        ax.axis("off")
    fig.subplots_adjust(left=0.003, right=0.997, top=0.8 if rows == 1 else 0.95, bottom=0.01, wspace=0.04,
                        hspace=0.18)
    out = REPO / "figures"
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_corruptions.{ext}", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    src = "preview draft" if args.from_preview else (args.image or DEFAULT_IMAGE)
    print(f"wrote figures/fig_corruptions.pdf/.png ({src}, severity {args.severity})")


if __name__ == "__main__":
    main()
