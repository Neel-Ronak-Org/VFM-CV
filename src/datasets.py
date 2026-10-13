"""COD dataset loaders (image + binary mask).

Expected layout (SINet-V2 packaging, see configs/data.yaml):
    <root>/<split_dir>/Imgs/*.jpg   RGB images
    <root>/<split_dir>/GT/*.png     masks, 0 = background, 255 = camouflaged object
Images and masks are paired by file stem.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
# Some packages name the mask folder differently; tried in order if `mask_subdir` is missing.
MASK_SUBDIR_ALTERNATIVES = ("GT", "GT_Object", "GT_Mask", "Mask", "masks")
REPO = Path(__file__).resolve().parents[1]
DATA_CONFIG = REPO / "configs" / "data.yaml"


def load_data_config(path: str | Path = DATA_CONFIG) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def read_rgb(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"cannot read image {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_mask(path: str | Path) -> np.ndarray:
    """Return a bool mask [H,W]. Anti-aliased edges are thresholded at 128."""
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise IOError(f"cannot read mask {path}")
    return m >= 128


@dataclass(frozen=True)
class Sample:
    name: str
    image_path: Path
    mask_path: Path


class CODDataset:
    """List-style dataset returning (name, rgb uint8 [H,W,3], mask bool [H,W])."""

    def __init__(self, split_dir: str | Path, image_subdir: str = "Imgs", mask_subdir: str = "GT",
                 name: str | None = None):
        self.root = Path(split_dir)
        self.name = name or self.root.name
        img_dir, mask_dir = self.root / image_subdir, self.root / mask_subdir
        if not mask_dir.is_dir():
            mask_dir = next((self.root / m for m in MASK_SUBDIR_ALTERNATIVES if (self.root / m).is_dir()), mask_dir)
        if not img_dir.is_dir() or not mask_dir.is_dir():
            raise FileNotFoundError(f"expected {img_dir} and {mask_dir}")
        imgs = {p.stem: p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS}
        masks = {p.stem: p for p in mask_dir.iterdir() if p.suffix.lower() in IMG_EXTS}
        self.missing_masks = sorted(set(imgs) - set(masks))
        self.missing_images = sorted(set(masks) - set(imgs))
        self.samples = [Sample(s, imgs[s], masks[s]) for s in sorted(set(imgs) & set(masks))]

    @classmethod
    def from_config(cls, dataset: str, cfg: dict | None = None, root: str | Path | None = None) -> "CODDataset":
        cfg = cfg or load_data_config()
        base = Path(root or cfg["root"])
        if not base.is_absolute():
            base = REPO / base
        sub = cfg["train"]["dir"] if dataset == "train" else cfg["test"][dataset]["dir"]
        return cls(base / sub, cfg["image_subdir"], cfg["mask_subdir"], name=dataset)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        s = self.samples[i]
        return s.name, read_rgb(s.image_path), read_mask(s.mask_path)


def verify_dataset(ds: CODDataset, expected: int | None = None, check_pixels: bool = True) -> dict:
    """Check counts, pairing, size alignment and mask binarity. Returns a report dict."""
    report = {
        "dataset": ds.name, "n_pairs": len(ds), "expected": expected,
        "count_ok": expected is None or len(ds) == expected,
        "missing_masks": len(ds.missing_masks), "missing_images": len(ds.missing_images),
        "size_mismatch": 0, "nonbinary_masks": 0, "empty_masks": 0,
    }
    heights, widths, areas = [], [], []
    if check_pixels:
        for s in ds.samples:
            img = cv2.imread(str(s.image_path), cv2.IMREAD_COLOR)
            m = cv2.imread(str(s.mask_path), cv2.IMREAD_GRAYSCALE)
            if img is None or m is None or img.shape[:2] != m.shape[:2]:
                report["size_mismatch"] += 1
                continue
            vals = np.unique(m)
            # "Binary" here allows a few anti-aliased edge values; flag masks where >1% of pixels are grey.
            if not set(vals.tolist()) <= {0, 255} and np.mean((m > 0) & (m < 255)) > 0.01:
                report["nonbinary_masks"] += 1
            if not (m >= 128).any():
                report["empty_masks"] += 1
            heights.append(m.shape[0])
            widths.append(m.shape[1])
            areas.append(float((m >= 128).mean()))
    if heights:  # dataset statistics for the paper's dataset section
        report.update(mean_height=round(float(np.mean(heights)), 1), mean_width=round(float(np.mean(widths)), 1),
                      min_side=int(min(min(heights), min(widths))), max_side=int(max(max(heights), max(widths))),
                      mean_object_fraction=round(float(np.mean(areas)), 4))
    report["ok"] = bool(report["count_ok"] and report["missing_masks"] == 0 and report["missing_images"] == 0
                        and report["size_mismatch"] == 0)
    return report
