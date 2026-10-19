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
        "unreadable": 0, "size_mismatch": 0, "nonbinary_masks": 0, "empty_masks": 0,
    }
    heights, widths, areas, mismatched = [], [], [], []
    if check_pixels:
        for s in ds.samples:
            img = cv2.imread(str(s.image_path), cv2.IMREAD_COLOR)
            m = cv2.imread(str(s.mask_path), cv2.IMREAD_GRAYSCALE)
            if img is None or m is None:
                report["unreadable"] += 1
                continue
            if img.shape[:2] != m.shape[:2]:
                # Not fatal: evaluation resizes predictions to the mask size (standard protocol).
                report["size_mismatch"] += 1
                mismatched.append(s.name)
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
    report["size_mismatch_names"] = ";".join(mismatched)
    report["ok"] = bool(report["count_ok"] and report["missing_masks"] == 0 and report["missing_images"] == 0
                        and report["unreadable"] == 0)
    return report


class CODTrainSet:
    """Training views for our adapter (torch-free; numpy arrays are collated by the DataLoader).

    Per sample, with an RNG seeded by (seed, epoch, index) so every epoch is reproducible and a run
    can resume mid-epoch:
      1. optional corruption of the ORIGINAL-resolution image (as in the benchmark, where corruptions
         are applied before a model resizes its input), drawn from `corruptions` at a random severity;
      2. the same geometric augmentation for every view: random crop keeping `crop_scale` of each side,
         horizontal flip, resize to img_size x img_size (cv2 bilinear, as at test time).
    No photometric augmentation (colour jitter, noise), so Ours-Clean never sees corruption-like inputs.

    paired=False: returns {"image", "mask", "corrupted"}; the image is corrupted with prob corruption_prob.
    paired=True:  returns {"image" (clean), "image_corr" (always corrupted), "mask"} for the consistency loss.
    Images are uint8 [3,S,S]; the mask is float32 [1,S,S] in [0,1] (bilinear-resized, soft edges).
    """

    def __init__(self, base: CODDataset, img_size: int, hflip: bool = True, crop_scale=(0.75, 1.0),
                 corruption_prob: float = 0.0, corruptions=(), severities=(1, 2, 3, 4, 5), paired: bool = False,
                 seed: int = 0):
        if (corruption_prob > 0 or paired) and not corruptions:
            raise ValueError("corruption augmentation needs a non-empty corruption list")
        self.base, self.img_size, self.hflip = base, img_size, hflip
        self.crop_scale = tuple(crop_scale)
        self.corruption_prob, self.corruptions, self.severities = corruption_prob, list(corruptions), list(severities)
        self.paired, self.seed, self.epoch = paired, seed, 0

    def __len__(self) -> int:
        return len(self.base)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _geometry(self, rng, h, w):
        s = rng.uniform(*self.crop_scale)
        ch, cw = max(1, int(round(h * s))), max(1, int(round(w * s)))
        y0, x0 = int(rng.integers(0, h - ch + 1)), int(rng.integers(0, w - cw + 1))
        return (y0, x0, ch, cw), bool(self.hflip and rng.random() < 0.5)

    def _apply(self, arr, geom, interp):
        (y0, x0, ch, cw), flip = geom
        arr = arr[y0:y0 + ch, x0:x0 + cw]
        if flip:
            arr = arr[:, ::-1]
        return cv2.resize(np.ascontiguousarray(arr), (self.img_size, self.img_size), interpolation=interp)

    def _corrupt(self, rng, img):
        from src.corruptions import apply_corruption

        c = self.corruptions[int(rng.integers(len(self.corruptions)))]
        s = int(self.severities[int(rng.integers(len(self.severities)))])
        return apply_corruption(img, c, s, seed=int(rng.integers(2**31)))

    def __getitem__(self, i: int) -> dict:
        rng = np.random.default_rng([self.seed, self.epoch, i])
        _, img, mask = self.base[i]
        if mask.shape != img.shape[:2]:
            mask = cv2.resize(mask.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        corrupt = self.paired or rng.random() < self.corruption_prob
        corr = self._corrupt(rng, img) if corrupt else None
        geom = self._geometry(rng, *img.shape[:2])
        out = {"mask": self._apply(mask.astype(np.float32), geom, cv2.INTER_LINEAR)[None]}
        chw = lambda a: self._apply(a, geom, cv2.INTER_LINEAR).transpose(2, 0, 1).copy()  # noqa: E731
        if self.paired:
            out["image"], out["image_corr"] = chw(img), chw(corr)
        else:
            out["image"], out["corrupted"] = chw(corr if corrupt else img), corrupt
        return out
