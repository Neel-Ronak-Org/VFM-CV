"""Deterministic image corruptions (ImageNet-C style) for the COD robustness benchmark.

Wraps the `imagecorruptions` package (Michaelis et al. 2019, a packaged version of
Hendrycks & Dietterich's ImageNet-C code). Two things are added:

1. A compatibility shim: imagecorruptions 1.1.2 was written for NumPy 1.x and
   scikit-image < 0.19. On NumPy 2 `fog` fails (`np.float_` removed) and `gaussian_blur`
   fails (`multichannel` renamed to `channel_axis`). We patch both at import time instead
   of pinning an old environment.
2. Per-image deterministic seeding: the seed is derived from (base_seed, dataset, image
   name, corruption, severity) with SHA-256, so the same corrupted image is produced on
   any machine and in any processing order. Generating on the fly is therefore
   equivalent to generating once and storing to disk.
"""
from __future__ import annotations

import hashlib
import random
import warnings
from pathlib import Path

import numpy as np
import yaml

# --- compatibility shim (must run before importing imagecorruptions) -----------------
if not hasattr(np, "float_"):
    np.float_ = np.float64  # type: ignore[attr-defined]

with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # pkg_resources deprecation warning inside the package
    import imagecorruptions.corruptions as _ic
    from imagecorruptions import corrupt as _corrupt
    from imagecorruptions import get_corruption_names

_skimage_gaussian = _ic.gaussian


def _gaussian_compat(*args, multichannel=None, **kwargs):
    if multichannel is not None and "channel_axis" not in kwargs:
        kwargs["channel_axis"] = -1 if multichannel else None
    return _skimage_gaussian(*args, **kwargs)


_ic.gaussian = _gaussian_compat
# --------------------------------------------------------------------------------------

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "corruptions.yaml"
ALL_CORRUPTIONS = tuple(get_corruption_names("all"))


def load_protocol(path: str | Path = CONFIG_PATH) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    names = [c for fam in cfg["families"].values() for c in fam]
    unknown = set(names) - set(ALL_CORRUPTIONS)
    if unknown:
        raise ValueError(f"Unknown corruptions in {path}: {sorted(unknown)}")
    if set(cfg["seen"]) | set(cfg["unseen"]) != set(names) or set(cfg["seen"]) & set(cfg["unseen"]):
        raise ValueError("seen/unseen must partition the corruption list")
    cfg["corruptions"] = names
    return cfg


def image_seed(base_seed: int, dataset: str, image_name: str, corruption: str, severity: int) -> int:
    """Stable 32-bit seed. Uses SHA-256, not Python's salted hash()."""
    key = f"{base_seed}|{dataset}|{Path(image_name).stem}|{corruption}|{severity}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "little")


def apply_corruption(img: np.ndarray, corruption: str, severity: int, seed: int | None = None) -> np.ndarray:
    """Corrupt an RGB uint8 image [H,W,3]. severity 0 returns the image unchanged.

    Output is uint8 with the same shape. imagecorruptions draws from NumPy's global RNG,
    so we seed it (and `random`) right before the call.
    """
    if severity == 0 or corruption in (None, "clean"):
        return img
    if corruption not in ALL_CORRUPTIONS:
        raise ValueError(f"Unknown corruption: {corruption}")
    if not 1 <= severity <= 5:
        raise ValueError(f"Severity must be 0-5, got {severity}")
    if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected RGB uint8 [H,W,3], got {img.dtype} {img.shape}")
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    out = _corrupt(img, corruption_name=corruption, severity=severity)
    out = np.asarray(out)
    if out.shape != img.shape:  # some corruptions can return float or resized output
        import cv2

        out = cv2.resize(out, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.clip(out, 0, 255).astype(np.uint8)


def corrupt_for_benchmark(img: np.ndarray, dataset: str, image_name: str, corruption: str,
                          severity: int, base_seed: int) -> np.ndarray:
    seed = image_seed(base_seed, dataset, image_name, corruption, severity)
    return apply_corruption(img, corruption, severity, seed)
