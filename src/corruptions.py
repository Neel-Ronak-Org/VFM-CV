"""Deterministic image corruptions for the COD robustness benchmark.

Two sources:

1. **COD10K-C corruptions (core benchmark).** Eight corruptions whose five severity
   parameters are stated in COD10K-C (Sayem, arXiv:2606.02603, Sec. 3); the values live
   in configs/corruptions.yaml. COD10K-C does not publish code, so the implementations
   below are ours. Where the paper is silent we made the simplest standard choice
   (PIL for Gaussian blur / brightness / contrast / JPEG, OpenCV for the motion-blur
   filter, clipping to [0,255]). Rain has no numeric definition in the paper, so its
   parameters are our own. See docs/EXPLAINER.md §2 for the full list of assumptions.

2. **ImageNet-C corruptions (optional extension)** via the `imagecorruptions` package
   (Michaelis et al. 2019, packaging Hendrycks & Dietterich 2019), with a compatibility
   shim for NumPy 2 / scikit-image >= 0.19.

Every corrupted image gets its own seed, derived with SHA-256 from (base_seed, dataset,
image name, corruption, severity). The same corrupted image is produced on any machine
and in any order, so corrupting on the fly equals generating once and storing.
"""
from __future__ import annotations

import functools
import hashlib
import io
import random
import warnings
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageEnhance, ImageFilter

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "corruptions.yaml"

CODC_CORRUPTIONS = ("gaussian_noise", "motion_blur", "gaussian_blur", "brightness",
                    "contrast", "fog", "jpeg_compression", "rain")
IMAGENET_C_CORRUPTIONS = (
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur", "motion_blur",
    "zoom_blur", "snow", "frost", "fog", "brightness", "contrast", "elastic_transform", "pixelate",
    "jpeg_compression", "speckle_noise", "gaussian_blur", "spatter", "saturate")


# ----------------------------------------------------------------------------------------
# Protocol
# ----------------------------------------------------------------------------------------
@functools.lru_cache(maxsize=4)
def _load_protocol_cached(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    params = cfg["params"]
    unknown = set(params) - set(CODC_CORRUPTIONS)
    if unknown:
        raise ValueError(f"Unknown core corruptions in {path}: {sorted(unknown)}")
    for name, p in params.items():
        if len(p) != len(cfg["severities"]):
            raise ValueError(f"{name}: need {len(cfg['severities'])} severity parameters, got {len(p)}")
    seen, unseen = set(cfg["seen"]), set(cfg["unseen"])
    if seen & unseen or seen | unseen != set(params):
        raise ValueError("seen/unseen must partition the core corruption list")
    fam = [c for f in cfg["families"].values() for c in f]
    if sorted(fam) != sorted(params):
        raise ValueError("families must list every core corruption exactly once")
    ext = set(cfg.get("extended_unseen", []))
    if not ext <= set(IMAGENET_C_CORRUPTIONS) or ext & set(params):
        raise ValueError("extended_unseen must be ImageNet-C corruptions not already in the core set")
    cfg["corruptions"] = list(params)
    return cfg


def load_protocol(path: str | Path = CONFIG_PATH) -> dict:
    return _load_protocol_cached(str(path))


def image_seed(base_seed: int, dataset: str, image_name: str, corruption: str, severity: int) -> int:
    """Stable 32-bit seed. Uses SHA-256, not Python's salted hash()."""
    key = f"{base_seed}|{dataset}|{Path(image_name).stem}|{corruption}|{severity}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "little")


# ----------------------------------------------------------------------------------------
# COD10K-C corruptions. img: RGB uint8 [H,W,3]; rng: seeded np.random.Generator.
# ----------------------------------------------------------------------------------------
def _gaussian_noise(img, sigma, rng):
    noisy = img.astype(np.float32) + rng.normal(0.0, sigma, img.shape).astype(np.float32)
    return np.clip(np.round(noisy), 0, 255).astype(np.uint8)


def _motion_blur(img, k, rng):
    kernel = np.full((1, int(k)), 1.0 / k, np.float32)   # horizontal box kernel of length k
    return cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REFLECT_101)


def _gaussian_blur(img, radius, rng):
    return np.asarray(Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius=radius)))


def _brightness(img, factor, rng):
    return np.asarray(ImageEnhance.Brightness(Image.fromarray(img)).enhance(factor))


def _contrast(img, factor, rng):
    return np.asarray(ImageEnhance.Contrast(Image.fromarray(img)).enhance(factor))


def _fog(img, alpha, rng):
    out = (1.0 - alpha) * img.astype(np.float32) + alpha * 255.0   # additive blend with white canvas
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def _jpeg(img, quality, rng):
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"))


RAIN_ANGLE_DEG = 80.0   # streak direction (90 = vertical); slight slant
RAIN_OPACITY = 0.6      # max blend weight of a streak towards white


def _rain_kernel(length: int, angle_deg: float = RAIN_ANGLE_DEG) -> np.ndarray:
    size = int(length) | 1  # odd
    k = np.zeros((size, size), np.float32)
    c = size // 2
    dx = np.cos(np.deg2rad(angle_deg)) * (length - 1) / 2
    dy = np.sin(np.deg2rad(angle_deg)) * (length - 1) / 2
    cv2.line(k, (int(round(c - dx)), int(round(c + dy))), (int(round(c + dx)), int(round(c - dy))), 1.0, 1)
    return k


def _rain(img, p, rng):
    """Synthetic rain streaks: random seed pixels smeared along a slanted line, blended to white."""
    h, w = img.shape[:2]
    seeds = (rng.random((h, w)) < p["density"]).astype(np.float32)
    seeds *= rng.uniform(0.5, 1.0, (h, w)).astype(np.float32)          # per-streak brightness
    streaks = cv2.filter2D(seeds, -1, _rain_kernel(p["length"]), borderType=cv2.BORDER_CONSTANT)
    streaks = cv2.GaussianBlur(np.clip(streaks, 0, 1), (3, 3), 0.5)[..., None] * RAIN_OPACITY
    out = img.astype(np.float32) * (1 - streaks) + 255.0 * streaks
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


_CODC_FUNCS = {
    "gaussian_noise": _gaussian_noise, "motion_blur": _motion_blur, "gaussian_blur": _gaussian_blur,
    "brightness": _brightness, "contrast": _contrast, "fog": _fog, "jpeg_compression": _jpeg, "rain": _rain,
}


# ----------------------------------------------------------------------------------------
# ImageNet-C (optional) -- lazy import with NumPy-2 / scikit-image compatibility shim
# ----------------------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def _imagenet_c():
    if not hasattr(np, "float_"):
        np.float_ = np.float64  # type: ignore[attr-defined]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # pkg_resources deprecation warning
        import imagecorruptions.corruptions as ic
        from imagecorruptions import corrupt

    sk_gaussian = ic.gaussian

    def gaussian_compat(*args, multichannel=None, **kwargs):
        if multichannel is not None and "channel_axis" not in kwargs:
            kwargs["channel_axis"] = -1 if multichannel else None
        return sk_gaussian(*args, **kwargs)

    ic.gaussian = gaussian_compat
    return corrupt


# ----------------------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------------------
def apply_corruption(img: np.ndarray, corruption: str, severity: int, seed: int | None = None,
                     source: str = "codc", protocol: dict | None = None) -> np.ndarray:
    """Corrupt an RGB uint8 image [H,W,3]; returns uint8 of the same shape.

    source="codc": COD10K-C definitions (default, core benchmark).
    source="imagenet_c": imagecorruptions package (its own severity scale).
    severity 0 or corruption "clean" returns the input unchanged.
    """
    if severity == 0 or corruption in (None, "clean"):
        return img
    if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected RGB uint8 [H,W,3], got {img.dtype} {img.shape}")
    if not 1 <= severity <= 5:
        raise ValueError(f"Severity must be 0-5, got {severity}")

    if source == "codc":
        if corruption not in _CODC_FUNCS:
            raise ValueError(f"Unknown COD10K-C corruption: {corruption}")
        protocol = protocol or load_protocol()
        rng = np.random.default_rng(seed)
        out = _CODC_FUNCS[corruption](img, protocol["params"][corruption][severity - 1], rng)
    elif source == "imagenet_c":
        if corruption not in IMAGENET_C_CORRUPTIONS:
            raise ValueError(f"Unknown ImageNet-C corruption: {corruption}")
        if seed is not None:  # imagecorruptions draws from the global RNGs
            np.random.seed(seed)
            random.seed(seed)
        out = np.asarray(_imagenet_c()(img, corruption_name=corruption, severity=severity))
        if out.shape != img.shape:
            out = cv2.resize(out, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
        out = np.clip(out, 0, 255)
    else:
        raise ValueError(f"unknown source {source!r}")
    return np.ascontiguousarray(out, dtype=np.uint8)


def source_of(corruption: str, protocol: dict | None = None) -> str:
    protocol = protocol or load_protocol()
    if corruption in protocol["params"]:
        return "codc"
    if corruption in protocol.get("extended_unseen", []):
        return "imagenet_c"
    raise ValueError(f"{corruption} is not in the protocol")


def corrupt_for_benchmark(img: np.ndarray, dataset: str, image_name: str, corruption: str,
                          severity: int, base_seed: int) -> np.ndarray:
    if severity == 0 or corruption == "clean":
        return img
    seed = image_seed(base_seed, dataset, image_name, corruption, severity)
    return apply_corruption(img, corruption, severity, seed, source=source_of(corruption))
