"""Common model interface for every baseline and for our model.

Each wrapper implements `_forward(img) -> logits or probabilities at any resolution`;
the base class handles device choice, fp16 autocast, inference mode, and resizing
back to the original resolution so `predict` always returns float32 [H,W] in [0,1].
"""
from __future__ import annotations

import abc
import time

import cv2
import numpy as np
import torch


def pick_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CODModel(abc.ABC):
    #: short identifier used in results CSVs, e.g. "SINet-V2"
    name: str = "base"
    #: provenance: source repo, commit/release, weights URL (also in each file's header)
    source: str = ""
    #: True only for "oracle" analysis models that need the ground truth (e.g. a GT-box prompt).
    #: The harness then calls set_gt(mask) before predict(). Never a fair competitor.
    needs_gt: bool = False
    #: precision used when the caller doesn't choose. Small specialist models run in fp32,
    #: as in their papers; large foundation models default to fp16 to fit in 4 GB.
    default_fp16: bool = True

    def __init__(self, device: str | None = None, fp16: bool | None = None):
        self.device = pick_device(device)
        fp16 = self.default_fp16 if fp16 is None else fp16
        # fp16 autocast only on CUDA; CPU falls back to fp32 for correctness and speed.
        self.fp16 = fp16 and self.device.type == "cuda"
        self.last_latency_ms: float | None = None

    def set_gt(self, gt: np.ndarray) -> None:  # only used when needs_gt is True
        raise NotImplementedError

    @abc.abstractmethod
    def _forward(self, img: np.ndarray) -> torch.Tensor:
        """img: RGB uint8 [H,W,3]. Return a [h,w] or [1,1,h,w] tensor of probabilities in [0,1]."""

    @torch.inference_mode()
    def predict(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.fp16):
            prob = self._forward(img)
        prob = prob.float().squeeze()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.last_latency_ms = (time.perf_counter() - t0) * 1000
        prob = prob.clamp(0, 1).cpu().numpy()
        if prob.shape != (h, w):
            prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
        return np.clip(prob, 0.0, 1.0).astype(np.float32)


def load_state_dict_file(model: torch.nn.Module, path, strict: bool = True) -> None:
    """Load a checkpoint saved by any of the usual conventions (raw state_dict, {'state_dict': ...},
    {'model': ...}, DataParallel 'module.' prefixes). Fails loudly on key mismatch when strict."""
    from pathlib import Path

    path = Path(path)
    load_state_dict_obj(model, read_state_dict(path), path.name, strict)


def read_state_dict(path) -> dict:
    """Read a checkpoint file into a plain state_dict (unwrapping common containers and prefixes)."""
    from pathlib import Path

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"weights not found: {path} (see configs/sources.yaml / scripts/download_data.py --weights)")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("state_dict", "model", "net"):
        if isinstance(ckpt, dict) and key in ckpt and isinstance(ckpt[key], dict):
            ckpt = ckpt[key]
            break
    return {k.removeprefix("module."): v for k, v in ckpt.items()}


def load_state_dict_obj(model: torch.nn.Module, state_dict: dict, name: str = "checkpoint", strict: bool = True) -> None:
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if strict and (missing or unexpected):
        raise RuntimeError(f"checkpoint {name} does not match the model: "
                           f"{len(missing)} missing (e.g. {missing[:3]}), {len(unexpected)} unexpected (e.g. {unexpected[:3]})")


def cod_postprocess(logits: torch.Tensor, size: tuple[int, int], minmax: bool = True) -> torch.Tensor:
    """The post-processing used by most official COD test scripts (SINet-V2, SAM2-UNet, ...):
    upsample the logits bilinearly (align_corners=False) to `size` (h, w), sigmoid, then optional
    per-image min-max normalisation (p - min) / (max - min + 1e-8)."""
    import torch.nn.functional as F

    prob = F.interpolate(logits.float(), size=size, mode="bilinear", align_corners=False).sigmoid()
    if minmax:
        prob = (prob - prob.min()) / (prob.max() - prob.min() + 1e-8)
    return prob


def normalize_imagenet(img: np.ndarray, size: int | tuple[int, int], device: torch.device,
                       resize: str = "pil") -> torch.Tensor:
    """Resize to `size` (h, w) and apply ImageNet mean/std, the preprocessing most COD models use.

    resize="pil" matches torchvision `transforms.Resize` on a PIL image (bilinear with
    antialiasing), which is what most official COD test scripts do. "cv2" is plain bilinear.
    """
    if isinstance(size, int):
        size = (size, size)
    if resize == "pil":
        from PIL import Image

        x = np.asarray(Image.fromarray(img).resize((size[1], size[0]), Image.BILINEAR), dtype=np.float32) / 255.0
    else:
        x = cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    return torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(device)


class _Constant(CODModel):
    """Debug model: predicts a centred blob. Used in tests and to smoke-test the harness."""
    name = "debug-centre"
    source = "n/a"

    def _forward(self, img):
        h, w = img.shape[:2]
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, h // 4), torch.linspace(-1, 1, w // 4), indexing="ij")
        return torch.sigmoid(4 * (0.5 - (xx**2 + yy**2)))
