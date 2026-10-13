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

    def __init__(self, device: str | None = None, fp16: bool = True):
        self.device = pick_device(device)
        # fp16 autocast only on CUDA; CPU falls back to fp32 for correctness and speed.
        self.fp16 = fp16 and self.device.type == "cuda"
        self.last_latency_ms: float | None = None

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


def normalize_imagenet(img: np.ndarray, size: int | tuple[int, int], device: torch.device) -> torch.Tensor:
    """Resize to `size` and apply ImageNet mean/std, the preprocessing most COD models use."""
    if isinstance(size, int):
        size = (size, size)
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
