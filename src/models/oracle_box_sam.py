"""ORACLE upper bound: zero-shot MobileSAM prompted with the ground-truth bounding box.

NOT A FAIR COMPETITOR. It is given the answer's location (a box around every ground-truth pixel),
which no real detector has. It is reported only as an analysis upper bound, to separate
"finding the object" (what COD models must do) from "delineating it once found" (what SAM is good at),
and to see how much corruption hurts SAM's segmentation even with a perfect prompt.

Model:    MobileSAM (Zhang et al., "Faster Segment Anything: Towards Lightweight SAM for Mobile
          Applications", arXiv:2306.14289), https://github.com/ChaoningZhang/MobileSAM (commit f706ad9),
          Apache-2.0. Vendored unchanged in third_party/mobile_sam/ (timm import path updated).
Weights:  https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt
          -> checkpoints/mobile_sam/mobile_sam.pt  (python scripts/download_data.py --weights mobile_sam)
          Strict load verified in the dev container.
Prompt:   one box = tight bounding box of all foreground pixels of the (clean) GT mask, scaled to the
          image size; multimask_output=False; SAM's own preprocessing (longest side 1024, pad).
          Images with multiple objects get one box around all of them. Empty mask -> all-zero map.
Output:   sigmoid of SAM's mask logits at the original resolution. No min-max.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.models.base import CODModel
from src.models.third_party.mobile_sam.build_sam import build_sam_vit_t
from src.models.third_party.mobile_sam.predictor import SamPredictor

REPO = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = REPO / "checkpoints" / "mobile_sam" / "mobile_sam.pt"


def gt_box(gt: np.ndarray, img_hw: tuple[int, int]) -> np.ndarray | None:
    ys, xs = np.nonzero(gt)
    if len(xs) == 0:
        return None
    sy, sx = img_hw[0] / gt.shape[0], img_hw[1] / gt.shape[1]
    return np.array([xs.min() * sx, ys.min() * sy, (xs.max() + 1) * sx, (ys.max() + 1) * sy], np.float32)


class OracleBoxMobileSAM(CODModel):
    name = "Oracle-Box-MobileSAM"
    source = "https://github.com/ChaoningZhang/MobileSAM@f706ad9 (oracle GT-box prompt)"
    needs_gt = True
    default_fp16 = False  # 10M parameters; fp32 is cheap and avoids SamPredictor dtype issues

    def __init__(self, weights: str | Path | None = DEFAULT_WEIGHTS, **kw):
        super().__init__(**kw)
        if weights is not None and not Path(weights).exists():
            raise FileNotFoundError(f"weights not found: {weights}")
        self.sam = build_sam_vit_t(str(weights) if weights is not None else None).to(self.device).eval()
        self.predictor = SamPredictor(self.sam)
        self._gt = None

    def set_gt(self, gt: np.ndarray) -> None:
        self._gt = np.asarray(gt) > 0

    def _forward(self, img: np.ndarray) -> torch.Tensor:
        if self._gt is None:
            raise RuntimeError("oracle model: call set_gt(mask) before predict()")
        box = gt_box(self._gt, img.shape[:2])
        self._gt = None  # never reuse a stale box
        if box is None:
            return torch.zeros(img.shape[:2])
        self.predictor.set_image(img)
        logits, _, _ = self.predictor.predict(box=box, multimask_output=False, return_logits=True)
        return torch.from_numpy(logits[0]).float().sigmoid()
