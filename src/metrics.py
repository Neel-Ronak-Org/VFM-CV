"""COD evaluation metrics.

Standard COD metrics come from PySODMetrics (github.com/lartpang/PySODMetrics), a
Python port verified against Fan et al.'s MATLAB toolbox:
  - S-alpha (structure measure, Fan et al. ICCV 2017)
  - E-phi mean (enhanced-alignment measure, Fan et al. IJCAI 2018), mean over thresholds
  - F-beta^w (weighted F-measure, Margolin et al. CVPR 2014)
  - MAE
Dice and IoU are added for comparison with COD10K-C. They are computed per image after
binarising the prediction at 0.5, then averaged over images. (COD10K-C's exact
definition is not confirmed; check before comparing numbers directly.)

All metrics are dataset-level averages over per-image scores, the usual COD protocol.
"""
from __future__ import annotations

import numpy as np
import py_sod_metrics as psm

METRIC_NAMES = ("Sm", "Em", "wFm", "MAE", "Dice", "IoU")
# Higher is better for everything except MAE.
HIGHER_IS_BETTER = {"Sm": True, "Em": True, "wFm": True, "MAE": False, "Dice": True, "IoU": True}


def _to_uint8(pred: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.float32)
    if pred.max() > 1.0 + 1e-6 or pred.min() < -1e-6:
        raise ValueError("pred must be in [0,1]")
    return np.round(pred * 255).astype(np.uint8)


def _gt_uint8(gt: np.ndarray) -> np.ndarray:
    gt = np.asarray(gt)
    return ((gt > 127) if gt.dtype == np.uint8 and gt.max() > 1 else (gt > 0.5)).astype(np.uint8) * 255


class CODMetrics:
    """Accumulate per-image metrics; call `results()` for dataset-level averages."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.sm = psm.Smeasure()
        self.em = psm.Emeasure()
        self.wfm = psm.WeightedFmeasure()
        self.mae = psm.MAE()
        self.dice: list[float] = []
        self.iou: list[float] = []

    def step(self, pred: np.ndarray, gt: np.ndarray) -> None:
        """pred: float [H,W] in [0,1]; gt: binary mask [H,W] (bool, {0,1} or {0,255})."""
        if pred.shape != gt.shape[:2]:
            raise ValueError(f"shape mismatch: pred {pred.shape} vs gt {gt.shape}")
        p8, g8 = _to_uint8(pred), _gt_uint8(gt)
        for m in (self.sm, self.em, self.wfm, self.mae):
            m.step(pred=p8, gt=g8)
        pb, gb = pred >= self.threshold, g8 > 0
        inter = np.logical_and(pb, gb).sum()
        psum, gsum = pb.sum(), gb.sum()
        union = psum + gsum - inter
        # Empty-vs-empty counts as a perfect match.
        self.dice.append(1.0 if psum + gsum == 0 else 2.0 * inter / (psum + gsum))
        self.iou.append(1.0 if union == 0 else inter / union)

    def __len__(self) -> int:
        return len(self.dice)

    def results(self) -> dict[str, float]:
        if not self.dice:
            raise RuntimeError("no images accumulated")
        return {
            "Sm": float(self.sm.get_results()["sm"]),
            "Em": float(self.em.get_results()["em"]["curve"].mean()),
            "wFm": float(self.wfm.get_results()["wfm"]),
            "MAE": float(self.mae.get_results()["mae"]),
            "Dice": float(np.mean(self.dice)),
            "IoU": float(np.mean(self.iou)),
        }
