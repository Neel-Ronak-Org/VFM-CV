"""FEDER (He, Li, Zhang, Tang, Zhang, Guo, Li, "Camouflaged Object Detection with Feature Decomposition
and Edge Reconstruction", CVPR 2023).

Source:   https://github.com/ChunmingHe/FEDER  (commit fac6b2a), MIT licence
Weights:  "well-trained model", Google Drive id 1MONpM9auqGlRoyaOKUe6wJgLZ-E6A4Dc (README §4)
          -> checkpoints/feder/  (python scripts/download_data.py --weights feder)
Network:  vendored in third_party/feder/ (import paths only). timm ResNet-50 (pretrained=False, as in
          the original Network class), channels=96.
Protocol note: per its README, FEDER is trained on COD10K-train only (3,040 images), not the
          4,040-image COD10K+CAMO set the other baselines use. Report this in the paper.

Inference follows the official Test.py: PIL resize to 384x384 (default --testsize), ImageNet
normalisation, output index 4 (final map f1), logits upsampled to full resolution, sigmoid,
per-image min-max. fp32 by default.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.models.base import CODModel, cod_postprocess, load_state_dict_file, normalize_imagenet
from src.models.third_party.feder.network import Network

REPO = Path(__file__).resolve().parents[2]
WEIGHTS_DIR = REPO / "checkpoints" / "feder"


def find_weights(folder: Path = WEIGHTS_DIR) -> Path:
    files = sorted(folder.rglob("*.pth")) if folder.exists() else []
    if len(files) == 1:
        return files[0]
    raise FileNotFoundError(f"expected exactly one .pth in {folder}, found {[f.name for f in files]}; pass weights=<path>")


class FEDER(CODModel):
    name = "FEDER"
    source = "https://github.com/ChunmingHe/FEDER@fac6b2a"
    default_fp16 = False
    input_size = 384

    def __init__(self, weights: str | Path | None | bool = True, minmax: bool = True, **kw):
        super().__init__(**kw)
        self.minmax = minmax
        self.net = Network(channels=96)
        if weights is True:
            weights = find_weights()
        if weights:
            load_state_dict_file(self.net, weights)
        self.net.to(self.device).eval()

    def _forward(self, img: np.ndarray) -> torch.Tensor:
        h, w = img.shape[:2]
        x = normalize_imagenet(img, self.input_size, self.device, resize="pil")
        return cod_postprocess(self.net(x)[4], (h, w), self.minmax)
