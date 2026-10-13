"""SAM2-UNet (Xiong et al., "SAM2-UNet: Segment Anything 2 Makes Strong Encoder for Natural and
Medical Image Segmentation", arXiv:2408.08870; Visual Intelligence 2025).

Source:   https://github.com/WZH0120/SAM2-UNet  (commit 01598e5), Apache-2.0
Weights:  COD checkpoint from the README's Google Drive folder
          https://drive.google.com/drive/folders/1w2fK8kLhtEmMWZ6G6w9_J17xwgfm3lev
          -> checkpoints/sam2_unet/<file>.pth   (python scripts/download_data.py --weights sam2_unet)
Network:  vendored in third_party/sam2_unet/. Porting change: the Hiera-L trunk is built directly
          with the arguments of sam2_hiera_l.yaml instead of via build_sam2 + hydra (the original
          then deleted every other SAM2 part). No other changes.

Why it is in the benchmark: SAM2-Adapter / SAM3-Adapter have no released COD weights, so SAM2-UNet
is our SAM2-based model (frozen Hiera-L encoder + small adapters + U-Net decoder).

Inference follows the official test.py: PIL resize to 352x352, ImageNet normalisation, first output,
logits upsampled to full resolution, sigmoid, per-image min-max. fp16 by default (216M parameters).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.models.base import CODModel, cod_postprocess, load_state_dict_file, normalize_imagenet
from src.models.third_party.sam2_unet.network import SAM2UNet as _SAM2UNet

REPO = Path(__file__).resolve().parents[2]
WEIGHTS_DIR = REPO / "checkpoints" / "sam2_unet"


def find_weights(folder: Path = WEIGHTS_DIR) -> Path:
    """The Drive folder holds checkpoints for several tasks; pick the camouflage one."""
    files = sorted(folder.rglob("*.pth")) if folder.exists() else []
    cod = [f for f in files if any(t in f.name.lower() for t in ("cod", "camo", "camouflag"))]
    if len(cod) == 1 or (len(files) == 1 and not cod):
        return (cod or files)[0]
    raise FileNotFoundError(f"could not pick a single COD checkpoint in {folder}; found {[f.name for f in files]}. "
                            "Pass weights=<path> explicitly.")


class SAM2UNet(CODModel):
    name = "SAM2-UNet"
    source = "https://github.com/WZH0120/SAM2-UNet@01598e5"
    default_fp16 = True
    input_size = 352

    def __init__(self, weights: str | Path | None | bool = True, minmax: bool = True, **kw):
        super().__init__(**kw)
        self.minmax = minmax
        self.net = _SAM2UNet()
        if weights is True:
            weights = find_weights()
        if weights:  # None/False = random init (tests only)
            load_state_dict_file(self.net, weights)
        self.net.to(self.device).eval()

    def _forward(self, img: np.ndarray) -> torch.Tensor:
        h, w = img.shape[:2]
        x = normalize_imagenet(img, self.input_size, self.device, resize="pil")
        return cod_postprocess(self.net(x)[0], (h, w), self.minmax)
