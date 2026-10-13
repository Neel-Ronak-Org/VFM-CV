"""ZoomNet (Pang, Zhao, Xiang, Zhang, Lu, "Zoom In and Out: A Mixed-scale Triplet Network for
Camouflaged Object Detection", CVPR 2022).

Source:   https://github.com/lartpang/ZoomNet  (commit 9c65e6c), MIT licence
Weights:  https://github.com/lartpang/ZoomNet/releases/download/v0.0.1/cod_zoomnet_r50_bs8_e40_2022-03-04.pth
          -> checkpoints/zoomnet/  (python scripts/download_data.py --weights zoomnet)
          Strict load verified in the dev container: 901/901 keys match.
Network:  vendored in third_party/zoomnet/. Porting changes: import paths, registry decorator
          removed, timm ResNet-50 built with pretrained=False (checkpoint holds all weights),
          timm.models.layers -> timm.layers.

Inference follows the official test.py with configs/zoomnet/cod_zoomnet.py (dataset msi_cod_te):
  1. ImageNet-normalise the full-resolution RGB image (albumentations Normalize, float32);
  2. cv2 INTER_LINEAR resize to 0.5x, 1.0x, 1.5x of 384x384 -> 192, 384, 576;
  3. sigmoid of the logits (384x384);
  4. cv2 INTER_LINEAR resize of the probability map to the mask size.
No min-max normalisation (test.sh does not pass --minmax-results) and no TTA (tta.enable=False).
fp32 by default.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from src.models.base import CODModel, load_state_dict_file
from src.models.third_party.zoomnet.network import ZoomNet as _ZoomNet

REPO = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = REPO / "checkpoints" / "zoomnet" / "cod_zoomnet_r50_bs8_e40_2022-03-04.pth"
MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255.0
STD = np.array([0.229, 0.224, 0.225], np.float32) * 255.0


class ZoomNet(CODModel):
    name = "ZoomNet"
    source = "https://github.com/lartpang/ZoomNet@9c65e6c"
    default_fp16 = False
    base_size = 384
    scales = (0.5, 1.0, 1.5)

    def __init__(self, weights: str | Path | None = DEFAULT_WEIGHTS, **kw):
        super().__init__(**kw)
        self.net = _ZoomNet()
        if weights is not None:
            load_state_dict_file(self.net, weights)
        self.net.to(self.device).eval()
        self.net.is_training = False

    def _forward(self, img: np.ndarray) -> torch.Tensor:
        x = (img.astype(np.float32) - MEAN) / STD
        ims = []
        for s in self.scales:
            n = int(self.base_size * s)
            r = cv2.resize(x, (n, n), interpolation=cv2.INTER_LINEAR)
            ims.append(torch.from_numpy(r.transpose(2, 0, 1)).unsqueeze(0).to(self.device))
        logits = self.net(data={"image0.5": ims[0], "image1.0": ims[1], "image1.5": ims[2]})
        # Base class resizes the probability map to the input size with cv2 INTER_LINEAR, as the
        # official script does (it resizes to the mask size; evaluate.py handles mask-size mismatches).
        return logits.float().sigmoid()
