"""PFNet (Mei, Ji, Wei, Yang, Wei, Fan, "Camouflaged Object Segmentation with Distraction Mining", CVPR 2021).

Source:   https://github.com/Mhaiyang/CVPR2021_PFNet  (commit 2c4cab0), BSD-style licence
Weights:  PFNet.pth from the project page https://mhaiyang.github.io/CVPR2021_PFNet/index.html
          -> checkpoints/pfnet/PFNet.pth  (manual download)
Network:  vendored in third_party/pfnet/. Porting change: the ImageNet ResNet-50 is only loaded when a
          backbone path is given; the PFNet checkpoint holds all weights.

Inference follows the official infer.py:
  1. torchvision Resize((416, 416)) on the PIL image (bilinear, antialiased), ToTensor, ImageNet Normalize;
  2. take the 4th output (predict1, already sigmoid-ed inside the network in eval mode);
  3. ToPILImage (float -> uint8 by x*255 truncation) and torchvision Resize((h, w)) back to the
     original size; the saved 8-bit PNG is what gets evaluated.
No min-max normalisation. fp32 by default.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.models.base import CODModel, load_state_dict_file, normalize_imagenet
from src.models.third_party.pfnet.network import PFNet as _PFNet

REPO = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = REPO / "checkpoints" / "pfnet" / "PFNet.pth"


class PFNet(CODModel):
    name = "PFNet"
    source = "https://github.com/Mhaiyang/CVPR2021_PFNet@2c4cab0"
    default_fp16 = False
    input_size = 416

    def __init__(self, weights: str | Path | None = DEFAULT_WEIGHTS, **kw):
        super().__init__(**kw)
        self.net = _PFNet(backbone_path=None)
        if weights is not None:
            load_state_dict_file(self.net, weights)
        self.net.to(self.device).eval()

    def _forward(self, img: np.ndarray) -> torch.Tensor:
        h, w = img.shape[:2]
        x = normalize_imagenet(img, self.input_size, self.device, resize="pil")
        prob = self.net(x)[3].float().squeeze()                      # sigmoid already applied
        u8 = (prob.clamp(0, 1) * 255).byte().cpu().numpy()           # ToPILImage: mul(255).byte()
        u8 = np.asarray(Image.fromarray(u8, mode="L").resize((w, h), Image.BILINEAR))
        return torch.from_numpy(u8.astype(np.float32) / 255.0)
