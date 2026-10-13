"""SINet-V2 (Fan, Ji, Cheng, Shao, "Concealed Object Detection", IEEE TPAMI 2022).

Source:   https://github.com/GewelsJI/SINet-V2  (commit 5b9844c94d10a03b815c7c7039c194166a9a8ba3)
Weights:  Net_epoch_best.pth, Google Drive id 1D3RKQ8Nzd0ArV_c47StVKEuaoYTwnclR (README "Prepare the data")
          -> checkpoints/sinet_v2/Net_epoch_best.pth  (python scripts/download_data.py --weights sinet_v2)
Licence:  Apache-2.0 LICENSE file; README says research and education use only.
Network:  vendored unchanged in third_party/sinet_v2/ (only import path / pretrained-path edits).

Inference follows the official MyTesting.py exactly:
  1. PIL bilinear resize to 352x352, ImageNet normalisation (torchvision Resize/ToTensor/Normalize);
  2. take the last output (S_3, "res2" in MyTesting.py);
  3. upsample the *logits* bilinearly (align_corners=False) to the original resolution, then sigmoid;
  4. per-image min-max normalisation: (p - min) / (max - min + 1e-8).
Step 4 is part of the official protocol and is kept for reproduction. Note that it stretches
any prediction to the full [0,1] range, which matters under corruption; set minmax=False to ablate.
Runs in fp32 by default (as in the paper); the model is small.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.models.base import CODModel, cod_postprocess, load_state_dict_file, normalize_imagenet
from src.models.third_party.sinet_v2.network import Network

REPO = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = REPO / "checkpoints" / "sinet_v2" / "Net_epoch_best.pth"


class SINetV2(CODModel):
    name = "SINet-V2"
    source = "https://github.com/GewelsJI/SINet-V2@5b9844c"
    default_fp16 = False
    input_size = 352

    def __init__(self, weights: str | Path | None = DEFAULT_WEIGHTS, minmax: bool = True, **kw):
        super().__init__(**kw)
        self.minmax = minmax
        self.net = Network(imagenet_pretrained=False)
        if weights is not None:  # None = random init (tests only)
            load_state_dict_file(self.net, weights)
        self.net.to(self.device).eval()

    def _forward(self, img: np.ndarray) -> torch.Tensor:
        h, w = img.shape[:2]
        x = normalize_imagenet(img, self.input_size, self.device, resize="pil")
        return cod_postprocess(self.net(x)[-1], (h, w), self.minmax)
