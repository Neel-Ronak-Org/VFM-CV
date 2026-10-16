"""SAM-Adapter (Chen et al., "SAM-Adapter: Adapting Segment Anything in Underperformed Scenes",
ICCV 2023 Workshops; arXiv:2304.09148).

Source:   https://github.com/tianrun-chen/SAM-Adapter-PyTorch  (main, commit 041277f), MIT licence
Weights:  the single released checkpoint, Google Drive id 13JilJT7dhxwMIgcdtnvdzr08vcbREFlR (README
          "Pre-trained Models") -> checkpoints/sam_adapter/  (python scripts/download_data.py --weights sam_adapter)
Network:  vendored in third_party/sam_adapter/ (SAM image encoder with the high-frequency/embedding
          prompt adapters, SAM mask decoder with a learned no-mask embedding, i.e. prompt-free).
          Loss modules and training code removed; no other changes.
Backbone: chosen automatically from the checkpoint (patch-embed width 768/1024/1280 -> ViT-B/L/H),
          with the encoder settings of configs/cod-sam-vit-{b,l,h}.yaml.

Inference follows test.py + datasets/wrappers.py ('val'): torchvision Resize((1024, 1024)) on the PIL
image, ImageNet normalisation, model.infer -> 1024x1024 logits, sigmoid. No min-max.
Protocol difference (explain in reproduction): the official script computes its metrics at 1024x1024
against a NEAREST-resized mask with its own metric code; we upsample the logits to the original
resolution and use the standard PySODMetrics protocol, so small gaps to the paper are expected.
fp16 by default (ViT-H is 0.6B parameters).

Low-memory mode (default on CUDA, `lowmem=True`), added 2026-10-16 after the first laptop run took
16-18 s/image: fp32 ViT-H weights (2.5 GB) + autocast's fp16 weight copies + the explicit 4096x4096x16
attention maps of the 4 global-attention blocks exceed 4 GB, and Windows silently spills GPU memory
into system RAM. The wrapper (vendored code untouched) therefore
  1. stores the image-encoder weights in fp16 (1.3 GB); the small mask decoder stays fp32;
  2. computes attention a few heads at a time (identical maths, checked by a unit test);
  3. keeps the FFT high-pass prompt in fp32.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import types

from src.models.base import CODModel, load_state_dict_obj, normalize_imagenet, read_state_dict
from src.models.third_party.sam_adapter.image_encoder import Attention, PromptGenerator, add_decomposed_rel_pos
from src.models.third_party.sam_adapter.network import SAM

REPO = Path(__file__).resolve().parents[2]
WEIGHTS_DIR = REPO / "checkpoints" / "sam_adapter"

_COMMON = dict(name="sam", img_size=1024, mlp_ratio=4, patch_size=16, qkv_bias=True, use_rel_pos=True,
               window_size=14, out_chans=256, prompt_embed_dim=256)
VARIANTS = {  # from configs/cod-sam-vit-{b,l,h}.yaml
    768: dict(_COMMON, embed_dim=768, depth=12, num_heads=12, global_attn_indexes=[2, 5, 8, 11]),
    1024: dict(_COMMON, embed_dim=1024, depth=24, num_heads=16, global_attn_indexes=[5, 11, 17, 23]),
    1280: dict(_COMMON, embed_dim=1280, depth=32, num_heads=16, global_attn_indexes=[7, 15, 23, 31]),
}
VARIANT_NAMES = {768: "ViT-B", 1024: "ViT-L", 1280: "ViT-H"}


def find_weights(folder: Path = WEIGHTS_DIR) -> Path:
    files = sorted([*folder.rglob("*.pth"), *folder.rglob("*.pt")]) if folder.exists() else []
    if len(files) == 1:
        return files[0]
    raise FileNotFoundError(f"expected exactly one checkpoint in {folder}, found {[f.name for f in files]}")


def detect_embed_dim(state_dict: dict) -> int:
    w = next(v for k, v in state_dict.items() if k.endswith("image_encoder.patch_embed.proj.weight"))
    return int(w.shape[0])


def chunked_attention_forward(self, x: torch.Tensor, heads_per_chunk: int = 4) -> torch.Tensor:
    """Drop-in replacement for the vendored Attention.forward that processes `heads_per_chunk`
    heads at a time. Per-head maths (scaled q.k^T + decomposed rel-pos, softmax, @v) is unchanged,
    so outputs match the original up to floating-point rounding; peak memory falls by ~num_heads/chunk."""
    B, H, W, _ = x.shape
    qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
    out = torch.empty_like(q)
    for i in range(0, q.shape[0], heads_per_chunk):
        qi, ki, vi = q[i:i + heads_per_chunk], k[i:i + heads_per_chunk], v[i:i + heads_per_chunk]
        attn = (qi * self.scale) @ ki.transpose(-2, -1)
        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, qi, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))
        out[i:i + heads_per_chunk] = attn.softmax(dim=-1) @ vi
        del attn
    x = out.view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
    return self.proj(x)


def _fft_fp32(self, x, rate):
    """Run the vendored FFT high-pass in fp32 whatever the module dtype (cuFFT half is fragile)."""
    return PromptGenerator.fft(self, x.float(), rate).to(x.dtype)


def enable_lowmem(net: SAM, half_encoder: bool, heads_per_chunk: int = 4) -> None:
    for m in net.image_encoder.modules():
        if isinstance(m, Attention):
            m.forward = types.MethodType(lambda self, x: chunked_attention_forward(self, x, heads_per_chunk), m)
        if isinstance(m, PromptGenerator):
            m.fft = types.MethodType(_fft_fp32, m)
    if half_encoder:
        enc = net.image_encoder.half()
        enc_forward = enc.forward
        enc.forward = lambda x: enc_forward(x.half()).float()   # decoder stays fp32


class SAMAdapter(CODModel):
    name = "SAM-Adapter"
    source = "https://github.com/tianrun-chen/SAM-Adapter-PyTorch@041277f"
    default_fp16 = True
    input_size = 1024

    def __init__(self, weights: str | Path | None | bool = True, embed_dim: int | None = None,
                 lowmem: bool | None = None, **kw):
        super().__init__(**kw)
        if weights is True:
            weights = find_weights()
        sd = read_state_dict(weights) if weights else None   # read once (ViT-H checkpoint is ~2.5 GB)
        if sd is not None and embed_dim is None:
            embed_dim = detect_embed_dim(sd)
        embed_dim = embed_dim or 768
        self.variant = VARIANT_NAMES[embed_dim]
        self.net = SAM(inp_size=self.input_size, encoder_mode=VARIANTS[embed_dim], loss="iou")
        if sd is not None:
            load_state_dict_obj(self.net, sd, Path(weights).name)
            del sd
        self.lowmem = (self.device.type == "cuda") if lowmem is None else lowmem
        if self.lowmem:
            enable_lowmem(self.net, half_encoder=self.fp16)
        self.net.to(self.device).eval()

    def _forward(self, img: np.ndarray) -> torch.Tensor:
        h, w = img.shape[:2]
        x = normalize_imagenet(img, self.input_size, self.device, resize="pil")
        logits = self.net.infer(x)
        return F.interpolate(logits.float(), size=(h, w), mode="bilinear", align_corners=False).sigmoid()
