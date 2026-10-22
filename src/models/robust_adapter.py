"""OUR MODEL: Robust Lightweight SAM Adapter (prompt-free, frozen MobileSAM encoder).

Backbone: MobileSAM (Zhang et al., arXiv:2306.14289), https://github.com/ChaoningZhang/MobileSAM
          (commit f706ad9, Apache-2.0), vendored in third_party/mobile_sam/. Its weights
          checkpoints/mobile_sam/mobile_sam.pt are loaded strictly and the image encoder stays FROZEN.
Design:   (see docs/EXPLAINER.md section 9)
  - Frozen TinyViT image encoder (6.07M params). BatchNorm stays in eval mode.
  - Trainable adapters inside the encoder's 10 transformer blocks (stages 1-3), zero-initialised so
    training starts from exactly the pretrained encoder:
      "bottleneck": AdaptFormer-style parallel adapter next to each block's MLP,
                    x + MLP(x) + Up(GELU(Down(LN(x)))),  hidden = dim * adapter_ratio
      "lora":       low-rank update of each attention qkv projection, W x + B A x   (ablation)
      "none":       encoder fully frozen, only the decoder learns (ablation)
    The convolutional stage 0 has no adapters and runs without gradients.
  - Prompt-free decoding: SAM's mask decoder (trainable, initialised from MobileSAM) receives
    LEARNED prompt tokens instead of user clicks. The first two tokens start as SAM's embedding of a
    box around the whole image; extra tokens start small and random. The dense prompt starts as SAM's
    "no mask" embedding and is learned too.
  - Output: one logit map from the decoder's first mask token (4x the embedding grid), upsampled to
    the input size. predict() returns its sigmoid (no min-max).
Input:    RGB resized to img_size x img_size (cv2 bilinear; same function in training and testing),
          normalised with SAM's pixel mean/std.
Checkpoint (written by src/train.py): only the trainable parameters + the model config; the frozen
          encoder always comes from mobile_sam.pt.
"""
from __future__ import annotations

import math
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from src.models.base import CODModel
from src.models.third_party.mobile_sam.modeling import MaskDecoder, PromptEncoder, TinyViT, TwoWayTransformer

REPO = Path(__file__).resolve().parents[2]
DEFAULT_SAM_WEIGHTS = REPO / "checkpoints" / "mobile_sam" / "mobile_sam.pt"
OURS_DIR = REPO / "checkpoints" / "ours"
SAM_PIXEL_MEAN = np.array([123.675, 116.28, 103.53], np.float32)
SAM_PIXEL_STD = np.array([58.395, 57.12, 57.375], np.float32)

MODEL_DEFAULTS = dict(img_size=1024, adapter="bottleneck", adapter_ratio=0.25, lora_rank=8, n_prompt_tokens=4)


def resize_image(img: np.ndarray, size: int) -> np.ndarray:
    """RGB uint8 [H,W,3] -> uint8 [size,size,3]. Used identically by training and inference."""
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)


def normalize_batch(x: torch.Tensor) -> torch.Tensor:
    """uint8/float [B,3,S,S] in 0..255 -> SAM-normalised float."""
    mean = torch.as_tensor(SAM_PIXEL_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.as_tensor(SAM_PIXEL_STD, device=x.device).view(1, 3, 1, 1)
    return (x.float() - mean) / std


class BottleneckAdapter(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.up = nn.Linear(hidden, dim)
        nn.init.zeros_(self.up.weight)  # zero output at init: the block is unchanged
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return self.up(self.act(self.down(self.norm(x))))


class MlpWithAdapter(nn.Module):
    """Replaces a TinyViT block's `mlp`, so the block computes x + MLP(x) + Adapter(x)."""

    def __init__(self, mlp: nn.Module, adapter: nn.Module):
        super().__init__()
        self.mlp = mlp
        self.adapter = adapter

    def forward(self, x):
        return self.mlp(x) + self.adapter(x)


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float | None = None):
        super().__init__()
        self.base = base
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        self.scale = (alpha or rank) / rank

    def forward(self, x):
        return self.base(x) + self.lora_b(self.lora_a(x)) * self.scale


def build_mobile_sam_parts(img_size: int):
    """MobileSAM's encoder/prompt encoder/decoder exactly as in build_sam_vit_t, but for any img_size
    (multiple of 64). The weights do not depend on img_size, so mobile_sam.pt loads strictly."""
    if img_size % 64:
        raise ValueError("img_size must be a multiple of 64")
    encoder = TinyViT(img_size=img_size, in_chans=3, num_classes=1000, embed_dims=[64, 128, 160, 320],
                      depths=[2, 2, 6, 2], num_heads=[2, 4, 5, 10], window_sizes=[7, 7, 14, 7], mlp_ratio=4.,
                      drop_rate=0., drop_path_rate=0.0, use_checkpoint=False, mbconv_expand_ratio=4.0,
                      local_conv_size=3, layer_lr_decay=0.8)
    grid = img_size // 16
    prompt_encoder = PromptEncoder(embed_dim=256, image_embedding_size=(grid, grid),
                                   input_image_size=(img_size, img_size), mask_in_chans=16)
    decoder = MaskDecoder(num_multimask_outputs=3,
                          transformer=TwoWayTransformer(depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8),
                          transformer_dim=256, iou_head_depth=3, iou_head_hidden_dim=256)
    return encoder, prompt_encoder, decoder


class RobustSAMAdapter(nn.Module):
    def __init__(self, img_size: int = 1024, adapter: str = "bottleneck", adapter_ratio: float = 0.25,
                 lora_rank: int = 8, n_prompt_tokens: int = 4, sam_weights: str | Path | None = DEFAULT_SAM_WEIGHTS,
                 grad_checkpointing: bool = False):
        super().__init__()
        if adapter not in ("bottleneck", "lora", "none"):
            raise ValueError(f"unknown adapter {adapter!r}")
        self.config = dict(img_size=img_size, adapter=adapter, adapter_ratio=adapter_ratio, lora_rank=lora_rank,
                           n_prompt_tokens=n_prompt_tokens)
        self.img_size = img_size
        self.grid = img_size // 16
        self.grad_checkpointing = grad_checkpointing

        self.encoder, prompt_encoder, self.decoder = build_mobile_sam_parts(img_size)
        if sam_weights is not None:
            self._load_mobile_sam(Path(sam_weights), prompt_encoder)
        self.pe_layer = prompt_encoder.pe_layer  # fixed random-Fourier positional encoding (a buffer)

        # Learned prompts, initialised from SAM's own prompt embeddings.
        with torch.no_grad():
            box = torch.tensor([[0.0, 0.0, img_size - 1.0, img_size - 1.0]])
            box_tokens, _ = prompt_encoder(points=None, boxes=box, masks=None)  # [1,2,256]
            tokens = torch.randn(1, n_prompt_tokens, 256) * 0.02
            k = min(2, n_prompt_tokens)
            tokens[:, :k] = box_tokens[:, :k]
        self.prompt_tokens = nn.Parameter(tokens)
        self.dense_embed = nn.Parameter(prompt_encoder.no_mask_embed.weight.detach().clone().view(256))

        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self._insert_adapters()
        # Decoder parts we never use (IoU head, hypernetworks of the 3 multimask tokens) stay frozen.
        for p in self.decoder.iou_prediction_head.parameters():
            p.requires_grad_(False)
        for mlp in self.decoder.output_hypernetworks_mlps[1:]:
            for p in mlp.parameters():
                p.requires_grad_(False)
        self.encoder.eval()

    def _load_mobile_sam(self, path: Path, prompt_encoder: nn.Module) -> None:
        if not path.exists():
            raise FileNotFoundError(f"MobileSAM weights not found: {path} (python scripts/download_data.py --weights mobile_sam)")
        sd = torch.load(path, map_location="cpu", weights_only=False)
        for prefix, module in (("image_encoder.", self.encoder), ("prompt_encoder.", prompt_encoder),
                               ("mask_decoder.", self.decoder)):
            part = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
            missing, unexpected = module.load_state_dict(part, strict=False)
            if missing or unexpected:
                raise RuntimeError(f"{path.name}: {prefix} mismatch: missing {missing[:3]}, unexpected {unexpected[:3]}")

    def _insert_adapters(self) -> None:
        mode = self.config["adapter"]
        for layer in self.encoder.layers[1:]:
            for blk in layer.blocks:
                if mode == "bottleneck":
                    hidden = max(8, int(round(blk.dim * self.config["adapter_ratio"])))
                    blk.mlp = MlpWithAdapter(blk.mlp, BottleneckAdapter(blk.dim, hidden))
                elif mode == "lora":
                    blk.attn.qkv = LoRALinear(blk.attn.qkv, self.config["lora_rank"])

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()  # frozen BatchNorm statistics and cached attention biases
        return self

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """SAM-normalised [B,3,S,S] -> image embedding [B,256,S/16,S/16].
        Same computation as TinyViT.forward_features, without its hard-coded 64x64 grid."""
        enc = self.encoder
        with torch.no_grad():  # stage 0 is frozen and has no adapters
            x = enc.layers[0](enc.patch_embed(x))
        use_ckpt = self.grad_checkpointing and torch.is_grad_enabled()
        for layer in enc.layers[1:]:
            for blk in layer.blocks:
                x = checkpoint(blk, x, use_reentrant=False) if use_ckpt else blk(x)
            if layer.downsample is not None:
                x = layer.downsample(x)
        b, _, c = x.shape
        x = x.view(b, self.grid, self.grid, c).permute(0, 3, 1, 2)
        return enc.neck(x)

    def decode(self, emb: torch.Tensor) -> torch.Tensor:
        """SAM's MaskDecoder.predict_masks for one prompt set per image (the original repeats every
        image embedding once per prompt, which would mix images inside a batch). Returns mask-token-0
        logits [B,1,4*grid,4*grid]."""
        dec = self.decoder
        b, c, h, w = emb.shape
        out_tokens = torch.cat([dec.iou_token.weight, dec.mask_tokens.weight], dim=0)
        tokens = torch.cat([out_tokens.unsqueeze(0).expand(b, -1, -1), self.prompt_tokens.expand(b, -1, -1)], dim=1)
        src = emb + self.dense_embed.view(1, c, 1, 1).to(emb.dtype)
        pos = self.pe_layer((h, w)).unsqueeze(0).expand(b, -1, -1, -1)
        hs, src = dec.transformer(src, pos, tokens)
        src = src.transpose(1, 2).reshape(b, c, h, w)
        up = dec.output_upscaling(src)
        hyper = dec.output_hypernetworks_mlps[0](hs[:, 1, :])  # mask token 0 -> [B, c/8]
        b, c8, h4, w4 = up.shape
        return (hyper.unsqueeze(1) @ up.view(b, c8, h4 * w4)).view(b, 1, h4, w4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: SAM-normalised [B,3,S,S]. Returns logits [B,1,S,S]."""
        logits = self.decode(self.encode(x))
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)

    # --- checkpoint helpers (only trainable parameters are saved) ---
    def trainable_names(self) -> list[str]:
        return [n for n, p in self.named_parameters() if p.requires_grad]

    def trainable_state_dict(self) -> dict:
        names = set(self.trainable_names())
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items() if k in names}

    def load_trainable_state_dict(self, sd: dict) -> None:
        expected = set(self.trainable_names())
        missing, extra = expected - set(sd), set(sd) - expected
        if missing or extra:
            raise RuntimeError(f"adapter checkpoint mismatch: missing {sorted(missing)[:3]}, unexpected {sorted(extra)[:3]}")
        self.load_state_dict(sd, strict=False)

    def count_params(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}


def find_checkpoint(run: str) -> Path | None:
    """final.pt when training finished, else the latest mid-training last.pt (with a warning)."""
    d = OURS_DIR / run
    if (d / "final.pt").exists():
        return d / "final.pt"
    if (d / "last.pt").exists():
        warnings.warn(f"{run}: training not finished, using {d / 'last.pt'}")
        return d / "last.pt"
    return None


def load_trained(path: str | Path, sam_weights: str | Path | None = DEFAULT_SAM_WEIGHTS,
                 map_location="cpu") -> tuple[RobustSAMAdapter, dict]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    cfg = {**MODEL_DEFAULTS, **{k: v for k, v in ckpt["model_config"].items() if k in MODEL_DEFAULTS}}
    net = RobustSAMAdapter(**cfg, sam_weights=sam_weights)
    net.load_trainable_state_dict(ckpt["trainable"])
    return net.eval(), ckpt


def low_res_view(img: np.ndarray, size: int) -> np.ndarray:
    """Resolution test: the image as a 352-px baseline would see it (PIL bilinear with antialiasing to
    size x size, as in normalize_imagenet), resized back to the original shape. This removes the detail
    a low-resolution model never receives, including much of the noise / JPEG / rain corruption."""
    from PIL import Image

    small = Image.fromarray(img).resize((size, size), Image.BILINEAR)
    return np.asarray(small.resize((img.shape[1], img.shape[0]), Image.BILINEAR))


class RobustAdapterCOD(CODModel):
    """CODModel wrapper. Subclasses fix the run name; `checkpoint=` overrides the file.
    `input_res` (resolution test only): first reduce the image to input_res x input_res, see low_res_view."""
    name = "Ours"
    run = None
    input_res: int | None = None
    source = "this repo (src/train.py), frozen MobileSAM f706ad9 encoder"
    default_fp16 = True

    def __init__(self, checkpoint: str | Path | None = None, sam_weights: str | Path | None = DEFAULT_SAM_WEIGHTS, **kw):
        super().__init__(**kw)
        path = Path(checkpoint) if checkpoint else (find_checkpoint(self.run) if self.run else None)
        if path is None or not path.exists():
            raise FileNotFoundError(f"no trained checkpoint for {self.name}: expected {OURS_DIR / str(self.run) / 'final.pt'} "
                                    f"(train with: python -m src.train --config configs/train/{self.run}.yaml)")
        self.net, ckpt = load_trained(path, sam_weights)
        self.net.to(self.device)
        self.checkpoint_path = path
        self.epoch = ckpt.get("epoch")
        self.finished = bool(ckpt.get("finished", False))
        if not self.finished:  # never let a half-trained model's results pass as the final one
            self.name = f"{self.name}@ep{self.epoch:.1f}-unfinished"
        self.source = f"{self.source}; {path.relative_to(REPO) if path.is_relative_to(REPO) else path} (epoch {self.epoch})"

    def _forward(self, img: np.ndarray) -> torch.Tensor:
        if self.input_res:
            img = low_res_view(img, self.input_res)
        x = torch.from_numpy(resize_image(img, self.net.img_size).transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        return self.net(normalize_batch(x)).float().sigmoid()


class OursClean(RobustAdapterCOD):
    name, run = "Ours-Clean", "ours_clean"


class OursAug(RobustAdapterCOD):
    name, run = "Ours-Aug", "ours_aug"


class OursFull(RobustAdapterCOD):
    name, run = "Ours-Full", "ours_full"


class OursFullEp15(RobustAdapterCOD):
    """First Ours-Full run (15 epochs; schedule fitted by a time budget). Kept for the epochs ablation."""
    name, run = "Ours-Full-15ep", "ours_full_ep15"


class OursFull512(RobustAdapterCOD):
    """Ours-Full trained and tested at 512 x 512 (configs/train/ours_full_512.yaml); resolution follow-up."""
    name, run = "Ours-Full-512", "ours_full_512"


# --- Resolution test (diagnostic, not new models): the same trained weights, fed a low-resolution view ---
class OursFullIn352(OursFull):
    name, input_res = "Ours-Full-in352", 352


class OursFullIn512(OursFull):
    name, input_res = "Ours-Full-in512", 512


class OursFullIn768(OursFull):
    name, input_res = "Ours-Full-in768", 768


class OursCleanIn352(OursClean):
    name, input_res = "Ours-Clean-in352", 352
