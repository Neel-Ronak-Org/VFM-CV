"""Training losses for our adapter.

structure_loss: weighted BCE + weighted IoU from PraNet (Fan et al., MICCAI 2020), also used by
SINet-V2. Pixels near object boundaries get up to 6x weight. PraNet uses a 31x31 window at 352x352
input; we scale the window with the training resolution (31 * S / 352, odd) so the weighted band
covers the same fraction of the image, and compute the box filter separably (exactly equal to
F.avg_pool2d(k, stride 1, padding k//2), but O(k) instead of O(k^2) per pixel).

consistency_loss: distance between the sigmoid maps of a clean and a corrupted view of the same
image. "mse" = mean squared difference of probabilities; "sym_kl" = symmetric per-pixel Bernoulli KL.
stopgrad_clean=True treats the clean prediction as a fixed target (gradients only through the
corrupted view), so the model learns to make corrupted predictions look like clean ones rather than
the reverse.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def boundary_window(size: int) -> int:
    k = int(round(31 * size / 352))
    return k if k % 2 else k + 1


def box_filter(x: torch.Tensor, k: int) -> torch.Tensor:
    """avg_pool2d(x, k, stride=1, padding=k//2) with zero padding counted, computed separably."""
    x = F.avg_pool2d(x, (1, k), stride=1, padding=(0, k // 2))
    return F.avg_pool2d(x, (k, 1), stride=1, padding=(k // 2, 0))


def structure_loss(logits: torch.Tensor, mask: torch.Tensor, k: int | None = None) -> torch.Tensor:
    """logits, mask: [B,1,H,W]; mask in [0,1]. Returns the batch mean of wBCE + wIoU."""
    logits, mask = logits.float(), mask.float()
    k = k or boundary_window(mask.shape[-1])
    weit = 1 + 5 * torch.abs(box_filter(mask, k) - mask)
    wbce = F.binary_cross_entropy_with_logits(logits, mask, reduction="none")
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(logits)
    inter = (pred * mask * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def consistency_loss(logits_clean: torch.Tensor, logits_corr: torch.Tensor, kind: str = "mse",
                     stopgrad_clean: bool = True) -> torch.Tensor:
    lc, lr = logits_clean.float(), logits_corr.float()
    if stopgrad_clean:
        lc = lc.detach()
    if kind == "mse":
        return ((torch.sigmoid(lr) - torch.sigmoid(lc)) ** 2).mean()
    if kind == "sym_kl":
        # log p and log(1-p) from logits, numerically stable
        lp_c, lq_c = F.logsigmoid(lc), F.logsigmoid(-lc)
        lp_r, lq_r = F.logsigmoid(lr), F.logsigmoid(-lr)
        p_c, p_r = lp_c.exp(), lp_r.exp()
        kl_cr = p_c * (lp_c - lp_r) + (1 - p_c) * (lq_c - lq_r)
        kl_rc = p_r * (lp_r - lp_c) + (1 - p_r) * (lq_r - lq_c)
        return 0.5 * (kl_cr + kl_rc).mean()
    raise ValueError(f"unknown consistency loss {kind!r}")
