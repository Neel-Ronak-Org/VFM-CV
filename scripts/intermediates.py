"""Intermediate outputs of our model for one image, for slides and the viva. Needs the dataset and a trained
checkpoint (run it on the laptop; a few seconds).

    python scripts/intermediates.py                                   # teaser image, motion blur 4, Ours (512)
    python scripts/intermediates.py --image COD10K-CAM-2-Terrestrial-23-Cat-1520 --corruption rain --severity 4
    python scripts/intermediates.py --model Ours-Full --corruption clean

Stages saved (figures/intermediates/<image>_<corruption><severity>/, one PNG per stage, plus panel.png):
  1 input        the test image as loaded
  2 corrupted    after the benchmark corruption (same seed as the benchmark, so identical to its input)
  3 net_input    resized to the model's square input (512 or 1024 px)
  4 embedding    frozen-encoder + adapter output, [256, S/16, S/16], first 3 principal components as RGB
  5 logits       decoder output before upsampling, [S/4, S/4], sigmoid shown
  6 probability  sigmoid map resized to the original image size (what predict() returns)
  7 overlay      probability > 0.5 drawn in red on the corrupted input
  8 ground_truth the GT mask
S-alpha and MAE of the final map against the GT are printed on the panel.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.corruptions import corrupt_for_benchmark, load_protocol  # noqa: E402
from src.datasets import CODDataset  # noqa: E402
from src.metrics import image_scores  # noqa: E402
from src.models import build_model  # noqa: E402
from src.models.robust_adapter import normalize_batch, resize_image  # noqa: E402

DEFAULT_IMAGE = "COD10K-CAM-3-Flying-61-Katydid-4063"  # the teaser image (results/qualitative_selection.csv)
TITLES = {"input": "1. Input", "corrupted": "2. Corrupted input", "net_input": "3. Network input",
          "embedding": "4. Encoder embedding (PCA)", "logits": "5. Decoder output (low-res)",
          "probability": "6. Camouflage map", "overlay": "7. Detection overlay", "ground_truth": "8. Ground truth"}


def pca_rgb(emb: np.ndarray) -> np.ndarray:
    """[C,H,W] -> [H,W,3] uint8 from the first 3 principal components (each stretched to 0-255)."""
    c, h, w = emb.shape
    x = emb.reshape(c, -1).T
    x = x - x.mean(0)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    p = x @ vt[:3].T
    p = (p - p.min(0)) / (np.ptp(p, 0) + 1e-8)
    return (p.reshape(h, w, 3) * 255).astype(np.uint8)


def gray(prob: np.ndarray) -> np.ndarray:
    return np.repeat((np.clip(prob, 0, 1) * 255).astype(np.uint8)[..., None], 3, axis=2)


def overlay(img: np.ndarray, prob: np.ndarray, color=(255, 40, 40), alpha: float = 0.55) -> np.ndarray:
    out = img.astype(np.float32).copy()
    m = prob > 0.5
    out[m] = (1 - alpha) * out[m] + alpha * np.array(color, np.float32)
    edges = cv2.Canny(m.astype(np.uint8) * 255, 50, 150) > 0
    out[cv2.dilate(edges.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0] = color
    return out.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    ap.add_argument("--dataset", default="COD10K")
    ap.add_argument("--corruption", default="motion_blur")
    ap.add_argument("--severity", type=int, default=4)
    ap.add_argument("--model", default="Ours-Full-512")
    args = ap.parse_args()
    if args.corruption == "clean":
        args.severity = 0

    ds = CODDataset.from_config(args.dataset)
    idx = next((i for i, s in enumerate(ds.samples) if s.name == args.image), None)
    if idx is None:
        raise SystemExit(f"{args.image} not found in {args.dataset}")
    name, img, gt = ds[idx]
    corrupted = corrupt_for_benchmark(img, args.dataset, name, args.corruption, args.severity,
                                      load_protocol()["base_seed"])

    model = build_model(args.model)
    net, dev = model.net.eval(), model.device
    net_in = resize_image(corrupted, net.img_size)
    x = normalize_batch(torch.from_numpy(net_in.transpose(2, 0, 1)).unsqueeze(0).to(dev))
    with torch.inference_mode(), torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
        emb = net.encode(x)
        logits = net.decode(emb)
    emb, logits = emb.float(), logits.float()
    low = logits.sigmoid()[0, 0].cpu().numpy()
    prob = F.interpolate(F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False),
                         size=img.shape[:2], mode="bilinear", align_corners=False).sigmoid()[0, 0].cpu().numpy()
    sc = image_scores(prob, gt)

    stages = {"input": img, "corrupted": corrupted, "net_input": net_in,
              "embedding": cv2.resize(pca_rgb(emb[0].cpu().numpy()), (net.img_size // 4,) * 2, interpolation=cv2.INTER_NEAREST),
              "logits": gray(low), "probability": gray(prob), "overlay": overlay(corrupted, prob),
              "ground_truth": gray(gt.astype(np.float32) / max(gt.max(), 1))}
    shapes = {"net_input": f"{net.img_size}x{net.img_size}x3", "embedding": f"256x{emb.shape[-1]}x{emb.shape[-1]}",
              "logits": f"{low.shape[0]}x{low.shape[1]}"}
    out = REPO / "figures" / "intermediates" / f"{name}_{args.corruption}{args.severity}"
    out.mkdir(parents=True, exist_ok=True)
    for k, v in stages.items():
        cv2.imwrite(str(out / f"{k}.png"), cv2.cvtColor(v, cv2.COLOR_RGB2BGR))

    fig, axes = plt.subplots(2, 4, figsize=(12, 6.4))
    for ax, (k, v) in zip(axes.flat, stages.items()):
        ax.imshow(v)
        ax.set_title(TITLES[k] + (f"\n{shapes[k]}" if k in shapes else f"\n{img.shape[0]}x{img.shape[1]}"), fontsize=10)
        ax.axis("off")
    fig.suptitle(f"{model.name} on {name}, {args.corruption} severity {args.severity}: "
                 f"S-alpha {sc['Sm']:.3f}, MAE {sc['MAE']:.3f}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "panel.png", dpi=200)
    print(f"wrote {out.relative_to(REPO)}/ (8 stages + panel.png); S-alpha {sc['Sm']:.3f}, MAE {sc['MAE']:.3f}")


if __name__ == "__main__":
    main()
