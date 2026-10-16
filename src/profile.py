"""Efficiency profile of any registered model: parameters, GFLOPs, latency/FPS, peak VRAM.

    python -m src.profile --models SINet-V2 SAM-Adapter          # writes results/efficiency.csv
    python -m src.profile --models SAM-Adapter --runs 5          # quick check

Protocol (Phase 5 / CLAUDE.md): batch size 1, each model's default precision (fp16 for the large
SAM-based models), inference mode, on the laptop GPU. Latency is END-TO-END `model.predict()` on a
fixed 1024x768 RGB image (preprocessing + network + resize back), after warm-up runs, with CUDA
synchronisation; FPS = 1000 / median latency. GFLOPs are counted once with torch's FlopCounterMode
(FLOPs = 2 x multiply-adds; GMACs is also reported). Peak VRAM is torch.cuda.max_memory_allocated
during the timed runs. If the peak exceeds the card's memory, Windows may be paging GPU memory to
system RAM, which makes everything very slow; the row is flagged `oversubscribed`.
"""
from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from src.models import build_model
from src.utils import REPO, git_hash

FIELDS = ["model", "params_M", "gflops", "gmacs", "latency_ms_median", "latency_ms_p90", "fps", "peak_vram_MB",
          "reserved_vram_MB", "gpu_total_MB", "oversubscribed", "precision", "device", "gpu_name", "image_hw",
          "runs", "git", "timestamp"]


def count_flops(model, img: np.ndarray) -> float | None:
    try:
        from torch.utils.flop_counter import FlopCounterMode

        with torch.inference_mode(), FlopCounterMode(display=False) as fc:
            with torch.autocast(device_type=model.device.type, dtype=torch.float16, enabled=model.fp16):
                model._forward(img)
        return float(fc.get_total_flops())
    except Exception as e:  # noqa: BLE001 -- FLOPs are nice-to-have; never fail the profile for them
        print(f"  (FLOP counting failed: {type(e).__name__}: {e})")
        return None


def n_params(model) -> float:
    mods = [getattr(model, a) for a in ("net", "sam") if hasattr(model, a)]
    return sum(p.numel() for m in mods for p in m.parameters()) / 1e6


def profile(name: str, runs: int, warmup: int, hw: tuple[int, int], device: str | None) -> dict:
    model = build_model(name, device=device)
    img = np.random.default_rng(0).integers(0, 256, (*hw, 3), dtype=np.uint8)
    if model.needs_gt:
        gt = np.zeros(hw, bool)
        gt[hw[0] // 4: 3 * hw[0] // 4, hw[1] // 4: 3 * hw[1] // 4] = True
    cuda = model.device.type == "cuda"
    flops = count_flops(model, img) if not model.needs_gt else None

    def once():
        if model.needs_gt:
            model.set_gt(gt)
        model.predict(img)
        return model.last_latency_ms

    for _ in range(warmup):
        once()
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    lat = [once() for _ in range(runs)]
    mb = (lambda b: round(b / 2**20)) if cuda else (lambda b: "")
    peak = mb(torch.cuda.max_memory_allocated()) if cuda else ""
    reserved_b = torch.cuda.max_memory_reserved() if cuda else 0
    total_b = torch.cuda.get_device_properties(0).total_memory if cuda else 0
    med = statistics.median(lat)
    return {
        "model": name, "params_M": round(n_params(model), 2),
        "gflops": round(flops / 1e9, 1) if flops else "", "gmacs": round(flops / 2e9, 1) if flops else "",
        "latency_ms_median": round(med, 1), "latency_ms_p90": round(float(np.percentile(lat, 90)), 1),
        "fps": round(1000.0 / med, 2), "peak_vram_MB": peak, "reserved_vram_MB": mb(reserved_b),
        "gpu_total_MB": mb(total_b), "oversubscribed": bool(cuda and reserved_b > 0.97 * total_b),
        "precision": "fp16" if model.fp16 else "fp32", "device": str(model.device),
        "gpu_name": torch.cuda.get_device_name(0) if cuda else "cpu", "image_hw": f"{hw[0]}x{hw[1]}",
        "runs": runs, "git": git_hash(), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def write_rows(path: Path, rows: list[dict]) -> None:
    """Replace each profiled model's previous row; keep the others."""
    old = list(csv.DictReader(open(path))) if path.exists() else []
    keep = [r for r in old if r["model"] not in {n["model"] for n in rows}]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(keep + rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--hw", nargs=2, type=int, default=[768, 1024], help="test image height width")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results/efficiency.csv")
    args = ap.parse_args(argv)
    out = Path(args.out) if Path(args.out).is_absolute() else REPO / args.out
    rows = []
    for name in args.models:
        print(f"profiling {name} ...")
        r = profile(name, args.runs, args.warmup, tuple(args.hw), args.device)
        rows.append(r)
        flag = "  <-- exceeds GPU memory: paging to system RAM" if r["oversubscribed"] else ""
        print(f"  {r['params_M']}M params, {r['gflops']} GFLOPs, {r['latency_ms_median']} ms ({r['fps']} FPS), "
              f"peak {r['peak_vram_MB']} MB / reserved {r['reserved_vram_MB']} MB of {r['gpu_total_MB']} MB{flag}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_rows(out, rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
