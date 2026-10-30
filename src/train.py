"""Train our Robust Lightweight SAM Adapter.

    python -m src.train --config configs/train/ours_full.yaml --profile 40     # speed + VRAM check only
    python -m src.train --config configs/train/ours_full.yaml                  # train
    python -m src.train --config configs/train/ours_full.yaml --resume         # continue after a stop/crash
    python -m src.train --config configs/train/ours_full.yaml --max-hours 8    # stop cleanly after ~8 h (resume later)
    python -m src.train --config ... --set train.batch_size=1 train.grad_accum=8 data.img_size=512   # VRAM ladder

Variants (configs/train/): ours_clean (no corruption), ours_aug (+50% seen-corruption augmentation),
ours_full (+ clean/corrupted consistency loss), plus ablations. Only the SEEN corruptions of the held-out
protocol (configs/corruptions.yaml) may be used for training; the script refuses unseen ones unless
augment.allow_unseen is set.

Outputs:
  checkpoints/ours/<run>/last.pt    resumable state (trainable params, optimizer, scaler, position),
                                    saved every epoch, every train.save_every_minutes and on stop/Ctrl+C
  checkpoints/ours/<run>/final.pt   trainable params + config, written when training finishes
  results/train/<run>/config.yaml, log.csv (loss curve), summary.json (git, seed, timings, VRAM)
  results/train/profile.csv         one row per --profile run (speed/VRAM ladder record)
Epoch-level determinism: the sample order and every augmentation are seeded by (seed, epoch, index),
so resuming mid-epoch continues the exact same sequence of batches.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import time
from pathlib import Path

import cv2
import torch
import yaml
from torch.utils.data import DataLoader

from src.corruptions import load_protocol
from src.datasets import CODDataset, CODTrainSet
from src.losses import consistency_loss, structure_loss
from src.models.base import pick_device
from src.models.robust_adapter import OURS_DIR, REPO, RobustSAMAdapter, normalize_batch
from src.utils import git_hash, seed_everything

LOG_ROOT = REPO / "results" / "train"
LOG_FIELDS = ["epoch", "step", "batch", "lr", "loss", "sup", "cons", "sec_per_batch", "peak_vram_mb", "elapsed_h", "eta_h"]
# Keys that may differ between a run and its resume without changing the experiment.
RESUME_SAFE_KEYS = {"data.num_workers", "train.time_budget_hours", "train.log_every", "train.save_every_minutes"}


# ---------------------------------------------------------------------------------------- config
def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path, overrides: list[str] | None = None, _top: bool = True) -> dict:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    if "base" in cfg:
        cfg = deep_merge(load_config(path.parent / cfg.pop("base"), _top=False), cfg)
    if not _top:
        return cfg
    for o in overrides or []:
        key, _, value = o.partition("=")
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        v = yaml.safe_load(value)
        if isinstance(v, str):  # YAML reads "1e-3" as text; accept it as a number
            try:
                v = float(v)
            except ValueError:
                pass
        node[leaf] = v
    if not cfg.get("run_name"):
        cfg["run_name"] = path.stem
    return cfg


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        out.update(flatten(v, f"{prefix}{k}.") if isinstance(v, dict) else {f"{prefix}{k}": v})
    return out


def resolve_corruptions(aug: dict, protocol: dict) -> list[str]:
    spec = aug.get("corruptions", "seen")
    corrs = protocol["seen"] if spec == "seen" else list(spec)
    unknown = set(corrs) - set(protocol["corruptions"])
    if unknown:
        raise ValueError(f"unknown corruptions {sorted(unknown)}")
    leaked = set(corrs) & set(protocol["unseen"])
    if leaked and not aug.get("allow_unseen", False):
        raise ValueError(f"{sorted(leaked)} are UNSEEN (held-out) corruptions; training on them breaks the protocol")
    return corrs


def build_trainset(cfg: dict, protocol: dict, data_root: str | Path | None = None) -> CODTrainSet:
    d, aug, cons = cfg["data"], cfg["augment"], cfg["consistency"]
    paired = cons["weight"] > 0
    use_corr = paired or aug["corruption_prob"] > 0
    base = CODDataset.from_config(d.get("split", "train"), root=data_root)
    return CODTrainSet(base, d["img_size"], hflip=d["hflip"], crop_scale=d["crop_scale"],
                       corruption_prob=aug["corruption_prob"],
                       corruptions=resolve_corruptions(aug, protocol) if use_corr else [],
                       severities=aug["severities"], paired=paired, seed=cfg["seed"])


def build_model(cfg: dict) -> RobustSAMAdapter:
    m = cfg["model"]
    weights = m.get("sam_weights")
    if weights and not Path(weights).is_absolute():
        weights = REPO / weights
    return RobustSAMAdapter(img_size=cfg["data"]["img_size"], adapter=m["adapter"], adapter_ratio=m["adapter_ratio"],
                            lora_rank=m["lora_rank"], n_prompt_tokens=m["n_prompt_tokens"], sam_weights=weights,
                            grad_checkpointing=cfg["train"]["grad_checkpointing"])


def lr_factor(step: int, total: int, warmup: int, min_ratio: float) -> float:
    """Linear warm-up, then cosine decay to min_ratio at `total` optimizer steps."""
    if step < warmup:
        return (step + 1) / warmup
    t = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * t))


def _worker_init(_):
    cv2.setNumThreads(1)  # avoid CPU oversubscription with several loader processes


# ---------------------------------------------------------------------------------------- training
def train(cfg: dict, *, resume: bool = False, profile: int = 0, max_hours: float | None = None,
          device: str | None = None, data_root: str | Path | None = None,
          ckpt_root: Path = OURS_DIR, log_root: Path = LOG_ROOT) -> dict:
    seed_everything(cfg["seed"])
    dev = pick_device(device)
    run, tc = cfg["run_name"], cfg["train"]
    ckpt_dir, log_dir = Path(ckpt_root) / run, Path(log_root) / run
    protocol = load_protocol()
    ds = build_trainset(cfg, protocol, data_root)
    net = build_model(cfg).to(dev).train()
    params = net.count_params()
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.cuda.reset_peak_memory_stats()

    decay, no_decay = [], []
    for n, p in net.named_parameters():
        if p.requires_grad:
            (no_decay if p.ndim <= 1 or n in ("prompt_tokens", "dense_embed") or "token" in n else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": tc["weight_decay"]},
                             {"params": no_decay, "weight_decay": 0.0}], lr=tc["lr"])
    amp = tc["amp"] if dev.type == "cuda" else "none"
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(amp)
    scaler = torch.amp.GradScaler("cuda", enabled=amp == "fp16")

    bs, accum = tc["batch_size"], tc["grad_accum"]
    batches_per_epoch = len(ds) // bs
    if batches_per_epoch == 0:
        raise RuntimeError(f"dataset has {len(ds)} images, fewer than batch_size {bs}")
    steps_per_epoch = math.ceil(batches_per_epoch / accum)
    paired = ds.paired
    lam, ckind, stopgrad = cfg["consistency"]["weight"], cfg["consistency"]["type"], cfg["consistency"]["stopgrad_clean"]

    state = {"epoch": 0, "batch": 0, "step": 0, "epochs": tc["epochs"], "epochs_fixed": False, "elapsed": 0.0}
    last = ckpt_dir / "last.pt"
    if resume and not profile:
        if not last.exists():
            raise FileNotFoundError(f"--resume: no {last}")
        ck = torch.load(last, map_location="cpu", weights_only=False)
        if ck["state"]["epoch"] >= ck["state"]["epochs"] and (ckpt_dir / "final.pt").exists():
            print(f"{run} already finished ({ck['state']['epochs']} epochs, {ck['state']['elapsed'] / 3600:.2f} h): "
                  f"{ckpt_dir / 'final.pt'} is the trained model. Nothing to resume.")
            return {"run": run, "status": "already_finished", "epochs_done": ck["state"]["epoch"]}
        diff = {k for k, v in flatten(cfg).items() if flatten(ck["train_config"]).get(k) != v} - RESUME_SAFE_KEYS
        if diff:
            raise RuntimeError(f"config differs from the run being resumed: {sorted(diff)}")
        net.load_trainable_state_dict(ck["trainable"])
        opt.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        state.update(ck["state"])
        print(f"resumed {run} at epoch {state['epoch']} batch {state['batch']} (step {state['step']})")
    if not profile:
        log_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    def save(path: Path, finished: bool = False) -> None:
        payload = {"trainable": net.trainable_state_dict(), "model_config": net.config, "train_config": cfg,
                   "epoch": state["epoch"] + state["batch"] / batches_per_epoch, "finished": finished,
                   "git": git_hash(), "params": params}
        if not finished:
            payload.update(optimizer=opt.state_dict(), scaler=scaler.state_dict(), state=dict(state))
        tmp = path.with_suffix(".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)

    def peak_mb() -> float | None:
        return round(torch.cuda.max_memory_allocated() / 2**20, 1) if dev.type == "cuda" else None

    def sync():
        if dev.type == "cuda":
            torch.cuda.synchronize()

    print(f"{run}: {len(ds)} images, img {cfg['data']['img_size']}, batch {bs} x accum {accum}, "
          f"{'paired clean+corrupted views' if paired else 'single view'}, device {dev}, amp {amp}, "
          f"trainable {params['trainable'] / 1e6:.2f}M / total {params['total'] / 1e6:.2f}M")
    t_start, t_last_save = time.time(), time.time()
    elapsed0 = state["elapsed"]
    timed, timed_n, n_batches = 0.0, 0, 0
    acc = {"loss": 0.0, "sup": 0.0, "cons": 0.0, "n": 0}
    bad_steps, status = 0, "finished"
    log_f = None
    if not profile:
        new = not (log_dir / "log.csv").exists() or not resume
        log_f = open(log_dir / "log.csv", "w" if new else "a", newline="")
        writer = csv.DictWriter(log_f, fieldnames=LOG_FIELDS)
        if new:
            writer.writeheader()
    opt.zero_grad(set_to_none=True)
    try:
        while state["epoch"] < state["epochs"]:
            epoch = state["epoch"]
            ds.set_epoch(epoch)
            g = torch.Generator().manual_seed(cfg["seed"] * 100003 + epoch)
            order = torch.randperm(len(ds), generator=g)[: batches_per_epoch * bs].tolist()
            start = state["batch"]
            t_epoch = time.time()
            loader = DataLoader(ds, batch_size=bs, sampler=order[start * bs:], num_workers=cfg["data"]["num_workers"],
                                pin_memory=dev.type == "cuda", worker_init_fn=_worker_init)
            sync()
            t_batch = time.perf_counter()
            for b, batch in enumerate(loader, start=start):
                x = batch["image"].to(dev, non_blocking=True)
                m = batch["mask"].to(dev, non_blocking=True)
                with torch.autocast(dev.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
                    if paired:
                        xc = batch["image_corr"].to(dev, non_blocking=True)
                        lc, lr = net(normalize_batch(torch.cat([x, xc]))).chunk(2)
                    else:
                        logits = net(normalize_batch(x))
                if paired:
                    sup = 0.5 * (structure_loss(lc, m) + structure_loss(lr, m))
                    cons = consistency_loss(lc, lr, ckind, stopgrad)
                    loss = sup + lam * cons
                else:
                    sup, cons = structure_loss(logits, m), torch.zeros((), device=dev)
                    loss = sup
                if not torch.isfinite(loss):
                    bad_steps += 1
                    print(f"warning: non-finite loss at epoch {epoch} batch {b}; skipping")
                    if bad_steps > 20:
                        raise RuntimeError("loss diverged (20 non-finite batches in a row)")
                    opt.zero_grad(set_to_none=True)
                    continue
                bad_steps = 0
                scaler.scale(loss / accum).backward()
                acc["loss"] += loss.item(); acc["sup"] += sup.item(); acc["cons"] += cons.item(); acc["n"] += 1  # noqa: E702
                boundary = (b + 1) % accum == 0 or b + 1 == batches_per_epoch
                if boundary:
                    lr_now = tc["lr"] * lr_factor(state["step"], state["epochs"] * steps_per_epoch,
                                                  tc["warmup_steps"], tc["min_lr"] / tc["lr"])
                    for grp in opt.param_groups:
                        grp["lr"] = lr_now
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_([p for grp in opt.param_groups for p in grp["params"]], tc["grad_clip"])
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                    state["step"] += 1
                state["batch"] = b + 1

                sync()
                now = time.perf_counter()
                n_batches += 1
                if n_batches > 5:  # skip start-up (workers spawning, cuDNN autotune)
                    timed += now - t_batch
                    timed_n += 1
                t_batch = now
                state["elapsed"] = elapsed0 + time.time() - t_start
                spb = timed / timed_n if timed_n else float("nan")

                if profile and n_batches >= profile:
                    return _report_profile(cfg, spb, batches_per_epoch, peak_mb(), dev, params, log_root)
                if log_f and boundary and state["step"] % tc["log_every"] == 0 and acc["n"]:
                    left = (state["epochs"] - epoch) * batches_per_epoch - state["batch"]
                    row = {"epoch": epoch, "step": state["step"], "batch": state["batch"], "lr": f"{lr_now:.3e}",
                           "loss": round(acc["loss"] / acc["n"], 5), "sup": round(acc["sup"] / acc["n"], 5),
                           "cons": round(acc["cons"] / acc["n"], 6), "sec_per_batch": round(spb, 4),
                           "peak_vram_mb": peak_mb() or "", "elapsed_h": round(state["elapsed"] / 3600, 3),
                           "eta_h": round(left * spb / 3600, 2) if timed_n else ""}
                    writer.writerow(row)
                    log_f.flush()
                    print(" ".join(f"{k}={v}" for k, v in row.items()))
                    acc = {"loss": 0.0, "sup": 0.0, "cons": 0.0, "n": 0}
                if boundary and not profile:
                    if max_hours and time.time() - t_start > max_hours * 3600:
                        status = "stopped"
                        break
                    if time.time() - t_last_save > tc["save_every_minutes"] * 60:
                        save(last)
                        t_last_save = time.time()
            if status == "stopped":
                break
            state["epoch"], state["batch"] = epoch + 1, 0
            if not state["epochs_fixed"] and tc.get("time_budget_hours") and not profile:
                # Fit the run to the budget using the real wall time of this epoch (loader start-up included);
                # timing only the first batches under-estimated speed by ~30% on the laptop.
                epoch_h = (time.time() - t_epoch) / 3600 * batches_per_epoch / (batches_per_epoch - start)
                fit = epoch + 1 + int((tc["time_budget_hours"] - state["elapsed"] / 3600) / epoch_h)
                state["epochs"] = max(epoch + 1, min(tc["epochs"], fit))
                state["epochs_fixed"] = True
                print(f"time budget {tc['time_budget_hours']} h, {epoch_h * 60:.1f} min/epoch -> {state['epochs']} epochs")
            if not profile:
                save(last)
                t_last_save = time.time()
                print(f"epoch {epoch + 1}/{state['epochs']} done, {state['elapsed'] / 3600:.2f} h")
    except KeyboardInterrupt:
        status = "interrupted"
        # The last partial accumulation group is dropped; resume restarts from the last optimizer step.
        state["batch"] -= state["batch"] % accum if state["batch"] < batches_per_epoch else 0
    finally:
        if log_f:
            log_f.close()
    if profile:  # fewer batches than requested (tiny dataset)
        return _report_profile(cfg, timed / max(timed_n, 1), batches_per_epoch, peak_mb(), dev, params, log_root)

    if status == "finished":
        save(ckpt_dir / "final.pt", finished=True)
    else:
        save(last)
        print(f"{status}: saved {last}; continue with --resume")
    summary = {"run": run, "status": status, "git": git_hash(), "seed": cfg["seed"], "device": str(dev),
               "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else "",
               "epochs_done": state["epoch"], "epochs_planned": state["epochs"], "optimizer_steps": state["step"],
               "hours": round(state["elapsed"] / 3600, 3), "sec_per_batch": round(timed / max(timed_n, 1), 4),
               "peak_vram_mb": peak_mb(), "params": params, "n_train_images": len(ds),
               "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    (log_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def _report_profile(cfg, spb, batches_per_epoch, peak, dev, params, log_root) -> dict:
    epoch_h = spb * batches_per_epoch / 3600
    row = {"run": cfg["run_name"], "img_size": cfg["data"]["img_size"], "batch_size": cfg["train"]["batch_size"],
           "grad_accum": cfg["train"]["grad_accum"], "adapter": cfg["model"]["adapter"],
           "paired": cfg["consistency"]["weight"] > 0, "grad_checkpointing": cfg["train"]["grad_checkpointing"],
           "amp": cfg["train"]["amp"], "num_workers": cfg["data"]["num_workers"], "device": str(dev),
           "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else "",
           "sec_per_batch": round(spb, 4), "peak_vram_mb": peak or "", "epoch_minutes": round(epoch_h * 60, 1),
           "hours_for_config_epochs": round(epoch_h * cfg["train"]["epochs"], 2),
           "trainable_params": params["trainable"], "git": git_hash(), "date": time.strftime("%Y-%m-%d %H:%M")}
    path = Path(log_root) / "profile.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    print(json.dumps(row, indent=2))
    if dev.type == "cuda" and peak:
        total = torch.cuda.get_device_properties(0).total_memory / 2**20
        if peak > 0.9 * total:
            print(f"WARNING: peak {peak:.0f} MB is >90% of {total:.0f} MB; on Windows this can spill into shared "
                  "memory and run many times slower. Step down the VRAM ladder (see docs/MODEL.md).")
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config entries")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--profile", type=int, default=0, metavar="N", help="time N batches, report speed/VRAM, save nothing")
    ap.add_argument("--max-hours", type=float, default=None, help="stop cleanly after this long (resume later)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--data-root", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    train(cfg, resume=args.resume, profile=args.profile, max_hours=args.max_hours, device=args.device,
          data_root=args.data_root)


if __name__ == "__main__":
    main()
