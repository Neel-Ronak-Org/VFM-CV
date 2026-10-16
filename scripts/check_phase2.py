"""Is Phase 2 done? One command that checks everything Checkpoint 2 needs and says what's left.

    python scripts/check_phase2.py

Checks, in order:
  1. data      - every dataset present with the expected number of image/mask pairs
  2. weights   - every required checkpoint present and loadable (FEDER may be clean-only instead)
  3. config    - the released SAM-Adapter config.yaml matches what our wrapper builds
  4. maps      - released prediction maps present and complete (SINet-V2, ZoomNet, FEDER)
  5. results   - a clean result for every model x dataset (benchmark.csv / released_maps.csv)
  6. reproduction - |S-alpha ours - reported| <= tolerance, and ours vs the authors' own maps
Prints [OK] / [TODO] / [FAIL] lines, then the exact commands still to run, and writes the same
report to results/phase2_status.md (commit it so Claude can read it). Exit code 0 = all done.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.datasets import CODDataset, load_data_config  # noqa: E402

DATASETS = ["CHAMELEON", "CAMO", "COD10K", "NC4K"]
# Models that must have a clean result from OUR code (benchmark.csv).
EVALUATED = ["SINet-V2", "PFNet", "ZoomNet", "SAM2-UNet", "SAM-Adapter", "Oracle-Box-MobileSAM"]
# Model -> weights key in configs/sources.yaml
WEIGHTS = {"SINet-V2": "sinet_v2", "PFNet": "pfnet", "ZoomNet": "zoomnet", "FEDER": "feder",
           "SAM2-UNet": "sam2_unet", "SAM-Adapter": "sam_adapter", "Oracle-Box-MobileSAM": "mobile_sam"}
# Released prediction maps used to check our ports (or, for FEDER without weights, as its clean result).
MAPS = {"SINet-V2": "data/released_maps/SINet-V2", "ZoomNet": "data/released_maps/ZoomNet",
        "FEDER": "data/released_maps/FEDER"}
MAPS_DELTA_WARN = 0.005  # ours vs authors' maps; bigger gaps deserve a look
# Accepted gaps (decided with the student) -> reported as INFO, never block completion.
KNOWN_GAPS = {
    ("FEDER", "COD10K"): "authors' released maps contain no COD10K folder and the weights link is dead "
                         "(decision 2026-10-16: skip; FEDER is clean-only on CHAMELEON/CAMO/NC4K)",
}
# Released-maps scoring is required only where the paper value could not be confirmed for every dataset.
MAPS_SCORING_OPTIONAL = {"SINet-V2": "paper S-alpha already matched on all 4 datasets (|Δ| <= 0.0004)"}
SLOW_MS = 2000  # clean mean latency above this makes the 40-condition corruption grid impractical


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Report:
    def __init__(self):
        self.lines: list[str] = []
        self.todo: list[str] = []
        self.failed = False
        self.incomplete = False

    def section(self, title: str):
        self.lines.append(f"\n## {title}")

    def ok(self, msg: str):
        self.lines.append(f"[OK]   {msg}")

    def info(self, msg: str):
        self.lines.append(f"[INFO] {msg}")

    def need(self, msg: str, cmd: str | None = None):
        self.incomplete = True
        self.lines.append(f"[TODO] {msg}")
        if cmd and cmd not in self.todo:
            self.todo.append(cmd)

    def fail(self, msg: str, cmd: str | None = None):
        self.failed = True
        self.lines.append(f"[FAIL] {msg}")
        if cmd and cmd not in self.todo:
            self.todo.append(cmd)


def _csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() and path.stat().st_size > 0 else None


# ---------------------------------------------------------------------------------------------
def check_data(r: Report) -> None:
    r.section("1. Data")
    cfg = load_data_config()
    specs = [("train", cfg["train"]["expected"])] + [(d, cfg["test"][d]["expected"]) for d in DATASETS]
    for name, expected in specs:
        try:
            n = len(CODDataset.from_config(name, cfg))
        except FileNotFoundError:
            r.need(f"{name}: not found", "python scripts/download_data.py --data")
            continue
        (r.ok if n == expected else r.fail)(f"{name}: {n} pairs (expected {expected})")


def _load_ok(path: Path) -> tuple[bool, str]:
    import torch

    head = path.read_bytes()[:64] if path.stat().st_size < 2**31 else open(path, "rb").read(64)
    if head.lstrip().startswith(b"<"):
        return False, "HTML page (Google Drive quota/virus-scan page), not a checkpoint"
    try:
        try:
            sd = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        except RuntimeError:  # legacy (non-zip) format can't be mmapped
            sd = torch.load(path, map_location="cpu", weights_only=False)
        n = len(sd.get("state_dict", sd)) if isinstance(sd, dict) else "?"
        return True, f"{path.stat().st_size / 2**20:.0f} MB, {n} tensors"
    except Exception as e:  # noqa: BLE001
        return False, f"cannot load ({type(e).__name__}); probably truncated"


def check_weights(r: Report, dd) -> dict[str, bool]:
    r.section("2. Weights")
    have = {}
    for model, key in WEIGHTS.items():
        entry = dd.SOURCES["weights"][key]
        files = dd._weight_files(entry)
        if model == "SAM-Adapter" and len(files) > 1:
            r.fail(f"SAM-Adapter: {len(files)} checkpoints in {entry['dest']} - keep only the COD one "
                   f"({', '.join(str(f.relative_to(REPO)) for f in files)})")
            have[model] = False
            continue
        if not files:
            have[model] = False
            if model == "FEDER":
                r.info("FEDER: no weights (official link dead) -> clean-only from released maps, checked in §4/§5")
            else:
                hint = entry.get("manual") or f"python scripts/download_data.py --weights {key}"
                r.need(f"{model}: weights missing ({entry['dest']})",
                       hint if hint.startswith("python") else f"# {model}: {hint}")
            continue
        f = files[0]
        good, why = _load_ok(f)
        have[model] = good
        if good:
            r.ok(f"{model}: {f.relative_to(REPO)} ({why})")
        else:
            r.fail(f"{model}: {f.relative_to(REPO)} is broken: {why}",
                   f"# delete {f.relative_to(REPO)} and download it again")
    return have


def check_sam_adapter_config(r: Report) -> None:
    r.section("3. SAM-Adapter released config vs our wrapper")
    cfg_path = REPO / "checkpoints" / "sam_adapter" / "config.yaml"
    if not cfg_path.exists():
        r.info("checkpoints/sam_adapter/config.yaml not present - skipped (keep it next to the checkpoint)")
        return
    from src.models.sam_adapter import VARIANTS

    cfg = yaml.safe_load(open(cfg_path))
    args = cfg.get("model", {}).get("args", {})
    enc = args.get("encoder_mode", {})
    embed = enc.get("embed_dim")
    if embed not in VARIANTS:
        r.fail(f"config embed_dim={embed} is not one of our variants {sorted(VARIANTS)}")
        return
    ours = VARIANTS[embed]
    keys = ["embed_dim", "depth", "num_heads", "global_attn_indexes", "window_size", "patch_size", "mlp_ratio",
            "qkv_bias", "use_rel_pos", "out_chans", "prompt_embed_dim"]
    diffs = [f"{k}: config={enc.get(k)!r} ours={ours.get(k)!r}" for k in keys if k in enc and enc.get(k) != ours.get(k)]
    if args.get("inp_size") not in (None, 1024):
        diffs.append(f"inp_size: config={args.get('inp_size')} ours=1024")
    # Settings that the vendored image encoder hard-codes; a different value in the config would mean a mismatch.
    hard = {"input_type": "fft", "freq_nums": 0.25, "prompt_type": "highpass", "tuning_stage": 1234,
            "handcrafted_tune": True, "embedding_tune": True, "adaptor": "adaptor", "scale_factor": 32}
    diffs += [f"{k}: config={enc[k]!r} encoder hard-codes {v!r}" for k, v in hard.items() if k in enc and enc[k] != v]
    if diffs:
        r.fail("SAM-Adapter config differs from the wrapper: " + "; ".join(diffs) + " -> tell Claude")
    else:
        r.ok(f"config matches the wrapper (embed_dim {embed}, depth {enc.get('depth')}, inp_size {args.get('inp_size', 1024)})")


def check_maps(r: Report, epm) -> None:
    r.section("4. Released prediction maps")
    cfg = load_data_config()
    for model, rel in MAPS.items():
        root = REPO / rel
        if not root.exists():
            cmd = ("python scripts/download_data.py --maps" if model != "FEDER"
                   else "# FEDER maps: browser-download the Drive folder, extract into data/released_maps/FEDER")
            r.need(f"{model}: {rel} not found", cmd)
            continue
        for d in DATASETS:
            if (model, d) in KNOWN_GAPS:
                r.info(f"{model} maps {d}: skipped - {KNOWN_GAPS[(model, d)]}")
                continue
            folder = epm.find_dataset_dir(root, d)
            try:
                n_gt = len(CODDataset.from_config(d, cfg))
            except FileNotFoundError:
                continue
            if folder is None:
                r.need(f"{model} maps: no {d} folder under {rel}")
                continue
            n = len(list(folder.rglob("*.png")))
            (r.ok if n >= n_gt else r.need)(f"{model} maps {d}: {n}/{n_gt} PNGs")


def check_results(r: Report, have: dict[str, bool]) -> None:
    r.section("5. Clean results")
    bench = _csv(REPO / "results" / "benchmark.csv")
    maps = _csv(REPO / "results" / "released_maps.csv")
    clean = bench[(bench.corruption == "clean") & (bench.severity == 0)] if bench is not None else pd.DataFrame()
    for m in EVALUATED + ["FEDER"]:
        for d in DATASETS:
            if (m, d) in KNOWN_GAPS:
                r.info(f"{m:21s} {d:9s} skipped - {KNOWN_GAPS[(m, d)]}")
                continue
            in_bench = len(clean) and ((clean.model == m) & (clean.dataset == d)).any()
            if in_bench:
                row = clean[(clean.model == m) & (clean.dataset == d)].iloc[-1]
                r.ok(f"{m:21s} {d:9s} Sm={row.Sm:.4f} ({row.precision})")
                continue
            if m == "FEDER" and not have.get("FEDER"):
                mrow = maps[(maps.model == "FEDER") & (maps.dataset == d)] if maps is not None else pd.DataFrame()
                if len(mrow):
                    miss = int(mrow.iloc[-1].get("n_missing", 0))
                    (r.ok if miss == 0 else r.need)(f"{m:21s} {d:9s} Sm={mrow.iloc[-1].Sm:.4f} (clean-only, released maps, {miss} missing)")
                else:
                    r.need(f"FEDER {d}: no clean-only result",
                           "python scripts/eval_prediction_maps.py --model FEDER --pred-root data/released_maps/FEDER")
                continue
            r.need(f"{m} {d}: no clean result",
                   f"python scripts/run_clean_baselines.py --models {m}" if have.get(m, True) else None)
    for m in ("SINet-V2", "ZoomNet"):
        if maps is None or not (maps.model == m).any():
            cmd = f"python scripts/eval_prediction_maps.py --model {m} --pred-root data/released_maps/{m}"
            if m in MAPS_SCORING_OPTIONAL:
                r.info(f"{m}: released maps not scored (optional: {MAPS_SCORING_OPTIONAL[m]}) - {cmd}")
            else:
                r.need(f"{m}: released maps not scored", cmd)
    if len(clean) and "mean_latency_ms" in clean:
        lat = clean.groupby("model").mean_latency_ms.median()
        for m, ms in lat.items():
            if ms > SLOW_MS:
                r.info(f"{m}: ~{ms / 1000:.1f} s/image on clean data - too slow for the 40-condition corruption grid; "
                       f"profile it: python -m src.profile --models {m} --runs 5")


def check_reproduction(r: Report, mr) -> None:
    r.section("6. Reproduction")
    bench = REPO / "results" / "benchmark.csv"
    if not bench.exists():
        r.need("no results yet")
        return
    df = mr.build(bench, REPO / "results" / "released_maps.csv", REPO / "configs" / "reported.yaml")
    tol = yaml.safe_load(open(REPO / "configs" / "reported.yaml"))["tolerance_Sm"]
    for _, row in df.iterrows():
        tag = f"{row.model:21s} {row.dataset:9s}"
        if row.model.startswith("Oracle"):
            continue
        if pd.notna(row.get("Sm_reported")):
            msg = f"{tag} ours {row.Sm_ours:.4f} vs paper {row.Sm_reported:.3f} (Δ {row.delta_Sm_vs_reported:+.4f})"
            if row.within_tolerance:
                r.ok(msg)
            else:
                r.fail(msg + f" - outside ±{tol}; explain or fix before Checkpoint 2")
        else:
            r.info(f"{tag} ours {row.Sm_ours:.4f}: no confirmed published value")
        if pd.notna(row.get("delta_Sm_vs_released_maps")):
            dlt = row.delta_Sm_vs_released_maps
            msg = f"{tag} ours vs authors' maps Δ {dlt:+.4f}"
            (r.ok if abs(dlt) <= MAPS_DELTA_WARN else r.info)(msg + ("" if abs(dlt) <= MAPS_DELTA_WARN else " (look into it)"))
    r.info("Oracle-Box-MobileSAM has no published value; inspect a few predictions at Checkpoint 2 "
           "(python -m src.evaluate --models Oracle-Box-MobileSAM --datasets CHAMELEON --save-pred 12 "
           "--out results/oracle_check.csv)")


def main() -> int:
    dd, epm, mr = _load_script("download_data"), _load_script("eval_prediction_maps"), _load_script("make_reproduction")
    r = Report()
    check_data(r)
    have = check_weights(r, dd)
    check_sam_adapter_config(r)
    check_maps(r, epm)
    check_results(r, have)
    check_reproduction(r, mr)
    if r.failed:
        verdict = "NOT DONE - something is wrong (see [FAIL] lines)"
    elif r.incomplete:
        verdict = "NOT DONE - work remaining (see [TODO] lines)"
    else:
        verdict = "PHASE 2 COMPLETE - ready for Checkpoint 2"
    out = ["# Phase 2 status", *r.lines, f"\n## Verdict\n{verdict}"]
    if r.todo:
        out += ["\n## Still to run", *[f"    {c}" for c in r.todo],
                "    python scripts/make_reproduction.py",
                "    python scripts/check_phase2.py"]
    text = "\n".join(out)
    print(text)
    (REPO / "results").mkdir(exist_ok=True)
    (REPO / "results" / "phase2_status.md").write_text(text + "\n", encoding="utf-8")
    return 0 if verdict.startswith("PHASE 2 COMPLETE") else 1


if __name__ == "__main__":
    sys.exit(main())
