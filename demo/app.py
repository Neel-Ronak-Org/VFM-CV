"""Gradio demo: camouflaged object detection under image corruptions.

    python demo/app.py                       # SINet-V2 vs our best trained adapter, GPU if available
    python demo/app.py --models SINet-V2 SAM2-UNet Ours-Full
    python demo/app.py --device cpu          # no GPU (slower)
    python demo/app.py --share               # temporary public link (e.g. to show on another device)

Tab "Detector": pick one of 12 test images (or upload / drop photos in demo/my_photos/), choose a
corruption and severity (0 = clean), and compare the models side by side as coloured overlays with each
model's latency. For test images, "Show ground truth" adds the GT and per-image S-measure / MAE.
Tab "Benchmark results": tables read from results/*.csv when the app starts (or on Refresh); nothing typed in.

Test images are read from your local data/raw (datasets are never bundled or redistributed). To choose
the 12 gallery images yourself, list them in demo/examples.txt, one "DATASET/image_name" per line.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.corruptions import apply_corruption, load_protocol  # noqa: E402
from src.datasets import IMG_EXTS, CODDataset, read_mask, read_rgb  # noqa: E402
from src.metrics import image_scores  # noqa: E402
from src.models import build_model  # noqa: E402
from src.models.base import pick_device  # noqa: E402

DEMO = REPO / "demo"
MY_PHOTOS = DEMO / "my_photos"
EXAMPLES_FILE = DEMO / "examples.txt"
GALLERY_PICK = {"COD10K": 6, "CAMO": 3, "CHAMELEON": 3}  # 12 images, chosen with a fixed seed
MAX_SIDE = 1600  # larger uploads are downscaled (phone photos) to keep corruption + inference fast
COLORS = [(255, 64, 64), (64, 200, 255), (80, 255, 80), (255, 200, 0), (220, 80, 255)]
GT_COLOR = (255, 255, 255)
OURS_PREFERENCE = ["Ours-Full", "Ours-Full-15ep", "Ours-Aug", "Ours-Clean"]


# ---------------------------------------------------------------------------------------- images
def pick_examples(per_dataset: dict[str, int] = GALLERY_PICK, seed: int = 0) -> list[dict]:
    """[{label, image, gt}] from local test sets: demo/examples.txt if present, else a seeded random pick."""
    out = []
    if EXAMPLES_FILE.exists():
        wanted = [ln.strip() for ln in EXAMPLES_FILE.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
        by_ds: dict[str, CODDataset] = {}
        for item in wanted:
            ds_name, _, stem = item.partition("/")
            try:
                ds = by_ds.setdefault(ds_name, CODDataset.from_config(ds_name))
            except (FileNotFoundError, KeyError):
                continue
            s = next((s for s in ds.samples if s.name == Path(stem).stem), None)
            if s:
                out.append({"label": f"{ds_name}/{s.name}", "image": s.image_path, "gt": s.mask_path})
        return out
    rng = np.random.default_rng(seed)
    for ds_name, n in per_dataset.items():
        try:
            ds = CODDataset.from_config(ds_name)
        except (FileNotFoundError, KeyError):
            continue
        for i in sorted(rng.choice(len(ds), size=min(n, len(ds)), replace=False)):
            s = ds.samples[i]
            out.append({"label": f"{ds_name}/{s.name}", "image": s.image_path, "gt": s.mask_path})
    return out


def my_photos() -> list[dict]:
    if not MY_PHOTOS.is_dir():
        return []
    return [{"label": f"my_photos/{p.name}", "image": p, "gt": None}
            for p in sorted(MY_PHOTOS.iterdir()) if p.suffix.lower() in IMG_EXTS]


def thumbnail(path: Path, side: int = 240) -> np.ndarray:
    img = read_rgb(path)
    s = side / max(img.shape[:2])
    return cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))), interpolation=cv2.INTER_AREA)


def cap_size(img: np.ndarray, gt: np.ndarray | None):
    s = MAX_SIDE / max(img.shape[:2])
    if s >= 1:
        return img, gt
    size = (int(img.shape[1] * s), int(img.shape[0] * s))
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    if gt is not None:
        gt = cv2.resize(gt.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    return img, gt


def overlay(img: np.ndarray, prob: np.ndarray, color) -> np.ndarray:
    """Tint the image by the predicted probability and outline the 0.5 contour."""
    a = 0.55 * prob[..., None]
    out = (img.astype(np.float32) * (1 - a) + np.array(color, np.float32) * a).astype(np.uint8)
    contours, _ = cv2.findContours((prob >= 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, max(2, img.shape[0] // 300))
    return out


# ---------------------------------------------------------------------------------------- models
def pick_ours(dev: str):
    """The first FINISHED variant in OURS_PREFERENCE (so a run still training never replaces a finished
    model); if none is finished, the first one that loads (labelled "...-unfinished")."""
    import warnings

    fallback = None
    for name in OURS_PREFERENCE:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = build_model(name, device=dev)
        except FileNotFoundError:
            continue
        if m.finished:
            return m
        fallback = fallback or m
    return fallback


def load_models(names: list[str] | None, device: str | None) -> dict:
    """name -> model or error string. Without --models: SINet-V2 + the best trained variant of ours."""
    dev = str(pick_device(device))
    models: dict = {}
    if names is None:
        names = ["SINet-V2"]
        pick = pick_ours(dev)
        if pick:
            models[pick.name] = pick
        else:
            models["Ours-Full"] = "not trained yet: python -m src.train --config configs/train/ours_full.yaml"
    for n in names:
        try:
            models[n] = build_model(n, device=dev)
        except Exception as e:  # show the reason in the UI instead of crashing the demo
            models[n] = f"unavailable: {type(e).__name__}: {e}"
    # keep SINet-V2 first, then ours, as the brief's layout
    order = [n for n in names if n in models] + [n for n in models if n not in names]
    return {n: models[n] for n in dict.fromkeys(order)}


def warmup(models: dict) -> None:
    dummy = np.full((480, 640, 3), 128, np.uint8)
    for m in models.values():
        if not isinstance(m, str) and not m.needs_gt:
            for _ in range(2):
                m.predict(dummy)


def corruption_choices(protocol: dict) -> dict[str, str]:
    """dropdown label -> corruption name"""
    out = {"none (clean)": "clean"}
    for c in protocol["corruptions"]:
        tag = "seen in training" if c in protocol["seen"] else "UNSEEN in training"
        out[f"{c} ({tag})"] = c
    return out


def detect(img, corruption: str, severity: int, gt, models: dict, show_gt: bool = False, seed: int = 0):
    """Core demo step (UI-independent, unit-tested). Returns (corrupted image, [(name, overlay|None, text)],
    gt overlay or None)."""
    img, gt = cap_size(img, gt)
    corrupted = apply_corruption(img, corruption, int(severity), seed=seed) if corruption != "clean" and severity > 0 else img
    results = []
    for i, (name, m) in enumerate(models.items()):
        if isinstance(m, str):
            results.append((name, None, f"**{name}**: {m}"))
            continue
        if m.needs_gt:
            if gt is None:
                results.append((name, None, f"**{name}**: needs a ground-truth mask (test images only)"))
                continue
            m.set_gt(gt)
        prob = m.predict(corrupted)
        text = f"**{name}**: {m.last_latency_ms:.0f} ms"
        if gt is not None and show_gt:
            g = gt if gt.shape == prob.shape else cv2.resize(gt.astype(np.uint8), prob.shape[::-1], interpolation=cv2.INTER_NEAREST) > 0
            sc = image_scores(prob, g)
            text += f" · S-measure {sc['Sm']:.3f} · MAE {sc['MAE']:.3f} (this image)"
        results.append((name, overlay(corrupted, prob, COLORS[i % len(COLORS)]), text))
    gt_img = overlay(corrupted, gt.astype(np.float32), GT_COLOR) if gt is not None and show_gt else None
    return corrupted, results, gt_img


# ---------------------------------------------------------------------------------------- results tab
def _summarize_module():
    spec = importlib.util.spec_from_file_location("summarize_benchmark", REPO / "scripts" / "summarize_benchmark.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def results_tables() -> dict[str, pd.DataFrame]:
    tables = {}
    bench_path = REPO / "results" / "benchmark.csv"
    if bench_path.exists():
        bench = pd.read_csv(bench_path).drop_duplicates(["model", "dataset", "corruption", "severity"], keep="last")
        clean = bench[bench.corruption == "clean"]
        if len(clean):
            t = clean.pivot_table(index="model", columns="dataset", values=["Sm", "MAE"], aggfunc="last")
            ds_order = [d for d in ["CHAMELEON", "CAMO", "COD10K", "NC4K"] if d in t.columns.levels[1]]
            t = t[[(m, d) for d in ds_order for m in ("Sm", "MAE") if (m, d) in t.columns]]
            t.columns = [f"{d} {m}" for m, d in t.columns]
            tables["clean"] = t.round(3).reset_index()
        try:
            summ = _summarize_module().summarize(bench, load_protocol())
            if len(summ):
                s = summ[summ.group.isin(["all", "seen", "unseen"])]
                s = s[["model", "dataset", "group", "n_conditions", "complete", "Sm_clean", "Sm_mcorr", "Sm_retention", "mCE_Sm"]]
                tables["robustness"] = s.round(3)
        except Exception as e:
            tables["robustness"] = pd.DataFrame({"error": [str(e)]})
        progress = bench[bench.corruption != "clean"].groupby(["model", "dataset"]).size().rename("conditions_done")
        if len(progress):
            tables["progress"] = progress.reset_index()
    eff = REPO / "results" / "efficiency.csv"
    if eff.exists():
        e = pd.read_csv(eff)
        tables["efficiency"] = e[[c for c in ["model", "params_M", "gflops", "latency_ms_median", "fps", "peak_vram_MB",
                                             "precision", "gpu_name"] if c in e.columns]]
    curves = []
    for log in sorted((REPO / "results" / "train").glob("*/log.csv")):
        df = pd.read_csv(log)
        if len(df):
            curves.append(df.assign(run=log.parent.name)[["run", "epoch", "step", "loss", "sup", "cons"]])
    if curves:
        tables["training"] = pd.concat(curves).groupby("run").tail(1).round(4)
    return tables


# ---------------------------------------------------------------------------------------- UI
def build_app(models: dict, examples: list[dict], device_label: str):
    protocol = load_protocol()
    choices = corruption_choices(protocol)
    gallery_items = lambda ex: [(thumbnail(e["image"]), e["label"]) for e in ex]  # noqa: E731

    def on_select(ex_list, evt: gr.SelectData):
        e = ex_list[evt.index]
        return read_rgb(e["image"]), (str(e["gt"]) if e["gt"] else None), f"Selected **{e['label']}**"

    def refresh_gallery():
        ex = examples + my_photos()
        return gr.update(value=gallery_items(ex)), ex

    def run(img, corr_label, severity, show_gt, gt_path):
        n = len(models)
        if img is None:
            return [None] + [None, ""] * n + [gr.update(value=None, visible=False)]
        gt = read_mask(gt_path) if gt_path else None
        if gt is not None and gt.shape != img.shape[:2]:
            gt = cv2.resize(gt.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        corrupted, results, gt_img = detect(img, choices[corr_label], int(severity), gt, models, show_gt)
        out = [corrupted]
        for _, ov, text in results:
            out += [ov, text]
        return out + [gr.update(value=gt_img, visible=gt_img is not None)]

    with gr.Blocks(title="Robust camouflage detection") as app:
        gr.Markdown("# Camouflaged object detection under image corruptions\n"
                    f"Models run on **{device_label}**. Pick a test image or upload a photo, then add a corruption. "
                    "Corruptions marked *UNSEEN* were never used when training our adapter.")
        ex_state = gr.State(examples + my_photos())
        gt_state = gr.State(None)
        with gr.Tab("Detector"):
            with gr.Row():
                with gr.Column(scale=1):
                    image = gr.Image(type="numpy", label="Input image (upload or pick below)", sources=["upload", "clipboard"])
                    corruption = gr.Dropdown(list(choices), value="none (clean)", label="Corruption")
                    severity = gr.Slider(0, 5, value=0, step=1, label="Severity (0 = clean)")
                    show_gt = gr.Checkbox(False, label="Show ground truth (test images only)")
                    run_btn = gr.Button("Detect", variant="primary")
                    picked = gr.Markdown("")
                with gr.Column(scale=2):
                    gallery = gr.Gallery(gallery_items(examples + my_photos()), label="Test images + demo/my_photos",
                                         columns=6, height=260, allow_preview=False)
                    refresh = gr.Button("Reload demo/my_photos", size="sm")
            with gr.Row():
                corrupted_out = gr.Image(label="Corrupted input", interactive=False)
                model_outs = []
                for name in models:
                    with gr.Column():
                        ov = gr.Image(label=name, interactive=False)
                        txt = gr.Markdown("")
                        model_outs += [ov, txt]
                gt_out = gr.Image(label="Ground truth", interactive=False, visible=False)
        with gr.Tab("Benchmark results"):
            gr.Markdown("Read from `results/*.csv` (the benchmark may still be running: see `complete` and "
                        "`n_conditions`). Retention = mean score under corruption / clean score.")
            tbl = {k: gr.Dataframe(label=lbl, interactive=False) for k, lbl in [
                ("clean", "Clean test sets: S-measure (higher is better) and MAE (lower is better)"),
                ("robustness", "Robustness: all / seen / unseen corruptions"),
                ("progress", "Corrupted conditions finished per model and dataset (of 40)"),
                ("efficiency", "Efficiency on the laptop GPU (results/efficiency.csv)"),
                ("training", "Our adapter: latest training log entry per run")]}
            reload_btn = gr.Button("Refresh", size="sm")

        def load_tables():
            t = results_tables()
            return [t.get(k, pd.DataFrame({"status": ["no results yet"]})) for k in tbl]

        inputs = [image, corruption, severity, show_gt, gt_state]
        outputs = [corrupted_out] + model_outs + [gt_out]
        gallery.select(on_select, [ex_state], [image, gt_state, picked]).then(run, inputs, outputs)
        image.upload(lambda: (None, "Uploaded image (no ground truth)"), None, [gt_state, picked])
        image.clear(lambda: (None, ""), None, [gt_state, picked])
        run_btn.click(run, inputs, outputs)
        corruption.change(run, inputs, outputs)
        severity.release(run, inputs, outputs)
        show_gt.change(run, inputs, outputs)
        refresh.click(refresh_gallery, None, [gallery, ex_state])
        reload_btn.click(load_tables, None, list(tbl.values()))
        app.load(load_tables, None, list(tbl.values()))
    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=None, help="default: SINet-V2 + best trained Ours variant")
    ap.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    dev = pick_device(args.device)
    t0 = time.time()
    models = load_models(args.models, args.device)
    for n, m in models.items():
        print(f"  {n}: {m if isinstance(m, str) else 'loaded'}")
    warmup(models)
    examples = pick_examples()
    if not examples:
        print("No local test images found (data/raw); upload photos or add them to demo/my_photos/.")
    label = "CPU" if dev.type == "cpu" else __import__("torch").cuda.get_device_name(0)
    print(f"ready in {time.time() - t0:.0f} s")
    build_app(models, examples, label).launch(share=args.share, server_port=args.port, inbrowser=True)


if __name__ == "__main__":
    main()
