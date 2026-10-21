"""Gradio demo: camouflaged object detection under image corruptions.

    python demo/app.py                       # SINet-V2 vs our best trained adapter, GPU if available
    python demo/app.py --models SINet-V2 SAM2-UNet Ours-Full   # which models are ticked at start
    python demo/app.py --device cpu          # no GPU (slower)
    python demo/app.py --share               # temporary public link (e.g. to show on another device)

Tab "Detector": upload your own photo (drag & drop) or pick one of 12 test images / demo/my_photos/,
tick up to 6 models to compare (baselines and every trained Ours variant; each loads on first use),
choose a corruption and severity (0 = clean), and compare the models side by side as coloured overlays
with each model's latency. For test images, "Show ground truth" adds the GT and per-image S-measure / MAE.
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
COLORS = [(255, 64, 64), (255, 140, 0), (255, 220, 0), (160, 120, 255), (255, 80, 200),
          (64, 200, 255), (0, 160, 255), (80, 255, 80), (0, 200, 140), (180, 255, 0)]  # one per PICKABLE model
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
PICKABLE = ["SINet-V2", "PFNet", "ZoomNet", "SAM2-UNet", "SAM-Adapter",
            "Ours-Clean", "Ours-Aug", "Ours-Full", "Ours-Full-15ep", "Ours-Full-in352"]
SLOW = {"SAM-Adapter": "~2.5 s/image, 2.5 GB GPU"}
MAX_SLOTS = 6  # side-by-side outputs shown at once


def _ours_run(name: str) -> str | None:
    from src.models.robust_adapter import OursClean, OursAug, OursFull, OursFullEp15, OursFullIn352
    runs = {c.name: c.run for c in (OursClean, OursAug, OursFull, OursFullEp15, OursFullIn352)}
    return runs.get(name)


def ours_status(name: str) -> str:
    """'finished' / 'unfinished' / 'missing' for an Ours variant, from its checkpoint files (nothing loaded)."""
    from src.models.robust_adapter import OURS_DIR
    d = OURS_DIR / str(_ours_run(name))
    return "finished" if (d / "final.pt").exists() else "unfinished" if (d / "last.pt").exists() else "missing"


def pick_ours() -> str | None:
    """The first FINISHED variant in OURS_PREFERENCE (so a run still training never replaces a finished
    model); if none is finished, the first one that exists (it is labelled "...-unfinished")."""
    status = {n: ours_status(n) for n in OURS_PREFERENCE}
    return (next((n for n in OURS_PREFERENCE if status[n] == "finished"), None)
            or next((n for n in OURS_PREFERENCE if status[n] == "unfinished"), None))


def pickable() -> list[str]:
    """Models offered in the picker: all baselines, plus the Ours variants that have a checkpoint."""
    return [n for n in PICKABLE if _ours_run(n) is None or ours_status(n) != "missing"]


def default_selection(names: list[str] | None) -> list[str]:
    if names:
        return names
    ours = pick_ours()
    return ["SINet-V2"] + ([ours] if ours else [])


class ModelPool:
    """Loads models the first time they are selected (and warms them up), then keeps them.
    A model that fails to load is remembered as an error string shown in its column."""

    def __init__(self, device: str | None):
        self.device = str(pick_device(device))
        self.models: dict = {}

    def get(self, name: str):
        if name not in self.models:
            try:
                m = build_model(name, device=self.device)
                warmup({name: m})
                self.models[name] = m
            except Exception as e:  # show the reason in the UI instead of crashing the demo
                self.models[name] = f"unavailable: {type(e).__name__}: {e}"
        return self.models[name]

    def subset(self, names: list[str]) -> dict:
        """name -> model (or error), loading as needed. On a CUDA out-of-memory error, models that are not
        selected are released and loading is retried once."""
        out = {}
        for n in names:
            m = self.get(n)
            if isinstance(m, str) and "out of memory" in m.lower():
                self.release(keep=names)
                del self.models[n]
                m = self.get(n)
            out[n] = m
        return out

    def release(self, keep: list[str]) -> None:
        for n in [n for n in self.models if n not in keep]:
            del self.models[n]
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


def load_models(names: list[str] | None, device: str | None) -> dict:
    """name -> model or error string, for the default selection (used by main() for a start-up check)."""
    return ModelPool(device).subset(default_selection(names))


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
        k = PICKABLE.index(name) if name in PICKABLE else i  # a model keeps its colour whatever else is shown
        results.append((name, overlay(corrupted, prob, COLORS[k % len(COLORS)]), text))
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
def build_app(pool: "ModelPool", examples: list[dict], device_label: str, selected: list[str]):
    protocol = load_protocol()
    choices = corruption_choices(protocol)
    gallery_items = lambda ex: [(thumbnail(e["image"]), e["label"]) for e in ex]  # noqa: E731

    def on_select(ex_list, evt: gr.SelectData):
        e = ex_list[evt.index]
        return read_rgb(e["image"]), (str(e["gt"]) if e["gt"] else None), f"Selected **{e['label']}**"

    def refresh_gallery():
        ex = examples + my_photos()
        return gr.update(value=gallery_items(ex)), ex

    def run(img, corr_label, severity, show_gt, gt_path, chosen):
        chosen = [n for n in PICKABLE if n in (chosen or [])] + [n for n in (chosen or []) if n not in PICKABLE]
        note = f"Showing the first {MAX_SLOTS} selected models." if len(chosen) > MAX_SLOTS else ""
        chosen = chosen[:MAX_SLOTS]
        if img is None:
            return [None, note] + [gr.update(visible=False)] * (2 * MAX_SLOTS) + [gr.update(value=None, visible=False)]
        gt = read_mask(gt_path) if gt_path else None
        if gt is not None and gt.shape != img.shape[:2]:
            gt = cv2.resize(gt.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        corrupted, results, gt_img = detect(img, choices[corr_label], int(severity), gt, pool.subset(chosen), show_gt)
        out = [corrupted, note]
        for i in range(MAX_SLOTS):
            if i < len(results):
                name, ov, text = results[i]
                out += [gr.update(value=ov, label=name, visible=True), gr.update(value=text, visible=True)]
            else:
                out += [gr.update(value=None, visible=False), gr.update(value="", visible=False)]
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
                    image = gr.Image(type="numpy", label="Your photo: drag & drop / click to upload (or pick a test image)",
                                     sources=["upload", "clipboard"])
                    model_pick = gr.CheckboxGroup(
                        pickable(), value=[n for n in selected if n in pickable()] or selected,
                        label=f"Models to compare (up to {MAX_SLOTS}; loaded on first use)",
                        info="; ".join(f"{k}: {v}" for k, v in SLOW.items()) + ". Ours-Full-in352 = Ours-Full fed a "
                             "352-px view (resolution test).")
                    corruption = gr.Dropdown(list(choices), value="none (clean)", label="Corruption")
                    severity = gr.Slider(0, 5, value=0, step=1, label="Severity (0 = clean)")
                    show_gt = gr.Checkbox(False, label="Show ground truth (test images only)")
                    run_btn = gr.Button("Detect", variant="primary")
                    picked = gr.Markdown("")
                with gr.Column(scale=2):
                    gallery = gr.Gallery(gallery_items(examples + my_photos()), label="Test images + demo/my_photos",
                                         columns=6, height=260, allow_preview=False)
                    refresh = gr.Button("Reload demo/my_photos", size="sm")
            slot_note = gr.Markdown("")
            with gr.Row():
                corrupted_out = gr.Image(label="Input (with corruption)", interactive=False)
                gt_out = gr.Image(label="Ground truth", interactive=False, visible=False)
            model_outs = []
            for r in range(0, MAX_SLOTS, 3):
                with gr.Row():
                    for i in range(r, min(r + 3, MAX_SLOTS)):
                        with gr.Column():
                            ov = gr.Image(label=f"model {i + 1}", interactive=False, visible=False)
                            txt = gr.Markdown("", visible=False)
                            model_outs += [ov, txt]
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

        inputs = [image, corruption, severity, show_gt, gt_state, model_pick]
        outputs = [corrupted_out, slot_note] + model_outs + [gt_out]
        gallery.select(on_select, [ex_state], [image, gt_state, picked]).then(run, inputs, outputs)
        image.upload(lambda: (None, "Uploaded image (no ground truth)"), None, [gt_state, picked]).then(run, inputs, outputs)
        model_pick.change(run, inputs, outputs)
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
    ap.add_argument("--models", nargs="+", default=None,
                    help="models ticked at start (default: SINet-V2 + best finished Ours variant); "
                         "any model can be ticked later in the page")
    ap.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    dev = pick_device(args.device)
    t0 = time.time()
    pool = ModelPool(args.device)
    selected = default_selection(args.models)
    for n, m in pool.subset(selected).items():  # load + warm up the default models now; others on first use
        print(f"  {n}: {m if isinstance(m, str) else 'loaded'}")
    print(f"  also selectable: {', '.join(n for n in pickable() if n not in selected)}")
    examples = pick_examples()
    if not examples:
        print("No local test images found (data/raw); upload photos or add them to demo/my_photos/.")
    label = "CPU" if dev.type == "cpu" else __import__("torch").cuda.get_device_name(0)
    print(f"ready in {time.time() - t0:.0f} s")
    build_app(pool, examples, label, selected).launch(share=args.share, server_port=args.port, inbrowser=True)


if __name__ == "__main__":
    main()
