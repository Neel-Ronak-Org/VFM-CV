"""Numbers for the course presentation, read only from results/*.csv (same rule as the paper).

    python presentation/make_numbers.py      # writes presentation/numbers.json, read by build_deck.js
"""
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.reporting import training_hours  # noqa: E402

R = REPO / "results"
MODELS = ["SINet-V2", "PFNet", "ZoomNet", "SAM2-UNet", "SAM-Adapter", "Ours-Full", "Ours-Full-512"]
DISPLAY = {"Ours-Full": "Ours (1024)", "Ours-Full-512": "Ours (512)"}
DATASETS = ["CHAMELEON", "CAMO", "COD10K", "NC4K"]


def r3(x):
    return None if pd.isna(x) else round(float(x), 3)


def main():
    rob = pd.read_csv(R / "summary_robustness.csv")
    allg = rob[rob.group == "all"].set_index(["model", "dataset"])
    cod = rob[rob.dataset == "COD10K"].set_index(["model", "group"])
    eff = pd.read_csv(R / "efficiency.csv").set_index("model")
    sig = pd.read_csv(R / "significance.csv")
    sel = pd.read_csv(R / "qualitative_selection.csv")
    data = pd.read_csv(R / "data_check.csv")

    def s(a, b, ds="COD10K", group="all"):
        row = sig[(sig.a == a) & (sig.b == b) & (sig.dataset == ds) & (sig.group == group)].iloc[0]
        return {"diff": r3(row.mean_diff), "lo": r3(row.ci_lo), "hi": r3(row.ci_hi), "sig": bool(row.significant)}

    out = {"models": [], "datasets": DATASETS, "images": {}, "significance": {}, "teaser": {}}
    for m in MODELS:
        e = eff.loc[m]
        out["models"].append({
            "name": DISPLAY.get(m, m), "key": m,
            "clean": r3(cod.loc[(m, "all"), "Sm_clean"]), "mcorr": r3(cod.loc[(m, "all"), "Sm_mcorr"]),
            "ret": r3(cod.loc[(m, "all"), "Sm_retention"]),
            "ret_seen": r3(cod.loc[(m, "seen"), "Sm_retention"]), "ret_unseen": r3(cod.loc[(m, "unseen"), "Sm_retention"]),
            "mcorr_ds": {d: r3(allg.loc[(m, d), "Sm_mcorr"]) if (m, d) in allg.index else None for d in DATASETS},
            "clean_ds": {d: r3(allg.loc[(m, d), "Sm_clean"]) if (m, d) in allg.index else None for d in DATASETS},
            "params_M": round(float(e.params_M), 1), "gflops": None if pd.isna(e.gflops) else round(float(e.gflops), 1),
            "latency_ms": round(float(e.latency_ms_median), 1), "fps": round(float(e.fps), 1),
            "vram_MB": int(e.peak_vram_MB), "precision": e.precision})
    out["gpu"] = eff.gpu_name.iloc[0]
    for d in data.itertuples():
        out["images"][d.dataset] = int(d.n_pairs)
    for key, (a, b, g) in {"ours_sinet": ("Ours-Full", "SINet-V2", "all"), "ours_zoom": ("Ours-Full", "ZoomNet", "all"),
                           "ours_sam2unet": ("Ours-Full", "SAM2-UNet", "all"),
                           "five_sinet": ("Ours-Full-512", "SINet-V2", "all"),
                           "five_ours_local": ("Ours-Full-512", "Ours-Full", "local"),
                           "full_clean_unseen": ("Ours-Full", "Ours-Clean", "unseen"),
                           "full_clean_seen": ("Ours-Full", "Ours-Clean", "seen"),
                           "full_clean_rain": ("Ours-Full", "Ours-Clean", "rain"),
                           "full_clean_motion": ("Ours-Full", "Ours-Clean", "motion_blur"),
                           "full_aug_all": ("Ours-Full", "Ours-Aug", "all")}.items():
        out["significance"][key] = s(a, b, group=g)
    out["significance"]["five_sinet_nc4k"] = s("Ours-Full-512", "SINet-V2", ds="NC4K")
    t = sel[sel.figure == "teaser"]
    out["teaser"] = {"image": t.image.iloc[0],
                     **{f"{r.model}_{r.severity}": r3(r.Sm) for r in t.itertuples()}}
    out["train_hours"] = {k: round(training_hours(run), 1) for k, run in
                          (("Ours-Full", "ours_full"), ("Ours-Full-512", "ours_full_512"))}
    out["params"] = json.loads((R / "train" / "ours_full" / "summary.json").read_text())["params"]
    out["train_images"] = int(data.set_index("dataset").loc["train", "n_pairs"]) if "train" in set(data.dataset) else None
    (REPO / "presentation" / "numbers.json").write_text(json.dumps(out, indent=1))
    print("wrote presentation/numbers.json")


if __name__ == "__main__":
    main()
