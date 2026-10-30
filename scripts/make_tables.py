"""Paper tables as LaTeX, generated ONLY from results files (never typed by hand).

    python scripts/significance.py     # first: results/significance.csv (used by Table 5)
    python scripts/make_tables.py                    # -> results/tables/*.tex
    python scripts/make_tables.py --out paper/tables # write straight into a LaTeX project instead

Files (each table is a complete table environment for \\input{}; needs booktabs):
  table1_clean.tex          clean S-alpha / E-phi / F-beta^w / MAE on the four test sets
  table2_robust_<DS>.tex    robustness on one dataset (COD10K is Table 2 in the paper; others for the appendix)
  table3_ablation.tex       training components (COD10K) and epochs / input resolution (CHAMELEON)
  table4_efficiency.tex     parameters, GFLOPs, latency, FPS, memory, training time
  table5_significance.tex   paired bootstrap 95 % CIs of the S-alpha difference under corruption
  table6_percorruption.tex  Dice retention per corruption on COD10K
  numbers.tex               one LaTeX macro for every number the paper's text quotes
Best value per column (excluding the oracle) is bold. Ours (512) is mean +- sample std over the seeds that
are complete for that dataset (SEED_GROUPS in src/reporting.py); the caption says how many.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.corruptions import load_protocol  # noqa: E402
from src.reporting import (DATASETS, MAIN_MODELS, MODELS, RESULTS, SEED_GROUPS, aggregate_seeds,  # noqa: E402
                           clean_scores, display, efficiency, load_bench, load_released, robustness,
                           training_hours)

OUT = RESULTS / "tables"
ORACLE = "Oracle-Box-MobileSAM"
DASH = "--"


def fmt(mean, std=None, digits=3, bold=False) -> str:
    if mean is None or pd.isna(mean):
        return DASH
    s = f"{mean:.{digits}f}"
    if std is not None and not pd.isna(std):
        s += "{\\tiny$\\pm$" + (f"{std:.{digits}f}".lstrip("0") or "0") + "}"
    return f"\\textbf{{{s}}}" if bold else s


def best_flags(values: dict, lower: bool) -> set:
    """Row keys holding the best value of a column (ties all bold)."""
    vals = {k: v for k, v in values.items() if v is not None and not pd.isna(v)}
    if not vals:
        return set()
    target = min(vals.values()) if lower else max(vals.values())
    return {k for k, v in vals.items() if round(v, 3) == round(target, 3)}


def seeds_note(counts: dict) -> str:
    n = [c for m, c in counts.items() if m in SEED_GROUPS and c > 1]
    return (f" Ours (512): mean $\\pm$ std over {min(n)}--{max(n)} seeds." if n and min(n) != max(n)
            else f" Ours (512): mean $\\pm$ std over {n[0]} seeds." if n else "")


def res_tex(model: str) -> str:
    return MODELS[model][2].replace("*", "$^*$")


def write(name: str, text: str, out: Path = OUT) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / name).write_text(text)
    print(f"wrote {out / name}")


# ------------------------------------------------------------------------------------------- Table 1
def table1(bench, released) -> str:
    metrics = [("Sm", "$S_\\alpha\\uparrow$", False), ("Em", "$E_\\phi\\uparrow$", False),
               ("wFm", "$F_\\beta^w\\uparrow$", False), ("MAE", "MAE$\\downarrow$", True)]
    rows = MAIN_MODELS + [ORACLE]
    data, counts = {}, {}
    for m in rows:
        for ds in DATASETS:
            mean, std, n = aggregate_seeds(lambda mm, d: clean_scores(bench, mm, d, released), m, ds)
            data[(m, ds)] = (mean, std)
            counts[m] = max(counts.get(m, 0), n)
    bold = {}
    for ds in DATASETS:
        for key, _, lower in metrics:
            vals = {m: (data[(m, ds)][0] or {}).get(key) for m in MAIN_MODELS}
            bold[(ds, key)] = best_flags(vals, lower)
    cols = "l" + "cccc" * len(DATASETS)
    head1 = " & " + " & ".join(f"\\multicolumn{{4}}{{c}}{{{ds}}}" for ds in DATASETS) + " \\\\"
    cmid = " ".join(f"\\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}" for i in range(len(DATASETS)))
    head2 = "Model & " + " & ".join(lbl for _ in DATASETS for _, lbl, _ in metrics) + " \\\\"
    body, group = [], None
    for m in rows:
        g = MODELS[m][1]
        if group is not None and g != group:
            body.append("\\midrule")
        group = g
        cells = []
        for ds in DATASETS:
            mean, std = data[(m, ds)]
            for key, _, _ in metrics:
                cells.append(fmt((mean or {}).get(key), (std or {}).get(key) if std else None,
                                 bold=m in bold[(ds, key)]))
        name = display(m) + ("$^\\dagger$" if m == "FEDER" else "$^\\ddagger$" if m == ORACLE else "")
        body.append(f"{name} & " + " & ".join(cells) + " \\\\")
    cap = ("Clean-image results on the four COD test sets. Best per column in bold (oracle excluded). "
           "$^\\dagger$FEDER: scored from the authors' released prediction maps (no COD10K maps released; "
           "trained on COD10K-train only). $^\\ddagger$Oracle: MobileSAM prompted with the ground-truth box, an "
           "analysis probe, not a competitor." + seeds_note(counts))
    return ("\\begin{table*}[t]\n\\centering\n\\caption{" + cap + "}\n\\label{tab:clean}\n\\footnotesize\n"
            "\\setlength{\\tabcolsep}{2.1pt}\n\\begin{tabular}{" + cols + "}\n\\toprule\n" + head1 + "\n" + cmid
            + "\n" + head2 + "\n\\midrule\n" + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")


# ------------------------------------------------------------------------------------------- Table 2
def table2(bench, dataset, protocol) -> str | None:
    cols = [("clean_Sm", "Clean", False), ("mcorr_Sm", "mCorr", False), ("ret_Sm", "Ret.", False),
            ("ret_Dice", "Ret.$_{\\mathrm{Dice}}$", False), ("ret_seen", "Seen", False), ("ret_unseen", "Unseen", False),
            ("ret_local", "Local", False), ("ret_global", "Global", False), ("mCE", "mCE$\\downarrow$", True)]
    rows = [m for m in MAIN_MODELS if m != "FEDER"] + [ORACLE]
    data, counts = {}, {}
    for m in rows:
        mean, std, n = aggregate_seeds(lambda mm, d: robustness(bench, mm, d, protocol), m, dataset)
        data[m], counts[m] = (mean, std), n
    if not any(data[m][0] for m in rows):
        return None
    bold = {k: best_flags({m: (data[m][0] or {}).get(k) for m in rows if m != ORACLE}, lower) for k, _, lower in cols}
    body, group = [], None
    for m in rows:
        g = MODELS[m][1]
        if group is not None and g != group:
            body.append("\\midrule")
        group = g
        mean, std = data[m]
        cells = [fmt((mean or {}).get(k), (std or {}).get(k) if std else None, digits=2 if k == "mCE" else 3,
                     bold=m in bold[k]) for k, _, _ in cols]
        name = display(m) + ("$^\\ddagger$" if m == ORACLE else "")
        body.append(f"{name} & {res_tex(m)} & " + " & ".join(cells) + " \\\\")
    head = "Model & Input & " + " & ".join(lbl for _, lbl, _ in cols) + " \\\\"
    cap = (f"Robustness on {dataset} (8 corruptions $\\times$ 5 severities). Clean and mCorr: $S_\\alpha$; "
           "Ret.: mCorr/clean ($S_\\alpha$ unless marked); Seen/Unseen: retention on the corruptions used / never "
           "used to train our models; Local: noise, blur, JPEG, rain; Global: brightness, contrast, fog; mCE: "
           "corruption error relative to SINet-V2. Input: side length in px (ZoomNet$^*$ also uses 0.5$\\times$ "
           "and 1.5$\\times$). $^\\ddagger$Oracle with the ground-truth box, not a competitor. Retention alone can favour weaker models; read it with mCorr and mCE."
           + seeds_note(counts))
    label = "tab:robust" if dataset == "COD10K" else f"tab:robust-{dataset.lower()}"
    return ("\\begin{table*}[t]\n\\centering\n\\caption{" + cap + "}\n\\label{" + label + "}\n\\footnotesize\n"
            "\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{llccccccccc}\n\\toprule\n" + head + "\n\\midrule\n" + "\n".join(body)
            + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")


# ------------------------------------------------------------------------------------------- Table 3
def table3(bench, protocol) -> str:
    cols = [("clean_Sm", "Clean"), ("mcorr_Sm", "mCorr"), ("mcorr_seen", "Seen"), ("mcorr_unseen", "Unseen"),
            ("mcorr_local", "Local"), ("mcorr_global", "Global")]
    blocks = [("(a) Training components, COD10K", "COD10K", [
        ("Ours-Clean", "Adapter, clean images only"), ("Ours-Aug", "+ corruption augmentation"),
        ("Ours-Full", "+ consistency loss (Ours, 1024)"), ("Ours-Full-512", "Ours, trained at 512 px")]),
        ("(b) Epochs and input resolution, CHAMELEON", "CHAMELEON", [
            ("Ours-Full-15ep", "Ours, 15 epochs"), ("Ours-Full", "Ours, 30 epochs (1024 px)"),
            ("Ours-Full-in768", "test-time view 768 px"), ("Ours-Full-in512", "test-time view 512 px"),
            ("Ours-Full-in352", "test-time view 352 px"), ("Ours-Full-512", "trained at 512 px")])]
    body, counts = [], {}
    for title, ds, items in blocks:
        body.append(f"\\multicolumn{{{len(cols) + 1}}}{{l}}{{\\textit{{{title}}}}} \\\\")
        data = {}
        for m, _ in items:
            mean, std, n = aggregate_seeds(lambda mm, d: robustness(bench, mm, d, protocol), m, ds)
            data[m] = (mean, std)
            counts[m] = max(counts.get(m, 0), n)
        bold = {k: best_flags({m: (data[m][0] or {}).get(k) for m, _ in items}, False) for k, _ in cols}
        for m, label in items:
            mean, std = data[m]
            cells = [fmt((mean or {}).get(k), (std or {}).get(k) if std else None, bold=m in bold[k]) for k, _ in cols]
            body.append(f"\\quad {label} & " + " & ".join(cells) + " \\\\")
        body.append("\\midrule")
    body = body[:-1]
    cap = ("Ablations. All columns are $S_\\alpha$: clean, and mean under corruption (mCorr) over all, seen, unseen, "
           "local and global corruptions. Test-time views shrink the input of the 1024-px model before inference; "
           "``trained at 512 px'' learns at that resolution. Best per block in bold." + seeds_note(counts))
    head = "Variant & " + " & ".join(lbl for _, lbl in cols) + " \\\\"
    return ("\\begin{table}[t]\n\\centering\n\\caption{" + cap + "}\n\\label{tab:ablation}\n\\footnotesize\n"
            "\\setlength{\\tabcolsep}{2.5pt}\n\\resizebox{\\columnwidth}{!}{%\n\\begin{tabular}{lcccccc}\n\\toprule\n"
            + head + "\n\\midrule\n" + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")


# ------------------------------------------------------------------------------------------- Table 4
def table4() -> str:
    eff = efficiency()
    rows = [m for m in MAIN_MODELS if m in eff.index]
    train = {"Ours-Full": training_hours("ours_full"), "Ours-Full-512": training_hours("ours_full_512")}
    cols = [("params_M", "Params (M)", True, 1), ("gflops", "GFLOPs", True, 1), ("latency_ms_median", "Latency (ms)", True, 1),
            ("fps", "FPS", False, 1), ("peak_vram_MB", "VRAM (MB)", True, 0)]
    bold = {k: best_flags({m: eff.loc[m, k] for m in rows if pd.notna(eff.loc[m, k])}, lower) for k, _, lower, _ in cols}
    body = []
    for m in rows:
        r = eff.loc[m]
        cells = [fmt(r[k], digits=d, bold=m in bold[k]) for k, _, _, d in cols]
        th = train.get(m)
        body.append(f"{display(m)} & {res_tex(m)} & {r['precision']} & " + " & ".join(cells)
                    + f" & {fmt(th, digits=1) if th else DASH} \\\\")
    gpu = eff["gpu_name"].dropna().iloc[0] if "gpu_name" in eff and len(eff) else "GPU"
    cap = (f"Efficiency on one {gpu} (4 GB), batch 1, end-to-end on a 768$\\times$1024 image (median over runs). "
           "Precision as in each model's official inference. Training: our full training run on the same laptop GPU.")
    head = "Model & Input & Prec. & " + " & ".join(lbl for _, lbl, _, _ in cols) + " & Train (h) \\\\"
    return ("\\begin{table}[t]\n\\centering\n\\caption{" + cap + "}\n\\label{tab:efficiency}\n\\footnotesize\n"
            "\\setlength{\\tabcolsep}{2.5pt}\n\\resizebox{\\columnwidth}{!}{%\n\\begin{tabular}{lllcccccc}\n\\toprule\n"
            + head + "\n\\midrule\n" + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")


# ------------------------------------------------------------------------------------------- Table 5
def table5() -> str | None:
    p = RESULTS / "significance.csv"
    if not p.exists():
        return None
    sig = pd.read_csv(p)
    rows = [("Ours-Full", "SINet-V2", "all"), ("Ours-Full", "ZoomNet", "all"), ("Ours-Full", "PFNet", "all"),
            ("Ours-Full", "SAM2-UNet", "all"), ("Ours-Full", "SAM-Adapter", "all"),
            ("Ours-Full-512", "SINet-V2", "all"), ("Ours-Full-512", "ZoomNet", "all"),
            ("Ours-Full-512", "Ours-Full", "local"), ("Ours-Full-512", "Ours-Full", "global"),
            ("Ours-Full", "Ours-Clean", "seen"), ("Ours-Full", "Ours-Clean", "unseen"), ("Ours-Full", "Ours-Clean", "rain")]
    order = ["COD10K", "NC4K", "CAMO", "CHAMELEON"]
    body = []
    for a, b, g in rows:
        cells = []
        for ds in order:
            r = sig[(sig.a == a) & (sig.b == b) & (sig.group == g) & (sig.dataset == ds)]
            if r.empty:
                cells.append(DASH)
                continue
            r = r.iloc[0]
            s = f"{r.mean_diff:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]"
            cells.append(f"\\textbf{{{s}}}" if r.significant else s)
        grp = "" if g == "all" else f" ({g.replace('_', ' ')})"
        body.append(f"{display(a)} $-$ {display(b)}{grp} & " + " & ".join(cells) + " \\\\")
    cap = ("Paired bootstrap tests (5{,}000 resamples): mean per-image difference in $S_\\alpha$ under corruption and "
           "its 95\\% CI. Bold: CI excludes zero. Groups: all 40 corrupted conditions unless stated; seen/unseen "
           "refer to the corruptions used / never used in our training.")
    head = "Comparison & " + " & ".join(f"{ds} (n={int(sig[sig.dataset == ds].n_images.max()) if (sig.dataset == ds).any() else 0})" for ds in order) + " \\\\"
    return ("\\begin{table*}[t]\n\\centering\n\\caption{" + cap + "}\n\\label{tab:significance}\n\\footnotesize\n"
            "\\begin{tabular}{lcccc}\n\\toprule\n" + head + "\n\\midrule\n" + "\n".join(body)
            + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")


# ------------------------------------------------------------------------------------------- Table 6
def table6(bench, protocol, dataset="COD10K") -> str:
    corrs = protocol["corruptions"]
    rows = [m for m in MAIN_MODELS if m != "FEDER"]
    data, counts = {}, {}
    for m in rows:
        mean, std, n = aggregate_seeds(lambda mm, d: robustness(bench, mm, d, protocol), m, dataset)
        data[m], counts[m] = (mean, std), n
    bold = {c: best_flags({m: (data[m][0] or {}).get(f"retDice_{c}") for m in rows}, False) for c in corrs}
    short = {"gaussian_noise": "G.~noise", "gaussian_blur": "G.~blur", "motion_blur": "M.~blur", "brightness": "Bright.",
             "contrast": "Contrast", "fog": "Fog", "jpeg_compression": "JPEG", "rain": "Rain"}
    head = "Model & " + " & ".join(
        f"{short.get(c, c)}$^{{{'s' if c in protocol['seen'] else 'u'}}}$" for c in corrs) + " \\\\"
    body = []
    for m in rows:
        mean, std = data[m]
        cells = [fmt((mean or {}).get(f"retDice_{c}"), (std or {}).get(f"retDice_{c}") if std else None,
                     bold=m in bold[c]) for c in corrs]
        body.append(f"{display(m)} & " + " & ".join(cells) + " \\\\")
    cap = (f"Dice retention per corruption on {dataset} (mean over severities 1--5 divided by clean Dice). "
           "$^s$seen / $^u$unseen in our training. Rain uses our own synthetic streaks." + seeds_note(counts))
    return ("\\begin{table*}[t]\n\\centering\n\\caption{" + cap + "}\n\\label{tab:percorruption}\n\\footnotesize\n"
            "\\begin{tabular}{l" + "c" * len(corrs) + "}\n\\toprule\n" + head + "\n\\midrule\n" + "\n".join(body)
            + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")


# ------------------------------------------------------------------------------------- numbers.tex
MODEL_CODE = {"SINet-V2": "Sinet", "PFNet": "Pfnet", "ZoomNet": "Zoom", "FEDER": "Feder", "SAM2-UNet": "Samtwo",
              "SAM-Adapter": "Samad", "Ours-Full": "Ours", "Ours-Full-512": "OursFive", "Ours-Clean": "OursClean",
              "Ours-Aug": "OursAug", "Ours-Full-15ep": "OursFifteen", "Ours-Full-in352": "OursInThree",
              "Ours-Full-in512": "OursInFive", "Ours-Full-in768": "OursInSeven", "Oracle-Box-MobileSAM": "Oracle"}
DS_CODE = {"COD10K": "Cod", "NC4K": "Nck", "CAMO": "Camo", "CHAMELEON": "Cham"}
KEY_CODE = {"clean_Sm": "Clean", "mcorr_Sm": "Mcorr", "ret_Sm": "Ret", "ret_Dice": "RetDice", "clean_Dice": "CleanDice",
            "ret_seen": "RetSeen", "ret_unseen": "RetUnseen", "ret_local": "RetLocal", "ret_global": "RetGlobal",
            "mcorr_seen": "McorrSeen", "mcorr_unseen": "McorrUnseen", "mcorr_local": "McorrLocal",
            "mcorr_global": "McorrGlobal", "mCE": "Mce"}
GROUP_CODE = {"all": "All", "seen": "Seen", "unseen": "Unseen", "local": "Local", "global": "Global"}


def camel(name: str) -> str:
    return "".join(w.capitalize() for w in name.split("_"))


def numbers(bench, released, protocol) -> str:
    """LaTeX macros for every number the paper's prose quotes, so text never contains a typed-in number.
    Names: \\N<model><dataset><key>, e.g. \\NOursCodMcorr; \\NOursFiveCodCleanStd for seed std;
    \\NSig<a><b><dataset><group>{Diff,Lo,Hi}; \\NEff<model>{Params,Gflops,Latency,Fps,Vram}; \\NTrain<model>."""
    out = ["% Generated by scripts/make_tables.py from results/*.csv. Do not edit by hand."]

    def macro(name, value):
        out.append(f"\\newcommand{{\\{name}}}{{{value}}}")

    for m, mc in MODEL_CODE.items():
        for ds, dc in DS_CODE.items():
            mean, std, n = aggregate_seeds(lambda mm, d: robustness(bench, mm, d, protocol), m, ds)
            if mean:
                for k, kc in KEY_CODE.items():
                    if mean.get(k) is not None:
                        macro(f"N{mc}{dc}{kc}", f"{mean[k]:.{2 if k == 'mCE' else 3}f}")
                        if std and std.get(k) is not None:
                            macro(f"N{mc}{dc}{kc}Std", f"{std[k]:.3f}")
                for c in protocol["corruptions"]:
                    macro(f"N{mc}{dc}RetDice{camel(c)}", f"{mean[f'retDice_{c}']:.3f}")
                    macro(f"N{mc}{dc}Mcorr{camel(c)}", f"{mean[f'mcorr_{c}']:.3f}")
                if n > 1:
                    macro(f"N{mc}{dc}Seeds", str(n))
            else:
                cs, _, _ = aggregate_seeds(lambda mm, d: clean_scores(bench, mm, d, released), m, ds)
                if cs:
                    macro(f"N{mc}{dc}Clean", f"{cs['Sm']:.3f}")
    eff = efficiency()
    for m, mc in MODEL_CODE.items():
        if m in eff.index:
            r = eff.loc[m]
            macro(f"NEff{mc}Params", f"{r['params_M']:.1f}")
            if pd.notna(r["gflops"]):
                macro(f"NEff{mc}Gflops", f"{r['gflops']:.1f}")
            macro(f"NEff{mc}Latency", f"{r['latency_ms_median']:.1f}")
            macro(f"NEff{mc}Fps", f"{r['fps']:.1f}")
            macro(f"NEff{mc}Vram", f"{r['peak_vram_MB']:.0f}")
    for m, run in (("Ours-Full", "ours_full"), ("Ours-Full-512", "ours_full_512"), ("Ours-Clean", "ours_clean"),
                   ("Ours-Aug", "ours_aug")):
        h = training_hours(run)
        if h:
            macro(f"NTrain{MODEL_CODE[m]}", f"{h:.1f}")
    sig_p = RESULTS / "significance.csv"
    if sig_p.exists():
        for _, r in pd.read_csv(sig_p).iterrows():
            if r.a in MODEL_CODE and r.b in MODEL_CODE:
                g = GROUP_CODE.get(r.group, camel(r.group))
                base = f"NSig{MODEL_CODE[r.a]}{MODEL_CODE[r.b]}{DS_CODE[r.dataset]}{g}"
                macro(base + "Diff", f"{r.mean_diff:+.3f}")
                macro(base + "Lo", f"{r.ci_lo:+.3f}")
                macro(base + "Hi", f"{r.ci_hi:+.3f}")
                macro(base + "Win", f"{100 * r.win_rate:.0f}")
    rep = RESULTS / "reproduction.csv"
    if rep.exists():
        r = pd.read_csv(rep).dropna(subset=["delta_Sm_vs_reported"])
        macro("NReproMaxDelta", f"{r.delta_Sm_vs_reported.abs().max():.4f}")
        macro("NReproModels", str(r.model.nunique()))
    dc = RESULTS / "data_check.csv"
    if dc.exists():
        for _, r in pd.read_csv(dc).iterrows():
            code = DS_CODE.get(r.dataset, "Train" if r.dataset == "train" else None)
            if code:
                macro(f"NSize{code}H", f"{r.mean_height:.0f}")
                macro(f"NSize{code}W", f"{r.mean_width:.0f}")
                macro(f"NImages{code}", f"{int(r.n_pairs):,}".replace(",", "{,}"))
    import json
    p = RESULTS / "train" / "ours_full" / "summary.json"
    if p.exists():
        d = json.loads(p.read_text())
        macro("NParamsTotal", f"{d['params']['total'] / 1e6:.2f}")
        macro("NParamsTrainable", f"{d['params']['trainable'] / 1e6:.2f}")
        macro("NParamsFrozen", f"{d['params']['frozen'] / 1e6:.2f}")
        macro("NTrainImages", f"{d['n_train_images']:,}".replace(",", "{,}"))
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT, help="output folder (default: results/tables)")
    out = ap.parse_args().out
    bench, released, protocol = load_bench(), load_released(), load_protocol()
    write("table1_clean.tex", table1(bench, released), out)
    for ds in DATASETS:
        t = table2(bench, ds, protocol)
        if t:
            write(f"table2_robust_{ds}.tex", t, out)
    write("table3_ablation.tex", table3(bench, protocol), out)
    write("table4_efficiency.tex", table4(), out)
    t5 = table5()
    if t5:
        write("table5_significance.tex", t5, out)
    else:
        print("skipped table5: run scripts/significance.py first")
    write("table6_percorruption.tex", table6(bench, protocol), out)
    write("numbers.tex", numbers(bench, released, protocol), out)


if __name__ == "__main__":
    main()
