"""Paper figures generated ONLY from results files (plus a drawn architecture diagram).

    python scripts/significance.py     # first (Fig. held-out uses its confidence intervals)
    python scripts/make_figures.py     # -> figures/fig_*.pdf and .png

  fig_architecture   our model (drawn; parameter counts are the measured ones from Table 4 / the code)
  fig_severity       S-alpha vs severity for each corruption on COD10K (paper Fig. 3); the severity-5 scores
                     of SAM2-UNet, Ours (512) and SINet-V2 are printed at the end of their curves
  fig_tradeoff       accuracy under corruption and retention vs latency on COD10K (supplementary);
                     fig_tradeoff_main = panel (a) at column width (paper Fig. 4)
  fig_resolution     retention under local vs global corruptions, ordered by input resolution
  fig_heldout        gain of our robustness training per corruption, seen vs unseen (COD10K)
Image figures (teaser, qualitative grid) need the models and data: scripts/make_qualitative.py on the laptop.
Ours (512) is averaged over its seeds where all are complete (SEED_GROUPS in src/reporting.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.corruptions import load_protocol  # noqa: E402
from src.reporting import (GLOBAL, LOCAL, MODELS, RESULTS, SEED_GROUPS, aggregate_seeds, display,  # noqa: E402
                           efficiency, load_bench, robustness)

OUT = REPO / "figures"
COL1, COL2 = 3.5, 7.16  # IEEE column / page width in inches
STYLE = {  # colour-blind-safe (Okabe-Ito); specialists grey, SAM-based orange, ours blue
    "SINet-V2": dict(color="#7f7f7f", ls="-", marker="o"),
    "PFNet": dict(color="#b0b0b0", ls="--", marker="s"),
    "ZoomNet": dict(color="#404040", ls=":", marker="^"),
    "SAM2-UNet": dict(color="#E69F00", ls="-", marker="D"),
    "SAM-Adapter": dict(color="#D55E00", ls="--", marker="v"),
    "Ours-Full": dict(color="#0072B2", ls="-", marker="*", lw=2.0),
    "Ours-Full-512": dict(color="#56B4E9", ls="-", marker="P", lw=2.0),
}
BENCH_MODELS = list(STYLE)
NAMES = {"gaussian_noise": "Gaussian noise", "gaussian_blur": "Gaussian blur", "motion_blur": "Motion blur",
         "brightness": "Brightness", "contrast": "Contrast", "fog": "Fog", "jpeg_compression": "JPEG", "rain": "Rain"}

plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
                     "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "savefig.dpi": 300})


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote figures/{name}.pdf/.png")


def members(model: str, bench: pd.DataFrame, dataset: str, n_cond: int) -> list[str]:
    """Seed runs of `model` that have every condition on `dataset`."""
    out = []
    for m in SEED_GROUPS.get(model, [model]):
        if len(bench[(bench.model == m) & (bench.dataset == dataset)]) >= n_cond:
            out.append(m)
    return out


def curve(bench, model, dataset, corruption, n_cond) -> np.ndarray | None:
    ms = members(model, bench, dataset, n_cond)
    if not ms:
        return None
    ys = []
    for m in ms:
        sub = bench[(bench.model == m) & (bench.dataset == dataset)]
        clean = sub[sub.corruption == "clean"].Sm.iloc[0]
        sev = sub[sub.corruption == corruption].sort_values("severity").Sm.to_numpy()
        ys.append(np.concatenate([[clean], sev]))
    return np.mean(ys, axis=0)


# --------------------------------------------------------------------------------------- figures
HIGHLIGHT = ["SAM2-UNet", "Ours-Full-512", "SINet-V2"]  # models whose severity-5 score is printed in Fig. 3


def spread(values: list[float], gap: float) -> list[float]:
    """Label y positions: same order as `values`, at least `gap` apart, centred on the originals."""
    order = np.argsort(values)
    ys = np.array(values, float)[order]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + gap)
    ys -= (ys.mean() - np.mean(values))
    out = np.empty_like(ys)
    out[order] = ys
    return list(out)


def fig_severity(bench, protocol, dataset="COD10K"):
    n_cond = len(protocol["corruptions"]) * len(protocol["severities"]) + 1
    corrs = protocol["seen"] + protocol["unseen"]
    fig, axes = plt.subplots(2, 4, figsize=(COL2, 3.0), sharex=True, sharey=True)
    for ax, c in zip(axes.flat, corrs):
        ends = {}
        for m in BENCH_MODELS:
            y = curve(bench, m, dataset, c, n_cond)
            if y is None:
                continue
            st = STYLE[m]
            ax.plot(range(6), y, color=st["color"], ls=st["ls"], marker=st["marker"], ms=3.2, lw=st.get("lw", 1.1),
                    label=display(m))
            if m in HIGHLIGHT:
                ends[m] = y[-1]
        ax.set_title(f"{NAMES[c]} ({'seen' if c in protocol['seen'] else 'unseen'})")
        ax.set_xticks(range(6), ["0", "1", "2", "3", "4", "5"])
        ax.set_xlim(-0.3, 6.6)
        names = list(ends)
        for m, yl in zip(names, spread([ends[m] for m in names], gap=0.017)):
            ax.annotate(f"{ends[m]:.3f}", (5, ends[m]), xytext=(5.25, yl), textcoords="data", fontsize=5.8,
                        color=STYLE[m]["color"], fontweight="bold", va="center", ha="left",
                        arrowprops=dict(arrowstyle="-", color=STYLE[m]["color"], lw=0.5, shrinkA=0, shrinkB=1))
    for ax in axes[1]:
        ax.set_xlabel("Severity (0 = clean)")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$S_\alpha$")
    h, lab = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=len(lab), frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "fig_severity")


# label placement per panel (points offset, alignment) so names don't collide with neighbours
LABEL_OFFSETS = (
    {"Ours-Full-512": (-6, 6, "right"), "SINet-V2": (7, -8, "left"), "Ours-Full": (6, 6, "left"),
     "SAM2-UNet": (9, 0, "left"), "SAM-Adapter": (-10, 0, "right")},
    {"Ours-Full-512": (-6, 6, "right"), "SINet-V2": (8, -6, "left"), "Ours-Full": (7, -9, "left"),
     "PFNet": (-8, 6, "right"), "SAM2-UNet": (9, 0, "left"), "SAM-Adapter": (-10, 0, "right")},
)


def fig_tradeoff(bench, protocol, dataset="COD10K"):
    eff = efficiency()
    pts = []
    for m in BENCH_MODELS:
        mean, _, n = aggregate_seeds(lambda mm, d: robustness(bench, mm, d, protocol), m, dataset)
        if mean and m in eff.index:
            pts.append((m, float(eff.loc[m, "latency_ms_median"]), mean["mcorr_Sm"], mean["ret_Sm"],
                        float(eff.loc[m, "params_M"])))
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.5))
    for ax, idx, ylab in ((axes[0], 2, r"Mean $S_\alpha$ under corruption (mCorr)"),
                          (axes[1], 3, r"Retention (mCorr / clean $S_\alpha$)")):
        for m, lat, mc, ret, par in pts:
            st = STYLE[m]
            y = (mc, ret)[idx - 2]
            ax.scatter(lat, y, s=18 + 9 * np.sqrt(par), color=st["color"], marker=st["marker"], edgecolor="black",
                       linewidth=0.4, zorder=3)
            dx, dy, ha = LABEL_OFFSETS[idx - 2].get(m, (6, 3, "left"))
            ax.annotate(display(m), (lat, y), xytext=(dx, dy), textcoords="offset points", fontsize=6.5, ha=ha)
        ax.set_xscale("log")
        ax.margins(x=0.12, y=0.12)
        ax.set_xlabel("Latency per image (ms, log scale), RTX 3050 Laptop")
        ax.set_ylabel(ylab)
    # Pareto front for accuracy under corruption vs latency
    front, best = [], -1.0
    for m, lat, mc, _, _ in sorted(pts, key=lambda p: p[1]):
        if mc > best:
            front.append((lat, mc))
            best = mc
    if len(front) > 1:
        axes[0].plot(*zip(*front), color="#0072B2", lw=0.8, ls="--", zorder=1, label="Pareto front")
        axes[0].legend(loc="lower right", frameon=False)
    axes[0].set_title(f"(a) Accuracy under corruption, {dataset}")
    axes[1].set_title(f"(b) Robustness, {dataset} (marker area $\\propto$ $\\sqrt{{\\mathrm{{params}}}}$)")
    fig.tight_layout()
    save(fig, "fig_tradeoff")
    # single-column version for the main paper: panel (a) only
    fig, ax = plt.subplots(figsize=(COL1, 2.4))
    for m, lat, mc, _, par in pts:
        st = STYLE[m]
        ax.scatter(lat, mc, s=18 + 9 * np.sqrt(par), color=st["color"], marker=st["marker"], edgecolor="black",
                   linewidth=0.4, zorder=3)
        dx, dy, ha = LABEL_OFFSETS[0].get(m, (6, 3, "left"))
        ax.annotate(display(m), (lat, mc), xytext=(dx, dy), textcoords="offset points", fontsize=6.5, ha=ha)
    if len(front) > 1:
        ax.plot(*zip(*front), color="#0072B2", lw=0.8, ls="--", zorder=1, label="Pareto front")
        ax.legend(loc="lower right", frameon=False)
    ax.set_xscale("log")
    ax.margins(x=0.15, y=0.12)
    ax.set_xlabel("Latency per image (ms, log scale)")
    ax.set_ylabel(r"Mean $S_\alpha$ under corruption, " + dataset)
    fig.tight_layout()
    save(fig, "fig_tradeoff_main")


def fig_resolution(bench, protocol, dataset="COD10K"):
    rows = []
    for m in BENCH_MODELS:
        mean, _, _ = aggregate_seeds(lambda mm, d: robustness(bench, mm, d, protocol), m, dataset)
        if mean:
            res = int(MODELS[m][2].rstrip("*"))
            rows.append((m, res, mean["ret_local"], mean["ret_global"]))
    rows.sort(key=lambda r: (r[1], r[2]))
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(COL1, 2.9))
    ax.bar(x - 0.2, [r[2] for r in rows], 0.38, color=[STYLE[r[0]]["color"] for r in rows], edgecolor="black",
           linewidth=0.4, label="Local (noise, blur, JPEG, rain)")
    ax.bar(x + 0.2, [r[3] for r in rows], 0.38, color="white", edgecolor=[STYLE[r[0]]["color"] for r in rows],
           linewidth=1.0, hatch="////", label="Global (brightness, contrast, fog)")
    lbl = [display(r[0]) if r[0].startswith("Ours") else f"{display(r[0])} ({MODELS[r[0]][2].rstrip('*')})" for r in rows]
    ax.set_xticks(x, lbl, rotation=35, ha="right",
                  fontsize=6.5)
    ax.set_ylim(min(r[2] for r in rows) - 0.01, 1.003)
    ax.set_ylabel(r"Retention ($S_\alpha$), " + dataset)
    ax.set_xlabel("Model (input side in px), ordered by input resolution", fontsize=7)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, frameon=False, fontsize=6.5)
    fig.tight_layout()
    save(fig, "fig_resolution")


def fig_heldout(bench, protocol, dataset="COD10K"):
    sig_p = RESULTS / "significance.csv"
    sig = pd.read_csv(sig_p) if sig_p.exists() else pd.DataFrame()
    base = robustness(bench, "Ours-Clean", dataset, protocol)
    aug = robustness(bench, "Ours-Aug", dataset, protocol)
    full = robustness(bench, "Ours-Full", dataset, protocol)
    if not (base and aug and full):
        print("skipped fig_heldout: Ours-Clean/Aug/Full not complete on", dataset)
        return
    corrs = protocol["seen"] + protocol["unseen"]
    x = np.arange(len(corrs))
    d_aug = [aug[f"mcorr_{c}"] - base[f"mcorr_{c}"] for c in corrs]
    d_full = [full[f"mcorr_{c}"] - base[f"mcorr_{c}"] for c in corrs]
    err = None
    if len(sig):
        lo, hi = [], []
        for c, d in zip(corrs, d_full):
            r = sig[(sig.a == "Ours-Full") & (sig.b == "Ours-Clean") & (sig.group == c) & (sig.dataset == dataset)]
            lo.append(d - r.ci_lo.iloc[0] if len(r) else 0)
            hi.append(r.ci_hi.iloc[0] - d if len(r) else 0)
        err = [lo, hi]
    fig, ax = plt.subplots(figsize=(COL1, 2.3))
    ax.axvspan(len(protocol["seen"]) - 0.5, len(corrs) - 0.5, color="#f2f2f2", zorder=0)
    ax.bar(x - 0.2, d_aug, 0.38, color="#56B4E9", edgecolor="black", linewidth=0.4, label="+ augmentation (Ours-Aug)")
    ax.bar(x + 0.2, d_full, 0.38, color="#0072B2", edgecolor="black", linewidth=0.4, yerr=err, capsize=1.5,
           error_kw=dict(lw=0.6), label="+ consistency (Ours, 1024), 95% CI")
    ax.axhline(0, color="black", lw=0.6)
    short = {"gaussian_noise": "G. noise", "gaussian_blur": "G. blur", "brightness": "Bright.", "jpeg_compression": "JPEG",
             "motion_blur": "M. blur", "contrast": "Contrast", "fog": "Fog", "rain": "Rain"}
    ax.set_xticks(x, [short[c] for c in corrs], fontsize=6.5, rotation=30, ha="right")
    ax.set_ylabel(r"$\Delta$ mCorr $S_\alpha$ vs. Ours-Clean")
    ax.text(len(protocol["seen"]) / 2 - 0.5, ax.get_ylim()[1] * 0.95, "seen in training", ha="center", va="top", fontsize=6.5)
    ax.text(len(protocol["seen"]) + len(protocol["unseen"]) / 2 - 0.5, ax.get_ylim()[1] * 0.95, "unseen (held out)",
            ha="center", va="top", fontsize=6.5)
    ax.legend(loc="center right", frameon=False, fontsize=6)
    ax.set_title(dataset)
    fig.tight_layout()
    save(fig, "fig_heldout")


def fig_architecture():
    """Drawn diagram; the numbers are the measured parameter counts (src/models/robust_adapter.py)."""
    fig, ax = plt.subplots(figsize=(COL2, 2.3))
    ax.set_xlim(0, 104)
    ax.set_ylim(0, 32)
    ax.axis("off")
    trained, frozen = "#0072B2", "#7f7f7f"

    def box(x, y, w, h, title, lines, on=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", linewidth=1.3 if on else 0.8,
                                    edgecolor=trained if on else frozen, facecolor="#e6f1f8" if on else "white"))
        ax.text(x + 1, y + h - 1.6, title, fontsize=7.5, fontweight="bold", va="top")
        for i, ln in enumerate(lines):
            ax.text(x + 1, y + h - 5.0 - 3.0 * i, ln, fontsize=6.5, va="top")

    def arrow(x0, y0, x1, y1, ls="-"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, lw=0.8, color="#404040", ls=ls))

    box(1, 16, 12, 14, "Input", ["RGB image", "resized to", "1024 (or 512)"])
    box(16, 16, 26, 14, "TinyViT encoder", ["frozen, 6.07 M parameters", "conv stage + 10 blocks"])
    ax.add_patch(FancyBboxPatch((17.2, 16.7), 23.6, 2.8, boxstyle="round,pad=0.2,rounding_size=0.8", linewidth=1.3,
                                edgecolor=trained, facecolor="#e6f1f8"))
    ax.text(18, 18.1, "+ adapter per block (0.20 M)", fontsize=6.5, va="center")
    box(46, 16, 22, 14, "SAM mask decoder", ["two-way transformer,", "4x upsampling", "3.51 M, trained"], on=True)
    box(46, 1, 22, 11, "Learned prompts", ["4 tokens + dense prompt", "replace user clicks"], on=True)
    box(72, 16, 14, 14, "Output", ["camouflage map,", "sigmoid, resized", "to original size"])
    arrow(13.6, 23, 15.6, 23)
    arrow(42.6, 23, 45.6, 23)
    arrow(57, 12.4, 57, 15.6)
    arrow(68.6, 23, 71.6, 23)
    ax.text(87.5, 29, "Training (Ours):", fontsize=7, fontweight="bold", va="top")
    for i, ln in enumerate(["clean + corrupted view", "(seen corruptions only)", "structure loss on both", "+ consistency (MSE,", "clean = fixed target)"]):
        ax.text(87.5, 25.6 - 3.0 * i, ln, fontsize=6.3, va="top")
    ax.add_patch(FancyBboxPatch((1, 1), 3, 2.4, boxstyle="round,pad=0.2", facecolor="#e6f1f8", edgecolor=trained, lw=1.3))
    ax.text(5.2, 2.2, "trained (3.71 M)", fontsize=6.5, va="center")
    ax.add_patch(FancyBboxPatch((20, 1), 3, 2.4, boxstyle="round,pad=0.2", facecolor="white", edgecolor=frozen, lw=0.8))
    ax.text(24.2, 2.2, "frozen (6.62 M)", fontsize=6.5, va="center")
    save(fig, "fig_architecture")


def main():
    bench, protocol = load_bench(), load_protocol()
    fig_architecture()
    fig_severity(bench, protocol)
    fig_tradeoff(bench, protocol)
    fig_resolution(bench, protocol)
    fig_heldout(bench, protocol)


if __name__ == "__main__":
    main()
