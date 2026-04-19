# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

"""
Generate publication-quality figures for the modeq v0.3 release.

Produces 4 figures (PNG @ 160 DPI + PDF) from the JSON data in
``results/``:

    fig01_int4_valley.{png,pdf}
        The motivation figure: naive per-row absmax INT quantization
        across 7 models and 6 bit widths. Shows the Qwen2-7B INT4
        valley of death on a log y-axis.

    fig02_8model_headline.{png,pdf}
        v0.3 headline: symmetric vs asymmetric INT3 nsep+pchan on 8
        models, expressed as % of FP16 baseline PPL. Horizontal
        dashed line marks our internal usability threshold (< 5% of
        baseline).

    fig03_ablation.{png,pdf}
        Four-method INT3 ablation on 8 models (naive, asym-only,
        v0.2 sym, v0.3 asym). Only (D) v0.3 asym is below the 5 %
        threshold on all 8 models.

    fig04_bits_vs_quality.{png,pdf}
        Pareto-style scatter: bits/value on x, ΔPPL % of baseline
        on y. Visualizes why (D) is the INT3-tier recommendation
        and why INT4 can still be justified for maximum quality.

Run from the repo root:

    python3 figures/make_figures.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = Path(__file__).resolve().parent

# Colors are reused across figures so the story stays visually coherent.
C_NAIVE   = "#7a7a7a"   # industry baseline: gray
C_V02     = "#d08418"   # modeq v0.2 symmetric: amber
C_V03     = "#2c62c9"   # modeq v0.3 asymmetric: blue -- the hero
C_ASYMONLY = "#c94141"  # asymmetric without nsep: red (it's the trap)
C_INT4    = "#27864b"   # reference INT4: green
C_THRESH  = "#8a8a8a"   # < 5 % threshold line: medium gray

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":          10,
    "axes.titlesize":     12,
    "axes.labelsize":     11,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "figure.dpi":         110,
    "savefig.dpi":        160,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.15,
})


def save_both(fig, name):
    """Save as PNG and PDF so the same figure works in web and paper."""
    fig.savefig(OUT / f"{name}.png")
    fig.savefig(OUT / f"{name}.pdf")
    print(f"  wrote {name}.png + {name}.pdf")


def load_json(name):
    return json.load(open(RESULTS / name))


def load_eight_model_ablation():
    """Return the 8-model ablation summary with baseline_ppl filled in.

    The JSON's ``eight_model_ablation_summary`` rows carry the four
    method deltas but not the baseline PPL; we look up baselines from
    the ``phase18_prior`` and ``phase19_runs`` blocks that are stored
    alongside. Returns a list of dicts with keys:
        label, year, A, B, C, D, baseline_ppl
    """
    src = load_json("phase19_ablation_generalization.json")
    baselines = {}
    for row in src.get("phase18_prior", []):
        baselines[row["label"]] = row["baseline_ppl"]
    for label, run in src.get("phase19_runs", {}).items():
        baselines[label] = run["baseline_ppl"]

    rows = []
    for r in src["eight_model_ablation_summary"]:
        r2 = dict(r)
        r2["baseline_ppl"] = baselines[r["label"]]
        rows.append(r2)
    return rows


# ---------------------------------------------------------------------------
# Figure 1: INT4 valley of death across 7 models
# ---------------------------------------------------------------------------
#
# Source: paper V5 top-of-README bit-width sweep (naive per-row absmax INT).
# The Qwen2-7B INT4 spike is the anchor of the entire modeq story, so the
# figure uses a log y-axis and highlights Qwen2-7B with a bold red line.
#
# Data reproduced from README.md at commit time.

BITWIDTH_DATA = {
    #                     INT2    INT3     INT4    INT5   INT6   INT8
    "Qwen2-7B (Mode 1)":  [122.0,  12.0,  1382.07,   5.0,  12.0,   0.0],
    "Pythia-6.9B":        [164.0,  22.0,     7.0,   2.0,   0.5,   0.0],
    "Llama-3.1-8B":       [ 88.0,   0.4,    0.05,  0.01,   0.0,   0.0],
    "Qwen2.5-14B":        [  7.0,   0.9,    0.2,   0.03,  0.01,   0.0],
    "Mistral-7B-v0.1":    [  2.7,   0.09,   0.02,  0.0,   0.0,    0.0],
    "GPT-J-6B":           [  6.0,   0.3,    0.03,  0.0,   0.0,    0.0],
    "Phi-3-medium-4k":    [  5.0,   0.2,    0.03,  0.01,  0.0,    0.0],
}
BITS_NAIVE = [2, 3, 4, 5, 6, 8]


def fig01_int4_valley():
    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    for model, ppls in BITWIDTH_DATA.items():
        ppls_plot = [max(p, 0.005) for p in ppls]  # log-safe floor
        if model == "Qwen2-7B (Mode 1)":
            ax.plot(BITS_NAIVE, ppls_plot, "-o",
                    color="#c94141", linewidth=2.5, markersize=7,
                    label=model, zorder=5)
        else:
            ax.plot(BITS_NAIVE, ppls_plot, "-o",
                    color="#9a9a9a", alpha=0.55, linewidth=1.2,
                    markersize=4, label=model)

    ax.set_yscale("log")
    ax.set_xlabel("Bit width (naive per-row absmax INT quantization)")
    ax.set_ylabel(r"$\Delta\mathrm{PPL}$ on WikiText-2 (log scale)")
    ax.set_title("The INT4 valley of death:\n"
                 "naive INT KV quantization fails non-monotonically on Qwen2-7B",
                 loc="left")

    ax.axhline(1.0, color=C_THRESH, linestyle=":", linewidth=0.8)
    ax.text(8.1, 1.05, "1.0 PPL reference",
            color=C_THRESH, fontsize=8, va="bottom", ha="right")

    ax.set_xticks(BITS_NAIVE)
    ax.grid(True, which="major", axis="y", alpha=0.25)
    ax.set_xlim(1.7, 8.3)

    ax.annotate("+1382 ΔPPL at INT4\nMode 1 signature →",
                xy=(4, 1382.07), xytext=(5.8, 800),
                fontsize=9.5, color="#c94141", ha="right",
                arrowprops=dict(arrowstyle="->", color="#c94141",
                                lw=1.2, connectionstyle="arc3,rad=-0.15"))

    ax.legend(loc="lower left", frameon=True, framealpha=0.95,
              facecolor="white", edgecolor="#cccccc", ncol=1,
              bbox_to_anchor=(0.015, 0.015))

    save_both(fig, "fig01_int4_valley")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: 8-model v0.3 headline (sym vs asym INT3 as % of baseline)
# ---------------------------------------------------------------------------

def fig02_8model_headline():
    eight = load_eight_model_ablation()
    # Sort so Mode 1 (Qwen2-7B) is first, then by v0.2 severity
    # (dramatically drops = most visible improvement) descending, for a
    # natural left-to-right reading of "where v0.3 helps most".
    eight_sorted = sorted(
        eight,
        key=lambda r: (r["label"] != "Qwen2-7B", -r["C"] / r["baseline_ppl"]),
    )

    labels = [r["label"] for r in eight_sorted]
    sym_pct = [r["C"] / r["baseline_ppl"] * 100 for r in eight_sorted]
    asym_pct = [r["D"] / r["baseline_ppl"] * 100 for r in eight_sorted]
    reduction_pct = [
        (sym - asym) / sym * 100 if sym > 0 else 0
        for sym, asym in zip(sym_pct, asym_pct)
    ]

    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 5.2))
    bar_sym  = ax.bar(x - width / 2, sym_pct,  width, color=C_V02,
                      label="v0.2 nsep+pchan symmetric INT3")
    bar_asym = ax.bar(x + width / 2, asym_pct, width, color=C_V03,
                      label="v0.3 nsep+pchan asymmetric INT3")

    ax.axhline(5.0, color=C_THRESH, linestyle="--", linewidth=1.1)
    ax.text(len(labels) - 0.5, 5.2, "< 5 % baseline (usability threshold)",
            color=C_THRESH, fontsize=9, va="bottom", ha="right")

    for xi, (sym, asym, red) in enumerate(zip(sym_pct, asym_pct, reduction_pct)):
        ax.text(xi - width / 2, sym + 0.3, f"{sym:.1f}%",
                ha="center", va="bottom", fontsize=8.5, color="#555555")
        ax.text(xi + width / 2, asym + 0.3, f"{asym:.1f}%",
                ha="center", va="bottom", fontsize=8.5,
                color=C_V03, fontweight="bold")
        ax.annotate(f"−{red:.0f}%",
                    xy=(xi, max(sym, asym) + 1.3),
                    ha="center", fontsize=8, color="#2c62c9",
                    fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right")
    ax.set_ylabel(r"$\Delta\mathrm{PPL}$ on WikiText-2 (% of FP16 baseline)")
    ax.set_title("At identical 3.0 bits/value, asymmetric per-channel reduces\n"
                 "WikiText-2 ΔPPL by an average of 66 % across 8 models",
                 loc="left")
    ax.set_ylim(0, max(max(sym_pct), max(asym_pct)) * 1.20)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95,
              facecolor="white", edgecolor="#cccccc")

    save_both(fig, "fig02_8model_headline")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: Four-method ablation (each component is necessary)
# ---------------------------------------------------------------------------

def fig03_ablation():
    eight = load_eight_model_ablation()

    methods = [
        ("A", "(A) naive per-row sym",     C_NAIVE),
        ("B", "(B) asym, no nsep",         C_ASYMONLY),
        ("C", "(C) v0.2 sym nsep+pchan",   C_V02),
        ("D", "(D) v0.3 asym nsep+pchan",  C_V03),
    ]

    labels = [r["label"] for r in eight]
    x = np.arange(len(labels))
    width = 0.20

    fig, ax = plt.subplots(figsize=(11, 5.5))

    for i, (key, label, color) in enumerate(methods):
        offset = (i - 1.5) * width
        pct = [max(r[key] / r["baseline_ppl"] * 100, 0.02) for r in eight]
        # clamp negative (Qwen3.5 / Gemma B-anomaly) to a tiny positive for
        # log visualization; annotate the true sign separately.
        raw = [r[key] / r["baseline_ppl"] * 100 for r in eight]
        ax.bar(x + offset, pct, width, color=color, label=label,
               alpha=0.92, edgecolor="white", linewidth=0.5)
        # mark catastrophic (> 5 %) bars with a subtle tick
        for xi, (p, raw_p) in enumerate(zip(pct, raw)):
            if raw_p > 5.0:
                ax.text(xi + offset, p * 1.05, "✗" if raw_p > 20 else "!",
                        ha="center", va="bottom", fontsize=9,
                        color=color, fontweight="bold")
            elif raw_p < 0:
                ax.text(xi + offset, 0.06, "−",
                        ha="center", va="bottom", fontsize=10,
                        color=color, fontweight="bold")

    ax.axhline(5.0, color=C_THRESH, linestyle="--", linewidth=1.1)
    ax.text(len(labels) - 0.5, 5.3, "< 5 % usability threshold",
            color=C_THRESH, fontsize=9, va="bottom", ha="right")

    ax.set_yscale("log")
    ax.set_ylim(0.05, 800)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right")
    ax.set_ylabel(r"$\Delta\mathrm{PPL}$ on WikiText-2 (% of FP16 baseline, log)")
    ax.set_title("Four-method INT3 ablation across 8 models:\n"
                 "only (D) nsep+pchan asym stays below 5 % on every model",
                 loc="left")
    ax.grid(True, axis="y", alpha=0.25, which="both")
    ax.legend(loc="upper left", ncol=2, frameon=True, framealpha=0.95,
              facecolor="white", edgecolor="#cccccc")

    save_both(fig, "fig03_ablation")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: bits/value vs ΔPPL — the Pareto plot
# ---------------------------------------------------------------------------

def fig04_bits_vs_quality():
    # The four 3-bit methods from Phase 19 + the two INT4 references from
    # Phase 14 (Qwen2-7B and Llama-3.1-8B only; the other 6 models'
    # INT4 numbers are not in our Phase-14/16 JSONs as such). We still
    # plot the 3-bit cloud, since the core Pareto argument is at 3 bits.
    eight = load_eight_model_ablation()

    def pct(r, key):
        return r[key] / r["baseline_ppl"] * 100

    # Horizontally offset each method inside the 3-bit column so the
    # per-model scatter and method-median markers are readable instead
    # of piling on top of each other at x=3.
    methods = [
        ("A",  "(A) naive per-row sym",     2.72,  C_NAIVE),
        ("B",  "(B) asym, no nsep",         2.90,  C_ASYMONLY),
        ("C",  "(C) v0.2 sym nsep+pchan",   3.08,  C_V02),
        ("D",  "(D) v0.3 asym nsep+pchan",  3.26,  C_V03),
    ]

    fig, ax = plt.subplots(figsize=(9.5, 5.6))

    # Per-model scatter (small, semi-transparent), plus method-median dot
    for key, label, bits, color in methods:
        ys = [pct(r, key) for r in eight]
        ys_plot = [max(y, 0.02) for y in ys]
        xs = np.full(len(ys_plot), bits) + np.random.uniform(
            -0.025, 0.025, size=len(ys_plot)
        )
        ax.scatter(xs, ys_plot, s=30, color=color, alpha=0.40, edgecolors="none")
        median = float(np.median(ys))
        ax.scatter([bits], [max(median, 0.02)], s=150, color=color,
                   edgecolors="white", linewidth=1.8, zorder=5, label=label)

    # INT4 reference points (Phase 14 measured values for Qwen2-7B + Llama).
    int4_phase14 = {
        "Qwen2-7B":      (0.0544, 8.3693),
        "Llama-3.1-8B":  (0.0133, 7.3840),
    }
    for model, (delta, base) in int4_phase14.items():
        ax.scatter([4.0], [max(delta / base * 100, 0.02)], marker="s",
                   s=85, color=C_INT4, edgecolors="white", linewidth=1.0,
                   alpha=0.85, zorder=4)
    ax.scatter([], [], marker="s", s=85, color=C_INT4,
               edgecolors="white", linewidth=1.0, label="INT4 sym (reference)")

    # Visual bracket indicating the 3-bit column spans all four methods.
    ax.annotate("", xy=(2.70, 0.045), xytext=(3.28, 0.045),
                arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.7))
    ax.text(2.99, 0.035, "3 bits / value", ha="center", va="top",
            fontsize=9, color="#555555")
    ax.text(4.0, 0.035, "4 bits / value", ha="center", va="top",
            fontsize=9, color="#555555")

    ax.axhline(5.0, color=C_THRESH, linestyle="--", linewidth=1.1)
    ax.text(4.1, 5.3, "< 5 % (usability)", color=C_THRESH,
            fontsize=9, va="bottom", ha="right")

    ax.set_yscale("log")
    ax.set_ylim(0.03, 500)
    ax.set_xlim(2.55, 4.3)
    ax.set_xticks([])   # replaced by the bracket + labels above
    ax.set_ylabel(r"$\Delta\mathrm{PPL}$ on WikiText-2 (% of FP16 baseline, log)")
    ax.set_title("Bits vs quality: v0.3 asym is the INT3-tier Pareto choice\n"
                 "(small = per-model, large = median across 8 models)",
                 loc="left")
    ax.grid(True, axis="y", alpha=0.25, which="both")
    ax.legend(loc="upper right", frameon=True, framealpha=0.95,
              facecolor="white", edgecolor="#cccccc")

    save_both(fig, "fig04_bits_vs_quality")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print(f"Writing figures into {OUT}")
    fig01_int4_valley()
    fig02_8model_headline()
    fig03_ablation()
    fig04_bits_vs_quality()
    print("Done.")


if __name__ == "__main__":
    np.random.seed(0)   # reproducible jitter in fig04
    main()
