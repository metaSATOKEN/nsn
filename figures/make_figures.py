# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

"""
Generate publication-quality figures for the modeq v0.3 release.

Produces 4 figures (PNG @ 160 DPI + PDF) from the JSON data in
``results/``:

    fig01_int4_valley.{png,pdf}
        The motivation figure: naive per-row absmax INT quantization
        across 7 models and 6 bit widths. Shows the non-monotonic
        +1382 ΔPPL spike at INT4 on Qwen2-7B -- an n=1 anomaly in
        our sample, not a property shared with the other 6 models.

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

    fig05_norm_dispersion.{png,pdf}
        Phase 23 mechanism figure: two side-by-side scatter panels
        showing that L0 V-cache per-token L2 norm CV predicts
        variant-(B) failure at Spearman rho ~= 0.88, while the
        corresponding K-cache CV does not (rho ~= 0.29). Directly
        supports §5.3 of the paper.

    fig06_nih_longctx.{png,pdf}
        Phase 24(b-d) long-context multi-needle NIH pooled result:
        left panel shows pass / wrong-code / hallucinate composition
        as stacked bars at 8K and 32K across 3 methods (FP16 / sym /
        asym); right panel shows per-model 32K pass rates so the
        pooled story is visibly consistent across Llama-3.1-8B,
        Qwen2-7B, and Qwen2.5-14B. Fisher exact p-values are
        annotated on the 32K bars in the left panel. Supports §5.7.

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
    ax.set_title("Naive INT KV quantization scales monotonically on 6 of 7 tested models;\n"
                 "Qwen2-7B is the only exception in our sample (n = 1 anomaly)",
                 loc="left")

    ax.axhline(1.0, color=C_THRESH, linestyle=":", linewidth=0.8)
    ax.text(8.1, 1.05, "1.0 PPL reference",
            color=C_THRESH, fontsize=8, va="bottom", ha="right")

    ax.set_xticks(BITS_NAIVE)
    ax.grid(True, which="major", axis="y", alpha=0.25)
    ax.set_xlim(1.7, 8.3)

    ax.annotate("+1382 ΔPPL at INT4\nQwen2-7B anomaly (1/27) →",
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
# Figure 5: V-CV vs (B) failure (rho ~= 0.88) vs K-CV vs (B) failure (rho ~= 0.29)
# ---------------------------------------------------------------------------
#
# Phase 23 (results/phase23_norm_dispersion.json) measures per-token L2
# norm coefficient of variation on L0 K and V caches for each of the 8
# benchmark models and pairs it with variant-(B)'s ΔPPL from Table 2.
# This figure puts V-CV (left) and K-CV (right) next to each other on
# a shared y-axis so the reader sees the V-CV monotone ramp and the
# K-CV scatter in one glance. The title carries both Spearman rho
# figures.

def _spearman_rho(xs, ys):
    """Spearman's rho without scipy (small n=8, ties unlikely)."""
    def rank(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        r = [0] * len(values)
        for rk, idx in enumerate(order, start=1):
            r[idx] = rk
        return r
    rx = rank(xs)
    ry = rank(ys)
    n = len(xs)
    d2 = sum((a - b) ** 2 for a, b in zip(rx, ry))
    return 1 - 6 * d2 / (n * (n * n - 1))


def fig05_norm_dispersion():
    src = load_json("phase23_norm_dispersion.json")
    rows = src["summary_table"]

    # Stable ordering: ascending by V-CV, which is roughly also the
    # "safe -> catastrophic" axis for (B) under our hypothesis.
    rows_v = sorted(rows, key=lambda r: r["L0_V_CV_mean"])

    v_cv  = [r["L0_V_CV_mean"] for r in rows_v]
    k_cv  = [r["L0_K_CV_mean"] for r in rows_v]
    b_pct = [r["B_variant_pct_of_base"] for r in rows_v]
    models = [r["model"] for r in rows_v]

    rho_v = _spearman_rho(v_cv,  b_pct)
    rho_k = _spearman_rho(k_cv,  b_pct)

    # Color-code each model by its (B) outcome so the reader can see
    # catastrophic vs OK vs B-negative at a glance without reading labels.
    def color_for(pct):
        if pct >= 100: return "#c94141"       # catastrophic: red
        if pct >=   5: return "#d08418"       # mild / borderline: amber
        if pct >=   0: return "#4a8c4a"       # OK: green
        return "#2c62c9"                       # negative (B-anomaly): blue

    colors = [color_for(p) for p in b_pct]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 5.3), sharey=True)

    # Shared y-axis: symlog so -2..+148 fits and the 0-line is visible.
    for ax in (axL, axR):
        ax.set_yscale("symlog", linthresh=5.0)
        ax.axhline(5.0, color=C_THRESH, linestyle="--", linewidth=1.0)
        ax.axhline(0.0, color="#c0c0c0", linestyle="-", linewidth=0.7)
        ax.axhline(100.0, color="#c94141", linestyle=":", linewidth=0.8,
                   alpha=0.55)
        ax.grid(True, axis="y", which="major", alpha=0.25)
        ax.set_ylim(-4, 400)

    # --- Left panel: V-CV (strong correlation) -----------------------------
    axL.scatter(v_cv, b_pct, s=110, c=colors, edgecolors="white",
                linewidth=1.3, zorder=5)
    # Least-squares line as a visual monotone guide (not a claim of linearity).
    zL = np.polyfit(v_cv, b_pct, 1)
    xx = np.linspace(-0.02, max(v_cv) + 0.05, 50)
    axL.plot(xx, np.polyval(zL, xx), color="#888888", linewidth=1.0,
             linestyle="--", alpha=0.7, zorder=2)

    axL.set_xlabel("L0 V-cache per-token L2 norm CV  ($\\sigma / \\mu$)")
    axL.set_ylabel("Variant-(B) $\\Delta\\mathrm{PPL}$  "
                   "(% of FP16 baseline, symlog)")
    axL.set_title(f"V-CV predicts (B) failure   "
                  f"Spearman $\\rho \\approx {rho_v:.2f}$",
                  loc="left", color="#2c62c9")
    axL.set_xlim(-0.03, 0.70)

    # --- Right panel: K-CV (weak correlation) ------------------------------
    axR.scatter(k_cv, b_pct, s=110, c=colors, edgecolors="white",
                linewidth=1.3, zorder=5)
    zR = np.polyfit(k_cv, b_pct, 1)
    xx2 = np.linspace(-0.02, max(k_cv) + 0.05, 50)
    axR.plot(xx2, np.polyval(zR, xx2), color="#888888", linewidth=1.0,
             linestyle="--", alpha=0.7, zorder=2)

    axR.set_xlabel("L0 K-cache per-token L2 norm CV  ($\\sigma / \\mu$)")
    axR.set_title(f"K-CV does not   "
                  f"Spearman $\\rho \\approx {rho_k:.2f}$",
                  loc="left", color="#c94141")
    axR.set_xlim(-0.03, 0.55)

    # Per-point labels. Keep them short and offset so they don't overlap.
    # We hand-tune offsets for the 8 known models to avoid collisions.
    offsets_v = {  # (dx, dy_symlog_units, ha)
        "Gemma-4-E4B":      ( 0.03, -1.8, "left"),
        "Qwen3.5-9B":       ( 0.03, -1.8, "left"),
        "GPT-J-6B":         (-0.02,  2.0, "right"),
        "Pythia-6.9B":      ( 0.03, -2.5, "left"),
        "Qwen2-7B":         ( 0.03,  2.0, "left"),
        "Mistral-7B-v0.1":  (-0.02, 45,   "right"),
        "Llama-3.1-8B":     (-0.02, 45,   "right"),
        "Qwen2.5-14B":      ( 0.03, -25,  "left"),
    }
    offsets_k = {
        "Gemma-4-E4B":      ( 0.03, -1.8, "left"),
        "Qwen3.5-9B":       ( 0.03,  2.0, "left"),
        "Qwen2.5-14B":      ( 0.03, 30,   "left"),
        "Llama-3.1-8B":     ( 0.03, 45,   "left"),
        "Pythia-6.9B":      (-0.02, -2.5, "right"),
        "GPT-J-6B":         ( 0.03,  2.0, "left"),
        "Mistral-7B-v0.1":  (-0.02, 45,   "right"),
        "Qwen2-7B":         (-0.02,  2.0, "right"),
    }

    for (ax, cv_arr, offsets) in [
        (axL, v_cv, offsets_v),
        (axR, k_cv, offsets_k),
    ]:
        for m, cv, pct in zip(models, cv_arr, b_pct):
            dx, dy, ha = offsets[m]
            ax.annotate(
                m, xy=(cv, pct), xytext=(cv + dx, pct + dy),
                fontsize=7.8, color="#333333", ha=ha, va="center",
            )

    # Shared threshold labels (on the left panel only to avoid clutter).
    axL.text(0.68, 5.3, "5 % usability", ha="right", va="bottom",
             fontsize=8, color=C_THRESH)
    axL.text(0.68, 105, "100 % baseline (catastrophic)",
             ha="right", va="bottom", fontsize=8, color="#c94141", alpha=0.8)

    # Category legend, shared.
    legend_handles = [
        mpatches.Patch(color="#c94141", label="(B) catastrophic (> 100 % base)"),
        mpatches.Patch(color="#d08418", label="(B) mild (5–100 % base)"),
        mpatches.Patch(color="#4a8c4a", label="(B) OK (< 5 % base)"),
        mpatches.Patch(color="#2c62c9", label="(B) negative ΔPPL (B-anomaly)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               frameon=True, framealpha=0.95, edgecolor="#cccccc",
               bbox_to_anchor=(0.5, -0.01))

    fig.suptitle("Why `nsep` is necessary: V-cache per-token L2 norm "
                 "dispersion predicts variant-(B) failure; K-cache does not",
                 fontsize=12, y=1.00)

    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    save_both(fig, "fig05_norm_dispersion")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6: Long-context NIH pooled result (Phases 24b / 24c / 24d)
# ---------------------------------------------------------------------------
#
# Two panels:
#   A (left, wider) : pooled stacked bars -- 6 bars (2 contexts x
#                     3 methods), stacked by pass / wrong-code /
#                     hallucinate. Fisher exact p-values annotated
#                     on the 32K bars.
#   B (right, narrow): per-model 32K pass rates as grouped bars,
#                      so the reader sees the pooled story replicates
#                      on each of the 3 individual models.
#
# Outcome colors are CHOSEN to match common visual conventions
# (green = correct, amber = wrong-but-plausible, red = hallucinate)
# and are independent of the method colors reused throughout fig01-05.

C_PASS      = "#3c8c4a"   # green: correct retrieval
C_WRONG     = "#d08418"   # amber: a different real needle retrieved
C_HALLUC    = "#c94141"   # red: characters not in any needle (or fused)


def _fisher_p_label(p):
    """Format a Fisher exact p-value for axis annotation."""
    if p >= 0.1:
        return f"p = {p:.2f} (n.s.)"
    if p >= 0.01:
        return f"p = {p:.3f}"
    if p >= 1e-4:
        return f"p = {p:.4f}"
    # very small p
    exp = int(np.floor(np.log10(p)))
    mant = p / (10 ** exp)
    return f"p ≈ {mant:.1f} × 10$^{{{exp}}}$"


def fig06_nih_longctx():
    # --- Load phase24b/c/d JSONs and pool -----------------------------------
    sources = [
        ("phase24b_nih_multineedle.json",        "Llama-3.1-8B"),
        ("phase24c_nih_multineedle_qwen2.json",  "Qwen2-7B"),
        ("phase24d_nih_multineedle_qwen25_14b.json", "Qwen2.5-14B"),
    ]
    per_model = {}
    for fname, label in sources:
        per_model[label] = load_json(fname)["results"]

    methods = [
        ("FP16 baseline",      C_NAIVE),
        ("sym INT3 (v0.2)",    C_V02),
        ("asym INT3 (v0.3)",   C_V03),
    ]
    method_labels_short = {
        "FP16 baseline":      "FP16",
        "sym INT3 (v0.2)":    "(C) sym",
        "asym INT3 (v0.3)":   "(D) asym",
    }

    # Pool pass/wrong/halluc across the 3 models at each (ctx, method).
    def pooled(ctx, method_name):
        p = w = h = 0
        for label in per_model:
            r = per_model[label][f"ctx{ctx}__{method_name}"]
            p += r["n_pass"]; w += r["n_wrong"]; h += r["n_halluc"]
        return p, w, h

    # Compute Fisher exact p-values (pooled sym / asym vs FP16 at each ctx).
    # We use a self-contained scipy-free implementation: for the sample
    # sizes here the two-sided p-value equals 2 * min(left, right) tail of
    # the hypergeometric, with a 1.0 cap. scipy's version is numerically
    # identical on integer counts of this magnitude.
    from math import comb

    def fisher_exact_2sided(a, b, c, d):
        n1 = a + b   # row 1 total
        n2 = c + d   # row 2 total
        k  = a + c   # col 1 total
        n  = n1 + n2
        # observed odds ratio direction -> compute tail probabilities
        p_obs = comb(n1, a) * comb(n2, c) / comb(n, k)
        ps = []
        for i in range(max(0, k - n2), min(n1, k) + 1):
            ps.append(comb(n1, i) * comb(n2, k - i) / comb(n, k))
        # two-sided: sum of all outcomes at least as extreme (<= p_obs)
        p = sum(pi for pi in ps if pi <= p_obs + 1e-15)
        return min(p, 1.0)

    # --- Build data for Panel A (stacked bars) ------------------------------
    contexts = [8192, 32768]
    stacked_data = {}
    for ctx in contexts:
        for mname, _ in methods:
            stacked_data[(ctx, mname)] = pooled(ctx, mname)

    # Fisher exact p-values at 32K for sym vs FP16, asym vs FP16
    fp16_p, fp16_w, fp16_h = stacked_data[(32768, "FP16 baseline")]
    sym_p,  sym_w,  sym_h  = stacked_data[(32768, "sym INT3 (v0.2)")]
    asym_p, asym_w, asym_h = stacked_data[(32768, "asym INT3 (v0.3)")]
    n_cell = 30  # 3 models x 10 trials
    p_sym_vs_fp16  = fisher_exact_2sided(fp16_p, n_cell - fp16_p,
                                         sym_p,  n_cell - sym_p)
    p_asym_vs_fp16 = fisher_exact_2sided(fp16_p, n_cell - fp16_p,
                                         asym_p, n_cell - asym_p)
    p_asym_vs_sym  = fisher_exact_2sided(sym_p,  n_cell - sym_p,
                                         asym_p, n_cell - asym_p)

    # --- Layout -------------------------------------------------------------
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(12, 5.6),
        gridspec_kw={"width_ratios": [1.45, 1.0]},
    )

    # --- Panel A: Stacked bars at 8K and 32K --------------------------------
    # Within each context block, 3 adjacent bars (FP16 / sym / asym).
    # Bars are drawn at numeric x positions.
    x_positions = []
    x_labels    = []
    bar_width   = 0.62
    gap_between_ctx = 1.0
    pos = 0.0
    ctx_centers = []
    for ctx in contexts:
        ctx_start = pos
        for mname, _ in methods:
            x_positions.append(pos)
            x_labels.append(method_labels_short[mname])
            pos += 1.0
        ctx_centers.append((ctx_start + pos - 1.0) / 2)
        pos += gap_between_ctx

    bars_pass = []; bars_wrong = []; bars_halluc = []
    method_color_per_bar = []
    for i, (ctx, mname) in enumerate(
        [(c, m[0]) for c in contexts for m in methods]
    ):
        p, w, h = stacked_data[(ctx, mname)]
        bars_pass.append(p); bars_wrong.append(w); bars_halluc.append(h)
        method_color_per_bar.append(dict(methods)[mname])

    x = np.array(x_positions)
    b1 = axL.bar(x, bars_pass,   bar_width, color=C_PASS,   label="pass",
                 edgecolor="white", linewidth=1.0)
    b2 = axL.bar(x, bars_wrong,  bar_width, bottom=bars_pass, color=C_WRONG,
                 label="wrong-code", edgecolor="white", linewidth=1.0)
    b3 = axL.bar(x, bars_halluc, bar_width,
                 bottom=[p + w for p, w in zip(bars_pass, bars_wrong)],
                 color=C_HALLUC, label="hallucinate",
                 edgecolor="white", linewidth=1.0)

    # Count labels inside bar segments (only if the segment is big enough).
    for xi, (p, w, h) in enumerate(zip(bars_pass, bars_wrong, bars_halluc)):
        if p >= 2:
            axL.text(x[xi], p / 2, f"{p}", ha="center", va="center",
                     fontsize=10, color="white", fontweight="bold")
        if w >= 2:
            axL.text(x[xi], p + w / 2, f"{w}", ha="center", va="center",
                     fontsize=9, color="white", fontweight="bold")
        if h >= 2:
            axL.text(x[xi], p + w + h / 2, f"{h}", ha="center", va="center",
                     fontsize=9, color="white", fontweight="bold")

    # Fisher exact annotations on the 32K bars (second context block).
    # FP16 32K bar is x_positions[3], sym is x_positions[4], asym is [5].
    y_annot = 32
    # Horizontal brackets for the two key comparisons.
    # sym vs FP16
    axL.annotate("", xy=(x_positions[3], y_annot + 1.3),
                 xytext=(x_positions[4], y_annot + 1.3),
                 arrowprops=dict(arrowstyle="-", color="#c94141", lw=1.2))
    axL.text((x_positions[3] + x_positions[4]) / 2, y_annot + 2.0,
             f"sym vs FP16: {_fisher_p_label(p_sym_vs_fp16)}",
             ha="center", va="bottom", fontsize=8.5,
             color="#c94141", fontweight="bold")
    # asym vs FP16
    axL.annotate("", xy=(x_positions[3], y_annot + 5.0),
                 xytext=(x_positions[5], y_annot + 5.0),
                 arrowprops=dict(arrowstyle="-", color="#3c8c4a", lw=1.2))
    axL.text((x_positions[3] + x_positions[5]) / 2, y_annot + 5.7,
             f"asym vs FP16: {_fisher_p_label(p_asym_vs_fp16)}",
             ha="center", va="bottom", fontsize=8.5,
             color="#3c8c4a", fontweight="bold")
    # asym vs sym
    axL.annotate("", xy=(x_positions[4], y_annot + 9.2),
                 xytext=(x_positions[5], y_annot + 9.2),
                 arrowprops=dict(arrowstyle="-", color="#2c62c9", lw=1.2))
    axL.text((x_positions[4] + x_positions[5]) / 2, y_annot + 9.9,
             f"asym vs sym: {_fisher_p_label(p_asym_vs_sym)}",
             ha="center", va="bottom", fontsize=8.5,
             color="#2c62c9", fontweight="bold")

    # Context labels beneath
    axL.set_xticks(x_positions)
    axL.set_xticklabels(x_labels, fontsize=9)
    for c, ctx in zip(ctx_centers, contexts):
        axL.text(c, -4.0, f"{ctx // 1024} K context",
                 ha="center", va="top", fontsize=10.5,
                 color="#333333", fontweight="bold")

    axL.set_ylabel("Trial count (pooled, $n = 30$)")
    axL.set_ylim(0, 50)
    axL.set_yticks([0, 10, 20, 30])
    axL.axhline(30, color="#aaaaaa", linestyle=":", linewidth=0.7)
    axL.grid(True, axis="y", alpha=0.25)
    axL.legend(loc="upper left", frameon=True, framealpha=0.95,
               facecolor="white", edgecolor="#cccccc",
               bbox_to_anchor=(0.015, 0.99), fontsize=9)
    axL.set_title("Pooled multi-needle NIH outcomes (3 models, $n = 30$)",
                  loc="left", fontsize=11)

    # --- Panel B: Per-model 32K pass rates (grouped bars) -------------------
    models = ["Llama-3.1-8B", "Qwen2-7B", "Qwen2.5-14B"]
    width_b = 0.26
    x_b = np.arange(len(models))

    for i, (mname, color) in enumerate(methods):
        ys = []
        for label in models:
            r = per_model[label][f"ctx32768__{mname}"]
            ys.append(r["n_pass"] / r["n_total"] * 100)
        offset = (i - 1) * width_b
        bars = axR.bar(x_b + offset, ys, width_b, color=color,
                       label=method_labels_short[mname],
                       edgecolor="white", linewidth=0.8)
        for xi, y in zip(x_b + offset, ys):
            axR.text(xi, y + 1.3, f"{y:.0f}%",
                     ha="center", va="bottom", fontsize=8.5,
                     color=color, fontweight="bold")

    axR.set_xticks(x_b)
    axR.set_xticklabels(
        [m.replace("-", "-\n", 1) for m in models],
        fontsize=8.5,
    )
    axR.set_ylabel("Pass rate at 32 K (%)")
    axR.set_ylim(0, 118)
    axR.set_yticks([0, 25, 50, 75, 100])
    axR.axhline(100, color="#aaaaaa", linestyle=":", linewidth=0.7)
    axR.grid(True, axis="y", alpha=0.25)
    axR.legend(loc="lower left", frameon=True, framealpha=0.95,
               facecolor="white", edgecolor="#cccccc", fontsize=8.5,
               bbox_to_anchor=(0.015, 0.015))
    axR.set_title("32 K pass rate per model  "
                  "(asym ≥ sym on 3 / 3)",
                  loc="left", fontsize=11)

    fig.suptitle(
        "v0.3 asym INT3 shows no significant 32K NIH lead-key retrieval "
        "loss vs FP16; v0.2 sym INT3 drops to 50 % with bit-mixing hallucinations",
        fontsize=11.5, y=1.00,
    )
    fig.tight_layout(rect=(0, 0.01, 1, 0.96))
    save_both(fig, "fig06_nih_longctx")
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
    fig05_norm_dispersion()
    fig06_nih_longctx()
    print("Done.")


if __name__ == "__main__":
    np.random.seed(0)   # reproducible jitter in fig04
    main()
