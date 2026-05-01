# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0
"""Common matplotlib style for Paper C figures.

Usage from each fig*.py:

    from _style import setup, save, OKABE_ITO, PROFILE_COLORS

    setup()
    fig, ax = ...
    save(fig, "fig01_R_distribution")

Saves both .pdf (vector) and .png (300 dpi) into the same folder
as the calling script.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


# Okabe-Ito color-blind-safe palette
# https://jfly.uni-koeln.de/color/
OKABE_ITO = {
    "black":          "#000000",
    "orange":         "#E69F00",
    "sky":            "#56B4E9",
    "green":          "#009E73",
    "yellow":         "#F0E442",
    "blue":           "#0072B2",
    "vermillion":     "#D55E00",
    "purple":         "#CC79A7",
    "grey":           "#999999",
}


# Per-profile colors used consistently across Fig 2, 3, 4
PROFILE_COLORS = {
    "uniform":             OKABE_ITO["grey"],
    "anti":                OKABE_ITO["vermillion"],
    "anti_full":           OKABE_ITO["vermillion"],
    "proxy_C2":            OKABE_ITO["yellow"],
    "proxy_correct":       OKABE_ITO["orange"],
    "full":                OKABE_ITO["blue"],          # the slope-1 R-only profile
    "full_slope":          OKABE_ITO["blue"],
    "half_slope":          OKABE_ITO["sky"],
    "half_slope_rg":       OKABE_ITO["green"],         # the recommended recipe
    "range_only":          OKABE_ITO["purple"],
    "shuffled_R_plus_r":   "#A6761D",                  # warm-brown
    "R_plus_shuffled_r":   "#7570B3",                  # muted-violet
}


# Per-condition colors for long-context plots (Fig 5)
LONGCTX_COLORS = {
    "FP16_orig":          OKABE_ITO["vermillion"],
    "BF16_orig":          OKABE_ITO["orange"],
    "FP32_patched":       OKABE_ITO["yellow"],
    "FP32_patched_off":   OKABE_ITO["yellow"],
    "FP16_clip_only":     OKABE_ITO["sky"],
    "FP16_full":          OKABE_ITO["green"],
    "uniform":            OKABE_ITO["grey"],
    "full":               OKABE_ITO["blue"],
}


def setup():
    """Apply the shared style. Call once at the top of each fig*.py."""
    mpl.rcParams.update({
        "font.family":          "DejaVu Sans",
        "font.size":            10,
        "axes.titlesize":       11,
        "axes.labelsize":       10,
        "xtick.labelsize":      9,
        "ytick.labelsize":      9,
        "legend.fontsize":      9,
        "legend.frameon":       True,
        "legend.framealpha":    0.92,
        "legend.edgecolor":     "0.7",
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "axes.grid":            True,
        "grid.color":           "0.85",
        "grid.linestyle":       "-",
        "grid.linewidth":       0.5,
        "axes.axisbelow":       True,
        "lines.linewidth":      2.0,
        "lines.markersize":     6,
        "savefig.dpi":          300,
        "savefig.bbox":         "tight",
        "savefig.pad_inches":   0.03,
        "pdf.fonttype":         42,
        "ps.fonttype":          42,
    })


def save(fig, basename: str, fmts=("pdf", "png")):
    """Save a figure under the calling script's folder, in both pdf and png.

    The basename should be the figure stem (e.g. "fig04_factorisation").
    """
    here = Path(__file__).resolve().parent
    for ext in fmts:
        out = here / f"{basename}.{ext}"
        fig.savefig(out)
        size_kb = os.path.getsize(out) / 1024
        print(f"  wrote {out.name}  ({size_kb:.1f} KB)")


def load_results(filename: str) -> dict:
    """Load a results JSON from paper_c/results_c/."""
    here = Path(__file__).resolve().parent
    results_path = here.parent / "results_c" / filename
    with open(results_path) as f:
        return json.load(f)


def add_value_labels(ax, rects, fmt="{:+.1f}", fontsize=7, dy=2):
    """Annotate bar tops with their values."""
    for r in rects:
        h = r.get_height()
        if h is None or (h != h):  # NaN guard
            continue
        ax.annotate(
            fmt.format(h),
            xy=(r.get_x() + r.get_width() / 2, h),
            xytext=(0, dy),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=fontsize,
        )


def kansai_pretty_model(label: str) -> str:
    """Pretty-print a model label for axes / legends."""
    repl = {
        "Mamba-130M":  "130M",
        "Mamba-370M":  "370M",
        "Mamba-790M":  "790M",
        "Mamba-1.4B":  "1.4B",
        "Mamba-2.8B":  "2.8B",
    }
    return repl.get(label, label)
