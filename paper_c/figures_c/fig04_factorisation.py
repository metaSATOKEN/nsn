# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0
"""Fig 4 — Lyapunov-vs-range factorisation on Mamba-2.8B-hf (§6.8).

Six profiles compared at b_0 in {4, 6} with matched-mean shift normalisation:

  uniform              no signal
  full_slope           R-only, slope 1
  half_slope_rg        R + r (recommended recipe)
  range_only           r-only, no R
  shuffled_R_plus_r    R-shuffled + r (preserves r placement)
  R_plus_shuffled_r    R + r-shuffled (preserves R placement)

Two side-by-side panels: ΔPPL (% of FP16 baseline) on a log-scaled
y-axis with achieved mean bit width annotated above each bar.

Source: paper_c/results_c/phase_c16b_lyapunov_vs_range_factorisation.json
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from _style import (
    setup,
    save,
    load_results,
    PROFILE_COLORS,
)


PROFILES = [
    "uniform",
    "full_slope",
    "half_slope_rg",
    "range_only",
    "shuffled_R_plus_r",
    "R_plus_shuffled_r",
]

PROFILE_LABELS = {
    "uniform":             "uniform",
    "full_slope":          "full_slope",
    "half_slope_rg":       "half_slope_rg",
    "range_only":          "range_only",
    "shuffled_R_plus_r":   "shuffled_R\n+r",
    "R_plus_shuffled_r":   "R+\nshuffled_r",
}


def main():
    setup()

    d = load_results("phase_c16b_lyapunov_vs_range_factorisation.json")
    sweep = d["results"]["Mamba-2.8B"]["sweep"]
    fp16_ppl = d["results"]["Mamba-2.8B"]["fp16_baseline_ppl"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)

    for col, b0 in enumerate([4, 6]):
        ax = axes[col]
        delta_pcts, mean_bs, colors = [], [], []
        for p in PROFILES:
            entry = sweep[f"b{b0}_{p}"]
            delta_pcts.append(entry["delta_pct"])
            mean_bs.append(entry["achieved_mean"])
            colors.append(PROFILE_COLORS[p])

        x = np.arange(len(PROFILES))
        # The recommended recipe bar gets a thicker edge for emphasis
        edges = ["black" if p == "half_slope_rg" else "0.6" for p in PROFILES]
        edge_widths = [1.6 if p == "half_slope_rg" else 0.8 for p in PROFILES]

        rects = ax.bar(
            x, delta_pcts, color=colors, alpha=0.92,
            edgecolor=edges, linewidth=edge_widths,
        )

        # Annotate ΔPPL value and achieved mean above each bar
        for i, (r, dp, mb) in enumerate(zip(rects, delta_pcts, mean_bs)):
            top = r.get_height()
            ax.annotate(
                f"{dp:+.1f}%",
                xy=(r.get_x() + r.get_width() / 2, top),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=8.5, fontweight="bold",
            )
            ax.annotate(
                f"$\\overline{{b}}={mb:.2f}$",
                xy=(r.get_x() + r.get_width() / 2, top),
                xytext=(0, 16),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=7.5, color="0.30",
            )

        # Reference line: FP16 baseline (ΔPPL = 0)
        ax.axhline(0.0, color="0.5", linestyle="--", linewidth=0.8, zorder=1)

        # Highlight the b_0=6 range_only mean-bit caveat (drawn next to the bar)
        if b0 == 6:
            range_only_idx = PROFILES.index("range_only")
            ax.annotate(
                "under-shoots\nmatched-mean target",
                xy=(range_only_idx, 40.171),
                xytext=(range_only_idx + 0.55, 90),
                textcoords="data",
                fontsize=7.0,
                color=PROFILE_COLORS["range_only"],
                ha="left", va="bottom", style="italic",
                arrowprops=dict(
                    arrowstyle="-",
                    color=PROFILE_COLORS["range_only"],
                    linewidth=0.8,
                    shrinkA=0, shrinkB=2,
                ),
            )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [PROFILE_LABELS[p] for p in PROFILES],
            rotation=18, ha="right",
        )
        ax.set_yscale("symlog", linthresh=10)
        ax.set_ylim(-5, 600 if col == 0 else 180)
        ax.set_title(
            f"$b_0 = {b0}$"
            + ("  (aggressive INT4 memory budget)" if b0 == 4
               else "  (moderate INT6 memory budget)"),
            pad=8,
        )
        if col == 0:
            ax.set_ylabel(r"$\Delta$PPL  [% of FP16 baseline]")

        # Subtle grouping: vertical bands for the two profile groups
        ax.axvspan(-0.5, 2.5, color="0.97", zorder=0)
        ax.axvspan(2.5, 5.5, color="0.92", zorder=0)
        ax.set_xlim(-0.5, 5.5)

        # Tier labels above the bars
        ymax = ax.get_ylim()[1]
        ax.text(1.0, ymax * 0.83, "reference profiles",
                ha="center", fontsize=8.5, color="0.30",
                style="italic")
        ax.text(4.0, ymax * 0.83, "factorisation profiles",
                ha="center", fontsize=8.5, color="0.30",
                style="italic")

    fig.suptitle(
        "Lyapunov-vs-Range Factorisation on Mamba-2.8B-hf  "
        "(matched mean bit width, FP16 baseline PPL = "
        f"{fp16_ppl:.2f})",
        fontsize=11.5, y=1.005,
    )

    fig.tight_layout()
    save(fig, "fig04_factorisation")
    plt.close(fig)


if __name__ == "__main__":
    main()
