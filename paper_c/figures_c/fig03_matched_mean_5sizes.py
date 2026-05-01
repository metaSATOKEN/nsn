# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0
"""Fig 3 — Matched-mean head-to-head across 5 state-spaces sizes (§6.7).

Two side-by-side panels: ΔPPL at b_0=4 (aggressive) and b_0=6 (moderate)
across the five state-spaces Mamba-1 sizes (130M to 2.8B), four profiles
(uniform / full_slope / half_slope / half_slope_rg), with mean bit width
matched across profiles via binary-search shift normalisation.

Source: paper_c/results_c/phase_c16_matched_mean_sweep.json
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from _style import (
    setup,
    save,
    load_results,
    PROFILE_COLORS,
    kansai_pretty_model,
)


PROFILES = ["uniform", "full_slope", "half_slope", "half_slope_rg"]
PROFILE_LABELS = {
    "uniform":         "uniform",
    "full_slope":      "full_slope",
    "half_slope":      "half_slope",
    "half_slope_rg":   "half_slope_rg",
}

MODELS = ["Mamba-130M", "Mamba-370M", "Mamba-790M", "Mamba-1.4B", "Mamba-2.8B"]


def main():
    setup()

    d = load_results("phase_c16_matched_mean_sweep.json")
    results = d["results"]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    n_models = len(MODELS)
    n_profiles = len(PROFILES)
    bar_w = 0.18
    group_centres = np.arange(n_models)

    for col, b0 in enumerate([4, 6]):
        ax = axes[col]

        for i, prof in enumerate(PROFILES):
            offsets = group_centres + (i - (n_profiles - 1) / 2) * bar_w
            vals = []
            mean_bs = []
            for m in MODELS:
                entry = results[m]["sweep"][f"b{b0}_{prof}"]
                vals.append(entry["delta_pct"])
                mean_bs.append(entry["achieved_mean"])

            edges = "black" if prof == "half_slope_rg" else "0.6"
            edge_w = 1.3 if prof == "half_slope_rg" else 0.6

            rects = ax.bar(
                offsets, vals, bar_w * 0.92,
                color=PROFILE_COLORS[prof],
                alpha=0.92,
                edgecolor=edges,
                linewidth=edge_w,
                label=PROFILE_LABELS[prof],
            )

            # Annotate ΔPPL value above each bar (compact)
            for r, v in zip(rects, vals):
                top = r.get_height()
                ax.annotate(
                    f"{v:+.0f}",
                    xy=(r.get_x() + r.get_width() / 2, top),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=6.4,
                    color="0.20",
                )

        ax.set_xticks(group_centres)
        ax.set_xticklabels([kansai_pretty_model(m) for m in MODELS])
        ax.set_xlabel("model size")

        ax.set_yscale("symlog", linthresh=5)
        if b0 == 4:
            ax.set_ylim(-2, 400)
            ax.set_ylabel(r"$\Delta$PPL  [% of FP16 baseline]")
            ax.set_title(
                "$b_0 = 4$  (aggressive INT4 memory budget)",
                pad=8,
            )
        else:
            ax.set_ylim(-1, 60)
            ax.set_title(
                "$b_0 = 6$  (moderate INT6 memory budget)",
                pad=8,
            )

        ax.axhline(0.0, color="0.5", linestyle="--", linewidth=0.7, zorder=1)

        if col == 0:
            ax.legend(
                loc="upper right", fontsize=8.5,
                title="profile", title_fontsize=8.5,
                ncol=1, framealpha=0.95,
            )

    fig.suptitle(
        "Matched-Mean Head-to-Head Across Five State-Spaces Sizes  "
        "(WikiText-2 test, ctx = 1024, n_windows = 5)",
        fontsize=11.5, y=1.005,
    )

    fig.tight_layout()
    save(fig, "fig03_matched_mean_5sizes")
    plt.close(fig)


if __name__ == "__main__":
    main()
