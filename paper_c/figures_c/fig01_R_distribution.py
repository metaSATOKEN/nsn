# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0
"""Fig 1 — The bit-allocation surface is heavy-tailed (§4.1, §4.2).

Two panels on Mamba-2.8B-hf calibrated from 1024 tokens of WikiText-2 train:

  (a) Empirical spectral-radius distribution across all 5,242,880 (c, s)
      state units, in six rho buckets. The right-most two buckets contain
      the long-memory tail (>= 0.99) where (1-rho)^-1 amplification is
      orders of magnitude.

  (b) Output-coupling heterogeneity across the 64 layers. Each bar is the
      Gamma_max / Gamma_min ratio within a single layer's 16 state indices.
      The heavy tail motivates per-(channel, state-unit) granularity.

Sources:
  paper_c/results_c/phase_c1b_mamba1_empirical.json   (rho buckets)
  paper_c/results_c/phase_c1c_gamma_extraction.json   (Gamma per layer)
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from _style import setup, save, load_results, OKABE_ITO


RHO_BUCKETS = [
    (r"$[0, 0.5)$",       OKABE_ITO["grey"]),
    (r"$[0.5, 0.8)$",     OKABE_ITO["sky"]),
    (r"$[0.8, 0.95)$",    OKABE_ITO["green"]),
    (r"$[0.95, 0.99)$",   OKABE_ITO["yellow"]),
    (r"$[0.99, 0.999)$",  OKABE_ITO["orange"]),
    (r"$[0.999, 1]$",     OKABE_ITO["vermillion"]),
]


def main():
    setup()

    # Panel (a): rho distribution
    d_rho = load_results("phase_c1b_mamba1_empirical.json")
    bucket_totals = [0] * 6
    for layer in d_rho["empirical_per_layer"]:
        if layer.get("status") != "ok":
            continue
        for i, c in enumerate(layer["rho_emp_bucket_counts"]):
            bucket_totals[i] += c
    total_units = sum(bucket_totals)
    bucket_pcts = [100.0 * b / total_units for b in bucket_totals]

    # Panel (b): Gamma max/min ratio per layer
    d_gamma = load_results("phase_c1c_gamma_extraction.json")
    gamma_ratios = [
        layer["gamma_max_minus_min_ratio"]
        for layer in d_gamma["per_layer"]
    ]
    n_layers = len(gamma_ratios)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    # --- Panel (a) ---
    ax = axes[0]
    x = np.arange(len(RHO_BUCKETS))
    colors = [c for _, c in RHO_BUCKETS]
    rects = ax.bar(x, bucket_pcts, color=colors, alpha=0.92,
                   edgecolor="0.4", linewidth=0.6)
    for r, pct in zip(rects, bucket_pcts):
        ax.annotate(
            f"{pct:.2f}%",
            xy=(r.get_x() + r.get_width() / 2, r.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=9, fontweight="bold",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for lbl, _ in RHO_BUCKETS], rotation=15, ha="right")
    ax.set_ylabel("fraction of $(c, s)$ state units  [%]")
    ax.set_xlabel(r"$\rho_g$  bucket")
    ax.set_title(
        r"(a) Empirical $\rho_g$ distribution on Mamba-2.8B-hf"
        + "\n"
        + r"5,242,880 state units; long-memory tail $\rho_g \geq 0.99$ is 8.87%",
        fontsize=10.5, pad=8,
    )
    ax.set_ylim(0, max(bucket_pcts) * 1.18)

    # Highlight the long-memory tail with a subtle bracket and annotation
    tail_total = bucket_pcts[-2] + bucket_pcts[-1]
    ax.annotate(
        f"long-memory tail\n{tail_total:.2f}% near $\\rho_g \\to 1$",
        xy=(5, bucket_pcts[-1]),
        xytext=(4.2, max(bucket_pcts) * 0.62),
        textcoords="data",
        fontsize=8.5, color=OKABE_ITO["vermillion"],
        ha="center", style="italic",
        arrowprops=dict(
            arrowstyle="->",
            color=OKABE_ITO["vermillion"],
            linewidth=0.9,
        ),
    )

    # --- Panel (b) ---
    ax = axes[1]
    sorted_ratios = sorted(gamma_ratios)
    layer_idx = np.arange(n_layers)
    ax.bar(
        layer_idx, sorted_ratios,
        color=OKABE_ITO["blue"], alpha=0.85,
        edgecolor="0.4", linewidth=0.3,
        width=0.85,
    )
    ax.set_yscale("log")
    ax.set_xlabel("layer rank  (sorted by $\\Gamma_{max}/\\Gamma_{min}$)")
    ax.set_ylabel(r"$\Gamma_{max} \,/\, \Gamma_{min}$  within layer")
    ax.set_title(
        r"(b) Output-coupling heterogeneity across 64 Mamba mixers"
        + "\n"
        + f"per-layer ratio range: "
        + f"{min(sorted_ratios):.0f}$\\times$ to "
        + f"{max(sorted_ratios):.0f}$\\times$",
        fontsize=10.5, pad=8,
    )
    ax.set_xlim(-0.5, n_layers - 0.5)

    # Median and tail annotations
    median_ratio = float(np.median(sorted_ratios))
    ax.axhline(
        median_ratio,
        color=OKABE_ITO["vermillion"], linestyle="--",
        linewidth=1.0, alpha=0.7,
    )
    ax.annotate(
        f"median = {median_ratio:.0f}$\\times$",
        xy=(n_layers - 1, median_ratio),
        xytext=(-2, 4),
        textcoords="offset points",
        ha="right", va="bottom",
        fontsize=8.5, color=OKABE_ITO["vermillion"],
        style="italic",
    )

    fig.suptitle(
        "Why a uniform bit budget is the wrong default for SSM state quantisation",
        fontsize=11.5, y=1.02, fontweight="bold",
    )

    fig.tight_layout()
    save(fig, "fig01_R_distribution")
    plt.close(fig)


if __name__ == "__main__":
    main()
