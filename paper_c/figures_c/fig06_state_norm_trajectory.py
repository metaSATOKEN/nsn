# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0
"""Fig 6 — Long-context state-norm and cliprate trajectory (§7.5).

Three stacked panels for Mamba-2.8B-hf on PG-19 at ctx=32K, recording
trajectories at 7 representative layers (0, 16, 32, 40, 48, 56, 63):

  (a) |h_t| p99 trajectory under FP32_patched_off (the unmodified
      recurrence). Late-context tail growth localises which layers
      operate in the rare-excursion regime.

  (b) peak rho_g trajectory (rho_p99 per chunk). Shows which layers
      approach the rho_g -> 1 recurrence-stability boundary as
      context grows.

  (c) Pre-clip cliprate (% of (c, s) units with |h_t| > r_g) under
      FP32_patched_off. Non-zero on every recorded layer at long
      context, with peak rates 0.1 to 2.3% depending on the layer.

Source: paper_c/results_c/phase_c10b_state_norm_pre_clip.json
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

from _style import setup, save, load_results, OKABE_ITO


CTX_TARGET = "32768"
COND = "FP32_patched_off"


def main():
    setup()

    d = load_results("phase_c10b_state_norm_pre_clip.json")
    cond_data = d["sweep_results"][CTX_TARGET][COND]
    trj = cond_data["trajectories"]

    # Layer ordering from JSON keys (already in numerical order)
    layers = list(trj.keys())
    n_layers = len(layers)

    # Use viridis to encode layer depth
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(n_layers - 1, 1)) for i in range(n_layers)]

    fig, axes = plt.subplots(3, 1, figsize=(8.6, 8.4), sharex=True)

    # The "t" field is already a token position (0, 512, 1024, ..., 32767)
    # so we use it directly on the x-axis.

    for li, lname in enumerate(layers):
        rows = trj[lname]
        ts = np.array([r["t"] for r in rows])
        h_p99 = np.array([r["h_abs_p99"] for r in rows])
        rho_p99 = np.array([r["rho_p99"] for r in rows])
        cliprate = 100.0 * np.array([r["cliprate_pre"] for r in rows])

        # Pretty layer label: "L0", "L16", ...
        layer_idx = lname.split(".")[2]
        label = f"layer {layer_idx}"

        axes[0].plot(ts, h_p99, color=colors[li], linewidth=1.6, label=label)
        axes[1].plot(ts, rho_p99, color=colors[li], linewidth=1.6, label=label)
        axes[2].plot(ts, cliprate, color=colors[li], linewidth=1.6, label=label)

    # --- Panel (a): state-norm p99 ---
    ax = axes[0]
    ax.set_yscale("log")
    ax.set_ylabel(r"$|h_t|$ p99")
    ax.set_title(
        r"(a) State-norm p99 trajectory — late-context tail growth",
        loc="left", pad=4, fontsize=10.5,
    )
    ax.legend(
        loc="lower right", ncol=4, fontsize=7.5,
        framealpha=0.9, columnspacing=0.8,
    )

    # --- Panel (b): peak rho ---
    ax = axes[1]
    ax.axhline(
        1.0, color=OKABE_ITO["vermillion"], linestyle=":",
        linewidth=1.0, alpha=0.8,
    )
    ax.text(
        500, 1.001, r"$\rho_g \to 1$ recurrence-stability boundary",
        fontsize=8, color=OKABE_ITO["vermillion"], style="italic",
        va="bottom",
    )
    ax.set_ylabel(r"peak $\rho_g$  (p99)")
    ax.set_title(
        r"(b) Peak spectral-radius trajectory — long-memory units approach $\rho \to 1$",
        loc="left", pad=4, fontsize=10.5,
    )
    ax.set_ylim(0.85, 1.005)

    # --- Panel (c): cliprate ---
    ax = axes[2]
    ax.set_ylabel(r"pre-clip rate  [% of $(c,s)$]")
    ax.set_xlabel("token position in 32K context")

    # Common x-axis ticks across all panels (8K / 16K / 24K / 32K)
    for a in axes:
        a.set_xticks([0, 8192, 16384, 24576, 32768])
        a.set_xticklabels(["0", "8K", "16K", "24K", "32K"])
        a.set_xlim(0, 33000)
    ax.set_title(
        r"(c) Pre-clip cliprate — fraction of state units with $|h_t| > r_g$ "
        r"on the unmodified recurrence",
        loc="left", pad=4, fontsize=10.5,
    )
    # Find a sensible y-max (the maximum cliprate observed)
    all_cliprate_pcts = [
        100.0 * r["cliprate_pre"]
        for lname in layers for r in trj[lname]
    ]
    ymax = max(all_cliprate_pcts) * 1.18 if all_cliprate_pcts else 3.0
    ax.set_ylim(0, ymax)

    fig.suptitle(
        "Five-stream mechanism at long context  "
        "(Mamba-2.8B-hf, PG-19 ctx = 32K, FP32_patched_off proxy for FP16_orig)",
        fontsize=11.5, y=1.005, fontweight="bold",
    )

    fig.tight_layout()
    save(fig, "fig06_state_norm_trajectory")
    plt.close(fig)


if __name__ == "__main__":
    main()
