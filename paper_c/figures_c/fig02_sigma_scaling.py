# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0
"""Fig 2 — sigma^2 scaling test on the bit-allocation surface (§4.5).

The strict bit-budget-optimal continuous profile sigma_g proportional to
R_g^{-1/2} is uniquely linear-response (sigma^2 scaling) under controlled
noise injection across a four-fold sigma_rel sweep on Mamba-2.8B-hf,
while every tested control profile is super-quadratic.

This is the strongest theoretical-side confirmation of the R_g surface.

Source: paper_c/results_c/phase_c3b_sigma_rel_sweep.json
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from _style import setup, save, load_results, PROFILE_COLORS


PROFILES = ["full", "proxy_correct", "proxy_C2", "anti_full", "uniform"]

PROFILE_LABELS = {
    "uniform":         r"uniform                $\sigma_g \propto 1$",
    "anti_full":       r"anti                       $\sigma_g \propto R_g^{+1/2}$",
    "proxy_C2":        r"proxy_C2          $\sigma_g \propto C_t^{-1}$",
    "proxy_correct":   r"proxy_correct $\sigma_g \propto \sqrt{1-\rho_g}$",
    "full":            r"full                       $\sigma_g \propto R_g^{-1/2}$",
}


def main():
    setup()

    d = load_results("phase_c3b_sigma_rel_sweep.json")
    sigma_rels = d["protocol"]["sigma_rels"]

    fig, ax = plt.subplots(1, 1, figsize=(7.6, 5.4))

    # Reference slope-2 line in the noise-injection space
    ref_x = np.array([sigma_rels[0], sigma_rels[-1]])

    for prof in PROFILES:
        ys = []
        for sr in sigma_rels:
            v = d["sweep_results"][str(sr)][prof]["pct_of_base"]
            ys.append(v)
        ax.plot(
            sigma_rels, ys,
            marker="o", markersize=7,
            color=PROFILE_COLORS[prof],
            label=PROFILE_LABELS[prof],
            linewidth=2.0,
            zorder=3 if prof == "full" else 2,
        )

    # Add a slope-2 reference line through the "full" profile's first point
    full_first = d["sweep_results"][str(sigma_rels[0])]["full"]["pct_of_base"]
    ref_y = full_first * (ref_x / sigma_rels[0]) ** 2
    ax.plot(
        ref_x, ref_y,
        color="0.55", linestyle=":", linewidth=1.4, zorder=1,
        label=r"slope $= 2$  (linear-response reference)",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\sigma_\mathrm{rel}$  (relative noise level)")
    ax.set_ylabel(r"$\Delta$PPL  [% of FP16 baseline]")
    ax.set_title(
        r"$\sigma^2$-scaling test on Mamba-2.8B-hf  "
        r"($\sigma$-controlled noise injection)"
        + "\n"
        + r"the strict-optimum profile $\sigma_g \propto R_g^{-1/2}$"
        + r" is uniquely linear-response across the 4$\times$ sweep",
        pad=8, fontsize=11,
    )
    ax.set_xticks(sigma_rels)
    ax.set_xticklabels([f"{x:g}" for x in sigma_rels])
    ax.set_xlim(sigma_rels[0] * 0.85, sigma_rels[-1] * 1.18)

    # Compute and annotate the empirical slope (k ~= 2 expected only for "full")
    full_ys = [d["sweep_results"][str(sr)]["full"]["pct_of_base"]
               for sr in sigma_rels]
    log_x = np.log(sigma_rels)
    log_y = np.log(full_ys)
    slope_full = float(np.polyfit(log_x, log_y, 1)[0])

    ax.text(
        sigma_rels[-1] * 0.97, full_ys[-1] * 1.18,
        f"full slope = {slope_full:.2f}",
        ha="right", va="bottom",
        fontsize=9, color=PROFILE_COLORS["full"],
        fontweight="bold",
    )

    ax.legend(
        loc="upper left",
        fontsize=8.5,
        title="profile",
        title_fontsize=9,
        framealpha=0.95,
    )

    fig.tight_layout()
    save(fig, "fig02_sigma_scaling")
    plt.close(fig)


if __name__ == "__main__":
    main()
