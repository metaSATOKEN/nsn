# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0
"""Fig 5 — Long-context behaviour and the calibration benefit (§7.1, §7.3).

Mamba-2.8B-hf evaluated on PG-19 sliding windows at ctx in {8K, 16K, 32K},
showing that the unquantised FP16 / BF16 paths diverge to PPL ~ 200 while
the calibrated-clipping path stays at PPL ~ 11. A four-condition ablation
(FP16_orig / BF16_orig / FP16_clip_only / FP16_full INT6) localises the
mechanism in calibrated clipping.

Source: paper_c/results_c/phase_c9_fp16_bf16_clip_ablation.json
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from _style import setup, save, load_results, LONGCTX_COLORS


CTX_LENS = [8192, 16384, 32768]
CTX_LABELS = ["8K", "16K", "32K"]

CONDITIONS = [
    ("FP16_orig",        "FP16 unquantised"),
    ("BF16_orig",        "BF16 unquantised"),
    ("FP16_clip_only",   r"FP16 clip-only (calibrated $r_g$ clamp, no quant)"),
    ("FP16_full",        r"FP16 full_slope INT6 (Realisation-B proxy)"),
]


def main():
    setup()

    d = load_results("phase_c9_fp16_bf16_clip_ablation.json")
    sweep = d["sweep_results"]

    fig, ax = plt.subplots(1, 1, figsize=(8.4, 5.2))

    for cond, label in CONDITIONS:
        ys = []
        for ctx in CTX_LENS:
            ys.append(sweep[str(ctx)][cond]["ppl"])
        marker = "o" if "clip" in cond or "full" in cond else "s"
        linestyle = "-" if "FP16" in cond and "orig" not in cond else "--"
        if cond == "BF16_orig":
            linestyle = ":"
        ax.plot(
            CTX_LENS, ys,
            marker=marker, markersize=8,
            color=LONGCTX_COLORS[cond],
            label=label,
            linestyle=linestyle,
            linewidth=2.2,
        )
        # Annotate the right-most point
        ax.annotate(
            f"{ys[-1]:.1f}",
            xy=(CTX_LENS[-1], ys[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            ha="left", va="center",
            fontsize=9, color=LONGCTX_COLORS[cond],
            fontweight="bold",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(CTX_LENS)
    ax.set_xticklabels(CTX_LABELS)
    ax.set_xlabel("context length  (PG-19 sliding window)")
    ax.set_ylabel("perplexity")

    # Reference band: roughly the FP16 short-context baseline
    ax.axhspan(8, 12, color="0.93", zorder=0)
    ax.text(
        CTX_LENS[0] * 1.02, 9.4, "short-context baseline regime",
        fontsize=8, color="0.4", style="italic",
    )

    # FP16 / BF16 overlap is the §7.3 finding (precision-independent divergence)
    ax.annotate(
        "FP16 and BF16\noverlap (precision-\nindependent divergence)",
        xy=(CTX_LENS[1], 117),
        xytext=(7600, 90),
        textcoords="data",
        fontsize=8, color="0.30",
        ha="left", va="top", style="italic",
        arrowprops=dict(
            arrowstyle="->",
            color="0.4", linewidth=0.8,
            connectionstyle="arc3,rad=-0.18",
        ),
    )

    ax.set_xlim(7000, 60000)
    ax.set_ylim(7, 320)

    ax.set_title(
        "Long-Context Behaviour on PG-19  (Mamba-2.8B-hf)"
        + "\n"
        + r"unquantised FP16 / BF16 diverge near 32K; calibrated $r_g$ clipping keeps PPL $\approx 11$",
        pad=8, fontsize=11,
    )

    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)

    fig.tight_layout()
    save(fig, "fig05_long_context")
    plt.close(fig)


if __name__ == "__main__":
    main()
