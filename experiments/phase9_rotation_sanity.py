# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

"""
Phase 9: Walsh-Hadamard rotation sanity check (v0.3 pre-flight).

Goal: before committing Colab GPU time to WikiText-2 evaluation on
Qwen2-7B / Llama-3.1-8B, empirically characterize WHEN rotation helps
and when it hurts the existing `nsep+pchan` INT3 pipeline. If
rotation uniformly hurts across realistic synthetic distributions,
v0.3 is dead on arrival and we need a different v0.3 angle.

We sweep five synthetic regimes intended to bracket real-model KV
statistics:

  1. i.i.d. Gaussian (no outliers) -> Mode-2 cartoon
  2. Single-channel outlier (10x magnitude) -> Mode-1 cartoon
  3. Multi-channel outlier (5 channels, 3x each) -> realistic pre-RoPE
  4. Heavy-tailed per-channel (Student-t nu=3) -> realistic post-RoPE
  5. Per-token intermittent outlier (outlier on 10% of tokens only) ->
     realistic "rare burst"

For each regime we compare plain `qa_nsep_pchan(x, 3)` to
`qa_nsep_pchan_rotated(x, 3)` by two metrics:
  - Per-element MSE of reconstruction
  - Per-channel max|dir| dispersion (should flatten under rotation)

Outcome interpretation:
  - If regime (2) is the only loss: our test synthetic was too cartoon.
    Proceed to real-model eval -- real Mode 1 likely resembles (3)-(5).
  - If (2), (3), (5) all lose: rotation is genuinely redundant with
    per-channel scaling and v0.3 needs a pivot.
  - If rotation wins on (3)-(5) even by a hair: promising signal,
    proceed to Colab.
"""

import os
import sys
from pathlib import Path

# Allow running from repo root without pip install
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import json
import torch

from modeq.quantize import qa_nsep_pchan, qa_nsep_pchan_rotated
from modeq.rotate import rotate


def _gen_gaussian(T, D, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(T, D, generator=g, dtype=torch.float32) * 0.3


def _gen_single_outlier(T, D, seed, outlier_chan=17, scale=10.0):
    x = _gen_gaussian(T, D, seed)
    g = torch.Generator().manual_seed(seed + 1)
    x[:, outlier_chan] = torch.randn(T, generator=g, dtype=torch.float32) * scale
    return x


def _gen_multi_outlier(T, D, seed, chans=(7, 17, 31, 47, 89), scale=3.0):
    x = _gen_gaussian(T, D, seed)
    for i, c in enumerate(chans):
        g = torch.Generator().manual_seed(seed + 10 + i)
        x[:, c] = torch.randn(T, generator=g, dtype=torch.float32) * scale
    return x


def _gen_student_t(T, D, seed, df=3.0):
    """Heavy-tailed per-channel (degrees of freedom = 3)."""
    g = torch.Generator().manual_seed(seed)
    # Student-t = N / sqrt(Chi2/df)
    normal = torch.randn(T, D, generator=g, dtype=torch.float32)
    chi2 = torch.distributions.Chi2(df).sample((T, D)).float()
    return normal / (chi2 / df).sqrt() * 0.3


def _gen_intermittent(T, D, seed, outlier_chan=17, p=0.1, scale=20.0):
    """Outlier on ~10% of tokens only."""
    x = _gen_gaussian(T, D, seed)
    g = torch.Generator().manual_seed(seed + 2)
    mask = torch.rand(T, generator=g) < p
    g2 = torch.Generator().manual_seed(seed + 3)
    spike = torch.randn(T, generator=g2, dtype=torch.float32) * scale
    x[mask, outlier_chan] = spike[mask]
    return x


def analyze(name, x, bits=3):
    # Per-channel max|direction| BEFORE rotation (on the unit direction
    # computed in nsep+pchan)
    n = x.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    d = x / n
    pre_chan_max = d.abs().amax(dim=0)  # per-channel max|direction|

    # Same AFTER rotation
    d_rot = rotate(d)
    rot_chan_max = d_rot.abs().amax(dim=0)

    # Quantize both ways and measure reconstruction error
    y_plain = qa_nsep_pchan(x, bits=bits)
    y_rot = qa_nsep_pchan_rotated(x, bits=bits)

    mse_plain = (x - y_plain).pow(2).mean().item()
    mse_rot = (x - y_rot).pow(2).mean().item()

    # Relative L2 (more interpretable than raw MSE)
    rel_plain = ((x - y_plain).norm(dim=-1) / n.squeeze(-1)).mean().item()
    rel_rot = ((x - y_rot).norm(dim=-1) / n.squeeze(-1)).mean().item()

    return {
        "name": name,
        "mse_plain": mse_plain,
        "mse_rot": mse_rot,
        "mse_ratio_rot_over_plain": mse_rot / max(mse_plain, 1e-12),
        "rel_l2_plain": rel_plain,
        "rel_l2_rot": rel_rot,
        "dir_chan_max_pre_mean": pre_chan_max.mean().item(),
        "dir_chan_max_pre_max": pre_chan_max.max().item(),
        "dir_chan_max_pre_std": pre_chan_max.std().item(),
        "dir_chan_max_rot_mean": rot_chan_max.mean().item(),
        "dir_chan_max_rot_max": rot_chan_max.max().item(),
        "dir_chan_max_rot_std": rot_chan_max.std().item(),
        "rotation_helps": mse_rot < mse_plain,
    }


def main():
    T, D = 1024, 128
    seed = 0
    regimes = {
        "1_gaussian_no_outlier": _gen_gaussian(T, D, seed),
        "2_single_outlier_10x":  _gen_single_outlier(T, D, seed),
        "3_multi_outlier_5x3x":  _gen_multi_outlier(T, D, seed),
        "4_student_t_df3":       _gen_student_t(T, D, seed),
        "5_intermittent_10pct":  _gen_intermittent(T, D, seed),
    }

    results = []
    for name, x in regimes.items():
        r = analyze(name, x, bits=3)
        results.append(r)
        verdict = "HELPS" if r["rotation_helps"] else "HURTS"
        print(
            f"{name:30s} | mse plain={r['mse_plain']:.5f} "
            f"rot={r['mse_rot']:.5f} (rel_l2: plain={r['rel_l2_plain']:.4f} "
            f"rot={r['rel_l2_rot']:.4f}) "
            f"dir_std pre={r['dir_chan_max_pre_std']:.3f} "
            f"rot={r['dir_chan_max_rot_std']:.3f} -> rotation {verdict}"
        )

    # Write to results/
    out = Path(__file__).resolve().parent.parent / "results"
    out.mkdir(exist_ok=True)
    with (out / "phase9_rotation_sanity.json").open("w") as f:
        json.dump({"regimes": results, "config": {"T": T, "D": D, "bits": 3}},
                  f, indent=2)

    # Decision signal
    helps_count = sum(r["rotation_helps"] for r in results)
    total = len(results)
    print()
    print(f"Decision signal: rotation HELPS on {helps_count}/{total} regimes.")
    if helps_count >= 3:
        print("-> PROCEED to Colab WikiText-2 evaluation.")
    elif helps_count >= 1:
        print("-> MIXED. Inspect which regimes win; real-model test will")
        print("   decide (outcome uncertain). Consider reducing scope.")
    else:
        print("-> PIVOT. Rotation appears redundant with per-channel scaling")
        print("   on every tested regime. v0.3 should use a different angle.")


if __name__ == "__main__":
    main()
