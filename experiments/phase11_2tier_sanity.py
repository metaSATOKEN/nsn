# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

"""
Phase 11: 2-tier nsep+pchan synthetic sanity check (v0.3 pivot).

After Phase 10 showed that Walsh-Hadamard rotation HURTS on top of
nsep+pchan on real Qwen2-7B Layer 0 K (3.68x MSE regression, 0/4
heads winning) because the distribution is a DISCRETE plateau rather
than heavy-tailed, we pivot to a different mechanism:

    Two-tier nsep+pchan: top-K highest-magnitude channels get
    bits_hi (default 6), remaining D-K channels get bits_lo
    (default 3). Effective bit rate ~3.09 bits/value at K=4,
    D=128. Data-oblivious per-head top-K selection at quantize time.

This script sweeps the same 5 synthetic regimes as Phase 9 and
compares:
  - qa_nsep_pchan(x, 3)                 [baseline INT3]
  - qa_nsep_pchan_2tier(x, 3, 6, k=4)   [v0.3 candidate]

If 2-tier beats baseline on the "plateau" regimes (2, 3) that real
Qwen2-7B resembles, we proceed to Phase 12 (real-model probe) and
then full WikiText-2 PPL.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from nsn.quantize import qa_nsep_pchan, qa_nsep_pchan_2tier


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
    g = torch.Generator().manual_seed(seed)
    normal = torch.randn(T, D, generator=g, dtype=torch.float32)
    chi2 = torch.distributions.Chi2(df).sample((T, D)).float()
    return normal / (chi2 / df).sqrt() * 0.3


def _gen_intermittent(T, D, seed, outlier_chan=17, p=0.1, scale=20.0):
    x = _gen_gaussian(T, D, seed)
    g = torch.Generator().manual_seed(seed + 2)
    mask = torch.rand(T, generator=g) < p
    g2 = torch.Generator().manual_seed(seed + 3)
    spike = torch.randn(T, generator=g2, dtype=torch.float32) * scale
    x[mask, outlier_chan] = spike[mask]
    return x


def _gen_qwen2_plateau(T, D, seed, chans=(7, 17, 31, 47, 89), scale=150.0):
    """Recreate the Phase 10 finding: 5-channel plateau at ~150 magnitude,
    rest at ~10. Tight imitation of Qwen2-7B Layer 0 K."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(T, D, generator=g, dtype=torch.float32) * 10.0
    for i, c in enumerate(chans):
        g2 = torch.Generator().manual_seed(seed + 10 + i)
        x[:, c] = torch.randn(T, generator=g2, dtype=torch.float32) * scale
    return x


def analyze(name, x, bits_lo=3, bits_hi=6, k=4):
    y_plain = qa_nsep_pchan(x, bits=bits_lo)
    y_2tier = qa_nsep_pchan_2tier(x, bits_lo=bits_lo, bits_hi=bits_hi, k=k)

    mse_plain = (x - y_plain).pow(2).mean().item()
    mse_2tier = (x - y_2tier).pow(2).mean().item()

    # Relative L2 error (more interpretable than raw MSE)
    n = x.norm(dim=-1).clamp(min=1e-12)
    rel_plain = ((x - y_plain).norm(dim=-1) / n).mean().item()
    rel_2tier = ((x - y_2tier).norm(dim=-1) / n).mean().item()

    return {
        "name": name,
        "mse_plain":        mse_plain,
        "mse_2tier":        mse_2tier,
        "mse_ratio_2tier_over_plain": mse_2tier / max(mse_plain, 1e-12),
        "rel_l2_plain":     rel_plain,
        "rel_l2_2tier":     rel_2tier,
        "2tier_helps":      mse_2tier < mse_plain,
    }


def main():
    T, D = 1024, 128
    seed = 0
    regimes = {
        "1_gaussian_no_outlier":  _gen_gaussian(T, D, seed),
        "2_single_outlier_10x":   _gen_single_outlier(T, D, seed),
        "3_multi_outlier_5x3x":   _gen_multi_outlier(T, D, seed),
        "4_student_t_df3":        _gen_student_t(T, D, seed),
        "5_intermittent_10pct":   _gen_intermittent(T, D, seed),
        "6_qwen2_plateau_150":    _gen_qwen2_plateau(T, D, seed),
    }

    results = []
    for name, x in regimes.items():
        r = analyze(name, x, bits_lo=3, bits_hi=6, k=4)
        results.append(r)
        verdict = "HELPS" if r["2tier_helps"] else "HURTS"
        rel_gain = (1 - r["mse_ratio_2tier_over_plain"]) * 100
        print(
            f"{name:28s} | mse plain={r['mse_plain']:.4f} "
            f"2tier={r['mse_2tier']:.4f} "
            f"(ratio={r['mse_ratio_2tier_over_plain']:.3f}, "
            f"gain={rel_gain:+.1f}%) "
            f"rel_l2 plain={r['rel_l2_plain']:.4f} "
            f"2tier={r['rel_l2_2tier']:.4f} -> 2-tier {verdict}"
        )

    # Save
    out = Path(__file__).resolve().parent.parent / "results"
    out.mkdir(exist_ok=True)
    with (out / "phase11_2tier_sanity.json").open("w") as f:
        json.dump({
            "regimes": results,
            "config":  {"T": T, "D": D, "bits_lo": 3, "bits_hi": 6, "k": 4,
                        "effective_bits_per_value": 3 + 3 * 4 / 128},
        }, f, indent=2)

    # Decision
    helps = sum(r["2tier_helps"] for r in results)
    total = len(results)
    print(f"\n2-tier HELPS on {helps}/{total} regimes.")
    plateau_r = next(r for r in results if r["name"] == "6_qwen2_plateau_150")
    if plateau_r["2tier_helps"] and plateau_r["mse_ratio_2tier_over_plain"] < 0.7:
        print(f"-> Qwen2-plateau shows >30% MSE reduction "
              f"({plateau_r['mse_ratio_2tier_over_plain']:.2f} ratio). "
              f"PROCEED to Phase 12 real-model probe.")
    elif plateau_r["2tier_helps"]:
        print(f"-> Qwen2-plateau wins but modestly "
              f"({plateau_r['mse_ratio_2tier_over_plain']:.2f} ratio). "
              f"Proceed cautiously; real-model test may amplify or wash out.")
    else:
        print(f"-> Qwen2-plateau LOST. 2-tier mechanism needs rethinking.")


if __name__ == "__main__":
    main()
