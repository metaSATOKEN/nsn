# Theory: Two Failure Modes of KV Cache INT Quantization

This document describes the theoretical framework that `modeq` is built on. The companion [paper](https://github.com/metaSATOKEN/norm-separated-quantization) has the full treatment with 7 appendices.

## The INT4 valley of death

When you quantize a per-row tensor $x$ of shape $[T, D]$ to INT$b$ using absmax symmetric quantization:

```
scale = absmax(x) / (2^(b-1) - 1)
x_q   = clamp(round(x / scale), -(2^(b-1)-1), 2^(b-1)-1) * scale
```

...the scale is dominated by the row's largest-magnitude value. If one channel holds a huge outlier (as Layer 0 K does on Qwen2-7B, with a K_max/K_mean ratio of 17.23×), every other channel's effective precision collapses.

Here's the counterintuitive part: **INT4 is uniquely catastrophic** on such models.

| Regime | Typical values round to | Outcome |
|:---|:---|:---|
| INT8 (qm=127) | ±7 to ±8 (many levels) | Fine — plenty of precision |
| INT6 (qm=31) | ±2 with some slop | OK |
| INT5 (qm=15) | ±1 cleanly | OK |
| **INT4 (qm=7)** | **±0.4 → round to 0 or ±1, noisy nonzero error** | **Catastrophic** |
| INT3 (qm=3) | ±0.17 → round to 0 cleanly | OK |
| INT2 (qm=1) | ±0.06 → round to 0 always | Bad but not catastrophic |

At INT4, normal values in an outlier-dominated row land precisely on the rounding boundary between 0 and ±1. The resulting nonzero errors don't cancel; they propagate through softmax and destroy attention. At INT3, the same values "zero out cleanly" — the error is large per-value but structured (many → 0), and attention survives.

Measured on Qwen2-7B (WikiText-2 ΔPPL):

```
INT2: +122.62   INT3: +11.92   INT4: +1382.07   INT5: +4.69   INT6: +11.67   INT8: +0.00
```

This is the **"INT4 valley of death"**. It's a specific mathematical interaction between outlier magnitude and grid size, not a monotonic function of bit width.

## Two empirically separable modes

We ran the same bit-width sweep on 7 models from 5 vendors. Only one (Qwen2-7B) showed the INT4 valley. The others were monotonic: more bits, less error, no surprises. A selective-compression experiment confirms the dichotomy causally.

**Causal test:** On each failing model, compress only Layer 0 with `nsep+pchan`; leave the other layers under naive INT4. What fraction of the damage does fixing Layer 0 alone rescue?

| Model (failure severity) | Naive INT4 ΔPPL | Fixing only Layer 0 ΔPPL | Rescue |
|:---|:-:|:-:|:-:|
| Qwen2-7B (catastrophic) | +1382 | +0.38 | **99.97%** |
| Pythia-12B (strong) | +14.8 | +14.8 | 0.3% |
| Pythia-6.9B (moderate) | +7.5 | +7.5 | 0.1% |
| Qwen2.5-14B (mild) | +0.19 | +0.18 | 4.3% |

Two cleanly separated patterns:

**Mode 1 — Layer-0-localized, outlier-driven.** A single K outlier at Layer 0 dominates the per-row scale. Errors at Layer 0 propagate multiplicatively through every subsequent attention, because Layer 0 K determines which tokens every later layer ends up reading. Fixing Layer 0 alone essentially cures the model. Observed in 1 of 27 models we've evaluated (Qwen2-7B only). Signature: K_max ≥ 15× at Layer 0.

**Mode 2 — Distributed, norm-variation-driven.** Per-token L2 norms span a wide range; no single layer's per-row absmax can fit both small and large norms simultaneously, so each layer contributes additive quantization error. Fixing one layer is worthless because the damage is distributed. This is the common case. Signature: K_max at Layer 0 < 10×.

Importantly, the **same vendor can produce either mode in different checkpoints**. Qwen2-7B (June 2024) is Mode 1; Qwen2.5-14B (October 2024) from the same vendor is Mode 2. The four-month recipe revision eliminated the Layer-0 K outlier.

## Why `nsep+pchan` handles both modes

The method has two components:

- **Norm separation** (`nsep`): decompose $x = n \cdot d$ where $n = \|x\|$ (stored exactly) and $d = x / n$ is the unit direction (quantized). Removes the token-wise dynamic range from the direction before quantization. Targets **Mode 2**.

- **Per-channel scaling** (`pchan`): quantize $d$ with a per-channel scale instead of a per-row scale. An outlier channel then gets its own scale and cannot crush the other channels' precision. Targets **Mode 1**.

Re-normalization after quantization brings the reconstructed direction back onto the unit sphere, eliminating systematic bias from quantization-induced norm drift.

**Why both are needed:** neither alone suffices. On Qwen2-7B (where both modes are present to some degree, though Mode 1 dominates):

- `nsep` alone: 4.1× improvement over naive
- `pchan` alone: 2.4× improvement over naive
- `nsep + pchan`: **744×** improvement over naive

The conjunction's multiplicative effect is the empirical fingerprint of two orthogonal failure modes being jointly handled.

## The INT3 sweet spot

Going below INT4 with `nsep+pchan`:

```
ΔPPL on Qwen2-7B (Mode 1):
  nsep+pchan INT4: +0.05
  nsep+pchan INT3: +0.40  ← 5.33× compression, still production quality
  nsep+pchan INT2: +325.96 ← breaks; direction needs more than 3 levels
```

INT3 works because per-channel scaling with 7 levels (symmetric: −3 to +3, total 7; or asymmetric: 0 to 7, total 8) is enough to preserve direction up to acceptable quantization error, while the norm is stored exactly. INT2 (symmetric: 3 levels −1, 0, +1) cannot represent arbitrary directions with enough resolution, and the method collapses. The "bit floor" for direction quantization is INT3.

Empirically, **4 of 7 models achieved our production threshold (ΔPPL < 5% of WikiText-2 baseline) at INT3 symmetric `nsep+pchan`** (the v0.2 default). One (Phi-3-medium at 5.2%) sat at the borderline; one was acceptable; one (Pythia-6.9B, a base model without instruction tuning, at 22%) was research-only. This was the main pragmatic contribution of `modeq` v0.2: 5.33× theoretical / 5.08× measured KV cache compression with a drop-in preprocessing step handling both catastrophic and common failure modes on a majority of tested models.

## v0.3 update: asymmetric per-channel is the new default (2026-04)

Phases 10, 13, and 14 (see `results/`) found that **asymmetric per-channel quantization** — storing per-channel `[min, max]` rather than a symmetric `[-absmax, +absmax]` window — reduces INT3 ΔPPL by an average **66.0% across 8 benchmarked models, 100% win rate**:

- Qwen2-7B (Mode 1): +0.402 → +0.103 (−74.3%)
- Llama-3.1-8B (Mode 2): +0.254 → +0.106 (−58.3%)
- Pythia-6.9B: +2.378 → +0.099 (−95.8%) — moves from "research only" into the production tier
- Qwen2.5-14B: +0.261 → +0.091 (−65.1%)
- Mistral-7B-v0.1: +0.124 → +0.055 (−56.0%)
- GPT-J-6B (head_dim=256): +0.187 → +0.070 (−62.5%)
- Qwen3.5-9B (hybrid attention, 2026): +0.275 → +0.094 (−65.7%)
- Gemma 4-E4B (multimodal, 2026): +1.879 → +0.930 (−50.5%)

**All 8 land in the < 5% baseline production tier under asym INT3 at identical 3.0 bits/value.**

Mechanism: post-RoPE K channels have strongly biased distributions. The symmetric absmax window wastes roughly half of its quantization levels on the empty side of zero. Asymmetric fits `[min, max]` tightly and uses all `2**bits` levels (8 instead of 7 at INT3 — a 14% base gain just from level count, on top of the much larger range-fit advantage). The storage cost is one additional FP16 per channel (+0.008 bits/value in real packing), so the v0.3 `Int3AsymNsepPchanCache` measures 5.09× compression vs FP16 — essentially identical to v0.2's 5.08× but with 58–96% less ΔPPL drift.

Above INT3, symmetric is still the v0.3 default. Phase 14 measured asymmetric INT4 as *worse* than symmetric INT4 on Llama-3.1-8B (+0.027 vs +0.013 ΔPPL). Above the INT3 bit floor, the zero-centering of symmetric quantization carries value that asym's extra level cannot compensate for. The v0.3 dispatcher applies asymmetric only at INT3 and returns to symmetric for INT4+.

Rotation (Walsh–Hadamard, similar to KVLinC / QuaRot style) was also tested in v0.3 development but *regressed* MSE by 3.68× on real Qwen2-7B Layer 0 K (Phase 10). The distribution is a discrete multi-channel plateau rather than heavy-tailed, which per-channel absmax already handles optimally. Rotation is retained in the package as a documented negative result (`qa_nsep_pchan_rotated`) but is not dispatched.

## References

- Companion paper: [Norm-Separated Quantization](https://github.com/metaSATOKEN/norm-separated-quantization) (V5, 23 pages)
- [Appendix H](https://github.com/metaSATOKEN/norm-separated-quantization/blob/main/paper/arc_compression.pdf) of the paper contains the full 11-model K/V outlier asymmetry study and the selective-compression causal experiments.
- v0.3 empirical data: `results/phase10_real_kv_distribution.json`, `phase13_full_sweep.json`, `phase14_asym_wikitext2_ppl.json`, `phase15_real_compression.json`, `phase16_asym_seven_model_sweep.json`, `phase17_2026_flagships.json`.
