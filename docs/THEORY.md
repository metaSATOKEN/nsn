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

## The common case, and one observed special case

We ran the same bit-width sweep on 7 models from 5 vendors. Only one (Qwen2-7B) showed the INT4 valley. The other six were monotonic: more bits, less error, no surprises. A selective-compression experiment shows that the Qwen2-7B anomaly is mechanistically distinct from the small-but-nonzero damage the other failing-model candidates sustain at lower bit widths.

**Causal test:** On each failing model, compress only Layer 0 with `nsep+pchan`; leave the other layers under naive INT4. What fraction of the damage does fixing Layer 0 alone rescue?

| Model (failure severity) | Naive INT4 ΔPPL | Fixing only Layer 0 ΔPPL | Rescue |
|:---|:-:|:-:|:-:|
| Qwen2-7B (catastrophic) | +1382 | +0.38 | **99.97%** |
| Pythia-12B (strong) | +14.8 | +14.8 | 0.3% |
| Pythia-6.9B (moderate) | +7.5 | +7.5 | 0.1% |
| Qwen2.5-14B (mild) | +0.19 | +0.18 | 4.3% |

The causal pattern separates cleanly, but the sample sizes are very unequal — one Mode 1 case, many Mode 2 cases:

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
  nsep+pchan INT3: +0.40  ← 5.33× compression, still usable (see paper §5 threshold)
  nsep+pchan INT2: +325.96 ← breaks; direction needs more than 3 levels
```

INT3 works because per-channel scaling with 7 levels (symmetric: −3 to +3, total 7; or asymmetric: 0 to 7, total 8) is enough to preserve direction up to acceptable quantization error, while the norm is stored exactly. INT2 (symmetric: 3 levels −1, 0, +1) cannot represent arbitrary directions with enough resolution, and the method collapses. The "bit floor" for direction quantization is INT3.

Empirically, **4 of 7 models achieved our usability threshold (ΔPPL < 5% of WikiText-2 baseline) at INT3 symmetric `nsep+pchan`** (the v0.2 default). One (Phi-3-medium at 5.2%) sat at the borderline; one was acceptable; one (Pythia-6.9B, a base model without instruction tuning, at 22%) was research-only. This was the main pragmatic contribution of `modeq` v0.2: 5.33× theoretical / 5.08× measured KV cache compression with a drop-in preprocessing step handling both catastrophic and common failure modes on a majority of tested models.

## v0.3 update: asymmetric per-channel is the new default (2026-04)

Phases 10, 13, and 14 (see `results/`) found that **asymmetric per-channel quantization** — storing per-channel `[min, max]` rather than a symmetric `[-absmax, +absmax]` window — reduces INT3 ΔPPL by an average **66.0% across 8 benchmarked models, 100% win rate**:

- Qwen2-7B (Mode 1): +0.402 → +0.103 (−74.3%)
- Llama-3.1-8B (Mode 2): +0.254 → +0.106 (−58.3%)
- Pythia-6.9B: +2.378 → +0.099 (−95.8%) — moves from "research only" below our usability threshold
- Qwen2.5-14B: +0.261 → +0.091 (−65.1%)
- Mistral-7B-v0.1: +0.124 → +0.055 (−56.0%)
- GPT-J-6B (head_dim=256): +0.187 → +0.070 (−62.5%)
- Qwen3.5-9B (hybrid attention, 2026): +0.275 → +0.094 (−65.7%)
- Gemma 4-E4B (multimodal, 2026): +1.879 → +0.930 (−50.5%)

**Under our primary evaluation protocol (fp16 load, 10 WikiText-2 windows at 1024 ctx), all 8 stay below the < 5 % baseline usability threshold under asym INT3 at identical 3.0 bits/value.** A follow-up bf16 / 30-window probe on Gemma-4-E4B places that specific model at 5.5 %–7.0 % of baseline, so we describe Gemma-4-E4B as borderline under the stricter probe rather than uniformly below threshold. See Phase 20 / 21 JSON in `results/` and `paper/nsn.md` §7 for the full breakdown.

Mechanism: post-RoPE K channels have strongly biased distributions. The symmetric absmax window wastes roughly half of its quantization levels on the empty side of zero. Asymmetric fits `[min, max]` tightly and uses all `2**bits` levels (8 instead of 7 at INT3 — a 14% base gain just from level count, on top of the much larger range-fit advantage). The storage cost is one additional FP16 per channel (+0.008 bits/value in real packing), so the v0.3 `Int3AsymNsepPchanCache` measures 5.09× compression vs FP16 — essentially identical to v0.2's 5.08× but with 58–96% less ΔPPL drift.

Above INT3, symmetric is still the v0.3 default. Phase 14 measured asymmetric INT4 as *worse* than symmetric INT4 on Llama-3.1-8B (+0.027 vs +0.013 ΔPPL). Above the INT3 bit floor, the zero-centering of symmetric quantization carries value that asym's extra level cannot compensate for. The v0.3 dispatcher applies asymmetric only at INT3 and returns to symmetric for INT4+.

Rotation (Walsh–Hadamard, similar to KVLinC / QuaRot style) was also tested in v0.3 development but *regressed* MSE by 3.68× on real Qwen2-7B Layer 0 K (Phase 10). The distribution is a discrete multi-channel plateau rather than heavy-tailed, which per-channel absmax already handles optimally. Rotation is retained in the package as a documented negative result (`qa_nsep_pchan_rotated`) but is not dispatched.

## v0.3 ablation: why each component is necessary (Phase 18 + 19)

After v0.3 shipped, we ran a four-method INT3 ablation on all 8 benchmarked models to answer a structural question: is modeq's v0.3 (D = `nsep + pchan + asym`) the *simplest* thing that works, or is one of the three components dispensable? The four methods were:

- **(A) naive per-row symmetric INT3** — no `pchan`, no `nsep`, no `asym`. Industry baseline.
- **(B) asym per-channel INT3, NO `nsep`** — `pchan` and `asym`, but not `nsep`. This is the "weight-quantization-style" recipe applied naively to KV cache.
- **(C) `nsep+pchan` symmetric INT3** — `nsep` and `pchan`, but not `asym`. This is modeq **v0.2**.
- **(D) `nsep+pchan` asymmetric INT3** — all three. This is modeq **v0.3**.

Usability-tier coverage (< 5 % baseline ΔPPL on WikiText-2) across 8 models. Table 2b / Appendix E of `paper/nsn.md` is the authoritative version; the row-year split here is a quick readable aggregate.

| Method | 2021 (GPT-J) | 2023 (Pythia, Mistral) | 2024 (Qwen2, Qwen2.5, Llama) | 2026 (Qwen3.5, Gemma) | Total under threshold |
|:---|:---:|:---:|:---:|:---:|:---:|
| (A) naive | 1 / 1 | 1 / 2 | 0 / 3 | 1 / 2 | **3 / 8** (over threshold on Pythia 💥, Qwen2 💥, Qwen2.5-14B, Llama mild, Gemma) |
| (B) asym no `nsep` | 0 / 1 | 0 / 2 | 1 / 3 | 2 / 2 | **3 / 8** (over threshold on 5 — 💥 catastrophic on Llama, Mistral, Qwen2.5-14B; mild on GPT-J 6.27 % and Pythia 5.88 %) |
| (C) `nsep+pchan` sym (v0.2) | 1 / 1 | 1 / 2 | 2 / 3 | 1 / 2 | **5 / 8** (over threshold on Pythia 22.1 %, Qwen2.5-14B 13.3 %, Gemma 9.1 %) |
| **(D) `nsep+pchan` asym (v0.3)** | **1 / 1** | **2 / 2** | **3 / 3** | **2 / 2** | **8 / 8** 🏆 |

💥 = catastrophic (> 100 % of baseline PPL).

Concretely, each component is empirically necessary:

**1. `pchan` is needed for Mode-1-style outlier channels.** Without per-channel scaling, method (A) fails catastrophically on Qwen2-7B (+11.9 ΔPPL, 142 % of baseline) and Pythia-6.9B (+22.5 ΔPPL, 209 %), because a per-row absmax scale is stretched wide to accommodate the outlier channel and mis-quantizes the other channels. Qwen2.5-14B (43.95 %), Llama-3.1-8B (6.03 %), and Gemma-4-E4B (26.45 %) are also over the 5 % threshold under (A).

**2. `nsep` is needed for Mode-2-style high-norm-variance distributions.** Without norm separation, method (B) fails our threshold on 5 of 8 — catastrophically on Llama-3.1-8B (+10.9 ΔPPL, 148 %), Mistral-7B-v0.1 (+7.5 ΔPPL, 120 %), and Qwen2.5-14B (+2.9 ΔPPL, 146 %), and mildly on GPT-J-6B (6.27 %) and Pythia-6.9B (5.88 %). The failure mechanism is specific to KV cache and does not appear in weight quantization: per-channel `[min, max]` computed on raw activations is dominated by high-norm tokens, and the resulting scale mis-quantizes low-norm tokens by a factor equal to the token-norm ratio. `nsep` normalizes each token to unit length before computing per-channel statistics, removing this contamination. The V-side of this mechanism dominates in practice (§5.3 / Appendix E: Spearman rho(V-CV, (B) failure) ~= 0.88 vs rho(K-CV, (B) failure) ~= 0.29), because V-cache error propagates linearly through the attention-weighted output while K-cache error is partly absorbed by softmax normalization.

**3. `asym` is needed for post-RoPE biased distributions.** Without asymmetric quantization, method (C, v0.2) lands in only 5 of 8 models — Pythia-6.9B at 22 % of baseline is the clearest failure. Asymmetric fits `[min, max]` tightly and avoids wasting half the quantization levels on the empty side of a symmetric window.

Only the full combination (D) succeeds on all 8 models. The asymmetric-per-channel-without-nsep recipe (B) that weight-quantization literature treats as "standard" is **actively dangerous on KV cache**: it can be up to 24× worse than the naive per-row baseline it is meant to improve, specifically on modern Llama-family architectures where token-wise norm variance is large.

### Footnote: accidental K-cache regularization on Gemma 4-E4B

On gemma-4-E4B (multimodal, 2026), method (B) produced a small but reproducible *negative* ΔPPL (−0.40 on WikiText-2 chars [0:48k], −0.15 on chars [48:96k], across fp16 and native bf16). A K-only / V-only split (Phase 21) isolated 87 % of the effect to the Key cache, consistent with accidental attention-logit entropy regularization: quantization noise on K softens the attention distribution, which helps when a multimodal-trained model is evaluated text-only. This effect does NOT generalize — (B) catastrophically regresses on 4 of the other 7 models — and does NOT affect the recommended recipe (D), which produces ordinary small-positive ΔPPL (+0.93) on gemma-4-E4B. We flag this as an open observation: K-cache-quantization-as-regularization in multimodal models may be worth instrumenting directly (attention entropy measurement) in future work. It is not a deployable method.

## v0.3 mechanism refinement: V-cache is the dominant driver (Phase 23)

The v0.2 "Mode 1 / Mode 2" framing above is a useful *causal* description — outlier-driven vs. norm-variance-driven — but it was introduced before we directly measured token-wise norm dispersion on the 8-model set. Phase 23 (`results/phase23_norm_dispersion.json`) is that direct measurement: for each model, tokenize the WikiText-2 test text with the model's own tokenizer, take three 1024-token segments, run each through a prefill pass, and compute the coefficient of variation ($\mathrm{CV} = \sigma / \mu$) of the per-token L2 norm on the Layer-0 K and V caches (pooled across heads).

The measurement refines the Phase 18 / 19 ablation-(B) failure story in two ways:

**1. It is V-cache dispersion, not K-cache dispersion, that predicts (B) failure.** The Spearman rank correlation between L0 V-CV and variant-(B) regression magnitude across the 8 models is $\rho_V \approx 0.881$ (Fisher-$z$ 95 % CI $\approx [0.46, 0.98]$); the same correlation computed against L0 K-CV is $\rho_K \approx 0.286$ (CI $\approx [-0.52, 0.82]$, straddling zero). Three catastrophic (B) failures (Llama-3.1-8B, Qwen2.5-14B, Mistral-7B-v0.1) are the three highest V-CV models; the two negative-ΔPPL models (Qwen3.5-9B, Gemma-4-E4B) are the two lowest. K-CV alone would have placed Qwen2-7B — which has the highest K-CV in the set at 0.472 — as a catastrophic-(B) candidate, but Qwen2-7B actually passes (B) at 4.29 % of baseline.

**2. The softmax absorbs K-cache error but not V-cache error.** Writing one attention head's output as $y = \mathrm{softmax}(QK^\top / \sqrt{D}) \cdot V$ and expanding to first order:

$$\Delta y \big|_{V \to V + \Delta V} \;=\; \mathrm{softmax}(QK^\top / \sqrt{D}) \cdot \Delta V$$

$$\Delta y \big|_{K \to K + \Delta K} \;\approx\; \tfrac{1}{\sqrt{D}}\,\mathrm{diag}(p) \cdot \bigl(Q \Delta K^\top - p\,(Q \Delta K^\top)\bigr) \cdot V$$

where $p = \mathrm{softmax}(QK^\top / \sqrt{D})$. The V perturbation is linear and un-damped — every per-token V error contributes to $y$ with weight equal to that token's attention score. The K perturbation is *doubly* damped: by $1/\sqrt{D}$ and by the softmax Jacobian (whose operator norm is bounded by $\max_i p_i (1 - p_i) \leq 1/4$). This is why per-token V-cache norm variance — if not neutralized by `nsep` — blows up the end-to-end ΔPPL while the corresponding K-cache variance does not.

**Confounder caveat.** The three catastrophic (B) failures in the benchmark cluster on a single architectural profile ($D = 128$, $H_{\mathrm{KV}} = 8$, GQA, SwiGLU, Llama-style RoPE). At $n = 8$ we cannot fully separate "V-CV is the causal knob" from "V-CV is a proxy for some other Llama-family-correlated variable". The V-CV / failure correlation is *consistent with* the mechanism above without confirming it causally.

**Architectural observation: QK-normalization on 2026 models.** Gemma-4-E4B's Layer-0 K-CV is measured at 0.000 and Qwen3.5-9B's at 0.034, versus 0.21–0.47 on the six older models. This is the signature of architectural QK-normalization — RMSNorm applied to $Q$ and $K$ before the attention dot product forces K-cache rows to approximately unit L2 norm by construction. On these two 2026 models, `nsep` on K is redundant with the architecture; on the six older models it is not. V-cache `nsep` is still required on all 8 models.

Practical implication: the diagnostic "measure L0 V-cache per-token L2 norm CV, and if it's in the 0.4–0.6 range, nsep is structurally required before per-channel quantization" is a cheap pre-flight check (one 1024-token forward pass; under 30 minutes on a single A100).

## v0.3 long-context validation: multi-needle NIH at 32 K (Phase 24)

Phases 24b / 24c / 24d (`results/phase24b_*.json`, `phase24c_*.json`, `phase24d_*.json`) tested whether the WikiText-2-measured advantage of (D) asymmetric INT3 over (C) symmetric INT3 carries through to long-context retrieval. We embedded three distractor needles — "Project Titan → XRQ7-BLUE-1984", "Project Orion → LMN3-GREEN-2037", "Project Vega → PQZ5-AMBER-1776" — in a WikiText-2 haystack at fixed positions $\{0.2, 0.5, 0.8\}$, asked for one specific label's code, and 3-way-classified the 15-token greedy decode (pass / wrong-code / hallucinate). 10 trials per (model, context, method) cell; pooled $n = 30$ across three models (Llama-3.1-8B, Qwen2-7B, Qwen2.5-14B) spanning two families, three scales, and both sides of the (B) failure classification.

**Pooled 32 K outcome (Fisher exact two-sided):**

| Method | Pass | Wrong-code | Hallucinate | vs FP16 | vs asym |
|:---|:---:|:---:|:---:|:---:|:---:|
| FP16 baseline | 29 / 30 | 1 / 30 | 0 / 30 | — | $p = 0.194$ (n.s.) |
| (C) sym INT3 (v0.2) | 15 / 30 | 6 / 30 | 9 / 30 | $p = 6.4 \times 10^{-5}$ | $p = 0.013$ |
| **(D) asym INT3 (v0.3)** | **25 / 30** | 5 / 30 | **0 / 30** | $p = 0.194$ (n.s.) | — |

At 32 K, the v0.3 recipe shows no significant difference from FP16 on this lead-key retrieval test ($p = 0.19$; $n = 30$ is too small to formally exclude moderate loss); v0.2 sym INT3 is significantly worse than both FP16 and v0.3. Pooled 8 K is saturated (asym 23 / 30 vs sym 22 / 30, $p = 1.00$) — the context is below the compression-sensitivity threshold for this protocol.

**Bit-mixing hallucinations under v0.2 sym INT3.** At 32 K, each of the three models produces cross-needle character fusions or out-of-data character introductions under sym INT3 — `"3X-RED-1984"` and `"XRZ5-BLUE"` on Llama-3.1-8B, `"LMQ3-GREEN"` and `"XZQ7-RED-"` on Qwen2-7B, `"WZQ7-BLUE"` and `"JQZ5-ORAN"` on Qwen2.5-14B. Across 60 v0.3 asym INT3 trials at 32 K, **zero such hallucinations occur**. The pattern is consistent with the Phase-23 V-cache mechanism: sym INT3's per-channel window, fit without sufficient asymmetric tightness on the biased direction-vector distribution, lets per-token V reconstruction error compound through 32 K attention weights, and the attention-weighted output occasionally synthesizes codes that interpolate across the three needles' V subspaces. The v0.3 recipe's tighter asymmetric fit reduces the per-token V error budget enough that the label-to-code binding survives intact.

**Open observation: 8 K anomaly on Qwen2-7B (Mode 1).** At 8 K context on the Mode-1 canonical Qwen2-7B alone, v0.3 asym (4 / 10 pass) slightly underperforms v0.2 sym (5 / 10 pass), with v0.3 producing within-needle digit-corruption patterns such as `"LMN5-GREEN-2037"` (should be `LMN3-GREEN-2037`) rather than the cross-needle fusions seen under v0.2 at 32 K. This anomaly does not appear at 32 K on Qwen2-7B (where v0.3 7 / 10 cleanly beats v0.2 3 / 10) nor at 8 K on the other Qwen-family model Qwen2.5-14B (where v0.3 is a perfect 10 / 10). It is consistent with an interaction between Mode-1 K-cache outlier structure (Qwen2-7B's L0 K-CV of 0.472 is the highest in our 8-model set) and per-channel $(m, M)$ estimation variance at short context. We flag it as a Mode-1-specific short-context observation for future probing; it does not disturb the 32 K long-context conclusion.

## References

- *Norm Separation is Necessary* (preprint draft, `paper/nsn.md`): this is the authoritative writeup; v0.19+ integrates Phase 22 / 23 / 24 / 26 results described in this document.
- Companion paper (2024 mechanism study): [Norm-Separated Quantization](https://github.com/metaSATOKEN/norm-separated-quantization) (V5, 23 pages), for the Mode 1 / Mode 2 framing and the 27-model bit-width survey.
- v0.3 empirical data (head): `results/phase10_real_kv_distribution.json`, `phase13_full_sweep.json`, `phase14_asym_wikitext2_ppl.json`, `phase15_real_compression.json`, `phase16_asym_seven_model_sweep.json`, `phase17_2026_flagships.json`.
- v0.3 ablation + anomaly probe: `results/phase18_ablation_vs_industry_naive.json`, `phase19_ablation_generalization.json`, `phase20_bf16_disambiguation.json`, `phase21_gemma_b_anomaly_probe.json`.
- v0.3 downstream + mechanism: `results/phase22_mmlu_downstream.json` (MMLU 5-shot), `phase23_norm_dispersion.json` (V-cache CV measurement), `phase24b_nih_multineedle.json`, `phase24c_nih_multineedle_qwen2.json`, `phase24d_nih_multineedle_qwen25_14b.json` (multi-needle NIH 32 K).
