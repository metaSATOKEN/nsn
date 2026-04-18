# modeq

**Mode-aware KV cache quantization for Pre-LN Transformers.** A practical INT3 default that cuts WikiText-2 ΔPPL by 58–74% vs the symmetric baseline at the same 3.0 bits/value, with 5.09× real-packed memory reduction and 10 lines of code.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-green.svg)](#installation)

---

## TL;DR

Naive INT4 KV quantization can fail catastrophically on certain models — **ΔPPL +1382 on Qwen2-7B** — while working fine on others. modeq identifies two empirically separable failure modes (Mode 1: Layer-0 outlier-driven; Mode 2: distributed) and ships a simple fix that beats the symmetric INT3 baseline by a large margin on both.

> **v0.3 (this release):** Switches the default INT3 quantizer to **asymmetric per-channel nsep+pchan** (`qa_nsep_pchan_asym`). At identical 3.0 bits/value, the v0.3 default cuts WikiText-2 ΔPPL by an average of **66.0% (range 50.5%–95.8%) across 8 models spanning 2021–2026** — including Qwen2-7B (−74.3%), Llama-3.1-8B (−58.3%), Pythia-6.9B (−95.8%), **Qwen3.5-9B hybrid attention (−65.7%)**, and **Gemma 4-E4B (−50.5%)**. **All 8 land in the `< 5% of baseline PPL` production tier**, including v0.2's "research only" Pythia-6.9B. No calibration, no codebook, no rotation — just using the full per-channel `[min, max]` range instead of the symmetric `[-absmax, +absmax]` window that wastes roughly half the quantization levels on biased post-RoPE K distributions.

```python
from modeq import Int3AsymNsepPchanCache

# Encode a key/value tensor [B, H, T, D]  (v0.3 default)
cached = Int3AsymNsepPchanCache.from_fp16(k)

# Reconstruct for attention
k_recon = cached.to_fp16()

# Actual bytes on disk / VRAM — 5.09× smaller than FP16
cached.size_bytes()
```

Four lines. No training. No calibration. Rescues Qwen2-7B's pathological INT4 case (+1382 → +0.10 ΔPPL) and cuts Llama-3.1-8B INT3 drift by more than half (+0.25 → +0.11) at the same bit budget.

> **Note on prior art.** Asymmetric per-channel quantization is standard in weight quantization (GPTQ, AWQ, etc.); the norm–direction split itself is prior art as of [PolarQuant (2025)](https://arxiv.org/abs/2502.02617) / [TurboQuant (ICLR 2026)](https://arxiv.org/abs/2504.19874). modeq's contribution is the **empirical demonstration that composing these two pieces on the KV cache, with no calibration or codebook, captures 58–74% of the available INT3-tier PPL headroom** — plus the failure-mode diagnosis that motivated looking there in the first place. See [Related work](#related-work-and-how-modeq-differs).

---

## The benchmark that motivates this library

WikiText-2 perplexity increase (ΔPPL) vs bit width on 7 models, naive absmax (no modeq):

| Model | base PPL | INT2 | INT3 | **INT4** | INT5 | INT6 | INT8 |
|:------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **Qwen2-7B** (Mode 1) | 8.37 | +122 | +12 | **+1382** | +5 | +12 | +0 |
| Pythia-6.9B (Mode 2) | 10.76 | +164 | +22 | +7 | +2 | +0.5 | +0 |
| Llama-3.1-8B (Mode 2) | 8.94 | +88 | +0.4 | +0.05 | +0.01 | +0 | +0 |
| Qwen2.5-14B (Mode 2) | 1.96 | +7 | +0.9 | +0.2 | +0.03 | +0.01 | +0 |
| Mistral-7B (Mode 2) | 6.30 | +2.7 | +0.09 | +0.02 | +0 | +0 | +0 |
| GPT-J-6B (Mode 2) | 10.29 | +6 | +0.3 | +0.03 | +0 | +0 | +0 |
| Phi-3-medium (Mode 2) | 2.13 | +5 | +0.2 | +0.03 | +0.01 | +0 | +0 |

One of these is not like the others. **Qwen2-7B at INT4 is 295× worse than at INT5, and 116× worse than at INT3.** This is the INT4 valley of death.

### Two failure modes, causally separated

We ran a selective-compression experiment: compress only Layer 0, leave every other layer uncompressed.

| Regime (Qwen2-7B) | ΔPPL |
|:------|:--|
| Naive INT4 on all 28 layers | +1382.07 |
| **Naive INT4 on layers 1–27, nsep+pchan on Layer 0 only** | **+0.38** (99.97% rescue) |
| Naive INT4 on Layer 0 only, nsep+pchan on layers 1–27 | +1368.18 (catastrophe preserved) |

Layer 0 is the single point of failure. Fixing it alone rescues essentially all the damage. On three other failing models (Pythia-6.9B, Pythia-12B, Qwen2.5-14B) the opposite holds: Layer 0 contributes < 5% of the damage.

This is what we mean by two modes:

- **Mode 1** — Layer-0-localized, outlier-driven. Naive INT4 breaks catastrophically. Rare (1/27 models we've tested).
- **Mode 2** — Distributed, norm-variation-driven. Naive INT scales monotonically with bit width. Common (26/27 models).

Both are handled by the same preprocessing: **norm separation + per-channel scaling** (`nsep+pchan`). v0.3 adds **asymmetric per-channel scaling** on top, which cuts ΔPPL by another large factor at the same bit budget.

---

## v0.3 headline: asymmetric INT3 crushes symmetric INT3

Applied to the KV cache across all layers, WikiText-2 sliding-window PPL (ctx=1024, 10 windows):

Initial Phase 14 measurement on the canonical Mode 1 / Mode 2 pair:

| Model | Method | bits/value | ΔPPL | vs sym INT3 |
|:------|:---|:-:|:-:|:-:|
| **Qwen2-7B** (Mode 1) | FP16 baseline | 16 | 0 | — |
|  | sym INT3 (v0.2 default) | 3.000 | **+0.4021** | — |
|  | **asym INT3 (v0.3 default)** | **3.000** | **+0.1034** | **−74.3%** 🔥 |
|  | sym INT4 | 4.000 | +0.0544 | −86.5% |
| **Llama-3.1-8B** (Mode 2) | FP16 baseline | 16 | 0 | — |
|  | sym INT3 (v0.2 default) | 3.000 | **+0.2541** | — |
|  | **asym INT3 (v0.3 default)** | **3.000** | **+0.1059** | **−58.3%** 🔥 |
|  | sym INT4 | 4.000 | +0.0133 | −94.8% |

The asym ↔ sym difference at **identical 3.0 bits/value** is the v0.3 headline. INT4 still wins on quality in absolute terms, but it costs 33% more bits; asym INT3 sits on the Pareto front for the INT3 tier.

**Where does the gain come from?** Post-RoPE K channels have strongly biased distributions. Symmetric absmax quantizes them on `[-|max|, +|max|]`, wasting roughly half the quantization levels on the empty side. Asymmetric fits `[min, max]` tightly and uses all 2^bits levels. At INT3 this is the difference between 7 symmetric levels (−3..3) and 8 asymmetric levels (0..7), plus the range-fit advantage.

### Generalization: 8-model sweep (2021–2026 coverage)

Phase 16 extended Phase 14 to four additional models from the v0.2 sweep. Phase 17 then added two 2026 flagships (Qwen3.5-9B, Gemma 4-E4B). Combined, this gives an **8-model, 2021–2026 validation** of the asym INT3 claim (Phi-3-medium-4k-instruct was the intended 7th Phase-16 model but failed to load under current `transformers` due to custom-code compat issues unrelated to modeq — see [Phi-3 note](#phi-3-medium-compatibility-note) below).

| Model | Mode | base PPL | sym INT3 Δ | **asym INT3 Δ** | reduction | asym % of base |
|:------|:-:|:-:|:-:|:-:|:-:|:-:|
| Qwen2-7B | Mode 1 | 8.37 | +0.4021 | **+0.1034** | **−74.3%** | 1.24% ✅ |
| Llama-3.1-8B | Mode 2 | 7.38 | +0.2541 | **+0.1059** | **−58.3%** | 1.43% ✅ |
| **Pythia-6.9B** | Mode 2 | 10.76 | +2.3781 | **+0.0994** | **−95.8%** 🚀 | 0.92% ✅ |
| Qwen2.5-14B | Mode 2 | 1.96 | +0.2608 | **+0.0909** | **−65.1%** | 4.64% ✅ |
| Mistral-7B-v0.1 | Mode 2 | 6.30 | +0.1243 | **+0.0547** | **−56.0%** | 0.87% ✅ |
| GPT-J-6B (head_dim=256) | Mode 2 | 10.29 | +0.1873 | **+0.0703** | **−62.5%** | 0.68% ✅ |
| **Qwen3.5-9B** (2026, hybrid attn) | — | 10.32 | +0.2750 | **+0.0942** | **−65.7%** | 0.91% ✅ |
| **Gemma 4-E4B** (2026, multimodal) | — | 20.60 | +1.8786 | **+0.9302** | **−50.5%** 🚀 | 4.51% ✅ |

**Aggregate (8 models): asym beats sym on 8/8 (100%), mean reduction −66.0%, range −50.5% to −95.8%, all 8 land in the < 5% of baseline production tier.**

Three specific callouts:

- **Pythia-6.9B jumps from "research only" to production.** Under v0.2's symmetric INT3 it drifted +22.1% of baseline (our "✗ research only" tier). Under v0.3's asymmetric INT3 at identical 3.0 bits/value it drifts +0.92% of baseline — the largest relative improvement in the 8-model sample and a clean move into the production tier.
- **Gemma 4-E4B is the second dramatic rescue.** Symmetric INT3 drifts 9.12% of baseline (research-only tier); asymmetric INT3 brings it to 4.51% (production tier). Largest absolute ΔPPL improvement (+0.948) in the set.
- **Qwen3.5-9B is a hybrid architecture** (8 standard-attention layers + 24 linear-attention layers, the latter having no K/V cache). modeq compresses only the standard-attention subset, and even so asym INT3 cuts ΔPPL by 65.7% vs symmetric at identical bit rate. The benefit is layer-local — it doesn't require full-model compression.
- **GPT-J-6B has head_dim=256, not 128.** Its −62.5% reduction confirms the asym recipe is head-dim agnostic at the per-channel level.

### Real-packed compression

Phase 15 measured bytes via `Int3AsymNsepPchanCache` on realistic config dimensions:

| Model config | FP16 | v0.2 sym INT3 | **v0.3 asym INT3** | v0.3 ratio |
|:------|:-:|:-:|:-:|:-:|
| Qwen2-7B (28 layers × 4 KV heads × 2048 ctx) | 112 MB | 21.93 MB | **21.98 MB** | **5.09×** |
| Llama-3.1-8B (32 layers × 8 KV heads × 2048 ctx) | 256 MB | 50.12 MB | **50.25 MB** | **5.09×** |

The asymmetric variant costs only **0.008 bits/value more** than symmetric (one extra FP16 per channel for `d_min`) — essentially identical compression, but with 58–74% less PPL drift. A practical free lunch.

Extrapolating the Llama ratio:

| Context | FP16 | v0.3 asym INT3 | Saves |
|:-:|:-:|:-:|:-:|
| 8K | 1.02 GB | 200 MB | 822 MB |
| 32K | 4.10 GB | 803 MB | 3.31 GB |
| 128K | 16.4 GB | 3.21 GB | 13.2 GB |

---

## INT3 at production quality (8 of 8 models, 2021–2026)

Applying `qa_nsep_pchan_asym` at INT3 — 3 bits per value, 5.09× compression vs FP16. Phase 14 + Phase 16 + Phase 17 summary:

| Model | Release | Δ PPL (v0.3 asym) | % of baseline | Compression | Verdict |
|:------|:--|:--|:--|:--|:--|
| GPT-J-6B | 2021 | +0.0703 | 0.68% | 5.09× (extrap.) | ✅ production |
| Pythia-6.9B | 2023 | +0.0994 | 0.92% | 5.09× (extrap.) | ✅ production |
| Mistral-7B-v0.1 | 2023 | +0.0547 | 0.87% | 5.09× (extrap.) | ✅ production |
| Qwen2-7B (Mode 1, rescued) | 2024 | +0.1034 | 1.24% | 5.09× (measured) | ✅ production |
| Qwen2.5-14B | 2024 | +0.0909 | 4.64% | 5.09× (extrap.) | ✅ production |
| Llama-3.1-8B | 2024 | +0.1059 | 1.43% | 5.09× (measured) | ✅ production |
| **Qwen3.5-9B** (hybrid attn) | **2026-02** | **+0.0942** | **0.91%** | 5.09× (extrap.) | ✅ production |
| **Gemma 4-E4B** (multimodal) | **2026-04** | **+0.9302** | **4.51%** | 5.09× (extrap.) | ✅ production |
| Phi-3-medium-4k-instruct | 2024 | — | — | — | ⚠ HF-compat (pending, see note) |

**8 of 8 runnable models cross our production threshold (< 5% baseline) under v0.3's asymmetric INT3 default, spanning GPT-J (2021) through Gemma 4 / Qwen3.5 (2026).** By the same threshold, v0.2's symmetric INT3 had 4 of 7 in production and 1 in "research only" (Pythia-6.9B at +22.1% baseline drift); v0.3 moves Pythia and two 2026 flagships (Qwen3.5, Gemma 4) fully into production at identical 3.0 bits/value.

### Phi-3-medium compatibility note

Phi-3-medium-4k-instruct uses a bundled `modeling_phi3.py` in its HF repo (commit `48d87cd5`) that is incompatible with current `transformers`. Loading it produces three successive errors as each patch exposes the next:

1. `KeyError: config.rope_scaling["type"]` — recent `transformers` renamed the key to `rope_type`.
2. After backfilling `type`: `ValueError: Unknown RoPE scaling type default` — Phi-3's custom code only handles `"su"` and `"yarn"`, not `transformers`' default fallback.
3. After disabling `rope_scaling`: `AttributeError: type object 'DynamicCache' has no attribute 'from_legacy_cache'` — Phi-3's code uses a cache API that was removed in recent `transformers`.

This is unrelated to modeq — the same bundled code breaks on any quantization project using current `transformers`. A clean fix would require either (a) pinning `transformers` to a version contemporaneous with the Phi-3 commit, (b) a bundled-code update from Microsoft, or (c) re-running against a maintained Phi-3 sibling (Phi-3.5-mini, Phi-3-small). We defer to v0.3.3+.

### What we mean by "production quality"

The "Verdict" column classifies by a specific, reproducible threshold on WikiText-2 ΔPPL.

| Tier | WikiText-2 ΔPPL criterion | Interpretation |
|:---|:---|:---|
| ✅ **production** | < 5% of baseline PPL | Conservative proxy for "acceptable for typical KV-compressed deployment". Downstream-task drift is not measured. |
| 🟡 **borderline** | 5–10% of baseline PPL | Usable in memory-constrained settings; monitor downstream task quality before deployment. |
| △ **acceptable** | 10–20% of baseline PPL | Quality drop measurable; acceptable trade-off if memory is the dominant constraint. |
| ✗ **research only** | > 20% of baseline PPL | Compression too aggressive for this model at this bit width; treat as an ablation, not a deployment recommendation. |

**Caveats that apply to any "production" verdict here:**

1. **Downstream task accuracy** (MMLU, GSM-8K, HumanEval, RAG) was not measured. WikiText-2 ΔPPL is a proxy.
2. **Inference throughput / latency** was not measured. Our implementation is Python-only and adds roughly +21% decode latency on 7B-class models. A fused CUDA kernel would remove this.
3. **Sample size for the INT3 asym claim** is 8 runnable models (Qwen2-7B, Llama-3.1-8B from Phase 14; Pythia-6.9B, Qwen2.5-14B, Mistral-7B-v0.1, GPT-J-6B from Phase 16; Qwen3.5-9B, Gemma 4-E4B from Phase 17). Phi-3-medium-4k-instruct was the intended 7th Phase-16 model and is pending an HF custom-code / transformers compatibility fix.

---

## How `nsep+pchan` works

Three ideas combined:

**Norm separation** — decompose each vector into its L2 length (stored exactly) and its unit direction (quantized). Removes token-wise dynamic-range variation before quantization. Addresses Mode 2.

**Per-channel scaling** — quantize each channel with its own scale instead of a per-row scale. Prevents an outlier channel from dominating the quantizer. Addresses Mode 1.

**Asymmetric per-channel (v0.3)** — store `d_min` and `d_step = (d_max − d_min) / (2^bits − 1)` per channel, quantize to unsigned indices `[0, 2^bits − 1]`. Fits biased post-RoPE distributions instead of wasting the empty side of a symmetric window. Costs one extra FP16 per channel and ~0.008 bits/value.

Neither `nsep` nor `pchan` alone is sufficient; their conjunction rescues Qwen2-7B's 1382 ΔPPL catastrophe down to +0.40 at symmetric INT3. Adding asymmetry drops it further, to +0.10 at the same bit rate.

The full diagnosis, including the 11-model K/V outlier asymmetry study and the causal-isolation experiment, is in the companion paper:

> **Sato, K.** "Norm-Separated Quantization: A Training-Free Fix for KV Cache INT4 Failures." 2026.
> [Zenodo DOI / paper PDF](https://github.com/metaSATOKEN/norm-separated-quantization)

---

## Installation

```bash
# From source (pip release pending)
git clone https://github.com/metaSATOKEN/modeq
cd modeq
pip install -e .

# For running benchmark scripts in experiments/
pip install -e ".[eval]"
```

Requires Python 3.9+ and PyTorch 2.0+.

---

## Usage

### Quantize a single tensor (simulated, v0.3 default)

```python
import torch
from modeq import qa_nsep_pchan_asym

k = torch.randn(1, 8, 2048, 128, dtype=torch.float16, device="cuda")  # [B, H, T, D]
k_q = qa_nsep_pchan_asym(k, bits=3)   # v0.3 default; same shape/dtype
```

### Real INT3 bit-packing (5.09× memory, v0.3 default)

```python
from modeq import Int3AsymNsepPchanCache

k = torch.randn(1, 8, 2048, 128, dtype=torch.float16, device="cuda")

# Encode
cached = Int3AsymNsepPchanCache.from_fp16(k)

# Reconstruct when reading from cache
k_recon = cached.to_fp16()

# Storage footprint
print(f"FP16: {k.element_size() * k.numel() / 1024:.1f} KB")
print(f"INT3: {cached.size_bytes() / 1024:.1f} KB")  # ~5.09× smaller
```

The v0.2 symmetric class (`Int3NsepPchanCache`) is still exported for backwards compatibility.

### Classify a new model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from modeq import measure_kv_outlier_ratio, classify_mode

model = AutoModelForCausalLM.from_pretrained("...")
tokenizer = AutoTokenizer.from_pretrained("...")

stats = measure_kv_outlier_ratio(model, tokenizer)
verdict = classify_mode(stats)
print(verdict["mode"], "-", verdict["recommendation"])
```

### Mode-aware dispatch

```python
from modeq import measure_kv_outlier_ratio, select_quantizer, describe_selection

stats = measure_kv_outlier_ratio(model, tokenizer)
quantize = select_quantizer(outlier_stats=stats, target_bits=3)
# v0.3: every mode at INT3 -> asym nsep+pchan.
# v0.2 residual mechanism still dispatched on Mode 1 + 4-bit budget.

x_q = quantize(k)

print(describe_selection(outlier_stats=stats, target_bits=3))
# {
#   "method": "nsep+pchan INT3 asymmetric (v0.3 default)",
#   "mode": "Mode 1",
#   "method_rationale": "INT3 tier: asymmetric per-channel (min + range) ...
#                        reduces WikiText-2 ΔPPL by 74.3% on Qwen2-7B (Mode 1)
#                        and 58.3% on Llama-3.1-8B (Mode 2) vs the symmetric
#                        baseline.",
#   "estimated_quality": "~ +0.10 ΔPPL on Qwen2-7B (Phase 14 measurement)",
#   ...
# }
```

---

## v0.3 release notes

Three empirical phases drove v0.3 (see `experiments/phase10` through `phase15`):

1. **Phase 10**: we evaluated Walsh–Hadamard rotation (à la QuaRot/KVLinC) as a way to spread Mode 1 outliers across channels. On real Qwen2-7B Layer 0 K, rotation **regressed MSE by 3.68×** (0/4 heads improved) because the distribution is a discrete multi-channel plateau rather than heavy-tailed — per-channel absmax already handles plateaus optimally. Rotation is now NOT dispatched in v0.3 (the quantizer function `qa_nsep_pchan_rotated` remains in the package as a documented negative result).

2. **Phase 13**: a full sweep of candidate variants (2-tier bit allocation, max- vs energy-based top-K selection, asymmetric, combinations) on real Qwen2-7B and Llama-3.1-8B Layer 0 K identified **asymmetric per-channel** as the single contribution responsible for essentially all of the available INT3-tier MSE headroom. 2-tier adds < 1% on top of asym.

3. **Phase 14**: WikiText-2 PPL confirmed the MSE → PPL transfer at **80–85% efficiency** (MSE −91% → PPL −74% on Qwen2-7B; MSE −68% → PPL −58% on Llama). This is in the normal range for quantization methods.

4. **Phase 15**: real-packed compression measurement via `Int3AsymNsepPchanCache` confirmed **5.09× compression ratio** for both target models; the asym metadata overhead is 0.008 bits/value.

### Honesty note on Mode 2 at INT4

Phase 14 also found that **asymmetric INT4 is *worse* than symmetric INT4 on Llama-3.1-8B** (+0.027 vs +0.013 ΔPPL) while tying on Qwen2-7B. Above INT3, the zero-centering of symmetric quantization appears to carry value that asym's extra level cannot make up for. Accordingly the v0.3 dispatcher uses asym at INT3 only and returns to symmetric for INT4+.

---

## v0.2: Per-channel RMS residual correction (+mode-aware dispatch)

Kept available in v0.3 for the 4-bit Mode 1 path. Phase 7 & 8 studies in `experiments/` validated a 1-bit residual correction step over `nsep+pchan`.

1. **Per-channel RMS magnitude is essential.** A fixed constant magnitude (e.g., 0.25, the E[|residual|] under uniform assumption) *backfires* on Mode 1 — it worsens INT2 ΔPPL from +326 to +463 on Qwen2-7B. The per-channel RMS adapts to the non-uniform Mode 1 residual distribution.

2. **INT3 + RMS residual surpasses INT4 on Mode 1 only.** On Qwen2-7B: ΔPPL +0.025 vs native INT4 +0.054 (2.2× precision). On Mode 2 (6 of 7 models): INT3+RMS improves over INT3 alone but doesn't close the gap to native INT4.

v0.3 dispatcher keeps this Mode-1-at-4-bit rule:

| Target bits | Mode 1 | Mode 2 / Unknown |
|:---|:---|:---|
| 2 | `qa_nsep_pchan_resid_rms(x, 2)` | `qa_nsep_pchan_resid_rms(x, 2)` |
| **3** | **`qa_nsep_pchan_asym(x, 3)` (v0.3)** | **`qa_nsep_pchan_asym(x, 3)` (v0.3)** |
| 4 | `qa_nsep_pchan_resid_rms(x, 3)` | `qa_nsep_pchan(x, 4)` |
| 5+ | `qa_nsep_pchan(x, bits)` | `qa_nsep_pchan(x, bits)` |

See `results/phase7_residual_correction.json`, `results/phase8_rms_residual_validation.json`, `results/phase14_asym_wikitext2_ppl.json`, `results/phase15_real_compression.json` for the full data behind this table.

---

## Related work and how modeq differs

### Asymmetric quantization is standard; norm–direction split is prior art

**Asymmetric per-channel quantization** (store min and range, map to unsigned indices) is standard in weight quantization: GPTQ, AWQ, RTN variants, LLM.int8(). Applied to KV cache it is less universally emphasized — KIVI uses asymmetric per-channel for K and per-token for V at INT2; KVQuant uses calibrated non-uniform quantization. modeq's v0.3 combines this standard technique with the `nsep+pchan` norm-decomposition pipeline from our v0.2 release and shows that the combination, with no calibration, captures 58–74% of the INT3-tier PPL headroom on real target models.

**Norm–direction decomposition** originates with **PolarQuant** ([arXiv:2502.02617](https://arxiv.org/abs/2502.02617), 2025): the scalar radius $r = \|x\|_2$ is stored in full precision, and the direction is represented as a hierarchy of polar angles quantized via k-means codebooks after a random orthogonal rotation. PolarQuant reports 4.2× compression on Llama-3.1-8B. **TurboQuant** ([ICLR 2026](https://arxiv.org/abs/2504.19874)) extends PolarQuant with a 1-bit QJL residual, reaching 6× compression at 3 bits/coord with a data-oblivious recipe.

**KVLinC** ([arXiv:2510.05373](https://arxiv.org/abs/2510.05373), Oct 2025): Hadamard rotation + lightweight linear correction adapters for KV quantization on Llama/Qwen2.5/Qwen3. Uses learned adapters on top of rotation.

### Differences

| | PolarQuant | TurboQuant | KVLinC | modeq v0.3 |
|:---|:---|:---|:---|:---|
| Norm stored exact | ✓ | ✓ | — | ✓ |
| Rotation | Random orthogonal | Random orthogonal | Hadamard | **Not used** (Phase 10 showed it regresses on real KV) |
| Direction quantization | k-means on polar angles | Lloyd-Max scalar + QJL residual | Rotated scalar + adapter | **Asymmetric per-channel absmax** |
| Data dependence | Requires codebook fit | **Data-oblivious** | Requires adapter training | **Data-independent** (per-tensor min/max only) |
| Calibration required | Yes | No | Yes (adapter) | **No** |
| Primary contribution | Polar codebook compression | Theoretical optimality | Adapter-based correction | **Failure-mode diagnosis + minimalist recipe** |

### What modeq contributes that isn't in prior work

1. **Two empirically separable failure modes.** Naive INT KV quantization fails in two distinct ways: Mode 1 (Layer-0 outlier-driven, catastrophic INT4 valley) and Mode 2 (distributed norm-variation). See [docs/THEORY.md](docs/THEORY.md). PolarQuant, TurboQuant, KVLinC do not separate or diagnose these modes.

2. **The INT4 valley of death.** Naive INT shows a non-monotonic ΔPPL vs bit-width curve on Mode 1 models: INT4 is catastrophically worse than both INT3 and INT5 (by 295× on Qwen2-7B). This grid-alignment interaction between outlier magnitude and quantization step size is not, to our knowledge, previously characterized.

3. **Layer-0 causal isolation.** On Qwen2-7B, applying `nsep+pchan` to Layer 0 alone rescues 99.97% of the catastrophic ΔPPL. No prior KV quantization work performs this kind of selective-layer intervention to isolate the site of failure.

4. **A practical INT3-tier default.** `qa_nsep_pchan_asym` is data-independent, requires no codebook, no rotation, no training. On the two models we evaluated end-to-end (Qwen2-7B Mode 1 + Llama-3.1-8B Mode 2), it cuts WikiText-2 ΔPPL by 58–74% vs the symmetric baseline at identical 3.0 bits/value — complementary rather than competitive with the more sophisticated compression machinery of PolarQuant/TurboQuant/KVLinC.

5. **A documented negative result on rotation.** Phase 10 measured Walsh-Hadamard rotation on real Qwen2-7B Layer 0 K and found a 3.68× MSE regression. The distribution is a discrete multi-channel plateau, and per-channel absmax is already optimal on plateaus. This is a concrete data point against universal rotation-based recipes on per-channel-scaled pipelines.

### Other related methods

- **KIVI** ([ICML 2024](https://arxiv.org/abs/2402.02750)): asymmetric 2-bit Key (per-channel) + Value (per-token) quantization. No norm separation.
- **KVQuant** ([NeurIPS 2024](https://www.stat.berkeley.edu/~mmahoney/pubs/neurips-2024-kvquant.pdf)): per-channel Key pre-RoPE quantization with calibration. Concurrent with KIVI.
- **SQuat** ([arXiv:2503.24358](https://arxiv.org/abs/2503.24358), 2025): constrains quantization error to be orthogonal to the query subspace. Complementary.

### Honest summary

**Compression engineering**: PolarQuant, TurboQuant, and KVLinC squeeze out more bits via codebooks / residuals / adapters. modeq does not compete with them at the 1–2 bit frontier.

**INT3-tier practical SOTA**: modeq v0.3 is a competitive, calibration-free, 10-line-of-code recipe that delivers 58–74% ΔPPL reduction over the symmetric nsep+pchan baseline at identical bit rate. That is the niche modeq serves.

**Diagnosis and mechanism**: modeq is uniquely positioned — we show *when and why* naive INT quantization breaks, with causal experiments. That's the gap in the literature modeq tries to fill.

---

## Status and limitations

This is a **v0.3 research prototype** (April 2026). It works, the numbers are reproducible, and we use it — but:

- **Python overhead**: current implementation adds ~+21% decode latency on 7B-class models because quantize/dequantize happens in Python. A fused CUDA kernel (Triton or native) would remove this; contributions welcome.
- **Not yet a drop-in for HuggingFace Transformers**: you wire `Int3AsymNsepPchanCache` against the model's `DynamicCache` manually for now. Proper integration is on the roadmap.
- **v0.3 PPL measured on 8 models** (Phase 14 + Phase 16 + Phase 17): GPT-J-6B (2021), Pythia-6.9B / Mistral-7B-v0.1 (2023), Qwen2-7B / Qwen2.5-14B / Llama-3.1-8B (2024), Qwen3.5-9B / Gemma 4-E4B (2026). 8 of 8 land in the production tier, all beating the symmetric baseline by 50–96%. Phi-3-medium-4k-instruct was the intended 7th Phase-16 model but is blocked by an HF custom-code / transformers version incompatibility unrelated to modeq.
- **Real bit-packing measured on 2 models**: `Int3AsymNsepPchanCache.size_bytes()` empirically validated on Qwen2-7B and Llama-3.1-8B (5.09× measured). The other 4 models were validated in simulation (quantize → dequantize → WikiText-2 PPL); their real-packed ratios would be essentially identical (within 0.01 bits/value for head_dim=128, slightly higher for GPT-J-6B's head_dim=256 due to per-channel metadata overhead).
- **Sample size for Mode 1**: only one catastrophic case (Qwen2-7B) in our 27-model sample. The Layer-0 mechanism is causally isolated, but how often the pattern occurs in the open-model population is unknown. Safe default: always use `qa_nsep_pchan_asym`, which handles both modes.
- **Pre-LN only**: Post-LN models (GPT-1 style) are not evaluated.
- **Long-context validation pending**: all PPL numbers are at ctx=1024 sliding window. 32K+ is a roadmap item.

---

## Citing this work

If you use modeq or its findings, please cite:

```bibtex
@software{sato2026modeq,
  author = {Sato, Kentaro},
  title  = {modeq: Mode-Aware KV Cache Quantization for Pre-LN Transformers},
  year   = {2026},
  url    = {https://github.com/metaSATOKEN/modeq}
}

@article{sato2026normsep,
  author = {Sato, Kentaro},
  title  = {Norm-Separated Quantization: A Training-Free Fix for KV Cache INT4 Failures},
  year   = {2026},
  doi    = {10.5281/zenodo.19602981}
}
```

---

## License

Apache License 2.0. See [LICENSE](LICENSE).
