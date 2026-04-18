# modeq

**Mode-aware KV cache quantization for Pre-LN Transformers.** Reach our "production" quality threshold (WikiText-2 ΔPPL < 5% of baseline) at INT3 on most model families we tested.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-green.svg)](#installation)

---

## TL;DR

Naive INT4 quantization of KV caches can fail catastrophically on certain models — **ΔPPL +1382 on Qwen2-7B** — while working fine on others. We identify **two empirically separable failure modes** (Mode 1: Layer-0 outlier-driven; Mode 2: distributed norm-variation-driven) and provide a simple, data-independent baseline that handles both: **INT3 with 5.08× real memory reduction** (measured on 2 models, extrapolated on the rest), landing within our "production" threshold (WikiText-2 ΔPPL < 5% of baseline) on **4 of 7 models tested** (a 5th is borderline at 5.2%) — see [What we mean by "production quality"](#what-we-mean-by-production-quality) for the precise criterion and caveats.

> **v0.2 (this release):** adds per-channel RMS residual correction (`qa_nsep_pchan_resid_rms`) and a mode-aware dispatcher (`select_quantizer`). On Mode 1 at roughly 4 bits/value + a small per-channel metadata (the RMS magnitudes), INT3 + RMS residual surpasses native INT4 by 2.2× precision on WikiText-2 ΔPPL.

```python
from modeq import Int3NsepPchanCache

# Encode a key/value tensor [B, H, T, D] with INT3 packing
cached = Int3NsepPchanCache.from_fp16(k)

# Reconstruct for attention
k_recon = cached.to_fp16()

# Actual bytes on disk / VRAM
cached.size_bytes()   # 5.08x smaller than k.nbytes
```

4 lines of code. No training. No calibration. No rotation. Rescues Qwen2-7B's pathological INT4 case (+1382 → +0.39 ΔPPL).

> **Note on prior art.** The norm-vs-direction split itself is not new: [PolarQuant (2025)](https://arxiv.org/abs/2502.02617) and [TurboQuant (ICLR 2026)](https://arxiv.org/abs/2504.19874) also store the L2 magnitude exactly and quantize the direction. modeq is deliberately simpler (per-channel absmax, no codebook, no rotation), and our main contribution is the **failure-mode diagnosis + causal mechanism** rather than the quantizer itself. See [Related work](#related-work-and-how-modeq-differs).

---

## The benchmark that motivates this library

WikiText-2 perplexity increase (ΔPPL) vs bit width on 7 models, naive absmax:

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
| **Naive INT4 on layers 1-27, nsep+pchan on Layer 0 only** | **+0.38** (99.97% rescue) |
| Naive INT4 on Layer 0 only, nsep+pchan on layers 1-27 | +1368.18 (catastrophe preserved) |

Layer 0 is the single point of failure. Fixing it alone rescues essentially all the damage. On three other failing models (Pythia-6.9B, Pythia-12B, Qwen2.5-14B) the opposite holds: Layer 0 contributes < 5% of the damage, and the rest is distributed across layers.

This is what we mean by two modes:

- **Mode 1** — Layer-0-localized, outlier-driven. Naive INT4 breaks catastrophically. Rare (1/27 models we've tested).
- **Mode 2** — Distributed, norm-variation-driven. Naive INT scales monotonically with bit width. Common (26/27 models).

Both are handled by the same preprocessing: **norm separation + per-channel scaling** (`nsep+pchan`).

---

## INT3 at production quality

Applying `nsep+pchan` at INT3 — 3 bits per value, 5.33× compression vs FP16 in theory. The ΔPPL column is WikiText-2 teacher-forced PPL with the KV cache compressed per method. The Compression column is **measured** real-packed bytes for the 2 models where we ran the full `Int3NsepPchanCache` path; for the other 5 models we list "—" because those PPL runs used simulated (fake) quantization, and the real-packing ratio would be the same 5.08× subject only to head-dim variation.

| Model | Δ PPL | % of baseline | Compression (measured) | Verdict |
|:------|:--|:--|:--|:--|
| Qwen2-7B (Mode 1, rescued) | +0.39 | 4.6% | 5.08× | ✅ production |
| Llama-3.1-8B (Mode 2) | +0.27 | 3.0% | 5.08× | ✅ production |
| Mistral-7B-v0.1 | +0.12 | 2.0% | (extrap.) | ✅ production |
| GPT-J-6B | +0.19 | 1.8% | (extrap.) | ✅ production |
| Phi-3-medium-4k | +0.11 | 5.2% | (extrap.) | 🟡 borderline |
| Qwen2.5-14B | +0.26 | 13.3% | (extrap.) | △ acceptable |
| Pythia-6.9B (base) | +2.38 | 22.1% | (extrap.) | ✗ research only |

**4 of 7 models cross our production threshold (< 5% baseline)**; a 5th (Phi-3-medium at 5.2%) sits just above it and is marked borderline. 1 is acceptable, and 1 (Pythia-6.9B, a base model without instruction tuning) is research-only. If a less strict absolute threshold is used (e.g., ΔPPL < 0.5 irrespective of baseline), 6 of 7 qualify; see the caveat on baseline dependence below.

### What we mean by "production quality"

The "Verdict" column above classifies by a specific, reproducible threshold on WikiText-2 ΔPPL. Here's what each tier means in this repository:

| Tier | WikiText-2 ΔPPL criterion | Interpretation |
|:---|:---|:---|
| ✅ **production** | < 5% of baseline PPL | We use this as a conservative proxy for "acceptable for typical KV-compressed deployment", in rough analogy to the PPL-drift budgets that INT4 KIVI/KVQuant methods report. In our sample this corresponds to < 0.5 absolute ΔPPL. Downstream-task drift is not measured. |
| 🟡 **borderline** | 5–10% of baseline PPL | Usable in memory-constrained settings, but we would recommend monitoring downstream task quality before deployment. |
| △ **acceptable** | 10–20% of baseline PPL | Quality drop measurable; acceptable trade-off if memory is the dominant constraint. |
| ✗ **research only** | > 20% of baseline PPL | Compression too aggressive for this model at this bit width; treat results as an ablation, not a deployment recommendation. |

**What this verdict does NOT cover** (important caveats):

1. **Downstream task accuracy** was not measured. WikiText-2 ΔPPL is a proxy; on tasks like MMLU, GSM-8K, HumanEval, or RAG accuracy, the actual quality drop may be larger or smaller than what ΔPPL suggests. KVQuant and KIVI report downstream-task accuracy; we do not, in v0.2.
2. **Inference throughput / latency** was not measured. Our implementation is Python-only and adds roughly +21% decode latency on 7B-class models. A fused CUDA kernel (not yet implemented) should remove this; until then, the "production" label applies to quality, not speed.
3. **Sample size**: one catastrophic model (Qwen2-7B) and seven total in the bit-width sweep. Claims about Mode 1 rescue rest on n=1 causally demonstrated; see the companion paper's Appendix H for the full diagnosis.
4. **Baseline dependence**: the "% of baseline PPL" metric penalizes very-low-baseline (high-capability) models. Qwen2.5-14B's ΔPPL +0.26 is tiny in absolute terms; it reads as 13% only because the baseline PPL is 1.96. An absolute-ΔPPL threshold (e.g., < 0.5) would classify most of our Mode 2 models as production-ready.

A model marked ✅ here means "our measurements suggest this should be acceptable at INT3 for many deployment use cases." It does not mean "we certified this model in your application." Evaluate on your own downstream task before shipping.

**Real memory, not simulated.** On Llama-3.1-8B with a 2048-token context:

```
FP16 KV cache:        92.01 MB
INT3-packed KV cache: 18.10 MB
Compression ratio:    5.08x (theoretical limit is 5.33x; our 4.7% overhead is FP16 norm/scale storage)
```

Extrapolating:

| Context | FP16 | INT3-packed | Saves |
|:-:|:-:|:-:|:-:|
| 8K | 368 MB | 72 MB | 296 MB |
| 32K | 1472 MB | 288 MB | 1.18 GB |
| 128K | 5888 MB | 1152 MB | 4.74 GB |

Simulated vs real packing: the bit-packing itself (pack → unpack on INT3 values) is bit-exact lossless; the end-to-end PPL difference between simulated and packed pipelines is **+0.000213 on Llama-3.1-8B (Mode 2)** and **+0.015 on Qwen2-7B (Mode 1)**. The small Mode 1 drift comes from storing the per-channel scales in FP16 after packing — the packed pipeline's effective precision is slightly different from the simulated path, but the effect is <0.02 ΔPPL on both tested models.

---

## Installation

```bash
# From source (pip release pending)
git clone https://github.com/metaSATOKEN/modeq
cd modeq
pip install -e .

# For running the benchmark scripts in experiments/
pip install -e ".[eval]"
```

Requires Python 3.9+ and PyTorch 2.0+.

---

## Usage

### Quantize a single tensor (simulated)

```python
import torch
from modeq import qa_nsep_pchan

k = torch.randn(1, 8, 2048, 128, dtype=torch.float16, device="cuda")  # [B, H, T, D]
k_q = qa_nsep_pchan(k, bits=3)   # ΔPPL-equivalent reconstruction, same shape
```

### Real INT3 bit-packing (5.08× memory)

```python
from modeq import Int3NsepPchanCache

k = torch.randn(1, 8, 2048, 128, dtype=torch.float16, device="cuda")

# Encode
cached = Int3NsepPchanCache.from_fp16(k)

# Reconstruct when reading from cache
k_recon = cached.to_fp16()

# Storage footprint
print(f"FP16: {k.element_size() * k.numel() / 1024:.1f} KB")
print(f"INT3: {cached.size_bytes() / 1024:.1f} KB")
```

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

### Mode-aware dispatch (v0.2)

Rather than manually choosing between `qa_nsep_pchan(x, 3)` and `qa_nsep_pchan_resid_rms(x, 3)` for your model, let modeq pick based on a measured outlier profile.

```python
from modeq import measure_kv_outlier_ratio, select_quantizer, describe_selection

stats = measure_kv_outlier_ratio(model, tokenizer)
quantize = select_quantizer(outlier_stats=stats, target_bits=4)
# On Mode 1 models (K @ L0 >= 15x): returns INT3 + per-channel RMS residual.
# On Mode 2 models: returns plain INT4 nsep+pchan (strictly better on Mode 2).

x_q = quantize(k)   # drop-in per-tensor compression

# Human-readable rationale
print(describe_selection(outlier_stats=stats, target_bits=4))
# {
#   "method": "nsep+pchan INT3 + per-channel RMS residual",
#   "mode": "Mode 1",
#   "method_rationale": "Mode 1, 4-bit budget. INT3 + RMS residual avoids
#                        the INT4 valley entirely and surpasses native
#                        INT4 (+0.025 vs +0.054 ΔPPL on Qwen2-7B, 2.2x
#                        precision).",
#   ...
# }
```

---

## v0.2: Per-channel RMS residual correction (+mode-aware dispatch)

The Phase 7 & 8 studies in `experiments/` validated a 1-bit residual correction step over `nsep+pchan`. Two findings shaped this v0.2 release:

1. **Per-channel RMS magnitude is essential.** A fixed constant magnitude (e.g., 0.25, the E[|residual|] under uniform assumption) *BACKFIRES* on Mode 1 models — it worsens INT2 ΔPPL from +326 to +463 on Qwen2-7B. The per-channel RMS adapts to the (highly non-uniform on Mode 1) residual distribution.

2. **INT3 + RMS residual surpasses INT4 on Mode 1 only.** On Qwen2-7B (Mode 1): ΔPPL +0.025 vs native INT4 +0.054 (2.2× precision). On Mode 2 (6 of 7 models): INT3+RMS improves over INT3 alone (16–72% reduction) but does not close the gap to native INT4. This is internally consistent with the two-mode theory: INT4 has a valley of death on Mode 1, so INT3+RMS can win there; no valley on Mode 2, so native INT4's cleaner levels win.

Together these imply a **mode-aware deployment rule**:

| Target bits | Mode 1 | Mode 2 / Unknown |
|:---|:---|:---|
| 3 | `qa_nsep_pchan(x, 3)` | `qa_nsep_pchan(x, 3)` |
| 4 | `qa_nsep_pchan_resid_rms(x, 3)` | `qa_nsep_pchan(x, 4)` |
| 5+ | `qa_nsep_pchan(x, bits)` | `qa_nsep_pchan(x, bits)` |

`modeq.select_quantizer()` (above) encapsulates this rule.

See `results/phase7_residual_correction.json` and `results/phase8_rms_residual_validation.json` for the full 7-model data.

---

## How `nsep+pchan` works

Two ideas combined:

**Norm separation** — decompose each vector into its L2 length (stored exactly) and its unit direction (quantized). Removes token-wise dynamic-range variation from the direction before quantization. Addresses Mode 2.

**Per-channel scaling** — quantize each channel with its own absmax-based scale instead of a per-row scale. Prevents an outlier channel from dominating the scale and crushing normal channels. Addresses Mode 1.

Neither component alone is sufficient: `nsep` alone rescues Qwen2-7B by 4.1×; `pchan` alone by 2.4×; together, **744×**. The conjunction is mechanistically necessary because a failing model may be dominated by either mode.

The full story, including the 11-model K/V outlier asymmetry study and the causal-isolation experiment, is in the companion paper:

> **Sato, K.** "Norm-Separated Quantization: A Training-Free Fix for KV Cache INT4 Failures." 2026.  
> [Zenodo DOI / paper PDF](https://github.com/metaSATOKEN/norm-separated-quantization)

---

## Related work and how modeq differs

KV cache quantization is an active area. Here's where modeq sits relative to recent work, and what's genuinely new versus what overlaps.

### Norm–direction decomposition is not new

**PolarQuant** ([arXiv:2502.02617](https://arxiv.org/abs/2502.02617), 2025) pioneered separating the KV vector's L2 magnitude from its direction for quantization: the scalar radius $r = \|x\|_2$ is stored in full precision, and the direction is represented as a hierarchy of polar angles that are then quantized via k-means codebooks after a random orthogonal rotation. PolarQuant reports 4.2× compression on Llama-3.1-8B. **TurboQuant** ([ICLR 2026](https://hackaday.com/2026/04/09/turboquant-reducing-llm-memory-usage-with-vector-quantization/), Google) extends PolarQuant with a 1-bit QJL residual correction, reaching 6× compression at 3 bits/coord.

modeq's `nsep+pchan` shares the broad structure of *storing the L2 norm exactly + quantizing the direction*. That idea itself is **prior art as of PolarQuant (Feb 2025)**. What differs is the direction representation and the study's focus.

### Differences

| | PolarQuant | TurboQuant | modeq |
|:---|:---|:---|:---|
| Direction representation | Hierarchical polar angles over log₂(d) levels | Scalar coordinates after random rotation | Unit vector, per-channel absmax |
| Quantization | k-means codebook on angles | Lloyd-Max scalar quantizer + 1-bit QJL residual | Absmax rounding (per-channel) |
| Data-dependence | Requires codebook fit from data | **Data-oblivious** (Gaussian after rotation, Lloyd-Max analytical) | **Data-independent** (absmax only) |
| Rotation / preconditioning | Random orthogonal | Random orthogonal | Not used |
| Primary evaluation | Llama-3.1-8B | Various sizes, ICLR 2026 benchmarks | 7 models × 6 bit widths |
| Primary contribution | Compression engineering | Compression engineering + theoretical optimality | **Failure-mode diagnosis + mechanism** |

Correction to an earlier draft: TurboQuant is data-oblivious, relying on the analytic Gaussian distribution induced by random rotation. Only PolarQuant's angle codebook is data-dependent. We thank GPT-5 for catching this mischaracterization.

### What modeq contributes that isn't in prior work

The bulk of `modeq`'s research value is not in the quantizer itself (PolarQuant covers the norm–direction split) but in the empirical and mechanistic framework around it:

1. **Two empirically separable failure modes.** Naive INT KV quantization doesn't fail one way — there are two: Mode 1 (Layer-0 outlier-driven, catastrophic INT4 valley, 1/27 models in our sample) and Mode 2 (distributed norm-variation-driven, monotonic, the common case). See [docs/THEORY.md](docs/THEORY.md). PolarQuant and TurboQuant do not separate or diagnose these modes.

2. **The INT4 valley of death.** Naive INT shows a non-monotonic ΔPPL vs bit-width curve on Mode 1 models: INT4 is catastrophically worse than both INT3 and INT5 (by 295× on Qwen2-7B). This grid-alignment interaction between outlier magnitude and quantization step size is not, to our knowledge, previously characterized.

3. **Layer-0 causal isolation.** On Qwen2-7B, applying `nsep+pchan` to Layer 0 alone — while leaving the other 27 layers under naive INT4 — rescues 99.97% of the catastrophic ΔPPL. The inverse (naive on Layer 0, `nsep+pchan` elsewhere) reproduces the catastrophe. No prior KV quantization work performs this kind of selective-layer intervention to isolate the site of failure.

4. **A simple universal baseline.** `nsep+pchan` is data-independent, requires no codebook, no rotation. On the 7 models we evaluated, INT3 `nsep+pchan` meets our "production" threshold (ΔPPL < 5% of WikiText-2 baseline PPL) on 4 of 7 (a 5th is borderline at 5.2%), and rescues the one Mode 1 pathology. This complements, rather than competes with, the more sophisticated compression machinery of PolarQuant/TurboQuant.

### Other related methods

- **KIVI** ([ICML 2024](https://arxiv.org/abs/2402.02750)): asymmetric 2-bit Key (per-channel) + Value (per-token) quantization. No norm separation.
- **KVQuant** ([NeurIPS 2024](https://www.stat.berkeley.edu/~mmahoney/pubs/neurips-2024-kvquant.pdf)): per-channel Key pre-RoPE quantization with calibration. Concurrent with and similar in spirit to KIVI.
- **SQuat** ([arXiv:2503.24358](https://arxiv.org/abs/2503.24358), 2025): constrains quantization error to be orthogonal to the query subspace. A different angle from norm decomposition; complementary.
- **MiniKV** (ACL 2025): pushes to 2-bit via selective precision. Orthogonal technique.

### Honest summary

**Compression engineering**: PolarQuant and TurboQuant are ahead (more bits squeezed out via codebooks + residuals).

**Diagnosis and mechanism**: modeq is uniquely positioned — we show *when and why* naive INT quantization breaks, with causal experiments, and provide the simplest baseline that works across both failure modes. That's the gap in the literature modeq tries to fill.

## Status and limitations

This is a **v0.2 research prototype** (released April 2026). It works, the numbers are real, and we use it ourselves — but:

- **Python overhead**: current implementation adds ~+21% decode latency on 7B-class models because quantize/dequantize happens in Python. A fused CUDA kernel (Triton or native) would remove this; we welcome contributions.
- **Not a drop-in for HuggingFace Transformers yet**: you have to wire up `Int3NsepPchanCache` (or the v0.2 `qa_nsep_pchan_resid_rms`) against the model's `DynamicCache` manually for now. A proper integration is on the roadmap.
- **Real bit-packing measured on 2 models**: `Int3NsepPchanCache.size_bytes()` was empirically validated on Qwen2-7B and Llama-3.1-8B (5.08× measured). For the other 5 models in the INT3 verdict table, the PPL numbers come from simulated quantization; the 5.33× compression ratio is inferred from head_dim rather than measured.
- **Sample size for Mode 1**: only one catastrophic case (Qwen2-7B) in our 27-model sample. The Layer-0 mechanism is causally isolated, but how often this pattern occurs in the open-model population is unknown. Safe default: always use `nsep+pchan`, which handles both modes.
- **Pre-LN only**: Post-LN models (GPT-1 style) are not evaluated here. Our companion paper discusses distributional differences.

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
