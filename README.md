# modeq

**Mode-aware KV cache quantization for Pre-LN Transformers.** Get INT3 compression with production quality, across every model family we tested.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-green.svg)](#installation)

---

## TL;DR

Naive INT4 quantization of KV caches can fail catastrophically on certain models — **ΔPPL +1382 on Qwen2-7B** — while working fine on others. We found two empirically separable failure modes and built a universal fix that delivers **INT3 compression with 5.08× real memory reduction at production quality** on both.

```python
from modeq import Int3NsepPchanCache

# Encode a key/value tensor [B, H, T, D] with INT3 packing
cached = Int3NsepPchanCache.from_fp16(k)

# Reconstruct for attention
k_recon = cached.to_fp16()

# Actual bytes on disk / VRAM
cached.size_bytes()   # 5.08x smaller than k.nbytes
```

4 lines of code. No training. No calibration. Works on Qwen2-7B's pathological case AND on everything else.

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

Applying `nsep+pchan` at INT3 — 3 bits per value, 5.33× compression vs FP16 — with our real bit-packing implementation:

| Model | Δ PPL | % of baseline | Compression | Verdict |
|:------|:--|:--|:--|:--|
| Qwen2-7B (Mode 1, rescued) | +0.39 | 4.6% | 5.08× | ✅ production |
| Llama-3.1-8B (Mode 2) | +0.27 | 3.0% | 5.08× | ✅ production |
| Mistral-7B-v0.1 | +0.12 | 2.0% | — | ✅ production |
| GPT-J-6B | +0.19 | 1.8% | — | ✅ production |
| Phi-3-medium-4k | +0.11 | 5.2% | — | 🟡 borderline |
| Qwen2.5-14B | +0.26 | 13.3% | — | △ acceptable |
| Pythia-6.9B (base) | +2.38 | 22.1% | — | ✗ research only |

5 of 7 models land in production quality. 1 is borderline. 1 (Pythia, a base model without instruction tuning) is research-only.

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

Simulated vs real packing: **PPL difference < 0.02** on both Mode 1 and Mode 2 (within numerical noise). Packing is lossless.

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

## Status and limitations

This is a v0.1 research prototype. It works, the numbers are real, and we use it ourselves — but:

- **Python overhead**: current implementation adds ~+21% decode latency on 7B-class models because quantize/dequantize happens in Python. A fused CUDA kernel (Triton or native) would remove this; we welcome contributions.
- **Not a drop-in for HuggingFace Transformers yet**: you have to wire up `Int3NsepPchanCache` against the model's `DynamicCache` manually for now. A proper integration is on the roadmap.
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
