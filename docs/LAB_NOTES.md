# Bit-Width Quantization Study (working title)

Prototype project. Will migrate to a real repo if Phase 1 confirms the hypothesis.

## Origin

Follow-up to the norm-separated-quantization paper (April 2026). That paper
observed an "INT3 > INT4 anomaly" on Qwen2-7B:

  - INT3 naive: ΔPPL +6.6
  - INT4 naive: ΔPPL +238
  - INT4 nsep+pchan: ΔPPL +0.32

The paper explains this as "cleaner zeroing at INT3 vs noisy non-zero
quantization at INT4 under outlier-dominated scales", but does not sweep the
full bit-width curve.

## Core Hypothesis

**Mode 1 failures (Layer-0-localized, outlier-driven) exhibit a U-shaped
ΔPPL-vs-bitwidth curve under naive INT quantization.**

Intuition: at very high precision (INT8+), normal channels are faithfully
represented and the outlier is one noisy entry. At very low precision
(INT2-INT3), most normal values round to zero, and attention is only
mildly perturbed. At middle precision (INT4-INT5) is the "worst of both":
normal values quantize to noisy nonzero levels that accumulate through
softmax.

Corollary: **nsep+pchan should be monotonic** (lower bits → more error,
smoothly), since it removes the outlier-domination and each bit is then
just about precision.

## Practical Goal

If confirmed, this opens the door to **sub-INT4 KV cache quantization**
(INT3, INT2) with nsep+pchan preprocessing, potentially achieving 8x
memory reduction vs FP16 while staying within usable ΔPPL bounds.

## Phase 1 (this experiment)

Bit-width sweep on Qwen2-7B (known Mode 1 case):
- bits: 2, 3, 4, 5, 6, 8
- methods: naive absmax, nsep+pchan
- metric: WikiText-2 PPL (5120 target tokens)

Expected: naive shows U-shape, nsep+pchan is monotonic.

## Next phases (pending Phase 1)

- Phase 2: Mode 2 models (Pythia-6.9B, Qwen2.5-14B) → confirm U-shape is Mode-1-specific
- Phase 3: INT2 viability survey across 15+ models
- Phase 4: HF transformers integration + pip-installable library
- Phase 5: Release & community
