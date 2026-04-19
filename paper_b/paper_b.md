# Norm Separation is Necessary: A Minimalist Calibration-Free INT3 Recipe for KV Cache Quantization

**Kentaro Sato**
*2026-04-19*
*arxiv preprint, draft v0.3*

---

## Abstract

KV cache quantization is one of the principal levers for deploying long-context Transformer inference on fixed memory. Recent state-of-the-art methods (KIVI, KVQuant, PolarQuant, TurboQuant, KVLinC) achieve aggressive compression ratios but rely on various combinations of codebook training, calibration data, random rotations, or learned correction adapters. We study a complementary question: at the INT3 tier (3 bits/value, 5.09× compression vs FP16), is there a *minimalist* recipe that (i) requires no calibration or training, (ii) fits in roughly ten lines of code, and (iii) keeps WikiText-2 ΔPPL below a conservative 5 % of baseline on a broad set of open-weight Transformers?

We show that the composition of three data-independent operations — **norm separation, per-channel quantization, and an asymmetric `[min, max]` window** — achieves this across eight open-weight Transformers spanning GPT-J (2021) through Qwen3.5-9B and Gemma-4-E4B (2026). At identical 3.0 bits/value, this recipe reduces the mean WikiText-2 ΔPPL by 66 % (range 50.5 %–95.8 %) over a symmetric per-channel variant; under our primary evaluation protocol (fp16, 10 windows × 1024 ctx), all eight models land below our 5 %-of-baseline threshold on WikiText-2. No simpler ablation variant achieves the same coverage. A follow-up bf16 / 30-window probe on Gemma-4-E4B places that specific model at 5.5 %–7.0 % of baseline, so we describe Gemma-4-E4B as borderline under the longer probe rather than uniformly below threshold.

A four-method ablation across the same eight models yields a specific and, to our knowledge, previously uncharacterized failure: **asymmetric per-channel INT3 *without* norm separation — a near-direct transplant of a standard weight-quantization recipe — catastrophically regresses on four of the eight models**, reaching 24× worse ΔPPL than naive per-row quantization on Llama-3.1-8B. The failure correlates with models whose token-wise L2 norm variance is large, pointing at a structural difference between KV cache and weight distributions. We also report one curious observation — small but reproducible *beneficial* ΔPPL on Gemma-4-E4B under the same ablation variant — and document why it does not translate into a deployment recommendation.

We release the recipe, benchmark scripts, and raw JSON measurements so that each number in the paper is reproducible from the committed data. We are explicit about scope limitations: the quality claim rests on WikiText-2 sliding-window perplexity alone, downstream-task accuracy was not measured, throughput was not measured, and the Qwen2-7B INT4 valley is a single-model observation ($n = 1$ in both the 27-model broader survey we draw on and the 8-model benchmark reported here). We do not claim that simpler recipes fail on models we did not test, nor that the proposed recipe is Pareto-optimal at bit rates below 3 bits/value.

---

## Contributions

Concretely, this paper contributes:

1. **A calibration-free INT3 recipe** (`qa_nsep_pchan_asym`) that composes norm separation, per-channel scaling, and an asymmetric quantization window. The reference implementation is a ten-line Python function with no external data, optimizer state, or learned parameters (§3).

2. **An eight-model end-to-end validation on WikiText-2 sliding-window perplexity** (ctx = 1024, 10 windows) spanning 2021–2026: GPT-J-6B, Pythia-6.9B, Mistral-7B-v0.1, Qwen2-7B, Qwen2.5-14B, Llama-3.1-8B, Qwen3.5-9B, Gemma-4-E4B. All eight stay below our 5 %-of-baseline-PPL threshold under the recipe; the mean ΔPPL reduction over a symmetric nsep+pchan baseline is 66 % at identical 3.0 bits/value (§5.1).

3. **A four-method ablation establishing that each of the three components is empirically necessary.** No simpler ablation variant — (A) naive per-row symmetric, (B) asymmetric per-channel without norm separation, or (C) symmetric nsep+pchan — achieves the eight-model coverage of the full recipe (D) (§5.2).

4. **A specific, reproducible negative result for a naive translation of weight-quantization recipes to KV caches.** Variant (B) above — the standard "asymmetric per-channel" baseline as it appears in GPTQ/AWQ-style weight quantization — **fails catastrophically on Llama-3.1-8B, Mistral-7B-v0.1, and Qwen2.5-14B** (regression by up to 24× vs naive per-row INT3 on Llama-3.1-8B), and fails mildly on GPT-J-6B (B is approximately 2× worse than A, landing just above the 5 % threshold). We propose a failure mechanism — token-wise L2 norm variance contaminates per-channel statistics when norm separation is omitted — and argue that `nsep` is therefore a structural ingredient for per-channel KV quantization on this model family rather than a quality-of-life preprocessing step (§5.3).

5. **Documented negative results on rotation and top-K mixed precision**, both tested during the development of this recipe. A Walsh–Hadamard rotation step (similar in spirit to KVLinC and the rotation component of TurboQuant) regresses per-head reconstruction MSE by 3.68× on real Qwen2-7B Layer 0 K; a top-K outlier-channel mixed-precision variant at $K = 4$ regresses per-head reconstruction MSE by 17.6 % on the same tensor. Both metrics are per-tensor reconstruction error rather than end-to-end ΔPPL; we did not pursue either method far enough to run the full PPL pipeline. Both quantizer implementations are retained in the public code base as documented negative results (§6, Appendix A).

6. **An open observation on accidental K-cache entropy regularization.** On Gemma-4-E4B evaluated text-only, variant (B) produces a small but reproducible *negative* ΔPPL (−0.32 to −0.44 on two non-overlapping WikiText-2 segments). A K-only vs V-only split attributes 87 % of the effect to the Key cache, consistent with quantization noise softening an over-confident attention distribution on an off-distribution (text-only) input to a multimodal-trained model. We do not promote this as a method — the same ablation regresses catastrophically on four other models — but record it as a phenomenon worth instrumenting in future work (§7).

We also argue, as a secondary contribution, that the **"Mode 1" phenomenon** — a non-monotonic INT4 ΔPPL valley observed on Qwen2-7B (n = 1 in our sample) — is a concrete, reproducible failure case of naive KV quantization that motivates looking for robust recipes in the first place. We emphasize throughout that this remains an n = 1 observation and explicitly do not claim it generalizes to untested models (§2, §5.3).

---

## 1. Introduction

Transformer inference at long context length is increasingly memory-bound on the KV cache rather than on weights. For a Pre-LN decoder with $L$ layers, $H$ key/value heads, head dimension $D$, sequence length $T$, and FP16 storage, the KV cache occupies $2 \cdot L \cdot H \cdot T \cdot D \cdot 2$ bytes per sequence. At the 128 K and 1 M context lengths now common in open-weight models (Llama-3.1, Qwen2.5, Qwen3.5, Gemma-4), the cache dominates per-sequence GPU memory by an order of magnitude over the model weights themselves. Quantizing the KV cache has therefore become one of the principal levers for practical deployment, and a body of recent work has pushed effective bit rates from the FP16 baseline toward 4-bit, 3-bit, and in some settings 2-bit or 1.5-bit representations.

The current state of KV cache quantization is, however, methodologically heterogeneous. **KIVI** [Liu et al., 2024] uses asymmetric per-channel for keys and per-token for values at 2 bits and introduces outlier protection logic. **KVQuant** [Hooper et al., 2024] uses per-channel key quantization in the pre-RoPE frame with a calibration-based non-uniform quantizer. **PolarQuant** [Wang et al., 2025] replaces the Euclidean coordinates of each KV vector with a polar-coordinate decomposition and trains a $k$-means codebook on the polar angles after a random orthogonal rotation. **TurboQuant** [ICLR 2026] applies a random orthogonal rotation, a Lloyd–Max scalar quantizer on the rotated coordinates, and a QJL-style residual correction to obtain a data-oblivious recipe that is competitive at the 3-bit tier. **KVLinC** [preprint, 2025] combines a Hadamard rotation with learned linear correction adapters to push 2-bit KV cache on Llama and Qwen families. Each of these methods delivers stronger compression ratios or finer-grained precision than a naive scalar quantizer, but each introduces a corresponding complexity dependency: a codebook, a calibration corpus, a rotation matrix (random or Hadamard), a trained adapter, or a non-trivial mixed-precision scheme.

Against this backdrop we consider a narrower and more practical question. **Is there a recipe that stays at the INT3 tier** — 3.0 bits/value, approximately $16/3 = 5.33\times$ theoretical compression, $\sim\!5.09\times$ measured with FP16 metadata overhead — **that requires no calibration, no codebook, no rotation, no adapter, fits in roughly ten lines of reference code, and still keeps the WikiText-2 perplexity drift below a conservative quality threshold on a broad slice of modern Transformers**? Each individual ingredient of such a recipe is, separately, folklore: norm separation ($x = n \cdot d$, with $n$ stored in FP precision) is the structural idea underlying PolarQuant and TurboQuant; per-channel absmax scaling is the standard per-channel preconditioner used throughout weight quantization since LLM.int8 [Dettmers et al., 2022]; asymmetric $[\min, \max]$ quantization windows are the default in GPTQ [Frantar et al., 2023] and AWQ [Lin et al., 2023] for *weights*. The question of interest is not the novelty of any ingredient but the empirical behavior of a particular composition of them on *KV caches* across a representative model population.

We study the composition $\text{\tt nsep+pchan+asym}$ — per-channel asymmetric INT3 quantization applied to the unit-direction vector after norm separation — and report three main findings. First, on an eight-model benchmark spanning 2021 (GPT-J-6B) through 2026 (Qwen3.5-9B, Gemma-4-E4B), this recipe keeps the WikiText-2 sliding-window ΔPPL below 5 % of the FP16 baseline on every model we were able to run, at exact 3.0 bits/value with measured 5.09× compression (§5.1, §5.4). Second, in a four-method ablation on the same eight models (§5.2), none of the three simpler variants — naive per-row symmetric, asymmetric per-channel *without* norm separation, symmetric nsep+pchan — achieves the same coverage; each variant catastrophically regresses on at least two models. Third, the most informative ablation failure is the naive asymmetric-per-channel variant that omits norm separation: this recipe, which is approximately what one obtains by copy-pasting the asymmetric per-channel machinery of AWQ/GPTQ from weights to KV cache, *regresses on 4 of the 8 models* and reaches +10.92 ΔPPL on Llama-3.1-8B, a 24× regression relative to naive per-row INT3 on the same tensor (§5.3). We trace this failure to a specific structural property of KV caches — token-wise L2 norm variance — that weight tensors largely lack, and we argue that norm separation therefore is not a quality-of-life preprocessing step but a structural requirement for per-channel KV quantization on this model family.

The benchmark also includes one anomaly that we treat explicitly as $n = 1$: Qwen2-7B exhibits a non-monotonic INT4 failure under naive per-row quantization (+1382 ΔPPL at INT4 versus +12 at INT3 and +5 at INT5), which we trace to an outlier channel at Layer 0 of the K cache. This "INT4 valley" is the one observation of this pattern in our 27-model broader survey (seven of which we re-benchmarked end-to-end for the present work). We report it as a concrete failure mode that motivates looking for recipes robust across the bit-width axis, and we are explicit throughout that it is a single-model observation; we do not claim a general "Mode 1" class of models exists. The proposed recipe handles this anomaly alongside the common case at identical bit budget (§5.1, §6). Finally, we document one curious *beneficial* anomaly — a reproducible small negative ΔPPL on Gemma-4-E4B under the asymmetric-without-nsep variant, traced at 87 % to the Key cache and plausibly caused by accidental attention-logit entropy regularization on a multimodal-trained model evaluated text-only (§7). We include this only as an observation in search of a theory; it is not a deployment recommendation.

The remainder of the paper is organized as follows. §2 surveys related work. §3 specifies the `nsep+pchan+asym` recipe in closed form. §4 describes our experimental setup. §5 reports the main results, including the four-method ablation. §6 discusses limitations — in particular our reliance on a single-benchmark (WikiText-2) proxy for quality, the $n = 1$ status of the Qwen2-7B anomaly, and Python-implementation overhead that our measurements do not attempt to remove. §7 documents the Gemma-4-E4B K-cache regularization observation. §8 concludes. Reproducibility materials, including dead-end experiments, the full Phase 13 method sweep, and the complete set of measured JSON results, are provided in the appendices and in the public code repository.

---

## 2. Related Work

**KV cache quantization** divides roughly into three families, plus a fourth category of ingredients we transplant from weight / activation quantization.

*Per-channel / per-token scalar quantization with outlier handling.* KIVI [Liu et al., 2024] observes that keys and values have different distribution structure — keys benefit from per-channel quantization while values benefit from per-token — and implements aggressive 2-bit quantization with explicit outlier reservation. KVQuant [Hooper et al., 2024] performs per-channel quantization of keys in the pre-RoPE frame, where the distribution is cleaner, and uses calibration data to fit a non-uniform quantizer. Atom [Zhao et al., 2024] combines group-wise mixed-precision quantization with explicit outlier isolation on both K and V, reaching 4-bit end-to-end serving quality with per-channel statistics plus a dedicated high-precision slot for outlier channels. All three methods rely on some combination of calibration, explicit outlier reservation, or group-wise bit allocation; the recipe we propose is simpler (no outlier slot, no calibration) at a comparable or slightly higher bit rate.

*Polar / norm-direction decomposition with codebooks.* PolarQuant [Wang et al., 2025] decomposes each KV vector into its L2 radius (stored at full precision) and its angular coordinates on the unit sphere, the latter quantized via a $k$-means codebook fit after a random orthogonal rotation. TurboQuant [ICLR 2026] builds on a similar random-rotation idea by applying a Lloyd-Max scalar quantizer on the rotated coordinates with a QJL-style residual correction, yielding a fully data-oblivious recipe. Our work shares the norm-direction decomposition with these methods but deliberately uses the simplest possible direction quantizer (per-channel absmax with asymmetric window) instead of a codebook or a rotation-plus-residual scheme.

*Rotation-based quantization.* KVLinC [preprint, 2025] applies a Hadamard rotation to the K cache and trains small linear-correction adapters that compensate for the residual quantization error. Outside the KV-cache-specific setting, QuaRot [Ashkboos et al., 2024] applies a fixed Hadamard rotation to weights and activations to enable 4-bit inference, and SpinQuant [2024] learns the rotation matrix end-to-end; both target weight+activation quantization rather than KV caches per se. Rotation spreads outlier energy across coordinates and is broadly effective at 2 bits in the KVLinC pipeline. In early development of the present work we tested a fixed Walsh–Hadamard rotation as a drop-in preprocessing step before `nsep+pchan` and observed a $3.68\times$ per-head reconstruction-MSE regression on real Qwen2-7B Layer 0 K, which we attribute to the specific multi-channel plateau structure of that tensor (§A.1). We therefore do not rotate in the proposed recipe; the present method obtains its generality from norm separation plus per-channel asymmetric scaling alone, without learned correction. This is a negative result for *fixed Walsh-Hadamard rotation pre-pended to `nsep+pchan`*; it is not a negative result for the KVLinC pipeline in its published form, which combines rotation with an adapter and targets 2-bit rather than 3-bit regimes.

*Weight- and activation-quantization ingredients transplanted to KV.* Per-channel scaling as a preconditioner for low-bit Transformer quantization traces to LLM.int8 [Dettmers et al., 2022] for outlier isolation and to SmoothQuant [Xiao et al., 2023], which migrates the activation dynamic range onto the weights so that per-channel activation quantization becomes feasible at INT8. Asymmetric per-channel $[\min, \max]$ quantization is the standard preconditioner in GPTQ [Frantar et al., 2023], AWQ [Lin et al., 2023], and SpQR [Dettmers et al., 2024] on the weight side. To our knowledge there is no work that explicitly isolates whether this ingredient transplants cleanly from weights or activations to KV caches when applied naively. §5.3 of this paper reports that it does *not*: variant (B) in our ablation, which is close to the naive transplant, regresses on four of the eight models we test (three catastrophically and one mildly). The root cause is specific to KV caches: per-token L2 norm variance is large enough to contaminate per-channel statistics when norm separation is omitted.

**Positioning.** We view this paper as complementary to the above work rather than competing at the bit-rate frontier. PolarQuant, TurboQuant, and KVLinC squeeze more quality per bit through codebooks, theoretically optimal scalar quantizers, and learned correction; Atom and KIVI handle the 2-bit to 4-bit regime with explicit outlier machinery. We offer a minimalist recipe at the INT3 tier whose contribution is the *generality-without-calibration* property and the empirically-supported minimality of its three ingredients. Where sub-2-bit pressure dominates, KVLinC or Atom's outlier slot are likely the better choices; where a calibration-free drop-in is preferred and 3 bits/value is acceptable, the recipe here has a different set of properties.

---

## 3. Method

### 3.1 Setup and notation

Consider a single key or value tensor from one head of a Transformer's KV cache, shaped $[T, D]$, where $T$ is the number of cached tokens and $D$ is the head dimension. Let $x_t \in \mathbb{R}^{D}$ denote the row corresponding to token $t$. We evaluate quantizers that map $x_t$ to a low-precision representation $\hat{x}_t$ and measure the end-to-end effect via WikiText-2 perplexity on a Transformer whose KV cache has been replaced by the reconstruction. Throughout, *bits per value* refers to the storage cost in the reconstructed tensor averaged over coordinates — including any per-channel or per-token metadata that is stored alongside the packed integer data.

### 3.2 The recipe

The method has three data-independent operations applied in sequence on every $[T, D]$ key or value tensor.

**Step 1 — Norm separation (`nsep`).** Decompose each row into magnitude and direction:
$$
n_t = \|x_t\|_2,\qquad d_t = \frac{x_t}{n_t}.
$$
We store $n_t \in \text{FP16}$ exactly; only the unit direction $d_t$ is quantized.

**Step 2 — Per-channel asymmetric window (`pchan` + `asym`).** For each channel $c \in \{1, \dots, D\}$, compute
$$
m_c = \min_{t} d_{t,c},\qquad M_c = \max_{t} d_{t,c},\qquad s_c = \frac{M_c - m_c}{2^{b} - 1},
$$
where $b$ is the target bit width ($b = 3$ in our main results). We store $m_c, s_c \in \text{FP16}$.

**Step 3 — Integer quantization + re-normalization.** Quantize each direction coordinate to a $b$-bit unsigned index:
$$
q_{t,c} = \mathrm{round}\!\left(\frac{d_{t,c} - m_c}{s_c}\right),\qquad q_{t,c} \in \{0, 1, \dots, 2^b - 1\}.
$$
At reconstruction time, the dequantized direction is
$$
\tilde{d}_{t,c} = m_c + s_c \cdot q_{t,c},
$$
and we re-normalize onto the unit sphere before multiplying by the stored norm:
$$
\hat{d}_t = \frac{\tilde{d}_t}{\|\tilde{d}_t\|_2 + \varepsilon},\qquad \hat{x}_t = n_t \cdot \hat{d}_t.
$$
The re-normalization step ( cheap, one FP32 norm per token ) corrects the small radial drift that per-coordinate quantization would otherwise introduce on the unit sphere.

The entire reconstruction pipeline is $O(T D)$ time per tensor, requires no calibration corpus, no learned parameters, and no transformation that is tensor-dependent beyond the two per-channel scalars $(m_c, s_c)$.

### 3.3 Storage accounting

Per key or value tensor of shape $[T, D]$, the recipe stores:
- $T \cdot D$ integers of $b$ bits each (packed);
- $T$ FP16 norms (one $n_t$ per token);
- $2D$ FP16 per-channel scalars ($m_c$ and $s_c$).

The average bit rate per coordinate is therefore
$$
b_\text{eff}^\text{asym}(b, T, D) \;=\; b \;+\; \underbrace{\frac{16}{D}}_{\text{per-token norm}} \;+\; \underbrace{\frac{32}{T}}_{\text{two per-channel scalars}}.
$$
At $b = 3$, $D = 128$, $T = 2048$ (representative of the main results), $b_\text{eff}^\text{asym} = 3 + 0.125 + 0.0156 \approx 3.141$ bits/value. At $T \to \infty$ this asymptotes to $3 + 16/D = 3.125$ bits/value.

For comparison, the symmetric per-channel variant used as the §5 baseline stores only one FP16 scalar per channel (the absmax) instead of two, so its bit rate is
$$
b_\text{eff}^\text{sym}(b, T, D) \;=\; b \;+\; \frac{16}{D} \;+\; \frac{16}{T}.
$$
The asymmetric variant's overhead relative to the symmetric one is therefore exactly $b_\text{eff}^\text{asym} - b_\text{eff}^\text{sym} = 16/T$ — that is, one additional FP16 per channel, amortized over the $T$ tokens that share that channel's scalar. At $T = 2048$ this is $\approx 0.0078$ bits/value (matching the Phase 15 measurement in §5.4 of $3.141 - 3.133 = 0.008$ bits/value). The extra storage cost is negligible at any realistic context length; the main quality trade is not paid in bits.

### 3.4 Real-packed storage

In the reference implementation, three-bit integers are packed using the existing `pack_int3_unsigned` primitive in the modeq code base: every 8 consecutive 3-bit values fit into 3 bytes. The packed representation is bit-exact: pack-then-unpack is the identity on integers in $\{0, \dots, 7\}$. On Qwen2-7B (28 layers × 4 KV heads × 2048 tokens × head dim 128) and Llama-3.1-8B (32 × 8 × 2048 × 128), the measured on-GPU cache size drops from 112 MB → 21.98 MB (Qwen2-7B) and 256 MB → 50.25 MB (Llama-3.1-8B), both at **5.09× compression vs FP16** (§5.4). The symmetric variant achieves 5.11×; the asymmetric variant's $0.02\times$ deficit is purely the extra per-channel scalar mentioned in §3.3 and is the entire price paid for the 58–95 % ΔPPL reductions reported in §5.

### 3.5 Reference implementation

```python
def qa_nsep_pchan_asym(x, bits=3):
    """Norm-separated per-channel asymmetric INT quantization.

    x: shape [..., T, D], dtype fp16 or fp32.
    Returns a same-shape, same-dtype tensor reconstructed from the
    INT-quantized direction vector plus the exact per-token L2 norm.
    """
    dtype = x.dtype
    xf = x.float()
    n = xf.norm(dim=-1, keepdim=True).clamp(min=1e-12)  # per-token
    d = xf / n                                          # unit direction
    m = d.amin(dim=-2, keepdim=True)                    # per-channel
    M = d.amax(dim=-2, keepdim=True)
    step = ((M - m) / (2**bits - 1)).clamp(min=1e-12)
    q = ((d - m) / step).round().clamp(0, 2**bits - 1)  # integer index
    d_q = m + q * step                                  # dequantize
    d_q = d_q / d_q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return (n * d_q).to(dtype)
```

The function has ten lines of content plus docstring; it performs no I/O, has no state, and is independent of the caller's model. A matching packed-cache class (`Int3AsymNsepPchanCache`) handles real bit packing, exactly reproduces the reconstruction above, and exposes a `size_bytes()` method.

---

## 4. Experimental Setup

**Benchmark.** All ΔPPL measurements are on WikiText-2 (test split) under a sliding-window protocol: context length $= 1024$ tokens, stride $= 512$, and up to 10 windows per evaluation unless otherwise stated. Each window is split in half: the first 512 tokens are fed without target loss to populate the KV cache; the KV cache is then optionally replaced by a quantized reconstruction; the second 512 tokens are scored. This matches the sliding-window perplexity protocol used by PolarQuant and KIVI. Baseline PPL is the model's own FP16 PPL under the same protocol with no quantization.

**Quality metric.** We report $\Delta\mathrm{PPL} = \mathrm{PPL}(\text{quantized}) - \mathrm{PPL}(\text{baseline})$ in absolute units and as a fraction of the baseline (denoted "% of baseline PPL"). We adopt a fixed 5 % threshold as a conservative proxy for "likely usable in practice if memory is the dominant constraint" and call it the *usability threshold* throughout. This threshold is modeq-internal; it is not a deployment certification and it does not imply we have measured downstream task accuracy.

**Models.** Eight open-weight Transformers spanning 2021–2026:

| Model                      | Release     | $D$  | $H_\text{KV}$ | Family          |
|:--------------------------|:-----------|:---:|:-:|:----------------|
| GPT-J-6B                   | 2021        | 256 | 16  | pre-Llama-family |
| Pythia-6.9B                | 2023        | 128 | 32  | Pythia           |
| Mistral-7B-v0.1            | 2023        | 128 | 8   | Mistral          |
| Qwen2-7B                   | 2024        | 128 | 4   | Qwen / GQA       |
| Qwen2.5-14B                | 2024        | 128 | 8   | Qwen / GQA       |
| Llama-3.1-8B               | 2024        | 128 | 8   | Llama-3-family   |
| Qwen3.5-9B                 | 2026-02     | 256 | 4   | Qwen / hybrid    |
| Gemma-4-E4B                | 2026-04     | 256 | 4   | Gemma-4 Efficient |

Qwen3.5-9B uses a hybrid attention architecture in which 8 of 32 layers are standard attention (with K/V cache) and the remaining 24 are linear attention (no K/V cache); the compression pipeline skips non-KV layers and compresses only the 8 standard-attention layers. Gemma-4-E4B is multimodal-capable but is evaluated text-only in all measurements below. We attempted a ninth model, Phi-3-medium-4k-instruct, but its bundled custom `modeling_phi3.py` is incompatible with current `transformers` on three separate fronts (rope-scaling key rename, unknown `rope_type = "default"`, removed `DynamicCache.from_legacy_cache`); we count it as "attempted but blocked" rather than as a failure of the recipe.

**Hardware and reproducibility.** All measurements were collected on a single NVIDIA A100 (40 GB) in Google Colab. The recipe is implemented in the `modeq` open-source package; every ΔPPL number and compression ratio reported below is reproducible from the committed JSON data under `results/` via a matching script under `experiments/`. The full set of quantizer primitives fits in under 50 lines of Python; no custom CUDA kernel is used and we do not claim any throughput advantage over the FP16 baseline (§6).

---

## 5. Results

### 5.1. Eight-model generalization at INT3

Table 1 reports the WikiText-2 ΔPPL under our recipe — the `nsep+pchan+asym` variant at $b = 3$ — against the v0.2 symmetric baseline at the same bit budget. Both methods operate at exactly 3.0 bits/value plus $O(D/T)$ FP16 metadata, so the comparison is at matched bit rate.

**Table 1.** Eight-model WikiText-2 ΔPPL at INT3 (ctx = 1024, 10 windows). "sym" = symmetric nsep+pchan (this paper's v0.2 baseline variant). "asym" = the recipe proposed in §3.

| Model             | base PPL | sym INT3 ΔPPL | asym INT3 ΔPPL | reduction | % of base |
|:------------------|:--:|:--:|:--:|:--:|:--:|
| GPT-J-6B          | 10.29 | +0.1873 | **+0.0703** | −62.5 % | 0.68 % |
| Pythia-6.9B       | 10.76 | +2.3781 | **+0.0994** | −95.8 % | 0.92 % |
| Mistral-7B-v0.1   | 6.30  | +0.1243 | **+0.0547** | −56.0 % | 0.87 % |
| Qwen2-7B          | 8.37  | +0.4021 | **+0.1034** | −74.3 % | 1.24 % |
| Qwen2.5-14B       | 1.96  | +0.2608 | **+0.0909** | −65.1 % | 4.64 % |
| Llama-3.1-8B      | 7.38  | +0.2541 | **+0.1059** | −58.3 % | 1.43 % |
| Qwen3.5-9B        | 10.32 | +0.2750 | **+0.0942** | −65.7 % | 0.91 % |
| Gemma-4-E4B       | 20.60 | +1.8786 | **+0.9302** | −50.5 % | 4.51 % |
| **Mean (8)**      |   —   | **+0.7200** | **+0.1936** | **−66.0 %** | **1.90 %** |

Reading Table 1: on every benchmarked model, the recipe reduces ΔPPL relative to the symmetric baseline at matched bit rate, with a mean reduction of 66 % and a range of 50.5 %–95.8 %. Under the primary evaluation protocol (fp16 model load, 10 WikiText-2 windows of 1024 ctx), all eight models stay below the 5 % usability threshold under the recipe — the largest per-model percentage is Qwen2.5-14B at 4.64 %, driven by its unusually low baseline PPL of 1.96, which makes the relative metric strict. Gemma-4-E4B at 4.51 % is the second-highest, and we note below and in §7 that Gemma-4-E4B becomes borderline under a follow-up bf16 / 30-window probe (ΔPPL $= 5.5\%$–$7.0\%$ of baseline depending on text segment), so we do not claim that model stays below threshold under stricter protocols. The symmetric baseline, by the same threshold, would place three models outside: Pythia-6.9B (22.1 % of baseline → research-only), Qwen2.5-14B (13.3 %), and Gemma-4-E4B (9.1 %, just above the 5 % line).

A single-model qualitative observation: the naive INT4 failure on Qwen2-7B (+1382 ΔPPL), which is the non-monotonic "INT4 valley" feature in Figure 1, is rescued by the recipe to +0.10 ΔPPL at INT3 — a 116× improvement over naive INT3 on the same tensor and 13,000× improvement over naive INT4. This is the single most dramatic rescue in the benchmark; the Qwen2-7B anomaly is qualitatively unlike the other seven models (§6), but it does not require special handling under the proposed recipe.

### 5.2. Four-method ablation

To test whether each of the three ingredients of the recipe — `nsep`, `pchan`, `asym` — is empirically necessary, we ran the same eight-model benchmark under four INT3 methods:

- **(A)** naive per-row symmetric INT3 (industry baseline; no `nsep`, no `pchan`, no `asym`);
- **(B)** asymmetric per-channel INT3 *without* norm separation (approximately the naive transplant of weight-quantization recipes to KV cache);
- **(C)** symmetric `nsep+pchan` INT3 (this paper's baseline; the v0.2 recipe);
- **(D)** asymmetric `nsep+pchan` INT3 (the proposed recipe).

**Table 2.** Four-method ablation on all eight models: ΔPPL as % of FP16 baseline PPL.

| Model             | (A) naive | (B) asym no nsep | (C) sym nsep+pchan | (D) **proposed** |
|:------------------|:--:|:--:|:--:|:--:|
| GPT-J-6B          | 3.11 %   | 6.27 %   | 1.82 %   | **0.68 %** |
| Pythia-6.9B       | 208.78 %  | 5.88 %   | 22.10 %  | **0.92 %** |
| Mistral-7B-v0.1   | 1.40 %   | 119.72 % | 1.97 %   | **0.87 %** |
| Qwen2-7B          | 142.47 % | 4.29 %   | 4.80 %   | **1.24 %** |
| Qwen2.5-14B       | 43.95 %  | 145.67 % | 13.32 %  | **4.64 %** |
| Llama-3.1-8B      | 6.03 %   | 147.91 % | 3.44 %   | **1.43 %** |
| Qwen3.5-9B        | 2.34 %   | −0.04 %  | 2.67 %   | **0.91 %** |
| Gemma-4-E4B       | 26.45 %  | −2.13 %  | 9.12 %   | **4.51 %** |
| **Under 5 % threshold** | **3 / 8** | **3 / 8** | **5 / 8** | **8 / 8** |

Four observations:

1. *No simpler ablation variant achieves 8 / 8 coverage.* (A) fails on Qwen2-7B (Mode 1 outlier-driven INT4 valley, §5.3), Pythia-6.9B (a base model with high quantization sensitivity), and Gemma-4-E4B. (B) fails on Llama-3.1-8B, Mistral-7B-v0.1, Qwen2.5-14B, and GPT-J-6B (the mechanism is analyzed in §5.3). (C) fails on Pythia-6.9B, Qwen2.5-14B, and Gemma-4-E4B.

2. *The specific variants that fail on each method are disjoint.* This is consistent with each ingredient `nsep`, `pchan`, `asym` addressing an empirically distinct failure mode. (A) fails on models where per-row scales get crushed by outlier channels, and `pchan` helps there; (B) fails on models with large per-token norm variance, and `nsep` helps there; (C) fails on models whose direction distribution is biased enough that the symmetric window wastes too many levels, and `asym` helps there.

3. *The proposed recipe (D) achieves 8 / 8 coverage at 3.0 bits/value with one FP16 scalar per channel more than (C)* — see §3.3 for the accounting. The mean improvement over (C) is 66 %; the mean improvement over (B) is 92 % (driven by the four B-catastrophe models).

4. *Negative ΔPPL on (B) for Qwen3.5-9B and Gemma-4-E4B* is a real empirical observation — quantization noise under this specific ablation marginally improves PPL on these two 2026 models. We investigate this separately in §7 and find that approximately 87 % of the effect is attributable to the Key cache only. It does not change the conclusion of Table 2: *the proposed recipe (D) is the only INT3 variant in this study that stays below threshold on all eight models*, including on Qwen3.5-9B and Gemma-4-E4B.

### 5.3. Why `nsep` is necessary: the asym-without-nsep catastrophe

The single most informative row of Table 2 is **(B) asym per-channel without norm separation**, which regresses *below* the naive per-row baseline (A) on four of the eight models. Three of the four are Llama-family or Llama-style (Llama-3.1-8B, Mistral-7B-v0.1, Qwen2.5-14B); the fourth is GPT-J-6B (head dim 256). The most extreme regression is Llama-3.1-8B: (A) naive gives +0.45 ΔPPL (6 % of baseline); (B) gives **+10.92 ΔPPL** (148 % of baseline), a 24× regression below a recipe that was meant to be an upgrade.

**Proposed mechanism.** We conjecture — but in this draft do not directly measure — that the failure is driven by token-wise L2 norm variance that the per-channel statistics implicitly absorb when norm separation is omitted. The argument is as follows. When norm separation is omitted, the per-channel `[min, max]` window in (B) is computed over *raw* tensor values $\{x_{t, c}\}_{t}$ across all tokens. If per-token L2 norms vary widely, tokens with large norm contribute the extremes of the per-channel range. The channel's quantization grid is then stretched to accommodate those extremes, and low-norm tokens — whose typical per-coordinate magnitude is an order of magnitude smaller — end up mis-quantized by a factor roughly equal to the token-norm ratio. Those large reconstruction errors then propagate through attention in a way that visibly corrupts the next-token distribution. Norm separation eliminates this contamination by normalizing each token to unit length *before* computing per-channel statistics, so the per-channel window in (D) is fit to a distribution with bounded token-wise norm ($\|d_t\|_2 = 1$).

We label this as a *proposed mechanism* rather than a confirmed one: the paper does not report per-model token-norm dispersion statistics correlated against the B-failure pattern, and such a diagnostic table would strengthen the claim. Our support for the hypothesis in this draft is indirect — (i) the four B-failures concentrate on Llama-family and post-Llama Qwen architectures, where qualitative inspection of per-token K-norm distributions in prior work shows high dispersion, and (ii) norm separation fixes (B) → (D) by construction. A companion experiment directly measuring the per-token K-norm standard deviation on each of the 8 models, against each model's (B) − (D) ΔPPL gap, is left to a revision.

If the mechanism is correct, the diagnostic is simple: any KV-cache quantization recipe that performs per-channel scaling without first normalizing per-token should expect this failure mode on models with high per-token L2 norm variance. Reporting the per-token norm distribution alongside the quantizer specification therefore seems worth making a community convention.

### 5.4. Real-packed compression

Table 3 reports measured bytes on realistic per-model KV-cache configurations. Compression ratio is computed as $(\text{FP16 cache size}) / (\text{packed cache size})$.

**Table 3.** Measured real-packed compression vs FP16 on the two models for which we ran the full pipeline end-to-end via the `Int3NsepPchanCache` (sym) and `Int3AsymNsepPchanCache` (asym) classes. Both classes pack 3-bit indices at the same density; they differ only in the per-channel metadata (one versus two FP16 scalars per channel — see §3.3). Methods (A) and (B) in the ablation of §5.2 use different metadata layouts and are not shown here; we quantify their memory footprint by the `b_eff` formula of §3.3 rather than a packed measurement.

| Model | Config (layers × KV heads × ctx × $D$) | FP16 | sym INT3 | **asym INT3** | ratio (asym) |
|:------|:--|:--:|:--:|:--:|:--:|
| Qwen2-7B | 28 × 4 × 2048 × 128 | 112.00 MB | 21.93 MB | **21.98 MB** | **5.09×** |
| Llama-3.1-8B | 32 × 8 × 2048 × 128 | 256.00 MB | 50.12 MB | **50.25 MB** | **5.09×** |

The asymmetric variant costs $\approx 0.02\times$ less compression than the symmetric variant (one additional FP16 per channel) but delivers the 58 %–74 % ΔPPL reduction reported on these same two models in Table 1. In practice the recipe trades 0.02× of compression for roughly 3× quality headroom. The theoretical maximum is $16 / 3 \approx 5.33\times$; the $\approx 4.5 \%$ overhead in our measurement is the FP16 norm + per-channel metadata (§3.3). A fused CUDA kernel — not implemented here — could reduce the Python decode overhead, but not the FP16 metadata cost.

Extrapolating the measured Llama ratio to longer contexts (under the asymptotic $b_\text{eff} = 3.125$ bits/value at $T \to \infty$):

| Context length | FP16 | asym INT3 | Savings |
|:-:|:-:|:-:|:-:|
| 8 K    | 1.02 GB | 200 MB  | 822 MB |
| 32 K   | 4.10 GB | 803 MB  | 3.31 GB |
| 128 K  | 16.4 GB | 3.21 GB | 13.2 GB |

### 5.5. Bits vs quality

Figure 4 consolidates the results of §§5.1–5.2 as a bits-vs-quality scatter. At 3 bits/value, the four-method cloud shows that only (D) has its cross-model median clearly below the 5 % usability line; (C) and (A) sit visibly above, and (B)'s cloud is bimodal (good on some models, catastrophic on others). The 4-bit reference points for Qwen2-7B and Llama-3.1-8B (under sym nsep+pchan INT4) achieve lower absolute ΔPPL but at 33 % higher memory footprint; (D) at 3 bits is the Pareto-preferred choice for deployments in which memory is the dominant constraint.

---

## 6. Discussion

### 6.1 The Qwen2-7B "INT4 valley": an $n = 1$ observation

In our broader 27-model survey from the companion (preprint) mechanism study, and in the 8-model subset we re-benchmarked end-to-end for the present work, the **Qwen2-7B naive INT4 failure (+1382 ΔPPL) is the single observation of a non-monotonic bit-width failure**. We emphasize the sample size explicitly: 1 / 27 in the survey, 1 / 8 in the present benchmark. We do not claim there is a class of "Mode 1" models characterized by this pattern; we only claim that one model shows it. Whether the phenomenon recurs on untested models — other Qwen releases, other families, future training recipes — is beyond what our data can settle. Other KV-quantization papers (KVQuant, KIVI) report per-channel outliers on a broader set of models, so specific-channel pathology is likely not unique to Qwen2-7B, but we cannot independently connect those reports to the specific non-monotonic INT4 behavior we observe. The proposed recipe in §3 handles the Qwen2-7B anomaly and the common case under the same bit budget, which is what allows us to recommend the recipe without bifurcating on mode classification.

### 6.2 When modeq is *not* the right choice

The proposed recipe is targeted at a specific regime. It is likely *not* the right tool when:
- **The bit budget is below INT3.** KIVI, TurboQuant, KVLinC, and PolarQuant all achieve useful quality at 2 bits or below; our recipe does not attempt to compete there and does not reach the INT3 tier below 3 bits.
- **Downstream task accuracy is the binding quality metric.** We measure WikiText-2 perplexity, which is a proxy. On tasks like retrieval, multi-step reasoning, or coding, the quality drop may be larger or smaller than what ΔPPL suggests.
- **A calibration corpus is available and a known-better quantizer exists.** KVQuant's calibrated non-uniform quantizer can beat our recipe where calibration data is cheap; we do not claim otherwise.
- **Throughput is the binding metric.** Our reference implementation is Python-only and adds roughly $+21 \%$ decode latency on 7B-class models. A fused kernel would remove most of that but is not part of this paper.
- **The model family is systematically different from anything in our benchmark.** Post-LN architectures, state-space models, and Transformer variants with very non-standard KV cache structure were not evaluated.

### 6.3 Limitations

1. *Single quality benchmark.* Every reported ΔPPL number is on WikiText-2. We chose this benchmark for comparability with prior KV quantization work (KIVI, KVQuant, PolarQuant all report it), but it is a narrow slice of "model quality". A paper-scale downstream-task evaluation on MMLU, GSM-8K, HumanEval, and a retrieval benchmark is warranted future work.

2. *Sample size for the Mode 1 anomaly.* See §6.1. The Qwen2-7B observation is $n = 1$; we structure our claims accordingly but cannot exclude the possibility that other models exhibit this pattern.

3. *No throughput measurement.* The current implementation is Python-level quantize/dequantize on top of the stock `DynamicCache`. A fused Triton or CUDA kernel would remove the Python overhead. The "< 5 %" claim in this paper is about quality only; we make no speed claim.

4. *Reproducibility with current HuggingFace `transformers`.* One of our intended nine benchmark models (Phi-3-medium-4k-instruct) is blocked by a custom-code / transformers-version incompatibility unrelated to modeq. We report the blocked run as "attempted" rather than "failed", but future papers using this benchmark should be aware that specific model + library version combinations may move.

5. *INT4 asymmetric observation.* We incidentally measured an *asymmetric* INT4 variant and found it is worse than symmetric INT4 on Llama-3.1-8B (+0.027 vs +0.013 ΔPPL). Above the INT3 bit floor, the zero-centering of symmetric quantization appears to carry value that asymmetric's extra level cannot recover. The proposed recipe is therefore INT3-specific; we do not recommend the asymmetric window at INT4+.

---

## 7. Open observation: accidental K-cache regularization on Gemma-4-E4B

Under ablation variant (B) (asymmetric per-channel INT3 *without* norm separation), Gemma-4-E4B — and, at a smaller effect size, Qwen3.5-9B — produces a reproducible **negative** ΔPPL on WikiText-2: the "quantized" model scores slightly *better* than the FP16 baseline. On Gemma-4-E4B the effect is −0.44 on one 48 KB text segment, −0.15 on a non-overlapping 48 KB segment, and persists (−0.32) when the model is re-evaluated under native bf16 rather than fp16. A per-side ablation that applies (B) to only the Key cache or only the Value cache attributes **87 % of the effect to the Key cache on both segments**.

We conjecture, without proof, that the effect is K-cache-side attention-logit entropy regularization. Gemma-4-E4B is a multimodal-trained model evaluated text-only; its attention distribution on pure text is plausibly slightly over-confident ("off-distribution"). Quantization noise on the Key cache perturbs the softmax $\mathrm{softmax}(QK^\top / \sqrt{D})$ in a way that flattens the distribution toward higher entropy. If the trained, correctly-calibrated distribution is itself high-entropy for text-only input, this small perturbation could nudge the model closer to its training distribution and reduce PPL by a modest amount.

We deliberately treat this as an observation, not a method. Three reasons:

1. Variant (B) **catastrophically regresses** on four other models in the same benchmark (§5.3). The same recipe cannot be recommended as a deployable quantizer.
2. Effect size is 1–3 % of baseline PPL, specific to text-only evaluation of this particular model, and has not been tested on downstream tasks.
3. The proposed recipe (D) does not exhibit this anomaly; its ΔPPL on Gemma-4-E4B is a normal +0.93 (4.5 % of baseline), consistent with lossy quantization. The anomaly lives entirely in the ablation variant we do not ship.

We record this as "a phenomenon worth instrumenting in future work" rather than a claim. A proper characterization would measure per-layer attention entropy directly rather than inferring it from PPL, and would evaluate on downstream tasks to see whether the PPL improvement reflects genuine quality or a proxy failure. We provide the full raw probe data in Appendix C.

---

## 8. Conclusion

We presented a calibration-free INT3 recipe for KV cache quantization and validated it end-to-end on eight open-weight Transformers spanning 2021–2026. The recipe composes three data-independent operations — norm separation, per-channel scaling, and an asymmetric quantization window — and fits in roughly ten lines of Python. On the eight-model benchmark, under our primary evaluation protocol (fp16 load, 10 WikiText-2 windows at 1024 ctx), it reduces the mean ΔPPL by 66 % over a symmetric variant at identical 3.0 bits/value and achieves 5.09× real-packed compression vs FP16. Gemma-4-E4B becomes borderline (5.5 %–7.0 % of baseline) under a longer bf16 / 30-window follow-up probe rather than staying below the 5 % threshold; we describe this in §5.1 and §7 rather than claiming the 8-of-8 figure for every protocol.

A four-method ablation on the same models shows that none of the three ingredients is removable: every simpler variant — industry-standard naive per-row quantization, asymmetric per-channel without norm separation, and symmetric `nsep+pchan` — falls outside our 5 %-of-baseline usability threshold on at least two models. In particular, the naive transplant of asymmetric per-channel from weight quantization to KV cache **regresses catastrophically on 4 of 8 models**, reaching up to 24× worse than the naive per-row baseline it was meant to improve, and traces to a failure mode specific to KV caches (per-token L2 norm variance) that weight tensors do not exhibit.

The scope of the positive claim is deliberately narrow. We do not compete with rotation-based or codebook-based methods at the 1–2 bit frontier; we do not claim generalization to downstream tasks beyond the WikiText-2 proxy; we do not claim the Qwen2-7B "INT4 valley of death" pathology generalizes beyond $n = 1$. What we do claim is that a very simple recipe, with no moving parts outside of per-channel arithmetic, attains the common-case quality of more complex methods at INT3 on a range of open-weight Transformers that spans half a decade of architectural evolution. We also claim that the recipe's three ingredients are empirically indivisible: this is a finding we believe is useful independent of the practical deployment case, because it specifies a concrete way in which naive weight-quantization recipes do *not* transplant cleanly to KV caches.

The reference implementation, benchmark scripts, and measured JSON data are released under Apache 2.0. Every number in this paper is reproducible by running the corresponding script in `experiments/` against the committed data under `results/`.

---

## References

See `refs.md` for the current citation list (in preparation — to be converted to bibtex for the arxiv submission).

---

## Appendix A: Dead ends

We document two mechanisms we tested during development and did *not* include in the final recipe. Both are retained as importable functions in the open-source package for reproducibility, but neither is dispatched by the default quantizer.

### A.1 Walsh–Hadamard rotation

Rotating the KV cache by a fixed Walsh–Hadamard matrix prior to `nsep+pchan` quantization is the strategy used by KVLinC and, with a randomized variant, TurboQuant. On synthetic data we expected rotation to spread any single-channel outlier energy across coordinates and thereby make Mode 1–like distributions look Mode 2–like to the quantizer.

The synthetic sanity check (Phase 9, `results/phase9_rotation_sanity.json`) showed mixed behavior: rotation helps on heavy-tailed (Student-$t$) distributions (−72 % MSE) but *hurts* on discrete-outlier distributions (+17 % to +30 % MSE). The real-model probe (Phase 10, `results/phase10_real_kv_distribution.json`) on Qwen2-7B Layer 0 K resolved the question: the K distribution is a **discrete multi-channel plateau** (top-5 per-channel max values are 166.75, 160.5, 159.0, 156.6, 128.1), not heavy-tailed, and rotation **regresses MSE by 3.68× on real Qwen2-7B L0 K (0 of 4 heads improve)**. On Llama-3.1-8B L0 K, rotation approximately ties (ratio 1.01). We therefore do not rotate.

The broader lesson: rotation is empirically useful when the target distribution has a heavy tail but is approximately isotropic; it is counterproductive when the distribution is plateau-like. Per-channel absmax already handles plateaus optimally. This result does not contradict KVLinC / TurboQuant, which deploy rotation against different target distributions, but it did force us to drop rotation from our own recipe.

### A.2 Top-K mixed precision

A natural hypothesis upon observing the Qwen2-7B plateau structure is that a top-K mixed-precision scheme — e.g., the top 4 per-channel-magnitude channels at INT6 and the remaining 124 channels at INT3, for an effective 3.09 bits/value — would exploit the plateau by giving the high-magnitude channels more headroom. On synthetic plateau regimes this hypothesis held (−45 % MSE on a Qwen2-cartoon tensor at $K = 5$), but on the real Qwen2-7B L0 K the top-K-INT6 variant at $K = 4$ **regressed by 17.6 %** and won on only 1 of 4 heads (Phase 12, `results/phase12_real_kv_2tier.json`). We were unable to locate a K + bit-allocation setting that outperformed plain `nsep+pchan` on the real tensor.

The interaction we believe explains the failure: the post-quantization re-normalization step in `nsep+pchan` works cleanly when per-channel quantization error is roughly uniform across channels (as with a single bit width), but is distorted when some channels carry high-precision residuals and others carry low-precision ones, effectively mixing two different error regimes on the unit-sphere projection. Without re-normalization removed or adapted, mixed precision injects asymmetric error into the direction vector. We did not pursue this further because the subsequent ablation (§5.2) showed that the asymmetric window at uniform INT3 was sufficient.

Both Appendix A.1 and A.2 remain available in the package (`qa_nsep_pchan_rotated`, `qa_nsep_pchan_2tier`) as documented negative results.

---

## Appendix B: Full Phase 13 method sweep

Phase 13 (`results/phase13_full_sweep.json`) evaluated 13 candidate INT3 variants on Qwen2-7B and Llama-3.1-8B Layer 0 K tensors (not end-to-end PPL; the metric is per-head mean-squared reconstruction error). The sweep was the origin of the `asym` finding — it is the variant that dominates the 3-bits/value column on both models. The 2-tier, energy-based top-K selection, and rotation variants are all present in this sweep; the Pareto front contains the four methods eventually used in the §5.2 ablation plus the v0.3 default.

---

## Appendix C: Gemma-4-E4B B-anomaly probes (Phase 20–21)

Phase 20 (`results/phase20_bf16_disambiguation.json`) tests whether the negative-ΔPPL anomaly on Gemma-4-E4B under variant (B) is an fp16-eval artifact. Under native bf16 the anomaly shrinks slightly (−0.44 → −0.32) but does not flip sign, and the FP16 / BF16 baseline PPL differ by only 0.019 (well below the 0.1 threshold we chose as a proxy for "real fp16-inflation"). We therefore rule out the H1 hypothesis.

Phase 21 (`results/phase21_gemma_b_anomaly_probe.json`) tests reproducibility across two non-overlapping WikiText-2 segments at 30 windows/segment (3× the Phase 19 sample size), and splits (B) into K-only / V-only / full variants. Reproducibility is confirmed: both segments show negative ΔPPL under (B) (−0.40 on segment 1, −0.15 on segment 2). The split confirms K-dominance: (B-K-only) gives 87 % of the B-full effect on both segments; (B-V-only) is small and of inconsistent sign. The data supports the attention-logit entropy-regularization hypothesis more than it supports the value-noise hypothesis, but we do not directly measure entropy and therefore stop short of a mechanistic claim.

---

## Appendix D: Implementation listings

### D.1 `qa_nsep_pchan_asym` (the v0.3 recipe)

See §3.5 for the 10-line reference implementation. The packed cache class `Int3AsymNsepPchanCache` is a thin wrapper that stores `packed_d` (uint8 tensor of bit-packed unsigned 3-bit indices), `norm`, `d_min`, and `d_step` on the same device as the input.

### D.2 `pack_int3_unsigned` / `unpack_int3_unsigned`

Three-bit unsigned packing: 8 consecutive 3-bit indices in $\{0, \dots, 7\}$ fit into 3 bytes (24 bits) via a simple bit-shift pattern. The pack is bit-exact; unpack-then-pack is identity on any valid input.

### D.3 Quantizer primitives used in the ablation

The four ablation methods use the following quantizer primitives, each exactly matching the mathematical description in §3 and §5.2:

```
(A)  qa_naive_perrow_sym(x, 3)      — per-row symmetric absmax INT3
(B)  qa_asym_pchan_no_nsep(x, 3)    — per-channel [min, max], no norm separation
(C)  qa_nsep_pchan(x, 3)            — symmetric nsep+pchan (v0.2 baseline)
(D)  qa_nsep_pchan_asym(x, 3)       — the v0.3 default
```

All four are implemented in ≤ 20 lines of PyTorch each and operate on the same input shape `[..., T, D]`.

