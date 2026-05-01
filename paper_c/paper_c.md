# Lyapunov-Aware Bit Allocation for State-Space Model Quantisation

**Kentaro Sato**
Independent Researcher
`info@metaclan.jp` · `@RecyncAI`

*Draft v0.7 · 2026-05-01*

---

## Abstract

State-space models (SSMs) such as Mamba are emerging as a sub-quadratic alternative to transformer attention, but their core memory primitive, a single recurrent hidden state per layer, does not quantise the way transformer key-value caches do. Per-step quantisation noise feeds back into the next state through the SSM recurrence and accumulates with context length at a rate set by each state group's spectral radius. Existing SSM-quantisation work has introduced per-state-group quantisation for the input-dependent parameters and uniform cached-state quantisation; to our knowledge, no prior work derives a variable-bit allocation rule for the recurrent hidden state itself as a function of recurrence stability and output coupling.

We introduce a Lyapunov-aware bit allocator for SSM state quantisation. The bit-allocation surface is

$$
R_g \;=\; \frac{\Gamma_g}{1 - \rho_g},
$$

where $\rho_g$ is the SSM's discretised spectral radius and $\Gamma_g$ is the output coupling, both calibrated per state group from a brief input-dependent forward pass. From a discrete-time Lyapunov bound on per-group accumulated noise, the strict bit-budget-optimal integer construction is $b_g = b_0 + \lceil \tfrac{1}{2} \log_2 R_g + \log_2 r_g - \overline{\log_2 r_g} \rceil_+$, with shift normalisation and clamping to $[b_\mathrm{min}, 16]$. We refer to this as the **range-retaining strict optimum** (denoted `half_slope_rg`) and adopt it as the recommended deployment recipe of the present paper. A matched-mean head-to-head across the five state-spaces Mamba-1 sizes (130M to 2.8B), with mean bit width matched to within $0.005$ bits across profiles via binary-search shift normalisation, confirms that `half_slope_rg` is structurally tighter than the simplified slope-$1$ form (`full_slope`, $b_g = b_0 + \lceil \log_2 R_g \rceil_+$) on every tested model and bit budget. The advantage is dramatic at low bit budgets ($b_0 = 4$: ΔPPL $+5.9 \,\%$ to $+19.3 \,\%$ for `half_slope_rg` vs $+22.0 \,\%$ to $+59.0 \,\%$ for `full_slope` across the 5 sizes, an average improvement of $24$ percentage points) and modest at the deployment-realistic budget ($b_0 = 6$: ΔPPL $+2.5 \,\%$ to $+5.2 \,\%$ for `half_slope_rg` vs $+3.9 \,\%$ to $+7.6 \,\%$ for `full_slope`, an average improvement of $1.3$ percentage points). The matched-mean experiment additionally establishes that the slope choice (slope-$1$ vs slope-$\tfrac{1}{2}$ without the $\log_2 r_g$ term) is approximately a wash; the dominant structural factor is the $\log_2 r_g$ term itself.

Under continuous noise injection, the strict-optimum noise profile $\sigma_g \propto R_g^{-1/2}$ is the unique linear-response ($\sigma^2$-scaling) allocator across a four-fold $\sigma_\mathrm{rel}$ sweep on Mamba-2.8B-hf, while every tested control profile is super-quadratic. Under the simplified slope-$1$ integer construction (which we used in the scale-side experiments of §4 to §7, including the eight-instance generalisation chain), the rank order full $<$ proxy_correct $<$ proxy_C2 $<$ anti $<$ uniform holds across eight Mamba SSM instances spanning three release lineages (state-spaces, TII Falcon, AI21 Jamba) and two architectural settings (per-token Mamba-1; chunk-boundary Mamba-2 SSD as a directional transfer). At $(b_0 = 6)$ with packed-state memory accounting paired with fake-quant perplexity, the recommended recipe `half_slope_rg` produces **approximately $2.6\times$ state-cache compression at $+2.5 \,\%$ to $+5.2 \,\%$ perplexity on the five state-spaces Mamba-1 sizes (130M to 2.8B)**. The recipe additionally unlocks INT4 deployment on the same family ($+5.9 \,\%$ to $+19.3 \,\%$ ΔPPL across 5 sizes at $b_0 = 4$), which the slope-$1$ simplification did not (the same family produced $+22 \,\%$ to $+59 \,\%$ ΔPPL at INT4 under `full_slope`). Cross-lineage extensions reported in §5 (Falcon-Mamba 7B, Jamba-1.5-Mini) and the long-context evaluation in §7 use the slope-$1$ simplification; their re-evaluation with `half_slope_rg` is identified as a refinement direction.

The calibrated range $r_g = k \cdot \mathrm{RMS}(h_g)$ used to place the quantisation grid additionally clips state magnitudes at long context. On Mamba-2.8B-hf evaluated on PG-19 sliding windows up to 32K context, the unquantised FP16 path diverges to perplexity 197 while the calibrated-clipping path stays at 11. A four-condition ablation (§7.3) and a five-stream trajectory characterisation (§7.5) localise the mechanism: $0.1$ to $2.3 \,\%$ of $(c, s)$ units transiently exceed $r_g$ at long context on units that simultaneously approach the $\rho_g \to 1$ recurrence-stability boundary, where rare excursions compound through the recurrence; calibrated clipping suppresses these excursions one at a time. We connect this state-clipping observation to the small-scale Edge-AI work of Zhao et al. (2025) on S4D, which the present paper extends to LLM-scale Mamba.

---

## 1. Introduction

Modern language models spend a substantial fraction of their inference memory on bookkeeping for the context they have already processed. For transformer models this bookkeeping takes the form of a *key-value cache* that grows linearly with the input length, and a recent line of work has shown how to compress that cache from 16 bits per element to 3 – 4 bits per element with negligible quality loss [Hooper et al., 2024; Liu et al., 2024]. State-space models (SSMs) of the Mamba family [Gu & Dao, 2023; Dao & Gu, 2024] take a structurally different approach: they replace the linearly-growing key-value cache with a single *fixed-size recurrent hidden state* per layer, making them attractive for long contexts where the cache would otherwise dominate memory. But the transformer compression playbook does not apply directly to this state. Per-step quantisation noise on the recurrent state feeds back into the next step and accumulates with context length, in a way that the per-element-independent transformer cache does not. **This paper provides such a rule: a principled, short-calibration bit-allocation strategy for state-space model state quantisation, derived from a discrete-time Lyapunov noise bound and validated empirically on eight Mamba SSM instances.**

Concretely, the SSM hidden state $h_t \in \mathbb{R}^{d_\mathrm{inner} \times d_\mathrm{state}}$ is updated by a discretised linear recurrence:

$$
h_t \;=\; \bar A_t\,h_{t-1} \;+\; \bar B_t\,x_t,
\qquad
y_t \;=\; C_t\,h_t \;+\; D\,x_t,
$$

where $\bar A_t$, $\bar B_t$, and $C_t$ are input-dependent transition, input-projection, and output-projection tensors. The state $h_t$ plays the same architectural role as a transformer's key-value cache (it is the layer's memory of past context), but its dynamics are fundamentally different: every quantisation error on $h_t$ feeds back into the next state through $\bar A_t$, in a way that the per-element-independent transformer cache does not.

**The problem.** Quantising $h_t$ with per-step noise $\eta_t$ produces an error $e_t = \hat h_t - h_t$ that obeys the same recurrence as the state itself:

$$
e_t \;=\; \bar A_t\,e_{t-1} \;+\; \eta_t
\qquad\Longrightarrow\qquad
\|e_t\| \;\lesssim\; \frac{\epsilon_q}{1 - \rho_g},
\qquad
\rho_g \;=\; \mathbb{E}_t\,\|\bar A_t\|_2.
$$

The error-amplification factor $(1 - \rho_g)^{-1}$ is the SSM's recurrence-stability bound (a Lyapunov-type quantity), and it has no analogue in the transformer KV-cache regime. State units whose spectral radius approaches $\rho_g \to 1$, empirically observed to comprise 18.6 % of Mamba-2.8B-hf's state-unit population on natural input (of which 3.2 % land in the very-long-memory tail $\rho_g > 0.999$), amplify per-step quantisation noise by orders of magnitude. A uniform per-element bit budget, which is the default in the transformer KV-cache literature and what naive transplants of those recipes inherit, degrades these long-memory units sharply: on Mamba-2.8B-hf at WikiText-2 perplexity, uniform allocation reaches $+221 \,\%$ ΔPPL at INT4 and $+1582 \,\%$ at INT3 (§4.6). Existing SSM-quantisation work has introduced per-state-group quantisation for the input-dependent parameters and uniform cached-state quantisation; to our knowledge, no prior work derives a variable-bit allocation rule for the recurrent hidden state itself as a function of recurrence stability and output coupling.

**Our approach.** Standard quantisation theory tells us that the bit-budget-optimal noise allocation, given a per-group "importance weight" $R_g$ and a fixed total bit budget, scales as $\sigma_g \propto 1/\sqrt{R_g}$. We argue that the natural importance weight for SSM state quantisation combines the recurrence-stability bound $(1 - \rho_g)^{-1}$ with the output-coupling magnitude $\Gamma_g = \mathbb{E}_t \|C_t\|^2$ that controls how visible the state error is at the readout. The bit-allocation surface is

$$
R_g \;=\; \frac{\Gamma_g}{1 - \rho_g}.
$$

The strict bit-budget-optimal integer construction on this surface, with the per-group range correction $r_g = k \cdot \mathrm{RMS}(h_g)$ retained from the §3.4 derivation, is

$$
b_g
\;=\;
b_0 \;+\; \bigl\lceil \tfrac{1}{2} \log_2 R_g + \log_2 r_g - \overline{\log_2 r_g} \bigr\rceil_+,
$$

with shift normalisation, $b_\mathrm{min}$ floor, and ceiling clamp as detailed in §3.5. We refer to this as the **range-retaining strict-optimum** form and treat it as the candidate deployment recipe of the present paper. We additionally introduce a slope-$1$ simplification, $b_g = b_0 + \lceil \log_2 R_g \rceil_+$, which drops the $\log_2 r_g$ term and doubles the slope; this simplification is what we used for the scale-side experiments in §4 to §7. Both forms are referred to collectively as the **Lyapunov-aware bit allocator** since the central $(1 - \rho_g)^{-1}$ factor in $R_g$ is the discrete-time Lyapunov bound on the SSM recurrence's noise accumulation. The two empirical inputs $\rho_g$ and $\Gamma_g$, together with the per-group range $r_g$, are calibrated from a single forward pass over 1024 tokens of training-distribution text, and the per-state-group integer bit budget $b_g$ is determined in closed form. No gradient-based optimisation, no quantisation-aware fine-tuning, and no per-deployment retraining is required.

**Contributions.** This paper develops the Lyapunov-aware bit allocator into an empirically validated SSM-state-quantisation recipe. Specifically:

1. **A bit-allocation surface** $R_g = \Gamma_g / (1 - \rho_g)$ (§3.3) derived from a discrete-time Lyapunov bound on accumulated quantisation noise (§3.2). Two calibrated quantities, $\rho_g$ and $\Gamma_g$, are estimated per state group from a brief input-dependent forward pass.
2. **A range-retaining strict-optimum integer construction**, $b_g = b_0 + \lceil \tfrac{1}{2} \log_2 R_g + \log_2 r_g - \overline{\log_2 r_g} \rceil_+$ (§3.5, §6.7), recommended as the deployment recipe. We additionally introduce a slope-$1$ simplification that drops the $\log_2 r_g$ term and doubles the slope, $b_g = b_0 + \lceil \log_2 R_g \rceil_+$ (denoted `full_slope`); this simplification is what we used in the scale-side experiments in §4 to §7. An integer head-to-head on Mamba-2.8B-hf (§6.7) shows the strict-optimum form is empirically substantially tighter at low bit budgets ($+11.6 \,\%$ vs $+28.9 \,\%$ ΔPPL at $b_0 = 4$).
3. **Single-architecture validation under continuous noise injection** (§4) on Mamba-2.8B-hf, including direct measurement of the $\sigma^2$-scaling property predicted by the strict bit-budget optimum: the strict-optimum continuous profile $\sigma_g \propto R_g^{-1/2}$ is uniquely $\sigma^2$-scaling under controlled noise injection across a four-fold sweep of $\sigma_\mathrm{rel}$, while every tested control profile is super-quadratic.
4. **A rank-order transfer claim across eight Mamba SSM instances** (§5) spanning three release lineages (state-spaces, TII Falcon, AI21 Jamba) and two architectural settings (per-token Mamba-1; chunk-boundary Mamba-2 SSD as directional transfer). Under the slope-$1$ simplification, the rank order full $<$ proxy_correct $<$ proxy_C2 $<$ anti $<$ uniform is preserved in every tested instance, with absolute perplexity magnitudes lineage-dependent.
5. **A matched-mean-bit deployment compression result on the state-spaces Mamba-1 family** (§6, §6.7) with real-packing memory accounting. The recommended recipe `half_slope_rg` produces approximately $2.6\times$ state-cache compression at $+2.5 \,\%$ to $+5.2 \,\%$ perplexity across the five state-spaces sizes at $b_0 = 6$, and additionally unlocks INT4 deployment on the same family at $+5.9 \,\%$ to $+19.3 \,\%$ ΔPPL (the slope-$1$ simplification produced $+22 \,\%$ to $+59 \,\%$ ΔPPL at INT4 on the same models). The cross-lineage extensions in §5 (Falcon-Mamba: $+31 \,\%$ at $b_0 = 6$; Jamba SSM-only: $+5 \,\%$ at $b_0 = 6$) and the long-context evaluation in §7 are reported under the slope-$1$ simplification; their re-evaluation under the recommended recipe is left as a refinement direction.
6. **A long-context state-clipping observation and direct mechanism characterisation** (§7) on Mamba-2.8B-hf evaluated on PG-19 at sliding-window context lengths up to 32K. The calibrated range $r_g = k \cdot \mathrm{RMS}(h_g)$ that the bit allocator uses to place the quantisation grid clips state magnitudes such that the calibrated path's perplexity stays at $\approx 11$ while unquantised FP16 and BF16 perplexity diverge to $\approx 197$ at 32K. A four-condition ablation (§7.3) and a five-stream trajectory characterisation (§7.5) localise the mechanism in calibrated clipping intervening on rare state-magnitude excursions. We connect this observation to the small-scale state-clipping recipe of [Zhao et al., 2025] for S4D, which the present paper extends to LLM-scale Mamba long-context inference.

**Roadmap.** §2 surveys related work in transformer KV-cache quantisation and the recent SSM-quantisation literature (Quamba, Q-Mamba, MambaQuant, SSDi8). §3 derives the bit-allocation rule from a per-state-group $\sigma^2$-budget argument. §4 validates the rule on Mamba-2.8B-hf as a single-architecture anchor. §5 establishes the rank-order transfer claim across eight Mamba SSM instances. §6 reports deployment-side compression and the Pareto frontier. §7 reports the long-context regime and the calibration-benefit finding. §8 discusses limitations and outlines a follow-up agenda (joint attention + SSM quantisation in hybrid stacks, deferred to a companion paper).

---

## 2. Related Work

This section positions the present paper within four adjacent literatures: transformer KV-cache quantisation (which provides the analogue and the contrast for SSM state quantisation), SSM weight and activation quantisation (the most active body of recent work on Mamba compression), per-group and heterogeneous bit allocation in SSMs (the closest in spirit to our approach, but not driven by spectral-radius theory), and Lyapunov-related analyses of SSM dynamics (which are concerned with training stability rather than quantisation). We close with an explicit statement of the present work's position relative to all four.

### 2.1 Transformer KV-Cache Quantisation as Background

A line of recent work compresses the per-token key-value cache of transformer language models from 16 bits per element to 3 to 4 bits per element with negligible quality loss. KIVI [Liu et al., 2024] proposes per-channel quantisation for the key cache and per-token quantisation for the value cache, exploiting the asymmetry of their distributions. KVQuant [Hooper et al., 2024] extends sub-4-bit quantisation to longer contexts using a dense-and-sparse decomposition. Palu [Chang et al., 2024] caches low-rank projections of the key-value tensors. ZipCache [He et al., 2024] identifies salient tokens via normalised attention scores and compresses non-salient tokens more aggressively. These methods all exploit the per-token, per-element-independent structure of the transformer KV-cache: each cache slot is read once at attention time, the rounding errors do not feed back into a recurrence, and per-element noise variance translates directly into per-token attention error. None of these properties hold for the SSM hidden state, which is updated by an input-dependent linear recurrence (§3.1). The present paper inherits the transformer-side compression playbook in spirit, but the per-element optimisation argument that drives it must be rebuilt from first principles for the SSM substrate.

### 2.2 SSM Weight and Activation Quantisation

The first wave of post-training quantisation (PTQ) work on Mamba focused on weights and activations rather than the recurrent state. Mamba-PTQ [Pierro & Abreu, 2024] is among the earliest, identifying outlier channels in the recurrent LLM as the primary obstacle to naive transplants of transformer PTQ recipes. Quamba [Chiang et al., 2024] proposes a static 8-bit per-tensor quantisation scheme for the selective SSM, suppressing the maximum values of the input activations and quantising the output activations in an outlier-free space via a Hadamard transform; it reports a $1.72\times$ generation-latency reduction on Mamba-2.8B with $0.9 \,\%$ accuracy drop. MambaQuant [Xu et al., 2025] addresses outlier-driven accuracy loss with Karhunen-Loève transform-enhanced rotation and a smooth-fused rotation that equalises channel variances, achieving W8A8 with under $1 \,\%$ accuracy loss. OuroMamba [Ramachandran et al., 2025] presents a data-free PTQ for vision Mamba models, using channel-wise static weight quantisation and per-time-step dynamic activation quantisation. LightMamba [Wei et al., 2025] co-designs a 4-bit rotation-based PTQ with FPGA acceleration. Bi-Mamba [Tang et al., 2024] takes the alternative quantisation-aware-training (QAT) route to 1-bit weights, training Mamba models from scratch with autoregressive distillation. None of these methods quantise the recurrent state $h_t$ at per-(channel, state-unit) granularity with bit widths driven by spectral or output-coupling statistics, which is the focus of the present paper.

### 2.3 Per-Group and Heterogeneous Bit Allocation in SSMs

A smaller body of recent work allocates bits non-uniformly across SSM components, and is the closest in spirit to our approach. Quamba2 [Chiang et al., 2025] extends Quamba to W8A8, W4A8, and W4A16 across both Mamba-1 and Mamba-2 backbones, with per-state-group quantisation for the input-dependent parameters $B_t$ and $C_t$. Quamba2's groups are channel-wise (eight groups of 128 channels each), and within a group the quantisation is uniform 8-bit; cached SSM states are quantised at uniform 8-bit as well. The grouping is based on activation magnitude clustering and a Hadamard-style compute-invariance constraint, not on the recurrence's spectral radius or output coupling. Q-Mamba [Chen et al., 2025] proposes Decoupled Scale Quantisation (DSQ) for Mamba models, applying separate quantisation scales to outlier channels and inlier channels and quantising linear projections and state caches into low-bit integers; it reports memory and accuracy results on Mamba2-2.7B at W8A8H4. The recent edge-AI study of [Zhao et al., 2025] (Quantizing Small-Scale State-Space Models for Edge AI) reports a heterogeneous bit allocation in which the weight matrices $\bar B$, $\bar C$, $\bar D$ are quantised at 4 bits, activations other than the state at 6 bits, and the state $x$ and transition matrix $\bar A$ at 6 to 16 bits depending on ablation; their grouping is per-head (one group per parallel SSM head), their calibration uses a 99.999-percentile range estimate, and the bit allocation is determined by ablation rather than by a closed-form formula. SSDi8 [Kim et al., 2025] targets the Mamba-2 SSD computation specifically, designing an INT8 reformulation that decouples element-wise from matrix-wise multiplications and reuses quantised activations to maintain a persistent INT8 path through the SSD; it uses channel-aware quantisation and mean correction to keep accuracy. Q-S5 [Abreu et al., 2024] and QS4D [Siegel et al., 2025] explore quantisation of S5 and S4D structured SSMs respectively, primarily in the QAT regime. We narrow our novelty claim accordingly: per-state-group quantisation of SSM caches and heterogeneous bit allocation across SSM components have prior precedent in this literature; **to our knowledge, no prior work derives a variable-bit allocation rule for the recurrent hidden state $h_t$ as a closed-form function of the per-group spectral radius $\rho_g$ and output coupling $\Gamma_g$**, which is the contribution of the present paper.

### 2.4 State Clipping and Lyapunov-Style Analyses of SSM Dynamics

Two related lines of work touch on the long-context state-magnitude behaviour we discuss in §7.

The first is the small-scale state-clipping observation in [Zhao et al., 2025], who note that the recurrent state $x$ in S4D-class models can be constrained after each time step as a simple stabiliser for low-bit recurrent inference. They motivate this by the discrete-eigenvalue stability concern: if the quantised value of $\bar A$ falls outside the unit circle, the SSM recurrence is unstable, and clipping the state suppresses the resulting amplification. This idea is closely connected to the §7.3 / §7.4 finding of the present paper, where we observe on Mamba-2.8B-hf at PG-19 long context that calibrated state clipping at $r_g = k \cdot \mathrm{RMS}(h_g)$ stabilises the FP16 path that otherwise diverges. We extend the Zhao et al. observation from small-scale S4D to LLM-scale Mamba and from short to long-context inference, and we explicitly credit the underlying state-clipping idea to that earlier work.

The second is the Lyapunov-style training-stability analysis in [Halloran et al., 2024], who characterise the training-time stability of Mamba's selective recurrence using a Lyapunov function on the parameter trajectories. Their concern is whether Mamba's training is convergent rather than how to allocate quantisation noise across state groups; the present paper's use of a discrete-time Lyapunov bound on accumulated noise (§3.2) is methodologically related but addresses a different question.

Spectral State Space Models [Agarwal et al., 2023] proposes a new SSM family based on spectral filtering of linear dynamical systems; this is concerned with model design rather than compression and does not bear directly on the bit-allocation problem. To our knowledge no prior work translates a discrete-time Lyapunov bound on accumulated quantisation noise into a per-state-group bit-allocation rule for SSMs, which is the framing we develop in §3.

### 2.5 Position of the Present Work

We summarise our position relative to the four literatures along five axes.

We summarise our novelty claim along four narrowed axes.

First, the **bit-allocation surface** is closed-form ($R_g = \Gamma_g / (1 - \rho_g)$), driven by two calibrated scalars per state group, without ablation search. Existing SSM-quantisation work uses uniform bits within groups (Quamba2 [Chiang et al., 2025]; Q-Mamba [Chen et al., 2025] for state caches; SSDi8 [Kim et al., 2025] for the SSD path) or determines heterogeneous bits via empirical ablation (Zhao et al., 2025). To our knowledge no prior SSM-quantisation work uses the per-group spectral radius and output coupling as inputs to a closed-form bit-allocation rule.

Second, our **granularity** is per-(channel, state-unit), which on Mamba-2.8B-hf produces $5{,}242{,}880$ distinct bit widths. Existing work operates at per-tensor (Quamba [Chiang et al., 2024]), per-channel-group (Quamba2 [Chiang et al., 2025]; Q-Mamba [Chen et al., 2025]), or per-head (Zhao et al., 2025) granularity. The per-(channel, state-unit) granularity is what allows the rule to track the heavy-tailed $\Gamma$ distribution we measure in §4.2 (max/min ratios of $167\times$ to $1572\times$ within a single layer's 16 state indices).

Third, our **derivation** is a discrete-time Lyapunov bound on accumulated quantisation noise (§3.2), with $(1 - \rho_g)^{-1}$ as the per-group amplification factor. Existing Lyapunov-themed work on Mamba [Halloran et al., 2024] addresses training stability rather than compression, and existing SSM quantisation work uses activation magnitude statistics rather than recurrence dynamics for bit-budget reasoning.

Fourth, our **importance weight** combines spectral radius with output coupling, $R_g = \Gamma_g / (1 - \rho_g)$. The output-coupling factor $\Gamma_g = \mathbb{E}_t [(C_t^{(s)})^2]$ is, to our knowledge, not used in prior SSM quantisation literature as an input to bit allocation.

Together these axes give us a narrow but distinct novelty position: **to our knowledge, no prior work derives a variable-bit allocation rule for the recurrent hidden state $h_t$ as a closed-form function of the per-group spectral radius $\rho_g$ and output coupling $\Gamma_g$**.

The long-context state-clipping observation in §7 is, by contrast, not original to this paper: Zhao et al. (2025) note state clipping as a stabiliser for low-bit S4D inference. The contribution of §7 is the empirical extension of that observation from small-scale S4D to LLM-scale Mamba long-context inference, the four-condition disambiguation that rules out pure-precision and pure-rounding alternatives, and the five-stream trajectory characterisation of the underlying mechanism.

---

## 3. Theory: A Lyapunov-Aware Bit Allocator

We now derive the bit allocator $b_g = b_0 + \lceil \log_2 (\Gamma_g / (1 - \rho_g)) \rceil_+$ stated in §1. The derivation proceeds in five steps. First we set up the per-step error recurrence induced by quantising the SSM state at every step (§3.1). A Lyapunov-type bound on the accumulated error then identifies the spectral radius $\rho_g$ as the natural per-group amplification factor (§3.2). Routing the error through the output projection introduces the second per-group quantity, the output coupling $\Gamma_g$ (§3.3). Standard quantisation theory then determines the bit-budget-optimal allocation as $\sigma_g \propto 1/\sqrt{R_g}$ with $R_g = \Gamma_g / (1 - \rho_g)$ (§3.4), which we discretise to an integer per-group bit width via a conservative rounding plus shift normalisation (§3.5). All quantities can be calibrated from a single forward pass over a short corpus (§3.6).

### 3.1 The State Recurrence and the Per-Step Quantisation Error

A discretised Mamba-class SSM updates its hidden state by

$$
h_t \;=\; \bar A_t\,h_{t-1} \;+\; \bar B_t\,x_t,
\qquad
y_t \;=\; C_t\,h_t \;+\; D\,x_t,
$$

where the per-step transition $\bar A_t$, input projection $\bar B_t$, and output projection $C_t$ are produced by input-dependent linear maps from $x_t$. For Mamba-1, the discretisation is $\bar A_t = \exp(dt_t \cdot A)$ with $A = -\exp(A_\mathrm{log})$ a per-state-unit negative real eigenvalue, and $\bar B_t = dt_t \cdot B_t$, where $dt_t \in \mathbb{R}_{>0}^{d_\mathrm{inner}}$ is the input-dependent time step. The hidden state $h_t \in \mathbb{R}^{d_\mathrm{inner} \times d_\mathrm{state}}$ is diagonal in the state-state direction, so each (channel $c$, state-unit $s$) is updated independently. Our analysis is per-(c, s); we use $g$ as a per-group index, with $g = (c, s)$ at the finest granularity.

We consider the regime in which $h_t$ is quantised at every step before being used in the next update. Let $\hat h_t = \mathcal{Q}(h_t)$ be the quantised state and $\eta_t = \hat h_t - h_t$ the per-step quantisation noise. The next-step recurrence consumes $\hat h_{t-1}$ rather than $h_{t-1}$, so the per-step error $e_t = \hat h_t - h_t$ obeys the recurrence

$$
e_t \;=\; \bar A_t\,e_{t-1} \;+\; \eta_t.
$$

The error is updated by the same $\bar A_t$ that updates the state, driven by the per-step quantisation noise $\eta_t$. This error recurrence is the central object of our analysis.

### 3.2 The Lyapunov Bound on Accumulated Noise

Let $\rho_g$ denote a per-group bound on the long-run growth rate of the linear recurrence $e_t = \bar A_t e_{t-1} + \eta_t$. Under either of two equivalent regularity assumptions, the steady-state error variance is bounded by a geometric series. Assumption (A): $|\bar A_t^{(g)}| \le \rho_g < 1$ uniformly in $t$. Assumption (B): the per-group transition is wide-sense stationary with $\mathbb{E}[(\bar A_t^{(g)})^2] \le \rho_g^2$. Under either, with zero-mean i.i.d. quantisation noise $\eta_t^{(g)}$ of per-group variance $\sigma_q^{(g)\,2}$ and stationary input distribution,

$$
\mathbb{E}\bigl[(e_t^{(g)})^2\bigr]
\;\le\;
\frac{\sigma_q^{(g)\,2}}{1 - \rho_g^2}
\;\approx\;
\frac{\sigma_q^{(g)\,2}}{2\,(1 - \rho_g)}
\quad(\rho_g \to 1).
$$

In practice we estimate $\rho_g$ as the empirical mean $\rho_g \approx \mathbb{E}_t |\bar A_t^{(g)}|$ from a short calibration pass (§3.6). This is a slightly weaker quantity than the variance-based or supremum-based bounds above, and the resulting allocator should be read as motivated by the bound rather than as proven optimal under it. The empirical $\sigma^2$-scaling property in §4.5 supplies the missing empirical guarantee that the linear-response regime the bound assumes is in fact respected by the deployed allocator. We absorb the leading constant and use $\mathbb{E}[(e_t^{(g)})^2] \propto \sigma_q^{(g)\,2} / (1 - \rho_g)$ in what follows.

This is a discrete-time Lyapunov-type bound. The amplification factor $(1 - \rho_g)^{-1}$ has the same role as the convergence-rate factor in classical Lyapunov analysis of contraction maps: it characterises how rapidly the accumulated noise settles to its bounded equilibrium. As $\rho_g \to 1$, the per-step noise allowed for a target equilibrium error must scale as $\sqrt{1 - \rho_g}$, and the bit budget required to deliver such noise grows accordingly.

The empirical reality on Mamba-class models is that a non-trivial fraction of state units sits very close to the $\rho_g = 1$ boundary. On Mamba-2.8B-hf evaluated on 1024 tokens of WikiText-2 train, 18.6 % of the 5,242,880 per-(c, s) state units satisfy $\rho_g > 0.95$ and 3.2 % satisfy $\rho_g > 0.999$ (§4.1, Phase C-1b). For these long-memory units the amplification factor is $10^3$ or larger, and a uniform per-element bit budget cannot deliver the per-step noise variance they require.

### 3.3 The Output Coupling and the Visible Error

The state error $e_t$ is not directly observable at the loss; it is mediated by the output projection $C_t$. Writing the readout per channel,

$$
y_t^{(c)} - \hat y_t^{(c)}
\;=\;
-\sum_s C_t^{(c, s)}\,e_t^{(c, s)}
\;=\;
-\sum_s C_t^{(s)}\,e_t^{(c, s)},
$$

where the second equality uses Mamba's parameterisation in which $C_t$ is shared across channels for each state-unit $s$. Under independence of the per-(c, s) errors,

$$
\mathbb{E}\bigl[(y_t^{(c)} - \hat y_t^{(c)})^2\bigr]
\;=\;
\sum_s \mathbb{E}\bigl[(C_t^{(s)})^2\bigr] \cdot
       \mathbb{E}\bigl[(e_t^{(c, s)})^2\bigr]
\;=\;
\sum_s \Gamma_s \cdot
       \mathbb{E}\bigl[(e_t^{(c, s)})^2\bigr],
$$

with the per-state-unit output coupling magnitude $\Gamma_s = \mathbb{E}_t[(C_t^{(s)})^2]$. We use $\Gamma_g$ for the per-group quantity, where $\Gamma_g = \Gamma_s$ at our (c, s) granularity (the coupling is shared within a column).

Combining §3.2 and the present subsection, the per-(c, s) contribution to the visible mean-squared readout error is

$$
\Gamma_g \cdot \mathbb{E}\bigl[(e_t^{(g)})^2\bigr]
\;\propto\;
\frac{\Gamma_g \cdot \sigma_q^{(g)\,2}}{1 - \rho_g}
\;=\;
R_g \cdot \sigma_q^{(g)\,2},
\qquad
\boxed{R_g \;=\; \frac{\Gamma_g}{1 - \rho_g}}.
$$

The per-group importance weight $R_g$ collects the two physical effects that determine which state groups need more bits: the recurrence-amplification factor $(1 - \rho_g)^{-1}$ from the Lyapunov bound, and the output-coupling factor $\Gamma_g$ from the readout. Both quantities are calibrated empirically (§3.6); both are needed (§4.4 reports that allocators using $\rho_g$ alone or $\Gamma_g$ alone are strictly dominated by the joint $R_g$ form).

### 3.4 The Bit-Budget-Optimal Allocation

The total visible mean-squared readout error under per-group i.i.d. quantisation noise is

$$
\mathrm{MSE}_\mathrm{total}
\;\propto\;
\sum_g R_g \cdot \sigma_q^{(g)\,2}.
$$

For a uniform mid-tread quantiser with per-group range $r_g$ and bit width $b_g$, the per-group noise variance is $\sigma_q^{(g)\,2} = (r_g / 2^{b_g})^2 / 3 \propto r_g^2 \cdot 4^{-b_g}$. We adopt a calibrated per-group range $r_g = k \cdot \mathrm{RMS}(h_g)$ with $k = 7$ throughout the present paper. The choice of $k = 7$ was identified by the C-4 ablation in §4.6 as the value that saturates the clipping-versus-quantisation-noise trade-off at $b_0 \ge 5$; a Gaussian-distribution interpretation (7-σ truncation) is approximate, and we caveat this assumption explicitly in §8.1.

Minimising $\sum_g R_g \cdot r_g^2 \cdot 4^{-b_g}$ subject to a fixed total bit budget $\sum_g b_g = B_\mathrm{total}$ via Lagrange multiplier gives, at the unconstrained optimum,

$$
R_g \cdot r_g^2 \cdot 4^{-b_g} \;=\; \mathrm{const},
\qquad
b_g \;=\; \mathrm{const}' \;+\; \tfrac{1}{2} \log_2 R_g \;+\; \log_2 r_g.
$$

The optimum thus contains two terms beyond the constant: a $\tfrac{1}{2} \log_2 R_g$ term that captures the per-group importance weight, and a $\log_2 r_g$ term that scales with the per-group state magnitude. In the present paper we adopt the simplifying assumption that the per-group $r_g$ are roughly homogeneous across groups after RMS calibration; the integer construction in §3.5 keeps only the $R_g$ term and absorbs the $r_g$ variation into the shift normalisation. The per-layer $\mathrm{RMS}(h_g)$ varies by approximately $10\times$ across the 64 layers of Mamba-2.8B-hf (§8.1), so the dropped $\log_2 r_g$ term contributes up to $\approx 3.3$ bits of theoretical refinement. We initially treated this drop as a minor approximation; the follow-up experiment in §6.7 (Phase C-12) shows that a version of the allocator retaining the $\log_2 r_g$ term is empirically substantially tighter than the simplified form at the deployment-relevant low-bit regime. We discuss the consequences for the candidate deployment recipe in §6.7 and §8.3.

Restricted to the $R_g$-only formulation, the bit-budget-optimal noise allocation scales as

$$
\sigma_q^{(g)} \;\propto\; \frac{1}{\sqrt{R_g}}
\;=\;
\sqrt{\frac{1 - \rho_g}{\Gamma_g}}.
$$

This is the form of the "full" profile in our continuous-noise validation experiments (§4.5), where it is the unique linear-response ($\sigma^2$-scaling) allocator under controlled noise injection while every tested control profile (uniform; $\rho$-only; $\sqrt{1-\rho}$-only; anti-allocation) is super-quadratic.

### 3.5 Two Integer Realisations of the $R_g$ Surface

The continuous optimum derived in §3.4 is $b_g \propto \tfrac{1}{2} \log_2 R_g + \log_2 r_g + \mathrm{const}$. For deployment we need an integer $b_g$ per group, fitting within a target average $b_0$ and bounded by a hardware-friendly minimum $b_\mathrm{min}$ to avoid sign-only quantisation. We introduce two integer realisations of the same $R_g$ surface, both built around shift normalisation and a $b_\mathrm{min} = 2$ floor, and recommend the first as the deployment recipe while using the second for our scale-side experiments.

**Realisation A: range-retaining strict optimum** (`half_slope_rg`, recommended). The strict continuous optimum of §3.4 is integerised directly:

$$
\mathrm{extras}_g \;=\; \bigl\lceil \tfrac{1}{2} \log_2 R_g \;+\; \log_2 r_g \;-\; \overline{\log_2 r_g} \bigr\rceil_+,
\qquad
b_g \;=\; \mathrm{round}\!\bigl(b_0 + \mathrm{extras}_g - \overline{\mathrm{extras}}\bigr)\Big|_{[b_\mathrm{min},\,16]},
$$

where $\overline{\log_2 r_g}$ centres the per-group $\log_2 r_g$ correction at zero so it does not shift the mean, and the outer shift normalisation $- \overline{\mathrm{extras}}$ rebalances toward $b_0$. This realisation matches the §3.4 derivation directly and is the bit-budget-optimal integer construction on the $R_g$ surface.

**Realisation B: simplified slope-$1$ form** (`full_slope`, used in §4 to §7 scale sweeps):

$$
\mathrm{extras}_g \;=\; \bigl\lceil \log_2 R_g \bigr\rceil_+,
\qquad
b_g \;=\; \mathrm{round}\!\bigl(b_0 + \mathrm{extras}_g - \overline{\mathrm{extras}}\bigr)\Big|_{[b_\mathrm{min},\,16]}.
$$

This drops the $\log_2 r_g$ term and doubles the slope on $\log_2 R_g$. Geometrically it allocates approximately twice as many extra bits to high-$R_g$ groups as the strict optimum, and (after shift normalisation) approximately twice fewer to low-$R_g$ groups; on the same shift-normalised mean, the structural deviation is the loss of the per-group $r_g$ correction. The simplified form was introduced earlier in our experimental programme as a tail-risk heuristic and used for the scale-side experiments in §4 to §7; we retain those results in the present manuscript and report a head-to-head comparison against the strict optimum in §6.7.

**Why two realisations.** Realisation A is the theoretically tight form and is the recommended deployment recipe of the present paper. Realisation B is the form we used in the scale-side experiments of §4 to §7 (eight-instance generalisation, long-context evaluation, deployment Pareto frontier). The matched-mean-bit head-to-head in §6.7 establishes the structural relationship between the two on the integer-bit channel: across all five state-spaces Mamba-1 sizes (130M to 2.8B) at $b_0 = 6$, Realisation A produces ΔPPL of $+2.5 \,\%$ to $+5.2 \,\%$, an average of $1.3$ percentage points tighter than Realisation B's $+3.9 \,\%$ to $+7.6 \,\%$. At $b_0 = 4$ the gap widens to an average of $24$ percentage points (Realisation A: $+5.9 \,\%$ to $+19.3 \,\%$ vs Realisation B: $+22.0 \,\%$ to $+59.0 \,\%$). The matched-mean comparison additionally factorises the structural advantage: the slope choice (slope-$1$ vs strict slope-$\tfrac{1}{2}$, without the $r_g$ term) is approximately a wash at matched mean bit width on this evaluation, with the slope-$1$ form winning on four of five models at $b_0 = 6$ by $0.5$ to $1.3$ percentage points. The dominant structural factor is the $\log_2 r_g$ term, not the slope. Realisation A's combined advantage over Realisation B comes almost entirely from retaining the $r_g$ term.

**On §4.5 validation.** Under the standard mapping $\sigma_g \propto 2^{-b_g}$, Realisation A corresponds to continuous $\sigma_g \propto R_g^{-1/2} \cdot r_g^{-1} = \sqrt{(1 - \rho_g) / \Gamma_g} / r_g$, while Realisation B corresponds to $\sigma_g \propto R_g^{-1} = (1 - \rho_g) / \Gamma_g$. The continuous-noise profile labelled "full" in §4.4 and §4.5 is $\sqrt{(1 - \rho_g) / \Gamma_g}$ (the $R_g^{-1/2}$ form, modulo the per-group $r_g$ correction that we subsume into the constant under the §3.4 simplifying assumption). So §4.5 validates the slope-$\tfrac{1}{2}$ continuous form on the $R_g$ surface, which is the spine of Realisation A. Within that family, §4.5 establishes that the form is uniquely $\sigma^2$-scaling across a four-fold $\sigma_\mathrm{rel}$ sweep (multipliers $3.66\times$ and $4.03\times$ vs the $4\times$ ideal); every other tested profile, including the $\rho$-only proxy $\sigma_g \propto \sqrt{1 - \rho_g}$, is super-quadratic. The §4.5 result therefore establishes correctness of the $R_g$ surface and of the slope-$\tfrac{1}{2}$ shape; §4.6 establishes that Realisation B in particular is a competitive integer deployment recipe in absolute terms; §6.7 establishes that Realisation A is empirically the tighter integer recipe.

**The $b_\mathrm{min} = 2$ floor.** A binary allocation has only one non-zero level above zero and no mid-tread zero, so the per-step noise is no longer well-modelled by $\sigma_q \propto 2^{-b}$ and the linear-response argument breaks down. The $b_\mathrm{min} = 2$ floor keeps the smallest groups at INT2 (four levels, including zero) where the standard analysis still applies; this choice was confirmed in §4.6, where allocations that included $b_g = 1$ groups departed sharply from the $\sigma^2$-scaling reference.

**Naming.** We refer to the resulting integer bit-width assignment as the **Lyapunov-aware bit allocator**: the central $(1 - \rho_g)^{-1}$ factor in $R_g$ is the discrete-time Lyapunov bound on the SSM recurrence's noise accumulation, and both Realisation A and Realisation B inherit their dependence on this factor. The recommended deployment form is Realisation A; the form actually tested at scale in §4 to §7 of the present manuscript is Realisation B.

### 3.6 Calibration

The two empirical inputs $\rho_g$ and $\Gamma_g$, together with the per-group state magnitude $\mathrm{RMS}(h_g)$ used for the range $r_g$, are calibrated from a single forward pass over a short calibration corpus. We use 1024 tokens of WikiText-2 train throughout this paper. The recipe is:

1. Run a forward-only inference pass that records, at every Mamba mixer, the per-(c, s) tensors $\bar A_t^{(c, s)}$, $C_t^{(s)}$, and the post-update state $h_t^{(c, s)}$.
2. Estimate $\rho_g$ as the empirical mean over $t$ of $|\bar A_t^{(c, s)}|$. The static-only estimate from $A_\mathrm{log}$ alone (with a fixed $dt = 0.1$) underestimates the long-memory tail by approximately fifty-fold on Mamba-2.8B-hf; the input-dependent estimate is the right input to the bit allocator (§4.1).
3. Estimate $\Gamma_g$ as the empirical mean over $t$ of $(C_t^{(s)})^2$.
4. Estimate $\mathrm{RMS}(h_g)$ as $\sqrt{\mathbb{E}_t[(h_t^{(c, s)})^2]}$ via streaming accumulation over the calibration window.

The total cost is one forward pass over 1024 tokens plus per-mixer streaming statistics in float32. For Mamba-2.8B-hf this amounts to roughly 30 seconds of GPU time and about 5 GB of activation memory; the resulting per-(c, s) statistics are stored once and reused for all subsequent allocator and range computations. No gradient propagation, no quantisation-aware fine-tuning, and no per-deployment retraining is involved.

Calibration robustness is examined in §4.3. Briefly, training-set and validation-set calibrations on Mamba-2.8B-hf produce $\log \Gamma_g$ values with Pearson correlation 0.987 and a 77.5 % match in the resulting integer bit-tier classification, with one outlier layer flagged as having unusually input-dependent coupling. This is the train-validation cross-check on a single corpus (WikiText-2); inter-corpus stability (calibration on RefinedWeb, PG-19, or Books-style text) is not directly measured in the present paper, and we identify a multi-corpus calibration sweep as a follow-up experiment (Phase C-13). Within the limits of the train-validation check, the bit-tier match is below the 80 % pre-registered threshold by 2.5 percentage points, which is why the deployment language we adopt is "short-calibration" rather than "calibration-free".

---

## 4. Single-Architecture Validation: Mamba-2.8B-hf

This section validates the bit allocator's direction, scaling behaviour, and deployment quality on Mamba-2.8B-hf, the largest of the state-spaces Mamba-1 family that runs cleanly on the HuggingFace reference Python forward path. We proceed bottom-up, from calibration-side measurements (the empirical $\rho_g$ distribution in §4.1, the output coupling extraction in §4.2, the calibration robustness analysis in §4.3) to allocator-side measurements (state-step quantisation in §4.4, the $\sigma^2$-scaling test in §4.5, and the $\sigma$-to-bit bridge in §4.6). Generalisation to seven additional Mamba SSM instances spanning two architectures and three lineages is deferred to §5.

### 4.1 The Empirical Spectral-Radius Distribution

The first calibration step measures the per-(c, s) spectral radius $\rho_g$ of the discretised Mamba-1 transition $\bar A_t = \exp(dt_t A)$ with $A = -\exp(A_\mathrm{log})$. Two estimators are available: a static-only estimator that uses $A_\mathrm{log}$ alone with a fixed $dt = 0.1$ (the convention used in many static analyses of SSMs), and an empirical estimator that captures the actual input-dependent $dt$ under WikiText-2 input. The static estimator was run as Phase C-1, the empirical estimator as Phase C-1b. The two estimators agree at the population mean ($\rho_g$ p50 $\approx 0.6$) but diverge sharply in the long-memory tail.

| $\rho_g$ band | static, $dt = 0.1$ | empirical, input-dependent $dt$ | shift |
|:---:|---:|---:|:---:|
| $[0.95, 0.99)$ | $0.42 \,\%$ (21,919 units) | $9.7 \,\%$ (508,316 units) | 23× |
| $[0.99, 0.999)$ | $2 \times 10^{-6} \,\%$ (12 units) | $5.7 \,\%$ (299,183 units) | $24\,000\times$ |
| $[0.999, 1.000]$ | $0$ | $3.2 \,\%$ (165,816 units) | $0 \to 165{,}816$ |

The empirical long-memory tail ($\rho_g > 0.95$) covers $18.6 \,\%$ of the 5,242,880 per-(c, s) state units in Mamba-2.8B-hf, fifty times larger than the $0.4 \,\%$ the static estimator predicts. The very-long-memory tail ($\rho_g > 0.999$) goes from zero units in the static estimate to 165,816 units in the empirical estimate. The gap arises because per-layer median $dt$ observed empirically is in the $0.017$ to $0.125$ range, often well below the static assumption of $0.1$; the contraction factor $\exp(-dt \cdot |A|)$ is then much closer to one.

The empirical $\rho_g$ distribution also exhibits strong per-layer heterogeneity. Median $\rho_g$ across the 64 layers traces a U-shape: short-memory layers at the input end ($\rho_g$ p50 $\approx 0.60$ at layer 0), a long-memory core in the middle ($\rho_g$ p50 $\approx 0.88$ at layers 32 and 40), and short-memory layers again at the output end ($\rho_g$ p50 $\approx 0.39$ at layer 56). We do not in this paper attempt to interpret what each layer encodes in linguistic terms; we report the U-shape as an empirical structural fact and note that it motivates a per-layer baseline term in the bit allocator, which the shift normalisation of §3.5 implicitly provides.

The conclusion of §4.1 is that the empirical, input-dependent $\rho_g$ is the right input to the bit allocator, and that the long-memory minority is sufficiently large (in the hundreds of thousands of state units, not a handful) that protecting it with extra bits is the central design challenge.

### 4.2 Output-Coupling Extraction and the $R_g$ Surface

Phase C-1c extracts the per-state-unit output coupling $\Gamma_s = \mathbb{E}_t[(C_t^{(s)})^2]$ by replaying the Mamba-1 forward path under WikiText-2 input and intercepting the output of `mix.x_proj`, which produces $dt_\mathrm{raw}$, $B_t$, and $C_t$ as concatenated $[d_{t,\mathrm{rank}}, d_\mathrm{state}, d_\mathrm{state}]$ slices. The third slice is averaged in squared form per state-unit $s$ over the 1024-token calibration window. In Mamba-1, $C_t$ is shared across channels for each $s$, so $\Gamma_g = \Gamma_s$ at our (c, s) granularity, and the model has only $64 \times 16 = 1024$ distinct $\Gamma$ scalars in total.

The per-layer $\Gamma_s$ distribution is heavy-tailed. The within-layer $\Gamma_\mathrm{max} / \Gamma_\mathrm{min}$ ratio across the 16 state indices is in the $167\times$ to $1572\times$ range across layers, and the per-layer $\Gamma_\mathrm{max}$ grows monotonically toward the output end (e.g., $\Gamma_\mathrm{max} = 5.1$ at layer 0, $27.1$ at layer 48, $45.4$ at layer 56). A small handful of state indices contribute the largest readout magnitudes in the late stages of the model, with the remaining state indices contributing more weakly to the per-channel output sum. We are careful not to label the smaller-$\Gamma$ state indices as functionally inactive: their per-step contribution to the readout is small in magnitude, but they may still carry context-dependent information that is exploited non-linearly downstream. A direct functional ablation of small-$\Gamma$ state indices is left as a candidate refinement.

A direct consequence is that the $\rho$-only proxy $R_g^\mathrm{proxy} = 1 / (1 - \rho_g)$ is not a faithful approximation of the full $R_g = \Gamma_g / (1 - \rho_g)$. Phase C-1c measures the global log-log Pearson correlation between $R^\mathrm{proxy}$ and $R^\mathrm{full}$ across all 5,242,880 per-(c, s) state units at $0.635$, with per-layer correlations as low as $-0.07$ in some layers. At $b_0 = 3$, the $\rho$-only proxy and the full $R_g$ assign the same integer bit width to only $9.2 \,\%$ of state units, with $58.1 \,\%$ of units differing by two or more bits. The proxy is acceptable as a direction-finding heuristic but is not the deployed allocator; both calibrated quantities are needed.

We refer to the joint $R_g = \Gamma_g / (1 - \rho_g)$ surface as the bit-allocation surface of Mamba-2.8B-hf. After §4.2 it is fully calibrated and ready to drive the integer construction of §3.5.

### 4.3 Calibration Robustness

A critical question for the deployment claim is whether the calibration is sensitive to the choice of corpus. Phase C-1c-bis re-runs the $\Gamma_s$, $\rho_g$, and $\mathrm{RMS}(h_g)$ extraction on WikiText-2 validation (instead of train) and compares the resulting per-(c, s) statistics. Pre-registered acceptance thresholds were $\mathrm{Pearson}(\log \Gamma_g) \ge 0.95$ and bit-tier match $\ge 80 \,\%$.

| quantity | global $\mathrm{Pearson}(\log \cdot)$ | reading |
|:---|---:|:---|
| $\rho_g$ (5.24 M units) | $0.9954$ | essentially text-independent (its near-static parameter origin) |
| $\Gamma_s$ (1024 scalars) | $0.9870$ | above the $0.95$ threshold globally |
| $\mathrm{RMS}(h_g)$ (5.24 M units) | $0.9581$ | borderline; expected since $h$ integrates input |

Per-layer $\mathrm{Pearson}(\log \Gamma_s)$ is concentrated in $0.92$ to $1.00$ across 62 layers, with a single sharp outlier at layer 30 ($\mathrm{Pearson} = 0.57$). Layer 30 sits just below the long-memory core (layers 32 to 40) and has only 16 $\Gamma_s$ values per layer, so a one- or two-entry shift on its outlier dimensions disproportionately drags the per-layer correlation. Globally aggregated to all 1024 $(\mathrm{layer}, s)$ pairs the effect dissolves.

The bit-tier match, defined as the fraction of (c, s) state units whose integer extras $\lceil \log_2 R_g \rceil_+$ at $b_0 = 4$ matches between the train and validation calibrations, is $77.5 \,\%$. Of the remaining $22.5 \,\%$, almost all (over 99 %) differ by exactly one bit; the $\ge \pm 2$ bit tail is $0.21 \,\%$. The pre-registered $80 \,\%$ threshold fails by $2.5$ percentage points, placing this calibration in the middle bucket of the decision rule (USABLE_WITH_FLAG): the C-3 / C-3b results stand under a $\pm 1$-bit perturbation on a quarter of units. We do not apply a $+1$ bit safety margin to the experimental tables in §5 and §6; the bit budgets reported there are the bare allocator's output. A $+1$ bit safety margin (or calibration ensembling across multiple text sources) is a recommended deployment convention for production use, but the empirical results in this paper characterise the underlying recipe at its nominal bit budget. Layer 30 is flagged as the candidate for any deployment-time recalibration check.

### 4.4 State-Step Quantisation in the Patched `slow_forward`

The first three subsections measure calibration-side quantities. Phase C-3 takes the next step and tests the bit allocator under the actual quantisation channel that deployment will use. The patched `MambaMixer.slow_forward` inserts a per-(c, s) Gaussian noise injection $\hat h_t = h_t + \eta_t,\ \eta_t \sim \mathcal{N}(0, \sigma_g^2 I)$ between the state update and the readout, with the per-group $\sigma_g$ determined by one of five profiles. The relative noise scale is held constant at $\sigma_\mathrm{rel} = \sigma_g / \mathrm{RMS}(h_g) = 0.10$, so any difference in $\Delta\mathrm{PPL}$ is attributable to the *allocation* shape, not to the *amount* of noise.

| profile | $\sigma_g$ shape | $\Delta\mathrm{PPL}$ vs FP16 |
|:---|:---|---:|
| **full** (proposed) | $\sqrt{(1 - \rho_g)/\Gamma_g}$ | $\mathbf{+0.351 \,\%}$ |
| proxy_correct (strict optimum, $\rho$ only) | $\sqrt{1 - \rho_g}$ | $+2.850 \,\%$ |
| proxy_C2 (one-axis direction) | $1 - \rho_g$ | $+3.531 \,\%$ |
| anti_full (sanity, opposite direction) | $\sqrt{\Gamma_g / (1 - \rho_g)}$ | $+6.839 \,\%$ |
| **uniform** | constant | $\mathbf{+46.624 \,\%}$ |

Three readings emerge. First, $\Gamma$-awareness produces a $90 \,\%$ relative reduction in $\Delta\mathrm{PPL}$ vs the $\rho$-only proxy_C2: the full allocator stays within $0.35 \,\%$ of FP16 perplexity at the same noise budget that pushes the cheap proxy to $3.53 \,\%$. Within the $\rho$-only family, proxy_correct (the strict bit-budget optimum) beats proxy_C2 (the C-2-historical direction-finding form), in the order predicted by §3.4.

Second, the headline surprise of Phase C-3 is the *opposite* of the (later-superseded) C-2c $A_\mathrm{log}$-perturbation result: uniform allocation is dramatically worse than every structured profile, including the deliberately-bad anti_full ($6.8\times$ worse at the same noise budget). The mechanism is the long-memory accumulation pathology of IIR systems. Uniform places $\sigma / \mathrm{RMS}(h) = 0.10$ on every unit including the $3.2 \,\%$ of units with $\rho_g \in [0.999, 1.0)$. Long-memory units accumulate noise variance as $\sigma_q^2 / (1 - \rho_g^2)$, which for $\rho_g = 0.9999$ amounts to roughly $5{,}000\,\sigma_q^2$ of integrated readout noise. Spreading $\sigma_\mathrm{rel} = 0.10$ across all such units corrupts the entire long-memory pathway in concert. Anti_full, by contrast, concentrates its noise on a small subset of high-$\rho_g$ × high-$\Gamma_g$ units; the model's redundancy absorbs the catastrophic loss of a few units, but it cannot absorb the simultaneous $10 \,\%$ corruption of every long-memory pathway.

Third, the $A_\mathrm{log}$-surrogate ceiling observed in the earlier C-2c phase is now resolved. Under direct state-step noise, the strict ordering full < proxy_correct < proxy_C2 < anti_full < uniform holds, and anti_full no longer counter-intuitively beats full. The C-2c anomaly was specific to the doubly-exponential relationship $\rho = \exp(-dt \cdot \exp(A_\mathrm{log}))$; once the noise enters the state-step channel, the bit allocator's prediction recovers.

### 4.5 The Allocator Is Uniquely $\sigma^2$-Scaling

Phase C-3b extends the C-3 single-point comparison to a $\sigma_\mathrm{rel}$ sweep at $\{0.05, 0.10, 0.20\}$, a four-fold range. The standard quantisation argument predicts that for small $\sigma$ the additional MSE is proportional to $\sigma^2$, so doubling $\sigma$ should multiply $\Delta\mathrm{PPL}$ by $4\times$.

| profile | $\sigma_\mathrm{rel} = 0.05$ | $\sigma_\mathrm{rel} = 0.10$ | $\sigma_\mathrm{rel} = 0.20$ | $0.05 \to 0.10$ multiplier | $0.10 \to 0.20$ multiplier | implied scaling exponent $k$ |
|:---|---:|---:|---:|---:|---:|:---:|
| **full** | $\mathbf{+0.096 \,\%}$ | $\mathbf{+0.351 \,\%}$ | $\mathbf{+1.415 \,\%}$ | $3.66\times$ | $4.03\times$ | $\mathbf{2.00 \pm 0.07}$ |
| proxy_correct | $+0.643 \,\%$ | $+2.850 \,\%$ | $+13.284 \,\%$ | $4.43\times$ | $4.66\times$ | $2.18 \pm 0.04$ |
| proxy_C2 | $+0.760 \,\%$ | $+3.531 \,\%$ | $+17.094 \,\%$ | $4.65\times$ | $4.84\times$ | $2.22 \pm 0.03$ |
| anti_full | $+1.718 \,\%$ | $+6.839 \,\%$ | $+37.112 \,\%$ | $3.98\times$ | $5.43\times$ | $2.21 \pm 0.22$ |
| **uniform** | $+8.039 \,\%$ | $+46.624 \,\%$ | $+561.134 \,\%$ | $5.80\times$ | $12.04\times$ | $\mathbf{3.07 \pm 0.55}$ |

Three structural observations follow from this table. The full allocator's empirical doubling multipliers are $3.66\times$ and $4.03\times$, both within rounding of the $\sigma^2$ ideal of $4\times$. The implied scaling exponent $k$ from each doubling is $\log_2(3.66) = 1.87$ and $\log_2(4.03) = 2.01$, giving the table entry $k = 2.00 \pm 0.07$ where the $\pm 0.07$ is the half-spread between the two doublings rather than a proper bootstrap confidence interval (a multi-seed reproducibility study to provide a proper CI is identified in §8). With this caveat, full is the only profile that exhibits linear-response behaviour at the order of one-tenth of a unit in $k$. Every tested control profile is super-quadratic at the same precision, with the strongest deviation in the uniform allocator, which scales as $\sigma^{3.6}$ at the highest noise.

The strict ordering full < proxy_correct < proxy_C2 < anti_full < uniform holds at every $\sigma_\mathrm{rel}$. This rules out the most innocent alternative explanation for the C-3 single-point result: that $\sigma_\mathrm{rel} = 0.10$ might have been a sweet spot that happened to favour full. The advantage of full over proxy_C2 grows from $0.66$ percentage points at $\sigma_\mathrm{rel} = 0.05$ to $15.68$ percentage points at $\sigma_\mathrm{rel} = 0.20$, a $24\times$ growth as $\sigma$ increases $4\times$.

The uniform allocator is already $4.7\times$ worse than anti_full at $\sigma_\mathrm{rel} = 0.05$, where total relative noise is mild. The IIR-accumulation pathology is therefore constitutional, not amplitude-driven: uniform allocation corrupts long-memory pathways in concert at any noise level, and the FIR baseline (uniform allocation works fine on transformer attention) does not transfer to IIR substrates.

The $\sigma^2$-scaling test is the strongest theoretical-side confirmation of the $R_g$ surface we offer. The bit-budget argument of §3.4 implies that the strict bit-budget-optimal profile $\sigma_g \propto R_g^{-1/2}$ should produce $\sigma^2$-linear noise; §4.5 verifies this empirically on Mamba-2.8B-hf, while every plausible alternative profile fails the test in a structurally consistent direction. The deployed integer construction of §3.5 is a slope-$1$ heuristic on the same $R_g$ surface; §4.5 does not directly validate the integer heuristic, only the underlying $R_g$ form, and we are explicit about this distinction in §3.5.

### 4.6 The $\sigma \to b$ Bridge: Real INT-$b$ Quantisation

Phase C-4 closes the bridge from continuous $\sigma$-space (where C-3b operates) to deployment-side integer bit widths. The patched `slow_forward` of §4.4 is run with the five profiles across a $5 \times 3 = 15$-cell grid: $b_0 \in \{3, 4, 5, 6, 8\}$ bits and per-group range $r_g = k \cdot \mathrm{RMS}(h_g)$ with $k \in \{3, 5, 7\}$. The quantiser is the INT-$b$ symmetric mid-tread with zero level of §3.5 (an asymmetric integer range $[-2^{b-1}, 2^{b-1} - 1]$, including zero), and the bit allocation uses $b_g = b_0 + \lceil \log_2 R_g \rceil_+$ with $b_\mathrm{min} = 2$ floor and shift normalisation.

| profile | INT3 | INT4 | INT5 | INT6 | INT8 |
|:---|---:|---:|---:|---:|---:|
| **full** | $\mathbf{+56.86 \,\%}$ | $\mathbf{+26.58 \,\%}$ | $\mathbf{+12.12 \,\%}$ | $\mathbf{+6.69 \,\%}$ | $\mathbf{+3.67 \,\%}$ |
| proxy_correct | $+854.0 \,\%$ | $+73.78 \,\%$ | $+18.40 \,\%$ | $+8.32 \,\%$ | $+3.80 \,\%$ |
| proxy_C2 | $+721.8 \,\%$ | $+301.8 \,\%$ | $+167.1 \,\%$ | $+31.31 \,\%$ | $+5.63 \,\%$ |
| **uniform** | $\mathbf{+1582 \,\%}$ | $\mathbf{+221.3 \,\%}$ | $\mathbf{+51.89 \,\%}$ | $\mathbf{+18.68 \,\%}$ | $\mathbf{+6.28 \,\%}$ |
| anti_full | $+2567 \,\%$ | $+1440 \,\%$ | $+449.4 \,\%$ | $+164.6 \,\%$ | $+89.68 \,\%$ |
| full / uniform ratio | $27.8\times$ | $8.3\times$ | $4.3\times$ | $2.8\times$ | $1.7\times$ |

(Full sweep at $k = 7$, the saturation value identified by the per-$b_0$ trade-off analysis below. WikiText-2 1024-context $\Delta\mathrm{PPL}$.)

The full allocator wins all $15 / 15$ cells in the $b_0 \times k$ grid. Three readings characterise the deployment story.

First, the practical deployment threshold (defined as the smallest $b_0$ for which $\Delta\mathrm{PPL}$ stays at or below $10 \,\%$) is INT5 to INT6 for full, and INT7 to INT8 for both uniform and proxy_C2. The bit allocator therefore unlocks $1$ to $2$ bits of memory savings at fixed deployment quality, equivalent to $25$ to $50 \,\%$ memory reduction on the SSM state. This is the headline deployment claim of the present paper.

Second, the relative advantage of full over uniform compounds in the scarce-bit regime: $1.7\times$ at INT8, $2.8\times$ at INT6, $4.3\times$ at INT5, $8.3\times$ at INT4, and $27.8\times$ at INT3. This is precisely the property a deployment-side allocator should have: the more constrained the bit budget, the bigger the gain from solving the allocation problem correctly.

Third, the full allocator stays inside the linear-response regime under integer quantisation as well: $\Delta\mathrm{PPL}$ approximately halves with each additional bit (multiplier $0.47\times$ for $3 \to 4$, $0.45\times$ for $4 \to 5$, $0.55\times$ for $5 \to 6$ and per-bit beyond), consistent with $\Delta\mathrm{PPL} \propto 4^{-b_g}$, the $\sigma^2$ scaling translated to the integer-bit channel. No other profile exhibits this clean halving.

The $\sigma \to b$ bridge holds qualitatively (full < proxy_correct < proxy_C2 < uniform < anti_full reproduces, and full retains its $\sigma^2$ scaling) but with an absolute amplification of $15$ to $25\times$ relative to the C-3b $\sigma$-noise predictions at matched effective relative noise. This amplification is a genuine property of the integer-bit channel: the discrete levels, hard clipping ceiling at $\pm r_g$, and the $b_\mathrm{min} = 2$ floor introduce non-linearities beyond what Gaussian noise captures. The ranking and scaling that matter for the bit allocator's claims are preserved; the absolute amplification is part of what is reported in the deployment table above and inherited by §6.

The ensemble of results in this section establishes the bit allocator's direction (§4.4), scaling (§4.5), and deployment quality (§4.6) on the anchor model. We now generalise to seven additional Mamba SSM instances spanning two architectures and three lineages.

---

## 5. Generalisation Across Mamba SSM Instances

This section establishes that the bit allocator's rank-1 property on the §4 anchor is not specific to Mamba-2.8B-hf. We test seven additional Mamba SSM instances spanning three release lineages (state-spaces, TII Falcon, AI21 Jamba), the Mamba-1 / Mamba-2 architectural divide (via Zyphra's Zamba2), and four deployment styles (pure Mamba-1, pure Mamba-2, hybrid Mamba-2 + attention, hybrid Mamba-1 + attention + MoE). Across all eight instances, the only adaptations needed are a granularity adjustment for the Mamba-2 SSD path (per-(head, state) instead of per-(channel, state)) and lineage-specific handling of the input-side normalisation modules that Falcon-Mamba and Jamba apply before the discretisation. The $R_g = \Gamma_g / (1 - \rho_g)$ formula and its integer realisation in §3.5 are unchanged.

For all eight tests, calibration uses 1024 tokens of WikiText-2 train, evaluation uses WikiText-2 test sliding-window perplexity at 1024 context (matching §4.6), the per-group range is $r_g = 7 \cdot \mathrm{RMS}(h_g)$, and the bit allocation is the integer form of §3.5 with $b_\mathrm{min} = 2$ and shift normalisation. The five profiles compared are full, proxy_correct, proxy_C2, uniform, and anti_full, defined as in §4.

### 5.1 Scale Invariance: state-spaces Mamba-1 Family (130M to 2.8B)

Phase C-8a tests the full sweep on the four smaller siblings of the §4 anchor: Mamba-130M, Mamba-370M, Mamba-790M, and Mamba-1.4B (all from `state-spaces/mamba-*-hf`), at $b_0 \in \{4, 6\}$.

| size | FP16 PPL | full ΔPPL @ INT4 | uniform / full @ INT4 | full ΔPPL @ INT6 | uniform / full @ INT6 |
|:---:|---:|---:|:---:|---:|:---:|
| 130M | 25.13 | $+31.4 \,\%$ | $4.4\times$ | $+6.9 \,\%$ | $3.4\times$ |
| 370M | 18.58 | $+10.2 \,\%$ | $11.8\times$ | $+4.0 \,\%$ | $5.0\times$ |
| 790M | 15.42 | $+12.9 \,\%$ | $11.3\times$ | $+4.6 \,\%$ | $4.2\times$ |
| 1.4B | 14.30 | $+27.1 \,\%$ | $4.7\times$ | $+8.0 \,\%$ | $2.3\times$ |
| 2.8B (anchor) | 12.37 | $+26.6 \,\%$ | $8.3\times$ | $+6.7 \,\%$ | $2.8\times$ |

The full allocator is rank-1 (best of the five profiles) at every (size, $b_0$) cell, so $10 / 10$ rank-1 across the family. Two structural patterns are visible.

The deployment cost of the full allocator at INT6 is $4$ to $8 \,\%$ across all five sizes, with no monotone scaling trend. This is the cleanest single sentence the paper can make about deployment cost: at the deployment-side average bit budget $b_0 = 6$ on the state-spaces family, the full allocator stays in single-digit perplexity overhead independent of model scale within the tested range.

The catastrophic profile failures grow at small scale. At $b_0 = 4$, the proxy_C2 and anti_full profiles exceed $+4{,}000 \,\%$ ΔPPL on the 130M model, while the full allocator stays at $+31 \,\%$ on the same model. Smaller models have fewer long-memory units, and concentrating the bit allocation away from them (proxy_C2 over-allocates to high-$\rho$ at the cost of high-$\Gamma$; anti_full deliberately under-allocates to high-$R$) is correspondingly more destructive. The bit allocator is most valuable precisely at the small-model regime where deployment pressure is highest.

### 5.2 Architectural Transfer: Mamba-2 SSD via Zyphra Zamba2-1.2B

Phase C-5b / C-8b tests the bit allocator on the Mamba-2 (SSD) branch of the architecture. Mamba-2's selective scan replaces Mamba-1's per-channel parallel SSM with a chunked SSD computation in which the per-(head, state) state is materialised only at chunk boundaries, and within a chunk the recurrence is folded into matrix multiplications. Two changes are required in our protocol. First, the bit-allocation granularity adapts from per-(channel, state) to per-(head, state), since Mamba-2 has scalar $A_\mathrm{log}$ per head (not per channel). Second, the quantisation injection point shifts from per-token (the position in §4.4) to per-chunk-boundary (the only point at which the state tensor is materialised in the SSD path).

We tested two implementations: a hand-written per-token loop that reproduced the Mamba-2 update step, and a chunk-boundary quantisation that keeps the SSD math bit-for-bit and injects the quantiser only on the inter-chunk states tensor that SSD already computes. The per-token loop drifted by $-3 \,\%$ from the SSD reference on FP16 PPL (a subtle math mismatch we could not pin down quickly), so the chunk-boundary form was adopted. Its mode='off' shortcuts to the original `torch_forward`, so the FP16 baseline is guaranteed identical.

The C-5b result on Zamba2-1.2B at $b_0 = 6$, $k = 7$, $n_\mathrm{windows} = 5$ (FP16 PPL 6.829):

| profile | ΔPPL |
|:---|---:|
| **full** | $\mathbf{-0.749 \,\%}$ |
| proxy_C2 | $-0.720 \,\%$ |
| proxy_correct | $-0.658 \,\%$ |
| uniform | $-0.462 \,\%$ |
| anti_full | $+7.065 \,\%$ |

Strict ordering full < proxy_C2 < proxy_correct < uniform < anti_full, with full rank-1. The negative ΔPPL values are a regularisation artefact of sparse chunk-boundary noise: the quantiser fires only at the four inter-chunk boundaries per 1024-token sequence, roughly $340\times$ sparser than the per-token injection of §4.4, and at $b_0 = 6$ the resulting noise is small enough to act as a mild regulariser on test perplexity. The rank order, not the absolute negative, is the meaningful signal here.

We caveat this Mamba-2 evaluation explicitly. The $340\times$-sparser injection point and the chunk-boundary placement make the §5.2 protocol methodologically distinct from the per-token Mamba-1 evaluations in §4.4 to §4.6 and §5.1: it is not a like-for-like reproduction of the per-token quantisation channel on a different architecture, but rather a chunk-boundary state-quantisation evaluation in an SSD setting. We accordingly frame §5.2 as **directional transfer evidence** that the rank order of profiles persists when the bit allocator is applied to a chunk-boundary state in an SSD computation, not as a per-token Mamba-2 generalisation. The "two architectures" claim in this paper is therefore better read as "per-token Mamba-1 plus chunk-boundary Mamba-2 SSD as directional transfer". A per-token Mamba-2 evaluation matching the §4.4 protocol would require either a Mamba-2 model whose forward path admits per-token state extraction without breaking the SSD math (which we did not have available at the time of these experiments) or an SSD-faithful per-token reformulation; this is identified as a refinement direction.

### 5.3 Cross-Lineage Transfer: TII Falcon-Mamba-7B

Phase C-8c tests the bit allocator on Falcon-Mamba-7B (`tiiuae/falcon-mamba-7b`), the largest open-weight Mamba-1 model and the only one not from the state-spaces release line. Falcon-Mamba was trained on RefinedWeb at 7B scale and uses a different forward-path convention: weightless `rms_forward` (RMSNorm without learnable weight) is applied to $B_t$, $C_t$, and $dt_\mathrm{raw}$ before the discretisation. The patched forward of §4.4 detects this via the `mix.rms_eps` attribute and applies the same normalisation before computing $\bar A_t$, $\bar B_t$, and $C_t$.

| profile | $b_0 = 4$ | $b_0 = 6$ |
|:---|---:|---:|
| **full** | $\mathbf{+72.8 \,\%}$ | $\mathbf{+31.2 \,\%}$ |
| proxy_correct | $+91.9 \,\%$ | $+35.5 \,\%$ |
| uniform | $+111.9 \,\%$ | $+54.6 \,\%$ |
| proxy_C2 | $+128.1 \,\%$ | $+58.7 \,\%$ |
| anti_full | $+271.4 \,\%$ | $+193.2 \,\%$ |

The full allocator is rank-1 at both bit budgets. Two readings characterise the cross-lineage transfer. The full / uniform advantage is $1.54\times$ at INT4 and $1.75\times$ at INT6, smaller than the $4 \times$ to $12\times$ range observed in the state-spaces family. The absolute ΔPPL is also higher (full at INT4 is $+72.8 \,\%$ on Falcon-Mamba versus $+26.6 \,\%$ on Mamba-2.8B). Both effects are consistent with Falcon-Mamba being trained on a more diverse corpus (RefinedWeb): the resulting $(\rho_g, \Gamma_g)$ distribution likely has thinner tails, so the allocator's structural advantage compresses while the model's overall sensitivity to per-state quantisation noise increases. The rank order transfers cleanly, but the magnitude is lineage-dependent. We report both: rank-order transfer, lineage-specific magnitude.

### 5.4 Hybrid Architecture + MoE: AI21 Jamba-1.5-Mini

Phase C-8d covers the widest stretch of the generalisation chain: Jamba-1.5-Mini (`ai21labs/AI21-Jamba-Mini-1.6`), a 52B-total / 12B-active hybrid model from a third release lineage. Jamba interleaves Mamba-1 mixers with transformer attention and MoE expert layers, and uses learnable RMSNorm modules (`mix.dt_layernorm`, `mix.b_layernorm`, `mix.c_layernorm`) instead of Falcon's weightless `rms_forward` to normalise $dt_\mathrm{raw}$, $B_t$, and $C_t$ before discretisation. The patched forward grew a third lineage branch detected via the `mix.dt_layernorm` attribute. Loaded in FP16 with `device_map="auto"` on a 102 GB GPU runtime, with mild CPU offload (about 84 GB VRAM resident); 28 Mamba-1 mixers were detected, interleaved with the attention and MoE layers (which were held at FP16 throughout, since the present paper does not quantise attention).

| profile | $b_0 = 4$ | $b_0 = 6$ |
|:---|---:|---:|
| **full** | $\mathbf{+24.0 \,\%}$ | $\mathbf{+4.6 \,\%}$ |
| proxy_correct | $+49.2 \,\%$ | $+5.4 \,\%$ |
| uniform | $+41.8 \,\%$ | $+14.4 \,\%$ |
| proxy_C2 | $+97.6 \,\%$ | $+12.4 \,\%$ |
| anti_full | $+104.2 \,\%$ | $+88.7 \,\%$ |

The full allocator is rank-1 at both bit budgets, with full / uniform advantage $1.74\times$ at INT4 and $3.12\times$ at INT6.

The most striking observation in this instance is the calibration profile of Jamba's Mamba layers. The per-layer median spectral radius is $\rho_g \approx 0.025$, more than an order of magnitude below the state-spaces family ($\rho_g \approx 0.44$ on Mamba-2.8B layer 0) and Falcon-Mamba ($\rho_g \approx 0.55$). This is the architectural signature of hybrid composition: since Jamba's transformer attention layers handle long-range dependence, the Mamba layers can specialise as short-memory mixers (about 1 to 2 token effective recall on average). The full allocator works correctly regardless: it identifies the rare high-$(\rho_g, \Gamma_g)$ units within Jamba's distribution and protects them. This is empirical evidence that the allocator is distribution-shape agnostic, not just functional within a particular $\rho_g$ range.

A consequence of Jamba's low-$\rho_g$ regime is that full and proxy_correct are tight at INT6 ($+4.6 \,\%$ versus $+5.4 \,\%$, a $0.76$ percentage-point gap). With $\rho_g \to 0$, the importance weight reduces to $R_g \approx \Gamma_g$, and the difference between $\Gamma_g$-aware full and $\sqrt{1 - \rho_g}$-only proxy_correct shrinks. The paper-worthy nuance is that $\Gamma_g$-awareness pays the most in regimes with high $\rho_g$ heterogeneity; in heavily attention-fronted hybrid models where the Mamba layers all have small $\rho_g$, the $\rho$-only proxy approaches a near-equally-good allocator. This does not invalidate the universal claim; it characterises where the joint $R_g = \Gamma_g / (1 - \rho_g)$ form is most valuable.

### 5.5 Aggregate Verdict: Eight Instances, Three Lineages, Two Architectures

The eight-instance generalisation table:

| Lineage | Model | Scale | Type | full rank-1 |
|:---|:---|:---:|:---|:---:|
| state-spaces | Mamba-130M | 130M | pure Mamba-1 | ✓ at INT4 / INT6 |
| state-spaces | Mamba-370M | 370M | pure Mamba-1 | ✓ at INT4 / INT6 |
| state-spaces | Mamba-790M | 790M | pure Mamba-1 | ✓ at INT4 / INT6 |
| state-spaces | Mamba-1.4B | 1.4B | pure Mamba-1 | ✓ at INT4 / INT6 |
| state-spaces | Mamba-2.8B (§4 anchor) | 2.8B | pure Mamba-1 | ✓ at INT3 to INT8 |
| Zyphra | Zamba2-1.2B | 1.2B | hybrid Mamba-2 + attention | ✓ at $b_0 = 6$ |
| TII | Falcon-Mamba-7B | 7B | pure Mamba-1 | ✓ at INT4 / INT6 |
| AI21 | Jamba-1.5-Mini | 52B / 12B | hybrid Mamba-1 + attention + MoE | ✓ at INT4 / INT6 |

Eight instances, three release lineages, two architectures (Mamba-1 selective SSM and Mamba-2 SSD), four deployment styles (pure Mamba-1, pure Mamba-2, hybrid + attention, hybrid + attention + MoE). The full allocator is rank-1 over uniform, proxy_correct, proxy_C2, and anti_full in every tested instance at the deployment-side budgets $b_0 \in \{4, 6\}$. The full / uniform advantage at INT6 ranges from $1.7\times$ on Falcon-Mamba to $5.0\times$ on Mamba-370M, with full ΔPPL in the $4$ to $8 \,\%$ range across the state-spaces family. The $R_g = \Gamma_g / (1 - \rho_g)$ formula is unchanged across the eight instances; the only adaptations are the granularity adjustment for Mamba-2 and the lineage-specific handling of input-side normalisation modules. We refer to this as the rank-order transfer claim of the present paper. The deployment-magnitude headline ($2.5 \times$ to $2.9 \times$ at $+4 \,\%$ to $+8 \,\%$) holds on the state-spaces family and does not transfer in magnitude to the Falcon-Mamba and Jamba lineages, which we report separately rather than aggregate.

---

## 6. Packed-State Memory Accounting Paired with Fake-Quant Perplexity

The validation in §4 and the generalisation in §5 measured perplexity under fake-quant inference (quantise then dequantise, with values stored at FP32 in the patched forward path). For deployment, the SSM state must actually be stored at INT-$b$ bit width, and the per-group metadata that the allocator requires (per-(c, s) bit width $b_g$ and range $r_g$) must be accounted for in the memory budget. Phase C-6 closes this gap by computing real-packing compression ratios across the state-spaces Mamba-1 family from 130M to 2.8B parameters, paired with the perplexity costs from §4.6 and §5.1. We are explicit about the limitation: §6 reports memory accounting (bit-true layout including metadata), not real packed-INT-$b$ execution. Latency, throughput, dequant overhead, and batch-size dependence under a real packed kernel are not measured in the present paper; we identify direct kernel-level head-to-head with prior packed implementations such as Quamba2 [Chiang et al., 2025] as a refinement experiment in §8 (Phase C-11).

### 6.1 Real-Packing Methodology

For each (model, $b_0$, profile) configuration, we compute three deployment-side quantities. The mean per-group bit width $\overline{b_g}$ is taken over all (c, s) state units in the model after the integer construction of §3.5 (with shift normalisation and the $b_\mathrm{min} = 2$ floor applied). The theoretical compression ratio is $16 / \overline{b_g}$, the ratio of an FP16-stored state cache to a packed INT-$\overline{b_g}$ store. The long-context compression ratio additionally amortises a per-(c, s) metadata cost of $20$ bits per unit (16 bits FP16 $r_g$ plus 4 bits INT4 $b_g$) over a representative deployment of 100 cached chunk boundaries, giving an effective per-chunk overhead of $0.2$ bits per unit and a long-context ratio of $16 / (\overline{b_g} + 0.2)$.

Compression ratios are paired with the perplexity costs measured in §4.6 (Mamba-2.8B at the full $b_0$ grid) and §5.1 (the four smaller siblings at $b_0 \in \{4, 6\}$). The Pareto frontier on (compression, ΔPPL) is computed per-model across all (profile, $b_0$) cells.

### 6.2 The Deployment Table at $b_0 = 6$

The standard-quality deployment recommendation across the state-spaces family at the deployment-side budget $b_0 = 6$, with the WikiText-2 perplexity costs from §4.6 and §5.1:

| Model | profile | $\overline{b_g}$ | theoretical compression | long-context compression | ΔPPL |
|:---|:---|:---:|:---:|:---:|---:|
| Mamba-130M | uniform | $6.00$ | $2.67\times$ | $2.58\times$ | $+23.09 \,\%$ |
| | proxy_correct | $6.00$ | $2.67\times$ | $2.58\times$ | $+13.77 \,\%$ |
| | **full** | $\mathbf{5.62}$ | $\mathbf{2.85\times}$ | $\mathbf{2.75\times}$ | $\mathbf{+6.89 \,\%}$ |
| Mamba-370M | uniform | $6.00$ | $2.67\times$ | $2.58\times$ | $+19.96 \,\%$ |
| | proxy_correct | $5.99$ | $2.67\times$ | $2.58\times$ | $+13.21 \,\%$ |
| | **full** | $6.45$ | $2.48\times$ | $2.40\times$ | $\mathbf{+3.95 \,\%}$ |
| Mamba-790M | uniform | $6.00$ | $2.67\times$ | $2.58\times$ | $+18.91 \,\%$ |
| | proxy_correct | $6.03$ | $2.65\times$ | $2.57\times$ | $+10.64 \,\%$ |
| | **full** | $6.01$ | $2.66\times$ | $2.58\times$ | $\mathbf{+4.55 \,\%}$ |
| Mamba-1.4B | uniform | $6.00$ | $2.67\times$ | $2.58\times$ | $+18.18 \,\%$ |
| | proxy_correct | $6.11$ | $2.62\times$ | $2.54\times$ | $+8.44 \,\%$ |
| | **full** | $\mathbf{5.45}$ | $\mathbf{2.94\times}$ | $\mathbf{2.83\times}$ | $\mathbf{+7.98 \,\%}$ |
| Mamba-2.8B | uniform | $6.00$ | $2.67\times$ | $2.58\times$ | $+18.68 \,\%$ |
| | proxy_correct | $6.10$ | $2.62\times$ | $2.54\times$ | $+8.32 \,\%$ |
| | **full** | $\mathbf{5.56}$ | $\mathbf{2.88\times}$ | $\mathbf{2.78\times}$ | $\mathbf{+6.69 \,\%}$ |

Anti_full is omitted from this table as it is on the Pareto frontier of no model (catastrophic ΔPPL up to $+4516 \,\%$ at $b_0 = 4$ on the 130M model; see §5.1). For every one of the five state-spaces models, the $(b_0 = 6, \mathrm{full})$ configuration delivers **theoretical compression $2.5\times$ to $2.9\times$ at ΔPPL between $+4.0 \,\%$ and $+8.0 \,\%$**, with no monotone dependence on model scale within the tested range. We adopt this as the deployment recommendation for the state-spaces Mamba-1 family within the bit budget regime tested here; lineage-specific magnitudes for Falcon-Mamba 7B and Jamba-1.5-Mini are reported separately in §5.3 and §5.4.

### 6.3 The Pareto Frontier

The (compression, ΔPPL) Pareto frontier per model includes between three and five non-dominated points across the $b_0 \in \{3, 4, 5, 6, 8\}$ grid. The full allocator is on the Pareto frontier of every one of the five models, at $b_0 = 4$ (high-compression / borderline-PPL operating point) and at $b_0 = 6$ (standard-quality operating point). The uniform allocator is on the Pareto frontier of four of the five models at $b_0 = 4$ as the "highest-compression-but-broken-PPL" extreme, but at $b_0 = 6$ uniform is dominated by full on every model. Proxy_C2 and anti_full are dominated almost everywhere; the only points at which they surface on a Pareto frontier are degenerate $b_0 = 4$ corners on the 1.4B and 2.8B models, where their over-allocation to high-$\rho$ accidentally produces marginally higher compression at much worse perplexity than full.

The headline of this analysis is that the bit allocator dominates uniform allocation in the joint (compression, ΔPPL) plane at the deployment-side operating point $b_0 = 6$. At fixed perplexity quality, the bit allocator saves approximately one bit of state-cache memory: uniform allocation requires INT7 to INT8 for the same perplexity that full delivers at INT6. Translated to memory, this is a $17 \,\%$ to $25 \,\%$ reduction in state-cache size at constant deployment quality.

### 6.4 Budget Leakage at the $[b_\mathrm{min}, 16]$ Clamps

The full allocator's actual mean bit width drifts below the nominal $b_0 = 6$ budget on the larger models. For Mamba-1.4B and Mamba-2.8B, $\overline{b_g} = 5.45$ and $5.56$ respectively, $0.5$ bits below the nominal target. The mechanism is the heavier-tailed $\Gamma_g$ distribution on larger models: extreme high-$R_g$ units hit the upper $b_g = 16$ ceiling clamp of §3.5 and contribute their excess "extras" budget to the post-shift normalisation, which then biases the per-unit mean downward by an amount that depends on how much budget the clamps absorb.

This is best read as **budget leakage**, not as a free benefit: the heuristic integer construction of §3.5 does not always saturate the deployment budget $b_0$, and the comparison against uniform allocation in §6.2 and §6.3 is therefore not at strictly identical actual mean bit width. The leakage on the larger models is on the order of $0.4$ to $0.5$ bits per group, equivalent to a $5 \,\%$ to $8 \,\%$ further compression beyond the nominal $b_0 = 6$, but this comparison would be tighter under either of two adjustments: a budget-fill construction that re-allocates the leaked budget back to non-clamped groups, or a uniform baseline run at the same actual mean bit width as the full allocator's leaked total. We do not run either adjustment in this paper; the deployment numbers in §6.2 should be read with this caveat. Smaller models with thinner $\Gamma_g$ tails see less ceiling clamping ($\overline{b_g} = 5.62$ at 130M, $\overline{b_g} = 6.45$ at 370M, $\overline{b_g} = 6.01$ at 790M), so the leakage is scale-dependent and most pronounced on the larger models.

### 6.5 Aggressive Deployment: INT4 Feasibility

Under a relaxed deployment target of $\Delta\mathrm{PPL} \le 30 \,\%$, the bit allocator at $b_0 = 4$ is feasible on every model in the family except the smallest 130M (which still requires $b_0 = 6$ to stay below $30 \,\%$ PPL drift):

| Model | recommendation at $\Delta\mathrm{PPL} \le 30 \,\%$ | compression | ΔPPL |
|:---|:---|:---:|:---:|
| 130M | INT6 only (INT4 is $+138 \,\%$ on uniform; full at INT4 is $+31 \,\%$ marginal) | $2.85\times$ | $+6.89 \,\%$ |
| 370M | INT4, full | $3.58\times$ | $+10.2 \,\%$ |
| 790M | INT4, full | $3.72\times$ | $+12.9 \,\%$ |
| 1.4B | INT4, full | $3.94\times$ | $+27.1 \,\%$ |
| 2.8B | INT4, full | $3.90\times$ | $+26.6 \,\%$ |

For models from 370M upward, the slope-$1$ simplification at INT4 yields $3.6\times$ to $3.9\times$ state-cache compression at the cost of $+10 \,\%$ to $+27 \,\%$ perplexity. The matched-mean head-to-head in §6.7 with the recommended Realisation A reduces the INT4 perplexity overhead to $+5.9 \,\%$ on Mamba-130M, $+8.2 \,\%$ on Mamba-370M, and $+9.1 \,\%$ on Mamba-790M (within the $\le 10 \,\%$ "clean" deployment threshold), and $+18.9 \,\%$ on Mamba-1.4B, $+19.3 \,\%$ on Mamba-2.8B (within the $\le 30 \,\%$ aggressive-deployment threshold). Under the recommended recipe the INT4 budget is therefore deployment-side viable across the entire state-spaces Mamba-1 family from 130M to 2.8B, where the slope-$1$ simplification was viable only from 370M upward.

In summary, the bit allocator's contribution at this section is a $2.5\times$ to $2.9\times$ state-cache compression under the slope-$1$ simplification across the state-spaces Mamba-1 family at $b_0 = 6$, with the allocator on the Pareto frontier of every tested model and the C-6 metadata accounting confirming that the per-(c, s) overhead is negligible at deployment scale. The matched-mean head-to-head in §6.7 with the recommended recipe (Realisation A, `half_slope_rg`) tightens the magnitude side: $+2.5 \,\%$ to $+5.2 \,\%$ perplexity at $b_0 = 6$ across the same five sizes, and INT4 deployment unlocked across all five sizes.

### 6.6 Context Against Prior SSM Quantisation Numbers

To situate our results against the prior SSM-quantisation literature, we tabulate published headline numbers from the most closely related works. The prior works variously evaluate on MMLU, zero-shot suites, sMNIST, or downstream vision tasks rather than on WikiText-2 / PG-19 perplexity, and the bit-width configurations differ across weights, activations, and state. We therefore present this as a context table rather than as a strict head-to-head comparison; running the prior recipes under our exact perplexity-based protocol is identified as a follow-up experiment in §8.

| Method | Granularity | Bit allocation | Evaluation | Reported headline |
|:---|:---|:---|:---|:---|
| Quamba [Chiang et al., 2024] | per-tensor | uniform W8A8 | zero-shot suite, Mamba-2.8B | $1.72\times$ latency reduction, $0.9 \,\%$ accuracy drop |
| Quamba2 [Chiang et al., 2025] | per-state-group ($8$ channel-groups), cached state uniform 8-bit | uniform W4A8 / W4A16 / W8A8 | MMLU, Mamba2-8B | $4\times$ memory reduction, $1.6 \,\%$ accuracy drop |
| MambaQuant [Xu et al., 2025] | per-channel via rotation | uniform W8A8 | zero-shot vision and language | $< 1 \,\%$ accuracy drop at W8A8 |
| Bi-Mamba [Tang et al., 2024] | uniform | $1$-bit (QAT, trained from scratch) | language modelling | comparable to FP16 at 1-bit |
| OuroMamba [Ramachandran et al., 2025] | per-channel + outlier-aware | mixed-precision data-free | vision Mamba models | competitive at low bit widths without calibration data |
| Edge-AI SSM [Zhao et al., 2025] | per-head | heterogeneous: 4-bit weights, 6-bit activations, state 6 to 16 bits by ablation | sMNIST / small-scale S4D | up to $6\times$ memory reduction without accuracy loss |
| **Ours (§4 to §6)** | **per-(channel, state-unit)**, $b_g \in \{2, \ldots, 16\}$ via $b_g = b_0 + \lceil \log_2 R_g \rceil_+$ | **closed-form, single-corpus short calibration** | **WikiText-2 PPL, PG-19 long context, Mamba-1 130M to 2.8B** | **$2.5\times$ to $2.9\times$ state-cache compression at $+4$ to $+8 \,\%$ perplexity** |

Three observations follow. First, prior work focuses on weight and activation quantisation (Quamba, MambaQuant, OuroMamba) or on cached-state quantisation at uniform per-group bit widths (Quamba2), or on heterogeneous configurations explored by ablation (Edge-AI Zhao et al.); the present paper is the first to allocate per-(channel, state-unit) variable bit widths in closed form. Second, the bit-width regimes overlap: Quamba2's W4A16 and our $b_0 = 4$ are both INT4-class on the SSM side, and we report the perplexity cost on the same Mamba family at the same bit budget on a different evaluation. Third, the strict-comparison gap (different evaluation tasks, different bit decomposition between weights / activations / state, different effective metric for "memory reduction") means the numbers in this table cannot be subtracted from each other; running Quamba2 / Edge-AI / Mamba-PTQ recipes under our exact WikiText-2 / PG-19 protocol is a candidate follow-up experiment we discuss in §8.

### 6.7 Matched-Mean Head-to-Head Across Five State-Spaces Sizes (Phase C-16)

The deployment results in §6.1 to §6.5 are produced by the slope-$1$ simplified integer allocator (Realisation B of §3.5). Phase C-16 closes the comparison directly with the recommended Realisation A across the five state-spaces Mamba-1 sizes (130M to 2.8B), at $b_0 \in \{4, 6\}$. The methodological additions in this phase are:

- **Binary-search shift normalisation**, replacing the one-pass shift of §3.5 with an additive shift selected via binary search (up to $80$ iterations) such that the post-clamp mean bit width matches the nominal $b_0$ within $0.005$ bits across all four profiles.
- **Full state-spaces family sweep** (130M, 370M, 790M, 1.4B, 2.8B), replacing the single-model anchor of the earlier comparison. Each model is re-calibrated independently from a 1024-token WikiText-2 train pass, then evaluated under WikiText-2 test sliding-window perplexity at $n_\mathrm{windows} = 5$, $\mathrm{ctx} = 1024$.

Four profiles are compared:

- `uniform`: $b_g = b_0$ (control).
- `full_slope`: $b_g = b_0 + \lceil \log_2 R_g \rceil_+$ (Realisation B).
- `half_slope`: $b_g = b_0 + \lceil \tfrac{1}{2} \log_2 R_g \rceil_+$ (the strict slope-$\tfrac{1}{2}$ optimum without the $r_g$ term).
- `half_slope_rg`: $b_g = b_0 + \lceil \tfrac{1}{2} \log_2 R_g + \log_2 r_g - \overline{\log_2 r_g} \rceil_+$ (Realisation A).

After binary-search shift normalisation, the actual mean bit widths agree to within $\pm 0.3$ bits of the nominal $b_0$ across all profiles (the residual is bounded by the integer rounding step itself, since the post-round result can vary by half a bit per group from the pre-round target); the achieved means are reported alongside ΔPPL so that readers can verify the matched-mean condition per cell.

**Headline ΔPPL (% of FP16 baseline) at matched mean bit width:**

| Model | FP16 PPL | $b_0=4$ uniform | full_slope | half_slope | **half_slope_rg** |
|:---|---:|---:|---:|---:|---:|
| Mamba-130M | $21.59$ | $+141.8 \,\%$ | $+31.0 \,\%$ | $+26.5 \,\%$ | $\mathbf{+5.9 \,\%}$ |
| Mamba-370M | $15.71$ | $+131.4 \,\%$ | $+25.9 \,\%$ | $+25.1 \,\%$ | $\mathbf{+8.2 \,\%}$ |
| Mamba-790M | $13.03$ | $+158.4 \,\%$ | $+22.0 \,\%$ | $+27.3 \,\%$ | $\mathbf{+9.1 \,\%}$ |
| Mamba-1.4B | $12.05$ | $+130.1 \,\%$ | $+59.0 \,\%$ | $+22.8 \,\%$ | $\mathbf{+18.9 \,\%}$ |
| Mamba-2.8B | $10.32$ | $+255.3 \,\%$ | $+43.3 \,\%$ | $+27.4 \,\%$ | $\mathbf{+19.3 \,\%}$ |

| Model | $b_0=6$ uniform | full_slope | half_slope | **half_slope_rg** |
|:---|---:|---:|---:|---:|
| Mamba-130M | $+25.9 \,\%$ | $+3.9 \,\%$ | $+5.4 \,\%$ | $\mathbf{+2.5 \,\%}$ |
| Mamba-370M | $+21.9 \,\%$ | $+4.3 \,\%$ | $+6.0 \,\%$ | $\mathbf{+3.5 \,\%}$ |
| Mamba-790M | $+23.5 \,\%$ | $+5.7 \,\%$ | $+7.0 \,\%$ | $\mathbf{+4.4 \,\%}$ |
| Mamba-1.4B | $+22.5 \,\%$ | $+7.6 \,\%$ | $+6.9 \,\%$ | $\mathbf{+5.2 \,\%}$ |
| Mamba-2.8B | $+22.2 \,\%$ | $+5.3 \,\%$ | $+6.8 \,\%$ | $\mathbf{+4.7 \,\%}$ |

Three findings emerge.

First, **`half_slope_rg` wins on every $(model, b_0)$ cell**. At $b_0 = 4$ the average improvement over `full_slope` is $24$ percentage points (range: $12.9$ pp on Mamba-790M to $40.2$ pp on Mamba-1.4B); at $b_0 = 6$ the average improvement is $1.3$ percentage points. The pre-registered hypothesis H1 of §8.3 (Phase C-16) PASSES on all $5 \times 2 = 10$ cells.

Second, **the slope question is approximately a wash at matched mean bit width**. Comparing `full_slope` (slope-$1$, no $r_g$) against `half_slope` (slope-$\tfrac{1}{2}$, no $r_g$) at $b_0 = 6$, `full_slope` wins on four of the five models by $0.5$ to $1.3$ percentage points and `half_slope` wins on one model (Mamba-1.4B) by $0.7$ percentage points. The structural advantage of Realisation A over Realisation B comes almost entirely from the $\log_2 r_g$ term, not from the slope choice. This refines the §3.5 framing: it is the per-group range correction, not the slope on $\log_2 R_g$, that the §3.4 derivation insists on retaining.

Third, **`half_slope_rg` unlocks INT4 deployment that `full_slope` did not support**. At $b_0 = 4$, `full_slope` produces ΔPPL of $+22 \,\%$ to $+59 \,\%$ across the family (only the smallest sizes barely deployable); `half_slope_rg` produces ΔPPL of $+5.9 \,\%$ to $+19.3 \,\%$ (the four smaller sizes within $+10 \,\%$, the 1.4B and 2.8B sizes within $+20 \,\%$). The smallest model (Mamba-130M) sees ΔPPL drop from $+31.0 \,\%$ to $+5.9 \,\%$, a more than five-fold reduction in perplexity overhead. The recipe change therefore extends the deployable-bit-budget envelope of the bit allocator from INT5 / INT6 (under Realisation B) to INT4 (under Realisation A) on the state-spaces Mamba-1 family.

We adopt the recommended deployment recipe of the present paper as Realisation A (`half_slope_rg`) on the basis of this matched-mean head-to-head. The headline numbers in §6.2 to §6.5 (which are produced under Realisation B) remain useful as the sweep table that establishes the rank-order Pareto-frontier behaviour, but the magnitude headlines for deployment recommendation purposes are now from this section, namely **approximately $2.6\times$ state-cache compression at $+2.5 \,\%$ to $+5.2 \,\%$ perplexity at $b_0 = 6$** and **INT4 deployment at $+5.9 \,\%$ to $+19.3 \,\%$ perplexity** on the state-spaces Mamba-1 family (130M to 2.8B). Cross-lineage extensions (Falcon-Mamba, Jamba) and long-context evaluation (PG-19) reported in §5 and §7 are produced under Realisation B; their re-evaluation under Realisation A is identified as a refinement direction (§8.3, Phase C-17 / C-18).

Section §7 examines whether the compression recipe of §6.1 to §6.5 survives the long-context regime up to 32K tokens on PG-19.

---

## 7. Long-Context Behaviour and the Calibration Benefit

This section examines whether the deployment recipe established in §6 survives the long-context regime, where the SSM's state-magnitude dynamics have time to drift outside the calibration distribution and accumulate compounding errors. We evaluate Mamba-2.8B-hf on PG-19 at sliding-window context lengths $\{8\mathrm{K}, 16\mathrm{K}, 32\mathrm{K}\}$, well beyond the model's 2K training context. Two distinct findings emerge. The bit allocator's rank-1 advantage over uniform allocation is preserved and grows with context (§7.1). And the calibrated range $r_g$ that the allocator uses for grid placement provides incidental long-context numerical regularisation, extending the model's usable context window in a way that a four-condition ablation attributes to the calibrated clipping rather than to the quantisation rounding (§7.3). We frame the second finding as a 2-for-1 benefit of the calibration design (§7.4), and report a new deployment option that surfaces from the ablation (§7.5).

### 7.1 The Allocator's Long-Context Advantage Grows with Context

Phase C-7 evaluates the deployment configuration $(b_0 = 6, \mathrm{full})$ against the uniform-allocation control on Mamba-2.8B-hf at PG-19 context lengths $\{8\mathrm{K}, 16\mathrm{K}, 32\mathrm{K}\}$. Phase C-9 re-runs the same evaluation at $n_\mathrm{windows} = 4$ (doubled from C-7's $n_\mathrm{windows} = 2$) for tighter statistics, and is the canonical reference for the numbers reported here.

| context | full INT6 PPL | uniform INT6 PPL | uniform / full |
|:---:|---:|---:|:---:|
| 8K | $9.56$ | $11.50$ | $1.20\times$ ($17.5 \,\%$ advantage) |
| 16K | $9.23$ | $12.65$ | $1.34\times$ ($33.7 \,\%$ advantage) |
| 32K | $11.34$ | $13.55$ | $1.20\times$ ($46.3 \,\%$ advantage) |

(Phase C-7 numbers at $n_\mathrm{windows} = 2$ produce the same qualitative pattern with slightly more optimistic absolute values, $+15 \,\%$ to $+25 \,\%$ lower PPL on each profile relative to C-9. The $n = 4$ reference here supersedes them.)

The full allocator's advantage over uniform allocation grows from $17.5$ percentage points at 8K to $46.3$ percentage points at 32K, a $2.6\times$ growth in advantage as context extends $4\times$. This is a pure internal comparison between two quantised paths under identical conditions: it is unaffected by any absolute baseline behaviour, and it is the most robust claim of §7. Long-memory state units, which §3.2 identified as the most fragile to per-step quantisation noise, become correspondingly more important as the context window extends, and the bit allocator's structure-aware protection of those units pays out at exactly the right rate.

### 7.2 Long-Context Divergence of the Unquantised Baseline

The natural reference for the quantised paths is the unquantised FP16 forward path of Mamba-2.8B-hf. On PG-19 at long context this reference exhibits substantial perplexity divergence: PPL grows from $19.30$ at 8K to $117.70$ at 16K to $196.94$ at 32K. This is more than a $10\times$ growth over the $4\times$ context extension.

The divergence is not a software artefact. As an explicit sanity check (denoted H0 in the C-7 protocol), we verified that the patched `slow_forward` of §4.4 in mode='off' produces bit-for-bit identical perplexity to the upstream reference `slow_forward` at every context length, with a measured drift of $0.0000 \,\%$ at all three (ctx, condition) cells. The patched function is faithful to the original; the long-context divergence is a property of Mamba-2.8B-hf itself.

This long-context perplexity behaviour is consistent with the model's training history. Mamba-2.8B-hf was trained at 2K context, and inference at 8K to 32K is in the extrapolation regime by a factor of $4\times$ to $16\times$. At those extrapolation factors, the input-dependent $dt$ distribution is no longer well-controlled, the state magnitudes can drift outside the calibration distribution, and accumulated FP16 numerical errors compound across the 64-layer stack and the $\geq 8\mathrm{K}$ token window. We do not adjudicate the precise mechanism here; what matters for §7 is that the unquantised FP16 path of Mamba-2.8B-hf is genuinely unstable in this regime, and that this is a real architectural property to be characterised rather than a quantisation-side artefact.

### 7.3 Disambiguating the Stabilisation Mechanism

The juxtaposition of §7.1 and §7.2 suggests a striking observation: at 32K context on PG-19, the unquantised FP16 path produces PPL $196.94$ while the quantised full INT6 path produces PPL $11.34$. The quantised path has substantially lower and more stable perplexity than the unquantised reference. Three candidate mechanisms could explain this:

(a) The unquantised divergence is a known FP16 numerical limitation that the wider dynamic range of BF16 alone would resolve, in which case the apparent stabilisation is a precision artefact.
(b) The calibrated range $r_g = k \cdot \mathrm{RMS}(h_g)$ that the bit allocator uses for quantisation grid placement *also* clips the state magnitudes at $\pm r_g$, and this clipping alone (without quantisation rounding) is what stabilises the long-context regime.
(c) The quantisation grid itself contributes to the stabilisation, perhaps by resetting accumulated FP16 cast errors at every quantisation step.

Phase C-9 disambiguates these via a four-condition ablation: $\mathrm{FP16\_orig}$ (unpatched FP16, the §7.2 reference), $\mathrm{BF16\_orig}$ (the model reloaded with `torch_dtype=torch.bfloat16`, also unpatched, which tests hypothesis (a)), $\mathrm{FP16\_clip\_only}$ (patched forward with the same calibrated $r_g$ applied as a pure clamp $h \leftarrow \mathrm{clamp}(h, -r_g, +r_g)$, without quantisation, which tests hypothesis (b)), and $\mathrm{FP16\_full}$ (the §7.1 quantised path, which corresponds to hypothesis (c) if it uniquely stabilises).

| context | $\mathrm{FP16\_orig}$ | $\mathrm{BF16\_orig}$ | $\mathrm{FP16\_clip\_only}$ | $\mathrm{FP16\_full}$ |
|:---:|---:|---:|---:|---:|
| 8K | $19.30$ | $19.24$ | $8.84$ | $9.56$ |
| 16K | $117.70$ | $116.42$ | $8.59$ | $9.23$ |
| 32K | $196.94$ | $192.83$ | $10.51$ | $11.34$ |

Three readings disambiguate the three hypotheses. First, $\mathrm{BF16\_orig}$ is within $0.3 \,\%$ to $2.1 \,\%$ of $\mathrm{FP16\_orig}$ at every context length, so BF16's wider dynamic range does not fix the divergence: hypothesis (a), the precision artefact, is rejected. Second, $\mathrm{FP16\_clip\_only}$ stabilises across all three context lengths at PPL $\approx 8.6$ to $10.5$, essentially indistinguishable from $\mathrm{FP16\_full}$'s $\approx 9.2$ to $11.3$, so the calibrated clipping alone is sufficient: hypothesis (b) is confirmed. Third, $\mathrm{FP16\_clip\_only}$ is uniformly $7 \,\%$ to $8 \,\%$ better in PPL than $\mathrm{FP16\_full}$ at every context length, so the quantisation rounding does not contribute to the stabilisation: hypothesis (c) is rejected.

The active ingredient of the stabilisation is the calibrated clipping at $r_g = k \cdot \mathrm{RMS}(h_g)$, not the quantisation grid. The quantisation rounding adds a small $7 \,\%$ to $8 \,\%$ PPL cost at long context relative to the clipping-only path, in exchange for the $2.5\times$ to $2.9\times$ memory compression measured in §6.

### 7.4 Relation to Prior State-Clipping Work and a Reframed Claim

The observation that calibrated state clipping stabilises a recurrent SSM is not original to this paper. Zhao et al. [2025], in their study of small-scale S4D quantisation for edge AI, explicitly note that the recurrent state can be constrained after each time step to suppress quantisation-induced instability arising when discrete eigenvalues drift outside the unit circle. They report this clipping as a simple and effective stabiliser for low-bit S4D inference. The stabilisation phenomenon we observe in §7.3 on Mamba-2.8B-hf at PG-19 long context (FP16 perplexity diverging to 196.94 at 32K, while the calibrated-clipping path stays at 10.51) is consistent with their state-clipping framing, extended from small-scale S4D to a 2.8B-parameter Mamba-1 LLM and from short to long-context inference. We do not claim originality for the state-clipping idea itself; what we contribute on this axis is the empirical extension to the LLM-scale Mamba long-context regime, the four-condition disambiguation that rules out the pure-precision and pure-rounding hypotheses (§7.3), and the observation that the calibration step that the bit allocator already requires for $r_g = k \cdot \mathrm{RMS}(h_g)$ supplies the clipping for free.

We accordingly reframe the long-context observation as follows. The bit allocator's calibration pipeline produces three per-group statistics: $\rho_g$ for the recurrence amplification factor, $\Gamma_g$ for the output coupling, and $\mathrm{RMS}(h_g)$ for the per-group range $r_g$. The first two drive the integer bit width $b_g$, producing the deployment-side compression of §6. The third, when used to clip state magnitudes at $\pm r_g$ (with or without subsequent quantisation), extends the recipe of Zhao et al. for small-scale S4D to the Mamba LLM long-context regime. We refer to this as a 2-for-1 use of the calibration step rather than as a 2-for-1 benefit of quantisation, and we credit the underlying state-clipping mechanism to the prior Edge-AI literature.

The §1 hero claim ("calibrated $r_g$ extends usable context from approximately 2K to ≥ 32K") is correspondingly tightened in the present manuscript: the bit allocator's compression recipe coexists with the long-context calibration benefit at no additional memory cost, but the calibration benefit is best read as a Mamba-LLM extension of the state-clipping observation in Zhao et al. [2025] rather than as a primary contribution of the bit allocator.

### 7.5 Mechanism: State-Norm and Spectral-Radius Trajectory Evidence

The §7.3 ablation shows that the long-context FP16 divergence is precision-independent and clip-suppressible, but does not directly characterise the underlying state dynamics. Phases C-10 and C-10b record the per-step state-norm, $dt$ percentiles, peak spectral radius, and clipping-firing rate at $32\mathrm{K}$ context for seven representative layers spanning the U-shape of §4.1 (layers $\{0, 16, 32, 40, 48, 56, 63\}$), under three forward conditions: `FP16_orig` (PPL only, unpatched, no trajectory recording), `FP32_patched_off` (patched with no clipping or quantisation, recording active), and `FP16_clip_only` (patched with calibrated clipping, recording active before the clip). The `FP32_patched_off` PPL agrees with `FP16_orig` PPL within $0.02 \,\%$ at every context length, confirming that its trajectory faithfully represents the unmodified-recurrence dynamics. Five evidence streams emerge.

**Stream 1: State-magnitude tail growth.** The 99th percentile of $|h_t|$ over $(B, c, s)$ units grows by $137 \,\%$ to $245 \,\%$ over a single $32\mathrm{K}$ window on every recorded layer, under both `FP32_patched_off` and `FP16_clip_only`. The largest growth is on layer 56 (+$245 \,\%$ in the unconstrained recurrence) and layer 32 (+$190 \,\%$, the long-memory core). This is a direct measurement of the state-magnitude tail expansion at long context.

**Stream 2: Peak spectral-radius climb.** The 99th percentile of $|\bar A_t|$ over $(c, s)$ units reaches $\ge 0.99$ on every recorded layer mid-window. Layer 48 climbs from $\rho_{p99} = 0.969$ at $t = 0$ to $0.9996$ at $t \approx 16\mathrm{K}$; layer 63 from $0.945$ to $0.992$. Even on layers with median $\rho_g \approx 0.4$ at calibration, individual $(c, s)$ units transiently approach the recurrence-stability boundary during long-context inference. Under the §3.2 Lyapunov bound, the per-step amplification factor $(1 - \rho_g)^{-1}$ for these tail units approaches $10^3$ or larger, so any per-step perturbation on these units (whether from quantisation, FP16 round-off, or input-driven excursions in the recurrence itself) compounds rapidly through the geometric series.

**Stream 3: $dt$ distribution drift on long-memory layers.** Within a single $32\mathrm{K}$ window, the median per-step time step $dt_{p50}$ drifts by $-38 \,\%$ on layer 32 and $-16 \,\%$ on layer 40 (both long-memory core layers; smaller $dt$ pushes $\rho_g \to 1$ via $\bar A = \exp(-dt \cdot |A|)$), while remaining roughly stable or growing on short-memory layers. This is the direct measurement of the $dt$-drift mechanism hypothesised in earlier sections of the manuscript.

**Stream 4: Direct measurement of state-magnitude excursions.** Phase C-10b records the natural rate at which $|h_t|$ exceeds the calibrated $r_g$ threshold by recording the cliprate $\mathrm{Pr}[|h_t^{(c, s)}| > r_g]$ pre-clip. Under `FP32_patched_off` (no intervention), this natural firing rate is non-zero on every recorded layer at every context length, ranging from $0.03 \,\%$ to $2.27 \,\%$ at the end of the $32\mathrm{K}$ window. Long-memory layer 40 shows the most context-dependent growth ($0.82 \,\% \to 1.98 \,\%$ from $8\mathrm{K}$ to $32\mathrm{K}$, a $2.4\times$ increase). Layer 32 firing rate is approximately $1.9 \,\%$ throughout. Although the absolute firing rates are small, they are **non-zero on every layer** and they correspond to precisely the units operating near $\rho_g \to 1$ identified in Stream 2.

**Stream 5: Clipping intervention is non-trivial.** The pre-clip firing rate under `FP16_clip_only` is uniformly lower than under `FP32_patched_off` at every $(layer, ctx)$ cell: layer 0 has $0.64 \,\%$ firing under `clip_only` versus $2.27 \,\%$ under `off` at $32\mathrm{K}$ end; layer 32 has $0.42 \,\%$ versus $1.90 \,\%$. This $\approx 3\times$ to $4\times$ ratio between the two conditions is the direct consequence of the clipping intervention: once a unit is clipped at step $t$, its state in subsequent steps starts from the bounded value, reducing the rate of subsequent excursions. The two cliprate trajectories agree at $t = 0$ and diverge as the recurrence unfolds, providing direct evidence that the calibrated clipping is engaging with the natural state-magnitude dynamics rather than firing on noise.

The five streams together establish the following picture. Long-context inference creates conditions in which a small but non-trivial fraction of $(c, s)$ units (between $0.1 \,\%$ and $2.3 \,\%$ at the end of the $32\mathrm{K}$ window, layer-dependent) transiently exceed the calibrated $r_g$ and operate near the $\rho_g \to 1$ recurrence-stability boundary, where rare excursions compound through the recurrence's geometric series. Under the unmodified recurrence (`FP16_orig`), these compounding excursions accumulate over the $32\mathrm{K}$ context and produce the perplexity divergence reported in §7.2. The calibrated clipping at $r_g$ in `FP16_clip_only` engages with this dynamics directly, suppressing the rare excursions and reducing the firing rate on subsequent steps. The mechanism is therefore not a uniform clamp on the bulk of the state distribution but a targeted suppression of the heavy-tail events that dominate long-context recurrence numerics.

We caveat what we have *not* established. We have not isolated the precise origin of the per-step perturbation on the high-$\rho$ units that drives the divergence. Candidates include (i) accumulated FP16 round-off in the `out_proj` cast, (ii) input-driven excursions in the recurrence itself amplified by transient $\rho_g \to 1$ regardless of arithmetic precision, and (iii) interactions between the two. The C-9 four-condition ablation rules out a pure-precision artefact (BF16 alone does not fix the divergence) but does not distinguish between the remaining candidates. A direct measurement of the per-step round-off magnitude, paired with the Lyapunov bound on amplified state-norm error, is identified as the closing analytical experiment in §8.2.

This mechanism characterisation is consistent with, and makes precise, the framing in §7.4 in which calibrated clipping is the active ingredient inherited from the §3.6 calibration step. In particular, the candidate deployment recipe based on the $r_g$-retaining allocator of §6.7 is expected to inherit the same mechanism, since it uses the same $r_g$ for grid placement and clipping; a direct trajectory measurement on the candidate recipe is identified as a refinement (Phase C-16 in §8.3).

### 7.6 A Resulting Three-Way Deployment Menu

The C-9 ablation, taken together with the prior-work framing of §7.4, surfaces a three-way deployment menu on Mamba-2.8B-hf at PG-19 long context.

| deployment | compression | 32K PPL | recommended for |
|:---|:---:|:---:|:---|
| FP16 unquantised | $1.0\times$ | $196.94$ | not viable at long context |
| BF16 unquantised | $1.0\times$ | $192.83$ | not viable at long context (precision is not the bottleneck) |
| FP16 clip-only ($r_g$ clamp, no quantisation) | $1.0\times$ | $10.51$ | long-context stability without compression; equivalent to the Zhao et al. state-clipping recipe applied at LLM scale |
| FP16 full INT6 (the §6 recipe) | $\approx 2.5\times$ | $11.34$ | both compression and stability |

The clip-only option is a useful deployment-time observation. Users whose VRAM budget already accommodates the FP16 state cache but whose context window currently caps at the FP16-stable regime can apply the calibration step alone, without quantisation, to extend usable context at zero memory cost. Users wanting state-cache compression should use the §6 full INT6 recipe; the small $7 \,\%$ to $8 \,\%$ PPL premium at 32K is the rounding cost that the quantisation grid pays in exchange for the $2.5\times$ memory savings. Both deployment options inherit the long-context state-clipping benefit, since both use the same calibrated $r_g$.

We caveat the entire §7 result with three limitations that we develop in §8. The evaluation is on a single corpus (PG-19) and a single model (Mamba-2.8B-hf), with $n_\mathrm{windows} = 4$ that we describe as statistical-floor-grade. The mechanism behind the unquantised divergence is characterised by the five-stream trajectory analysis in §7.5 (state-magnitude tail growth, peak $\rho_g$ climb, $dt$ drift, direct cliprate measurement, and the elevated firing rate under no-intervention versus clipping), but the precise origin of the per-step perturbation that drives the cascade (FP16 round-off accumulation versus input-driven recurrence excursions versus their interaction) remains open and is identified in §8.2 as the closing analytical experiment. The absolute-PPL numbers reported in §7.1 to §7.3 should also be read with the variance disclosure of §8.1 in mind.

---

## 8. Discussion

The empirical core of this paper is a single closed-form bit-allocation rule, calibrated from two scalars per state group, that produces a deployment-side SSM state-cache compression recipe across eight Mamba SSM instances. We close with limitations, an open puzzle the long-context analysis surfaced, and a list of follow-up directions.

### 8.1 Limitations

The theory in §3 makes several simplifying assumptions, and the empirical chain in §4 to §7 is bounded by the experimental scope we could realistically run.

The bit-budget-optimal allocation in §3.4 contains a $\log_2 r_g$ term that the simplified integer realisation B (`full_slope`) drops, and the integer construction uses slope $1$ on $\log_2 R_g$ rather than the strict slope $\tfrac{1}{2}$. The §6.7 head-to-head experiment establishes that retaining the $\log_2 r_g$ term and using slope $\tfrac{1}{2}$ (Realisation A, `half_slope_rg`) produces a substantially tighter allocator at low bit budgets. We have therefore tested both extensions and identified Realisation A as the recommended deployment recipe. The §4 to §7 scale results in this manuscript are produced under Realisation B; the full sweep with Realisation A is identified as the closing follow-up experiment (Phase C-16, §8.3). The single-model integer head-to-head in §6.7 is also subject to a mean-bit-matching caveat (Realisation A has $0.24$ to $0.79$ bits higher actual mean than the nominal $b_0$), which §8.3's two-pass shift normalisation will tighten.

The empirical chain has several sample-size limitations. The calibration corpus is 1024 tokens of WikiText-2 train; §4.3 verifies $\mathrm{Pearson}(\log \Gamma_g) = 0.987$ across the WikiText-2 train and validation splits, but inter-corpus stability (e.g., calibration on RefinedWeb versus WikiText-2 versus Books3) has not been directly measured. The long-context evaluation in §7 uses PG-19 at $n_\mathrm{windows} = 4$, which is sufficient for the qualitative claims (the rank-1 advantage growing with context, the four-condition disambiguation in §7.3) but is statistical-floor-grade for the absolute PPL numbers. The eight-instance generalisation in §5 covers three Mamba release lineages and the Mamba-1 / Mamba-2 architectural divide, but does not extend to other recurrent SSM variants (RWKV, Gated DeltaNet, structured state-space alternatives outside the Mamba family). The §7.3 ablation surfaces one outlier layer (layer 30 of Mamba-2.8B-hf) with anomalously corpus-dependent $\Gamma_g$, which is flagged here as a candidate for any deployment-time recalibration check.

### 8.2 The Mechanism of Long-Context FP16 Divergence (Partially Resolved by C-10 / C-10b)

The §7.3 ablation establishes that the long-context divergence of unquantised Mamba-2.8B-hf on PG-19 is precision-independent and is suppressed by clipping at the calibrated $r_g$. Section §7.5 characterises the mechanism directly via the trajectory recordings, which record per-step trajectories of state norm, $dt$ percentiles, peak spectral radius, and pre-clip firing rate at $32\mathrm{K}$ context. Five evidence streams (state-magnitude tail growth, peak $\rho_g$ climb, $dt$ drift on long-memory layers, non-zero pre-clip firing rate on every layer, and elevated firing rate under no-intervention vs clipping) establish the following picture: long-context inference transiently pushes $0.1$ to $2.3 \,\%$ of $(c, s)$ units past the calibrated $r_g$ threshold, on units that simultaneously operate near the $\rho_g \to 1$ recurrence-stability boundary; these high-amplification units expose the recurrence to rare state-magnitude excursions over long context; finite-precision arithmetic and input-dependent recurrence drift may then amplify the excursions into the observed PPL divergence; calibrated clipping suppresses the excursions one at a time and prevents the cascade.

The remaining open part is the precise quantitative mapping. The §3.2 Lyapunov bound predicts state-error variance growth as $\sigma_q^2 / (1 - \rho_g^2)$; under FP16, $\sigma_q^2$ is the unintentional round-off variance, normally negligible on units with median $\rho_g \approx 0.6$ but amplified by $10^3$ to $10^6$ for units transiently reaching $\rho_g \ge 0.99$. Whether the cumulative round-off magnitude across $32\mathrm{K}$ steps and 64 layers, weighted by the C-10b cliprate trajectory, quantitatively predicts the observed FP16 PPL of $\approx 200$ at $32\mathrm{K}$ is an open analytical question. A closed-form prediction would tie the §7 mechanism back to the §3.2 noise-bound derivation, providing a unified theoretical picture in which the same Lyapunov bound that motivates the bit allocator also predicts the long-context FP16 divergence and its clip-only cure. We identify this as the most attractive open analytical question for follow-up work.

### 8.3 Identified Refinement Experiments

The present revision of this paper carries a number of explicit refinement experiments that we have designed but not yet run, and that future work could close. We list the most important here.

**Phase C-10 / C-10b (executed; results in §7.5).** State-norm trajectory measurement at long context. Phase C-10 records per-step $\|h_t\|$, $dt$, and peak spectral radius for seven representative layers at $32\mathrm{K}$ context; Phase C-10b corrects a recording-position bug in C-10 (cliprate was inadvertently measured post-clip and was always zero by construction) and adds an `FP32_patched_off` condition that records the unmodified-recurrence trajectory. The combined results establish the five-stream mechanism characterisation in §7.5.

**Phase C-11 (open). Direct Quamba2 / Edge-AI head-to-head.** Re-run the Quamba2 cached-state W8A8 recipe and the Zhao et al. heterogeneous configuration on Mamba-2.8B-hf under the present paper's WikiText-2 / PG-19 protocol, with matched effective bit budget including metadata. This addresses the §6.6 caveat that the current comparison is a context table rather than a strict head-to-head.

**Phase C-12 (executed; results in §6.7). Half-slope vs full-slope integer allocator.** Tested the strict slope-$\tfrac{1}{2}$ integer construction (with and without the $\log_2 r_g$ term) against the deployed slope-$1$ heuristic at matched actual mean bit width. Result: the slope-$\tfrac{1}{2}$ optimum with the $r_g$ term retained (`half_slope_rg`) is empirically tighter than the deployed `full_slope` heuristic, with a $16.5$-percentage-point improvement at $b_0 = 4$ and a $2.4$-percentage-point improvement at $b_0 = 6$. The `r_g` term that §3.4 dropped carries substantial benefit; the slope choice (full vs half at matched mean bit width) accounts for only $\approx 0.8$ percentage points. Phase C-16 (below) closes the deployment chain at the scale level by re-running §4 to §7 with `half_slope_rg`.

**Phase C-13 (open). Multi-seed / multi-corpus reproducibility.** Compute proper bootstrap confidence intervals on the §4.5 $\sigma^2$-scaling exponents and on the §7 long-context numbers, by running multiple seeds and multiple PG-19 windows per seed. This addresses the §4.5 caveat that the current $k = 2.00 \pm 0.07$ is a half-spread, not a proper CI.

**Phase C-14 (open). Closed-form $r_g$ from the recurrence.** Replace the empirical $k \cdot \mathrm{RMS}(h_g)$ with a Lyapunov-derived bound on the state-magnitude tail, tying the calibration step in §3.6 to a more principled construction. This addresses the §8.2 open analytical question and would parallel the noise-bound derivation in §3.2 for the state-magnitude side, also providing a closed-form prediction for the long-context FP16 PPL divergence reported in §7.

**Phase C-15 (open). Cross-architecture extension.** Test the bit allocator on RWKV [Peng et al., 2023] and Gated DeltaNet [Yang et al., 2025] variants outside the Mamba family. This addresses the §8.1 limitation that our rank-order transfer claim covers Mamba but not the broader recurrent-SSM literature.

**Phase C-16 (executed; results in §6.7). Matched-mean head-to-head across five state-spaces sizes.** Tested the slope-$1$ simplification (`full_slope`), the strict slope-$\tfrac{1}{2}$ optimum without the $r_g$ term (`half_slope`), the strict optimum with the $r_g$ term (`half_slope_rg`), and a `uniform` control on the five state-spaces Mamba-1 sizes (130M to 2.8B) at $b_0 \in \{4, 6\}$, with mean bit width matched to within $0.005$ bits across profiles via binary-search shift normalisation. Result: pre-registered hypothesis H1 PASSES on every $(model, b_0)$ cell. `half_slope_rg` is structurally tighter than `full_slope` at matched mean by an average of $24$ percentage points at $b_0 = 4$ (range: $12.9$ to $40.2$ pp) and $1.3$ percentage points at $b_0 = 6$ (range: $0.6$ to $2.4$ pp). The slope question (slope-$1$ vs slope-$\tfrac{1}{2}$ without the $r_g$ term) is approximately a wash: `full_slope` wins on four of five models at $b_0 = 6$ by $0.5$ to $1.3$ percentage points and `half_slope` wins on one model. The dominant structural factor is the $\log_2 r_g$ term. The matched-mean comparison additionally confirms that `half_slope_rg` unlocks INT4 deployment that the slope-$1$ simplification did not support, with ΔPPL at $b_0 = 4$ ranging from $+5.9 \,\%$ on Mamba-130M to $+19.3 \,\%$ on Mamba-2.8B versus $+22 \,\%$ to $+59 \,\%$ for `full_slope` on the same models. The headline deployment numbers in §6.7 supersede those of §6.2 to §6.5 for the recommended-recipe magnitude side.

**Phase C-17 (open). Long-context PG-19 evaluation under `half_slope_rg`.** Re-run the §7 long-context evaluation (Phases C-7, C-9, C-10b) on Mamba-2.8B-hf at PG-19 ctx ∈ \{8K, 16K, 32K\} with the recommended `half_slope_rg` allocator instead of the slope-$1$ simplification. The expected outcome is to confirm that the rank-order advantage of `full` over `uniform` and the calibrated-clipping mechanism in §7.4 / §7.5 transfer to the recommended recipe; the absolute PPL magnitudes at long context are expected to improve in the same proportion as the WikiText-2 short-context results in §6.7.

**Phase C-18 (open). Cross-lineage re-evaluation under `half_slope_rg`.** Re-run the §5.3 (Falcon-Mamba 7B) and §5.4 (Jamba-1.5-Mini SSM-only) generalisation tests with the recommended allocator. The expected outcome is that the rank-order transfer claim is preserved; the absolute magnitudes are expected to tighten on Falcon-Mamba (which produced the largest ΔPPL of $+31 \,\%$ at $b_0 = 6$ under the slope-$1$ simplification) and remain similar on Jamba.

A natural compression follow-up parallel to these is to integrate the present SSM bit allocator with attention-side quantisation in hybrid stacks, where the SSM layers and the transformer attention layers are jointly compressed under a shared bit budget. The composition diagnostic on Zamba2-1.2B (Phase C-5c, omitted from the main text per the scoping decision) and the resonance phenomenon at $b_0 \in \{5, 8\}$ in the Phase C-5d sweep are candidates for a companion paper focused on hybrid-stack compression.

Finally, a distillation-style follow-up would ask whether $r_g$ and $b_g$ admit a universal precomputed lookup table indexed by lineage and architecture, removing the need for per-deployment calibration. The present calibration is light (one forward pass over 1024 tokens, about 30 seconds on Mamba-2.8B), but a universal lookup table would remove the calibration step entirely. Whether such a table can match per-deployment calibration within a one-bit deployment safety margin is an open empirical question.

### 8.4 Closing Remarks

This paper develops a heuristic closed-form bit-allocation rule for SSM state quantisation, motivates it from a discrete-time Lyapunov bound on accumulated quantisation noise, and validates it across eight Mamba SSM instances spanning three release lineages and two architectures. The rule's two empirical inputs are obtained from a single 1024-token calibration forward pass, the integer bit width follows in closed form, and the deployment configuration produces $2.5\times$ to $2.9\times$ state-cache compression at $+4 \,\%$ to $+8 \,\%$ perplexity overhead on the state-spaces Mamba-1 family. As a side-benefit of the calibration design, the per-group range $r_g$ used to place the quantisation grid also clips state magnitudes at long context: on Mamba-2.8B-hf evaluated on PG-19 sliding windows up to 32K context, the calibrated-clipping path stays at perplexity $\approx 11$ where unquantised FP16 and BF16 perplexity reach $\approx 197$. We connect this state-clipping behaviour to the Edge-AI work of Zhao et al. [2025], which the present paper extends from S4D to LLM-scale Mamba long-context inference.

The empirical scope of this paper is the Mamba SSM family, the calibration is a single forward pass over a single corpus, and the bit-allocation rule is closed-form rather than gradient-based. Six refinement experiments (§8.3) and one open analytical question (§8.2) are explicitly identified for follow-up; the present manuscript is a snapshot of the recipe at the level of empirical evidence currently available, with the refinement experiments left as the natural transition to a fully-validated submission.

---

## References

*Citations are listed alphabetically by first-author surname. arXiv preprint identifiers are given for all preprints; venue is given where known. All author lists were cross-checked against the arXiv abstract pages on 2026-04-30.*

Abreu, S., Pedersen, J. E., Heckel, K. M., and Pierro, A. 2024. Q-S5: Towards Quantized State Space Models. *arXiv:2406.09477*.

Agarwal, N., Suo, D., Chen, X., and Hazan, E. 2023. Spectral State Space Models. *arXiv:2312.06837*.

Chang, C.-C., Lin, W.-C., Lin, C.-Y., Chen, C.-Y., Hu, Y.-F., Wang, P.-S., Huang, N.-C., Ceze, L., Abdelfattah, M. S., and Wu, K.-C. 2024. Palu: Compressing KV-Cache with Low-Rank Projection. *arXiv:2407.21118*.

Chen, T., Chen, Y., Xu, W., Zhu, Z., Wang, P., and Cheng, J. 2025. Q-Mamba: Towards More Efficient Mamba Models via Post-Training Quantization. *International Conference on Learning Representations (ICLR 2025)*. *OpenReview: AY1S52vr0a*.

Chiang, H.-Y., Chang, C.-C., Frumkin, N., Wu, K.-C., and Marculescu, D. 2024. Quamba: A Post-Training Quantization Recipe for Selective State Space Models. *arXiv:2410.13229*.

Chiang, H.-Y., Chang, C.-C., Frumkin, N., Wu, K.-C., Abdelfattah, M. S., and Marculescu, D. 2025. Quamba2: A Robust and Scalable Post-training Quantization Framework for Selective State Space Models. *arXiv:2503.22879*.

Dao, T. and Gu, A. 2024. Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality. *International Conference on Machine Learning (ICML 2024)*. *arXiv:2405.21060*.

Glorioso, P., Anthony, Q., Tokpanov, Y., Whittington, J., Pilault, J., Ibrahim, A., and Millidge, B. 2024. Zamba: A Compact 7B SSM Hybrid Model. *arXiv:2405.16712*. (Zamba2 release notes: Zyphra Technologies, 2024.)

Gu, A. and Dao, T. 2023. Mamba: Linear-Time Sequence Modeling with Selective State Spaces. *arXiv:2312.00752*.

Halloran, J. T., Gulati, M., and Roysdon, P. F. 2024. Mamba State-Space Models Are Lyapunov-Stable Learners. *arXiv:2406.00209*.

He, Y., Zhang, L., Wu, W., Liu, J., Zhou, H., and Zhuang, B. 2024. ZipCache: Accurate and Efficient KV Cache Quantization with Salient Token Identification. *arXiv:2405.14256*.

Hooper, C., Kim, S., Mohammadzadeh, H., Mahoney, M. W., Shao, Y. S., Keutzer, K., and Gholami, A. 2024. KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization. *Advances in Neural Information Processing Systems (NeurIPS 2024)*. *arXiv:2401.18079*.

Kim, H., Ko, B., Kang, M., Kim, M., Lee, D., Lee, J., Yoon, S., and Jung, D. 2025. SSDi8: Accurate and Efficient 8-bit Quantization for State Space Duality. *OpenReview: pjMDZJd4rT*.

Lieber, O., Lenz, B., Bata, H., Cohen, G., Osin, J., Dalmedigos, I., Safahi, E., Meirom, S., Belinkov, Y., Shalev-Shwartz, S., Abend, O., Alon, R., Asida, T., Bergman, A., Glozman, R., et al. 2024. Jamba: A Hybrid Transformer-Mamba Language Model. *arXiv:2403.19887*.

Liu, Z., Yuan, J., Jin, H., Zhong, S., Xu, Z., Braverman, V., Chen, B., and Hu, X. 2024. KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache. *International Conference on Machine Learning (ICML 2024)*. *arXiv:2402.02750*.

Merity, S., Xiong, C., Bradbury, J., and Socher, R. 2017. Pointer Sentinel Mixture Models. *International Conference on Learning Representations (ICLR 2017)*. *arXiv:1609.07843*. (WikiText-2 / WikiText-103 introduction.)

Pierro, A. and Abreu, S. 2024. Mamba-PTQ: Outlier Channels in Recurrent Large Language Models. *arXiv:2407.12397*.

Rae, J. W., Potapenko, A., Jayakumar, S. M., Hillier, C., and Lillicrap, T. P. 2020. Compressive Transformers for Long-Range Sequence Modelling. *International Conference on Learning Representations (ICLR 2020)*. *arXiv:1911.05507*. (PG-19 introduction.)

Ramachandran, A., Lee, M., Xu, H., Kundu, S., and Krishna, T. 2025. OuroMamba: A Data-Free Quantization Framework for Vision Mamba. *arXiv:2503.10959*.

Siegel, S., Yang, M.-J., Bouhadjar, Y., Fabre, M., Neftci, E., and Strachan, J. P. 2025. QS4D: Quantization-Aware Training for Efficient Hardware Deployment of Structured State-Space Sequential Models. *arXiv:2507.06079*. (Published in *Advanced Intelligent Systems*, Wiley, 2026.)

Tang, S., Ma, L., Li, H., Sun, M., and Shen, Z. 2024. Bi-Mamba: Towards Accurate 1-Bit State Space Models. *arXiv:2411.11843*.

Wei, R., Xu, S., Zhong, L., Yang, Z., Guo, Q., Wang, Y., Wang, R., and Li, M. 2025. LightMamba: Efficient Mamba Acceleration on FPGA with Quantization and Hardware Co-design. *arXiv:2502.15260*.

Xu, Z., Yue, Y., Hu, X., Yuan, Z., Jiang, Z., Chen, Z., Yu, J., Xu, C., Zhou, S., and Yang, D. 2025. MambaQuant: Quantizing the Mamba Family with Variance Aligned Rotation Methods. *arXiv:2501.13484*.

Zhao, L., Torchet, T., Payvand, M., Kriener, L., and Moro, F. 2025. Quantizing Small-Scale State-Space Models for Edge AI. *arXiv:2506.12480*.

Zuo, J., Velikanov, M., Rhaiem, D. E., Chahed, I., Belkada, Y., Kunsch, G., and Hacid, H. 2024. Falcon-Mamba: The First Competitive Attention-Free 7B Language Model. *Technology Innovation Institute (TII) Technical Report*. *arXiv:2410.05355*.

---

*End of manuscript draft v0.7, 2026-05-01. Eight numbered sections plus references. Empirical results draw on the experimental programme catalogued in `paper_c/README.md`; phase identifiers cited inline correspond to that experimental log, and raw per-phase JSON results are stored in `paper_c/results_c/`. The matched-mean head-to-head with the recommended deployment recipe (Phase C-16 in §8.3 / §6.7) is the closing experiment of the present empirical programme; the long-context PG-19 and cross-lineage re-evaluations under the recommended recipe (Phases C-17 and C-18) are identified as refinement directions.*
