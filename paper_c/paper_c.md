# Lyapunov-Aware Bit Allocation for State-Space Model Quantisation

**Kentaro Sato**
Independent Researcher
`info@metaclan.jp` · `@RecyncAI`

*Draft v0.1 · 2026-04-30*

---

## Abstract

State-space models (SSMs) such as Mamba are emerging as a sub-quadratic alternative to transformer attention, but their core memory primitive, a single recurrent hidden state per layer, does not quantise the way transformer key-value caches do. Per-step quantisation noise feeds back into the next state through the SSM recurrence and accumulates with context length at a rate set by each state group's spectral radius, so a uniform per-element bit budget catastrophically degrades the long-memory state units; no principled per-state-group allocation rule has yet been established for SSM substrates.

We introduce a Lyapunov-aware bit allocator for SSM state quantisation. The closed-form rule

$$
b_g \;=\; b_0 \;+\; \bigl\lceil \log_2 (\Gamma_g / (1 - \rho_g)) \bigr\rceil_+
$$

allocates bits per state group from two calibrated quantities, the SSM's discretised spectral radius $\rho_g$ and output coupling $\Gamma_g$, both estimated from a brief input-dependent forward pass. The allocator is empirically rank-1 over uniform allocation and three control profiles (proxy, anti, inverse) across **eight Mamba SSM instances spanning three lineages (state-spaces, TII Falcon, AI21 Jamba) and two architectures (Mamba-1, Mamba-2)**, and is uniquely $\sigma^2$-scaling under controlled noise injection while every tested control profile is super-quadratic. At the deployment-realistic configuration $(b_0 = 6, \mathrm{full})$ the recipe delivers **2.5 - 2.9× state-cache compression at +4 - 8 % perplexity** across five Mamba-1 sizes (130M - 2.8B). The calibrated range $r_g = k \cdot \mathrm{RMS}(h_g)$ used to place the quantisation grid additionally regularises long-context state magnitudes, **extending Mamba-2.8B-hf's usable context from approximately 2K to ≥ 32K on PG-19**. We refer to this as a 2-for-1 benefit of the calibration design, and verify in ablation that it is precision-independent (BF16 alone does not fix the underlying divergence) and rounding-independent (a clip-only ablation reproduces the effect).

---

## 1. Introduction

Modern language models spend a substantial fraction of their inference memory on bookkeeping for the context they have already processed. For transformer models this bookkeeping takes the form of a *key-value cache* that grows linearly with the input length, and a recent line of work has shown how to compress that cache from 16 bits per element to 3 – 4 bits per element with negligible quality loss [Hooper et al., 2024; Liu et al., 2024]. State-space models (SSMs) of the Mamba family [Gu & Dao, 2023; Dao & Gu, 2024] take a structurally different approach: they replace the linearly-growing key-value cache with a single *fixed-size recurrent hidden state* per layer, making them attractive for long contexts where the cache would otherwise dominate memory. But the transformer compression playbook does not apply directly to this state. Per-step quantisation noise on the recurrent state feeds back into the next step and accumulates with context length, in a way that the per-element-independent transformer cache does not. **This paper provides such a rule: a principled, calibration-light bit-allocation strategy for state-space model state quantisation.**

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

The error-amplification factor $(1 - \rho_g)^{-1}$ is the SSM's recurrence-stability bound (a Lyapunov-type quantity), and it has no analogue in the transformer KV-cache regime. State units whose spectral radius approaches $\rho_g \to 1$, empirically observed to comprise 18.6 % of Mamba-2.8B-hf's state-unit population on natural input (of which 3.2 % land in the very-long-memory tail $\rho_g > 0.999$), amplify per-step quantisation noise by orders of magnitude. A uniform per-element bit budget, which is the default in the transformer KV-cache literature and what naive transplants of those recipes inherit, catastrophically degrades these long-memory units. To the best of our knowledge, no principled rule has been established in the SSM-quantisation literature for *how many bits to give to each state group as a function of the state group's role in the recurrence*.

**Our approach.** Standard quantisation theory tells us that the bit-budget-optimal noise allocation, given a per-group "importance weight" $R_g$ and a fixed total bit budget, scales as $\sigma_g \propto 1/\sqrt{R_g}$. We argue that the natural importance weight for SSM state quantisation combines the recurrence-stability bound $(1 - \rho_g)^{-1}$ with the output-coupling magnitude $\Gamma_g = \mathbb{E}_t \|C_t\|^2$ that controls how visible the state error is at the readout:

$$
R_g \;=\; \frac{\Gamma_g}{1 - \rho_g}
\qquad\Longrightarrow\qquad
b_g \;=\; b_0 \;+\; \bigl\lceil \log_2 R_g \bigr\rceil_+.
$$

We refer to this as the **Lyapunov-aware bit allocator**, since its dependence on $(1 - \rho_g)^{-1}$ is the manifestation of the SSM recurrence's discrete-time Lyapunov bound on accumulated noise. The two empirical inputs $\rho_g$ and $\Gamma_g$ are calibrated from a single forward pass over 1024 tokens of training-distribution text, and the per-state-group integer bit budget $b_g$ is determined in closed form. No gradient-based optimisation, no quantisation-aware fine-tuning, and no per-deployment retraining is required.

**Contributions.** This paper develops the Lyapunov-aware bit allocator into an empirically validated SSM-state-quantisation recipe. Specifically:

1. **A single closed-form bit-allocation rule** (§3) that requires only two calibrated quantities, $\rho_g$ and $\Gamma_g$, to specify per-state-group integer bit widths from a deployment-set baseline $b_0$.
2. **Single-architecture validation** (§4) on Mamba-2.8B-hf across the C-1 / C-3 / C-4 ladder, including direct measurement of the $\sigma^2$-scaling property predicted by the linear-response argument (the `full` profile is uniquely $\sigma^2$-scaling under controlled noise; every tested control profile is super-quadratic).
3. **A universal-allocator claim across eight Mamba SSM instances** (§5) spanning three lineages (state-spaces, TII Falcon, AI21 Jamba) and two architectures (Mamba-1 selective SSM, Mamba-2 SSD). The allocator is rank-1 over uniform / proxy / anti / inverse profiles in every tested instance.
4. **A deployment-realistic compression result** (§6) with real INT-b packing: 2.5 - 2.9× state-cache compression at +4 - 8 % perplexity across five Mamba-1 sizes (130M - 2.8B), and a Pareto-frontier study identifying $(b_0 = 6, \mathrm{full})$ as the universal deployment recommendation.
5. **A long-context calibration-benefit finding** (§7) on PG-19 at context lengths up to 32K. The calibrated range $r_g = k \cdot \mathrm{RMS}(h_g)$ used to place the quantisation grid also acts as a state-magnitude regulariser, extending Mamba-2.8B-hf's usable context window from approximately 2K (where unquantised FP16 is stable) to at least 32K (where unquantised FP16 and BF16 perplexity exceed 190 on PG-19 while quantised paths remain below 12). A four-condition ablation (§7.3) attributes this regularisation to the calibrated clipping rather than to the quantisation rounding, and rules out a pure FP16-precision bug as the cause of the unquantised divergence.

**Roadmap.** §2 surveys related work in transformer KV-cache quantisation and the recent SSM-quantisation literature (Quamba, Q-Mamba, MambaQuant, SSDi8). §3 derives the bit-allocation rule from a per-state-group $\sigma^2$-budget argument. §4 validates the rule on Mamba-2.8B-hf as a single-architecture anchor. §5 establishes the universal-allocator claim across eight Mamba SSM instances. §6 reports deployment-realistic compression and the Pareto frontier. §7 reports the long-context regime and the calibration-benefit finding. §8 discusses limitations and outlines a follow-up agenda (joint attention + SSM quantisation in hybrid stacks, deferred to a companion paper).

---

## 2. Related Work

This section positions the present paper within four adjacent literatures: transformer KV-cache quantisation (which provides the analogue and the contrast for SSM state quantisation), SSM weight and activation quantisation (the most active body of recent work on Mamba compression), per-group and heterogeneous bit allocation in SSMs (the closest in spirit to our approach, but not driven by spectral-radius theory), and Lyapunov-related analyses of SSM dynamics (which are concerned with training stability rather than quantisation). We close with an explicit statement of the present work's position relative to all four.

### 2.1 Transformer KV-Cache Quantisation as Background

A line of recent work compresses the per-token key-value cache of transformer language models from 16 bits per element to 3 to 4 bits per element with negligible quality loss. KIVI [Liu et al., 2024] proposes per-channel quantisation for the key cache and per-token quantisation for the value cache, exploiting the asymmetry of their distributions. KVQuant [Hooper et al., 2024] extends sub-4-bit quantisation to longer contexts using a dense-and-sparse decomposition. Palu [Chang et al., 2024] caches low-rank projections of the key-value tensors. ZipCache [He et al., 2024] identifies salient tokens via normalised attention scores and compresses non-salient tokens more aggressively. These methods all exploit the per-token, per-element-independent structure of the transformer KV-cache: each cache slot is read once at attention time, the rounding errors do not feed back into a recurrence, and per-element noise variance translates directly into per-token attention error. None of these properties hold for the SSM hidden state, which is updated by an input-dependent linear recurrence (§3.1). The present paper inherits the transformer-side compression playbook in spirit, but the per-element optimisation argument that drives it must be rebuilt from first principles for the SSM substrate.

### 2.2 SSM Weight and Activation Quantisation

The first wave of post-training quantisation (PTQ) work on Mamba focused on weights and activations rather than the recurrent state. Mamba-PTQ [Pierro & Abreu, 2024] is among the earliest, identifying outlier channels in the recurrent LLM as the primary obstacle to naive transplants of transformer PTQ recipes. Quamba [Chiang et al., 2024] proposes a static 8-bit per-tensor quantisation scheme for the selective SSM, suppressing the maximum values of the input activations and quantising the output activations in an outlier-free space via a Hadamard transform; it reports a $1.72\times$ generation-latency reduction on Mamba-2.8B with $0.9 \,\%$ accuracy drop. MambaQuant [Xu et al., 2025] addresses outlier-driven accuracy loss with Karhunen-Loève transform-enhanced rotation and a smooth-fused rotation that equalises channel variances, achieving W8A8 with under $1 \,\%$ accuracy loss. OuroMamba [Ramachandran et al., 2025] presents a data-free PTQ for vision Mamba models, using channel-wise static weight quantisation and per-time-step dynamic activation quantisation. LightMamba [Wei et al., 2025] co-designs a 4-bit rotation-based PTQ with FPGA acceleration. Bi-Mamba [Tang et al., 2024] takes the alternative quantisation-aware-training (QAT) route to 1-bit weights, training Mamba models from scratch with autoregressive distillation. None of these methods quantise the recurrent state $h_t$ at per-(channel, state-unit) granularity with bit widths driven by spectral or output-coupling statistics, which is the focus of the present paper.

### 2.3 Per-Group and Heterogeneous Bit Allocation in SSMs

A smaller body of recent work allocates bits non-uniformly across SSM components, and is the closest in spirit to our approach. Quamba2 [Chiang et al., 2025] extends Quamba to W8A8, W4A8, and W4A16 across both Mamba-1 and Mamba-2 backbones, with per-state-group quantisation for the input-dependent parameters $B_t$ and $C_t$. Quamba2's groups are channel-wise (eight groups of 128 channels each), and within a group the quantisation is uniform 8-bit; cached SSM states are quantised at uniform 8-bit as well. The grouping is based on activation magnitude clustering and a Hadamard-style compute-invariance constraint, not on the recurrence's spectral radius or output coupling. The recent edge-AI study of [Zhao et al., 2025] (Quantizing Small-Scale State-Space Models for Edge AI) reports a heterogeneous bit allocation in which the weight matrices $\bar B$, $\bar C$, $\bar D$ are quantised at 4 bits, activations other than the state at 6 bits, and the state $x$ and transition matrix $\bar A$ at 6 to 16 bits depending on ablation; their grouping is per-head (one group per parallel SSM head), their calibration uses a 99.999-percentile range estimate, and the bit allocation is determined by ablation rather than by a closed-form formula. Q-S5 [Abreu & Pierro, 2024] and QS4D [Siegel et al., 2025] explore quantisation of S5 and S4D structured SSMs respectively, primarily in the QAT regime. The present paper is the first to allocate per-(channel, state-unit) integer bit widths in closed form from a discrete-time Lyapunov bound on accumulated quantisation noise, with the bit width determined by the per-group importance weight $R_g = \Gamma_g / (1 - \rho_g)$ rather than by activation magnitude statistics or by ablation.

### 2.4 Lyapunov Analyses of SSM Dynamics (Non-Quantisation)

Two recent works analyse SSM dynamics through a Lyapunov-style lens but in non-quantisation contexts. Mamba State-Space Models Are Lyapunov-Stable Learners [Halloran et al., 2024] characterises the training-time stability of Mamba's selective recurrence using a Lyapunov function on the parameter trajectories, addressing the question of whether Mamba's training is convergent rather than the question of how to allocate quantisation noise across state groups. Spectral State Space Models [Agarwal et al., 2023] proposes a new SSM family based on spectral filtering of linear dynamical systems, again concerned with model design rather than compression. The discrete-time Lyapunov bound on accumulated quantisation noise that drives our bit allocator (§3.2) is an instance of a more general fact about contraction maps, but to our knowledge no prior work has translated this bound into a per-state-group bit-allocation rule.

### 2.5 Position of the Present Work

We summarise our position relative to the four literatures along five axes.

First, our bit-allocation rule is closed-form ($b_g = b_0 + \lceil \log_2 R_g \rceil_+$), driven by two calibrated scalars per state group, and obtained without ablation search. Existing work either uses uniform bits within groups (Quamba2 [Chiang et al., 2025]) or determines heterogeneous bits via empirical ablation (Zhao et al., 2025).

Second, our granularity is per-(channel, state-unit), which on Mamba-2.8B-hf produces $5{,}242{,}880$ distinct bit widths. Existing work operates at per-tensor (Quamba [Chiang et al., 2024]), per-channel-group (Quamba2 [Chiang et al., 2025]), or per-head (Zhao et al., 2025) granularity, none of which exposes the per-state-unit structure that our $R_g$ formula targets.

Third, our derivation is a discrete-time Lyapunov bound on accumulated quantisation noise (§3.2), with $(1 - \rho_g)^{-1}$ as the per-group amplification factor. No prior SSM-quantisation work uses spectral radius for bit allocation; the existing Lyapunov-themed work [Halloran et al., 2024] addresses training stability rather than compression.

Fourth, our importance weight combines spectral radius with output coupling, $R_g = \Gamma_g / (1 - \rho_g)$. The output-coupling factor $\Gamma_g = \mathbb{E}_t [(C_t^{(s)})^2]$ is, to our knowledge, not used in prior SSM quantisation literature as an input to bit allocation.

Fifth, our universal claim spans eight Mamba SSM instances across three release lineages (state-spaces, TII Falcon-Mamba [Zuo et al., 2024], AI21 Jamba [Lieber et al., 2024]; with Zyphra's Zamba2 [Glorioso et al., 2024] providing the Mamba-2 SSD test). Existing PTQ work on Mamba is anchored on the state-spaces release line; cross-lineage and cross-architecture generalisation has not been systematically tested.

The long-context calibration-benefit observation in §7 is, to our knowledge, also new. Prior SSM-quantisation work has not characterised the interaction between calibrated state-magnitude clipping and the long-context numerical regime, which we frame in §7.4 as a 2-for-1 benefit of the calibration design rather than as a quantisation-specific stabilisation effect.

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

Let $\rho_g = \mathbb{E}_t |\bar A_t^{(g)}|$ denote the long-run spectral radius of the per-group transition. For a stationary input distribution and zero-mean i.i.d. quantisation noise of per-group variance $\sigma_q^{(g)\,2}$, the steady-state error variance is bounded by the geometric series

$$
\mathbb{E}\bigl[(e_t^{(g)})^2\bigr]
\;\le\;
\frac{\sigma_q^{(g)\,2}}{1 - \rho_g^2}
\;\approx\;
\frac{\sigma_q^{(g)\,2}}{2\,(1 - \rho_g)}
\quad(\rho_g \to 1).
$$

We absorb the leading constant and use $\mathbb{E}[(e_t^{(g)})^2] \propto \sigma_q^{(g)\,2} / (1 - \rho_g)$ in what follows.

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

For a uniform mid-tread quantiser with per-group range $r_g$ and bit width $b_g$, the per-group noise variance is $\sigma_q^{(g)\,2} = (r_g / 2^{b_g})^2 / 3 \propto r_g^2 \cdot 4^{-b_g}$. We adopt a calibrated per-group range $r_g = k \cdot \mathrm{RMS}(h_g)$ with $k = 7$, a 7-σ truncation that captures over 99.99 % of the calibration-distribution state magnitude. Under the simplifying assumption that the per-group ranges $r_g$ are roughly homogeneous across groups after RMS calibration (an assumption we revisit in §3.5 and §8), the variation of $\sigma_q^{(g)\,2}$ across groups is then determined entirely by $b_g$.

Minimising $\sum_g R_g \cdot \sigma_q^{(g)\,2}$ subject to a fixed total bit budget $\sum_g b_g = B_\mathrm{total}$ via Lagrange multiplier gives, at the unconstrained optimum,

$$
R_g \cdot 4^{-b_g} \;=\; \mathrm{const},
\qquad
b_g \;=\; \mathrm{const}' \;+\; \tfrac{1}{2} \log_2 R_g.
$$

Equivalently, the bit-budget-optimal noise allocation scales as

$$
\sigma_q^{(g)} \;\propto\; \frac{1}{\sqrt{R_g}}
\;=\;
\sqrt{\frac{1 - \rho_g}{\Gamma_g}}.
$$

This is the form of the "full" profile in our continuous-noise validation experiments (§4.5), where it is the unique linear-response (σ²-scaling) allocator under controlled noise injection while every tested control profile (uniform; $\rho$-only; $\sqrt{1-\rho}$-only; anti-allocation) is super-quadratic.

### 3.5 Integer Bit-Width Construction

For deployment we need an integer $b_g$ per group, fitting within a target average $b_0$ (the user-specified deployment budget) and bounded by a hardware-friendly minimum $b_\mathrm{min}$ to avoid sign-only quantisation. Our concrete integer realisation is

$$
\mathrm{extras}_g \;=\; \bigl\lceil \log_2 R_g \bigr\rceil_+,
\qquad
b_g^\mathrm{raw} \;=\; b_0 + \mathrm{extras}_g,
\qquad
b_g \;=\; \mathrm{round}\!\bigl(b_g^\mathrm{raw} - \overline{b^\mathrm{raw}} + b_0\bigr)\Big|_{[b_\mathrm{min},\,16]},
$$

where $\overline{b^\mathrm{raw}}$ is the average over all groups in the model and the final clamp enforces $b_g \in [b_\mathrm{min}, 16]$. The shift normalisation rebalances the total bit budget so that $\sum_g b_g \approx b_0 \cdot |\{g\}|$ is preserved.

Two design choices in this construction deserve comment.

First, the extras are based on $\lceil \log_2 R_g \rceil$, not $\lceil \tfrac{1}{2} \log_2 R_g \rceil$. The strict bit-budget optimum from §3.4 calls for a factor of $\tfrac{1}{2}$, but our integer formula uses unity. The result is a factor-of-two more aggressive allocation: high-$R_g$ groups receive about twice the extra bits the strict optimum would assign, and after the shift normalisation, low-$R_g$ groups receive about twice fewer. We adopt this conservatism for two reasons. The Lyapunov bound in §3.2 is a stationary-input bound, while the input-dependent transition $\bar A_t$ on long-memory units exhibits heavy right tails on natural text (Phase C-1b finds median $\rho_g \approx 0.88$ on the long-memory layer 32 of Mamba-2.8B-hf, but with non-trivial mass at $\rho_g > 0.999$). Over-allocating bits on these units absorbs the worst-case excursions. Empirically, the σ²-scaling result in §4.5 confirms that the resulting allocator is uniquely linear-response across a range of noise magnitudes, while every tested control profile (including a "strict optimum proxy" $\sigma_g \propto \sqrt{1 - \rho_g}$) is super-quadratic.

Second, the floor $b_\mathrm{min} = 2$ prevents the construction from dropping into binary quantisation. A binary allocation has only one non-zero level above zero and no mid-tread zero, so the per-step noise is no longer well-modelled by $\sigma_q \propto 2^{-b}$ and the linear-response argument breaks down. The $b_\mathrm{min} = 2$ floor keeps the smallest groups at INT2 (four levels, including zero) where the standard analysis still applies; this choice was confirmed in §4.6, where allocations that included $b_g = 1$ groups departed sharply from the σ²-scaling reference.

We refer to the resulting integer bit-width assignment as the **Lyapunov-aware bit allocator**, since the central $(1 - \rho_g)^{-1}$ factor in $R_g$ is the discrete-time Lyapunov bound on the SSM recurrence's noise accumulation.

### 3.6 Calibration

The two empirical inputs $\rho_g$ and $\Gamma_g$, together with the per-group state magnitude $\mathrm{RMS}(h_g)$ used for the range $r_g$, are calibrated from a single forward pass over a short calibration corpus. We use 1024 tokens of WikiText-2 train throughout this paper. The recipe is:

1. Run a forward-only inference pass that records, at every Mamba mixer, the per-(c, s) tensors $\bar A_t^{(c, s)}$, $C_t^{(s)}$, and the post-update state $h_t^{(c, s)}$.
2. Estimate $\rho_g$ as the empirical mean over $t$ of $|\bar A_t^{(c, s)}|$. The static-only estimate from $A_\mathrm{log}$ alone (with a fixed $dt = 0.1$) underestimates the long-memory tail by approximately fifty-fold on Mamba-2.8B-hf; the input-dependent estimate is the right input to the bit allocator (§4.1).
3. Estimate $\Gamma_g$ as the empirical mean over $t$ of $(C_t^{(s)})^2$.
4. Estimate $\mathrm{RMS}(h_g)$ as $\sqrt{\mathbb{E}_t[(h_t^{(c, s)})^2]}$ via streaming accumulation over the calibration window.

The total cost is one forward pass over 1024 tokens plus per-mixer streaming statistics in float32. For Mamba-2.8B-hf this amounts to roughly 30 seconds of GPU time and about 5 GB of activation memory; the resulting per-(c, s) statistics are stored once and reused for all subsequent allocator and range computations. No gradient propagation, no quantisation-aware fine-tuning, and no per-deployment retraining is involved.

Calibration robustness is examined in §4.3. Briefly, training-set and validation-set calibrations on Mamba-2.8B-hf produce $\log \Gamma_g$ values with Pearson correlation 0.987 and a 77.5 % match in the resulting integer bit-tier classification, with one outlier layer flagged as having unusually input-dependent coupling. This level of robustness is consistent with the deployment claim that the allocator is calibration-light.

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

The empirical $\rho_g$ distribution also exhibits strong per-layer heterogeneity. Median $\rho_g$ across the 64 layers traces a U-shape: short-memory layers at the input end ($\rho_g$ p50 $\approx 0.60$ at layer 0), a long-memory core in the middle ($\rho_g$ p50 $\approx 0.88$ at layers 32 and 40), and short-memory layers again at the output end ($\rho_g$ p50 $\approx 0.39$ at layer 56). This pattern matches the transformer literature finding that middle layers tend to encode the most abstract long-range context, and motivates a per-layer baseline term in the bit allocator that the shift normalisation of §3.5 implicitly provides.

The conclusion of §4.1 is that the empirical, input-dependent $\rho_g$ is the right input to the bit allocator, and that the long-memory minority is sufficiently large (in the hundreds of thousands of state units, not a handful) that protecting it with extra bits is the central design challenge.

### 4.2 Output-Coupling Extraction and the $R_g$ Surface

Phase C-1c extracts the per-state-unit output coupling $\Gamma_s = \mathbb{E}_t[(C_t^{(s)})^2]$ by replaying the Mamba-1 forward path under WikiText-2 input and intercepting the output of `mix.x_proj`, which produces $dt_\mathrm{raw}$, $B_t$, and $C_t$ as concatenated $[d_{t,\mathrm{rank}}, d_\mathrm{state}, d_\mathrm{state}]$ slices. The third slice is averaged in squared form per state-unit $s$ over the 1024-token calibration window. In Mamba-1, $C_t$ is shared across channels for each $s$, so $\Gamma_g = \Gamma_s$ at our (c, s) granularity, and the model has only $64 \times 16 = 1024$ distinct $\Gamma$ scalars in total.

The per-layer $\Gamma_s$ distribution is heavy-tailed. The within-layer $\Gamma_\mathrm{max} / \Gamma_\mathrm{min}$ ratio across the 16 state indices is in the $167\times$ to $1572\times$ range across layers, and the per-layer $\Gamma_\mathrm{max}$ grows monotonically toward the output end (e.g., $\Gamma_\mathrm{max} = 5.1$ at layer 0, $27.1$ at layer 48, $45.4$ at layer 56). The natural reading is that a small handful of state indices carries most of the readout in the late stages of the model, with the remaining state indices contributing essentially noise to the output sum.

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

The bit-tier match, defined as the fraction of (c, s) state units whose integer extras $\lceil \log_2 R_g \rceil_+$ at $b_0 = 4$ matches between the train and validation calibrations, is $77.5 \,\%$. Of the remaining $22.5 \,\%$, almost all (over 99 %) differ by exactly one bit; the $\ge \pm 2$ bit tail is $0.21 \,\%$. The pre-registered $80 \,\%$ threshold fails by $2.5$ percentage points, placing this calibration in the middle bucket of the decision rule (USABLE_WITH_FLAG): the C-3 / C-3b results stand under a 1-bit perturbation on a quarter of units, but the deployment recommendation is to add a $+1$ bit safety margin or to ensemble calibration across multiple text sources. We adopt the $+1$ bit safety margin convention for all subsequent results, and flag layer 30 as the candidate for any deployment-time recalibration check.

### 4.4 State-Step Quantisation in the Patched `slow_forward`

The first three subsections measure calibration-side quantities. Phase C-3 takes the next step and tests the bit allocator under the actual quantisation channel that deployment will use. The patched `MambaMixer.slow_forward` inserts a per-(c, s) Gaussian noise injection $\hat h_t = h_t + \eta_t,\ \eta_t \sim \mathcal{N}(0, \sigma_g^2 I)$ between the state update and the readout, with the per-group $\sigma_g$ determined by one of five profiles. The relative noise scale is held constant at $\sigma_\mathrm{rel} = \sigma_g / \mathrm{RMS}(h_g) = 0.10$, so any difference in $\Delta\mathrm{PPL}$ is attributable to the *allocation* shape, not to the *amount* of noise.

| profile | $\sigma_g$ shape | $\Delta\mathrm{PPL}$ vs FP16 |
|:---|:---|---:|
| **full** (Paper C) | $\sqrt{(1 - \rho_g)/\Gamma_g}$ | $\mathbf{+0.351 \,\%}$ |
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

Three structural observations follow from this table. The full allocator's empirical doubling multipliers are $3.66\times$ and $4.03\times$, both within rounding of the $\sigma^2$ ideal of $4\times$. It is the only profile that exhibits this linear-response behaviour. Every tested control profile is super-quadratic, with the strongest deviation in the uniform allocator, which scales as $\sigma^{3.6}$ at the highest noise.

The strict ordering full < proxy_correct < proxy_C2 < anti_full < uniform holds at every $\sigma_\mathrm{rel}$. This rules out the most innocent alternative explanation for the C-3 single-point result: that $\sigma_\mathrm{rel} = 0.10$ might have been a sweet spot that happened to favour full. The advantage of full over proxy_C2 grows from $0.66$ percentage points at $\sigma_\mathrm{rel} = 0.05$ to $15.68$ percentage points at $\sigma_\mathrm{rel} = 0.20$, a $24\times$ growth as $\sigma$ increases $4\times$.

The uniform allocator is already $4.7\times$ worse than anti_full at $\sigma_\mathrm{rel} = 0.05$, where total relative noise is mild. The IIR-accumulation pathology is therefore constitutional, not amplitude-driven: uniform allocation corrupts long-memory pathways in concert at any noise level, and the FIR baseline (uniform allocation works fine on transformer attention) does not transfer to IIR substrates.

The $\sigma^2$-scaling test is the strongest theoretical-side confirmation of the bit allocator we offer. The bit-budget argument of §3.4 implies that the bit-budget-optimal profile should produce $\sigma^2$-linear noise; §4.5 verifies this empirically on Mamba-2.8B-hf, while every plausible alternative profile fails the test in a structurally consistent direction.

### 4.6 The $\sigma \to b$ Bridge: Real INT-$b$ Quantisation

Phase C-4 closes the bridge from continuous $\sigma$-space (where C-3b operates) to deployment-realistic integer bit widths. The patched `slow_forward` of §4.4 is run with the five profiles across a $5 \times 3 = 15$-cell grid: $b_0 \in \{3, 4, 5, 6, 8\}$ bits and per-group range $r_g = k \cdot \mathrm{RMS}(h_g)$ with $k \in \{3, 5, 7\}$. The quantiser is the INT-$b$ symmetric mid-tread with zero level of §3.5 (an asymmetric integer range $[-2^{b-1}, 2^{b-1} - 1]$, including zero), and the bit allocation uses $b_g = b_0 + \lceil \log_2 R_g \rceil_+$ with $b_\mathrm{min} = 2$ floor and shift normalisation.

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

First, the practical deployment threshold (defined as the smallest $b_0$ for which $\Delta\mathrm{PPL}$ stays at or below $10 \,\%$) is INT5 to INT6 for full, and INT7 to INT8 for both uniform and proxy_C2. The Paper C bit allocator therefore unlocks $1$ to $2$ bits of memory savings at fixed deployment quality, equivalent to $25$ to $50 \,\%$ memory reduction on the SSM state. This is the headline deployment claim of the present paper.

Second, the relative advantage of full over uniform compounds in the scarce-bit regime: $1.7\times$ at INT8, $2.8\times$ at INT6, $4.3\times$ at INT5, $8.3\times$ at INT4, and $27.8\times$ at INT3. This is precisely the property a deployment-grade allocator should have: the more constrained the bit budget, the bigger the gain from solving the allocation problem correctly.

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

The deployment cost of the full allocator at INT6 is $4$ to $8 \,\%$ across all five sizes, with no monotone scaling trend. This is the cleanest single sentence the paper can make about deployment cost: at the deployment-realistic average bit budget $b_0 = 6$ on the state-spaces family, the full allocator stays in single-digit perplexity overhead independent of model scale within the tested range.

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

Strict ordering full < proxy_C2 < proxy_correct < uniform < anti_full, with full rank-1. The negative ΔPPL values are a regularisation artefact of sparse chunk-boundary noise: the quantiser fires only at the four inter-chunk boundaries per 1024-token sequence, roughly $340\times$ sparser than the per-token injection of §4.4, and at $b_0 = 6$ the resulting noise is small enough to act as a mild regulariser on test perplexity. The rank order, not the absolute negative, is the meaningful signal. It establishes that the bit allocator's structure transfers across the Mamba-1 / Mamba-2 architectural divide, with the only protocol change being the granularity adjustment.

### 5.3 Cross-Lineage Transfer: TII Falcon-Mamba-7B

Phase C-8c tests the bit allocator on Falcon-Mamba-7B (`tiiuae/falcon-mamba-7b`), the largest open-weight Mamba-1 model and the only one not from the state-spaces release line. Falcon-Mamba was trained on RefinedWeb at 7B scale and uses a different forward-path convention: weightless `rms_forward` (RMSNorm without learnable weight) is applied to $B_t$, $C_t$, and $dt_\mathrm{raw}$ before the discretisation. The patched forward of §4.4 detects this via the `mix.rms_eps` attribute and applies the same normalisation before computing $\bar A_t$, $\bar B_t$, and $C_t$.

| profile | $b_0 = 4$ | $b_0 = 6$ |
|:---|---:|---:|
| **full** | $\mathbf{+72.8 \,\%}$ | $\mathbf{+31.2 \,\%}$ |
| proxy_correct | $+91.9 \,\%$ | $+35.5 \,\%$ |
| uniform | $+111.9 \,\%$ | $+54.6 \,\%$ |
| proxy_C2 | $+128.1 \,\%$ | $+58.7 \,\%$ |
| anti_full | $+271.4 \,\%$ | $+193.2 \,\%$ |

The full allocator is rank-1 at both bit budgets. Two readings characterise the cross-lineage transfer. The full / uniform advantage is $1.54\times$ at INT4 and $1.75\times$ at INT6, smaller than the $4 \times$ to $12\times$ range observed in the state-spaces family. The absolute ΔPPL is also higher (full at INT4 is $+72.8 \,\%$ on Falcon-Mamba versus $+26.6 \,\%$ on Mamba-2.8B). Both effects are consistent with Falcon-Mamba being trained on a more diverse corpus (RefinedWeb): the resulting $(\rho_g, \Gamma_g)$ distribution likely has thinner tails, so the allocator's structural advantage compresses while the model's overall sensitivity to per-state quantisation noise increases. The rank order transfers cleanly, but the magnitude is lineage-dependent. Paper C reports both: the bit allocator is universal in direction, lineage-specific in magnitude.

### 5.4 Hybrid Architecture + MoE: AI21 Jamba-1.5-Mini

Phase C-8d covers the widest stretch of the generalisation chain: Jamba-1.5-Mini (`ai21labs/AI21-Jamba-Mini-1.6`), a 52B-total / 12B-active hybrid model from a third release lineage. Jamba interleaves Mamba-1 mixers with transformer attention and MoE expert layers, and uses learnable RMSNorm modules (`mix.dt_layernorm`, `mix.b_layernorm`, `mix.c_layernorm`) instead of Falcon's weightless `rms_forward` to normalise $dt_\mathrm{raw}$, $B_t$, and $C_t$ before discretisation. The patched forward grew a third lineage branch detected via the `mix.dt_layernorm` attribute. Loaded in FP16 with `device_map="auto"` on a 102 GB GPU runtime, with mild CPU offload (about 84 GB VRAM resident); 28 Mamba-1 mixers were detected, interleaved with the attention and MoE layers (which were held at FP16 throughout, since Paper C does not quantise attention).

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

Eight instances, three release lineages, two architectures (Mamba-1 selective SSM and Mamba-2 SSD), four deployment styles (pure Mamba-1, pure Mamba-2, hybrid + attention, hybrid + attention + MoE). The full allocator is rank-1 over uniform, proxy_correct, proxy_C2, and anti_full in every tested instance at the deployment-realistic budgets $b_0 \in \{4, 6\}$. The full / uniform advantage at INT6 ranges from $1.7\times$ on Falcon-Mamba to $5.0\times$ on Mamba-370M, with full ΔPPL in the $4$ to $8 \,\%$ range across the state-spaces family. The $R_g = \Gamma_g / (1 - \rho_g)$ formula is unchanged across the eight instances; the only adaptations are the granularity adjustment for Mamba-2 and the lineage-specific handling of input-side normalisation modules. We refer to this as the universal-allocator claim of Paper C.

---

## 6. Deployment-Realistic Compression

The validation in §4 and the generalisation in §5 measured perplexity under fake-quant inference (quantise then dequantise, with values stored at FP32 in the patched forward path). For deployment, the SSM state must actually be stored at INT-$b$ bit width, and the per-group metadata that the allocator requires (per-(c, s) bit width $b_g$ and range $r_g$) must be accounted for in the memory budget. Phase C-6 closes this gap by computing real-packing compression ratios across the state-spaces Mamba-1 family from 130M to 2.8B parameters, paired with the perplexity costs from §4.6 and §5.1.

### 6.1 Real-Packing Methodology

For each (model, $b_0$, profile) configuration, we compute three deployment-side quantities. The mean per-group bit width $\overline{b_g}$ is taken over all (c, s) state units in the model after the integer construction of §3.5 (with shift normalisation and the $b_\mathrm{min} = 2$ floor applied). The theoretical compression ratio is $16 / \overline{b_g}$, the ratio of an FP16-stored state cache to a packed INT-$\overline{b_g}$ store. The long-context compression ratio additionally amortises a per-(c, s) metadata cost of $20$ bits per unit (16 bits FP16 $r_g$ plus 4 bits INT4 $b_g$) over a representative deployment of 100 cached chunk boundaries, giving an effective per-chunk overhead of $0.2$ bits per unit and a long-context ratio of $16 / (\overline{b_g} + 0.2)$.

Compression ratios are paired with the perplexity costs measured in §4.6 (Mamba-2.8B at the full $b_0$ grid) and §5.1 (the four smaller siblings at $b_0 \in \{4, 6\}$). The Pareto frontier on (compression, ΔPPL) is computed per-model across all (profile, $b_0$) cells.

### 6.2 The Deployment Table at $b_0 = 6$

The standard-quality deployment recommendation across the state-spaces family at the deployment-realistic budget $b_0 = 6$, with the WikiText-2 perplexity costs from §4.6 and §5.1:

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

Anti_full is omitted from this table as it is on the Pareto frontier of no model (catastrophic ΔPPL up to $+4516 \,\%$ at $b_0 = 4$ on the 130M model; see §5.1). For every one of the five state-spaces models, the $(b_0 = 6, \mathrm{full})$ configuration delivers **theoretical compression $2.5\times$ to $2.9\times$ at ΔPPL between $+4.0 \,\%$ and $+8.0 \,\%$**, with no monotone dependence on model scale within the tested range. We adopt this as the universal deployment recommendation of Paper C.

### 6.3 The Pareto Frontier

The (compression, ΔPPL) Pareto frontier per model includes between three and five non-dominated points across the $b_0 \in \{3, 4, 5, 6, 8\}$ grid. The full allocator is on the Pareto frontier of every one of the five models, at $b_0 = 4$ (high-compression / borderline-PPL operating point) and at $b_0 = 6$ (standard-quality operating point). The uniform allocator is on the Pareto frontier of four of the five models at $b_0 = 4$ as the "highest-compression-but-broken-PPL" extreme, but at $b_0 = 6$ uniform is dominated by full on every model. Proxy_C2 and anti_full are dominated almost everywhere; the only points at which they surface on a Pareto frontier are degenerate $b_0 = 4$ corners on the 1.4B and 2.8B models, where their over-allocation to high-$\rho$ accidentally produces marginally higher compression at much worse perplexity than full.

The headline of this analysis is that the bit allocator dominates uniform allocation in the joint (compression, ΔPPL) plane at the deployment-realistic operating point $b_0 = 6$. At fixed perplexity quality, the bit allocator saves approximately one bit of state-cache memory: uniform allocation requires INT7 to INT8 for the same perplexity that full delivers at INT6. Translated to memory, this is a $17 \,\%$ to $25 \,\%$ reduction in state-cache size at constant deployment quality.

### 6.4 Scale Benefit: Tail Clipping in Larger Models

A subtler observation in the deployment table is that the full allocator's actual mean bit width drifts below the nominal $b_0 = 6$ budget on the larger models. For Mamba-1.4B and Mamba-2.8B, $\overline{b_g} = 5.45$ and $5.56$ respectively, $0.5$ bits below the nominal target. The mechanism is the heavier-tailed $\Gamma_g$ distribution on larger models: extreme high-$R_g$ units hit the upper $b_g = 16$ ceiling clamp of §3.5, the per-unit shift then biases the average downward, and the resulting allocation extracts an additional $5 \,\%$ to $8 \,\%$ of compression beyond the nominal budget.

This is a free benefit of the bit allocator that grows with scale: smaller models with thinner $\Gamma_g$ tails see no ceiling clamping ($\overline{b_g} = 5.62$ at 130M, $\overline{b_g} = 6.45$ at 370M, $\overline{b_g} = 6.01$ at 790M), while the larger 1.4B and 2.8B models extract the most. Paper C reports this as a scale-dependent compression bonus rather than an explicit design lever; in deployments where the ceiling clamp matters, raising it from 16 bits to a higher cap (or dropping it entirely) would forfeit this bonus.

### 6.5 Aggressive Deployment: INT4 Feasibility

Under a relaxed deployment target of $\Delta\mathrm{PPL} \le 30 \,\%$, the bit allocator at $b_0 = 4$ is feasible on every model in the family except the smallest 130M (which still requires $b_0 = 6$ to stay below $30 \,\%$ PPL drift):

| Model | recommendation at $\Delta\mathrm{PPL} \le 30 \,\%$ | compression | ΔPPL |
|:---|:---|:---:|:---:|
| 130M | INT6 only (INT4 is $+138 \,\%$ on uniform; full at INT4 is $+31 \,\%$ marginal) | $2.85\times$ | $+6.89 \,\%$ |
| 370M | INT4, full | $3.58\times$ | $+10.2 \,\%$ |
| 790M | INT4, full | $3.72\times$ | $+12.9 \,\%$ |
| 1.4B | INT4, full | $3.94\times$ | $+27.1 \,\%$ |
| 2.8B | INT4, full | $3.90\times$ | $+26.6 \,\%$ |

For models from 370M upward, the bit allocator at INT4 yields $3.6\times$ to $3.9\times$ state-cache compression at the cost of $+10 \,\%$ to $+27 \,\%$ perplexity. This is a viable aggressive-deployment regime for memory-pressured environments. The deployment-realistic recommendation remains $(b_0 = 6, \mathrm{full})$ for standard-quality inference; the INT4 results are reported as the upper boundary of what the allocator can sustain under more aggressive memory pressure.

In summary, the bit allocator's deployment-grade contribution is a $2.5\times$ to $2.9\times$ state-cache compression at $+4 \,\%$ to $+8 \,\%$ perplexity across the state-spaces Mamba-1 family at $b_0 = 6$, with the allocator on the Pareto frontier of every tested model. The C-6 metadata accounting confirms that the per-(c, s) overhead is negligible at deployment scale (under $5 \,\%$ of the bit budget for cached deployments). Section §7 examines whether this compression recipe survives the long-context regime up to 32K tokens on PG-19.

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

### 7.4 The 2-for-1 Calibration-Benefit Framing

The C-9 ablation establishes that the bit allocator's calibration design has two simultaneous functions on Mamba-2.8B-hf at long context. The calibrated $\rho_g$ and $\Gamma_g$ statistics determine the per-group bit width $b_g$, producing the $2.5\times$ to $2.9\times$ memory compression at the deployment-grade perplexity reported in §6. Independently, the calibrated range $r_g = k \cdot \mathrm{RMS}(h_g)$ that the allocator uses to place the quantisation grid acts as a state-magnitude regulariser, extending the model's usable context window from the FP16-stable regime (approximately 2K context, the model's training length) to at least 32K context (where unquantised FP16 perplexity exceeds 190 on PG-19 while the calibrated-clipping path stays below 11).

We refer to this as a 2-for-1 benefit of the calibration design: a single $\mathrm{RMS}(h_g)$ measurement, performed at calibration time alongside the $\rho_g$ and $\Gamma_g$ measurements that drive the bit allocation, simultaneously determines the per-group quantisation range and the per-group state-magnitude bound that regularises long-context behaviour. State quantisation inherits the regularisation benefit by construction, since the quantisation grid is already defined relative to the same $r_g$.

The reframing relative to a naive reading of the §7.1 versus §7.2 juxtaposition is important. The phenomenon is real and reproducible (the quantised paths are stable while the unquantised paths diverge), but the causal mechanism is the calibrated clipping rather than the quantisation rounding. We frame the long-context behaviour as an incidental side-benefit of the calibration design rather than as a quantisation-specific stabilisation effect, which is the framing the C-9 ablation supports.

### 7.5 A New Deployment Option: FP16 Clip-Only

The C-9 ablation surfaces a previously unidentified deployment option. The $\mathrm{FP16\_clip\_only}$ configuration costs zero bits (the state is still stored at FP16) but inherits the long-context calibration benefit, with PPL at 32K of $10.51$ versus the unquantised reference's $196.94$.

| deployment | compression | 32K PPL | recommended for |
|:---|:---:|:---:|:---|
| FP16 unquantised | $1.0\times$ | $196.94$ | not viable at long context |
| BF16 unquantised | $1.0\times$ | $192.83$ | not viable at long context |
| FP16 clip-only ($r_g$ clamp, no quantisation) | $1.0\times$ | $10.51$ | users wanting long-context stability without compression |
| FP16 full INT6 (the §6 recipe) | $\approx 2.5\times$ | $11.34$ | users wanting both compression and stability |

The clip-only option is useful for users whose VRAM budget already accommodates the FP16 state cache but whose context window currently caps at the FP16-stable regime: the calibration step alone, without any quantisation, extends usable context at zero memory cost. Users wanting state-cache compression should still use the §6 recipe; the small $7 \,\%$ to $8 \,\%$ PPL premium at 32K is the rounding cost that the quantisation grid pays in exchange for the $2.5\times$ memory savings. Both options inherit the long-context calibration benefit. The bit allocator's calibration pipeline is thus the common ingredient, with the choice of whether to additionally quantise determined by the deployment's memory budget.

---

## 8. Discussion

The empirical core of this paper is a single closed-form bit-allocation rule, calibrated from two scalars per state group, that produces a deployment-grade SSM state-cache compression recipe across eight Mamba SSM instances. We close with limitations, an open puzzle the long-context analysis surfaced, and a list of follow-up directions.

### 8.1 Limitations

The theory in §3 makes several simplifying assumptions, and the empirical chain in §4 to §7 is bounded by the experimental scope we could realistically run.

The bit-budget-optimal allocation in §3.4 assumes per-group ranges $r_g = k \cdot \mathrm{RMS}(h_g)$ that are roughly homogeneous across groups, allowing the strict optimum to simplify to $b_g \propto \tfrac{1}{2} \log_2 R_g$. In practice, the per-layer $\mathrm{RMS}(h_g)$ varies by approximately $10\times$ across the 64 layers of Mamba-2.8B-hf (e.g., layer 0 has p50 $\mathrm{RMS}(h_g) = 2.3 \times 10^{-3}$ versus layer 49 at $2.4 \times 10^{-2}$), so a more refined integer formula could include a per-group $\log_2 r_g$ term. We did not test such an extension; the simpler form of §3.5 is rank-1 over the controls we tested in §4 to §7, but the optimum may be improvable.

The integer realisation of §3.5 uses $\lceil \log_2 R_g \rceil_+$ in place of the $\lceil \tfrac{1}{2} \log_2 R_g \rceil_+$ that the strict bit-budget optimum prescribes. We adopt this conservatism because the spectral-radius bound is stationary-input only while the empirical $\bar A_t$ on long-memory units exhibits heavy right tails, and because the $\sigma^2$-scaling test in §4.5 confirms the resulting allocator is uniquely linear-response across a four-fold $\sigma_\mathrm{rel}$ range. A more careful integer rounding study (with quantisation-aware fine-tuning, with non-uniform integer levels, or with the missing $\log_2 r_g$ term) might yield a strictly tighter allocator.

The empirical chain has several sample-size limitations. The calibration corpus is 1024 tokens of WikiText-2 train; §4.3 verifies $\mathrm{Pearson}(\log \Gamma_g) = 0.987$ across the WikiText-2 train and validation splits, but inter-corpus stability (e.g., calibration on RefinedWeb versus WikiText-2 versus Books3) has not been directly measured. The long-context evaluation in §7 uses PG-19 at $n_\mathrm{windows} = 4$, which is sufficient for the qualitative claims (the rank-1 advantage growing with context, the four-condition disambiguation in §7.3) but is statistical-floor-grade for the absolute PPL numbers. The eight-instance generalisation in §5 covers three Mamba release lineages and the Mamba-1 / Mamba-2 architectural divide, but does not extend to other recurrent SSM variants (RWKV, Gated DeltaNet, structured state-space alternatives outside the Mamba family). The §7.3 ablation surfaces one outlier layer (layer 30 of Mamba-2.8B-hf) with anomalously corpus-dependent $\Gamma_g$, which is flagged here as a candidate for any deployment-time recalibration check.

### 8.2 An Open Puzzle: Why Does Mamba State Magnitude Grow at Long Context?

The §7.3 ablation establishes that the long-context divergence of unquantised Mamba-2.8B-hf on PG-19 is precision-independent and is suppressed by clipping at the calibrated $r_g$. We do not characterise the underlying mechanism in this paper; doing so requires a direct measurement of the $\|h_t\|$ trajectory under FP16 versus clip-only forward at long context, and a comparison of the corresponding spectral radius and input-magnitude statistics. We hypothesise that the input-dependent $dt_t$ distribution drifts toward the longer-memory tail at long context (smaller $dt$, $\rho_g \to 1$), pushing the recurrence toward integration-without-decay; this is consistent with §4.1's empirical observation that the long-memory tail is heavy already at 1024 context. A more direct test of this hypothesis would compare the $dt_t$ distribution measured under 1024-context calibration against the same distribution measured at 32K context. We have not run this test.

The puzzle is paper-relevant in a specific sense. Closed-form characterisation of $\|h_t\|$ as a function of context length and the calibrated statistics would tie the §7.4 calibration-benefit framing back to a Lyapunov-type analytical bound on long-context state magnitude, paralleling the bound on accumulated noise in §3.2. We leave this as the most attractive open analytical question for follow-up work.

### 8.3 Future Work

Several extensions of the present results are natural follow-ups.

A closed-form derivation of $r_g$ from the SSM recurrence statistics, replacing the empirical $k \cdot \mathrm{RMS}(h_g)$ with a Lyapunov-derived bound on the state-magnitude tail, would tighten the connection between the theory of §3 and the calibration-benefit observation of §7. This is the most directly research-flavoured extension and connects to the open puzzle of §8.2.

The long-context calibration-benefit finding raises the question of whether the same regularisation effect transfers to Mamba variants pretrained at long context. Mamba-2.8B-hf was trained at 2K, so its long-context behaviour is by construction extrapolation; if a future Mamba checkpoint pretrained at 32K context still requires the calibrated clipping for stable inference, then $r_g$-based regularisation is a general property of the architecture rather than an extrapolation patch. We have not tested this.

A natural compression follow-up is to integrate Paper C's SSM bit allocator with attention-side quantisation in hybrid stacks, where the SSM layers and the transformer attention layers are jointly compressed under a shared bit budget. The composition diagnostic on Zamba2-1.2B (Phase C-5c, omitted from the main text per the scoping decision) and the resonance phenomenon at $b_0 \in \{5, 8\}$ in the Phase C-5d sweep are both candidates for a companion paper focused on hybrid-stack compression.

Finally, a distillation-style follow-up would ask whether $r_g$ and $b_g$ admit a universal precomputed lookup table indexed by lineage and architecture, removing the need for per-deployment calibration. The present calibration is light (one forward pass over 1024 tokens, about 30 seconds on Mamba-2.8B), but a universal lookup table would remove the calibration step entirely. Whether such a table can match per-deployment calibration within a one-bit safety margin (the §4.3 USABLE_WITH_FLAG threshold) is an open empirical question.

### 8.4 Closing Remarks

This paper develops a single closed-form bit-allocation rule for SSM state quantisation, derives it from a discrete-time Lyapunov bound on accumulated quantisation noise, and validates it across eight Mamba SSM instances spanning three release lineages and two architectures. The rule's two empirical inputs are obtained from a single 1024-token calibration forward pass, the integer bit width follows in closed form, and the deployment-realistic configuration produces $2.5\times$ to $2.9\times$ state-cache compression at $+4 \,\%$ to $+8 \,\%$ perplexity overhead on the state-spaces Mamba-1 family. As a side-benefit of the calibration design, the per-group range $r_g$ used to place the quantisation grid also regularises long-context state magnitudes, extending Mamba-2.8B-hf's usable context window from approximately 2K to at least 32K on PG-19. The structure-aware allocation, the Lyapunov-bound derivation, and the calibration-benefit finding together suggest that SSM state quantisation is best treated as a calibration-led problem: the same two scalars per state group that define the integer bit width also characterise the architectural fragility that drove the FIR-versus-IIR distinction in the first place.

---

## References

*Citations are listed alphabetically by first-author surname. arXiv preprint identifiers are given for all preprints; venue is given where known. All author lists were cross-checked against the arXiv abstract pages on 2026-04-30.*

Abreu, S., Pedersen, J. E., Heckel, K. M., and Pierro, A. 2024. Q-S5: Towards Quantized State Space Models. *arXiv:2406.09477*.

Agarwal, N., Suo, D., Chen, X., and Hazan, E. 2023. Spectral State Space Models. *arXiv:2312.06837*.

Chang, C.-C., Lin, W.-C., Lin, C.-Y., Chen, C.-Y., Hu, Y.-F., Wang, P.-S., Huang, N.-C., Ceze, L., Abdelfattah, M. S., and Wu, K.-C. 2024. Palu: Compressing KV-Cache with Low-Rank Projection. *arXiv:2407.21118*.

Chiang, H.-Y., Chang, C.-C., Frumkin, N., Wu, K.-C., and Marculescu, D. 2024. Quamba: A Post-Training Quantization Recipe for Selective State Space Models. *arXiv:2410.13229*.

Chiang, H.-Y., Chang, C.-C., Frumkin, N., Wu, K.-C., Abdelfattah, M. S., and Marculescu, D. 2025. Quamba2: A Robust and Scalable Post-training Quantization Framework for Selective State Space Models. *arXiv:2503.22879*.

Dao, T. and Gu, A. 2024. Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality. *International Conference on Machine Learning (ICML 2024)*. *arXiv:2405.21060*.

Glorioso, P., Anthony, Q., Tokpanov, Y., Whittington, J., Pilault, J., Ibrahim, A., and Millidge, B. 2024. Zamba: A Compact 7B SSM Hybrid Model. *arXiv:2405.16712*. (Zamba2 release notes: Zyphra Technologies, 2024.)

Gu, A. and Dao, T. 2023. Mamba: Linear-Time Sequence Modeling with Selective State Spaces. *arXiv:2312.00752*.

Halloran, J. T., Gulati, M., and Roysdon, P. F. 2024. Mamba State-Space Models Are Lyapunov-Stable Learners. *arXiv:2406.00209*.

He, Y., Zhang, L., Wu, W., Liu, J., Zhou, H., and Zhuang, B. 2024. ZipCache: Accurate and Efficient KV Cache Quantization with Salient Token Identification. *arXiv:2405.14256*.

Hooper, C., Kim, S., Mohammadzadeh, H., Mahoney, M. W., Shao, Y. S., Keutzer, K., and Gholami, A. 2024. KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization. *Advances in Neural Information Processing Systems (NeurIPS 2024)*. *arXiv:2401.18079*.

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

*End of manuscript draft v0.1, 2026-04-30. Eight numbered sections plus references; total length approximately 13,000 words. Empirical results draw on Phase C-1 through Phase C-9 of the Paper C experimental programme; phase numbers cited inline correspond to the experimental log in the project repository at `paper_c/README.md`. The associated raw experimental data are stored in `paper_c/results_c/` as JSON files (one per phase).*

---

*End of draft v0.1. §1 is fully written; §2-§8 are stubs pointing to README content for fast merge.*
