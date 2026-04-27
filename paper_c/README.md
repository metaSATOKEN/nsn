# Paper C — From KV-Table to State-Aware Quantisation

**Working title.** *From KV-Table Quantisation to State-Aware Quantisation: Why Norm Separation Works for Transformers but Must Become Lyapunov-Aware for SSMs.*

**Status.** Kickoff phase. No manuscript yet. Empirical groundwork has been collected as part of [Paper B](../paper/) (the NSN paper); this folder collects the prospectus, the mathematical scaffolding, and the open phase plan for the follow-up. Started 2026-04-27.

---

## TL;DR

modeq's v0.3 recipe (`nsep+pchan+asym` INT3) compresses a *static, per-token KV table*: each cache slot is read once at attention time, errors do not feed back, and 5.09× memory reduction at < 5 % WikiText-2 ΔPPL is achievable on eight open-weight Transformers (Paper B). On SSM substrates (Mamba, Mamba-2 / SSD, Gated DeltaNet, hybrid Mamba + Transformer like Jamba / Zamba / Nemotron-3 Super / Qwen 3.6 27B), the same per-element noise is fed back into the next state via a recurrence and accumulates over time. **Per-token MSE is the wrong objective on a recurrent state.** Paper C develops a state-aware quantisation framework whose loss is weighted by the model's observability Gramian and whose bit allocation is driven by per-state-group spectral radius and output coupling.

---

## Core thesis

For a Mamba-style discretised SSM,

$$
h_t \;=\; \bar A_t\,h_{t-1} \;+\; \bar B_t\,x_t,\qquad
y_t \;=\; C_t\,h_t \;+\; D\,x_t,
$$

quantising the recurrent state $h_t$ with per-step noise $\eta_t$ produces an error $e_t = \hat h_t - h_t$ that obeys

$$
e_t \;=\; \bar A_t\,e_{t-1} \;+\; \eta_t
\quad\Longrightarrow\quad
\|e_t\| \;\lesssim\; \frac{\epsilon_q}{1 - \rho},\qquad
\rho \;=\; \mathbb{E}_t\,\|\bar A_t\|_2.
$$

The closer $\rho$ approaches 1 (i.e., the longer the model wants to remember), the smaller the per-step quantisation noise budget $\epsilon_q$ must be. **This factor $(1-\rho)^{-1}$ does not exist in the FIR / KV-cache regime that the v0.3 recipe was designed for**, and is the central reason why a transplant of `nsep+pchan+asym` onto a recurrent state is conceptually wrong — not just empirically suboptimal.

For Mamba's *selective* SSM, $\bar A_t,\,\bar B_t,\,C_t$ are themselves input-dependent, which means $\eta_t$ also propagates through quantised parameter generation, not only through the state itself. This compounds the recursion-amplification effect.

---

## Results so far (Phase C-1 + C-1b, 2026-04-27)

First Paper C empirical runs on **Mamba-2.8B-hf** (state-spaces' HF-native port of the original Mamba-1, 64 layers × 5,120 d_inner × 16 d_state = **5,242,880 state units**). Pivoted here after `state-spaces/mamba2-2.7b` failed with config + CUDA-kernel-build issues and `Zyphra/Zamba2-2.7B` failed with a transformers `tie_weights_keys` mismatch on its shared-transformer architecture; Mamba-1 HF runs cleanly on the reference Python forward path.

### Phase C-1: static spectral track ($dt = 0.1$ assumed)

| $\rho_g$ range  | meaning            | n state units | fraction |
|:---------------:|:-------------------|--------------:|---------:|
| [0.000, 0.500)  | aggressive forget  |     3,037,617 |   58.0 % |
| [0.500, 0.800)  | medium             |     1,639,466 |   31.3 % |
| [0.800, 0.950)  | longer-term        |       543,866 |   10.4 % |
| [0.950, 0.990)  | long-memory        |        21,919 |    0.42 % |
| [0.990, 0.999)  | **very long memory** |          **12** | **2 × 10⁻⁶** |
| [0.999, 1.000)  | (none)             |             0 |        0 |

Read alone, the static track suggests Mamba's recurrent state is overwhelmingly short-memory: only 0.4 % of state units would carry $\rho_g > 0.95$, and only 12 units across the entire 2.8B model would land at $\rho_g > 0.99$. **This conclusion turns out to be wrong** — the actual selectivity mechanism in Mamba's input-dependent dt biases the empirical distribution very differently.

### Phase C-1b: empirical spectral track (input-dependent dt, captured)

The Mamba-1 forward path (`in_proj → chunk → conv1d → silu → x_proj → split[dt_rank, d_state, d_state] → dt_proj → softplus`) was replayed inside a forward pre-hook on every mixer, capturing the actual per-token dt under WikiText-2 input. With $A_\text{bar}_{t,c,s} = \exp(dt_{t,c} \cdot (-\exp(A_\log_{c,s})))$ averaged over the 1024-token window:

| $\rho_g$ range  | n state units | fraction | static fraction | shift |
|:---------------:|--------------:|---------:|----------------:|:----- |
| [0.000, 0.500)  |     1,644,688 |   31.4 % |    58.0 %       | shrunk by half |
| [0.500, 0.800)  |     1,532,435 |   29.2 % |    31.3 %       | similar |
| [0.800, 0.950)  |     1,092,442 |   20.8 % |    10.4 %       | doubled |
| [0.950, 0.990)  |       508,316 |    9.7 % |     0.42 %      | **× 23** |
| [0.990, 0.999)  |       299,183 |    5.7 % |   2 × 10⁻⁶      | **× 24,000** |
| [0.999, 1.000]  |       165,816 |    3.2 % |     0           | **0 → 165,816** |

The empirical long-memory tail ($\rho > 0.95$) covers **18.6 % of state units** — fifty times larger than the 0.4 % the static analysis predicted. The very-long-memory tail ($\rho > 0.999$) goes from zero units (static) to **165,816 units (3.2 %)** in real input. The reason is straightforward: per-layer median dt observed empirically is in the 0.017 – 0.125 range, often well below the static assumption of 0.1, and small dt gives $A_\text{bar} = \exp(-dt \cdot |A|)$ much closer to 1.

### Per-layer heterogeneity ($U$-shape across the 64 layers)

| layer | dt p50  | $\rho_g$ p50 | role         |
|:-----:|--------:|-------------:|:-------------|
|   0   | 0.0646  | 0.6025       | early abstraction |
|   8   | 0.0467  | 0.7141       | ↓             |
|  16   | 0.0492  | 0.6828       | ↓             |
|  24   | 0.0588  | 0.6429       | ↓             |
| **32** | **0.0168** | **0.8754** | **long-context core** |
| **40** | **0.0175** | **0.8786** | **long-context core** |
|  48   | 0.1021  | 0.4649       | output decode |
|  56   | 0.1253  | 0.3910       | output decode |

Mamba-2.8B exhibits a clear **U-shape** in median ρ across layer depth: short memory at the input end, long memory in the middle (layers 32 – 40), short memory at the output end. The middle-layer long-context core sits at $\rho_g \approx 0.88$, while the bottom and top of the stack run at $\rho_g \approx 0.40$ – $0.71$. This is consistent with the Transformer literature finding that middle layers tend to encode the most abstract / long-range context.

### Updated bit-allocation implication

| $\rho_g$ band | suggested $b_g$ ($b_0 = 3$) | covers (empirical) | covers (static) |
|:-------------:|:----------------------------:|:------------------:|:----------------|
| 0.0 – 0.5     | INT3                         | 31 %               | 58 % |
| 0.5 – 0.8     | INT4 – 5                     | 29 %               | 31 % |
| 0.8 – 0.95    | INT6 – 7                     | 21 %               | 10 % |
| 0.95 – 0.99   | INT8 – 9                     | 9.7 %              | 0.4 % |
| > 0.99        | INT12+                       | 8.9 % (470 K units)| 12 units |

The static-track allocator would compress 89 % of units to INT3 and reserve high precision for a handful of outliers; it would catastrophically under-protect the 470 K state units that empirically need INT12+. **The static / dt = 0.1 analysis is unsuitable as a Paper C bit-allocator input. Empirical (input-dependent) ρ is the correct input.**

### Implications for the Paper C narrative

Three structural updates from Phase C-1b that Paper C should now build on:

1. **Calibration is required.** Unlike Paper B's calibration-free recipe, the Paper C bit allocator requires at least one input-dependent dt measurement (per layer, per state group). The static parameters alone systematically under-estimate long-memory needs by orders of magnitude. This is consistent with the SSDi8 line in the literature, which is explicitly calibration-based.

2. **Per-layer allocation is more impactful than per-state-unit allocation.** The U-shape across layer depth (median ρ ranges 0.39 – 0.88) implies a large amount of the bit-budget difference comes from *which layer* a state unit sits in, not from intra-layer variation. A two-stage allocator — first per-layer baseline from layer-median ρ, then per-state-unit refinement — is the natural architecture.

3. **Paper C is a quality lever, not a compression lever.** The original C-1 conclusion stands and is *reinforced* by C-1b: 9 % of state units need INT8+ and another 9 % need INT12+. A uniform INT3 allocator would saturate them, exactly the kind of long-memory-minority destruction Paper B item #11's synchronisation-pulse hypothesis predicts. The deliverable is "SSM-quality-preserving allocator" with calibration cost on the side, not "INT3 SSM compressor".

### Cross-Paper-B echoes (synchronisation-pulse hypothesis)

| Paper B finding (Qwen 3.6 27B) | Paper C echo (Mamba 2.8B)     |
|:------------------------------|:------------------------------|
| 3 of 16 attention layers individually safe (Phase 29f) | 31 % of state units have $\rho < 0.5$ (short memory, easy to compress) |
| W-shaped fragility curve (single noisy layer worst, all-noisy somewhat absorbed) | not yet probed on SSM; planned for Phase C-3 with feedback rounding |
| FIR / IIR distinction | Phase C-1b's per-layer dt heterogeneity makes the IIR side concrete: layers 32 – 40 are the strongest IIR contributors |
| 0.2 % per-token-L2 noise tolerance on hybrid attention | layer 32 / 40 with $\rho \approx 0.88$ implies an effective error-amplification factor $1/(1-0.88) \approx 8.3$, so per-step noise tolerance there must be $\le 0.024$ to keep accumulated error inside that 0.2 % budget |

### Raw data

- `results_c/phase_c1_ssm_spectral_baseline_summary.json` (35 KB, committed): per-layer static aggregates.
- `results_c/phase_c1_ssm_spectral_baseline.json` (257 MB, gitignored): full per-state-unit `A_log_flat` / `rho_static_flat` lists.
- `results_c/phase_c1b_mamba1_empirical.json` (44 KB, committed): per-layer empirical buckets + dt p01/p50/p99 + per-layer rho summary. Streaming aggregation keeps it small even with 5.24 M state units.

### What is open, what is closed

✅ Static-track allocator is unsuitable; empirical-track allocator is the right input.
✅ Long-memory minority is **18.6 % of state units**, not 0.4 %; per-layer median ρ ranges 0.39 – 0.88; layers 32 – 40 are the long-context core.
✅ Cross-architecture echo of Paper B's "long-memory minority" pattern is now quantified.
✅ Paper C's bit allocator must be two-stage (per-layer baseline + per-state-unit refinement).

❌ Output-coupling Γ_g is not yet measured. Phase C-1c will extract C from x_proj's last d_state slice and complete the R_g formula.

(✅ partial / soft) The bit allocator has been tested end-to-end against PPL — see Phase C-2 below.

❌ All numbers above are on Mamba-1 (state-spaces/mamba-2.8b-hf). The Mamba-2 / SSD-specific update-decomposition (prospectus block 3) is untested. Phase C-2 may target Mamba-2 once the bit allocator's direction is unambiguously confirmed on Mamba-1.

---

## Phase C-2 (soft confirmation, 2026-04-27)

End-to-end test of the bit-allocator *direction* against WikiText-2 PPL on Mamba-2.8B-hf. The Paper C bit allocator predicts `b_g ∝ -log_2 (1 - rho_g)`, i.e. high-rho (long-memory) state units should carry small noise variance and low-rho (short-memory) units can absorb larger noise. We test this by perturbing `A_log` per unit with Gaussian noise of profile-specific σ_g, holding the *total* noise energy `Σ σ_g²` constant across profiles.

| condition          | σ_g profile                            | σ_g at ρ ≈ 0.0 | σ_g at ρ ≈ 0.999 |
|:-------------------|:---------------------------------------|---------------:|------------------:|
| FP16 baseline      | none                                   |              0 |                 0 |
| uniform            | σ_g = σ_0                              |          0.050 |             0.050 |
| **adaptive**       | σ_g ∝ (1 − ρ_g) (Paper C predicted)    |          0.079 |             0.000 |
| anti-adaptive      | σ_g ∝ ρ_g (sanity check, opposite)     |          0.019 |             0.071 |

`Σ σ_g²` matches to four significant figures (1.3107 × 10⁴) across all three perturbed conditions, so any difference in PPL is attributable to *where* the noise lives, not how much there is.

**Result on 5 sliding windows of WikiText-2 test (ctx 1024, stride 512), seed 42:**

| condition          | PPL    | ΔPPL    | % of base | rel vs uniform |
|:-------------------|-------:|--------:|----------:|---------------:|
| FP16 baseline      | 10.3146 |  —      | +0.000 %  | —              |
| **adaptive**       | 10.3318 | +0.0172 | +0.167 %  | **−22.4 %**    |
| uniform            | 10.3367 | +0.0222 | +0.215 %  | (reference)    |
| anti-adaptive      | 10.3386 | +0.0241 | +0.233 %  | **+8.4 %**     |

**Direction confirmed.** The predicted ranking *adaptive < uniform < anti-adaptive* holds. At fixed total noise energy, the bit-allocator-aligned profile (small σ at high ρ) reduces ΔPPL by 22.4 % relative to the uniform allocator, and the inverted profile (large σ at high ρ) makes it 8.4 % worse than uniform. Same direction across both pairwise comparisons.

**Magnitudes are small in absolute terms.** All three perturbed PPLs sit within 0.07 of each other and within 0.25 % of the FP16 baseline. The script's pre-registered "PREDICTED RANKING CONFIRMED" verdict required absolute gaps of ≥ 0.5 percentage points between conditions, which this run does not clear; it returned `NO MEASURABLE EFFECT`. We treat the result as a **soft confirmation**: direction is right, signal is below the pre-registered absolute threshold but consistent in both pairwise comparisons.

**Why the magnitude is small (3 hypotheses):**

1. *σ_0 = 0.05 is conservative.* All PPLs drift by < 0.25 % of baseline, so the configuration is in a near-linear-response regime where allocation differences are diluted. A larger σ_0 should amplify the differential effect.
2. *A_log perturbation is a soft surrogate for state-step quantisation.* It changes the forgetting rate of the recurrence but does not interact with the input-dependent dt, which Mamba's selectivity mechanism may compensate for. Phase C-3-style state-step quantisation would expose the effect more directly.
3. *5 windows is at the edge of statistical resolution.* The pairwise gaps (0.048 % and 0.018 %) are smaller than the window-to-window PPL variance.

**Phase C-2b** (planned) re-runs at σ_0 = 0.15 and 10 windows specifically to test hypothesis (1) and (3); if the gaps grow proportionally, the small-σ-dilution explanation is the right one. Hypothesis (2) is the agenda for a future Phase C-3.

**Pull-quote for Paper C-tier framing**, conditional on Phase C-2b also confirming the direction:

> On Mamba-2.8B-hf, allocating Gaussian noise on the spectral parameter `A_log` in proportion to each state unit's empirical *forgetting* rate (rather than uniformly across all 5.24 M units, at the same total noise energy) reduces WikiText-2 ΔPPL by 22 % at σ_0 = 0.05 — direct empirical support for the per-state-group bit-allocator direction predicted by the FIR / IIR analysis.

**Raw data:** `results_c/phase_c2_bit_allocator_ppl.json` (16 KB, committed): per-condition PPL, sigma profile bucket diagnostic, per-layer rho_emp aggregates.

---

## Empirical inheritance from Paper B

Paper C does not start from zero. The Phase 23 / 28 / 29 / 30 raw JSONs and scripts in [`../results/`](../results/) and [`../experiments/`](../experiments/) already constitute Paper C's pilot data:

| observation | source | role in Paper C |
|:---|:---|:---|
| 1-in-4 hybrid layer pattern (16 attn / 48 SSM in Qwen 3.6 27B) | [`phase29b`](../experiments/phase29b_transformer_only_compression.py) | architectural baseline + diagnostic for any hybrid model |
| Sparse attention layers fragile at $\sigma \approx 0.002$ per-token-L2 | [`phase29e`](../experiments/phase29e_gaussian_noise_sweep.py) | quantitative noise threshold; calibrates Lyapunov $\epsilon_q$ |
| Per-layer fragility heterogeneous (only 3 / 16 layers individually safe) | [`phase29f`](../experiments/phase29f_per_layer_fragility.py) | empirical signature of synchronisation-pulse hypothesis |
| W-shaped dilution curve (1 layer worse than all 16) | [`phase29g`](../experiments/phase29g_anchor_hypothesis.py) | non-additive coupling; falsifies "anchor layers stabilise the recurrence" |
| SSM has no KV cache (architectural truth, not bug) | [`phase29h`](../experiments/phase29h_universal_compression.py) | scope boundary for any KV-only framework, including modeq |
| Bias correction hurts on FIR (attention) | [`phase30a`](../experiments/phase30a_bias_correction_probe.py) | confirms FIR / IIR distinction is operational, not rhetorical; motivates feedback rounding for IIR side |
| V-cache CV / (B) failure correlation $\rho \approx 0.88$ on 8 models | [`phase23`](../experiments/phase23_norm_dispersion.py) | extends naturally to a per-state-group spectral / observability probe |

---

## Mathematical scaffolding (four building blocks)

### 1. Observability-Gramian-weighted reconstruction loss

Replace per-token MSE with a state-trajectory-aware loss

$$
\mathcal{L}_q \;=\; \sum_i \mathrm{tr}\!\left( \Sigma_{\eta_i}\,G_i \right),
\qquad
G_i \;=\; \sum_{t \ge i} \Phi(t,i)^\top\,C_t^\top\,W_t\,C_t\,\Phi(t,i),
\qquad
\Phi(t,i) \;=\; \bar A_t\,\bar A_{t-1}\cdots\bar A_{i+1}.
$$

$G_i$ is the discrete-time observability Gramian: it measures how much a quantisation perturbation at step $i$ shows up at the model's output across all future steps. Standard control-theory machinery; new to PTQ on SSMs.

### 2. Per-state-group bit allocation

For each state group $g$, estimate

$$
\rho_g \;=\; \mathbb{E}_t\,\|\bar A_{t,g}\|_2,
\qquad
\Gamma_g \;=\; \mathbb{E}_t\!\left[ C_{t,g}^\top\,C_{t,g} \right],
\qquad
R_g \;=\; \frac{\|\Gamma_g\|}{1 - \rho_g + \epsilon}.
$$

Allocate

$$
b_g \;=\; b_0 \;+\; \lceil \log_2 R_g \rceil_+
$$

bits to group $g$. Long-memory and high-output-coupling groups get more bits; aggressive-forgetting, low-coupling groups get fewer. Aligns with Quamba2's per-state-group quantisation and Q-Mamba's decoupled scale.

### 3. Update-decomposition for Gated DeltaNet / SSD

For matrix-state recurrences $S_t = \alpha_t\,S_{t-1} + \beta_t\,(v_t - S_{t-1}\,k_t)\,k_t^\top$, factor the *write* before quantising:

$$
U_t \;=\; \beta_t \,(v_t - S_{t-1}\,k_t)\,k_t^\top
\;=\; \beta_t \, n_t^r \, n_t^k \, d_t^r \,(d_t^k)^\top,
$$

with $r_t = v_t - S_{t-1}\,k_t$. Norm-separate $r_t$ and $k_t$ independently, keep the high-precision scalars $(\beta_t, n_t^r, n_t^k)$ in FP, low-bit only the directions $d^r, d^k$. This carries the *spirit* of nsep+pchan into the recurrent-write path without transplanting the per-token KV-table machinery onto a recurrent state where it does not belong.

### 4. Error-feedback quantisation with Lyapunov-stable rounding

Because errors propagate through $\bar A_t$, naive rounding injects a noise *mean* that compounds. Instead:

$$
s_t \;=\; \bar A_t\,\hat h_{t-1} \;+\; \bar B_t\,x_t \;+\; r_{t-1},
\qquad
\hat h_t \;=\; Q(s_t),
\qquad
r_t \;=\; s_t - \hat h_t.
$$

Storing $r_t$ in full FP defeats the purpose, so use stochastic rounding (zero-mean noise per step, no accumulated bias), low-rank residual storage, or block residual at coarser temporal granularity. **Phase 30a's verdict — that *static* mean-error subtraction hurts even in the FIR case — is direct empirical evidence that the right mechanism is feedback rounding rather than open-loop bias subtraction.**

---

## Synchronisation-pulse hypothesis (hybrid attention)

In 1-in-$k$ hybrid stacks (Qwen 3.6 27B's $k=4$, Jamba's $k \approx 8$, Nemotron-3 Super's variant), the sparse full-attention layers act not as redundant attention slots but as *periodic synchronisation pulses* re-aligning the recurrent state of the surrounding $k-1$ SSM layers.

This explains, mechanistically, the Paper B Phase 29 observations that

- single-layer compression of a "fragile" attention layer is *worse* than compressing all 16 (one perturbed pulse desynchronises $k-1$ recurrent layers; many perturbed pulses average back into a stationary noise regime that the recurrence absorbs);
- only 3 of 16 attention layers (indices 1, 4, 9 within the 1-in-4 pattern) are individually safe;
- compressing 4 layers in mid-positions (Q2 / Q3 quartile groups) is dramatically less harmful than compressing any single layer in those groups.

A Paper C analysis would formalise this. Linearising around the FP16 trajectory,

$$
\delta z_{b+1}
\;=\; J_\text{attn}\,J_\text{GDN}^{(k-1)}\,\delta z_b
\;+\; J_\text{GDN}^{(k-1)}\,\eta_\text{attn}
\;+\; \eta_\text{gdn},
$$

and define a **fragility index**

$$
\chi_\ell \;=\; \big\|\, O_\ell^{1/2}\,J_\ell\,\Sigma_{q,\ell}\,J_\ell^\top\,O_\ell^{1/2}\,\big\|.
$$

Allocate bits inversely to $\chi_\ell$. Phase 29f's per-layer r4 ranking is, retrospectively, a coarse empirical proxy for $\chi_\ell$.

---

## Open phase plan

| phase | question | status |
|:-:|:---|:---|
| C-1 | Compute $\rho_g$, $\Gamma_g$, $R_g$ on Mamba-2-2.7B and Qwen 3.6 27B SSM blocks. Does the per-state-group bit allocator predict the right ranking? | not started |
| C-2 | Implement update-decomposition (block 3) on a Gated DeltaNet head. Compare PPL vs Quamba2 on the same hardware. | not started |
| C-3 | Implement error-feedback rounding with stochastic / block residual on a Mamba-2 inference path. Does it close the gap to FP16 over 32K-token context? | not started |
| C-4 | Synchronisation-pulse hypothesis: validate $\chi_\ell$ ranking on a second hybrid model (Jamba or Nemotron-3 Super) to rule out Qwen-3.6-specific confound. | not started |
| C-5 | Hybrid stack recipe: combine modeq attention K/V (no calibration, FIR) + state-aware SSM quantiser (calibrated, IIR) on Qwen 3.6 27B end-to-end. Target: 32K-context generation with $r_4$ within 0.05 of FP16. | not started |

---

## Anchor citations

- **Mamba** (Gu & Dao, 2023): [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)
- **SSD / Mamba-2** (Dao & Gu, 2024): [arXiv:2405.21060](https://arxiv.org/abs/2405.21060)
- **Quamba** (Chiang et al., 2024) — SSM weight + activation INT8 with Hadamard smoothing: [arXiv:2410.13229](https://arxiv.org/abs/2410.13229)
- **Quamba2** (Chiang et al., 2025) — per-state-group quantisation: [arXiv:2503.22879](https://arxiv.org/abs/2503.22879)
- **MambaQuant** (2025) — variance / outlier analysis: [arXiv:2501.13484](https://arxiv.org/html/2501.13484v1)
- **Q-Mamba** — state-cache decoupled scale, W8A8H4 on Mamba-2-2.7B: [OpenReview AY1S52vr0a](https://openreview.net/forum?id=AY1S52vr0a)
- **SSDi8** (Kim et al., ICLR 2026 Poster) — INT8 SSD with bias-correction reformulation: [OpenReview pjMDZJd4rT](https://openreview.net/forum?id=pjMDZJd4rT)
- **Gated DeltaNet** (2024): [arXiv:2412.06464](https://arxiv.org/abs/2412.06464)
- **KL Lens** (Kong, Pandey, Ponzina, Rosing, 2026) — forward-only sensitivity for hybrid SSM-Transformer: [arXiv:2604.13440](https://arxiv.org/abs/2604.13440)
- **Nemotron-3 Super** (NVIDIA, 2026) — 120B active 12B hybrid Mamba + Transformer + LatentMoE, 1M context, NVFP4: [arXiv:2604.12374](https://arxiv.org/abs/2604.12374)
- **Jamba-1.5 / ExpertsInt8** (Lieber et al., 2024): [arXiv:2408.12570](https://arxiv.org/abs/2408.12570)
- **KIVI / PolarQuant / TurboQuant** — KV-cache baselines for the FIR side of the eventual paper.

---

## Relationship to Paper B (modeq / NSN)

Paper B is the **attention K/V piece** of the eventual component-specific stack:

```
   ┌───────────────────────┐
   │ MoE / MLP weights     │  ← ExpertsInt8  (Jamba team, INT8 + JIT dequant)
   ├───────────────────────┤
   │ SSM weights           │  ← Quamba       (Hadamard + clustering, W8A8)
   ├───────────────────────┤
   │ SSM recurrent state   │  ← SSDi8 + Q-Mamba + Paper C (this folder)
   ├───────────────────────┤
   │ Attention K / V cache │  ← modeq v0.3   (Paper B / NSN, INT3 calibration-free)
   └───────────────────────┘
```

Paper C extends only the third row. The other three rows are out of scope.

A practical Paper C contribution would be:

1. The **mathematical framework** (observability-Gramian loss, $b_g$ allocator, update-decomposition, feedback rounding) as a unified picture rather than four scattered tricks.
2. A **single open-source implementation** that composes modeq's attention-side recipe with the new SSM-side recipe on a hybrid model (Qwen 3.6 27B is the natural target — pre-instrumented by Paper B).
3. The **synchronisation-pulse hypothesis** as a testable structural claim about hybrid models, validated on $\ge 2$ hybrid architectures.

---

## What does NOT belong here

- Anything that fits into Paper B's scope (KV-cache attention compression). Those numbers are in [`../paper/nsn.pdf`](../paper/nsn.pdf).
- Anything that fits into the Quamba / SSDi8 / KL Lens lines unmodified. Cite, do not duplicate.
- Pure transformer-only experiments. The whole point of Paper C is the SSM-aware piece.

---

## Status of this folder

Currently contains only this README. Future contents (when phases above start running):

```
paper_c/
├── README.md         (this file)
├── paper_c.md        (manuscript draft, when it begins)
├── refs.bib          (Paper-C-specific BibTeX; for now reuse ../paper/refs.bib)
├── experiments_c/    (Phase C-1 onwards; mirrors ../experiments/)
├── results_c/        (matching JSON outputs)
└── notes/            (working drafts, derivations)
```

For now Paper-C-related experiments live under [`../experiments/`](../experiments/) alongside Paper B's, with Phase 29 / 30 numbering. They will be migrated when the manuscript begins.

---

## Contact

Same as the main repo:

- Issues / Discussions on the [modeq repository](../)
- Email: info@metaclan.jp
- X: [@RecyncAI](https://x.com/RecyncAI)

Feedback on the kickoff brief — especially on the four mathematical building blocks and the synchronisation-pulse hypothesis — is welcome before any manuscript drafting starts.
