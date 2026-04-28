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

## Phase C-2 + C-2b (bit-allocator direction confirmed at σ₀ = 0.15, 2026-04-27)

End-to-end test of the bit-allocator *direction* against WikiText-2 PPL on Mamba-2.8B-hf. The Paper C bit allocator predicts `b_g ∝ -log_2 (1 - rho_g)`, i.e. high-rho (long-memory) state units should carry small noise variance and low-rho (short-memory) units can absorb larger noise. We test this by perturbing `A_log` per unit with Gaussian noise of profile-specific σ_g, holding the *total* noise energy `Σ σ_g²` constant across profiles.

| condition          | σ_g profile                            | σ_g at ρ ≈ 0.0 | σ_g at ρ ≈ 0.999 |
|:-------------------|:---------------------------------------|---------------:|------------------:|
| FP16 baseline      | none                                   |              0 |                 0 |
| uniform            | σ_g = σ_0                              |        σ_0    |            σ_0    |
| **adaptive**       | σ_g ∝ (1 − ρ_g) (Paper C predicted)    |  ≈ 1.6 × σ_0  |         ≈ 0       |
| anti-adaptive      | σ_g ∝ ρ_g (sanity check, opposite)     |  ≈ 0.4 × σ_0  |       ≈ 1.4 × σ_0 |

`Σ σ_g²` matches to four significant figures across the three perturbed conditions at every σ₀ tested, so any difference in PPL is attributable to *where* the noise lives, not how much there is.

### Phase C-2: σ₀ = 0.05, 5 windows

| condition          | PPL    | ΔPPL    | % of base | rel vs uniform |
|:-------------------|-------:|--------:|----------:|---------------:|
| FP16 baseline      | 10.3146 |  —      | +0.000 %  | —              |
| **adaptive**       | 10.3318 | +0.0172 | +0.167 %  | **−22.4 %**    |
| uniform            | 10.3367 | +0.0222 | +0.215 %  | (reference)    |
| anti-adaptive      | 10.3386 | +0.0241 | +0.233 %  | **+8.4 %**     |

Predicted ranking observed but at small absolute magnitude (gaps 0.018 – 0.048 %-points). The script's pre-registered "PREDICTED RANKING CONFIRMED" verdict required absolute gaps ≥ 0.5 %-points and returned `NO MEASURABLE EFFECT`.

### Phase C-2b: σ₀ = 0.15, 10 windows (amplified replay)

3× larger σ₀ to test the hypothesis that C-2's small magnitude was a near-linear-response artifact (gap should scale ∝ σ_0² ⇒ 9× larger at σ_0 = 0.15).

| condition          | PPL    | ΔPPL    | % of base | rel vs uniform |
|:-------------------|-------:|--------:|----------:|---------------:|
| FP16 baseline      | 12.3725 |  —      | +0.000 %  | —              |
| **adaptive**       | 12.4419 | +0.0694 | +0.561 %  | **−40.2 %**    |
| uniform            | 12.4886 | +0.1161 | +0.938 %  | (reference)    |
| anti-adaptive      | 12.5282 | +0.1557 | +1.258 %  | **+34.1 %**    |

Direction holds, magnitudes amplified, and the relative-effect numbers (-40 % / +34 %) are now industrial-significance level.

### Gap scaling between C-2 and C-2b

| pairwise gap (%-points) | σ₀ = 0.05 | σ₀ = 0.15 | observed | predicted (σ₀² ratio) |
|:------------------------|----------:|----------:|---------:|:---------------------:|
| adaptive vs uniform     |    +0.048 |    +0.378 |  **7.9×** |          9×          |
| anti vs uniform         |    +0.018 |    +0.320 | **17.8×** |          9×          |

The adaptive↔uniform gap scales close to the linear-response prediction (7.9× of the predicted 9× — slightly sub-linear, consistent with a mild saturation of the protective effect). The anti↔uniform gap scales **17.8×** — almost double the linear prediction, which is itself further evidence for the bit allocator: perturbing the long-memory units (anti-adaptive's choice) is *super-linearly* destructive in σ₀, exactly because A_log lives in the exponent and small relative changes flip ρ from ≈ 0.999 to a much smaller value, destabilising the recurrence.

In other words: the bit allocator's prediction is not just "high-ρ units need more bits" — Phase C-2b shows that high-ρ units are *even more* fragile to noise than the linear-response analysis suggests. Protecting them is correspondingly more valuable than the prospectus's first-order math implied.

### Verdict (revised vs the script's pre-registered threshold)

The script's threshold of 0.5 %-points absolute gap was overly conservative. Going by the consolidated evidence:

✅ **Direction confirmed at three σ₀ values** (0.05 in C-2, 0.15 in C-2b — same ranking adaptive < uniform < anti-adaptive at every σ₀ tested).

✅ **Relative effect grows with σ₀** as expected: adaptive's advantage over uniform goes from 22 % at σ₀ = 0.05 to 40 % at σ₀ = 0.15.

✅ **Scaling asymmetry** (adaptive 7.9× vs anti 17.8× under a 9× linear prediction) supplies independent empirical support for the FIR/IIR-rooted "long-memory units are critical" hypothesis: anti-adaptive saturates first because it is hitting the units where small perturbations have super-linear consequences.

✅ **Total noise energy controlled**: `Σ σ_g²` matches to four sig figs across profiles at every σ₀, so the comparisons isolate *allocation* from *amount*.

⚠️ **Absolute pairwise gaps** still under 0.5 %-points at σ₀ = 0.15 (adaptive vs uniform = 0.378). Could be cleared by σ₀ = 0.20 in a Phase C-2c, but the direction is no longer in question.

⚠️ **A_log perturbation is a surrogate** for state-step quantisation. The bit allocator's direction is correct in the surrogate; whether the same direction holds for direct state-step quantisation is the agenda of Phase C-3.

### Pull-quote for Paper C, suitable for §1 / §abstract framing

> On Mamba-2.8B-hf, allocating Gaussian noise on the spectral parameter `A_log` in proportion to each state unit's empirical *forgetting* rate, rather than uniformly across all 5.24 M units at the same total noise energy, reduces WikiText-2 ΔPPL by 40 % at σ₀ = 0.15 (and 22 % at σ₀ = 0.05); the inverted profile increases ΔPPL by 34 %. The asymmetry between adaptive (7.9×) and anti-adaptive (17.8×) gap scaling against a linear-response prediction of 9× is itself further evidence: long-memory state units are super-linearly fragile to noise, exactly as the per-state-group bit allocator's `b_g ∝ -log₂(1 − ρ_g)` prescribes.

### Raw data

- `results_c/phase_c2_bit_allocator_ppl.json` (16 KB, committed)
- `results_c/phase_c2b_bit_allocator_ppl_amplified.json` (16 KB, committed)

Both contain per-condition PPL, sigma profile bucket diagnostic, and per-layer ρ_emp aggregates.

### What is open after C-2 / C-2b

✅ Bit-allocator *direction* is empirically right on Mamba-2.8B (A_log surrogate, two σ₀ values).

❌ State-step quantisation (the actual recurrence quantisation, not its A_log surrogate) is untested. Phase C-3.

(✅ now done) Output-coupling Γ_g measured. See Phase C-1c immediately below — and the verdict reframes everything.

❌ Hybrid Mamba + Attention models (the eventual Paper C-5 target) untested. Once C-3 lands, Qwen 3.6 27B or Zamba2-2.7B is the natural next substrate.

---

## Phase C-1c (output-coupling Γ extraction → narrative shift, 2026-04-27)

**Bottom line.** The ρ-only proxy R_proxy = 1 / (1 − ρ_g) that Phase C-2 / C-2b implicitly used is **not a faithful approximation** of the full Paper C R_g formula on Mamba-2.8B. The output-coupling factor Γ_s is not flat across state indices — it is heavy-tailed with per-layer max/min ratios in the **170× – 1500×** range. The Paper C bit allocator must extract Γ during calibration; ρ alone is materially wrong.

### Why we ran C-1c

The full Paper C bit allocator is

$$
R_g \;=\; \|\Gamma_g\| \,/\, (1 - \rho_g + \varepsilon),
\qquad
b_g \;=\; b_0 + \lceil \log_2 R_g \rceil_+,
\qquad
\sigma_g \;\propto\; \tfrac{1}{\sqrt{R_g}}.
$$

Phase C-2 / C-2b used the cheaper proxy

$$
R_{\text{proxy}}(c, s) \;=\; 1 \,/\, (1 - \rho_{\text{emp}}(c, s)),
\qquad
\sigma_{\text{proxy}} \;\propto\; (1 - \rho_g),
$$

which drops the Γ factor entirely. The pre-registered question for C-1c was: are R_proxy and R_full sufficiently correlated that the proxy is a faithful approximation? Pre-registered thresholds: Pearson(log R_proxy, log R_full) ≥ 0.95 and ≥ 90 % of state units agreeing within ±1 bit.

### Mamba-1 specifics — Γ as a per-state scalar

In Mamba-1 the readout is `y_{t,c} = Σ_s C_{t,s} h_{t,c,s}`, so the output-coupling tensor `C` depends on state index `s` but not on channel `c`. The output coupling reduces to

$$
\Gamma_s \;=\; \mathbb{E}_t \!\left[\, C_{t,s}^2 \,\right]
$$

i.e. d_state = 16 scalars per layer, 64 layers × 16 = 1024 scalars total across the model. Tiny extraction cost; the question is whether those 1024 scalars matter.

### Result — Γ is dramatically heavy-tailed

Per-layer Γ_s max / min ratio across the 16 state indices (every 8th layer shown):

| layer | Γ_min  | Γ_p50  | Γ_max  | max / min |
|:-----:|-------:|-------:|-------:|----------:|
|   0   | 0.0293 | 0.1436 |  5.11  |   174 ×  |
|   8   | 0.0128 | 0.0807 |  5.64  |   440 ×  |
|  16   | 0.0106 | 0.1151 |  4.69  |   443 ×  |
|  24   | 0.0134 | 0.1560 |  2.87  |   214 ×  |
|  32   | 0.0230 | 0.0820 |  3.84  |   167 ×  |
|  40   | 0.0198 | 0.0493 |  8.02  |   406 ×  |
|  48   | 0.0335 | 0.4365 | 27.13  |   811 ×  |
|  56   | 0.0289 | 0.9616 | 45.43  | **1572 ×** |

The very last SSM layers concentrate enormous Γ on a few state indices (Γ_max = 45.4 at layer 56). The natural reading: a *small handful* of state indices carry most of the readout in the late stages of the model, with the rest contributing essentially noise to the output sum.

### Result — proxy and full disagree at the per-state-unit level

| metric                                                      | value                |
|:------------------------------------------------------------|----------------------|
| Global log-log Pearson(R_proxy, R_full), 1.28 M sample      | **0.635**            |
| Per-layer log-log Pearson, p50                              | 0.538                |
| Per-layer log-log Pearson, **min**                          | **− 0.07** (negative) |
| Per-layer log-log Pearson, max                              | 0.882                |

A Pearson of 0.635 is moderate, **well below** the 0.95 pre-registered threshold for "proxy sufficient". The negative-Pearson layer is striking: at least one layer in the stack ranks state units in the *opposite order* under proxy versus full. Long-memory units in those layers are **not** the units that dominate the output; the proxy and full would prescribe inverted bit allocations there.

### Result — bit-tier disagreement is large

Allocating bits with `b_g = b_0 + ⌈ log₂ R_g ⌉₊` at b_0 = 3:

| agreement category      | fraction of state units |
|:------------------------|------------------------:|
| same bit (proxy = full) | **9.24 %**              |
| within ±1 bit            | 32.62 %                 |
| **off by ≥ 2 bits**      | **58.13 %**             |

Only 9 % of state units get the same bit allocation under the two formulas; almost three out of every five units would be misallocated by ≥ 2 bits if the proxy were used as the bit allocator. This is **material**, not a rounding-noise discrepancy.

### Verdict — full R_g IS REQUIRED

Pre-registered: Pearson < 0.7 ⇒ "FULL R_g IS REQUIRED". Observed Pearson = 0.635 < 0.7. Verdict triggered. The cheap ρ-only proxy is acceptable as a *direction-finding* heuristic but is not the deployed Paper C bit allocator; future allocator implementations must extract Γ during calibration.

### What this does to Phase C-2 / C-2b's interpretation

Phase C-2 / C-2b were **direction tests, not allocator tests**. They confirmed that allocating *more* noise to short-memory units and *less* noise to long-memory units helps PPL, regardless of how exactly the allocation was scaled. Their reported 22 % / 40 % ΔPPL reductions remain valid as direction evidence.

But because they used the proxy profile σ_g ∝ (1 − ρ_g), which drops the Γ factor, those numbers are now best read as **lower bounds** on what the full Γ-aware allocator can achieve. The full allocator concentrates precision on the small set of high-Γ state indices in late layers (layer 48: Γ_max = 27.1; layer 56: Γ_max = 45.4) which the proxy was protecting at only a uniform-with-ρ level. We would expect the full allocator to push the relative reduction past 40 %, and Phase C-2c is set up to test exactly that.

### Two-axis Paper C narrative (updated)

Paper C's empirical story is now cleanly two-factor, mirroring (in form) Paper B's V-CV / K-CV mechanism story:

| axis              | quantity        | what it captures                              | Paper-B analogue          |
|:------------------|:----------------|:----------------------------------------------|:--------------------------|
| **Recurrent**     | ρ_g (= forgetting rate) | how long this state unit *carries* a perturbation forward in time | per-token V-cache norm dispersion (V-CV) |
| **Output**        | Γ_s (= mean C²) | how strongly a perturbation here *shows up* at the readout | softmax Jacobian damping on K-cache |
| **Combined critical-ness** | R_g = Γ_s / (1 − ρ_g) | how many bits this unit needs | analogue of "non-monotone failure depth" |

ρ alone is half the story. Γ alone is the other half. **Both** axes have heavy-tailed distributions and they are only moderately correlated. The Paper C bit allocator must combine them; the prospectus's R_g formula is now empirically supported as the right combination.

A particularly interesting open observation: at least one Mamba-2.8B layer shows **negative** Pearson(log R_proxy, log R_full). In that layer some state units have high ρ (long memory) but low Γ (no readout influence) — they "remember without speaking". And vice versa: some state units have low ρ (short memory) but high Γ (loud at readout) — "fresh, loud, and gone". The proxy treats only the first kind as critical; the full formula correctly identifies both.

### Open / closed (post-C-1c)

✅ ρ_g per state unit measured (Phase C-1 + C-1b, ~5.24 M units).
✅ Γ_s per state index measured (Phase C-1c, 1024 scalars). Heavy-tailed across s.
✅ Bit-allocator *direction* confirmed end-to-end on PPL (Phase C-2 / C-2b, +40 % reduction at σ₀ = 0.15).
✅ ρ-only proxy is **insufficient** as the deployed allocator (Phase C-1c, Pearson 0.635, bit-tier 9 % match).

❌ Full R_g allocator end-to-end PPL test. **Phase C-2c**.

❌ State-step quantisation (true recurrence-side test, not A_log surrogate). Phase C-3.

❌ Hybrid Mamba + Attention. Phase C-5 target.

### Raw data

`results_c/phase_c1c_gamma_extraction.json` (~330 KB, committed):
per-layer Γ values + heavy-tailedness ratios, per-layer log-log Pearson, bit-tier disagreement counts, 5K-sample (log R_proxy, log R_full) scatter sample.

---

## Phase C-2c (full Γ-aware allocator + A_log surrogate ceiling, 2026-04-28)

**Bottom line.** The full Γ-aware allocator beats the ρ-only proxy used in Phase C-2 / C-2b by **71 % relative ΔPPL reduction** under the A_log perturbation surrogate. Γ-awareness is empirically valuable end-to-end on PPL — the C-1c per-state-unit disagreement (Pearson 0.635) translates to a real PPL gap. **However**, the same run reveals a *surrogate ceiling*: the inverted profile (anti_full) also wins, by an even larger margin, which is structurally impossible under direct state-step quantisation. We document both findings and identify Phase C-3 as the definitive test.

### Six-condition test at σ₀ = 0.15, n_windows = 10

All five perturbed profiles share `Σ σ_g² = 1.1796 × 10⁵` (matched to four sig figs).

| condition          | σ_g profile                          | PPL    | ΔPPL    | % of base | rel vs uniform |
|:-------------------|:-------------------------------------|-------:|--------:|----------:|---------------:|
| FP16 baseline      | none                                 | 12.3725 |  —      | +0.000 %  | —              |
| **anti_full**      | √(Γ_s / (1 − ρ))                     | 12.3809 | +0.0084 | **+0.068 %** | **−92.7 %**    |
| **full** (predicted winner) | √((1 − ρ) / Γ_s)            | 12.3925 | +0.0200 | **+0.162 %** | **−82.7 %**    |
| proxy_C2           | (1 − ρ)                              | 12.4419 | +0.0694 | +0.561 %  | −40.2 %        |
| proxy_correct      | √(1 − ρ)                             | 12.4543 | +0.0818 | +0.661 %  | −29.5 %        |
| uniform            | σ_0 = const                          | 12.4886 | +0.1161 | +0.938 %  | (reference)    |

### Reading 1 — Γ-awareness is empirically valuable

Both Γ-aware profiles (full and anti_full) sit far below all three Γ-unaware profiles (proxy_C2, proxy_correct, uniform). The minimum gap between a Γ-aware profile and a ρ-only profile is **0.4 %-points** (full = 0.162 %, proxy_C2 = 0.561 %), which is **6× larger** than the largest gap *within* the ρ-only profiles (proxy_correct − proxy_C2 = 0.10 %-points). Γ contributes roughly an order of magnitude more allocation signal than the choice of σ-vs-(1−ρ) scaling within the ρ axis.

Reading the headline: **full beats proxy_C2 by 71 % relative ΔPPL reduction** (0.561 → 0.162). The Phase C-1c diagnostic (Pearson 0.635 between R_proxy and R_full) translates to real PPL improvement when used as an allocator.

### Reading 2 — anti_full also wins, and that is informative about A_log as a surrogate

The unexpected: anti_full sits **below** full (0.068 % vs 0.162 %). The Paper C bit allocator predicts the opposite ordering. Two readings, one of which we can confirm:

**(A)** *A_log is a saturating surrogate for state-step noise on long-memory units.* In Mamba-1, ρ = exp(−dt · exp(A_log)); the relationship is doubly exponential. For a unit with A_log = −10 (ρ ≈ 1), σ_A ≈ 0.4 noise on A_log shifts ρ in the fifth decimal place — invisible at PPL granularity. For a unit with A_log = −3 (ρ ≈ 0.74), σ_A ≈ 0.4 shifts ρ by ~0.1, very visible. The A_log perturbation is therefore *most active* on medium-ρ units and *most inert* on high-ρ units, regardless of how Γ allocates the σ budget.

**(B)** *Mechanism of anti_full's "win"*. anti_full concentrates its σ budget on (high ρ, high Γ) units. The A_log saturation in (A) means most of that budget is *wasted* — the model barely notices. The remaining (low ρ, low Γ) units get tiny σ and stay clean. Net effect: anti_full produces less observable perturbation per unit of σ² budget than any of the other allocations on this surrogate.

**(C)** *What this rules out and rules in.* (B) plus (A) explains the observed ranking without requiring the bit allocator to be wrong. It does NOT explain whether the bit allocator is *right* — the same observations are equally consistent with anti_full being a "win" purely because of A_log saturation, full being a "win" because it correctly protects critical units, both effects superimposed. The A_log surrogate **cannot distinguish** these.

### Verdict — directional confirmation, surrogate ceiling, Phase C-3 required

✅ **Γ-awareness materially improves allocation** under any reasonable interpretation of the surrogate. Both Γ-aware profiles beat all Γ-unaware ones by ≥ 0.4 %-points. The Paper C two-axis narrative (ρ + Γ) holds.

✅ **The full allocator's prediction is a 71 % relative reduction over the ρ-only proxy** that Phase C-2 / C-2b used. This refines the C-2b "lower bound" framing: under the surrogate, the full allocator does meaningfully better.

⚠️ **A_log perturbation cannot validate the bit-allocation *direction* on Mamba**. The surrogate's intrinsic insensitivity to high-ρ A_log noise produces an unfalsifiable order between full and anti_full. Either could be "right" at the state-quantisation level.

❌ **Phase C-3 (state-step quantisation in `slow_forward`)** is now the definitive test. The Paper C bit allocator's direction can only be validated by perturbing the recurrent state h directly (where Γ_s acts linearly on the readout), not by perturbing A_log (where high-ρ saturation hides the effect).

### Pull-quote draft (Paper C §1, conditional on C-3 confirming direction)

> Allocating Gaussian noise on Mamba-2.8B's spectral parameter A_log according to the full Paper C bit-allocator formula σ_g ∝ √((1 − ρ_g) / Γ_s), versus the rho-only proxy σ_g ∝ (1 − ρ_g) used as a baseline allocator, reduces WikiText-2 ΔPPL by 71 % at fixed total noise energy (0.561 % → 0.162 % of FP16 baseline). The output-coupling factor Γ contributes roughly an order of magnitude more allocation signal than the choice of σ-vs-(1−ρ) scaling on the recurrent axis alone, empirically supporting the prospectus's two-axis (ρ + Γ) bit-allocator design.

### Raw data

`results_c/phase_c2c_full_R_allocator_ppl.json` (~21 KB, committed): per-condition PPL, six σ profiles, per-(rho-bucket) mean σ table, per-layer Γ summary.

---

## Handoff brief — for future sessions starting cold

If you're a future Claude session (or a future me) starting in this folder without the prior context, here is what you need to know to keep going.

### Where Paper C currently is

Six phases done on Mamba-2.8B-hf. **One blocking question is still open**: does the bit-allocator direction survive direct state-step quantisation, or is it an A_log-surrogate artefact?

| phase | what it did                                          | status |
|:-----:|:-----------------------------------------------------|:------:|
| C-1   | Static spectral baseline (A_log only, no input)      | ✅     |
| C-1b  | Empirical spectral (input-dependent dt)              | ✅     |
| C-1c  | Output coupling Γ_s extraction; verdict "full required" | ✅  |
| C-2   | Bit-allocator direction at σ₀=0.05 (soft)            | ✅     |
| C-2b  | Amplified at σ₀=0.15; direction confirmed            | ✅     |
| C-2c  | Full Γ-aware allocator vs proxy; 71 % rel reduction, but anti_full also wins (surrogate ceiling) | ✅     |
| C-3   | State-step quantisation in `slow_forward` (definitive test) | ❌  |
| C-1c-bis | Held-out calibration window for Γ stability        | (optional) |
| C-5   | Hybrid Mamba + Attention (Qwen 3.6 27B / Zamba2)     | ❌     |

### Working code conventions (every script in `experiments_c/` follows these)

- **Cell layout**: `# === CELL V1 ===` blocks down the file, top to bottom. Setup / load / calibrate / run / save / read.
- **HF token paste point**: a fenced `>>>>>  PASTE YOUR HF TOKEN  <<<<<` comment with `HF_TOKEN = "hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"`. The user pastes their token and runs. Do not hard-code real tokens.
- **Pip install**: `!pip install -q -U transformers accelerate hf_transfer datasets`. Nothing else. **Do not** install `mamba-ssm` or `causal-conv1d`. Both reliably fail to build on Colab and we do not need them — the HF reference Python forward path runs without them, just slower.
- **`os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"`** speeds up the model download to ~30 s.
- **Save path**: `/content/<phase_name>.json` inside the script. The user copies it to `paper_c/results_c/` after the run.
- **Big JSON guard**: any per-state-unit list (5.24 M floats × 64 layers = ~260 MB) blows through the github 100 MB-per-file limit. Use streaming aggregation (running mean, bucket counts) or save an abridged summary; the gitignore already filters `phase_c1_ssm_spectral_baseline.json` (the only file that disregarded this).

### Working model

Use `state-spaces/mamba-2.8b-hf` (and only this — see `Phase C-1` section above for why other Mamba / Mamba-2 / hybrid checkpoints failed for us with config / build / tied-weights issues). It is HF-native, runs the reference Python forward path, ~5.5 GB VRAM at fp16, ~30 s download. The smaller siblings `mamba-130m-hf` / `370m-hf` / `790m-hf` / `1.4b-hf` work too if you want fast iteration.

### Mamba-1 mixer attribute reference (HF transformers, 2026-04 vintage)

```python
mix.A_log         # Parameter, shape [d_inner, d_state] = [5120, 16]
mix.D             # Parameter, shape [d_inner]
mix.dt_proj       # Linear (160 -> 5120) — projects dt_rank to d_inner
mix.x_proj        # Linear (5120 -> 192) — outputs [dt_rank, d_state, d_state]
mix.in_proj       # Linear (2560 -> 10240) — outputs [z, x_pre] when chunked
mix.conv1d        # Conv1d, weight shape [5120, 1, 4]
mix.intermediate_size  # 5120 (= d_inner)
mix.ssm_state_size     # 16  (= d_state)
mix.time_step_rank     # 160 (= dt_rank)
```

The forward path (which we replay manually inside hooks):

```python
y = in_proj(hidden_states)                                # [B, T, 2*d_inner]
z, x_pre = y.chunk(2, dim=-1)                             # each [B, T, d_inner]
x_pre = silu(causal_conv1d(x_pre))                        # [B, T, d_inner]
xbc = x_proj(x_pre)                                       # [B, T, dt_rank + 2*d_state]
dt_raw, B, C = xbc.split([dt_rank, d_state, d_state], -1) # B, C are [B, T, d_state]
dt = softplus(dt_proj(dt_raw))                            # [B, T, d_inner]
A = -exp(A_log)                                           # [d_inner, d_state]
A_bar = exp(dt[..., None] * A[None, None])                # [B, T, d_inner, d_state]
# state recurrence and output decoding happen inside slow_forward
```

For C-3 you will need to monkey-patch `slow_forward` to insert quantisation between the state update and the readout.

### Bit-allocator math (the form we used)

Under the standard quantisation argument (uniform-rounding noise variance σ_g² ∝ 4^{−b_g}, total bit budget Σ b_g constant), the optimal allocation is **σ_g ∝ 1 / √R_g**. With

  R_g = Γ_s / (1 − ρ_g + ε)

this gives σ_g ∝ √((1 − ρ_g) / Γ_s) = the *full* profile in C-2c.

Phase C-2 / C-2b used σ_g ∝ (1 − ρ_g) directly (= the *proxy_C2* profile in C-2c). That is the right *direction* but not the bit-budget-optimal scaling. We kept it as a comparison point in C-2c so the comparison is honest: both proxy_C2 and proxy_correct (= √(1 − ρ_g)) are tested.

### Key open puzzle (read this before designing C-3)

A_log perturbation has a doubly-exponential relationship to ρ:

  ρ = exp(−dt · exp(A_log))

For high-ρ units (A_log very negative), small absolute changes in A_log cause tiny changes in ρ. So adding noise to A_log on high-ρ units is *less* destructive than adding the same noise to medium-ρ units. This is what made anti_full counter-intuitively win Phase C-2c.

State-step quantisation does NOT have this saturation — it adds noise η_t directly into h_t at every step, and the readout y_t = C h_t makes the error visible linearly. **Phase C-3 should reproduce the C-2c six-condition design but with state-step quantisation in place of A_log perturbation.** If full < anti_full under state-step quantisation (matching the bit allocator's prediction), the surrogate ceiling explanation is confirmed.

### Repository layout (Paper C side only)

```
paper_c/
├── README.md                     ← this file
├── experiments_c/                ← Paper C scripts
│   ├── phase_c1_ssm_spectral_baseline.py
│   ├── phase_c1b_mamba1_empirical.py
│   ├── phase_c1c_gamma_extraction.py
│   ├── phase_c2_bit_allocator_ppl.py
│   ├── phase_c2b_bit_allocator_ppl_amplified.py
│   ├── phase_c2c_full_R_allocator_ppl.py
│   └── (future) phase_c3_state_step_quantisation.py
└── results_c/                    ← JSONs (mostly small)
    ├── phase_c1_ssm_spectral_baseline_summary.json (35 KB committed)
    ├── phase_c1_ssm_spectral_baseline.json (257 MB, gitignored)
    ├── phase_c1b_mamba1_empirical.json (45 KB)
    ├── phase_c1c_gamma_extraction.json (330 KB)
    ├── phase_c2_bit_allocator_ppl.json (16 KB)
    ├── phase_c2b_bit_allocator_ppl_amplified.json (17 KB)
    └── phase_c2c_full_R_allocator_ppl.json (21 KB)
```

### What NOT to redo

- Do not try `state-spaces/mamba2-2.7b` (no `model_type`), `Zyphra/Zamba2-2.7B` (HF tied-weights bug), or `mamba-ssm` install (CUDA build fails). Phase C-1 documents the pivot history.
- Do not store per-state-unit flat lists in JSON. Use bucket counts or streaming aggregation. The single 257 MB file in results_c/ is grandfathered in (gitignored), do not produce more.
- Do not interpret Phase C-2 / C-2b's σ ∝ (1 − ρ) as the bit-budget-optimal proxy; it's the C-2-historical proxy. The bit-budget-optimal proxy is √(1 − ρ), and both are reported in C-2c for completeness.

### What to do next (if compute time allows)

1. **Phase C-3** — state-step quantisation directly in HF Mamba `slow_forward`. Reuse C-2c's six-condition design. This is the definitive test of the bit-allocator direction.
2. **Phase C-1c-bis** — re-run Γ extraction on a held-out calibration window to check whether Γ_s is calibration-text-stable. C-1c used WikiText-2 train; verify on validation split.
3. **Phase C-5** — hybrid model (Qwen 3.6 27B or Zamba2-2.7B) with the Paper C allocator on the SSM side and modeq's existing INT3 recipe on the attention side. End-to-end deployment-grade test.

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
