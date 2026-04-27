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
