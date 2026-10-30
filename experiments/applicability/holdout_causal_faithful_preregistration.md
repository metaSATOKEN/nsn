# Hold-out causal-faithful re-evaluation: does the NSN family advantage survive the production-faithful scale?

A pre-registered re-evaluation of the load-bearing hold-out cells under the production-faithful
(causal-faithful) quantiser, answering the post-K1 adversarial-review charge that the "NSN wins
on three of four hold-out models" claim (Section 6.2, Section 7.1) is partly an in-harness
artefact: Section 9 shows the in-harness full-window scale is pessimistic to the plain-asymmetric
(KIVI-style) family by up to 2.0 pp, and the OLMoE in-harness NSN margin (0.98 pp) sits inside
that correction, so whether NSN strictly beats plain asymmetric on OLMoE at deployment scale is
not yet established. This experiment settles it by measuring the hold-out cells under the same
A0-to-A3 causal-faithful protocol Section 9 applied to the calibration cells.

To be committed (locked) before the run. Repo state at lock: `0e37dbd` (Tier-1 review response).

## 1. Claim under test

Section 7.1 reports, in-harness at INT4, that norm separation is the more robust family out of
sample: cleaner than plain asymmetric on Bamba (+0.50% vs +0.99%), OLMoE (+0.89% vs +1.87%), and
Qwen3-30B (+1.06% vs +3.97%), beaten only on Granite. The margins are in-harness. Section 9 found
the A0-to-A3 correction is up to -2.0 pp on the plain-asymmetric family (Gemma4 INT4), so an
in-harness NSN margin smaller than that correction may not survive a production-faithful scale.

**Per cell, the question is whether the paired gap (plain-asymmetric INT4 minus NSN INT4) stays
positive (NSN cleaner) under A3.**

## 2. Pre-registered prediction (locked, genuinely uncertain)

The A0-to-A3 correction shrinks the plain-asymmetric penalty (its full-window scale over-counts
out-of-window outliers), so it narrows the NSN advantage. Modal prediction by margin size:

- **Qwen3-30B (in-harness margin 2.91 pp)**: comfortably larger than the 2.0 pp correction, so
  NSN should remain clearly cleaner under A3 (gap CI lower bound > 0).
- **OLMoE (in-harness margin 0.98 pp)**: smaller than the correction, so the honest modal outcome
  is a **tie** under A3 (gap CI includes 0), i.e. NSN and plain asymmetric are not separable at
  deployment scale on OLMoE.
- **Bamba (in-harness margin 0.49 pp)**: also within the correction; likely a tie under A3.
- **Granite (NSN loses in-harness by 0.51 pp)**: plain asymmetric stays ahead or ties.

So the honest pre-registered expectation is that the in-harness "NSN wins 3 of 4" becomes, under
the production-faithful scale, something closer to **1 clear win (Qwen3-30B) + ties + 1 loss**.
The qualitative paper claim (family preference is model-specific, not class-predictable) does not
depend on the count: it rests on the winner flipping across models at all (it does) and on the
Qwen3-30B margin, which survives the correction. This experiment establishes the honest count.

## 3. Models, cells, configs

- **Primary (load-bearing)**: `allenai/OLMoE-1B-7B-0924` (uniform) and `Qwen/Qwen3-30B-A3B-Base`
  (uniform). These are the two cells the review names; OLMoE is the one inside the correction.
- **Secondary (for the full count)**: `ibm-ai-platform/Bamba-9B-v2` (hotspot) and
  `ibm-granite/granite-3.0-3b-a800m` (uniform), to report all four hold-out models consistently.
- Recipes: `nsn` INT4 and `kivi` (plain per-channel asymmetric) INT4, plus the FP16 baseline;
  V handled by family as in Section 9 (per-token asymmetric, identical across recipes).
- Configs: A0 (in-harness full-window) -> A1 (causal per-group scale) -> A2 (+ FP16 residual
  buffer) -> A3 (+ post-RMSNorm hook on QK-norm models only; others terminate at A2). The same
  four-configuration ablation as Section 9 (`phase_e1` / `phase_w1d`).
- ctx = 1024 (matches the Section 7.1 hold-out measurement context); seed 20260510; N=30
  random windows; N_boot=5000; chunked NLL.

## 4. Success criteria (pre-registered, per cell)

- **NSN strictly cleaner (survives)** if the A3 paired gap (plain minus NSN) CI lower bound > 0.
- **Tie under production-faithful** if the A3 gap CI includes 0.
- **NSN loses** if the A3 gap CI upper bound < 0.

Aggregate: report the production-faithful win/tie/loss count across the four hold-out models, and
state explicitly whether the in-harness "3 of 4" holds, weakens to a smaller clear-win count, or
is unchanged.

## 5. Honest-reporting commitments (locked now)

- Whatever the A3 outcome, Section 6.2 / Section 7.1 are updated to the production-faithful count;
  if OLMoE becomes a tie, the paper says "NSN is clearly cleaner on Qwen3-30B, a tie on OLMoE and
  Bamba under the production-faithful scale, and behind on Granite", not "wins 3 of 4".
- The qualitative model-specific-preference claim is kept only on what survives (the family
  flipping across models + the Qwen3-30B margin); the count is reported as the honest bound.
- Every cell's A0-to-A3 trajectory is released, not only the headline gap.

## 6. Harness and verification (to build next session)

A self-contained paste-and-run file (no git clone; the Colab-code convention) that vendors the
A0-A3 causal-faithful quant from `phase_e1` / `phase_w1d` (bit-identical), the N=30 random-window
bootstrap, and the multi-auto-class loader; cells set to the four hold-out models at INT4; paired
cross-recipe gap with CI per config; GPU-free self-tests at the bottom; save-first JSON to
`results/holdout_causal_faithful.json`. OLMoE and Qwen3-30B are the must-run cells; Bamba and
Granite are run if budget allows.

## 7. Scope

- Decides only the production-faithful family count on the hold-out cells; does not revisit the
  conservative INT5 anchor (settled) or the long-context separation (Section 6.5, settled).
- WikiText-2 only; A0-A3 in-harness-to-production-faithful scale (no kernel).
- ctx=1024 (the Section 7.1 context); a long-context hold-out family check is separate.

## 8. Outcome (filled after the run)

(To be completed after `results/holdout_causal_faithful.json` exists: the A0-A3 trajectory per
cell, the A3 paired gap and CI, the production-faithful win/tie/loss count, and the resulting
Section 6.2 / Section 7.1 wording.)
