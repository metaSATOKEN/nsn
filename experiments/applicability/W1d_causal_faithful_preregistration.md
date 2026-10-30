# W1d: does the long-context family separation survive the production-faithful scale?

A pre-registered causal-faithful re-evaluation of the §6.5 load-bearing cells, deciding
whether the long-context recipe-family separation (NSN-INT5 in band, plain-asym-INT5 out
of band) is real or an artefact of the in-harness full-window (A0) scale. Directly
answers the adversarial-review charge (round 7, W1) and the deciding follow-up §6.5
already registers.

The charge: §6.5 reports the family separation on the full-window A0 scale, but §9
establishes that the A0 scale is pessimistic to the plain-asymmetric (KIVI-core) family
by up to 2.0 pp on Gemma4 once the hook position is matched (A0 to A3), while the NSN side
is essentially unmoved (-0.13 pp on the 8K NSN-INT5 cell). The 26B-A4B separation,
+1.26 to +1.53 pp, sits inside that correction. So under the production-faithful scale the
plain-asymmetric recipe could return to the band and the separation could vanish. This
experiment measures it.

Committed before the measurement; repo at `949cb7a` (the hedged state).

## 1. Claim under test

Notation: A0 = full-window scale (the §6.5 measurement); A1 = causal per-group scale; A2
= A1 + FP16 residual buffer; A3 = A2 + post-RMSNorm hook (the production-faithful config
on QK-norm models, which Gemma4 is). Quant bit-identical to `phase_e1`. The clean band is
+/-0.5% delta-PPL. The cross-recipe gap is the paired per-window (plain-asym-INT5 minus
NSN-INT5) delta-PPL, both measured on the same N=30 random windows, under one config.

**H-survive (the separation is real).** At A3, on each load-bearing cell, the cross-recipe
gap has a 95% bootstrap CI lower bound above 0 (plain-asym-INT5 is still significantly
worse than NSN-INT5), AND plain-asym-INT5 is still out of band (its A3 marginal CI lower
bound above +0.5%) while NSN-INT5 stays near-lossless. If this holds, the §6.5 family
separation survives the production-faithful scale and is a genuine finding.

**H-collapse (the separation is an A0 artefact).** At A3, the cross-recipe gap CI includes
0 or favours plain-asym, OR plain-asym-INT5 returns to the band. If this holds on the
load-bearing cells, the §6.5 separation cannot be distinguished from the in-harness scale
artefact §9 identifies, and §6.5 is downgraded (Section 5).

## 2. Pre-registered prediction (locked, honest, genuinely uncertain)

From the §9 anchor (KIVI-INT4 A0 to A3 shift = -2.00 pp at 8K) the INT5 shift is expected
smaller (INT5 is gentler and the hook-position effect scales with quant coarseness), but
its magnitude at long context is unknown. So:

| cell | A0 gap (plain-asym minus NSN, measured) | predicted A3 outcome |
|---|---|---|
| 26B-A4B 8K | +1.53 pp | at risk: if the plain-asym A0->A3 shift exceeds ~1.5 pp the gap closes |
| 26B-A4B 32K | +1.26 pp | most at risk: the smallest gap, inside the INT4-scale correction |
| 12B 64K | ~2.2 pp (marginal CIs non-overlapping) | more likely to survive: gap exceeds the 2.0 pp anchor |

Predicted modal outcome: a split, the 12B 64K surviving and the 26B-A4B 32K marginal or
collapsing. We do not predict a clean all-survive or all-collapse, and we report each cell.

## 3. Models, cells, configs

- Gemma4-26B-A4B-base at ctx 8K and 32K (8K reproduces the `e1` NSN-INT5 8K A0 cell,
  +0.105%, as a harness cross-check, and carries the largest A0 gap; 32K is the §6.5
  headline and the most vulnerable gap).
- Gemma4-12B at ctx 64K (the extreme cell, larger A0 gap).
- Recipes per cell: `nsn` INT5 and `kivi` (plain per-channel asym) INT5, V handled by
  family as in `phase_e1` / §6.5 (NSN family V = NSN; plain-asym family V = per-token).
- Configs A0, A1, A2, A3 per recipe. E/M/S held FP16.
- N=30 random windows, seed 20260510, N_boot=5000, chunked NLL (per-token identical to the
  full upcast, removes the long-context logits OOM).

## 4. Success criteria (pre-registered, per cell)

- **survive** if the A3 cross-recipe gap CI lower bound > 0 AND plain-asym-INT5 A3 marginal
  CI lower bound > +0.5%.
- **collapse** if the A3 gap CI includes 0 or is negative, OR plain-asym-INT5 A3 marginal CI
  upper bound < +0.5% (back in band).
- **marginal** otherwise (gap positive but plain-asym near the band edge), reported as such.

Aggregate verdict for §6.5: the separation is upheld only on the cells that **survive**;
any **collapse** on a load-bearing cell triggers the Section 5 downgrade for that claim.

## 5. Honest-reporting commitments (locked now, before the result)

- If the load-bearing cells collapse, §6.5 is downgraded to an exploratory observation that
  cannot exclude the in-harness scale artefact, Contribution 6 is withdrawn, and the
  abstract/intro long-context family sentences are removed, keeping only the established
  anchor-robustness result (NSN-INT5 near-lossless across context, which does not depend on
  the plain-asym comparison). We commit to this now.
- If they survive, the §6.5 hedge is lifted to a confirmed in-band-vs-out-of-band claim at
  the production-faithful scale, with the W2 attribution kept (norm separation's value, not
  K-localisation routing; §7.1).
- A split is reported as a split: the separation holds at the contexts that survive and is
  not established at those that collapse.
- The A3 config is the same production-faithful standard §9 applies to the calibration
  cells; no per-cell redefinition of the band or the configs.

## 6. Harness and verification

`phase_w1d_causal_faithful_longctx.py`: the `phase_e1` A0-A3 quant and hooks (including the
A3 post-norm path), the `phase_w1b` chunked NLL and N=30 random-window bootstrap, and the
`phase_w1c` multi-auto-class loader for the 12B unified model. Cross-recipe gaps computed
per config on the shared windows. Quant vendored bit-identical from `phase_e1`; GPU-free
toy-tensor self-tests at the bottom. Output `results/w1d_causal_faithful_longctx.json`,
save-first after every cell. `analyze_w1d.py` (or inline) scores Section 4.

## 7. Scope

- Decides §6.5 only; does not revisit the conservative-anchor robustness (established and
  scale-stable on the NSN side) or any other section.
- WikiText-2 only; in-harness vs production-kernel magnitudes remain the §10 boundary (A3
  is the production-faithful *scale*, still in the forward-hook harness, not a kernel).
- ctx beyond 64K is out of budget on these models (§6.5).

## 8. Outcome (filled after the run)

Run: `results/w1d_causal_faithful_longctx.json` (all three cells, no errors; 64K fit at
93 GB). **H-survive holds on all three load-bearing cells.** The round-7 W1 charge is
refuted by data.

| cell | plain-asym A0 to A3 | NSN A0 to A3 | gap at A3 | verdict |
|---|---|---|---|---|
| 26B-A4B 8K  | +1.638% to +1.273% (shift -0.365 pp, OUT) | +0.105% to -0.024% (in) | +1.30 pp [+0.99, +1.61] | survive |
| 26B-A4B 32K | +1.492% to +1.234% (shift -0.258 pp, OUT) | +0.236% to +0.074% (in) | +1.16 pp [+1.02, +1.31] | survive |
| 12B 64K     | +2.627% to +2.282% (shift -0.345 pp, OUT) | +0.407% to -0.243% (in) | +2.53 pp [+2.01, +3.14] | survive |

The mechanism: the reviewer extrapolated the §9 KIVI-INT4 A0-to-A3 correction (-2.00 pp) to
INT5. Measured, the plain-asymmetric INT5 correction at long context is only -0.26 to
-0.37 pp (INT5 is gentler, so the hook-position effect is much smaller), far short of the
gap (+1.16 to +2.53 pp), so the separation survives. A bonus: the 12B NSN-INT5 anchor,
at the band edge under A0 (+0.41%), is clearly in band under A3 (-0.24%); the
production-faithful scale helps the anchor, since the A0 edge reading was the pessimistic
full-window artefact.

**Action taken (per §5):** the §6.5 hedge is lifted to a confirmed claim at the
production-faithful scale, with the W2 attribution kept (norm separation's value, §7.1);
the abstract, intro, contribution 6, §6.4, §6.5, §8.1, §8.2, §9, and Figure 3 are updated
accordingly. The result is also a long-context extension of the §9 causal-faithful table.
