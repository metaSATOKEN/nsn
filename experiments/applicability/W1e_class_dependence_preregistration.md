# W1e: is the long-context family separation K-localisation-class-dependent?

A pre-registered test of whether the §6.5 long-context recipe-family separation (plain-asym
INT5 leaves the band, NSN-INT5 stays in it) is predicted by the K-localisation diagnostic
class, or is a general property of norm separation. Answers the round-8 adversarial-review
charge: §6.5 honestly attributes the load-bearing factor to norm separation itself (Paper
B), not the K-localisation classification, so as stated §6.5 supports the recipe rather than
this paper's diagnostic. If the separation occurs only on the hotspot class and not on
weak-local or uniform models, the K-localisation classification predicts when the recipe
family matters at long context, and the result becomes a property of the diagnostic.

Committed before the measurement; repo at `8330886` (the §6.5 un-hedged + reframed state).

## 1. Claim under test

The §6.5 separation is established on two **hotspot** Gemma4 models (W1d): at A0, the
cross-recipe gap (plain-asym-INT5 minus NSN-INT5) is +1.53 pp at 8K and +1.26 pp at 32K on
the 26B-A4B, plain-asym out of band, NSN in band. W1e measures the same A0 long-context
family comparison on a **uniform** and a **weak-local** model and compares the gap across
K-localisation classes.

**H-class-dependent (the diagnostic is load-bearing).** The separation scales with the
K-localisation class: the uniform model does not separate at long context (plain-asym-INT5
stays in band, cross-recipe gap CI includes 0 or stays well inside the band), while the
hotspot Gemma4 does (gap +1.26 pp at 32K). The weak-local model is intermediate. If this
holds, the K-localisation classification predicts when the recipe family matters at long
context, so §6.5 is a property of the diagnostic, and Contribution 6 is a diagnostic result.

**H-class-independent.** The uniform and weak-local models also separate at long context
(plain-asym-INT5 out of band, gap comparable to the hotspot), so the separation does not
depend on the K-localisation class. If this holds, §6.5 is a general long-context property
of norm separation (a Paper B extension), and Contribution 6 is reframed accordingly,
keeping its honest §6.5 attribution.

## 2. Pre-registered prediction (locked, genuinely uncertain)

The hotspot structure (concentrated K fragility) is what the plain-asymmetric full-window
scale is expected to fail on at long context, which predicts H-class-dependent. But §7.1
finds norm separation the more robust family out of sample regardless of class at the
aggressive tier, which predicts some NSN advantage even on uniform models, i.e. partial
H-class-independence. The honest modal prediction is a **monotone relationship with
K_spread**: gap(hotspot Gemma4 +1.26 pp at 32K) > gap(weak-local Qwen3.6) > gap(uniform
Qwen3-30B-A3B), with the uniform gap small enough that plain-asym-INT5 is at or inside the
band. A clean uniform non-separation is the strong outcome; a uniform separation as large
as the hotspot is the null (H-class-independent).

## 3. Models, cells, configs

- `Qwen/Qwen3-30B-A3B-Base`: **uniform** (K_spread 0.79), head_dim 128 (non-256, so a
  separation here also dents the Gemma-head_dim-256-specificity concern), 48 layers,
  max context 32768.
- `Qwen/Qwen3.6-27B`: **weak-local** (K_spread 1.04), head_dim 256, 64 layers, the
  long-context hero of the testbed (measured to 8K previously).
- Anchor (not re-run): Gemma4-26B-A4B-base, **hotspot** (K_spread 3.98), from W1d.
- Contexts: 16K and 32K (both models support 32K; matched to the Gemma4 long-context cells).
- Recipes: `nsn` INT5 and `kivi` (plain per-channel asym) INT5, plus the FP16 baseline,
  V handled by family as in W1d.
- Config: A0 full-window (the class-dependence is an A0-vs-A0 cross-model comparison; the
  A0 separation is already shown to survive A3 on the hotspot in W1d, and a uniform that
  does not separate at A0 separates even less at A3, which is cleaner for plain-asym).
- N=30 random windows, seed 20260510, N_boot=5000, chunked NLL.

## 4. Success criteria (pre-registered, per model)

- **separates** if the A0 cross-recipe gap CI lower bound > 0 AND plain-asym-INT5 is out of
  band (A0 marginal CI lower bound > +0.5%).
- **does not separate** if plain-asym-INT5 is in band (A0 marginal CI upper bound < +0.5%)
  OR the gap CI includes 0.

Aggregate verdict: **H-class-dependent holds** if the uniform model does not separate while
the hotspot does (and the weak-local is intermediate or does not separate); the gap ordered
by K_spread is the supporting evidence. **H-class-independent holds** if the uniform model
separates comparably to the hotspot.

## 5. Honest-reporting commitments (locked now)

- If H-class-dependent: §6.5 and Contribution 6 are upgraded to a diagnostic result (the
  K-localisation classification predicts when the recipe family matters at long context),
  with the cross-class gap ordering as evidence; abstract / intro / §8.2 updated.
- If H-class-independent: §6.5 is positioned as a general long-context property of norm
  separation (a Paper B extension), Contribution 6 keeps its honest §6.5 attribution and
  does not claim a diagnostic-routing value; the cross-class result is reported as the
  honest bound either way.
- A monotone-but-not-clean outcome (uniform separates, but less than hotspot) is reported as
  partial class-dependence, neither over- nor under-claimed.
- Every model's result is reported; the uniform null is not buried.

## 6. Harness and verification

`phase_w1e_class_dependence.py`: vendored from `phase_w1d` with the config set restricted to
A0, the cells set to the uniform and weak-local models, and the per-model separation verdict;
chunked NLL and the N=30 random-window bootstrap from `phase_w1b`; the multi-auto-class loader
from `phase_w1c`. Quant bit-identical from `phase_e1`. GPU-free self-tests at the bottom.
Output `results/w1e_class_dependence.json`, save-first after every recipe.

## 7. Scope

- Decides only the attribution of §6.5 (diagnostic property vs recipe property). Does not
  revisit the production-faithful survival (W1d, settled) or the conservative anchor.
- WikiText-2 only; A0 in-harness scale (the class comparison is scale-internal).
- Two new models (one per non-hotspot class); a fuller class sweep is out of budget here.

## 8. Outcome (filled after the run)

Run: `results/w1e_class_dependence.json` (no errors). **H-class-independent holds; the
K-localisation class does not predict the separation.**

| model | class | K_spread | A0 gap (plain minus NSN) | plain-asym band | verdict |
|---|---|---|---|---|---|
| Gemma4-26B-A4B (anchor) | hotspot | 3.98 | +1.26 pp (32K, W1d) | out | separates |
| Qwen3-30B-A3B | uniform | 0.79 | +0.97 pp (16K), +1.03 pp (32K) | out | **separates** |
| Qwen3.6-27B | weak-local | 1.04 | -0.30 pp (16K), +0.12 pp (32K) | in | **no separation** |

The separation is non-monotone in K_spread (the uniform model at 0.79 separates, the
weak-local at 1.04 does not), so the K-localisation class does not predict it: **H-class-
dependent is refuted**, the §6.5-elevation hypothesis does not hold. The pattern instead
tracks the §7.1 model-specific quantiser-family preference: Qwen3-30B-A3B prefers norm
separation (§7.1: NSN cleaner by +2.91 pp at INT4, ctx=1024) and separates at long context;
Qwen3.6 prefers plain asymmetric (§6.2 / §7.1) and does not. The result is the long-context
manifestation of §7.1 (family preference is model-specific, not class-predictable) at INT5.

**Two gains beyond the negative.** (1) The separation reproduces on a **non-Gemma,
head_dim-128** model (Qwen3-30B-A3B), so it is not specific to the Gemma family or head_dim
256 (a partial answer to the round-8 generality charge). (2) The Qwen3.6 boundary bounds the
margin honestly to norm-separation-preferring models.

**Action taken (per §5, H-class-independent path):** §6.5 keeps the honest attribution
(a property of norm separation, not the K-localisation diagnostic); Contribution 6 does not
claim a diagnostic-routing value. §6.5, §8.2, and Contribution 6 are updated to report the
cross-class outcome, the §7.1 manifestation, and the non-Gemma generality. The §6.5-to-
diagnostic elevation is reported as not supported.
