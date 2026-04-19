# modeq figures

Publication-quality figures generated from the JSON files in
[`../results/`](../results). Regenerate with:

```bash
python3 figures/make_figures.py
```

Each figure is produced as both `.png` (@ 160 DPI, for web / README) and
`.pdf` (vector, for paper inclusion). Colors and typography are shared
across figures.

---

## fig01_int4_valley — "why modeq exists"

Naive per-row absmax INT quantization across 7 models and 6 bit widths,
log y-axis. Source: companion paper V5 top-of-README sweep.

Reading: **6 of 7 models scale monotonically** with bit width — more
bits, smoothly less ΔPPL. The **Qwen2-7B** curve (red) is the
exception: **+1382 ΔPPL at INT4** vs +12 at INT3 and +5 at INT5. This
non-monotonic spike (which we informally call the "INT4 valley") is
the one observation in our 27-model survey that motivates modeq's
mechanism work. It is **n = 1** in our sample; we do not claim the
pattern generalizes to untested models.

Good for: blog intro, paper motivation, talk opening slide. Pair with
the n = 1 caveat in the main [README](../README.md#the-common-case-and-one-observed-special-case)
when used in a published context.

## fig02_8model_headline — "v0.3 at a glance"

Paired bars of v0.2 symmetric INT3 vs v0.3 asymmetric INT3 ΔPPL, as
**% of each model's FP16 baseline PPL**, across 8 models (Phase 14 +
Phase 16 + Phase 17). Dashed line at 5 % marks our modeq-internal
usability threshold (see [README §"What we mean by the ✅ usable
label"](../README.md#what-we-mean-by-the--usable-label)).

Reading: every model drops below the 5 % line under v0.3 (right-hand
blue bar). Per-model reduction % annotated above each pair. Models are
sorted by severity of v0.2 drift, so the left side is where v0.3 helps
most visibly (Pythia-6.9B moves from 22 % to < 1 %).

Good for: Twitter headline, README hero figure, paper's main table in
graphical form.

## fig03_ablation — "each component is necessary"

Four-method INT3 ablation from Phase 18 + Phase 19, 8 models, log y.
Methods (A) naive per-row, (B) asymmetric per-channel *without* norm
separation, (C) modeq v0.2 symmetric, (D) modeq v0.3 asymmetric. Cross /
exclamation-mark annotations mark where each method crosses the
usability threshold.

Reading: only the v0.3 composition (D, blue) stays under 5 % on every
model. Method (B) — the weight-quantization-style recipe — fails
catastrophically on 4 of 8 models, which is the empirical content of
"norm separation is structurally necessary for KV cache per-channel
quantization". Bars near the bottom (Qwen3.5, Gemma 4 under method B)
are the negative-ΔPPL anomaly probed separately in Phase 20 + 21.

Good for: paper ablation section, reviewer-facing README.

## fig04_bits_vs_quality — "Pareto view"

Bits-per-value (x) against ΔPPL as % of baseline (y, log). Each method
gets its own column at the 3-bit tier; INT4 reference squares appear at
the 4-bit tier for the two models where we measured it (Qwen2-7B,
Llama-3.1-8B in Phase 14). Per-model small dots show the spread; the
large circle is the cross-model median.

Reading: at 3 bits, (D) v0.3 asym is the only method whose cross-model
median sits clearly below the 5 % usability line. Going to 4 bits
(green squares) improves absolute quality further but costs 33 % more
bits; (D) at 3 bits is the practical Pareto choice when memory is
the dominant constraint.

Good for: summary figure in a deck, one-shot explanation of "why INT3
and not INT4".

---

## Reproducibility

All four figures are regenerable by running `make_figures.py` against
the committed `results/*.json`. The only externally-sourced data is the
bit-width sweep in `fig01` (7 models × 6 bits), which is reproduced at
the top of the main `README.md` and sourced from the companion paper's
Phase 1–8 experiments.

If you modify any quantizer in `src/modeq/quantize.py` and re-run the
Colab experiment scripts in `experiments/`, updated JSONs will land in
`results/` and the figures will regenerate with the new numbers without
any further code changes.
