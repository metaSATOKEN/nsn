# Paper: Lyapunov-Aware Bit allocation for SSMs (LAB)

This folder contains the formal preprint for the modeq Paper C: a closed-form,
short-calibration bit-allocation strategy for state-space model state
quantisation, derived from a discrete-time Lyapunov noise bound.

**Zenodo DOI:** TBD (assigned at v1.0.0 release)

## What this paper is about

State-space models (SSMs) such as Mamba replace the transformer's linearly
growing key-value cache with a fixed-size recurrent hidden state. Per-step
quantisation noise on that state feeds back into the next step and accumulates
with context length. Existing SSM-quantisation work uses uniform bits across
state groups; we derive a per-state-group bit allocation rule from the
discrete-time Lyapunov bound on accumulated quantisation noise.

The bit-allocation surface

```
R_g = Γ_g / (1 - ρ_g)
```

combines the recurrence-stability factor `(1 - ρ_g)^-1` with the output-
coupling magnitude `Γ_g`, both calibrated per state group from a single
1024-token forward pass. The recommended recipe `half_slope_rg` is the
closed-form rounded integer realisation of the continuous bit-budget optimum
on this surface, retaining a per-group range correction `r_g`:

```
b_g = b_0 + ⌈ ½ log_2 R_g + log_2 r_g − mean(log_2 r_g) ⌉_+
```

## Headline results

On the state-spaces Mamba-1 family (130M to 2.8B), at matched mean bit width
under packed-state memory accounting paired with fake-quant perplexity:

- **2.6× state-cache compression at +2.5% to +5.2% perplexity** at the
  moderate-memory budget `b_0 = 6`.
- **Aggressive INT4 memory budget viable** across all five sizes at
  **+5.9% to +19.3% ΔPPL**, where the slope-1 simplification produced
  +22% to +59% on the same models.
- Lyapunov-vs-range factorisation on Mamba-2.8B-hf argues against a
  range-aware-only explanation: stripping the `R_g` signal collapses ΔPPL
  to +124% at `b_0 = 4`, and randomising the per-group placement of either
  `R_g` or `r_g` collapses the advantage at matched mean bits.
- σ²-scaling test: the strict-optimum profile σ_g ∝ R_g^(-1/2) is the only
  tested profile with linear-response behaviour (empirical slope = 1.94 vs
  the σ² ideal of 2).

Side benefit on Mamba-2.8B-hf at PG-19 sliding windows up to 32K context:
the calibrated range `r_g` additionally clips state-magnitude excursions, so
the calibrated-clipping path stays at perplexity ≈ 11 where unquantised FP16
diverges to ≈ 197. This extends the small-scale Edge-AI state-clipping
recipe of [Zhao et al., 2025] to LLM-scale Mamba long-context inference.

## Files

- **[paper_c.pdf](paper_c.pdf)** — the preprint PDF (~36 pages, Zenodo-ready)
- **[paper_c.md](paper_c.md)** — authoring source (Markdown + YAML frontmatter,
  built via pandoc → xelatex)
- **[Makefile](Makefile)** — pandoc + xelatex build, with delegate targets
  for figure regeneration
- **[figures_c/](figures_c/)** — 6 figures (PDF + PNG) with reproducible
  scripts and a shared `_style.py` module
- **[results_c/](results_c/)** — raw JSON results from each experimental
  phase (calibration outputs, sweep data, trajectory recordings)
- **[experiments_c/](experiments_c/)** — Python scripts for each phase
  (drop-in runnable in Colab; one script per phase)
- **[EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md)** — long-form research log:
  16 phases with why-we-ran-it / methodology / results / verdict, plus
  open / closed lists, dead ends, and a handoff brief for cold-start sessions

## About the "LAB" short-name

Throughout the repository, commit messages, and the citation key
(`sato2026lab`), this paper is referred to by the short-name **LAB**, after
*Lyapunov-Aware Bit-allocation*. This complements the Paper B short-name
**NSN** (*Norm Separation is Necessary*); together they form the modeq
two-paper preprint set on cache-state quantisation.

## Building

```bash
# requires pandoc + xelatex (TinyTeX works) and python3 + matplotlib + numpy
cd paper_c
make pdf       # rebuild paper_c.pdf
make figures   # rebuild all 6 figures (PDF + PNG)
make all       # both
```

## See also

- [../paper/README.md](../paper/README.md) — modeq Paper B (NSN) preprint
- [../README.md](../README.md) — engineer-facing repo entry point
- [EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md) — long-form research log
