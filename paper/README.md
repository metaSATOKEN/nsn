# Paper: Norm Separation is Necessary (NSN)

This folder contains the formal preprint for the NSN v1.0.0 INT3 KV cache
quantization recipe (reference implementation: the `nsn` Python package).

**Zenodo DOI:** [10.5281/zenodo.19724817](https://doi.org/10.5281/zenodo.19724817)

## Files

- **[nsn.pdf](nsn.pdf)** — the preprint PDF (27 pages, Zenodo-ready)
- **[nsn.md](nsn.md)** — the authoring source (Markdown + YAML frontmatter,
  built via pandoc → xelatex)
- **[nsn.tex](nsn.tex)** — the Markdown → LaTeX intermediate (for eventual
  arXiv migration once endorsement is obtained)
- **[refs.bib](refs.bib)** — BibTeX references, disambiguated (Wang 2025a /
  2025b, etc.)
- **[refs.md](refs.md)** — the human-written reference source that `refs.bib`
  is mechanically converted from

## About the "NSN" short-name

Throughout the repository, commit messages, and the citation key
(`sato2026nsn`), this paper is referred to by the short-name **NSN**,
after its title *Norm Separation is Necessary: A Minimalist Calibration-Free
INT3 Recipe for KV Cache Quantization*. The title phrase reflects the
paper's central empirical finding: removing the norm-separation step from
the v0.3 recipe causes catastrophic regressions on 3 of 8 benchmark models
(§5.2, §5.3).

## Building

```bash
# assumes pandoc + xelatex on PATH (TinyTeX works)
cd paper
pandoc nsn.md -o nsn.pdf \
    --pdf-engine=xelatex \
    -V mainfont="Times New Roman" \
    -V monofont="Menlo" \
    -V geometry:margin=0.9in \
    --resource-path=.:..
pandoc nsn.md -o nsn.tex
```

## See also

- [../README.md](../README.md) — engineer-facing repo entry point
- [../EXPERIMENT_REPORT.md](../EXPERIMENT_REPORT.md) — long-form writeup
  with dead ends, mechanism notes, and open observations not elevated to
  the paper
- [../docs/THEORY.md](../docs/THEORY.md) — two-failure-modes background
  and the v0.2 → v0.3 update
