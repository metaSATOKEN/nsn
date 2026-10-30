# NSN: KV Quantization by Separating Norm and Direction

**Configuration, Applicability, and the Cost of Using Compressed Representations**  
Kentaro Sato · Independent Researcher · [info@metaclan.jp](mailto:info@metaclan.jp)

NSN compresses a language model's attention memory (the KV cache) by keeping each vector's magnitude separately and quantizing its direction. The asymmetric configuration uses per-channel minima and step sizes, then renormalizes the reconstructed direction. Its bit width is configurable: the study examines both 3-bit compression and higher-precision 5-bit settings.

This repository connects three questions: **when prediction quality is retained, how much storage is actually saved, and what using the compressed representation costs.** A smaller KV cache does not automatically imply a lower whole-GPU memory peak or a faster complete inference path.

**Read the [English integrated paper](papers/nsn-integrated/NSN-2026-10-29-EN.md) or [Japanese paper](papers/nsn-integrated/NSN-2026-10-29-JA.md).** Both explain NSN from the beginning. The [looped-model study](papers/loop-kv/README.md) provides additional implementation detail.

[Earlier edition / library users](LEGACY.md) · [日本語 README](README.ja.md) · [Run commands](RUNNING.md) · [Reproduction](REPRODUCIBILITY.md) · [Evidence coverage](DATA_COVERAGE.md) · [Licenses](LICENSES.md)

## What the experiments show

| Question | Observation | Evaluation setting |
| --- | --- | --- |
| Which configuration retained quality? | Asymmetric NSN was the only one of four 3-bit configurations below 5% PPL degradation for all 8 tested models in the primary comparison. | Configuration-level comparison; the paper reports additional conditions where this did not hold. |
| Is NSN restricted to 3 bits or dense Transformers? | The precision-oriented 5-bit setting met the quality diagnostic in 4 models with SSM-plus-attention or MoE architectures. | Attention K/V; whole-window QDQ, with 3/4/5-bit comparisons. |
| Does an actual packed cache work in looped models? | Final KV storage fell by 67.68% for Ouro and 64.39% for the extended Nanbeige evaluation. | Four texts per model; 255 and 256 teacher-forced predictions per text, respectively. Metadata and uncompressed tails are included. |
| Does reduced storage mean faster execution? | Fused reconstruction reduced compressed-path costs. In the Nanbeige continuation experiment, initialization and updates offset most forward-pass savings, yielding approximately equal total time to checked uncompressed execution. | Fixed implementation, batch size 1; quality, stored bytes, GPU peaks, and time are separate measurements. |

Here, quality refers to the paper's PPL diagnostic, not a general generation-quality guarantee. The six [integrated figures](papers/nsn-integrated/figures/) and the appendices explain the comparisons and numerical criteria.

The public GPU entrypoints are [readable, expanded Python sources](experiments/README.md). Compressed OneCell files stay in the author's local Colab workflow.

## Start with the level of evidence you need

1. **Read and inspect:** the papers, figures, fixed plans, code, summary tables and most logs are in the Git repository.
2. **Check the download:** run the standard-library hash verifier. This performs no model calls or numerical reanalysis.
3. **Inspect raw arrays or reproduce:** obtain the companion evidence ZIPs, restore a new workspace, and use that workspace's environment. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

```sh
.venv/bin/python -B tools/verify_package.py
```

Use the `.venv` belonging to this repository root. The source research workspace has its own instructions in its root `RUNNING.md`. Do not substitute a local Mac interpreter into the Colab launchers.

## Repository map

| Path | Contents |
| --- | --- |
| `papers/nsn-integrated/` | Current Japanese/English integrated paper, six figures, tables, and generation inputs |
| `papers/loop-kv/` | Current looped-model companion paper and numerical-audit code |
| `src/nsn/` | NSN reference implementation: quantization, storage and packing |
| `data/` | Configuration, applicability, storage and kernel measurements; a separate catalogue of contextual records |
| `experiments/` | Code grouped by the same research questions, plus readable AC08/NB05/NB06 model integrations |
| `runs/` | Six selected experiments' saved metadata and logs; large files are in companion assets |
| `scripts/`, `plans/` | Looped-model analysis implementations and fixed experiment specifications |
| `tools/` | Hash verification, safe workspace restoration, and offline reproduction entrypoints |
| `metadata/` | Source mapping, per-file hashes, asset manifest, and packaging checks |

The repository is organized around the integrated paper's questions rather than the earlier projects. Start with [the experiment index](experiments/README.md) and [the data index](data/README.md). The earlier standalone manuscripts are not bundled; the published NSN paper remains cited below. Frozen integrated-paper baselines are retained only as generator inputs.

Internal workflow ledgers and operating records are not distributed. Required scientific protocols are in `plans/prespecified/`, and numerical checks are in [data/validation](data/validation/README.md).

Acquisition-time source identifiers in original records are resolved through [source-layout.json](metadata/source-layout.json). The mapper verifies unchanged source bytes and does not recreate the old project folders.

## Availability, attribution, and license

The integrated v1.0 package is hosted at [https://github.com/metaSATOKEN/nsn](https://github.com/metaSATOKEN/nsn), with **17 evidence ZIPs and SHA256SUMS in the [integrated-v1.0 Release](https://github.com/metaSATOKEN/nsn/releases/tag/integrated-v1.0)**. The ZIPs total 6.30 GB and remain outside Git history. Download all 17 ZIPs into one directory without renaming them, then follow [REPRODUCIBILITY.md](REPRODUCIBILITY.md). Filenames, sizes, hashes and download URLs are listed in [release-assets.json](metadata/release-assets.json). GitHub's automatic source archives do not include these companion arrays.

**The repository remains private pending the author's public-access decision.** Authorized repository users can access its Release assets. No integrated-paper DOI has been assigned.

The earlier public NSN paper is [*Norm Separation is Necessary*, v1.0.0](https://doi.org/10.5281/zenodo.19724817). The integrated paper combines its reanalysis with subsequent implementation and looped-model evaluations. See [publication provenance](papers/nsn-integrated/publication-provenance-v0.1-2026-10-03.md).

Author-owned code is licensed under [Apache-2.0](LICENSE); author-owned papers, figures and data under [CC BY 4.0](LICENSES/CC-BY-4.0.txt). Third-party material retains its original terms; see [LICENSES.md](LICENSES.md). Citation metadata is in [CITATION.cff](CITATION.cff).

The current PDF exports retain an obsolete contact address and are excluded from this candidate; use the corrected Markdown editions. [Publishing checklist](PUBLISHING.md).
