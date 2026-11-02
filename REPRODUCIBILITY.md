# Reproduction guide

Obtain the 17 companion ZIPs and `SHA256SUMS` from the [integrated-v1.0 Release](https://github.com/metaSATOKEN/nsn/releases/tag/integrated-v1.0). Put them together in `../github-release-assets/` (or pass another directory with `--assets-dir`). The repository and Release currently require authorized access. GitHub's source ZIP/tarball contains the repository only.

Use the research root's `.venv/bin/python`, as described in [RUNNING.md](RUNNING.md). The three operations below have different effects: hash checking reads files, restoration creates a new copy, and scientific reproduction calculates values and writes new outputs.

## 1. Check the repository and companion ZIPs

From a standalone copy with its own root environment:

```sh
.venv/bin/python -B tools/verify_package.py
.venv/bin/python -B tools/verify_package.py --assets-dir ../github-release-assets
```

The second command checks every ZIP's bytes, SHA-256 and member inventory. It does not execute any archived Python. See [release-assets.json](metadata/release-assets.json) for the required filenames. A repository-only PASS does not mean external arrays are present.

## 2. Restore a separate working copy

Choose a destination that does not exist, outside the source repository and asset directory:

```sh
.venv/bin/python -B tools/prepare_workspace.py --assets-dir ../github-release-assets --destination /tmp/nsn-work-new --dry-run
.venv/bin/python -B tools/prepare_workspace.py --assets-dir ../github-release-assets --destination /tmp/nsn-work-new
```

The restorer copies only inventoried repository files, extracts only listed regular ZIP members, and checks each restored hash. Existing destinations are refused. No model, network, scientific calculation or original-file modification is involved.

## 3. Recalculate in that copy

Prepare **that working copy's** root `.venv` from an explicitly chosen Python executable. From its root, review and install the fixed CPU analysis dependencies:

```sh
.venv/bin/python -m pip install -r requirements-offline.txt
.venv/bin/python -B tools/verify_package.py --require-data
.venv/bin/python -B tools/reproduce.py --output-dir /tmp/nsn-reproduced-new
```

The output directory must be new and outside the working copy. The historical analysis dependencies were checked with Python 3.13.5; the source research environment currently uses 3.13.17. Record the actual resolved environment rather than treating interpreter selection as proof of numerical reproducibility.

The command writes its results to the new output directory and compares them to the data and figures distributed with the two papers.

| Output | Comparison |
| --- | --- |
| Evidence extraction | Recompute stored-value comparisons, window-bootstrap summaries and timing summaries against 1,730 recorded checks. |
| NB05 numerical audit | Recalculate 432 target NLLs from 48 saved logit arrays and check the recorded tolerance. |
| Initial tables and figures | Regenerate the initial measurements and five figures for each paper. |
| NB06 continuation | Reaggregate all 15,420 saved loss components. Rebuild both sets of paper data, the paper-level audits and Figure 6. |

`reproduction-report.json` distinguishes byte or numerical equality from PNG pixel equality. The Markdown files in [papers](papers/README.md) are the text to cite. For the original manuscript-generation workflow, use the [fixed v1.0 source and its instructions](https://github.com/metaSATOKEN/nsn/blob/integrated-v1.0/REPRODUCIBILITY.md) with the same evidence ZIPs.

The latest [CPU verification](metadata/publication-validation.json) records the table comparisons and per-figure results.

## Resolve original source identifiers

Run `.venv/bin/python -B tools/source_layout.py` to check the mapped source and scientific-evidence files against their original hashes. For example:

```sh
.venv/bin/python -B tools/source_layout.py --resolve modeq/results/phase18_ablation_vs_industry_naive.json
```

The identifier is provenance, not an expected old directory. The result points to `data/configuration/phase18_ablation_vs_industry_naive.json`. The two analysis readers use this same explicit mapping; numerical loops and tolerances are unchanged.

## Inspect or rerun model experiments

The public distributions are [ordinary, expanded source files](experiments/README.md), not compressed OneCell payloads. The original OneCells remain in the author's private Colab workflow.

```sh
.venv/bin/python -B experiments/nb06/run_colab.py
```

Without `--run`, this checks source hashes only. Actual execution requires the documented Colab/Linux GPU environment, explicit `--run`, model/data access and the experiment-specific fixed dependencies. It downloads dependencies/model files and performs GPU inference. Those dependencies differ from `requirements-offline.txt`. Do not run these large models on a local Mac.

## Limits of the included evidence

The original six run directories are distributed except for execution caches and a duplicate original ZIP. Earlier parent campaigns are not included in full, so verifiers that require them may need additional original inputs. No data are supplied for positions the original experiment did not save: NB06 has raw logits at 384 retained rows, with loss components and hashes elsewhere. Original quality thresholds, failed checks, technical replicates and dependent windows retain their original interpretation. See [DATA_COVERAGE.md](DATA_COVERAGE.md).
