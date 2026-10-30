# Evidence coverage

The repository and companion assets form one versioned evidence package. The repository contains readable metadata, plans, code, summaries, tables, manuscripts and figures. Arrays (`.npy`/`.npz`) and source files larger than 8 MiB are distributed as ZIP assets with their original paths and bytes.

## Selected experiments

| Experiment | Evidence included across repository + assets |
| --- | --- |
| KV01 | K/V split diagnostic, saved inputs/logits, cache traces, original records and verifier source |
| IV01 | Input-validation fixtures, arrays, timing records and fixed implementation |
| VF01 | Bridge-control inputs, reconstructed arrays, timing/fixture records and fixed implementation |
| AC08 | Ouro model integration, raw logits, query/KV captures, timing and cache traces |
| NB05 | Initial Nanbeige integration, 48 saved-logit trajectories, 432 target/NLL records and query diagnostics |
| NB06 | Continuation to 256 predictions, 60 trajectories/15,420 forward records and 32 actual-query captures |

NB06 retained raw FP32 logits only at 8 fixed positions in each of 48 non-warmup trajectories (384 rows). Other positions retain NLL components and logit hashes. Packaging cannot restore logits that the experiment never saved. Technical replicates and overlapping windows are not additional independent texts.

For these six run directories, every retained non-cache file is included except a redundant embedded original ZIP; exclusions are enumerated in [run-exclusions.json](metadata/run-exclusions.json). The package does **not** contain all earlier parent campaigns. A verifier requiring parent archives still needs those inputs. Original failure statuses and fixed numerical criteria are preserved.

## Earlier evidence and paper reproduction

The 135 original JSON results are retained byte-for-byte: 20 are directly reaggregated and 115 remain catalogue-only context. They are grouped in [data/](data/README.md). The current extraction performs 1,730 checks. The previous total of 1,732 included hashes of two historical manuscript bodies; those two non-numerical checks are explicitly removed because those manuscripts are no longer distributed. Every other check and all scientific values retain the same comparisons. These are aggregate or per-window records, not replacement raw token logits. Current paper builders, templates, frozen baseline manuscripts and NB06 supplement specification are included.

[source-layout.json](metadata/source-layout.json) maps original identifiers to the integrated file layout. Original `source` values and baseline extraction records retain their acquisition-time names; use `tools/source_layout.py --resolve ORIGINAL_IDENTIFIER` to locate the current file. No old-named directory or compatibility symlink is needed. [source-files.json](metadata/source-files.json) records each distributed copy's hash, size, original source and any publication-only edits. [release-assets.json](metadata/release-assets.json) lists every external file and its containing ZIP. `tools/prepare_workspace.py` recreates their paths without numerical changes. The repository's `package-files.json` additionally covers publication documentation and tools.

## Materials, provenance and exclusions

Material records retain the PG19 split, selected book identifiers, tokenizer/dataset revisions and actual token sequences. Model outputs and captured arrays are research evidence. Model weights, complete upstream datasets, local environments, credentials and general project diaries are not bundled. Upstream licenses and terms apply to third-party content; see [LICENSES.md](LICENSES.md).

Some archived logs contain original `/content/` or author-local paths, old manuscript contact fields, historical names and dated status statements. They are evidence, not the current execution instructions. Current commands use relative package paths and the root environment. The canonical author contact is **info@metaclan.jp**.

## Publication versus scientific validation

Packaging checks establish file identity, transport restoration and usable references. They do not rerun GPU inference or turn historical within-project verification into independent replication. Current preparation results are recorded in [metadata/publication-validation.json](metadata/publication-validation.json); earlier scientific checks retain their own dates and scope.

## Private workflow exclusions

Internal ledgers, operating records and delivery bookkeeping are excluded. Six prespecified scientific protocols and sixteen numerical-validation/code/publication-metadata files are retained separately, with original bytes and source identifiers. The exclusion does not remove any of the 1,730 retained checks. Current evidence indexes point to the public records; frozen manuscript inputs retain historical link text solely for reproducible generation.
