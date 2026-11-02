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

## Earlier measurements and paper data

Of the 135 original result files, 20 supply the reaggregation and 115 provide context in the catalogue. [Data index](data/README.md)

The paper data include the condition-level estimates, their source identifiers and the continuation results. An earlier measured looped-model table is also retained as a fixed input to the integrated analysis. [Papers and data](papers/README.md) · [Source resolution](metadata/source-layout.json) · [Reproduction instructions](REPRODUCIBILITY.md)

The runs retain model settings, sampled material identifiers and token sequences. The upstream model and dataset licenses apply to third-party materials. [Third-party terms](LICENSES.md)
