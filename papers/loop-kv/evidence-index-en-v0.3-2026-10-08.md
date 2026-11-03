# Supplement S1: Evidence Index

October 8, 2026 / English S1 v0.3, corresponding to [English manuscript v1.0](manuscript-en-v1.0-2026-10-08.md) and [Japanese S1 v0.5](evidence-index-v0.5-2026-10-08.md). Evidence for the initial two-model comparison is retained; §12 adds the NB06 continuation evaluation.

Publication metadata updated October 29, 2026: licenses, author declarations and navigation are synchronized with the current package. Measured values and scientific conclusions are unchanged from the original edition.

This index is an entry point to retained artifacts in the release candidate. Code, summaries, and readable logs are in the repository; large saved arrays are in companion ZIPs (see [coverage](../../DATA_COVERAGE.md)). Prespecified plans and original records are preserved; paper-specific presentations do not overwrite them. The repository is [https://github.com/metaSATOKEN/nsn](https://github.com/metaSATOKEN/nsn); [README](../../README.md) records its access status and the availability of companion ZIPs. A permanent archive DOI for the integrated paper has not been assigned. Linked historical reports retain their original language, primarily Japanese.

## 1. Entry Points for the Main Results

|Paper content|Result report|Original artifacts|Recalculation with a separate implementation within the project|
|---|---|---|---|
|Ouro: four texts not previously used in this study; Tables 3/4/7/8 and Figures 2/4/5|[AC08](../../reports/ac08-results-v0.1-2026-10-02.md)|[Run](../../runs/loop-kv-ac08-20261002-095022-41c2554c/)|[Audit](../../data/validation/ac08/)|
|Nanbeige: four texts not previously used in this study; Tables 3/4/7/8 and Figures 2/4/5|[NB05](../../reports/nb05-results-v0.1-2026-10-03.md)|[Run](../../runs/loop-kv-nb05-20261003-090604-18baabf5/)|[Audit](../../data/validation/nb05/)|
|Same-arithmetic bridge control; Table 5 and Figures 1/3|[VF01](../../reports/vf01-results-v0.1-2026-10-02.md)|[Run](../../runs/loop-kv-vf01-20261002-072452-1d1c10c0/)|[Audit](../../data/validation/vf01/)|
|Consolidated input validation; Table 6|[IV01](../../reports/iv01-results-v0.1-2026-10-01.md)|[Run](../../runs/loop-kv-iv01-20261001-142907-0c002415/)|[Audit](../../data/validation/iv01/)|

Each run's `summary.json`, `run_status.json`, manifest, and SHA256 list provide entry points. Audit directories document result import, comparison with code retained before distribution, recalculation, and correspondence to formal values. Verification did not consist of blindly executing code from the returned ZIP.

## 2. Claims, Tables, and Data Fields

|Paper content|Machine-readable source|Fields and units used|
|---|---|---|
|Table 4 / Figure 2 left: Ouro memory|[AC08 descriptive statistics](../../reports/ac08-descriptive-statistics.json)|`terminal_storage`, `allocated_peak`, `reserved_peak`, and `recorded_prefill_inclusive_peak` under `memory`; bytes converted to MiB|
|Table 4 / Figure 2 right: Nanbeige memory|[NB05 descriptive statistics](../../data/validation/nb05/supplementary-analysis.json)|Each `books[].terminal_kv_bytes` and `memory`; identical across texts and replicates|
|Table 5 / Figure 3: bridge control|[VF01 analysis](../../data/validation/vf01/analysis.json)|`main_time_ms`, `timing.cuda_ms.*.main`, and `main_memory_bytes`; 6 main pairs from 2 distinct texts|
|Table 7 / Figure 4: per-text quality|[Paper data](data/paper-data-v0.3-2026-10-08.json)|`quality`; 255 predictions and the last 64 for Ouro, 8 post-compression predictions for Nanbeige; PPLs are not averaged directly|
|Table 8 / Figure 5 left: Ouro time ratios|[AC08 descriptive statistics](../../reports/ac08-descriptive-statistics.json)|`time_ratio_summary` and `window_time_ratio_summary`; median of 8 paired ratios|
|Table 8 / Figure 5 right: Nanbeige time ratios|[NB05 descriptive statistics](../../data/validation/nb05/supplementary-analysis.json)|`timing.cuda_ms.decode` / `phase_sum`; median of 3 paired ratios per text, then median across 4 texts|
|Table 6: validation cost|[IV01 descriptive statistics](../../reports/iv01-descriptive-statistics.json)|Representative values and paired ratios for 18 saved model inputs; 192 decisions on 64 fixtures are not independent texts|
|Formulas and actual packed size|[NSN arithmetic](../../scripts/loop_kv_nb05_primitives.py), [codec](../../scripts/loop_kv_nb05_codec.py)|64×128 block; 3072 code bytes + 640 metadata bytes + 12 header bytes = 3724 bytes; summed over K/V, all heads and slots, and BF16 tails|

Figure 1 is a schematic of the readout paths. Figures 2–5 are generated from extracted paper data. Figure 2 uses the uncompressed value of each memory metric as its denominator. Figure 3 shows path-level values, Figure 4 per-text PPL changes, and Figure 5 paired time ratios aggregated at prespecified units. Source SVGs are retained in `figures/`. All five figures already have English labels and are shared unchanged with the Japanese version; captions have been translated and checked against the figures. “Decode” in the figures refers to model decoding; “restore” refers to KV reconstruction. “Checked native” denotes the checked uncompressed path.

The extraction and plotting code is [build_paper_assets_v0_2.py](code/build_paper_assets_v0_2.py). Hashes of 8 source files are included in the paper JSON. Figure generation maps previously verified values to the paper; it is separate from rechecking raw logits. The CPU cross-check of Nanbeige logits conducted for Japanese v0.4 is described in Section 8.

## 3. Prespecified Plans and Recalculation

- **AC08:** [Design](../../plans/loop-kv-ac08-design-v0.1-2026-10-02.md) / [prespecification card](../../plans/prespecified/loop-kv-ac08-v0.1-2026-10-02.md) / [verifier](../../scripts/verify_loop_kv_ac08.py).
- **NB05:** [Design](../../plans/public/loop-kv-nb05-design-v0.1-2026-10-03.md) / [prespecification card](../../plans/public/loop-kv-nb05-v0.1-2026-10-03.md) / [verifier](../../scripts/verify_loop_kv_nb05.py).
- **VF01:** [Design](../../plans/public/loop-kv-vf01-design-v0.1-2026-10-02.md) / [prespecification card](../../plans/public/loop-kv-vf01-v0.1-2026-10-02.md) / [verifier](../../scripts/verify_loop_kv_vf01.py).
- **IV01:** [Design](../../plans/loop-kv-iv01-design-v0.1-2026-10-01.md) / [prespecification card](../../plans/prespecified/loop-kv-iv01-v0.1-2026-10-01.md) / [verifier](../../scripts/verify_loop_kv_iv01.py).

Numerical thresholds are retained in the contract and numerical modules imported by each verifier and in the input JSON, and are checked against distribution-time hashes. AC08 agreement with official incremental execution, VF01 bridge agreement, and NB05 GQA error budgets are distinct conditions. Their success does not overwrite the earlier unmet BF16 full-input/incremental-input agreement criterion.

For reaggregation, preserve the original run, inspect verifier side effects, and use a working copy. In particular, the AC08 verifier CLI writes `completion_verification.json`, so it should not be run directly on original evidence. To regenerate only paper tables and figures, run from the research root:

```bash
.venv/bin/python -B tools/reproduce.py --output-dir /tmp/nsn-results
```

Use a fresh restored workspace as described in the [reproduction guide](../../REPRODUCIBILITY.md).

## 4. Distinctions Between Numerical Checks

- **Original values:** Original summaries, logits, caches, schedules, and stopping records are unchanged.
- **NB05 derived floating-point values:** The original recalculation differed in 22 fields by at most 3.553e−15, within the prespecified relative/absolute tolerance of 1e−12. This differs from exact agreement of discrete values or raw arrays.
- **VF01:** H2 (numerics) and H3 (time) were met. H1 remains unmet because of the incorrectly constructed noncontiguous fixture. [NC01](../../reports/nc01-results-v0.1-2026-10-02.md) is a separate follow-up check.
- **AC08:** The actual-query bridge and fused paths agreed, but differences from the existing reconstruction routine remained. Bitwise agreement of entire logit trajectories across methods is not substituted for the quality criterion.
- **NB05:** Meeting stage-specific budgets at 16 captured points does not mean bitwise agreement with uncompressed attention.

## 5. Index of Preliminary Work and Failure Records

These records explain and trace the conditions leading to the main results. They are not pooled into the paper's main sample.

|Report|Content|
|---|---|
|[P0-DIAG01](../../reports/p0-diag01-results-v0.1-2026-09-30.md)|10 diagnostic conditions completed; no BF16 candidate passed|
|[P0-FP32-01](../../reports/p0-fp32-01-results-v0.1-2026-09-30.md)|FP32 reference: all 16 conditions across 4 books and 4 recurrences met the criterion|
|[P0R-P1A](../../reports/p0r-p1a-results-v0.1-2026-09-30.md)|BF16 path validation passed; 64 quantization conditions at 4 recurrences completed|
|[P0RB-P1B](../../reports/p0rb-p1b-results-v0.1-2026-09-30.md)|Within the 5% quality threshold at 1, 2, and 8 recurrences|
|[KV01](../../reports/kv01-results-v0.1-2026-09-30.md)|K-side PPL loss was somewhat larger; mean losses for the two sides were approximately additive|
|[LP01](../../reports/lp01-results-v0.1-2026-09-30.md)|Between-recurrence differences at equal compressed volume fell within the fixed mean-difference range|
|[LC01](../../reports/lc01-results-v0.1-2026-10-01.md)|Continuous compression remained within the quality threshold at 2,048 tokens; interval differences were not uniform|
|[LC02](../../reports/lc02-results-v0.1-2026-10-01.md)|The continuous-compression quality threshold was also met on 8 books not previously used in this study|
|[PK01](../../reports/pk01-results-v0.1-2026-10-01.md)|Actual 3-bit storage, prediction, timing, and an unmet numerical-check criterion|
|[PD01](../../reports/pd01-results-v0.1-2026-10-01.md)|Standalone 3-bit unpacking was costly; grouping validation waits changed time by approximately 5%|
|[PU01](../../reports/pu01-results-v0.1-2026-10-01.md)|Consolidated 3-bit unpacking preserved values and storage while shortening the model path by approximately 18.5%|
|[PU02](../../reports/pu02-results-v0.1-2026-10-01.md)|Value agreement and approximately 19% time reduction persisted through 3 compression updates|
|[RF01](../../reports/rf01-results-v0.1-2026-10-01.md)|Reconstructed values and checks were preserved, with a further approximately 15.7% time reduction relative to the preceding implementation|
|[QK01](../../reports/qk01-results-v0.1-2026-10-01.md)|K reconstruction and score fusion completed normally; component time decreased by approximately 32.7%|
|[AC01](../../reports/ac01-results-v0.1-2026-10-01.md)|Stopped at the softmax audit after all measurements; not a type error|
|[AD01](../../reports/ad01-results-v0.1-2026-10-01.md)|Small FP32 differences reproduced; warmed-up fused attention was 20.2% faster than the nonfused path|
|[AC02](../../reports/ac02-results-v0.1-2026-10-01.md)|After warming up all lengths, model time decreased by approximately 7.6–7.7% relative to the nonfused path|
|[AF01](../../reports/af01-results-v0.1-2026-10-01.md)|Fusion through softmax and V combination reduced component time by 26.8%|
|[AC03](../../reports/ac03-results-v0.1-2026-10-01.md)|Fusion reduced model time by 7.8%; trade-offs relative to the uncompressed path were measured|
|[DC01](../../reports/dc01-results-v0.1-2026-10-01.md)|Costs of V reconstruction and input validation were identified|
|[AC04](../../reports/ac04-results-v0.1-2026-10-02.md)|Grouped input validation reduced model decoding time by 15.4%|
|[AC05](../../reports/ac05-results-v0.1-2026-10-02.md)|Validation-related time savings and output agreement persisted after 3 compression updates|
|[AC06](../../reports/ac06-results-v0.1-2026-10-02.md)|Quality and validation-related time savings checked on 4 books not previously used in this study|
|[NC01](../../reports/nc01-results-v0.1-2026-10-02.md)|Valid inputs passed; noncontiguous inputs stopped before computation|
|[AC07](../../reports/ac07-results-v0.1-2026-10-02.md)|Time savings from V fusion persisted in the full model|
|[NB01](../../reports/nb01-results-v0.1-2026-10-02.md)|Small Nanbeige integration check passed|
|[NB02](../../reports/nb02-results-v0.1-2026-10-02.md)|Nanbeige actual 3-bit KV storage and full reconstruction passed|
|[NB03](../../reports/nb03-results-v0.1-2026-10-02.md)|Fused component transfer to Nanbeige GQA passed|
|[NB04](../../reports/nb04-results-v0.1-2026-10-03.md)|NB04 results v0.1, 2026-10-03|

Stops and fixes are described in the public reports above. Internal workflow ledgers are not distributed. This history is not accumulated into an independent sample count.

## 6. Relationship to the Integrated Paper

This paper details implementation and evaluation in looped models; the [integrated NSN paper v2.0](../nsn-integrated/NSN-2026-10-29-EN-v2.0.md) brings together configuration, applicability, actual storage, and readout costs. The earlier public NSN version has been checked in the [publication provenance record](../../metadata/publication-provenance/README.md). Author-owned papers, figures, and data use CC BY 4.0, code uses Apache-2.0, and third-party terms are retained (see [licenses](../../LICENSES.md)). The repository is [https://github.com/metaSATOKEN/nsn](https://github.com/metaSATOKEN/nsn); [README](../../README.md) records its access status and the availability of companion ZIPs. A permanent archive DOI for the integrated paper has not been assigned.

## 7. Short Implementation and Literature Checks Added in Japanese v0.3

The function body of [nsn_int3_core.py](code/nsn_int3_core.py), shown in Listing 1, contains 10 lines excluding imports and the function signature. It follows `nsn_encode`, `nsn_decode`, and `nsn_qdq` in the [frozen arithmetic](../../scripts/loop_kv_nb05_primitives.py). It reflects the current rules, including the `1e-12` directional denominator and FP16 metadata rounding; it is not an unchanged reproduction of the entire explanatory code from the earlier NSN manuscript.

The [comparison results](../../data/validation/qdq-core/core-validation.json) show bitwise agreement with the CPU reference in 24 conditions: 8 synthetic input types × 3 dtypes (FP32 / FP16 / BF16). Inputs include all-zero and constant arrays, zero rows, varying norms, and strided inputs. This is not a new model-quality measurement, a proof for arbitrary inputs, or a bitwise-equivalence test of the GPU fused kernel. Reference-source and short-code SHA256 hashes and per-fixture results are saved. The [check script](../../data/validation/qdq-core/check_core.py) extracts only three functions from the reference code and runs them on the CPU.

Actual storage additionally requires the [codec](../../scripts/loop_kv_nb05_codec.py), cache management, checks, and attention paths. Listing 1's uint8 code array is not counted as actual 3-bit storage. Results in Sections 4–6, all 5 figures, Tables 4–8, and the [paper JSON](data/paper-data-v0.3-2026-10-08.json) remain unchanged from Japanese v0.2 and use the same derived assets.

Section 2 of the paper cites the primary sources used for each comparison.

## 8. Numerical Conditions and Separate Reproduction Entry Points

Appendix B, Table B1 corresponds to code fixed before measurement; Table B2 corresponds to saved diagnostic results. The [numerical-audit JSON](data/numerical-audit-v0.4-2026-10-03.json) stores maximum differences, pass counts, and source hashes. Error measures assess reconstruction and computational consistency for the same compressed input. Prediction-quality loss relative to the uncompressed path is assessed with PPL in Table 7.

|Check|Fixed-code entry point|Reference distinction|
|---|---|---|
|Reconstructed K/V and QK score|[QK01 contract](../../scripts/loop_kv_qk01_contract.py), `k_metrics` and `score_metrics`; corresponding functions in [NB05 numerics](../../scripts/loop_kv_nb05_numeric.py)|Ouro's score reference uses frozen-reconstruction K; Nanbeige's fused-score reference uses fused-reconstruction K. Nanbeige checks K/V reconstruction error separately|
|Context|[AF01 contract](../../scripts/loop_kv_af01_contract.py), `context_metrics`; same-named NB05 function|FP64 softmax of saved FP32 scores and V reconstructed by that path|
|Context including reference-V differences|[VF01 contract](../../scripts/loop_kv_vf01_contract.py), `context_budget`; [VF01 verifier](../../scripts/verify_loop_kv_vf01.py), `numerics`|Budget includes V reconstruction differences; score checks handle QK error|
|SDPA output|NB05 numerics, `sdpa_metrics`|FP64 attention using each SDPA path's own reconstructed K/V; scale is a weighted sum of absolute values|
|Bridge, ordinary calls, and actual forward passes|[AC08 diagnostics](../../scripts/loop_kv_ac08_audit.py); [NB05 diagnostics](../../scripts/loop_kv_nb05_diagnostic.py)|Bitwise agreement for identical inputs and arithmetic. Nanbeige distinguishes bridge K/V agreement from scale-dependent final-output errors|
|NLL|[AC08 verifier](../../scripts/verify_loop_kv_ac08.py); [NB05 verifier](../../scripts/verify_loop_kv_nb05.py)|Recalculation from saved logits; counts differ from the number of distinct texts in the quality evaluation|

For Japanese v0.4, saved AC08, NB05, and VF01 diagnostic summaries were extracted, and NB05's 48 saved FP32 logit trajectories (432 predictions) were recalculated on the CPU. The maximum NLL difference was 7.106e−15, within the existing relative/absolute tolerance of 1e−12. [Per-prediction checks](../../data/validation/nb05/nb05-nll-checks.json) are retained. Ouro's maximum over 8,925 predictions is quoted from a previously saved verification result produced by a separate implementation within the same project. Neither that revision nor this translation reran the GPU experiments or fully reverified AC08 raw arrays.

There are three reproduction entry points:

1. **Main tables and figures:** the paper JSON and plotting code in Sections 2–3 read existing summaries.
2. **Diagnostic extraction and saved-NLL verification:** the [CPU script](code/extract_numerical_audit_v0_4.py) writes to a separate output directory without changing original runs or formal summaries. It requires NumPy and the Python standard library and writes two files: supplementary JSON and per-prediction checks. Contract-code hashes are included in the supplementary JSON.
3. **Original numerical verification and GPU reruns:** use the verifiers in Section 3, raw arrays in the original runs, prespecified plans, and distributed code. CPU extraction does not rerun a kernel. No GPU-dependent work was performed for the revision or translation.

```bash
python3 papers/loop-kv/code/extract_numerical_audit_v0_4.py \
  --research-root . --output-dir /tmp/loop-kv-numerical-audit
```

The script reads stage-specific maxima from saved summaries, checks recorded hashes for NB05 raw files, and recalculates NLL. It does not import or execute experimental Python code. It stops if the output directory already contains files with the same names.

## 9. Main Execution Paths, Checks, and Synchronization

Table 8 compares the following **complete checked paths**. The scope of the change differs even when both are called fusion. Savings were not obtained by moving checks outside the timing interval.

|Model and path|Reconstruction arithmetic and attention|Input/output checks and host confirmation points|
|---|---|---|
|Ouro `NATIVE`|BF16 K/V; official SDPA call|Check decoding position and all-allowed mask; Q/K and V structure/finiteness; attention-output finiteness|
|Ouro `PACKED_GROUPED`|Internal K reconstruction; full V array from frozen RF01 reconstruction; dedicated QK, softmax, and V combination|Check finiteness after full V reconstruction. Group checks of Q, K metadata/tail, and reconstructed V; confirm flags on the host. Check fused-computation flags and final output|
|Ouro `PACKED_KV_FUSED`|Internal tile-based K/V reconstruction, with fusion-specific V arithmetic|Check packed Q/K/V structure and metadata/tail finiteness; confirm computation flags and output. The outer model adapter also checks output finiteness|
|Nanbeige `NATIVE`|BF16 K/V repeated for 6-to-1 GQA mapping and passed to SDPA|Common Q/K/V structure and finiteness checks; host confirmation of grouped flags; output-finiteness check|
|Nanbeige `PACKED_SDPA`|Full K/V arrays produced by PyTorch unpacking and reference NSN reconstruction, followed by GQA mapping and SDPA|Same raw-input checks and flag confirmation as NATIVE; reconstruction/output checks|
|Nanbeige `GQA_FUSED`|Q heads mapped to KV heads 6-to-1; tile-internal K/V reconstruction and attention|Common raw-input checks, followed by partial/merge kernel flags and output checks|

Ouro's paths correspond to the [model adapter](../../scripts/loop_kv_ac08_attention.py), [grouped input validation](../../scripts/loop_kv_iv01_kernel.py), [V fusion](../../scripts/loop_kv_vf01_kernel.py), [V input checks](../../scripts/loop_kv_vf01_checks.py), and [RF01 reconstruction](../../scripts/loop_kv_rf01_kernel.py). Nanbeige's correspond to its [model adapter](../../scripts/loop_kv_nb05_attention.py), [kernel paths](../../scripts/loop_kv_nb05_kernel.py), and [common checks](../../scripts/loop_kv_nb05_checks.py).

Nanbeige's `TILE_SDPA` materializes full K/V using the same arithmetic as fusion before calling SDPA. It is a **bridge for local diagnostics**, not an arm in the main model comparison in Table 8. Ouro's same-arithmetic V bridge is likewise reported separately as a component comparison in Table 5.

`bool(tensor)`, position `item()` calls, and finiteness-flag confirmation involve waiting for GPU results on the CPU. The [AC08 runner](../../scripts/loop_kv_ac08.py) and [Nanbeige timing wrapper](../../scripts/loop_kv_nb05_measure.py) set CUDA-event and wall-time boundaries, synchronizing before measurement and at its end. Evidence copies to the CPU and file writes occur outside those intervals. Reconstruction arithmetic, validation targets, and call counts differ across paths; Table 8 does not isolate the causal effect of removing the full V array alone.

SDPA uses **automatic backend selection; the selected backend for individual calls was not identified**. Ouro's [execution settings](../../runs/loop-kv-ac08-20261002-095022-41c2554c/execution_settings.json) record whether flash, efficient, math, and cuDNN backends are enabled, not which one was chosen for each call. Nanbeige's [example environment record](../../runs/loop-kv-nb05-20261003-090604-18baabf5/books/b0/environment.json) records the PyTorch-default policy. No backend identity is inferred beyond the saved evidence.

## 10. Timing Positions and Compression-Update Order

Here, \(L\) is the number of tokens processed by the forward pass: the KV length at attention. Token-array indices are zero-based.

|Ouro set|KV length \(L\)|Count|
|---|---|---:|
|All decoding positions|769–1023|255|
|Initial excluded positions|769–776|8|
|Capture exclusions|769, 831, 833, 895, 897, 959, 961, 1023|8|
|Intersection of the exclusion sets|769|1|
|Main timing positions|All decoding positions minus the union of the two exclusion sets|240|
|Additional compression after the forward pass|832, 896, 960|3|

These match `timed_lengths`, `capture_lengths`, and `flush_post_lengths` in the [fixed input JSON](../../scripts/loop_kv_ac08_inputs.json). Compression positions 832, 896, and 960 are included in main timing. Quality uses all 255 predictions; main timing uses 240 positions. Captures cover slots 0, 95, and 191 at each capture position. The exclusion sets were not changed to recalculate results.

Compression updates follow `post_prefill` and `post_decode` in the [Ouro base cache](../../scripts/loop_kv_pk01_cache.py) and [Nanbeige cache](../../scripts/loop_kv_nb05_packed.py):

```text
prefill: attention with uncompressed K/V -> model output
post_prefill: compress old complete 64-token blocks, retaining at least 64 tail tokens
each decoding step:
    append new K/V to the BF16 tail
    attention using the packed prefix and BF16 tail -> model output
    post_decode: if the tail has at least 128 tokens, compress its oldest 64
    retain the remaining tail values in BF16
```

Ouro compresses 704 tokens after a 768-token prefill. Attention at L=832 sees 704 compressed tokens plus a 128-token tail; after that forward pass, the state is 768 compressed tokens plus a 64-token tail. Another 64 tokens are compressed at L=896 and L=960. At terminal L=1023, the state is 896 compressed tokens plus a 127-token tail. FORWARD and POST are timed separately and added for main timing. Prefill and initial compression are excluded from main timing and included in the initial-cost-inclusive metric.

Nanbeige compresses 64 tokens after a 128-token prefill. After 8 decoding steps, it has 64 compressed tokens plus a 72-token tail. No subsequent compression update is triggered in these trajectories. The coexistence of shorter decoding and a longer total including initial compression is a result for this short usage interval.

## 11. Materials, Tokenizers, and Prediction Targets

The PG19 candidate list is tied to revision `4d28bd77e66947ad3835cf78ed7aaeb4dd87ad8b`. Text files were retrieved from `https://storage.googleapis.com/deepmind-gutenberg/train/{book_id}.txt`. Fixed candidates, seeds, and exclusions are documented in the plans linked in Section 3 and each `selection_lock.json`. Actual URLs, text hashes, token sequences, and token hashes are in the corresponding `materials.json`.

|Model|Four book IDs|Pinned tokenizer revision|Input and evaluation targets|
|---|---|---|---|
|Ouro|38233, 57533, 3921, 58192|`fcfcfbc429c7b10ee73dfc718033351d8936d2a8`|First 1024 tokens. Prefill inputs: indices 0–767. Main target-token indices: 769–1023 (255 predictions). Prefill target 768 is excluded from main quality|
|Nanbeige|9667, 26438, 45277, 28881|`4a38e817c14b9f2c4b69b1e05cbda90395872f6b`|First 137 tokens. Prefill inputs: indices 0–127. Main target-token indices: 129–136 (8 predictions). Prefill target 128 is excluded from main quality|

- Ouro: [materials](../../runs/loop-kv-ac08-20261002-095022-41c2554c/materials.json) / [selection lock](../../runs/loop-kv-ac08-20261002-095022-41c2554c/selection_lock.json) / [selection and tokenization code](../../scripts/loop_kv_ac08_materials.py).
- Nanbeige: [materials](../../runs/loop-kv-nb05-20261003-090604-18baabf5/materials.json) / [selection lock](../../runs/loop-kv-nb05-20261003-090604-18baabf5/selection_lock.json) / [selection and tokenization code](../../scripts/loop_kv_nb05_materials.py).

Downloaded bytes were decoded as UTF-8 and tokenized with `add_special_tokens=False, truncation=True` and `max_length=1024` or 137. No opening material or preface was removed after download, no custom whitespace normalization was applied, and no chat template or manual BOS/EOS was added. Tokenizer-internal normalization follows the pinned version. Ouro used the pinned `AutoTokenizer` defaults; Nanbeige used `use_fast=False, legacy=True`. This documents the preprocessing already used, rather than introducing a new procedure.

Main-quality targets and saved raw-logit counts differ. For example, the Nanbeige CPU check covers 9 predictions × 48 trajectories, including the prefill prediction. Its 8 main-quality predictions are distinct from the 9 predictions used in numerical-consistency checks.

## 12. Continuation Evidence and Regeneration of the Supplemented Edition

The continuation comparison was fixed before measurement as NB06. Entry points are the [design](../../plans/loop-kv-nb06-design-v0.1-2026-10-08.md), [prespecified card](../../plans/prespecified/loop-kv-nb06-v0.1-2026-10-08.md), [input contract](../../plans/loop-kv-nb06-input-contract-v0.1-2026-10-08.json), [implementation](../../scripts/loop_kv_nb06.py), [fixed verifier](../../scripts/verify_loop_kv_nb06.py), [original results](../../runs/loop-kv-nb06-20261008-075410-5621380a/summary.json), and [input tokens](../../runs/loop-kv-nb06-20261008-075410-5621380a/materials.json).

The quality, timing, and mean_additive_cost_cuda_ms fields of the [reaggregated results](../../reports/nb06-results-v0.1-2026-10-08.json) map to the quality table, cumulative-time table, and descriptive cost breakdown. The [reaggregation code](../../scripts/analyze_loop_kv_nb06_results.py) checks values against original step records. The 32 actual-query diagnostics are in books[].actual_q of the original summary; memory is in books[].memory_peak_bytes and terminal_kv_bytes. The [verification record](../../data/validation/nb06/verification-summary.json) and [paper-level NB06 numerical table](data/numerical-audit-nb06-v0.1-2026-10-08.json) are also retained.

The [reproduction guide](../../REPRODUCIBILITY.md) describes how to check the saved logits and regenerate the continuation table data and Figure 6. The materials are the same four texts; quality and timing are not pooled with NB05. Original ZIP SHA256: 2ef013414a6a7130bb55ddd0ff32328698b965a1f3347fa023b2400d5d2246fa. Original summary SHA256: 6da2ac29e8490ef5a36364b5637ce56d47bc00554605674a38f3ec93bd4ddc8a.
