# Measurements by research question

| Group | Supports | Original result JSONs directly reaggregated |
| --- | --- | ---: |
| [configuration](configuration/) | Four-method comparison, precision/segment boundaries, norm-dispersion and K/V diagnostics | 10 |
| [applicability](applicability/) | Architecture/bit-width and causal/context/model-class comparisons | 4 |
| [storage](storage/) | Stored bytes, actual cache PPL and reconstruction timing | 3 |
| [kernels](kernels/) | Production/compiled/fused QK component comparisons | 3 |
| [additional-records](additional-records/) | 115 catalogue-only contextual records; not silently pooled into the main comparisons | 0 |

All 135 JSON files preserve their original bytes. The [frozen extraction catalogue](../papers/nsn-integrated/data/extraction-v0.1/result-catalog.json) records which 20 were selected for numerical extraction. Its `path` and other original `source` fields are acquisition-time identifiers. [source-layout.json](../metadata/source-layout.json) maps them to the files here without requiring the old project trees.

Looped-model arrays and trajectories stay under [runs](../runs/) and the companion Release ZIPs. [DATA_COVERAGE.md](../DATA_COVERAGE.md) describes their granularity. The [CPU reproduction command](../REPRODUCIBILITY.md) is the entrypoint for recalculation; all source evidence remains read-only.

[Validation evidence](validation/README.md) contains the numerical checks used by the papers. Internal workflow ledgers and delivery records are excluded from this distribution.
