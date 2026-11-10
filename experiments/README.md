# Experiments in the integrated paper

**Looking for the fused GPU kernels?** Open the [Ouro packed K/V kernel](ac08/source/loop_kv_vf01_kernel.py) or the [Nanbeige GQA kernel](nb06/source/loop_kv_nb06_kernel.py) directly. The [root README kernel guide](../README.md#gpu-kernels-where-to-start) maps reconstruction, attention, cache updates, and verification to source files and functions.

Follow the research question you want to inspect. Earlier source files are grouped by purpose, rather than by the name of the project in which they were written.

| Question | Code | Data |
| --- | --- | --- |
| Which quantizer configuration works, and where does it fail? | [Configuration / K–V diagnostics](configuration/README.md) | [Configuration results](../data/configuration/) |
| Does the setting transfer across architectures and context policies? | [Applicability](applicability/README.md) | [Applicability results](../data/applicability/) |
| How many bytes are stored, and what does reading them cost? | [Storage and reconstruction](storage/README.md) | [Storage results](../data/storage/) |
| What changes with kernel implementations? | [QK components](kernels/README.md) | [Kernel results](../data/kernels/) |
| Does packed reconstruction work inside looped models? | [AC08](ac08/README.md), [NB05](nb05/README.md), [NB06](nb06/README.md) | [Runs and companion assets](../DATA_COVERAGE.md) |

For the quantizer itself, start with [src/nsn](../src/README.md) and the [small paper QDQ core](../papers/nsn-integrated/code/nsn_int3_core.py). For reported numbers, use [CPU saved-data reproduction](../REPRODUCIBILITY.md).

AC08/NB05/NB06 provide visible expanded sources and a check-only default launcher. `--run` explicitly starts the original Colab GPU workflow. The earlier configuration/applicability/storage/kernel files are retained original experiment code; their individual READMEs explain the execution context. No compressed executable OneCell payload is used as the public entrypoint.
