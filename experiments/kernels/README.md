# Fused QK component measurements

For **full packed K/V attention**, go directly to the [Ouro kernel](../ac08/source/loop_kv_vf01_kernel.py), [Nanbeige GQA kernel](../nb06/source/loop_kv_nb06_kernel.py), or [kernel reading guide](../../README.md#gpu-kernels-where-to-start). This directory contains the separate **QK-score component** study, which stores one `uint8` element per code.

Production, compilation-corroboration, and fused Triton QK components. These use their original code-storage contracts.

Start with [phase_k1c_triton_fused_kernel.py](phase_k1c_triton_fused_kernel.py).

- [phase_k1_production_kernel_benchmark.py](phase_k1_production_kernel_benchmark.py)
- [phase_k1b_compile_corroboration.py](phase_k1b_compile_corroboration.py)
- [phase_k1c_triton_fused_kernel.py](phase_k1c_triton_fused_kernel.py)

These are unchanged original experiment sources and preregistration records. Some `.py` files contain Colab `!pip` commands and placeholder tokens; inspect them in their original notebook context. They are not the local CPU reanalysis entrypoint. Use the [reproduction guide](../../REPRODUCIBILITY.md) for saved results.

Corresponding measurements: [data index](../../data/README.md). Original names, hashes and new locations are recorded in [source-layout.json](../../metadata/source-layout.json).
