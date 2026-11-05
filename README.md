# NSN: KV Quantization by Separating Norm and Direction

**How far can we shrink a language model's KV cache while retaining prediction quality—and what does it take to turn that smaller cache into cheaper inference?**

A language model stores keys and values from earlier tokens in its **KV cache**, so it can attend to the past without recomputing everything. This memory grows with context length. Low-bit quantization makes it smaller, but also changes the values used by attention and adds work to reconstruct them.

This project develops and evaluates **asymmetric NSN**: store each vector's magnitude separately, and represent its direction with a small set of integers. The integrated study connects the method's prediction quality to its actual storage and execution costs. It identifies useful settings across model architectures, then follows the compressed representation all the way into attention.

[Read the paper](papers/nsn-integrated/NSN-2026-10-29-EN-v2.0.md) · [日本語 README](README.ja.md) · [日本語論文](papers/nsn-integrated/NSN-2026-10-29-JA-v2.0.md) · [Code](experiments/README.md)

## What this study adds

**1. Evidence for choosing asymmetric NSN, and for adjusting its precision.** In the primary comparison, asymmetric NSN was the only one of four 3-bit configurations to keep PPL degradation below 5% in all **8 tested models**. Its value here is broad coverage with one configuration; other configurations achieved lower PPL on some models. The same method also supports higher bit widths: a **5-bit setting** kept mean PPL changes within ±0.5% in **4 models with SSM-plus-attention or MoE architectures**. This gives a concrete way to trade code size for precision.

**2. Working compressed caches in looped Transformers.** We connected actual bit-packed 3-bit KV to **Ouro and Nanbeige**, including cache updates and the mapping between loops, layers, and heads. Terminal KV storage fell by **67.68% in Ouro** and **64.39% in the extended Nanbeige evaluation**, counting metadata and the uncompressed tail. These are measured storage reductions from representations used for subsequent predictions.

**3. Reconstructing V where attention uses it reduced readout time.** We compared two paths that use the same packed V (the values that attention combines in a weighted sum) and the same reconstruction arithmetic. The bridge control writes the reconstructed values for all past positions to an array, which attention then reads. The fused kernel reconstructs V one block at a time and uses each block directly in the weighted sum, skipping the write and subsequent read of the full-length array. In this component comparison, the fused path took **28.28% less time** than the bridge control.

Quality here means the paper's **PPL diagnostic**: how well the model predicts the supplied text, with a 5% degradation threshold. The four-method result compares complete configurations. The hybrid/MoE results quantize attention K/V using ranges estimated over each full evaluation window; the separate packed-cache evaluations use teacher forcing on **4 texts per model**, with **255 predictions per text for Ouro and 256 for the Nanbeige extension**. This defines the evidence for choosing settings; long free-running generation and a ranking against external quantizers remain open questions.

## Why separate magnitude from direction?

The vectors `(3, 4)` and `(30, 40)` point in the same direction, but their lengths are 5 and 50. When raw KV values are quantized directly per channel, large-magnitude tokens can widen the range and leave a coarse grid for smaller tokens.

Asymmetric NSN separates these two jobs:

1. **Keep magnitude:** store each token vector's norm separately.
2. **Compress direction:** normalize the vector, then assign integer levels over each channel's observed minimum-to-maximum range. The range need not be symmetric around zero.
3. **Reconstruct for attention:** restore the direction, normalize its length after rounding, and multiply by the saved norm.

The quantizer uses the input block's statistics, with no additional quantizer training or separate calibration corpus. Bit width remains a design choice: 3 bits provide 8 levels; 5 bits provide 32.

![NSN separates vector magnitude and direction, quantizes direction, then reconstructs it with the stored norm.](papers/nsn-integrated/figures/figure-01-nsn.png)

### Asymmetric NSN in 10 lines

The function below shows the **3-bit quantization and reconstruction core** from the paper. Input `x` is a validated floating-point KV block with shape `[..., 64, 128]`: 64 tokens and 128 dimensions per head. It assumes finite values and metadata within the FP16 storage range, and returns reconstructed values with the same shape and dtype.

```python
import torch

def nsn_int3_qdq(x):
    a = x.float()
    n = torch.linalg.vector_norm(a, dim=-1, keepdim=True)
    d = a / n.clamp_min(1e-12)
    m = d.amin(dim=-2, keepdim=True)
    s = (d.amax(dim=-2, keepdim=True) - m) / 7
    q = ((d - m) / torch.where(s == 0, torch.ones_like(s), s)).round().clamp(0, 7).to(torch.uint8)
    n, m, s = (t.half().float().contiguous() for t in (n, m, s))
    d = (m + s * q.float().contiguous()).contiguous()
    y = n * (d / torch.linalg.vector_norm(d, dim=-1, keepdim=True).clamp_min(1e-12))
    return torch.where(n == 0, torch.zeros_like(y), y).to(x.dtype)
```

This example quantizes and reconstructs floating-point values; bit packing, cache updates, and attention integration are implemented separately. See the [matching reference function](papers/nsn-integrated/code/nsn_int3_core.py), [method and arithmetic details](papers/nsn-integrated/NSN-2026-10-29-EN-v2.0.md#24-the-computational-core-and-implementation-differences), and [readable model-integration code](experiments/README.md).

## When does smaller storage become an execution benefit?

The Nanbeige measurements make the distinction concrete:

| Evaluation | Observed outcome |
| --- | --- |
| Initial 8-step continuation | Fused decoding was about **3% faster** than checked uncompressed execution, but time including prefill and initial compression was about **15% longer**. |
| Extension to 256 steps, including four compression updates | KV storage was **64.39% smaller**, while total time including initialization and updates was **approximately equal** to checked uncompressed execution. |

Compression saved capacity in this workload. Turning that saving into a time advantage required accounting for **reconstruction, validation, initial packing, and updates**. The paper also measures whole-GPU memory peaks separately from KV storage, because weights and temporary arrays remain in memory. These cost breakdowns help identify which part of an implementation needs work.

## Explore the research

| To understand… | Start here |
| --- | --- |
| The method, findings, and their scope | Integrated paper: [English](papers/nsn-integrated/NSN-2026-10-29-EN-v2.0.md) / [Japanese](papers/nsn-integrated/NSN-2026-10-29-JA-v2.0.md) with figures and appendices · [English PDF](papers/nsn-integrated/NSN_Integrated_EN_v2.0.pdf) |
| How the looped-model caches and kernels work | [Looped-model companion paper](papers/loop-kv/README.md) · [English PDF](papers/loop-kv/Loop-kv_NSN_EN_v1.0.pdf) |
| How the experiments were implemented | [Readable Python sources and experiment index](experiments/README.md) |
| Which measurements support a result | [Data index](data/README.md) and [evidence coverage](DATA_COVERAGE.md) |
| The original edition or the retained library API | [Earlier edition and library guide](LEGACY.md) |

The integrated paper introduces NSN from the beginning. Experiment sources are expanded, readable Python; no compressed OneCell payload needs to be deciphered.

## Verify or reproduce

The repository contains papers, figures, code, tables, plans, and readable logs. Large saved arrays are distributed as **17 ZIPs (6.30 GB total) plus SHA256SUMS** in the [integrated-v1.0 Release](https://github.com/metaSATOKEN/nsn/releases/tag/integrated-v1.0). Download all 17 into one directory without renaming them. GitHub's automatic source archives contain only the repository.

From a standalone checkout, use its root `.venv` as described in [RUNNING.md](RUNNING.md):

```sh
.venv/bin/python -B tools/verify_package.py
```

This checks file identity without running a model. [REPRODUCIBILITY.md](REPRODUCIBILITY.md) explains how to verify the companion ZIPs, restore a new workspace, and regenerate results with the required dependencies. [release-assets.json](metadata/release-assets.json) lists filenames, sizes, hashes, and download URLs.

## Edition, citation, and license

Kentaro Sato · Independent Researcher · [info@metaclan.jp](mailto:info@metaclan.jp)

The integrated Markdown manuscripts are **v2.0, 29 October 2026**. They combine a reanalysis of the earlier [*Norm Separation is Necessary*](https://doi.org/10.5281/zenodo.19724817) with subsequent implementation and looped-model evaluations. The version DOI for the integrated paper is [10.5281/zenodo.23938878](https://doi.org/10.5281/zenodo.23938878). See [CITATION.cff](CITATION.cff).

Code: [Apache-2.0](LICENSE). Author-owned papers, figures, and data: [CC BY 4.0](LICENSES/CC-BY-4.0.txt). Third-party terms are retained in [LICENSES.md](LICENSES.md).

The repository and evidence Release require authorized access.
