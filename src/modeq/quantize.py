# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

"""
Core quantization primitives: naive absmax and nsep+pchan.

All functions take a tensor `x` with last dim = head_dim and
return a same-shape, same-dtype tensor quantized to INT-`bits`
precision via simulated quantization (quantize then dequantize
in place). For true bit-packing, see `modeq.pack` and
`modeq.cache`.
"""

import torch


def qa_perrow(x, bits):
    """
    Naive absmax per-row quantization at arbitrary bit width.

    Each row (last dim) uses a single scale = absmax / qm.
    This is the baseline method that breaks catastrophically on
    Mode 1 (Layer-0 outlier-driven) models at INT4.

    Parameters
    ----------
    x : torch.Tensor
        Tensor with last dim to be quantized.
    bits : int
        Target bit width in {2, 3, 4, 5, 6, 7, 8}. Uses symmetric
        quantization with qm = 2**(bits-1) - 1.

    Returns
    -------
    torch.Tensor
        Quantized-and-dequantized tensor with same shape and dtype.
    """
    dtype = x.dtype
    x = x.float()
    qm = 2 ** (bits - 1) - 1
    s = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / qm
    return (((x / s).round().clamp(-qm, qm)) * s).to(dtype)


def qa_perchan(x, bits):
    """
    Per-channel absmax quantization at arbitrary bit width.

    Each column (second-to-last dim reduction) uses its own scale.
    This isolates outlier channels and prevents them from dominating
    the scale for normal channels -- the core of LLM.int8 and
    SmoothQuant ideas applied per-channel to the KV cache.

    Parameters
    ----------
    x : torch.Tensor, shape [..., T, D]
        Tensor to quantize. The scale is computed over the T dim
        (typically the token dim), separately for each of the D
        channels.
    bits : int
        Target bit width.

    Returns
    -------
    torch.Tensor
        Quantized-and-dequantized tensor with same shape and dtype.
    """
    dtype = x.dtype
    x = x.float()
    qm = 2 ** (bits - 1) - 1
    s = x.abs().amax(dim=0, keepdim=True).clamp(min=1e-12) / qm
    return (((x / s).round().clamp(-qm, qm)) * s).to(dtype)


def qa_nsep_pchan(x, bits):
    """
    Norm-separated per-channel quantization (the core method).

    Decomposes each vector into its L2 norm (kept at FP precision)
    and its unit direction (quantized per-channel), then reconstructs.
    This handles both failure modes of naive INT quantization:

        Mode 1 (outlier channels): per-channel scaling isolates them.
        Mode 2 (norm variation):   norm separation removes token-wise
                                   dynamic range from the direction.

    Re-normalization after quantization brings the reconstructed
    direction back onto the unit sphere, avoiding systematic bias
    from quantization-induced norm drift.

    Parameters
    ----------
    x : torch.Tensor, shape [..., T, D]
        Tensor to quantize.
    bits : int
        Target bit width. Best at INT3 for 5.33x compression with
        near-baseline quality on most Pre-LN Transformers.

    Returns
    -------
    torch.Tensor
        Quantized-and-dequantized tensor with same shape and dtype.
    """
    dtype = x.dtype
    x = x.float()
    n = x.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    d = x / n
    d_q = qa_perchan(d, bits).float()
    d_q = d_q / d_q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return (n * d_q).to(dtype)


def apply_method(x, method, bits):
    """
    Dispatch quantization by method name.

    Parameters
    ----------
    x : torch.Tensor
    method : str
        One of {"baseline", "fp16", "naive", "nsep+pchan"}.
        "baseline"/"fp16" return x unchanged.
    bits : int
        Target bit width (ignored for baseline).
    """
    if method in ("baseline", "fp16"):
        return x
    if method == "naive":
        return qa_perrow(x, bits)
    if method == "nsep+pchan":
        return qa_nsep_pchan(x, bits)
    raise ValueError(f"Unknown method: {method}")
