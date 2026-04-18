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

    The scale is computed over the token axis (``dim=-2``),
    independently for each of the D channels. This isolates outlier
    channels and prevents them from dominating the scale for normal
    channels -- the core of LLM.int8 and SmoothQuant ideas applied
    per-channel to the KV cache.

    Supports 2D ``[T, D]``, 3D ``[H, T, D]``, and 4D ``[B, H, T, D]``
    inputs via the ``dim=-2`` convention.

    Parameters
    ----------
    x : torch.Tensor, shape [..., T, D]
        Tensor to quantize. ``T`` is the token axis; ``D`` is the
        (head) channel axis.
    bits : int
        Target bit width.

    Returns
    -------
    torch.Tensor
        Quantized-and-dequantized tensor with same shape and dtype.

    Notes
    -----
    In v0.2, this function incorrectly used ``dim=0`` (which
    coincides with the token axis only for 2D input). On 4D
    ``[B, H, T, D]`` inputs with ``B=1`` -- the common README
    example shape -- the reduction became element-wise, making the
    quantizer an identity. The experimental PPL numbers were
    unaffected because the experiment scripts slice to 2D per head
    before calling the quantizer, but the public API was broken.
    Thanks to GPT-5 for surfacing this in a cross-review.
    """
    dtype = x.dtype
    x = x.float()
    qm = 2 ** (bits - 1) - 1
    s = x.abs().amax(dim=-2, keepdim=True).clamp(min=1e-12) / qm
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


def qa_nsep_pchan_resid_rms(x, bits_base):
    """
    nsep+pchan with 1-bit per-channel RMS residual correction (v0.2).

    After the base quantization, the per-value residual is sign-stored
    with a magnitude set per-channel to the RMS of residuals in that
    channel. Adds 1 bit/value of storage plus D FP16 scalars per head
    for the RMS magnitudes. Total: approximately (bits_base + 1)
    bits/value at typical head dimensions.

    On Mode 1 models (Qwen2-7B in our sample), INT3 + RMS residual
    surpasses INT4 alone by 2.2x precision at roughly 4+epsilon
    bits/value, escaping the INT4 "valley of death".

    On Mode 2 models, INT3 + RMS residual improves over INT3 alone
    by 16-72% but does not surpass INT4 alone.

    IMPORTANT: the per-channel RMS is essential. A fixed constant
    magnitude (e.g., 0.25, the E[|residual|] under uniform assumption)
    BACKFIRES on Mode 1 due to outlier-driven non-uniform residual
    distribution. See `qa_nsep_pchan_resid_fixed` for the fixed-
    magnitude variant, retained only for ablation studies.

    Parameters
    ----------
    x : torch.Tensor, shape [..., T, D]
    bits_base : int
        Base quantization bit width. 3 is the recommended default.

    Returns
    -------
    torch.Tensor, same shape and dtype as x.
    """
    dtype = x.dtype
    x = x.float()
    n = x.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    d = x / n

    qm = 2 ** (bits_base - 1) - 1
    s = d.abs().amax(dim=-2, keepdim=True).clamp(min=1e-12) / qm
    d_scaled = d / s
    d_q = d_scaled.round().clamp(-qm, qm)

    # Per-channel RMS residual correction (reduce over token axis dim=-2)
    residual = d_scaled - d_q
    r_rms = (residual ** 2).mean(dim=-2, keepdim=True).sqrt().clamp(min=1e-12)
    r_1bit = r_rms * residual.sign()

    d_corrected = (d_q + r_1bit) * s
    d_corrected = d_corrected / d_corrected.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return (n * d_corrected).to(dtype)


def qa_nsep_pchan_resid_fixed(x, bits_base, magnitude=0.25):
    """
    nsep+pchan with 1-bit fixed-magnitude residual correction (ablation).

    Uses a constant magnitude for the residual correction instead of the
    adaptive per-channel RMS. The default 0.25 is the optimal choice for
    uniform-distributed residuals on [-0.5, 0.5].

    Known failure: on Mode 1 models, this variant BACKFIRES: the residual
    distribution is non-uniform (heavy-tailed due to outliers), so a
    fixed constant over-corrects small residuals and under-corrects large
    ones. Empirically, fixed-0.25 worsens INT2 ΔPPL from +326 to +463 on
    Qwen2-7B. Use `qa_nsep_pchan_resid_rms` instead for any model where
    Mode 1 behavior is possible.

    Retained here for ablation studies and for the special case of
    deliberately-clean Mode 2 models where the simpler constant works
    nearly as well as RMS.

    Parameters
    ----------
    x : torch.Tensor, shape [..., T, D]
    bits_base : int
    magnitude : float, default 0.25

    Returns
    -------
    torch.Tensor, same shape and dtype as x.
    """
    dtype = x.dtype
    x = x.float()
    n = x.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    d = x / n

    qm = 2 ** (bits_base - 1) - 1
    s = d.abs().amax(dim=-2, keepdim=True).clamp(min=1e-12) / qm
    d_scaled = d / s
    d_q = d_scaled.round().clamp(-qm, qm)

    residual = d_scaled - d_q
    r_1bit = magnitude * residual.sign()

    d_corrected = (d_q + r_1bit) * s
    d_corrected = d_corrected / d_corrected.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return (n * d_corrected).to(dtype)


def apply_method(x, method, bits):
    """
    Dispatch quantization by method name.

    Parameters
    ----------
    x : torch.Tensor
    method : str
        One of {"baseline", "fp16", "naive", "nsep+pchan",
        "nsep+pchan+resid-rms", "nsep+pchan+resid-fixed"}.
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
    if method == "nsep+pchan+resid-rms":
        return qa_nsep_pchan_resid_rms(x, bits)
    if method == "nsep+pchan+resid-fixed":
        return qa_nsep_pchan_resid_fixed(x, bits)
    raise ValueError(f"Unknown method: {method}")
