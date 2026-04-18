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

from .rotate import rotate, unrotate


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


def qa_nsep_pchan_asym(x, bits):
    """
    Asymmetric per-channel nsep+pchan quantization (v0.3 SOTA candidate).

    Quantizes the unit direction on the range ``[min_c, max_c]`` per
    channel rather than the symmetric ``[-|max_c|, +|max_c|]`` window.
    This uses all ``2**bits`` levels (e.g., 8 for INT3) across the
    channel's actual value range, instead of symmetric quantization's
    ``2*(2**(bits-1) - 1) + 1`` levels (e.g., 7 for INT3) centered on
    zero - of which roughly half are wasted when post-RoPE K channels
    have strongly biased distributions (empirically the case on
    Qwen2-7B Layer 0 K; see Phase 10/13 real-KV probes).

    Empirical gain at identical bit rate (Phase 13, real Layer 0 K,
    INT3, per-head MSE averaged over heads):

        Qwen2-7B  (Mode 1):  +91.2% MSE reduction vs symmetric INT3
        Llama-3.1-8B (Mode 2): +68.1% MSE reduction vs symmetric INT3

    At INT3 this costs one extra FP16 scalar per channel per tensor
    (storing ``d_min``; the range is derived as ``d_max - d_min``).
    For typical GQA configurations (n_kv_heads=4-8, head_dim=128)
    the metadata overhead is ~0.02 bits/value amortized over
    real-world context lengths - lost in the noise next to the
    precision gain.

    Parameters
    ----------
    x : torch.Tensor, shape [..., T, D]
        Input tensor. Last dim D is the head channel axis, second-to-last
        T is the token axis.
    bits : int
        Target bit width. 3 is the recommended default; 4 is the upper
        reference.

    Returns
    -------
    torch.Tensor, same shape and dtype as x.

    Notes
    -----
    * Unlike symmetric nsep+pchan, the number of representable levels
      is ``2**bits`` (not ``2**(bits-1) - 1`` twice), so asymmetric
      INT3 uses 8 levels vs symmetric's 7 - a 14% base gain even
      ignoring the distribution-fit advantage.
    * The norm-separation and post-quantization re-normalization steps
      are identical to `qa_nsep_pchan`, so asymmetric composes
      cleanly with the rest of the pipeline.
    """
    dtype = x.dtype
    xf = x.float()
    n = xf.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    d = xf / n

    d_min = d.amin(dim=-2, keepdim=True)                # [..., 1, D]
    d_max = d.amax(dim=-2, keepdim=True)
    d_range = (d_max - d_min).clamp(min=1e-12)

    levels = 2 ** bits - 1                               # 0..levels inclusive
    step = d_range / levels

    d_q = ((d - d_min) / step).round().clamp(0, levels) * step + d_min

    d_q = d_q / d_q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return (n * d_q).to(dtype)


def qa_nsep_pchan_2tier(x, bits_lo=3, bits_hi=6, k=4):
    """
    Two-tier nsep+pchan: top-K highest-magnitude channels get more bits.

    Motivation (empirical, v0.3 pre-flight, Phase 10 probe):
    Real Qwen2-7B Layer 0 K is a "plateau" pattern -- 5 channels all
    carry magnitudes in the 130-170 range while the remaining 123
    channels sit near 10-15. The plain INT3 nsep+pchan pipeline
    quantizes every channel with the same 7 levels, so the plateau
    channels (which dominate the MSE energy budget) end up with
    granularity ~160/7 = 23, while the small channels have tight
    ~2/7 = 0.3 granularity they don't need. Rotation does not help
    (Phase 10: 3.68x MSE regression) because per-channel absmax
    scaling is already optimal for discrete plateau structure.

    The remedy is to REALLOCATE bits rather than rearrange values:

        * Top-K channels (ranked by per-channel absmax on the direction)
          get `bits_hi` (default 6) -> 31 levels, granularity drops 4x
          on the plateau and dominates the MSE reduction.
        * Remaining D-K channels get `bits_lo` (default 3) -> 7 levels,
          same as baseline INT3 nsep+pchan.

    Effective bit rate (simulated-dequant space):
        bits/value = bits_lo + (bits_hi - bits_lo) * K / D
    For bits_lo=3, bits_hi=6, K=4, D=128: 3 + 3*4/128 = 3.09 bits/value
    -- within 3% of plain INT3, so we call it "INT3-class" despite
    the small surcharge.

    Per-head metadata: K channel indices at ceil(log2(D)) bits each
    (e.g., 4 * 7 = 28 bits per KV head per layer). Negligible at
    any realistic context length.

    Parameters
    ----------
    x : torch.Tensor, shape [..., T, D]
        Last dim D is the head_dim. Supports 2D [T, D], 3D [H, T, D],
        and 4D [B, H, T, D]. Top-K selection is computed independently
        for each leading-dim slice (so e.g. every (batch, head) picks
        its own top-K set).
    bits_lo : int, default 3
        Bit width for the non-plateau (D - K) channels.
    bits_hi : int, default 6
        Bit width for the top-K plateau channels. Should satisfy
        bits_hi > bits_lo; the 6-vs-3 default gives a 4x finer
        granularity on the outliers for a 3% bit-budget surcharge.
    k : int, default 4
        Number of channels that receive `bits_hi`. Must satisfy
        1 <= k < D.

    Returns
    -------
    torch.Tensor, same shape and dtype as x.

    Notes
    -----
    * Bit allocation is DATA-OBLIVIOUS in the sense that no external
      calibration corpus is needed: the top-K choice is made purely
      from the tensor being quantized at the moment of the call.
    * The decision is per-leading-dim (e.g., per-head). On GQA models
      this means K * n_kv_heads indices, not K * n_attention_heads.
    """
    if not (1 <= k < x.shape[-1]):
        raise ValueError(f"k must be in [1, D); got k={k}, D={x.shape[-1]}")
    if bits_hi <= bits_lo:
        raise ValueError(
            f"bits_hi ({bits_hi}) must exceed bits_lo ({bits_lo}) for the "
            f"2-tier reallocation to reduce MSE. Use qa_nsep_pchan for "
            f"uniform bit allocation."
        )

    dtype = x.dtype
    xf = x.float()

    # (1) Norm separation (identical to qa_nsep_pchan).
    n = xf.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    d = xf / n  # unit-direction

    # (2) Per-channel absmax on the direction. Shape: [..., 1, D]
    per_chan_max = d.abs().amax(dim=-2, keepdim=True).clamp(min=1e-12)

    # (3) Pick the top-K channel indices INDEPENDENTLY for each leading
    # slice (per-batch, per-head). Squeeze the singleton token axis first.
    pcm = per_chan_max.squeeze(-2)                # [..., D]
    _, topk_idx = pcm.topk(k, dim=-1)             # [..., K]
    mask = torch.zeros_like(pcm, dtype=torch.bool)
    mask.scatter_(-1, topk_idx, True)             # [..., D] True for top-K
    mask = mask.unsqueeze(-2)                     # [..., 1, D] broadcast-ready

    # (4) Quantize both tiers from the SAME per-channel absmax, only
    # the number of levels differs. This keeps the method a pure
    # bit-reallocation over the unmodified nsep+pchan scale source.
    qm_lo = 2 ** (bits_lo - 1) - 1
    qm_hi = 2 ** (bits_hi - 1) - 1
    s_lo = per_chan_max / qm_lo                   # [..., 1, D]
    s_hi = per_chan_max / qm_hi

    d_lo = (d / s_lo).round().clamp(-qm_lo, qm_lo) * s_lo
    d_hi = (d / s_hi).round().clamp(-qm_hi, qm_hi) * s_hi

    d_q = torch.where(mask, d_hi, d_lo)           # per-channel select

    # (5) Re-normalize onto the unit sphere to remove quantization-
    # induced norm drift, then multiply by the exact stored norm.
    d_q = d_q / d_q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return (n * d_q).to(dtype)


def qa_nsep_pchan_rotated(x, bits):
    """
    Walsh-Hadamard rotated nsep+pchan quantization (v0.3).

    Rotates the last dim via a deterministic Walsh-Hadamard matrix,
    applies nsep+pchan INT-`bits`, and unrotates. The rotation
    spreads any single-channel outlier magnitude uniformly across
    all channels before per-channel scales are computed, so the
    method sees an approximately Mode-2 distribution even when the
    input is Mode 1.

    Empirically the rotation is worth its cost only for Layer 0 of
    Mode 1 models (see :func:`modeq.rotate.should_rotate` for the
    dispatch policy). Mode 2 models gain essentially nothing
    because their per-channel scales are already well-balanced.

    Parameters
    ----------
    x : torch.Tensor, shape [..., T, D]
        Last dim D must be a power of 2 (typical: 64, 128, 256).
    bits : int
        Base bit width for the inner nsep+pchan call. 3 is the
        recommended INT3 target.

    Returns
    -------
    torch.Tensor, same shape and dtype as x.

    Notes
    -----
    Because H is orthonormal, the per-vector L2 norm is preserved
    exactly through rotation. The nsep+pchan decomposition thus
    sees the SAME norm distribution as in unrotated space; only the
    direction's per-channel scale profile changes.
    """
    dtype = x.dtype
    # Rotate in FP32 for numerical stability; qa_nsep_pchan runs in
    # FP32 internally so this is a no-extra-cast pipeline.
    x_rot = rotate(x)                       # FP32
    x_rot_q = qa_nsep_pchan(x_rot, bits)    # FP32 in, FP32 out
    x_back = unrotate(x_rot_q)              # FP32
    return x_back.to(dtype)


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
    if method == "nsep+pchan+rot":
        return qa_nsep_pchan_rotated(x, bits)
    if method == "nsep+pchan+2tier":
        # "bits" is interpreted as bits_lo; bits_hi defaults to bits+3,
        # K defaults to 4. For finer control, call qa_nsep_pchan_2tier
        # directly.
        return qa_nsep_pchan_2tier(x, bits_lo=bits, bits_hi=bits + 3, k=4)
    if method == "nsep+pchan+asym":
        return qa_nsep_pchan_asym(x, bits)
    raise ValueError(f"Unknown method: {method}")
