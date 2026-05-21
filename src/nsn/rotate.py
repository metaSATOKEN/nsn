# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

"""
Walsh-Hadamard rotation helpers for KV cache quantization.

An orthonormal rotation applied before quantization spreads any
single-channel outlier magnitude uniformly across all channels,
turning a Mode-1 distribution (one dominant outlier channel) into
an approximately Mode-2 distribution (distributed norm variation).
Because the rotation is orthonormal, per-vector L2 norm is
preserved exactly, so our norm-separated quantizer `qa_nsep_pchan`
still sees identical norms and only the direction's per-channel
scales change.

We use the deterministic Walsh-Hadamard matrix (WHT) for three
reasons:

* **Data-free**: no calibration, no training, no RNG seed.
* **Symmetric and self-inverse** (after 1/sqrt(d) scaling): the
  forward and backward passes use the same matrix, and rotation
  is its own undo.
* **+-1 entries only**: a fused CUDA kernel could implement the
  transform in n*log(n) adds/subtracts per vector without any
  multiplies.

Constraints:

* Head dimension `d` must be a power of 2 (common for LLMs:
  64, 128, 256). Non-power-of-2 can be padded or handled via a
  generalized random orthogonal matrix; we do not implement that
  fallback in v0.3.

This module is used by :func:`nsn.quantize.qa_nsep_pchan_rotated`
and by :func:`nsn.mode_aware.select_quantizer` when the caller
supplies ``layer_idx`` and the mode is Mode 1 (rotate-when-needed
policy; see :func:`should_rotate`).
"""

import math
from typing import Optional

import torch


_HADAMARD_CACHE: dict = {}


def hadamard_matrix(
    d: int,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Construct the dxd Walsh-Hadamard matrix, scaled to be orthonormal.

    Parameters
    ----------
    d : int
        Dimension. Must be a positive power of 2.
    dtype : torch.dtype, default torch.float32
        Output dtype. FP32 is strongly recommended for numerical
        stability when composing with quantization.
    device : torch.device, optional
        Target device. Defaults to CPU.

    Returns
    -------
    torch.Tensor, shape (d, d)
        Orthonormal matrix H satisfying H @ H.T = I. For the
        Walsh-Hadamard construction, H is also symmetric
        (H == H.T), so rotation is self-inverse.

    Raises
    ------
    ValueError
        If d is not a positive power of 2.
    """
    if d <= 0 or (d & (d - 1)) != 0:
        raise ValueError(
            f"hadamard_matrix: d must be a positive power of 2, got {d}. "
            f"Typical LLM head dims (64, 128, 256) all satisfy this; "
            f"non-power-of-2 requires padding or a random orthogonal "
            f"fallback not implemented in v0.3."
        )

    H = torch.ones(1, 1, dtype=dtype, device=device)
    while H.shape[0] < d:
        top = torch.cat([H, H], dim=1)
        bot = torch.cat([H, -H], dim=1)
        H = torch.cat([top, bot], dim=0)
    return H / math.sqrt(d)


def cached_hadamard(
    d: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """
    Return a cached Walsh-Hadamard matrix keyed by (d, dtype, device).

    Building the matrix is O(d^2) and wasteful to repeat. A single
    (d=128, FP32) matrix is 64 KiB, so caching across the whole
    decode loop costs essentially nothing.
    """
    key = (d, dtype, str(device))
    cached = _HADAMARD_CACHE.get(key)
    if cached is None:
        cached = hadamard_matrix(d, dtype=dtype, device=device)
        _HADAMARD_CACHE[key] = cached
    return cached


def rotate(x: torch.Tensor) -> torch.Tensor:
    """
    Apply Walsh-Hadamard rotation to the last dim of `x`.

    The rotation is applied in FP32 for numerical stability. If `x`
    is FP16/BF16, the output is returned in FP32; callers can cast
    back explicitly. Upstream quantizers typically run in FP32
    internally, so this is usually what you want.

    Parameters
    ----------
    x : torch.Tensor, shape [..., D]
        Input tensor. Last dim D must be a power of 2.

    Returns
    -------
    torch.Tensor, shape [..., D], dtype FP32
        Rotated tensor. L2 norm along the last dim is preserved
        exactly (up to FP32 roundoff).
    """
    d = x.shape[-1]
    H = cached_hadamard(d, dtype=torch.float32, device=x.device)
    return x.float() @ H


def unrotate(x: torch.Tensor) -> torch.Tensor:
    """
    Inverse Walsh-Hadamard rotation on the last dim of `x`.

    For the symmetric Walsh-Hadamard H, unrotate == rotate (H is
    self-inverse after orthonormal scaling). We keep the function
    separate for call-site readability.

    Parameters
    ----------
    x : torch.Tensor, shape [..., D]

    Returns
    -------
    torch.Tensor, shape [..., D], dtype FP32
    """
    d = x.shape[-1]
    H = cached_hadamard(d, dtype=torch.float32, device=x.device)
    # For Walsh-Hadamard, H == H.T, so @ H.T is the same as @ H.
    # We write @ H.T for readability and in case someone swaps in a
    # non-symmetric orthonormal matrix later.
    return x.float() @ H.T


def should_rotate(mode: Optional[str], layer_idx: Optional[int]) -> bool:
    """
    Minimum-effective-dose rotation policy.

    Returns True only when rotation is empirically worth its cost:

    * Mode 1 (outlier-driven) AND layer_idx == 0.

    For Mode 2 models the rotation overhead is not justified by the
    marginal quality gain at INT3+; the unrotated nsep+pchan is
    already effectively lossless on such models. For non-Layer-0
    layers of Mode 1 models, outlier concentration is much weaker
    (by construction of our Mode 1 diagnosis), so rotation is again
    not worth it. This matches the `Layer 0 causal isolation`
    finding: ~99.97% of Mode 1 damage is attributable to Layer 0 K.

    Parameters
    ----------
    mode : str, optional
        Output of :func:`nsn.detect.classify_mode` or a manual
        label. Case- and whitespace-insensitive. None returns False
        (no rotation) - caller must opt in by providing a mode.
    layer_idx : int, optional
        Layer index in the transformer stack (0 is the first block).
        None returns False (no rotation) - caller must specify layer.

    Returns
    -------
    bool
        True to apply Hadamard rotation before quantization.
    """
    if mode is None or layer_idx is None:
        return False
    mode_n = mode.lower().replace(" ", "").replace("-", "")
    return mode_n == "mode1" and layer_idx == 0
