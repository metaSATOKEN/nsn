# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

"""
Real 3-bit packing for INT3 tensors.

Eight 3-bit values are packed into 24 bits = 3 uint8 bytes,
giving exact 3 bits/value storage. Combined with nsep+pchan
quantization (see `modeq.cache`), this delivers 5.33x
compression vs FP16 in theory, measured at 5.08x on
Llama-3.1-8B and Qwen2-7B (overhead ~5% from FP16 storage of
per-token norms and per-channel scales).

The implementation uses int64 intermediate arithmetic for
readability. A fused CUDA kernel would pack with native uint
bit-ops; the Python version here adds ~20% decode latency,
which a kernel could remove entirely (low arithmetic intensity).
"""

import torch


def pack_int3_signed(x_int8, strict=True):
    """
    Pack signed INT3 values (range [-3, 3]) into compact uint8 bytes.

    Parameters
    ----------
    x_int8 : torch.Tensor, dtype int8
        Input tensor with last dim N, where N is divisible by 8.
        Values must be in [-3, 3] (INT3 symmetric range).
    strict : bool, default True
        If True (default), raise a ValueError when ``x_int8`` contains
        values outside [-3, 3]. If False, silently clamp them to the
        range -- useful for numerical-edge cases but dangerous in
        general because unpacking can surface an out-of-range value
        (shifted [0, 7] -> [-3, 4]) that wasn't in the advertised
        symmetric set.

    Returns
    -------
    torch.Tensor, dtype uint8
        Packed tensor with last dim 3*N/8.

    Notes
    -----
    Bit layout: each group of 8 values packs into 3 bytes (24 bits).
    Value i occupies bits [3*i, 3*i+2] of the 24-bit group,
    stored LSB-first across the 3 output bytes.
    """
    *prefix, N = x_int8.shape
    assert N % 8 == 0, f"Last dim must be divisible by 8, got {N}"

    if strict:
        xmin = int(x_int8.min().item())
        xmax = int(x_int8.max().item())
        if xmin < -3 or xmax > 3:
            raise ValueError(
                f"pack_int3_signed requires values in [-3, 3] (INT3 "
                f"symmetric range); got [{xmin}, {xmax}]. Pass "
                f"strict=False to silently clamp."
            )

    # Shift [-3, 3] to [0, 6] (unsigned 3-bit, value 7 unused)
    u = (x_int8.to(torch.int64) + 3).clamp(0, 7)
    u = u.view(*prefix, N // 8, 8)

    # Pack 8 values into a single int64 (24 bits used)
    shifts = torch.arange(8, device=u.device, dtype=torch.int64) * 3
    packed64 = (u << shifts).sum(dim=-1)  # [..., N/8]

    # Split into 3 bytes (LSB first)
    b0 = (packed64 & 0xFF).to(torch.uint8)
    b1 = ((packed64 >> 8) & 0xFF).to(torch.uint8)
    b2 = ((packed64 >> 16) & 0xFF).to(torch.uint8)

    return torch.stack([b0, b1, b2], dim=-1).view(*prefix, 3 * N // 8)


def unpack_int3_signed(bytes_in, N):
    """
    Unpack compact bytes back to signed INT3 values.

    Parameters
    ----------
    bytes_in : torch.Tensor, dtype uint8
        Packed tensor with last dim 3*N/8 (output of pack_int3_signed).
    N : int
        Original number of values (must be divisible by 8).

    Returns
    -------
    torch.Tensor, dtype int8
        Unpacked tensor with last dim N, values in [-3, 3].
    """
    *prefix, P = bytes_in.shape
    assert P == 3 * N // 8, f"Expected {3*N//8} bytes for N={N}, got {P}"

    grouped = bytes_in.view(*prefix, N // 8, 3)
    b0, b1, b2 = grouped[..., 0], grouped[..., 1], grouped[..., 2]
    packed64 = (
        b0.to(torch.int64)
        | (b1.to(torch.int64) << 8)
        | (b2.to(torch.int64) << 16)
    )

    shifts = torch.arange(8, device=packed64.device, dtype=torch.int64) * 3
    mask = 7  # 3-bit mask
    u = (packed64.unsqueeze(-1) >> shifts) & mask  # [..., N/8, 8]

    # Shift [0, 6] back to [-3, 3]
    return (u - 3).view(*prefix, N).to(torch.int8)
