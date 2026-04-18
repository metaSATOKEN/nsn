# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

"""Unit tests for modeq.pack (INT3 bit-packing primitives)."""

import pytest
import torch

from modeq.pack import pack_int3_signed, unpack_int3_signed


def test_roundtrip_random():
    """Random INT3 values pack/unpack bit-exact."""
    torch.manual_seed(0)
    x = torch.randint(-3, 4, (2, 3, 64), dtype=torch.int8)
    packed = pack_int3_signed(x)
    unpacked = unpack_int3_signed(packed, N=64)
    assert torch.equal(x, unpacked)


def test_roundtrip_extremes():
    """Min (-3) and max (+3) values survive packing."""
    x = torch.tensor([[-3, -2, -1, 0, 1, 2, 3, -3] * 4], dtype=torch.int8)
    packed = pack_int3_signed(x)
    unpacked = unpack_int3_signed(packed, N=x.shape[-1])
    assert torch.equal(x, unpacked)


def test_compression_ratio():
    """Packed size is exactly 3N/8 bytes for N values."""
    N = 128
    x = torch.zeros(4, N, dtype=torch.int8)
    packed = pack_int3_signed(x)
    assert packed.shape[-1] == 3 * N // 8, (
        f"Expected {3*N//8} packed bytes, got {packed.shape[-1]}"
    )
    # 3 bits/value vs int8's 8 bits/value = 2.67x
    # vs fp16's 16 bits/value = 5.33x
    assert x.numel() * 2 / packed.numel() == pytest.approx(16 / 3)


def test_dim_requirement():
    """Last dim must be divisible by 8."""
    x = torch.zeros(4, 7, dtype=torch.int8)  # 7 not divisible by 8
    with pytest.raises(AssertionError):
        pack_int3_signed(x)


def test_strict_range_check():
    """pack_int3_signed with strict=True should reject OOR values."""
    x = torch.tensor([[-5, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.int8)  # -5 < -3
    with pytest.raises(ValueError):
        pack_int3_signed(x)

    x_over = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 7]], dtype=torch.int8)  # 7 > 3
    with pytest.raises(ValueError):
        pack_int3_signed(x_over)


def test_strict_false_silently_clamps():
    """pack_int3_signed with strict=False silently clamps OOR values."""
    x = torch.tensor([[-5, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.int8)
    packed = pack_int3_signed(x, strict=False)
    unpacked = unpack_int3_signed(packed, N=8)
    # -5 shifted +3 = -2 clamped to [0, 7] = 0, then unpack -> -3
    assert unpacked[0, 0].item() == -3


def test_multi_batch():
    """Batched tensors round-trip correctly."""
    torch.manual_seed(42)
    x = torch.randint(-3, 4, (2, 4, 8, 128), dtype=torch.int8)
    packed = pack_int3_signed(x)
    assert packed.shape == (2, 4, 8, 48)  # 3 * 128 / 8
    unpacked = unpack_int3_signed(packed, N=128)
    assert torch.equal(x, unpacked)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_roundtrip():
    """Pack/unpack works on GPU."""
    torch.manual_seed(0)
    x = torch.randint(-3, 4, (2, 3, 64), dtype=torch.int8, device="cuda")
    packed = pack_int3_signed(x)
    unpacked = unpack_int3_signed(packed, N=64)
    assert torch.equal(x, unpacked)
    assert packed.device == x.device
    assert unpacked.device == x.device
