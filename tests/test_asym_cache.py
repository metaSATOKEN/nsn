# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

"""Tests for pack_int3_unsigned and Int3AsymNsepPchanCache (v0.3)."""

import pytest
import torch

from modeq.pack import (
    pack_int3_unsigned,
    unpack_int3_unsigned,
)
from modeq.cache import Int3AsymNsepPchanCache


# ---------------------------------------------------------------------------
# pack_int3_unsigned / unpack_int3_unsigned
# ---------------------------------------------------------------------------

def test_unsigned_roundtrip_full_range():
    """All 8 levels survive the pack/unpack roundtrip."""
    x = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.int8)
    packed = pack_int3_unsigned(x)
    unpacked = unpack_int3_unsigned(packed, N=8)
    assert torch.equal(x, unpacked)


def test_unsigned_roundtrip_random():
    torch.manual_seed(0)
    x = torch.randint(0, 8, (2, 3, 64), dtype=torch.int8)
    packed = pack_int3_unsigned(x)
    unpacked = unpack_int3_unsigned(packed, N=64)
    assert torch.equal(x, unpacked)


def test_unsigned_compression_ratio():
    """Packed size = 3N/8 bytes."""
    N = 128
    x = torch.zeros(4, N, dtype=torch.int8)
    packed = pack_int3_unsigned(x)
    assert packed.shape[-1] == 3 * N // 8


def test_unsigned_strict_range():
    with pytest.raises(ValueError, match=r"\[0, 7\]"):
        pack_int3_unsigned(torch.tensor([[-1, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.int8))
    with pytest.raises(ValueError, match=r"\[0, 7\]"):
        pack_int3_unsigned(torch.tensor([[8, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.int8))


def test_unsigned_strict_false_clamps():
    """strict=False clamps input to [0, 7]."""
    x = torch.tensor([[10, -2, 3, 5, 0, 7, 1, 4]], dtype=torch.int8)
    packed = pack_int3_unsigned(x, strict=False)
    unpacked = unpack_int3_unsigned(packed, N=8)
    # 10 -> 7, -2 -> 0, rest stay
    assert unpacked[0, 0].item() == 7
    assert unpacked[0, 1].item() == 0
    assert unpacked.min().item() >= 0
    assert unpacked.max().item() <= 7


# ---------------------------------------------------------------------------
# Int3AsymNsepPchanCache round-trip
# ---------------------------------------------------------------------------

def test_asym_cache_roundtrip_2d():
    """2D [T, D] input round-trips through cache close to the FP16 input."""
    torch.manual_seed(0)
    x = torch.randn(64, 128, dtype=torch.float16)
    cache = Int3AsymNsepPchanCache.from_fp16(x)
    y = cache.to_fp16()
    assert y.shape == x.shape
    assert y.dtype == torch.float16

    # Reconstructed should be close (but not identical) to input.
    err = (x.float() - y.float()).abs().mean().item()
    assert 1e-3 < err < 1.0, (
        f"Reconstruction error unexpected: {err}"
    )


def test_asym_cache_roundtrip_4d():
    """4D [B, H, T, D] input also works."""
    torch.manual_seed(0)
    x = torch.randn(1, 8, 64, 128, dtype=torch.float16)
    cache = Int3AsymNsepPchanCache.from_fp16(x)
    y = cache.to_fp16()
    assert y.shape == x.shape


def test_asym_cache_matches_simulated():
    """Real-packed cache reconstruction should match simulated
    qa_nsep_pchan_asym reconstruction within 1e-3."""
    from modeq.quantize import qa_nsep_pchan_asym

    torch.manual_seed(0)
    x = torch.randn(1, 4, 32, 128, dtype=torch.float16)
    y_sim = qa_nsep_pchan_asym(x, bits=3)

    cache = Int3AsymNsepPchanCache.from_fp16(x)
    y_cache = cache.to_fp16()

    diff = (y_sim.float() - y_cache.float()).abs().max().item()
    assert diff < 1e-2, (
        f"Packed cache reconstruction diverges from simulated by {diff}"
    )


def test_asym_cache_head_dim_requirement():
    """head_dim must be divisible by 8 for 3-bit packing."""
    x = torch.randn(64, 130, dtype=torch.float16)  # 130 not divisible by 8
    with pytest.raises(AssertionError, match="divisible by 8"):
        Int3AsymNsepPchanCache.from_fp16(x)


def test_asym_cache_size_bytes():
    """size_bytes() matches the sum of each stored tensor's byte count."""
    torch.manual_seed(0)
    x = torch.randn(64, 128, dtype=torch.float16)
    cache = Int3AsymNsepPchanCache.from_fp16(x)
    expected = (
        cache.packed_d.numel() * 1
        + cache.norm.numel() * 2
        + cache.d_min.numel() * 2
        + cache.d_step.numel() * 2
    )
    assert cache.size_bytes() == expected


def test_asym_cache_compression_vs_fp16():
    """Real compression ratio for a realistic tensor shape."""
    T, D = 1024, 128
    x = torch.randn(1, 8, T, D, dtype=torch.float16)
    fp16_bytes = x.numel() * 2  # 2 bytes/value
    cache = Int3AsymNsepPchanCache.from_fp16(x)
    cached_bytes = cache.size_bytes()
    ratio = fp16_bytes / cached_bytes
    # Theoretical: 16 bits / (3.094 bits + a little metadata) ~ 5.0x
    # We expect 4.5x to 5.2x for this shape.
    assert 4.0 < ratio < 5.5, (
        f"Unexpected compression ratio: fp16={fp16_bytes}, "
        f"cached={cached_bytes}, ratio={ratio:.2f}x"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_asym_cache_cuda():
    """Cache works on GPU."""
    torch.manual_seed(0)
    x = torch.randn(1, 4, 64, 128, dtype=torch.float16, device="cuda")
    cache = Int3AsymNsepPchanCache.from_fp16(x)
    y = cache.to_fp16()
    assert y.device == x.device
    assert y.shape == x.shape


# ---------------------------------------------------------------------------
# to_fp16_compiled (v0.3.12): torch.compile-accelerated unpack
# ---------------------------------------------------------------------------

def test_to_fp16_compiled_matches_eager_cpu():
    """Compiled unpack must match eager to_fp16() within FP16 rounding.
    CPU path, so small tensors only; the full Phase-27 CUDA measurement
    lives under experiments/phase27_decode_latency.py.
    """
    torch.manual_seed(0)
    x = torch.randn(4, 64, 128, dtype=torch.float16)
    cache = Int3AsymNsepPchanCache.from_fp16(x)

    eager    = cache.to_fp16()
    compiled = cache.to_fp16_compiled()

    assert eager.shape == compiled.shape
    assert eager.dtype == compiled.dtype
    # Within FP16 rounding (the compile path processes through fp32
    # intermediaries identically to to_fp16 but the operator order
    # may differ slightly after fusion).
    max_abs_diff = (eager.float() - compiled.float()).abs().max().item()
    assert max_abs_diff < 1e-2, f"compiled vs eager diverges: {max_abs_diff}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_to_fp16_compiled_matches_eager_cuda():
    """Same check on CUDA, where the compile path actually matters."""
    torch.manual_seed(0)
    x = torch.randn(8, 1024, 128, dtype=torch.float16, device="cuda")
    cache = Int3AsymNsepPchanCache.from_fp16(x)

    eager    = cache.to_fp16()
    compiled = cache.to_fp16_compiled()

    assert eager.shape == compiled.shape
    max_abs_diff = (eager.float() - compiled.float()).abs().max().item()
    assert max_abs_diff < 1e-2, f"compiled vs eager diverges: {max_abs_diff}"


def test_compiled_unpack_is_cached_at_class_level():
    """The first call populates Int3AsymNsepPchanCache._compiled_unpack;
    subsequent calls reuse it instead of recompiling every time.
    """
    # Reset any prior compile cache so this test is self-contained
    Int3AsymNsepPchanCache._compiled_unpack = None

    x = torch.randn(2, 16, 64, dtype=torch.float16)
    cache = Int3AsymNsepPchanCache.from_fp16(x)
    _ = cache.to_fp16_compiled()

    # After one call, the class-level cache must be non-None
    assert Int3AsymNsepPchanCache._compiled_unpack is not None

    # A second instance should reuse the same compiled object
    first_ref = Int3AsymNsepPchanCache._compiled_unpack
    cache2 = Int3AsymNsepPchanCache.from_fp16(x)
    _ = cache2.to_fp16_compiled()
    assert Int3AsymNsepPchanCache._compiled_unpack is first_ref
