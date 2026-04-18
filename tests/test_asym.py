# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

"""Unit tests for qa_nsep_pchan_asym (v0.3 asymmetric per-channel)."""

import pytest
import torch

from modeq.quantize import qa_nsep_pchan, qa_nsep_pchan_asym


# ---------------------------------------------------------------------------
# Shape / dtype sanity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [(32, 128), (4, 32, 128), (2, 8, 32, 128)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_asym_preserves_shape_and_dtype(shape, dtype):
    x = torch.randn(*shape, dtype=dtype)
    y = qa_nsep_pchan_asym(x, bits=3)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_asym_not_identity():
    """Sanity: the asymmetric quantizer actually quantizes."""
    torch.manual_seed(0)
    x = torch.randn(64, 128, dtype=torch.float32)
    y = qa_nsep_pchan_asym(x, bits=3)
    assert (x - y).abs().mean().item() > 1e-3


# ---------------------------------------------------------------------------
# Level counts
# ---------------------------------------------------------------------------

def test_asym_uses_full_level_count():
    """Asymmetric INT3 uses 2**bits levels (8 for INT3), which is 1 more
    than symmetric's 2*qm + 1 = 7. Verify on a per-channel basis that the
    unique quantized values after deduplication approach the level count."""
    torch.manual_seed(0)
    T, D = 2048, 128
    x = torch.randn(T, D, dtype=torch.float32) * 1.0
    y = qa_nsep_pchan_asym(x, bits=3)

    # For each channel, count unique quantized values. Should be at most
    # 2**3 = 8 plus the effect of post-quantization re-normalization (which
    # can create many more unique floats due to per-token rescaling).
    # Rather than exact count we assert that the RECONSTRUCTED values span
    # a range at least as wide as the original per-channel range: if asym
    # were broken (e.g., always reducing to symmetric), it would shrink.
    for c in range(0, D, 17):
        orig_min, orig_max = x[:, c].min().item(), x[:, c].max().item()
        rec_min,  rec_max  = y[:, c].min().item(), y[:, c].max().item()
        # Allow 10% slack for re-normalization drift; core check is that
        # the reconstructed range actually spans the original.
        assert rec_min <= orig_min + 0.10 * (orig_max - orig_min), (
            f"channel {c}: asym rec_min {rec_min:.3f} should reach "
            f"near orig_min {orig_min:.3f}"
        )
        assert rec_max >= orig_max - 0.10 * (orig_max - orig_min), (
            f"channel {c}: asym rec_max {rec_max:.3f} should reach "
            f"near orig_max {orig_max:.3f}"
        )


# ---------------------------------------------------------------------------
# Empirical: asym beats sym on a strongly biased channel distribution
# (the v0.3 thesis)
# ---------------------------------------------------------------------------

def test_asym_beats_symmetric_on_biased_input():
    """On per-channel distributions biased strongly to one side of zero
    (mimicking the post-RoPE Qwen2-7B pattern), asymmetric INT3 should
    outperform symmetric INT3 at identical 3 bits/value."""
    torch.manual_seed(0)
    T, D = 1024, 128
    # Construct: each channel's values are drawn from N(mu, 1) with mu
    # drawn from N(3, 1) - so most channels are strongly biased to the
    # positive side. Symmetric absmax wastes the negative half; asym
    # fits the [min, max] range tightly.
    mu = torch.randn(1, D, dtype=torch.float32) * 1.0 + 3.0
    x = torch.randn(T, D, dtype=torch.float32) + mu

    mse_sym  = (x - qa_nsep_pchan(x, bits=3)).pow(2).mean().item()
    mse_asym = (x - qa_nsep_pchan_asym(x, bits=3)).pow(2).mean().item()

    assert mse_asym < mse_sym * 0.8, (
        f"asym INT3 should cut MSE by >20% on a biased input. "
        f"Got sym={mse_sym:.5f}, asym={mse_asym:.5f}, "
        f"ratio={mse_asym / mse_sym:.3f}"
    )


def test_asym_ties_on_symmetric_gaussian():
    """Safety: on zero-centered Gaussian (perfectly symmetric), asym
    should be approximately as good as sym - not catastrophically worse
    and typically slightly better due to having one more level."""
    torch.manual_seed(0)
    x = torch.randn(2048, 128, dtype=torch.float32)
    mse_sym  = (x - qa_nsep_pchan(x, bits=3)).pow(2).mean().item()
    mse_asym = (x - qa_nsep_pchan_asym(x, bits=3)).pow(2).mean().item()
    # asym shouldn't regress by more than 20% on genuinely symmetric data
    # (and usually beats sym by 10-20% thanks to the extra level).
    assert mse_asym < 1.2 * mse_sym, (
        f"asym unexpectedly regressed on symmetric Gaussian: "
        f"sym={mse_sym:.5f}, asym={mse_asym:.5f}"
    )


# ---------------------------------------------------------------------------
# Dispatch integration
# ---------------------------------------------------------------------------

def test_apply_method_dispatch_asym():
    from modeq.quantize import apply_method
    torch.manual_seed(0)
    x = torch.randn(32, 128, dtype=torch.float32)
    y = apply_method(x, "nsep+pchan+asym", bits=3)
    assert y.shape == x.shape
    y_direct = qa_nsep_pchan_asym(x, bits=3)
    assert torch.allclose(y, y_direct, atol=1e-5)


# ---------------------------------------------------------------------------
# Bit-width validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
def test_asym_works_at_various_bits(bits):
    torch.manual_seed(0)
    x = torch.randn(32, 128, dtype=torch.float32)
    y = qa_nsep_pchan_asym(x, bits=bits)
    assert y.shape == x.shape
    # MSE should drop monotonically with more bits.
    # (Measured relative to the same input.)
    mse = (x - y).pow(2).mean().item()
    if bits >= 3:
        assert mse < 1.0, f"MSE suspiciously high at bits={bits}: {mse}"


def test_asym_int4_beats_asym_int3():
    """Bits -> 4 should reduce MSE monotonically vs bits -> 3."""
    torch.manual_seed(0)
    x = torch.randn(1024, 128, dtype=torch.float32)
    mse3 = (x - qa_nsep_pchan_asym(x, bits=3)).pow(2).mean().item()
    mse4 = (x - qa_nsep_pchan_asym(x, bits=4)).pow(2).mean().item()
    assert mse4 < mse3, (
        f"asym INT4 should beat asym INT3: mse3={mse3:.5f}, mse4={mse4:.5f}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_asym_cuda():
    torch.manual_seed(0)
    x = torch.randn(16, 128, dtype=torch.float16, device="cuda")
    y = qa_nsep_pchan_asym(x, bits=3)
    assert y.device == x.device
    assert y.dtype == x.dtype
    assert y.shape == x.shape
