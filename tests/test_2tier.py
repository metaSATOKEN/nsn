# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

"""Unit tests for qa_nsep_pchan_2tier (v0.3 pivot, top-K mixed precision)."""

import pytest
import torch

from nsn.quantize import qa_nsep_pchan, qa_nsep_pchan_2tier


# ---------------------------------------------------------------------------
# Shape / dtype sanity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [(32, 128), (4, 32, 128), (2, 8, 32, 128)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_2tier_preserves_shape_and_dtype(shape, dtype):
    x = torch.randn(*shape, dtype=dtype)
    y = qa_nsep_pchan_2tier(x, bits_lo=3, bits_hi=6, k=4)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_2tier_not_identity():
    """Sanity: quantizer actually quantizes (not a no-op)."""
    torch.manual_seed(0)
    x = torch.randn(64, 128, dtype=torch.float32)
    y = qa_nsep_pchan_2tier(x, bits_lo=3, bits_hi=6, k=4)
    assert (x - y).abs().mean().item() > 1e-3


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------

def test_2tier_rejects_bad_k():
    x = torch.randn(16, 128)
    with pytest.raises(ValueError, match="k must"):
        qa_nsep_pchan_2tier(x, k=0)
    with pytest.raises(ValueError, match="k must"):
        qa_nsep_pchan_2tier(x, k=128)        # k must be strictly < D
    with pytest.raises(ValueError, match="k must"):
        qa_nsep_pchan_2tier(x, k=200)


def test_2tier_rejects_bits_hi_le_lo():
    x = torch.randn(16, 128)
    with pytest.raises(ValueError, match="bits_hi"):
        qa_nsep_pchan_2tier(x, bits_lo=4, bits_hi=3, k=4)
    with pytest.raises(ValueError, match="bits_hi"):
        qa_nsep_pchan_2tier(x, bits_lo=3, bits_hi=3, k=4)


# ---------------------------------------------------------------------------
# Correctness: 2-tier beats plain on the plateau pattern (v0.3 thesis)
# ---------------------------------------------------------------------------

def test_2tier_beats_plain_on_plateau_input():
    """Core claim: on a Qwen2-7B-like plateau (5 channels with magnitude
    ~150, rest ~10), 2-tier INT3+6 should reduce MSE substantially
    compared to plain INT3."""
    torch.manual_seed(0)
    T, D = 1024, 128
    x = torch.randn(T, D, dtype=torch.float32) * 10.0
    # Plateau: 5 channels at ~150 magnitude
    plateau_chans = [7, 17, 31, 47, 89]
    for c in plateau_chans:
        x[:, c] = torch.randn(T, dtype=torch.float32) * 150.0

    mse_plain = (x - qa_nsep_pchan(x, bits=3)).pow(2).mean().item()
    mse_2tier = (x - qa_nsep_pchan_2tier(x, bits_lo=3, bits_hi=6, k=5)).pow(2).mean().item()

    # Expect at least a 1.5x MSE reduction on this synthetic plateau.
    # On a similar but stronger plateau (lower noise floor), ratios of
    # 3x+ are achievable; the 1.5x floor protects against regressions
    # without being brittle to random-seed details.
    assert mse_2tier < mse_plain / 1.5, (
        f"2-tier should cut MSE by at least 1.5x on a plateau input: "
        f"plain={mse_plain:.5f}, 2tier={mse_2tier:.5f}, "
        f"ratio={mse_plain / mse_2tier:.2f}x"
    )


def test_2tier_close_to_plain_on_uniform_gaussian():
    """Safety: on uniform Gaussian (no plateau), 2-tier and plain should
    produce similar MSE -- 2-tier is wasting bits on arbitrary channels
    but not CATASTROPHICALLY worse."""
    torch.manual_seed(0)
    x = torch.randn(512, 128, dtype=torch.float32) * 0.3

    mse_plain = (x - qa_nsep_pchan(x, bits=3)).pow(2).mean().item()
    mse_2tier = (x - qa_nsep_pchan_2tier(x, bits_lo=3, bits_hi=6, k=4)).pow(2).mean().item()

    # 2-tier should actually help a LITTLE on Gaussian too (since it
    # spends more bits on the top-variance channels), but if it's ever
    # more than 2x worse we have a bug.
    assert mse_2tier < 2 * mse_plain, (
        f"2-tier is unexpectedly much worse than plain on Gaussian: "
        f"plain={mse_plain:.5f}, 2tier={mse_2tier:.5f}"
    )


# ---------------------------------------------------------------------------
# Top-K is selected independently per leading dim (per-head etc.)
# ---------------------------------------------------------------------------

def test_2tier_per_head_topk_is_independent():
    """For 3D input [H, T, D], each head should pick its own top-K
    (verified by constructing heads with outliers on disjoint channels
    and checking that each head's reconstruction is accurate on its
    own outliers)."""
    torch.manual_seed(0)
    H, T, D = 4, 256, 128
    x = torch.randn(H, T, D, dtype=torch.float32) * 1.0

    # Head h gets its outlier on channel (h * 10 + 5).
    outlier_chans = []
    for h in range(H):
        c = h * 10 + 5
        outlier_chans.append(c)
        x[h, :, c] = torch.randn(T, dtype=torch.float32) * 100.0

    y_2tier = qa_nsep_pchan_2tier(x, bits_lo=3, bits_hi=6, k=4)
    y_plain = qa_nsep_pchan(x, bits=3)

    # Per-head, per-outlier-channel reconstruction error should be
    # noticeably smaller under 2-tier (since each head's top-K must
    # include that head's outlier channel).
    for h, c in enumerate(outlier_chans):
        err_plain = (x[h, :, c] - y_plain[h, :, c]).pow(2).mean().item()
        err_2tier = (x[h, :, c] - y_2tier[h, :, c]).pow(2).mean().item()
        assert err_2tier < err_plain, (
            f"Head {h} outlier channel {c}: 2-tier should win, "
            f"got plain={err_plain:.4f}, 2tier={err_2tier:.4f}"
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_2tier_k_equals_1():
    """K=1 (only the single largest channel gets bits_hi) still works."""
    torch.manual_seed(0)
    x = torch.randn(64, 128)
    y = qa_nsep_pchan_2tier(x, bits_lo=3, bits_hi=6, k=1)
    assert y.shape == x.shape


def test_2tier_k_near_d():
    """K = D-1 (almost every channel gets bits_hi) still works."""
    torch.manual_seed(0)
    x = torch.randn(64, 128)
    y = qa_nsep_pchan_2tier(x, bits_lo=3, bits_hi=6, k=127)
    assert y.shape == x.shape
    # Should be close to pure qa_nsep_pchan(x, 6) since 127/128 channels
    # run in INT6 mode; the reconstruction error should be smaller than
    # plain INT3.
    mse_2tier = (x - y).pow(2).mean().item()
    mse_plain_int3 = (x - qa_nsep_pchan(x, bits=3)).pow(2).mean().item()
    assert mse_2tier < mse_plain_int3


# ---------------------------------------------------------------------------
# apply_method dispatch integration
# ---------------------------------------------------------------------------

def test_apply_method_dispatch_2tier():
    from nsn.quantize import apply_method
    torch.manual_seed(0)
    x = torch.randn(32, 128, dtype=torch.float32)
    y = apply_method(x, "nsep+pchan+2tier", bits=3)
    assert y.shape == x.shape
    # Dispatch should use default K=4, bits_hi=6.
    y_direct = qa_nsep_pchan_2tier(x, bits_lo=3, bits_hi=6, k=4)
    assert torch.allclose(y, y_direct, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_2tier_cuda():
    torch.manual_seed(0)
    x = torch.randn(16, 128, dtype=torch.float16, device="cuda")
    y = qa_nsep_pchan_2tier(x, bits_lo=3, bits_hi=6, k=4)
    assert y.device == x.device
    assert y.dtype == x.dtype
