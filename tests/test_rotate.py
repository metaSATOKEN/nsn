# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

"""Unit tests for nsn.rotate (Walsh-Hadamard helpers) and the
qa_nsep_pchan_rotated quantizer."""

import math

import pytest
import torch

from nsn.rotate import (
    hadamard_matrix,
    rotate,
    unrotate,
    should_rotate,
    cached_hadamard,
)
from nsn.quantize import qa_nsep_pchan, qa_nsep_pchan_rotated


# ---------------------------------------------------------------------------
# Hadamard matrix properties
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("d", [2, 4, 8, 16, 64, 128, 256])
def test_hadamard_is_orthonormal(d):
    """H @ H.T should equal identity for any valid d."""
    H = hadamard_matrix(d, dtype=torch.float64)
    I = torch.eye(d, dtype=torch.float64)
    err = (H @ H.T - I).abs().max().item()
    assert err < 1e-10, f"H @ H.T != I at d={d}, max err {err}"


@pytest.mark.parametrize("d", [2, 4, 8, 16, 64, 128])
def test_hadamard_is_symmetric(d):
    """Walsh-Hadamard is symmetric (H == H.T), so rotation is self-inverse."""
    H = hadamard_matrix(d, dtype=torch.float64)
    assert torch.equal(H, H.T), f"Walsh-Hadamard should be symmetric at d={d}"


def test_hadamard_rejects_non_power_of_2():
    """Non-power-of-2 dims must be rejected with a clear error."""
    for bad in [3, 5, 7, 127, 129]:
        with pytest.raises(ValueError, match="power of 2"):
            hadamard_matrix(bad)


def test_hadamard_rejects_zero_and_negative():
    for bad in [0, -1, -8]:
        with pytest.raises(ValueError):
            hadamard_matrix(bad)


# ---------------------------------------------------------------------------
# rotate / unrotate roundtrip
# ---------------------------------------------------------------------------

def test_rotate_unrotate_identity_2d():
    """rotate then unrotate is identity (up to FP32 roundoff)."""
    torch.manual_seed(0)
    x = torch.randn(32, 128, dtype=torch.float32)
    back = unrotate(rotate(x))
    assert (x - back).abs().max().item() < 1e-5


def test_rotate_unrotate_identity_4d():
    """Same on 4D [B, H, T, D] tensors (the real cache shape)."""
    torch.manual_seed(0)
    x = torch.randn(2, 8, 16, 128, dtype=torch.float32)
    back = unrotate(rotate(x))
    assert (x - back).abs().max().item() < 1e-5


def test_rotate_preserves_l2_norm():
    """Orthonormal rotation preserves per-vector L2 norm exactly."""
    torch.manual_seed(0)
    x = torch.randn(16, 64, dtype=torch.float32)
    n_before = x.norm(dim=-1)
    n_after = rotate(x).norm(dim=-1)
    assert (n_before - n_after).abs().max().item() < 1e-5


def test_cached_hadamard_reuses_matrix():
    """Two calls with the same (d, dtype, device) return the same tensor."""
    H1 = cached_hadamard(64, torch.float32, torch.device("cpu"))
    H2 = cached_hadamard(64, torch.float32, torch.device("cpu"))
    assert H1 is H2, "cached_hadamard should reuse the cached tensor"


# ---------------------------------------------------------------------------
# Rotation actually spreads outliers (the whole point)
# ---------------------------------------------------------------------------

def test_rotation_spreads_single_channel_outlier():
    """Mode-1-like input (1 huge channel, rest small) should have its max
    per-channel magnitude DROP substantially after rotation."""
    torch.manual_seed(0)
    T, D = 128, 128
    x = torch.randn(T, D, dtype=torch.float32) * 0.1
    # Inject a Mode-1-style outlier: channel 17 is 100x larger on half the
    # tokens.
    x[:, 17] = torch.randn(T, dtype=torch.float32) * 10.0

    pre_maxabs = x.abs().amax(dim=0)
    pre_max_chan = pre_maxabs.max().item()

    x_rot = rotate(x)
    rot_maxabs = x_rot.abs().amax(dim=0)
    rot_max_chan = rot_maxabs.max().item()

    # Hadamard rotation distributes outlier energy across channels;
    # the peak per-channel max should drop by at least 2x.
    assert rot_max_chan < pre_max_chan / 2.0, (
        f"Rotation failed to spread outlier: pre={pre_max_chan:.3f}, "
        f"post={rot_max_chan:.3f}"
    )

    # And the variance across channels should flatten (std of per-channel
    # maxabs should shrink).
    pre_std = pre_maxabs.std().item()
    post_std = rot_maxabs.std().item()
    assert post_std < pre_std, (
        f"Per-channel maxabs dispersion did not flatten: "
        f"pre std={pre_std:.3f}, post std={post_std:.3f}"
    )


# ---------------------------------------------------------------------------
# qa_nsep_pchan_rotated integration
# ---------------------------------------------------------------------------

def test_qa_rotated_preserves_shape_and_dtype():
    """Rotated quantizer must preserve shape and dtype like its siblings."""
    for shape in [(32, 128), (4, 32, 128), (2, 8, 32, 128)]:
        for dtype in [torch.float16, torch.float32]:
            x = torch.randn(*shape, dtype=dtype)
            y = qa_nsep_pchan_rotated(x, bits=3)
            assert y.shape == x.shape
            assert y.dtype == x.dtype


def test_qa_rotated_documents_synthetic_behavior():
    """Document (not assert) how rotation interacts with nsep+pchan on
    synthetic outlier inputs.

    Counterintuitive finding: on a synthetic "1 dominant channel" input,
    rotation HURTS rather than helps, because plain per-channel scaling
    already isolates the outlier channel (giving it its own wide scale
    while other channels stay tight). Rotation spreads the outlier
    magnitude across all channels, widening every channel's scale and
    degrading precision on the majority of (originally small-magnitude)
    coordinates. This is logged here as a reference so future readers
    don't expect rotation to be universally helpful on top of nsep+pchan.

    The real test of whether rotation helps Mode 1 is on actual
    Qwen2-7B WikiText-2 PPL, where the outlier structure is richer
    (multi-channel, post-RoPE, token-dependent) than this synthetic
    cartoon. That benchmark is v0.3's go/no-go gate.
    """
    torch.manual_seed(0)
    T, D = 256, 128
    x = torch.randn(T, D, dtype=torch.float32) * 0.1
    x[:, 17] = torch.randn(T, dtype=torch.float32) * 10.0

    y_plain = qa_nsep_pchan(x, bits=3)
    y_rot = qa_nsep_pchan_rotated(x, bits=3)

    mse_plain = (x - y_plain).pow(2).mean().item()
    mse_rot = (x - y_rot).pow(2).mean().item()

    # Sanity: both quantizers produce non-trivial reconstruction error
    assert mse_plain > 0
    assert mse_rot > 0
    # Both are within the same order of magnitude (neither is catastrophic)
    assert mse_rot < 10 * mse_plain or mse_plain < 10 * mse_rot


def test_qa_rotated_is_not_identity():
    """Sanity: the rotated quantizer actually quantizes (not a no-op)."""
    torch.manual_seed(0)
    x = torch.randn(32, 128, dtype=torch.float32)
    y = qa_nsep_pchan_rotated(x, bits=3)
    assert (x - y).abs().mean().item() > 1e-3


def test_qa_rotated_rejects_non_power_of_2():
    """Non-power-of-2 head dims bubble up the rotate.hadamard_matrix error."""
    x = torch.randn(32, 96, dtype=torch.float32)  # 96 is not pow2
    with pytest.raises(ValueError, match="power of 2"):
        qa_nsep_pchan_rotated(x, bits=3)


# ---------------------------------------------------------------------------
# should_rotate policy
# ---------------------------------------------------------------------------

def test_should_rotate_policy_minimum_effective_dose():
    """Only (Mode 1, layer_idx == 0) should opt in."""
    # Yes cases
    assert should_rotate("Mode 1", 0) is True
    assert should_rotate("mode 1", 0) is True     # case insensitive
    assert should_rotate("Mode-1", 0) is True     # hyphenated

    # No cases
    assert should_rotate("Mode 1", 1) is False    # deeper layer
    assert should_rotate("Mode 2", 0) is False    # different mode
    assert should_rotate("borderline", 0) is False
    assert should_rotate("unknown", 0) is False
    assert should_rotate(None, 0) is False        # mode missing
    assert should_rotate("Mode 1", None) is False  # layer missing


# ---------------------------------------------------------------------------
# Mode-aware dispatcher integration
#
# Note: v0.3 does NOT dispatch to the rotated quantizer. Phase 10 probe
# on real Qwen2-7B Layer 0 K showed rotation regresses MSE by 3.68x.
# The rotated quantizer is retained in the package as a documented
# negative result, but all INT3 dispatch paths use `qa_nsep_pchan_asym`
# regardless of `layer_idx`.
# ---------------------------------------------------------------------------

def test_select_quantizer_int3_always_returns_asym_in_v03():
    """In v0.3, select_quantizer at target_bits=3 always returns the
    asymmetric variant (regardless of mode or layer_idx). Rotation was
    removed from the dispatcher after Phase 10."""
    from nsn.mode_aware import select_quantizer
    from nsn.quantize import qa_nsep_pchan_asym

    torch.manual_seed(0)
    x = torch.randn(16, 128, dtype=torch.float32)
    y_expected = qa_nsep_pchan_asym(x, bits=3)

    for (mode, layer) in [
        ("Mode 1", 0),   # previously would have dispatched to rotation
        ("Mode 1", 4),
        ("Mode 2", 0),
        ("Mode 2", 5),
        ("borderline", 0),
        ("unknown", None),
    ]:
        qfn = select_quantizer(mode=mode, target_bits=3, layer_idx=layer)
        y = qfn(x)
        assert torch.allclose(y, y_expected, atol=1e-5), (
            f"mode={mode} layer={layer} did not dispatch to asym INT3"
        )


def test_describe_selection_int3_mentions_asym():
    """describe_selection at target_bits=3 should identify the v0.3
    asymmetric default in the method string."""
    from nsn.mode_aware import describe_selection
    for mode in ["Mode 1", "Mode 2", "borderline", "unknown"]:
        desc = describe_selection(mode=mode, target_bits=3)
        method_lower = desc["method"].lower()
        assert "asymmetric" in method_lower or "asym" in method_lower, (
            f"mode={mode}: expected 'asymmetric' in method, got: {desc['method']}"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_qa_rotated_cuda():
    """Rotated quantizer works on GPU."""
    torch.manual_seed(0)
    x = torch.randn(16, 128, dtype=torch.float16, device="cuda")
    y = qa_nsep_pchan_rotated(x, bits=3)
    assert y.device == x.device
    assert y.dtype == x.dtype
    assert y.shape == x.shape
