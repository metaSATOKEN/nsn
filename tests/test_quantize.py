# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

"""Unit tests for nsn.quantize (core quantization primitives)."""

import pytest
import torch

from nsn.quantize import qa_perrow, qa_perchan, qa_nsep_pchan, apply_method


def test_qa_perrow_shape_and_dtype():
    x = torch.randn(4, 128, dtype=torch.float16)
    y = qa_perrow(x, bits=4)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_qa_nsep_pchan_preserves_norm_approximately():
    """After nsep+pchan, the reconstructed norm should match the input."""
    torch.manual_seed(0)
    x = torch.randn(16, 128, dtype=torch.float32)
    y = qa_nsep_pchan(x, bits=4)
    rel_err = ((x.norm(dim=-1) - y.norm(dim=-1)).abs() / x.norm(dim=-1)).mean()
    assert rel_err < 0.01, f"Norm-preservation error too large: {rel_err:.4f}"


def test_nsep_beats_naive_on_outliers():
    """On a tensor with outlier channels, nsep+pchan should have lower MSE."""
    torch.manual_seed(0)
    x = torch.randn(64, 128, dtype=torch.float32)
    # Inject a big outlier into channel 0
    x[:, 0] *= 20.0

    y_naive = qa_perrow(x, bits=4)
    y_nsep = qa_nsep_pchan(x, bits=4)

    mse_naive = ((x - y_naive) ** 2).mean()
    mse_nsep = ((x - y_nsep) ** 2).mean()
    assert mse_nsep < mse_naive, (
        f"nsep+pchan (MSE {mse_nsep:.4f}) should beat naive "
        f"(MSE {mse_naive:.4f}) on outlier-containing tensor"
    )


def test_apply_method_dispatch():
    x = torch.randn(4, 128, dtype=torch.float16)
    assert apply_method(x, "baseline", bits=4) is x
    assert apply_method(x, "fp16", bits=4) is x
    assert apply_method(x, "naive", bits=4).shape == x.shape
    assert apply_method(x, "nsep+pchan", bits=4).shape == x.shape
    with pytest.raises(ValueError):
        apply_method(x, "unknown", bits=4)


# -- Regression tests for the 4D identity bug found in v0.2 --
# The pre-fix code used `dim=0` where `dim=-2` was meant. On 4D
# [B, H, T, D] input with B=1, this made qa_nsep_pchan effectively
# the identity. Tests below verify non-trivial quantization for 4D
# shapes and that 2D and 4D paths produce consistent results.

def test_qa_nsep_pchan_quantizes_4d_input():
    """4D [B, H, T, D] input should produce non-trivial quantization.

    Regression for the v0.2 bug where qa_nsep_pchan returned x unchanged
    on 4D inputs because the per-channel scale reduced over dim=0
    (batch) instead of dim=-2 (tokens).
    """
    torch.manual_seed(0)
    x = torch.randn(1, 8, 32, 128, dtype=torch.float32)
    y = qa_nsep_pchan(x, bits=3)
    diff = (x - y).abs().mean().item()
    assert diff > 1e-3, (
        f"4D input should produce non-trivial quantization error; "
        f"got mean abs diff {diff:.3e}. This is the v0.2 2D-only bug."
    )


def test_qa_nsep_pchan_2d_and_4d_produce_consistent_quantization():
    """Per-head 2D quantization and full 4D quantization should agree
    up to per-element error levels (not zero, but of the same order
    of magnitude).
    """
    torch.manual_seed(0)
    x4 = torch.randn(1, 4, 16, 64, dtype=torch.float32)
    y4 = qa_nsep_pchan(x4, bits=3)

    # Slice-per-head 2D path (the experimental harness style)
    y_per_head = torch.zeros_like(x4)
    for h in range(x4.shape[1]):
        y_per_head[0, h] = qa_nsep_pchan(x4[0, h], bits=3)

    # Both approaches should quantize; they will not be bit-identical
    # because dim=-2 reduces over the same axis (token) in both
    # cases but per-head vs per-batch-head reductions differ only by
    # the presence of the batch dim, which for B=1 produces identical
    # scale tensors. So the results MUST match.
    assert torch.allclose(y4, y_per_head, atol=1e-5, rtol=1e-5), (
        "4D path and per-head 2D path should produce identical "
        "quantizations for B=1 (both reduce over T via dim=-2)."
    )


def test_qa_perchan_4d_nontrivial():
    """Direct qa_perchan test on 4D input."""
    from nsn.quantize import qa_perchan
    torch.manual_seed(0)
    x = torch.randn(1, 4, 32, 96, dtype=torch.float32)
    y = qa_perchan(x, bits=4)
    diff = (x - y).abs().mean().item()
    assert diff > 1e-4, (
        f"4D qa_perchan should actually quantize; got mean abs diff "
        f"{diff:.3e}"
    )
