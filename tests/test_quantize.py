# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

"""Unit tests for modeq.quantize (core quantization primitives)."""

import pytest
import torch

from modeq.quantize import qa_perrow, qa_perchan, qa_nsep_pchan, apply_method


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
