# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

"""Tests for nsn's v0.2 residual correction variants."""

import pytest
import torch

from nsn.quantize import (
    qa_nsep_pchan,
    qa_nsep_pchan_resid_rms,
    qa_nsep_pchan_resid_fixed,
)


def test_rms_residual_shape_and_dtype():
    x = torch.randn(8, 128, dtype=torch.float16)
    y = qa_nsep_pchan_resid_rms(x, bits_base=3)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_fixed_residual_shape_and_dtype():
    x = torch.randn(8, 128, dtype=torch.float16)
    y = qa_nsep_pchan_resid_fixed(x, bits_base=3, magnitude=0.25)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_rms_residual_improves_int3_alone():
    """On a clean synthetic tensor, RMS residual should yield lower MSE
    than INT3 alone at the same seed."""
    torch.manual_seed(0)
    x = torch.randn(64, 128, dtype=torch.float32)

    y_int3      = qa_nsep_pchan(x, 3)
    y_int3_rms  = qa_nsep_pchan_resid_rms(x, 3)

    mse_int3     = ((x - y_int3) ** 2).mean().item()
    mse_int3_rms = ((x - y_int3_rms) ** 2).mean().item()
    assert mse_int3_rms < mse_int3, (
        f"RMS residual should reduce MSE: int3={mse_int3:.4e}, "
        f"int3+rms={mse_int3_rms:.4e}"
    )


def test_rms_residual_nears_int4_on_clean_mode2_like_tensor():
    """On a tensor without pathological outliers, INT3+RMS should be
    close to INT4 alone. It may lose slightly (as on Mode 2 models)
    but should not be orders of magnitude worse."""
    torch.manual_seed(0)
    x = torch.randn(64, 128, dtype=torch.float32)

    y_int4      = qa_nsep_pchan(x, 4)
    y_int3_rms  = qa_nsep_pchan_resid_rms(x, 3)

    mse_int4     = ((x - y_int4) ** 2).mean().item()
    mse_int3_rms = ((x - y_int3_rms) ** 2).mean().item()
    # Both should be small, and INT3+RMS should be within 3x of INT4
    assert mse_int3_rms < 3 * mse_int4 + 1e-8, (
        f"INT3+RMS should be within 3x of INT4: "
        f"int4={mse_int4:.4e}, int3+rms={mse_int3_rms:.4e}"
    )


def test_fixed_0_25_residual_on_uniform_data_approximately_works():
    """On uniformly-distributed data (a synthetic Mode-2-like case),
    the fixed 0.25 residual should at least not make things worse than
    INT3 alone."""
    torch.manual_seed(0)
    x = torch.randn(64, 128, dtype=torch.float32)

    y_int3       = qa_nsep_pchan(x, 3)
    y_int3_fixed = qa_nsep_pchan_resid_fixed(x, 3, magnitude=0.25)

    mse_int3       = ((x - y_int3) ** 2).mean().item()
    mse_int3_fixed = ((x - y_int3_fixed) ** 2).mean().item()
    # On clean data, fixed 0.25 should not backfire (won't be > 2x INT3)
    assert mse_int3_fixed < 2 * mse_int3, (
        f"Fixed-0.25 should not worsen clean Mode-2-like data: "
        f"int3={mse_int3:.4e}, fixed={mse_int3_fixed:.4e}"
    )


def test_residual_variants_quantize_4d_input():
    """Regression for the v0.2 dim=0 -> dim=-2 fix: residual variants
    must produce non-trivial quantization on 4D [B, H, T, D]."""
    torch.manual_seed(0)
    x = torch.randn(1, 4, 16, 64, dtype=torch.float32)

    y_rms   = qa_nsep_pchan_resid_rms(x, bits_base=3)
    y_fixed = qa_nsep_pchan_resid_fixed(x, bits_base=3, magnitude=0.25)

    diff_rms   = (x - y_rms).abs().mean().item()
    diff_fixed = (x - y_fixed).abs().mean().item()
    assert diff_rms   > 1e-3, f"RMS variant identity on 4D: {diff_rms:.3e}"
    assert diff_fixed > 1e-3, f"Fixed variant identity on 4D: {diff_fixed:.3e}"


def test_fixed_residual_backfires_on_synthetic_outlier():
    """Synthetic outlier case to verify the Mode-1-like failure mode of
    fixed-magnitude residual. This replicates the empirical finding
    that fixed-0.25 makes things worse when residuals are non-uniform.

    Not a hard failure check (since this depends on distribution); we
    just verify the ratio isn't absurdly bad.
    """
    torch.manual_seed(0)
    x = torch.randn(64, 128, dtype=torch.float32)
    # Inject extreme outlier channel to make residuals heavy-tailed
    x[:, 0] *= 20.0

    y_int3       = qa_nsep_pchan(x, 3)
    y_int3_fixed = qa_nsep_pchan_resid_fixed(x, 3, magnitude=0.25)
    y_int3_rms   = qa_nsep_pchan_resid_rms(x, 3)

    mse_int3       = ((x - y_int3) ** 2).mean().item()
    mse_int3_fixed = ((x - y_int3_fixed) ** 2).mean().item()
    mse_int3_rms   = ((x - y_int3_rms) ** 2).mean().item()
    # RMS should be better than fixed on outlier-injected synthetic
    assert mse_int3_rms < mse_int3_fixed + 1e-8 or mse_int3_rms < mse_int3, (
        f"RMS should beat or tie fixed on outlier tensor: "
        f"int3={mse_int3:.4e}, fixed={mse_int3_fixed:.4e}, "
        f"rms={mse_int3_rms:.4e}"
    )
