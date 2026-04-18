# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

"""Tests for modeq.mode_aware dispatch logic."""

import pytest
import torch

from modeq.mode_aware import select_quantizer, describe_selection


def _fake_outlier_stats(layer_0_k, k_max=None, n_layers=32):
    """Construct a minimal outlier_stats dict with the given L0 K."""
    km = k_max if k_max is not None else max(layer_0_k, 5.0)
    return {
        "n_cache_layers": n_layers,
        "per_layer": [{"layer": i, "k_max": layer_0_k if i == 0 else 4.0,
                       "k_avg": layer_0_k / 2, "v_max": 5.0, "v_avg": 3.0}
                      for i in range(n_layers)],
        "k_max_overall": km,
        "v_max_overall": 8.0,
        "worst_k_layer": 0,
        "worst_v_layer": 0,
        "layer_0_k_max": layer_0_k,
    }


def test_select_mode1_4bit_uses_residual():
    """Mode 1 + 4 bits -> should use INT3+RMS residual quantizer."""
    stats = _fake_outlier_stats(layer_0_k=17.0)
    qfn = select_quantizer(outlier_stats=stats, target_bits=4)

    x = torch.randn(8, 128, dtype=torch.float16)
    y = qfn(x)
    assert y.shape == x.shape

    desc = describe_selection(outlier_stats=stats, target_bits=4)
    assert "residual" in desc["method"].lower()
    assert desc["mode"] == "Mode 1"


def test_select_mode2_4bit_uses_plain_int4():
    """Mode 2 + 4 bits -> should use plain INT4 (no residual)."""
    stats = _fake_outlier_stats(layer_0_k=5.0)
    desc = describe_selection(outlier_stats=stats, target_bits=4)
    assert "residual" not in desc["method"].lower()
    assert "INT4" in desc["method"]
    assert desc["mode"] == "Mode 2"


def test_select_mode1_3bit_uses_plain_int3():
    """At 3-bit budget, Mode 1 still gets plain INT3 (residual would
    cost extra bits; the value is at 4-bit budget)."""
    stats = _fake_outlier_stats(layer_0_k=17.0)
    desc = describe_selection(outlier_stats=stats, target_bits=3)
    assert "residual" not in desc["method"].lower()
    assert "INT3" in desc["method"]


def test_explicit_mode_overrides_stats():
    """Explicit mode parameter takes precedence over stats."""
    stats = _fake_outlier_stats(layer_0_k=5.0)  # would classify as Mode 2
    desc = describe_selection(mode="Mode 1", target_bits=4,
                               outlier_stats=stats)
    assert desc["mode"] == "Mode 1"
    assert "residual" in desc["method"].lower()


def test_unknown_mode_uses_safe_default():
    """Unknown / unspecified mode should pick the safe nsep+pchan INT4."""
    desc = describe_selection(mode="unknown", target_bits=4)
    assert "residual" not in desc["method"].lower()
    assert "INT4" in desc["method"]


def test_no_info_raises_on_quantizer():
    """Calling select_quantizer with neither mode nor stats should raise."""
    with pytest.raises(ValueError):
        select_quantizer(target_bits=4)


def test_quantizer_callable_produces_right_shape():
    """Returned callable should produce a same-shape tensor when applied."""
    stats = _fake_outlier_stats(layer_0_k=17.0)
    qfn = select_quantizer(outlier_stats=stats, target_bits=4)
    x = torch.randn(4, 8, 128, dtype=torch.float16)
    y = qfn(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_borderline_hedges_toward_mode1():
    """Borderline at 4 bits -> use residual (conservative)."""
    desc = describe_selection(mode="borderline", target_bits=4)
    assert "residual" in desc["method"].lower()


def test_describe_matches_select_for_borderline_low_bits():
    """At <=2 bits the borderline callable is INT2+RMS; description
    must say so (regression: previously said plain INT2 without
    residual)."""
    desc = describe_selection(mode="borderline", target_bits=2)
    assert "residual" in desc["method"].lower(), (
        f"borderline + 2-bit should advertise residual, got: {desc['method']}"
    )

    qfn = select_quantizer(mode="borderline", target_bits=2)
    x = torch.randn(16, 128, dtype=torch.float32)
    y = qfn(x)
    # residual-corrected quantizer is NOT the identity
    assert (x - y).abs().mean().item() > 1e-4


def test_describe_matches_select_for_mode1_low_bits():
    """Mode 1 + <=2 bits: silently upgraded to INT3 (pure INT2 broken
    on outlier channels). Description must make the upgrade explicit
    so the caller sees the true bit cost."""
    desc = describe_selection(mode="Mode 1", target_bits=2)
    method_lower = desc["method"].lower()
    rationale_lower = desc["method_rationale"].lower()

    # Method string must say INT3 (actual cost), not INT2
    assert "int3" in method_lower, (
        f"Mode 1 + 2-bit should advertise INT3, got: {desc['method']}"
    )
    # Method or rationale must explicitly flag the upgrade so the user
    # cannot miss that they were not actually given 2-bit storage.
    assert ("upgrad" in method_lower) or ("upgrad" in rationale_lower), (
        f"Mode 1 + 2-bit description must flag the INT2->INT3 upgrade; "
        f"got method='{desc['method']}' rationale='{desc['method_rationale']}'"
    )
    # target_bits preserved as reported-input
    assert desc["target_bits"] == 2


def test_describe_matches_select_for_unknown_low_bits():
    """At <=2 bits the unknown callable is INT2+RMS; description must
    say so and must NOT claim 'never produced catastrophic DeltaPPL'
    at that budget (that claim applies to INT3+ only)."""
    desc = describe_selection(mode="unknown", target_bits=2)
    assert "residual" in desc["method"].lower(), (
        f"unknown + 2-bit should advertise residual, got: {desc['method']}"
    )
    # The "never catastrophic" claim must be scoped to INT3+ budgets,
    # not blanket-applied to the low-bit fallback.
    assert "never produced catastrophic" not in desc["method_rationale"], (
        f"Rationale at <=2 bits should not carry the INT3+ safety claim; "
        f"got: {desc['method_rationale']}"
    )

    qfn = select_quantizer(mode="unknown", target_bits=2)
    x = torch.randn(16, 128, dtype=torch.float32)
    y = qfn(x)
    assert (x - y).abs().mean().item() > 1e-4
