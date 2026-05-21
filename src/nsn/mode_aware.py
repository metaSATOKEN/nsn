# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

"""
Mode-aware quantizer dispatch (v0.3).

Given a model's measured K outlier profile (from
:func:`nsn.detect.measure_kv_outlier_ratio`), return the best
`nsep+pchan` variant for the requested bit budget.

v0.3 empirical basis (Phases 10-15, April 2026):

* **INT3 tier (3 bits/value, 5.09x vs FP16)**:
  Per-channel ASYMMETRIC quantization is the universal default.
  Phase 14 WikiText-2 PPL cuts symmetric INT3 ΔPPL by 74.3% on
  Qwen2-7B and 58.3% on Llama-3.1-8B at identical bit rate. The
  gain comes from fitting the full [min, max] per-channel range
  instead of symmetric [-absmax, +absmax], which wastes roughly
  half the quantization levels on strongly biased post-RoPE K
  distributions (empirically observed across both Mode 1 and
  Mode 2 models).

* **INT4 tier (4 bits/value)**: Plain symmetric nsep+pchan
  remains the default. Phase 14 shows asymmetric INT4 is actually
  *worse* than symmetric on Llama-3.1-8B (+0.027 vs +0.013 ΔPPL)
  while tying on Qwen2-7B. Above INT3, the zero-centering of
  symmetric quantization appears to carry value that asym's extra
  level cannot make up for.

* **4-bit Mode 1 special case**: INT3 + per-channel RMS residual
  (v0.2 mechanism) remains the best 4-bit-budget choice on Mode 1
  (Qwen2-7B: +0.025 ΔPPL, beats plain INT4 +0.054). This is
  retained as the 4-bit Mode 1 dispatch.

* **<=2 bits**: RMS residual correction on INT2 base. Pure INT2
  is too lossy on both modes.

Rotation (Walsh-Hadamard) was evaluated in v0.3 dev and is NOT
dispatched. Phase 10 probe on real Qwen2-7B Layer 0 K showed
rotation REGRESSES MSE by 3.68x because the distribution is a
discrete multi-channel plateau rather than heavy-tailed, which
per-channel absmax already handles optimally. The rotation
quantizer remains in the package (:func:`qa_nsep_pchan_rotated`)
as a documented negative result.
"""

from typing import Callable, Optional

import torch

from .quantize import (
    qa_nsep_pchan,
    qa_nsep_pchan_resid_rms,
    qa_nsep_pchan_asym,
)
from .detect import classify_mode


def select_quantizer(
    mode: Optional[str] = None,
    outlier_stats: Optional[dict] = None,
    target_bits: int = 4,
    layer_idx: Optional[int] = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Return the recommended quantizer for the given mode and bit budget.

    Either ``mode`` (explicit) or ``outlier_stats`` (from
    :func:`nsn.detect.measure_kv_outlier_ratio`, auto-classified) must
    be provided.

    Parameters
    ----------
    mode : str, optional
        One of {"Mode 1", "Mode 2", "borderline", "unknown"}.
        If not given, inferred from ``outlier_stats`` via
        :func:`classify_mode`.
    outlier_stats : dict, optional
        Output of :func:`measure_kv_outlier_ratio`. Used to derive
        ``mode`` when ``mode`` is not given.
    target_bits : int, default 4
        Target bit width per value (3 or 4 are the supported sweet
        spots; 2 is very lossy and 5+ diminishing returns).
    layer_idx : int, optional
        Reserved for future layer-conditional dispatch. In v0.3 the
        dispatcher does not consult this parameter (Walsh-Hadamard
        rotation was removed after Phase 10 showed a 3.68x MSE
        regression on real Qwen2-7B Layer 0 K). Retained in the
        signature for forward API compatibility.

    Returns
    -------
    callable
        A function ``f(x) -> x_quantized`` applying the selected
        quantizer. Can be passed to a cache compression loop.

    Raises
    ------
    ValueError
        If neither ``mode`` nor ``outlier_stats`` is provided.
    """
    del layer_idx  # reserved in the API but unused in v0.3 dispatch

    if mode is None and outlier_stats is None:
        raise ValueError(
            "Either 'mode' or 'outlier_stats' must be provided. "
            "Pass outlier_stats=None to fall through to 'unknown'."
        )
    if mode is None:
        classification = classify_mode(outlier_stats)
        mode = classification["mode"]

    mode_normalized = mode.lower().replace(" ", "").replace("-", "")

    # ---- INT3 tier: asymmetric is universal v0.3 default ----
    if target_bits == 3:
        return lambda x: qa_nsep_pchan_asym(x, 3)

    # ---- <=2 bits: RMS residual on INT2 base ----
    if target_bits <= 2:
        return lambda x: qa_nsep_pchan_resid_rms(x, 2)

    # ---- 4-bit tier ----
    if target_bits == 4:
        if mode_normalized == "mode1":
            # On Mode 1, INT3 + RMS residual (~4+D/head bits) beats
            # plain INT4 alone (+0.025 vs +0.054 ΔPPL on Qwen2-7B).
            return lambda x: qa_nsep_pchan_resid_rms(x, 3)
        if mode_normalized == "borderline":
            # Hedge toward Mode 1 to avoid the INT4 valley if mode
            # is uncertain.
            return lambda x: qa_nsep_pchan_resid_rms(x, 3)
        # Mode 2 and unknown: plain sym INT4 is best at 4-bit budget.
        # Asymmetric INT4 is empirically worse on Llama (+0.027 vs
        # +0.013); do not use it here.
        return lambda x: qa_nsep_pchan(x, 4)

    # ---- >=5 bits: plain sym is clean everywhere ----
    return lambda x: qa_nsep_pchan(x, target_bits)


def describe_selection(
    mode: Optional[str] = None,
    outlier_stats: Optional[dict] = None,
    target_bits: int = 4,
    layer_idx: Optional[int] = None,
) -> dict:
    """
    Human-readable description of the mode-aware selection decision.

    Returns a dict containing the chosen method name, its rationale
    ("why this and not something else"), and estimated quality for
    the given mode.
    """
    del layer_idx  # reserved; unused in v0.3

    if mode is None and outlier_stats is not None:
        classification = classify_mode(outlier_stats)
        mode = classification["mode"]
        mode_reasoning = classification["reasoning"]
    else:
        mode_reasoning = f"Mode explicitly specified as '{mode}'."

    mode_n = (mode or "unknown").lower().replace(" ", "").replace("-", "")

    # ----- INT3 -----
    if target_bits == 3:
        method = "nsep+pchan INT3 asymmetric (v0.3 default)"
        rationale = (
            "INT3 tier: asymmetric per-channel (min + range) is the "
            "v0.3 universal default. At identical 3.0 bits/value this "
            "reduces WikiText-2 ΔPPL by 74.3% on Qwen2-7B (Mode 1) and "
            "58.3% on Llama-3.1-8B (Mode 2) vs the symmetric baseline. "
            "Mechanism: post-RoPE K channels are strongly biased; "
            "symmetric absmax wastes ~half the quantization range."
        )
        if mode_n == "mode1":
            est_quality = "~ +0.10 ΔPPL on Qwen2-7B (Phase 14 measurement)"
        elif mode_n == "mode2":
            est_quality = "~ +0.11 ΔPPL on Llama-3.1-8B (Phase 14 measurement)"
        else:
            est_quality = "~ +0.10 to +0.15 ΔPPL typical at INT3 asym"

    # ----- <=2 bits -----
    elif target_bits <= 2:
        method = f"nsep+pchan INT{target_bits} + per-channel RMS residual"
        rationale = (
            f"<=2-bit budget. Pure INT{target_bits} alone is empirically "
            f"too lossy on both Mode 1 and Mode 2 (e.g., INT2 alone gives "
            f"ΔPPL +22.8 on Llama-3.1-8B). RMS residual correction costs "
            f"roughly 1 extra bit/value + small metadata but reduces the "
            f"damage to ~+0.7 ΔPPL."
        )
        est_quality = "~ +0.5 to +2 ΔPPL (uncertain at very low bits)"

    # ----- 4-bit -----
    elif target_bits == 4:
        if mode_n == "mode1":
            method = "nsep+pchan INT3 + per-channel RMS residual"
            rationale = (
                "Mode 1, 4-bit budget. INT3 + RMS residual avoids the INT4 "
                "valley entirely and surpasses native INT4 (+0.025 vs +0.054 "
                "ΔPPL on Qwen2-7B)."
            )
            est_quality = "~ +0.025 ΔPPL on Qwen2-7B"
        elif mode_n == "borderline":
            method = "nsep+pchan INT3 + per-channel RMS residual"
            rationale = (
                "Borderline (10-15x Layer 0 K). Conservative default: "
                "treat as Mode 1 to hedge against the INT4 valley."
            )
            est_quality = "uncertain; hedged via Mode 1 defaults"
        else:
            method = "nsep+pchan INT4 (symmetric)"
            rationale = (
                "Mode 2 / unknown, 4-bit budget. Plain symmetric INT4 is "
                "the best choice at 4-bit budget on Mode 2. Note that "
                "asymmetric INT4 is empirically *worse* than symmetric at "
                "this bit width on Llama-3.1-8B (+0.027 vs +0.013 ΔPPL)."
            )
            est_quality = {
                "mode2":   "~ +0.013 ΔPPL on Llama-3.1-8B (Phase 14)",
                "unknown": "~ +0.02 to +0.08 ΔPPL typical",
            }.get(mode_n, "~ +0.02 to +0.08 ΔPPL typical")

    # ----- 5+ bits -----
    else:
        method = f"nsep+pchan INT{target_bits} (symmetric)"
        rationale = (
            f">=5-bit budget. Symmetric INT{target_bits} is clean on both "
            f"Mode 1 and Mode 2; residual correction and asymmetric variants "
            f"offer no measurable improvement above 4 bits."
        )
        est_quality = "very small ΔPPL"

    return {
        "method": method,
        "mode": mode or "unknown",
        "target_bits": target_bits,
        "mode_reasoning": mode_reasoning,
        "method_rationale": rationale,
        "estimated_quality": est_quality,
    }
