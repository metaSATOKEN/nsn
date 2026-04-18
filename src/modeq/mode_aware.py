# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

"""
Mode-aware quantizer dispatch.

Given a model's measured K outlier profile (from
:func:`modeq.detect.measure_kv_outlier_ratio`), select the best
`nsep+pchan` variant for the requested bit budget.

The empirical basis for the dispatch rules (7-model validation,
April 2026):

* **Mode 1** (Layer-0 K outlier >= 15x; observed only in Qwen2-7B):
  * 3-bit budget -> ``qa_nsep_pchan(x, 3)``  (escapes INT4 valley)
  * 4-bit budget -> ``qa_nsep_pchan_resid_rms(x, 3)``
    * DeltaPPL = +0.025 on Qwen2-7B, vs INT4 alone +0.054.
    * This is Mode-1-specific: the INT4 "valley of death"
      makes INT3+RMS beat native INT4.
  * 5+ bit budget -> ``qa_nsep_pchan(x, bits)``  (INT5+ is clean)

* **Mode 2** (Layer-0 K < 10x; 6/7 models in our sample):
  * 3-bit budget -> ``qa_nsep_pchan(x, 3)``
  * 4-bit budget -> ``qa_nsep_pchan(x, 4)`` (clean, beats INT3+RMS)
  * 5+ bit budget -> ``qa_nsep_pchan(x, bits)``

* **Borderline** (10x <= Layer-0 K < 15x; no model in our sample):
  * Conservative default: treat as Mode 1 (use residual).

* **Unknown** (profile not measured): pragmatic default is
  ``qa_nsep_pchan`` without residual at the requested bit budget.
  nsep+pchan INT4 did not cause catastrophic degradation on any
  of the 27 models evaluated in the companion paper.
"""

from typing import Callable, Optional

import torch

from .quantize import (
    qa_nsep_pchan,
    qa_nsep_pchan_resid_rms,
)
from .detect import classify_mode


def select_quantizer(
    mode: Optional[str] = None,
    outlier_stats: Optional[dict] = None,
    target_bits: int = 4,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Return the recommended quantizer for the given mode and bit budget.

    Either ``mode`` (explicit) or ``outlier_stats`` (from
    :func:`modeq.detect.measure_kv_outlier_ratio`, auto-classified) must
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
    if mode is None and outlier_stats is None:
        raise ValueError(
            "Either 'mode' or 'outlier_stats' must be provided. "
            "Pass outlier_stats=None to fall through to 'unknown'."
        )
    if mode is None:
        classification = classify_mode(outlier_stats)
        mode = classification["mode"]

    mode_normalized = mode.lower().replace(" ", "").replace("-", "")

    if mode_normalized == "mode1":
        if target_bits <= 3:
            return lambda x: qa_nsep_pchan(x, 3)
        elif target_bits == 4:
            # INT3 + RMS residual (~4+D/head bits) beats INT4 alone
            # on Mode 1 by 2.2x precision.
            return lambda x: qa_nsep_pchan_resid_rms(x, 3)
        else:
            return lambda x: qa_nsep_pchan(x, target_bits)

    if mode_normalized == "mode2":
        if target_bits <= 2:
            # INT2 alone is too lossy on Mode 2 (e.g., Llama-3.1-8B
            # DeltaPPL +22.8 per Phase 7). RMS residual on an INT2
            # base is less lossy (+0.71 on Llama) at roughly 3 eff.
            # bits. Caller asked for 2 bits; we give a related
            # scheme at 3 eff. bits rather than return an
            # empirically broken quantizer.
            return lambda x: qa_nsep_pchan_resid_rms(x, 2)
        return lambda x: qa_nsep_pchan(x, target_bits)

    if mode_normalized == "borderline":
        # Conservative default: assume Mode 1 behavior.
        if target_bits == 4:
            return lambda x: qa_nsep_pchan_resid_rms(x, 3)
        if target_bits <= 2:
            return lambda x: qa_nsep_pchan_resid_rms(x, 2)
        return lambda x: qa_nsep_pchan(x, target_bits)

    # Fallback for "unknown" or unrecognized strings
    if target_bits <= 2:
        return lambda x: qa_nsep_pchan_resid_rms(x, 2)
    return lambda x: qa_nsep_pchan(x, target_bits)


def describe_selection(
    mode: Optional[str] = None,
    outlier_stats: Optional[dict] = None,
    target_bits: int = 4,
) -> dict:
    """
    Human-readable description of the mode-aware selection decision.

    Returns a dict containing the chosen method name, its rationale
    ("why this and not something else"), and estimated quality for
    the given mode.
    """
    if mode is None and outlier_stats is not None:
        classification = classify_mode(outlier_stats)
        mode = classification["mode"]
        mode_reasoning = classification["reasoning"]
    else:
        mode_reasoning = f"Mode explicitly specified as '{mode}'."

    mode_n = (mode or "unknown").lower().replace(" ", "").replace("-", "")

    if mode_n == "mode1":
        if target_bits <= 3:
            method = "nsep+pchan INT3"
            rationale = (
                "Mode 1, 3-bit budget. INT3 escapes the INT4 valley of death. "
                "Residual correction at 3 bits would cost extra bits."
            )
            est_quality = "~ +0.40 DeltaPPL on Qwen2-7B (Phase 1 reference)"
        elif target_bits == 4:
            method = "nsep+pchan INT3 + per-channel RMS residual"
            rationale = (
                "Mode 1, 4-bit budget. INT3 + RMS residual avoids the INT4 "
                "valley entirely and surpasses native INT4 (+0.025 vs +0.054 "
                "DeltaPPL on Qwen2-7B, 2.2x precision)."
            )
            est_quality = "~ +0.025 DeltaPPL on Qwen2-7B"
        else:
            method = f"nsep+pchan INT{target_bits}"
            rationale = (
                f"Mode 1, >=5-bit budget. INT5+ is clean even on Mode 1 "
                f"(sufficient precision for outlier channels)."
            )
            est_quality = "~ <0.01 DeltaPPL"
    elif mode_n == "mode2":
        if target_bits <= 2:
            method = f"nsep+pchan INT{target_bits} + per-channel RMS residual"
            rationale = (
                f"Mode 2, {target_bits}-bit budget. INT{target_bits} alone is "
                f"empirically too lossy on Mode 2 (e.g., INT2 alone gives "
                f"DeltaPPL +22.8 on Llama-3.1-8B). We upgrade to RMS "
                f"residual, which costs roughly 1 extra bit/value + small "
                f"metadata but reduces the damage to ~+0.7 DeltaPPL on the "
                f"same model."
            )
            est_quality = "~ +0.5 to +2 DeltaPPL (uncertain at very low bits)"
        else:
            method = f"nsep+pchan INT{target_bits}"
            rationale = (
                f"Mode 2, {target_bits}-bit budget. Mode 2 scales monotonically "
                f"with bit width; INT4 alone beats INT3+residual by a small but "
                f"consistent margin (no valley to escape). Simplest choice is best."
            )
            est_quality = {
                3: "~ +0.1 to +0.3 DeltaPPL typical",
                4: "~ +0.02 to +0.08 DeltaPPL typical",
            }.get(target_bits, "very small")
    elif mode_n == "borderline":
        method = ("nsep+pchan INT3 + RMS residual" if target_bits == 4
                  else f"nsep+pchan INT{target_bits}")
        rationale = (
            "Borderline (10-15x Layer 0 K). No model in our 7-model sample "
            "fell in this range. Conservative default: treat as Mode 1 to "
            "hedge against INT4 valley. If Mode-2-like, INT3+RMS loses at "
            "most 0.05 DeltaPPL to INT4 alone."
        )
        est_quality = "uncertain; hedged via Mode 1 defaults"
    else:
        method = f"nsep+pchan INT{target_bits}"
        rationale = (
            f"Unknown / unmeasured profile. Default to the safe choice: "
            f"nsep+pchan INT{target_bits} without residual. In the 27-model "
            f"sample across all experiments, this choice never produced "
            f"catastrophic DeltaPPL (worst case +0.24)."
        )
        est_quality = "varies; generally safe"

    return {
        "method": method,
        "mode": mode or "unknown",
        "target_bits": target_bits,
        "mode_reasoning": mode_reasoning,
        "method_rationale": rationale,
        "estimated_quality": est_quality,
    }
