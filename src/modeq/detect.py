# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

"""
Mode detection: measure K/V outlier profile and classify
a model as Mode 1 (Layer-0 outlier-driven) or Mode 2
(distributed norm-variation-driven).

Empirical thresholds from the 7-model study:

    Mode 1 signature: K_max at Layer 0 >= 15x, AND naive INT4
                      ΔPPL on WikiText-2 > 1000 (Qwen2-7B only
                      in our sample).
    Mode 2 signature: either K_max < 10x or K_max concentrated
                      at late layers (>= half-depth).

The `classify_mode` function below uses only the Layer 0 K
heuristic since PPL is expensive to measure; measure_kv_outlier_ratio
can be run on a small probe text in under a minute.
"""

import numpy as np
import torch


def _outlier_stats(x):
    """
    Per-channel max / mean-of-max ratio for one head.

    Parameters
    ----------
    x : torch.Tensor, shape [T, D]

    Returns
    -------
    (ratio, max_absmax, mean_absmax) : tuple of float
    """
    xa = x.float().abs()
    col_absmax = xa.amax(dim=0)
    m = col_absmax.mean().item()
    mx = col_absmax.amax().item()
    if m < 1e-12:
        return float("nan"), mx, m
    return mx / m, mx, m


def measure_kv_outlier_ratio(model, tokenizer, probe_text=None, device="cuda"):
    """
    Run a short forward pass and measure per-layer K/V outlier
    ratios across all heads.

    Parameters
    ----------
    model : PreTrainedModel
        HuggingFace model with `use_cache=True` support.
    tokenizer : PreTrainedTokenizer
    probe_text : str, optional
        Text to use as probe. Defaults to a ~200-token English
        passage that touched both narrative and technical registers.
    device : str, default "cuda"

    Returns
    -------
    dict
        {
            "n_cache_layers": int,
            "per_layer": list of dicts with k_max, k_avg, v_max, v_avg
                         for each layer index,
            "k_max_overall": float,
            "v_max_overall": float,
            "worst_k_layer": int,
            "worst_v_layer": int,
            "layer_0_k_max": float,
        }
    """
    if probe_text is None:
        probe_text = (
            "The scientific method represents a fundamental approach to "
            "understanding the natural world through systematic observation. "
            "Transformer architectures have revolutionized natural language "
            "processing by enabling attention over arbitrary positions. "
            "Quantization of neural activations to low bit widths introduces "
            "approximation error. When outlier channels dominate the "
            "per-row scale, normal channels lose precision. "
            "The Mona Lisa hangs in the Louvre. Photosynthesis converts "
            "carbon dioxide and water into glucose using light energy "
            "captured by chlorophyll."
        )

    ids = tokenizer.encode(probe_text, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model(input_ids=ids, use_cache=True)
        past = out.past_key_values

    if hasattr(past, "layers"):
        n_layers = len(past.layers)
        get_k = lambda li: past.layers[li].keys
        get_v = lambda li: past.layers[li].values
    else:
        n_layers = len(past)
        get_k = lambda li: past[li][0]
        get_v = lambda li: past[li][1]

    per_layer = []
    for li in range(n_layers):
        k = get_k(li)[0]  # [H, T, D]
        v = get_v(li)[0]
        k_ratios = [_outlier_stats(k[h])[0] for h in range(k.shape[0])]
        v_ratios = [_outlier_stats(v[h])[0] for h in range(v.shape[0])]
        per_layer.append(
            {
                "layer": li,
                "k_avg": float(np.nanmean(k_ratios)),
                "k_max": float(np.nanmax(k_ratios)),
                "v_avg": float(np.nanmean(v_ratios)),
                "v_max": float(np.nanmax(v_ratios)),
            }
        )

    all_k = [r["k_max"] for r in per_layer]
    all_v = [r["v_max"] for r in per_layer]
    return {
        "n_cache_layers": n_layers,
        "per_layer": per_layer,
        "k_max_overall": float(np.max(all_k)),
        "v_max_overall": float(np.max(all_v)),
        "worst_k_layer": int(np.argmax(all_k)),
        "worst_v_layer": int(np.argmax(all_v)),
        "layer_0_k_max": float(all_k[0]),
    }


def classify_mode(outlier_stats):
    """
    Classify a model as Mode 1 or Mode 2 based on measured K/V
    outlier statistics.

    Heuristic:
        Mode 1: Layer-0 K_max >= 15.0
                (observed only in Qwen2-7B in our sample; this is
                the catastrophic 'INT4 valley of death' pattern.)
        Mode 2: otherwise
                (the common case; naive INT at sufficient bits works,
                and nsep+pchan improves it smoothly.)

    Parameters
    ----------
    outlier_stats : dict
        Output of `measure_kv_outlier_ratio`.

    Returns
    -------
    dict
        {
            "mode": "Mode 1" | "Mode 2",
            "confidence": str,
            "reasoning": str,
            "recommendation": str,
        }
    """
    l0_k = outlier_stats["layer_0_k_max"]
    k_max = outlier_stats["k_max_overall"]
    worst_k_layer = outlier_stats["worst_k_layer"]
    n_layers = outlier_stats["n_cache_layers"]

    if l0_k >= 15.0:
        return {
            "mode": "Mode 1",
            "confidence": "high",
            "reasoning": (
                f"Layer 0 K_max = {l0_k:.2f}x (>= 15x threshold). This model "
                "matches the Qwen2-7B signature: outlier-dominated Layer 0 K. "
                "Naive INT4 will likely show catastrophic failure (INT4 valley "
                "of death). INT3 naive is typically much better than INT4."
            ),
            "recommendation": (
                "Use qa_nsep_pchan_asym(x, 3) (or Int3AsymNsepPchanCache) "
                "for 5.09x compression; Phase 14 measured +0.10 ΔPPL on "
                "Qwen2-7B at 3.0 bits/value. For 4-bit budget, Mode 1 "
                "still benefits from qa_nsep_pchan_resid_rms(x, 3) "
                "(+0.025 ΔPPL, beats plain INT4 due to the INT4 valley). "
                "Layer-0-selective compression with the v0.2 symmetric "
                "pipeline also rescues this signature (99.97% on Qwen2-7B)."
            ),
        }
    elif l0_k >= 10.0:
        return {
            "mode": "borderline",
            "confidence": "medium",
            "reasoning": (
                f"Layer 0 K_max = {l0_k:.2f}x (10-15x window). No model in "
                "the 7-model study fell in this range; behavior is unknown."
            ),
            "recommendation": (
                "Use qa_nsep_pchan_asym(x, 3) (Int3AsymNsepPchanCache) as "
                "the safe default -- it hedges cleanly across Mode 1 and "
                "Mode 2 at 3.0 bits/value with a 66% average ΔPPL "
                "reduction vs symmetric. Consider running WikiText-2 PPL "
                "with naive INT4 if you need a definitive mode call."
            ),
        }
    else:
        return {
            "mode": "Mode 2",
            "confidence": "high" if k_max < 11 else "medium",
            "reasoning": (
                f"Layer 0 K_max = {l0_k:.2f}x (< 10x). Overall K_max = "
                f"{k_max:.2f}x at layer {worst_k_layer}/{n_layers}. "
                "Matches Mode 2 signature: no Layer-0 outlier. Naive INT "
                "quantization scales monotonically with bit width."
            ),
            "recommendation": (
                "Use qa_nsep_pchan_asym(x, 3) (Int3AsymNsepPchanCache) "
                "for 5.09x compression. Phase 14-17 validated on 7 Mode-2 "
                "models (Llama-3.1-8B, Pythia-6.9B, Qwen2.5-14B, "
                "Mistral-7B, GPT-J-6B, Qwen3.5-9B, Gemma 4-E4B); all 7 "
                "landed in the < 5% baseline production tier under asym. "
                "Plain symmetric INT4 (qa_nsep_pchan(x, 4)) is also fine "
                "on Mode 2 with ΔPPL < 0.1 typical -- use when you have "
                "the 4-bit budget to spare."
            ),
        }
