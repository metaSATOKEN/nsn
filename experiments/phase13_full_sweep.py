# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Phase 13: Full sweep on real KV (v0.3 pivot search)
# ============================================================
# Phase 12 found that:
#   * Llama-3.1-8B (Mode 2) -- 2-tier HELPS, 8/8 heads, -12.8% MSE
#   * Qwen2-7B  (Mode 1) -- 2-tier HURTS, 1/4 heads, +17.6% MSE
#
# Three hypotheses for the Qwen2-7B regression:
#   H1: plateau extends past K=4; K=8/16 might help.
#   H2: re-normalization interacts badly with non-uniform
#       quantization error structure.
#   H3: top-K-by-max selection is noisy with only 177 tokens;
#       mean-abs (channel energy) is more stable.
#
# We test all three in one Colab run:
#   (A) K sweep:     {2, 4, 8, 16, 32} with max-based top-K
#   (B) selection:   max-based  vs mean-abs-based top-K at K=8
#   (C) asymmetric:  per-channel min+range (non-symmetric) quantization,
#                    with and without 2-tier composition
#
# Output: ranked table of (method, model) -> mse ratio vs plain INT3.
# Best-performing Qwen2-7B method AND best non-regressing Llama
# method -> candidate for WikiText-2 PPL evaluation.
#
# Runtime: ~10-15 min on Colab A100.
# ============================================================


# =============================================================
# === CELL V1 === Setup
# =============================================================

import gc
import torch
try:
    del mdl, tok
except NameError:
    pass
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

!pip install -q -U transformers accelerate hf_transfer
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# ─────────────────────────────────────────────────────────────
# >>>>>  PASTE YOUR HF TOKEN  <<<<<
HF_TOKEN = "hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
# ─────────────────────────────────────────────────────────────
from huggingface_hub import login
login(token=HF_TOKEN)

import json
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer


# =============================================================
# === CELL V2 === Quantizer zoo (all inline)
# =============================================================

def qa_nsep_pchan(x, bits):
    """Plain symmetric per-channel INT (baseline)."""
    dtype = x.dtype
    xf = x.float()
    n = xf.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    d = xf / n
    qm = 2**(bits-1) - 1
    s = d.abs().amax(dim=-2, keepdim=True).clamp(min=1e-12) / qm
    dq = ((d / s).round().clamp(-qm, qm)) * s
    dq = dq / dq.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return (n * dq).to(dtype)


def qa_nsep_pchan_2tier(x, bits_lo=3, bits_hi=6, k=4, select_by="max"):
    """2-tier: top-K channels get bits_hi, rest bits_lo. Selection by
    'max' (channel abs-max) or 'energy' (mean squared magnitude)."""
    dtype = x.dtype
    xf = x.float()
    n = xf.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    d = xf / n

    if select_by == "max":
        score = d.abs().amax(dim=-2, keepdim=True).clamp(min=1e-12)
    elif select_by == "energy":
        score = d.pow(2).mean(dim=-2, keepdim=True).clamp(min=1e-12).sqrt()
    else:
        raise ValueError(f"unknown select_by: {select_by}")

    per_chan_max = d.abs().amax(dim=-2, keepdim=True).clamp(min=1e-12)
    pcm = score.squeeze(-2)
    _, topk_idx = pcm.topk(k, dim=-1)
    mask = torch.zeros_like(pcm, dtype=torch.bool)
    mask.scatter_(-1, topk_idx, True)
    mask = mask.unsqueeze(-2)

    qm_lo = 2**(bits_lo-1) - 1
    qm_hi = 2**(bits_hi-1) - 1
    s_lo = per_chan_max / qm_lo
    s_hi = per_chan_max / qm_hi

    d_lo = (d / s_lo).round().clamp(-qm_lo, qm_lo) * s_lo
    d_hi = (d / s_hi).round().clamp(-qm_hi, qm_hi) * s_hi
    d_q = torch.where(mask, d_hi, d_lo)

    d_q = d_q / d_q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return (n * d_q).to(dtype)


def qa_nsep_pchan_asym(x, bits):
    """Per-channel ASYMMETRIC quantization (min+range rather than absmax).

    For bits=3: 8 levels over [min_c, max_c] per channel, whereas the
    symmetric variant uses 7 levels over [-absmax, +absmax]. Asym fits
    biased (post-RoPE) channel distributions better.
    """
    dtype = x.dtype
    xf = x.float()
    n = xf.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    d = xf / n

    d_min = d.amin(dim=-2, keepdim=True)
    d_max = d.amax(dim=-2, keepdim=True)
    d_range = (d_max - d_min).clamp(min=1e-12)
    levels = 2**bits - 1   # 0..levels
    step = d_range / levels

    d_q = ((d - d_min) / step).round().clamp(0, levels) * step + d_min

    d_q = d_q / d_q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return (n * d_q).to(dtype)


def qa_nsep_pchan_asym_2tier(x, bits_lo=3, bits_hi=6, k=4, select_by="max"):
    """Composition: asymmetric + 2-tier. Top-K channels use INT-bits_hi
    asymmetric; rest use INT-bits_lo asymmetric."""
    dtype = x.dtype
    xf = x.float()
    n = xf.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    d = xf / n

    d_min = d.amin(dim=-2, keepdim=True)
    d_max = d.amax(dim=-2, keepdim=True)
    d_range = (d_max - d_min).clamp(min=1e-12)

    if select_by == "max":
        score = d.abs().amax(dim=-2, keepdim=True)
    elif select_by == "energy":
        score = d.pow(2).mean(dim=-2, keepdim=True).sqrt()
    else:
        raise ValueError(f"unknown select_by: {select_by}")

    pcm = score.squeeze(-2)
    _, topk_idx = pcm.topk(k, dim=-1)
    mask = torch.zeros_like(pcm, dtype=torch.bool)
    mask.scatter_(-1, topk_idx, True)
    mask = mask.unsqueeze(-2)

    levels_lo = 2**bits_lo - 1
    levels_hi = 2**bits_hi - 1
    step_lo = d_range / levels_lo
    step_hi = d_range / levels_hi

    d_lo = ((d - d_min) / step_lo).round().clamp(0, levels_lo) * step_lo + d_min
    d_hi = ((d - d_min) / step_hi).round().clamp(0, levels_hi) * step_hi + d_min
    d_q = torch.where(mask, d_hi, d_lo)

    d_q = d_q / d_q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return (n * d_q).to(dtype)


# =============================================================
# === CELL V3 === Real KV extraction
# =============================================================

WIKITEXT_PROMPT = (
    " = Robert Boulter = \n\n Robert Boulter is an English film , "
    "television and theatre actor . He had a guest @-@ starring role on "
    "the television series The Bill in 2000 . This was followed by a "
    "starring role in the play Herons written by Simon Stephens , "
    "which was performed in 2001 at the Royal Court Theatre . He had a "
    "guest role in the television series Judge John Deed in 2002 . In "
    "2004 Boulter landed a role as \" Craig \" in the episode \" Teddy "
    "'s Story \" of the television series The Long Firm ; he starred "
    "alongside actors Mark Strong and Derek Jacobi . He was cast in "
    "the 2005 theatre productions of the Philip Ridley play Mercury "
    "Fur , which was performed at the Drum Theatre in Plymouth and the "
    "Menier Chocolate Factory in London ."
)


def extract_layer0_k(model_id, prompt, dtype=torch.float16):
    print(f"  Loading {model_id} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map="auto", trust_remote_code=True,
    )
    mdl.eval()

    captured = {}
    def hook(module, inputs, output):
        captured["k"] = output.detach().cpu()
    try:
        layer0 = mdl.model.layers[0]
    except AttributeError:
        layer0 = mdl.get_decoder().layers[0]
    handle = layer0.self_attn.k_proj.register_forward_hook(hook)
    inputs = tok(prompt, return_tensors="pt").to(mdl.device)
    with torch.no_grad():
        _ = mdl(**inputs)
    handle.remove()

    raw = captured["k"][0]
    cfg = mdl.config
    head_dim = getattr(cfg, "head_dim",
                       cfg.hidden_size // cfg.num_attention_heads)
    n_kv_heads = getattr(cfg, "num_key_value_heads",
                         cfg.num_attention_heads)
    T = raw.shape[0]
    k = raw.view(T, n_kv_heads, head_dim).transpose(0, 1).contiguous().float()

    del mdl, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return k, {"head_dim": head_dim, "n_kv_heads": n_kv_heads, "T": T}


# =============================================================
# === CELL V4 === Method zoo for the sweep
# =============================================================

def method_list(D):
    """All candidate methods to benchmark (name, callable, eff_bits)."""
    def eff_2tier(bits_lo, bits_hi, k):
        return bits_lo + (bits_hi - bits_lo) * k / D

    methods = [
        # --- baseline ---
        ("plain_INT3",              lambda x: qa_nsep_pchan(x, 3),                                         3.000),
        ("plain_INT4",              lambda x: qa_nsep_pchan(x, 4),                                         4.000),

        # --- (A) K sweep with 2-tier (INT3+6), max-based ---
        ("2tier_3+6_K2_max",        lambda x: qa_nsep_pchan_2tier(x, 3, 6, 2,  "max"),      eff_2tier(3, 6, 2)),
        ("2tier_3+6_K4_max",        lambda x: qa_nsep_pchan_2tier(x, 3, 6, 4,  "max"),      eff_2tier(3, 6, 4)),
        ("2tier_3+6_K8_max",        lambda x: qa_nsep_pchan_2tier(x, 3, 6, 8,  "max"),      eff_2tier(3, 6, 8)),
        ("2tier_3+6_K16_max",       lambda x: qa_nsep_pchan_2tier(x, 3, 6, 16, "max"),      eff_2tier(3, 6, 16)),
        ("2tier_3+6_K32_max",       lambda x: qa_nsep_pchan_2tier(x, 3, 6, 32, "max"),      eff_2tier(3, 6, 32)),

        # --- (B) Selection criterion: max vs energy ---
        ("2tier_3+6_K8_energy",     lambda x: qa_nsep_pchan_2tier(x, 3, 6, 8,  "energy"),   eff_2tier(3, 6, 8)),
        ("2tier_3+6_K16_energy",    lambda x: qa_nsep_pchan_2tier(x, 3, 6, 16, "energy"),   eff_2tier(3, 6, 16)),

        # --- (C) Asymmetric quantization ---
        ("asym_INT3",               lambda x: qa_nsep_pchan_asym(x, 3),                                    3.000),
        ("asym_INT4",               lambda x: qa_nsep_pchan_asym(x, 4),                                    4.000),
        ("asym_2tier_3+6_K4_max",   lambda x: qa_nsep_pchan_asym_2tier(x, 3, 6, 4,  "max"),  eff_2tier(3, 6, 4)),
        ("asym_2tier_3+6_K8_max",   lambda x: qa_nsep_pchan_asym_2tier(x, 3, 6, 8,  "max"),  eff_2tier(3, 6, 8)),
        ("asym_2tier_3+6_K8_energy",lambda x: qa_nsep_pchan_asym_2tier(x, 3, 6, 8,  "energy"),eff_2tier(3, 6, 8)),
    ]
    return methods


def evaluate_method(k_tensor, method_callable):
    """Per-head MSE aggregated mean for a method."""
    n_heads, T, D = k_tensor.shape
    mse_list, wins_vs_plain = [], 0

    plain_mse_per_head = []
    for h in range(n_heads):
        x = k_tensor[h].float()
        y_p = qa_nsep_pchan(x, bits=3)
        plain_mse_per_head.append(float((x - y_p).pow(2).mean()))

    for h in range(n_heads):
        x = k_tensor[h].float()
        y_m = method_callable(x)
        mse = float((x - y_m).pow(2).mean())
        mse_list.append(mse)
        if mse < plain_mse_per_head[h]:
            wins_vs_plain += 1

    mean_mse       = sum(mse_list) / len(mse_list)
    mean_plain_mse = sum(plain_mse_per_head) / len(plain_mse_per_head)
    return {
        "mse_mean":          mean_mse,
        "mse_per_head":      mse_list,
        "plain_mse_mean":    mean_plain_mse,
        "ratio_vs_plain":    mean_mse / max(mean_plain_mse, 1e-12),
        "gain_pct":          (1 - mean_mse / max(mean_plain_mse, 1e-12)) * 100,
        "heads_win_vs_plain": wins_vs_plain,
        "n_heads":           n_heads,
    }


# =============================================================
# === CELL V5 === Run sweep
# =============================================================

MODELS = [
    {"id": "Qwen/Qwen2-7B",           "label": "Qwen2-7B",     "expected_mode": "Mode 1"},
    {"id": "meta-llama/Llama-3.1-8B", "label": "Llama-3.1-8B", "expected_mode": "Mode 2"},
]

all_results = {}
for m in MODELS:
    print(f"\n============================================================")
    print(f"=== {m['label']} (expected {m['expected_mode']}) ===")
    print(f"============================================================")
    try:
        k, meta = extract_layer0_k(m["id"], WIKITEXT_PROMPT)
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        all_results[m["label"]] = {"error": f"{type(e).__name__}: {e}"}
        continue

    print(f"  K shape: {tuple(k.shape)} (n_kv_heads, T, head_dim)")
    D = k.shape[-1]

    methods = method_list(D)
    model_results = []

    # Header
    print(f"\n  {'method':<32s}  {'eff_bits':>8s}  {'mse':>9s}  "
          f"{'ratio':>6s}  {'gain':>7s}  {'wins':>6s}")
    print(f"  {'-'*32}  {'-'*8}  {'-'*9}  {'-'*6}  {'-'*7}  {'-'*6}")

    for (name, fn, eff_bits) in methods:
        try:
            r = evaluate_method(k, fn)
            model_results.append({
                "method":    name,
                "eff_bits":  eff_bits,
                **r,
            })
            print(f"  {name:<32s}  {eff_bits:>8.3f}  "
                  f"{r['mse_mean']:>9.5f}  "
                  f"{r['ratio_vs_plain']:>6.3f}  "
                  f"{r['gain_pct']:>+6.1f}%  "
                  f"{r['heads_win_vs_plain']}/{r['n_heads']}")
        except Exception as e:
            print(f"  {name:<32s}  FAILED: {type(e).__name__}: {e}")

    all_results[m["label"]] = {
        "model_id":      m["id"],
        "expected_mode": m["expected_mode"],
        "meta":          meta,
        "methods":       model_results,
    }


# =============================================================
# === CELL V6 === Summary & save
# =============================================================

print("\n" + "=" * 72)
print(f"Phase 13 summary ({datetime.now().isoformat(timespec='seconds')})")
print("=" * 72)

# Per-model leaderboard (sorted by gain_pct descending)
for label, r in all_results.items():
    if "error" in r:
        continue
    print(f"\n### {label} ({r['expected_mode']}) leaderboard ###")
    sorted_methods = sorted(r["methods"], key=lambda m: -m["gain_pct"])
    print(f"  {'method':<32s}  {'eff_bits':>8s}  {'gain':>7s}  {'wins':>6s}")
    for mm in sorted_methods:
        print(f"  {mm['method']:<32s}  {mm['eff_bits']:>8.3f}  "
              f"{mm['gain_pct']:>+6.1f}%  "
              f"{mm['heads_win_vs_plain']}/{mm['n_heads']}")

# Find the best INT3-class method that helps BOTH models
print("\n### Cross-model winners (gain > 0 on both; eff_bits <= 3.25) ###")
qwen_methods = {mm["method"]: mm for mm in all_results.get("Qwen2-7B",     {}).get("methods", [])}
llama_methods = {mm["method"]: mm for mm in all_results.get("Llama-3.1-8B", {}).get("methods", [])}

both_win = []
for name in sorted(qwen_methods.keys()):
    if name not in llama_methods:
        continue
    q, l = qwen_methods[name], llama_methods[name]
    if q["eff_bits"] > 3.25:
        continue
    if q["gain_pct"] > 0 and l["gain_pct"] > 0:
        both_win.append({
            "method":    name,
            "eff_bits":  q["eff_bits"],
            "qwen_gain": q["gain_pct"],
            "llama_gain": l["gain_pct"],
        })

if both_win:
    print(f"  {'method':<32s}  {'eff_bits':>8s}  {'qwen':>7s}  {'llama':>7s}")
    for r in sorted(both_win, key=lambda x: -(x["qwen_gain"] + x["llama_gain"])):
        print(f"  {r['method']:<32s}  {r['eff_bits']:>8.3f}  "
              f"{r['qwen_gain']:>+6.1f}%  {r['llama_gain']:>+6.1f}%")
else:
    print("  (no method beats plain on BOTH at <=3.25 eff_bits)")

# Save
out_path = "/content/phase13_full_sweep.json"
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved -> {out_path}")
