# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Phase 12: Real-Model 2-Tier Validation (v0.3 pivot)
# ============================================================
# Phase 10 (real Qwen2-7B Layer 0 K) killed the rotation idea
# (3.68x MSE regression). The distribution is a DISCRETE plateau
# (top-5 channels ~130-167 magnitude, rest ~10), not heavy-tailed,
# so rotation's outlier-spreading offers no benefit on top of
# per-channel absmax.
#
# Phase 11 synthetic sweep showed 2-tier nsep+pchan (top-K at 6
# bits, rest at 3 bits) HELPS on all 6 regimes including a 44.7%
# MSE reduction on the Qwen2-plateau cartoon at the 3.09 eff-bits
# target.
#
# This phase repeats Phase 10's hook-based real-KV extraction but
# compares plain INT3 vs 2-tier INT3+6 (K=4) instead of rotation.
# Decision:
#   HELPS  -> proceed to full WikiText-2 PPL eval
#   TIES   -> synthetic overstates the gain; rethink
#   HURTS  -> pivot again (would be a surprise given Phase 11)
#
# Runtime: ~5-10 min on Colab A100 (model loads + 1 forward pass each)
# ============================================================


# =============================================================
# === CELL V1 === Setup (paste your HF token below)
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
# === CELL V2 === Inline quantizers (plain INT3 + 2-tier)
# =============================================================
# Self-contained: does NOT depend on `pip install modeq`.

def qa_nsep_pchan(x, bits):
    """Plain norm-separated per-channel INT quantization."""
    dtype = x.dtype
    xf = x.float()
    n = xf.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    d = xf / n
    qm = 2**(bits-1) - 1
    s = d.abs().amax(dim=-2, keepdim=True).clamp(min=1e-12) / qm
    dq = ((d / s).round().clamp(-qm, qm)) * s
    dq = dq / dq.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return (n * dq).to(dtype)


def qa_nsep_pchan_2tier(x, bits_lo=3, bits_hi=6, k=4):
    """Top-K highest-magnitude channels get bits_hi, rest get bits_lo.

    Effective bits/value = bits_lo + (bits_hi - bits_lo) * k / D.
    For defaults (3, 6, k=4, D=128): 3 + 12/128 = 3.09 bits/value.
    """
    dtype = x.dtype
    xf = x.float()
    n = xf.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    d = xf / n

    per_chan_max = d.abs().amax(dim=-2, keepdim=True).clamp(min=1e-12)
    pcm = per_chan_max.squeeze(-2)
    _, topk_idx = pcm.topk(k, dim=-1)
    mask = torch.zeros_like(pcm, dtype=torch.bool)
    mask.scatter_(-1, topk_idx, True)
    mask = mask.unsqueeze(-2)

    qm_lo = 2**(bits_lo-1) - 1
    qm_hi = 2**(bits_hi-1) - 1
    s_lo  = per_chan_max / qm_lo
    s_hi  = per_chan_max / qm_hi

    d_lo = (d / s_lo).round().clamp(-qm_lo, qm_lo) * s_lo
    d_hi = (d / s_hi).round().clamp(-qm_hi, qm_hi) * s_hi
    d_q  = torch.where(mask, d_hi, d_lo)

    d_q = d_q / d_q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return (n * d_q).to(dtype)


# =============================================================
# === CELL V3 === Real KV extraction via forward hook
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
    """Run one forward pass and capture Layer 0 K via forward hook."""
    print(f"  Loading {model_id} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
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

    raw = captured["k"][0]  # (T, n_kv_heads * head_dim)
    cfg = mdl.config
    head_dim   = getattr(cfg, "head_dim",
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
# === CELL V4 === Per-head plain vs 2-tier comparison
# =============================================================

def compare_plain_vs_2tier(k_tensor, bits_lo=3, bits_hi=6, top_k=4):
    n_heads, T, D = k_tensor.shape
    assert top_k < D, f"top_k {top_k} must be < D {D}"

    mse_plain, mse_2tier = [], []
    for h in range(n_heads):
        x = k_tensor[h].float()
        y_p = qa_nsep_pchan(x, bits=bits_lo)
        y_t = qa_nsep_pchan_2tier(x, bits_lo=bits_lo, bits_hi=bits_hi, k=top_k)
        mse_plain.append(float((x - y_p).pow(2).mean()))
        mse_2tier.append(float((x - y_t).pow(2).mean()))

    mse_p = torch.tensor(mse_plain)
    mse_t = torch.tensor(mse_2tier)
    return {
        "bits_lo":                 bits_lo,
        "bits_hi":                 bits_hi,
        "top_k":                   top_k,
        "n_heads":                 n_heads,
        "mse_plain_per_head":      mse_plain,
        "mse_2tier_per_head":      mse_2tier,
        "mse_plain_mean":          float(mse_p.mean()),
        "mse_2tier_mean":          float(mse_t.mean()),
        "mse_ratio_2tier_over_plain": float(mse_t.mean() / mse_p.mean().clamp(min=1e-12)),
        "gain_pct":                float((1 - mse_t.mean() / mse_p.mean().clamp(min=1e-12)) * 100),
        "n_heads_2tier_wins":      int((mse_t < mse_p).sum()),
        "effective_bits_per_value": bits_lo + (bits_hi - bits_lo) * top_k / D,
    }


# =============================================================
# === CELL V5 === Run on Qwen2-7B and Llama-3.1-8B
# =============================================================

MODELS = [
    {"id": "Qwen/Qwen2-7B",           "label": "Qwen2-7B",      "expected_mode": "Mode 1"},
    {"id": "meta-llama/Llama-3.1-8B", "label": "Llama-3.1-8B",  "expected_mode": "Mode 2"},
]

all_results = {}
for m in MODELS:
    print(f"\n=== {m['label']} (expected {m['expected_mode']}) ===")
    try:
        k, meta = extract_layer0_k(m["id"], WIKITEXT_PROMPT)
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        all_results[m["label"]] = {"error": f"{type(e).__name__}: {e}"}
        continue

    print(f"  K shape: {tuple(k.shape)} (n_kv_heads, T, head_dim)")
    # Quick distribution recap (to confirm we still see the plateau)
    per_chan_max = k.reshape(-1, k.shape[-1]).abs().amax(dim=0)
    top5 = per_chan_max.topk(5).values.tolist()
    print(f"  per_chan_max top-5: {[round(v, 2) for v in top5]}")

    cmp_ = compare_plain_vs_2tier(k, bits_lo=3, bits_hi=6, top_k=4)
    print(f"  INT3 mse plain={cmp_['mse_plain_mean']:.5f} "
          f"2tier={cmp_['mse_2tier_mean']:.5f} "
          f"ratio={cmp_['mse_ratio_2tier_over_plain']:.3f} "
          f"gain={cmp_['gain_pct']:+.1f}%")
    print(f"  heads where 2tier wins: {cmp_['n_heads_2tier_wins']}/"
          f"{cmp_['n_heads']}")
    print(f"  effective bits/value: {cmp_['effective_bits_per_value']:.3f}")

    ratio = cmp_["mse_ratio_2tier_over_plain"]
    if ratio < 0.70:
        verdict = "2-TIER DOMINATES (>30% MSE reduction)"
    elif ratio < 0.90:
        verdict = "2-TIER HELPS (10-30% MSE reduction)"
    elif ratio < 1.05:
        verdict = "2-TIER ~ TIES"
    else:
        verdict = "2-TIER HURTS (regression)"
    print(f"  -> {verdict}")

    all_results[m["label"]] = {
        "model_id":         m["id"],
        "expected_mode":    m["expected_mode"],
        "meta":             meta,
        "per_chan_max_top5": top5,
        "comparison":       cmp_,
        "verdict":          verdict,
    }


# =============================================================
# === CELL V6 === Summary + save JSON
# =============================================================

print("\n" + "=" * 60)
print(f"Phase 12 summary ({datetime.now().isoformat(timespec='seconds')})")
print("=" * 60)

for label, r in all_results.items():
    if "error" in r:
        print(f"\n{label}: ERROR -- {r['error']}")
        continue
    c = r["comparison"]
    print(f"\n{label} ({r['expected_mode']}):")
    print(f"  plain INT3  MSE: {c['mse_plain_mean']:.5f}")
    print(f"  2-tier      MSE: {c['mse_2tier_mean']:.5f}")
    print(f"  ratio:           {c['mse_ratio_2tier_over_plain']:.3f} "
          f"(gain {c['gain_pct']:+.1f}%)")
    print(f"  2tier wins:      {c['n_heads_2tier_wins']}/{c['n_heads']} heads")
    print(f"  effective bits:  {c['effective_bits_per_value']:.3f} bits/value")
    print(f"  verdict:         {r['verdict']}")

out_path = "/content/phase12_real_kv_2tier.json"
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved -> {out_path}")

# Final decision
print("\n" + "=" * 60)
print("DECISION")
print("=" * 60)
qwen = all_results.get("Qwen2-7B", {})
llama = all_results.get("Llama-3.1-8B", {})

qwen_helps = isinstance(qwen, dict) and "comparison" in qwen and \
             qwen["comparison"]["mse_ratio_2tier_over_plain"] < 0.90
llama_ok = isinstance(llama, dict) and "comparison" in llama and \
           llama["comparison"]["mse_ratio_2tier_over_plain"] < 1.10

if qwen_helps and llama_ok:
    print("  -> PROCEED to full WikiText-2 PPL evaluation.")
    print("     Qwen2-7B gains meaningfully and Llama doesn't regress.")
elif qwen_helps:
    print("  -> PROCEED WITH CARE. Qwen2 gains but Llama loses or ties;")
    print("     make 2-tier a Mode-1-conditional dispatch rather than")
    print("     universal default.")
else:
    print("  -> PIVOT AGAIN. Synthetic gains did not transfer to real KV.")
