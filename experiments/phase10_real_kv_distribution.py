# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Phase 10: Real-Model KV Distribution Probe (v0.3 pre-flight)
# ============================================================
# Phase 9 (synthetic, 5 regimes) finding:
#   Walsh-Hadamard rotation on top of nsep+pchan INT3 *HELPS*
#   on heavy-tailed distributions (Student-t df=3, 3.5x better)
#   but *HURTS* on discrete channel-outlier distributions
#   (per-channel absmax already optimal there).
#
# Open question: is real Qwen2-7B Layer 0 K closer to
#   - regime 2/3 (discrete outlier -> rotation hurts)    or
#   - regime 4   (heavy-tailed    -> rotation helps 3.5x) ?
#
# This phase answers that by loading the real models, pulling
# Layer 0 K via a forward hook, measuring distribution shape
# (kurtosis, Hill tail index, per-channel ratio), and running
# plain vs rotated INT3 reconstruction MSE head-by-head.
#
# Decision:
#   HELPS  -> proceed to full WikiText-2 PPL evaluation (v0.3)
#   HURTS  -> pivot v0.3 to a different angle
#   MIXED  -> inspect which models win and re-scope
#
# Models (both gated -> need HF token):
#   Qwen/Qwen2-7B             (Mode 1 reference)
#   meta-llama/Llama-3.1-8B   (Mode 2 reference)
#
# Runtime: ~5-10 min (model loads + 1 forward pass each)
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

import json, math
import torch.nn.functional as F
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer


# =============================================================
# === CELL V2 === Rotation primitives + quantizers (inline)
# =============================================================
# v0.3 candidate mechanism: deterministic Walsh-Hadamard
# rotation before nsep+pchan INT3. Self-contained -- does NOT
# depend on installing the modeq package.

def hadamard_matrix(d, dtype=torch.float32, device=None):
    """dxd orthonormal Walsh-Hadamard. d must be a power of 2."""
    if d <= 0 or (d & (d - 1)) != 0:
        raise ValueError(f"d must be positive power of 2, got {d}")
    H = torch.ones(1, 1, dtype=dtype, device=device)
    while H.shape[0] < d:
        top = torch.cat([H, H], dim=1)
        bot = torch.cat([H, -H], dim=1)
        H = torch.cat([top, bot], dim=0)
    return H / math.sqrt(d)


_HADAMARD_CACHE = {}
def cached_hadamard(d, dtype, device):
    key = (d, dtype, str(device))
    if key not in _HADAMARD_CACHE:
        _HADAMARD_CACHE[key] = hadamard_matrix(d, dtype=dtype, device=device)
    return _HADAMARD_CACHE[key]


def rotate(x):
    d = x.shape[-1]
    H = cached_hadamard(d, torch.float32, x.device)
    return x.float() @ H


def unrotate(x):
    d = x.shape[-1]
    H = cached_hadamard(d, torch.float32, x.device)
    return x.float() @ H.T


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


def qa_nsep_pchan_rotated(x, bits):
    """Walsh-Hadamard rotated nsep+pchan (v0.3 candidate)."""
    dtype = x.dtype
    x_rot   = rotate(x)
    x_rot_q = qa_nsep_pchan(x_rot, bits)
    x_back  = unrotate(x_rot_q)
    return x_back.to(dtype)


# =============================================================
# === CELL V3 === Distribution statistics
# =============================================================

def channel_stats(k_tensor):
    """
    k_tensor: (n_heads, T, head_dim). Returns per-channel stats
    aggregated over (head, token) as one sample axis.
    """
    n_heads, T, D = k_tensor.shape
    flat = k_tensor.reshape(n_heads * T, D).float()

    per_chan_max     = flat.abs().amax(dim=0)                          # (D,)
    per_chan_meanabs = flat.abs().mean(dim=0)                          # (D,)
    per_chan_ratio   = per_chan_max / per_chan_meanabs.clamp(min=1e-12)

    return {
        "D": int(D),
        "per_chan_max_mean":   float(per_chan_max.mean()),
        "per_chan_max_max":    float(per_chan_max.max()),
        "per_chan_max_std":    float(per_chan_max.std()),
        "per_chan_max_top5":   [float(v) for v in per_chan_max.topk(5).values],
        "per_chan_ratio_mean": float(per_chan_ratio.mean()),
        "per_chan_ratio_max":  float(per_chan_ratio.max()),
    }


def kurtosis_per_channel(k_tensor):
    """
    Excess kurtosis per channel. Normal ~0, Student-t df=3 ~6+,
    single-outlier channel pattern shows 1 spiky channel only.
    """
    n_heads, T, D = k_tensor.shape
    flat = k_tensor.reshape(n_heads * T, D).float()
    mu       = flat.mean(dim=0, keepdim=True)
    centered = flat - mu
    var      = (centered ** 2).mean(dim=0).clamp(min=1e-12)
    m4       = (centered ** 4).mean(dim=0)
    kurt     = m4 / (var ** 2) - 3.0
    return {
        "kurt_mean_over_channels":         float(kurt.mean()),
        "kurt_max_over_channels":          float(kurt.max()),
        "kurt_min_over_channels":          float(kurt.min()),
        "kurt_median_over_channels":       float(kurt.median()),
        "kurt_frac_channels_heavy_tailed": float((kurt > 3.0).float().mean()),
    }


def hill_tail_index(values, k_frac=0.01):
    """
    Hill estimator on top k_frac fraction of |values|.
    Lower = heavier tail. Gaussian ~3+, Student-t df=3 ~3.
    """
    v = values.abs().float().flatten()
    n = v.numel()
    k = max(2, int(k_frac * n))
    topk = v.topk(k).values
    topk, _ = topk.sort(descending=True)
    x_k = topk[-1].clamp(min=1e-12)
    denom = torch.log(topk[:-1] / x_k).mean()
    if denom <= 0:
        return float("inf")
    return float(1.0 / denom)


def compare_plain_vs_rotated(k_tensor, bits=3):
    """Per-head plain vs rotated INT3 reconstruction MSE."""
    n_heads, T, D = k_tensor.shape
    assert (D & (D-1)) == 0, f"head_dim {D} must be power of 2"

    mse_plain, mse_rot = [], []
    for h in range(n_heads):
        x = k_tensor[h].float()
        y_p = qa_nsep_pchan(x, bits=bits)
        y_r = qa_nsep_pchan_rotated(x, bits=bits)
        mse_plain.append(float((x - y_p).pow(2).mean()))
        mse_rot.append(float((x - y_r).pow(2).mean()))

    mse_p = torch.tensor(mse_plain)
    mse_r = torch.tensor(mse_rot)
    return {
        "bits":                     bits,
        "n_heads":                  n_heads,
        "mse_plain_per_head":       mse_plain,
        "mse_rot_per_head":         mse_rot,
        "mse_plain_mean":           float(mse_p.mean()),
        "mse_rot_mean":             float(mse_r.mean()),
        "mse_ratio_rot_over_plain": float(mse_r.mean() / mse_p.mean().clamp(min=1e-12)),
        "n_heads_rotation_helps":   int((mse_r < mse_p).sum()),
    }


# =============================================================
# === CELL V4 === Real KV extraction via forward hook
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
    """
    Load model, run one forward pass, hook into Layer 0 k_proj,
    return captured K reshaped to (n_kv_heads, T, head_dim).
    """
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

    # Find Layer 0 attention's k_proj
    try:
        layer0 = mdl.model.layers[0]
    except AttributeError:
        layer0 = mdl.get_decoder().layers[0]
    handle = layer0.self_attn.k_proj.register_forward_hook(hook)

    inputs = tok(prompt, return_tensors="pt").to(mdl.device)
    with torch.no_grad():
        _ = mdl(**inputs)
    handle.remove()

    # captured["k"] shape: (1, T, n_kv_heads * head_dim)
    raw = captured["k"][0]  # (T, n_kv_heads * head_dim)

    cfg = mdl.config
    head_dim   = getattr(cfg, "head_dim",
                         cfg.hidden_size // cfg.num_attention_heads)
    n_kv_heads = getattr(cfg, "num_key_value_heads",
                         cfg.num_attention_heads)
    T = raw.shape[0]

    # (T, n_kv_heads, head_dim) -> (n_kv_heads, T, head_dim)
    k = raw.view(T, n_kv_heads, head_dim).transpose(0, 1).contiguous().float()

    # Free GPU memory before next model
    del mdl, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return k, {"head_dim": head_dim, "n_kv_heads": n_kv_heads, "T": T}


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

    print(f"  K shape: {tuple(k.shape)}  (n_kv_heads, T, head_dim)")
    print(f"  meta: {meta}")

    stats = channel_stats(k)
    kurt  = kurtosis_per_channel(k)
    hill  = hill_tail_index(k.flatten(), k_frac=0.01)
    cmp_  = compare_plain_vs_rotated(k, bits=3)

    print(f"  per_chan_ratio_max (outlier signature): "
          f"{stats['per_chan_ratio_max']:.2f}x")
    print(f"  per_chan_max top-5: "
          f"{[round(v, 3) for v in stats['per_chan_max_top5']]}")
    print(f"  kurtosis mean/max/heavy-tail frac: "
          f"{kurt['kurt_mean_over_channels']:.2f} / "
          f"{kurt['kurt_max_over_channels']:.2f} / "
          f"{kurt['kurt_frac_channels_heavy_tailed']:.2%}")
    print(f"  Hill tail index (top 1%): {hill:.2f}  "
          f"(lower=heavier; Gaussian~3, Student-t df3~3)")
    print(f"  INT3 mse: plain={cmp_['mse_plain_mean']:.5f} "
          f"rot={cmp_['mse_rot_mean']:.5f} "
          f"ratio={cmp_['mse_ratio_rot_over_plain']:.3f} "
          f"(heads where rot wins: {cmp_['n_heads_rotation_helps']}/"
          f"{cmp_['n_heads']})")

    ratio = cmp_["mse_ratio_rot_over_plain"]
    if ratio < 0.95:
        verdict = "ROTATION HELPS on real Layer 0 K"
    elif ratio < 1.05:
        verdict = "ROTATION ~ TIES on real Layer 0 K"
    else:
        verdict = "ROTATION HURTS on real Layer 0 K"
    print(f"  -> {verdict}")

    all_results[m["label"]] = {
        "model_id":                 m["id"],
        "expected_mode":            m["expected_mode"],
        "meta":                     meta,
        "channel_stats":            stats,
        "kurtosis":                 kurt,
        "hill_tail_index_top1pct":  hill,
        "int3_comparison":          cmp_,
        "verdict":                  verdict,
    }


# =============================================================
# === CELL V6 === Summary + save JSON
# =============================================================

print("\n" + "=" * 60)
print(f"Phase 10 summary ({datetime.now().isoformat(timespec='seconds')})")
print("=" * 60)

for label, r in all_results.items():
    if "error" in r:
        print(f"\n{label}: ERROR -- {r['error']}")
        continue
    c = r["int3_comparison"]
    print(f"\n{label} ({r['expected_mode']}):")
    print(f"  plain INT3 MSE: {c['mse_plain_mean']:.5f}")
    print(f"  rot   INT3 MSE: {c['mse_rot_mean']:.5f}")
    print(f"  ratio (rot/plain): {c['mse_ratio_rot_over_plain']:.3f}")
    print(f"  heads rotation wins: {c['n_heads_rotation_helps']}/{c['n_heads']}")
    print(f"  kurtosis mean: {r['kurtosis']['kurt_mean_over_channels']:.2f}")
    print(f"  Hill tail index: {r['hill_tail_index_top1pct']:.2f}")
    print(f"  verdict: {r['verdict']}")

# Persist results
out_path = "/content/phase10_real_kv_distribution.json"
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved -> {out_path}")

# Final decision banner
print("\n" + "=" * 60)
print("DECISION")
print("=" * 60)
total = sum(1 for r in all_results.values() if "error" not in r)
helped = sum(1 for r in all_results.values()
             if isinstance(r, dict) and "HELPS" in r.get("verdict", ""))
if total == 0:
    print("  No models evaluated successfully; cannot decide.")
elif helped == total:
    print("  -> PROCEED to full WikiText-2 PPL evaluation (v0.3 alive).")
elif helped >= 1:
    print("  -> MIXED. Rotation helps on some but not all targets.")
    print("     Run full PPL eval only on the winners.")
else:
    print("  -> PIVOT. Rotation is empirically redundant with nsep+pchan")
    print("     on our real target models. Rethink v0.3 angle.")
