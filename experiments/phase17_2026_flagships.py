# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Phase 17: asym INT3 on 2026 flagship open models
# ============================================================
# v0.3.2 Phase 16 validated asym INT3 on 6 models spanning 2023-
# early 2024 architectures. This phase checks whether the claim
# generalizes to the latest flagship open models released after
# our v0.2 sweep:
#
#   Qwen3.5-9B      (Alibaba, released 2026-02-16; dense, base
#                    variant to avoid the default thinking-mode
#                    wrapper; 262K native context extendable to
#                    1M; post-Qwen2 recipe iteration)
#   gemma-4-E4B     (Google DeepMind, released 2026-04-02;
#                    "Efficient" dense variant in the Gemma 4
#                    family; 256K context; multimodal-capable
#                    but we feed text only; Apache 2 license)
#
# Both fit comfortably in an A100 40GB FP16.
#
# If asym INT3 beats sym INT3 on BOTH at 3.0 bits/value, the
# combined 8-model claim becomes "validated on every 2023-2026
# open-weight model we could run" -- a concrete generalization
# signal beyond the v0.2 sweep.
#
# Methods (per model; ctx=1024, sliding window 10 windows):
#   baseline (FP16)
#   nsep+pchan INT3 symmetric   (v0.2 reference)
#   nsep+pchan INT3 asymmetric  (v0.3 hero)
#   nsep+pchan INT4 symmetric   (1-bit-higher reference)
#
# Runtime: ~40-60 min on Colab A100.
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

# Qwen3.5 and Gemma 4 both need up-to-date transformers.
!pip install -q -U transformers accelerate hf_transfer datasets
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# ─────────────────────────────────────────────────────────────
# >>>>>  PASTE YOUR HF TOKEN  <<<<<
HF_TOKEN = "hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
# ─────────────────────────────────────────────────────────────
from huggingface_hub import login
login(token=HF_TOKEN)

import json, numpy as np
import torch.nn.functional as F
from datetime import datetime
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# =============================================================
# === CELL V2 === Quantizers (inline, identical to Phase 14/16)
# =============================================================

def qa_nsep_pchan_sym(x, bits):
    dtype = x.dtype
    xf = x.float()
    n = xf.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    d = xf / n
    qm = 2**(bits-1) - 1
    s = d.abs().amax(dim=0, keepdim=True).clamp(min=1e-12) / qm
    dq = ((d / s).round().clamp(-qm, qm)) * s
    dq = dq / dq.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return (n * dq).to(dtype)


def qa_nsep_pchan_asym(x, bits):
    dtype = x.dtype
    xf = x.float()
    n = xf.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    d = xf / n
    d_min = d.amin(dim=0, keepdim=True)
    d_max = d.amax(dim=0, keepdim=True)
    d_range = (d_max - d_min).clamp(min=1e-12)
    levels = 2**bits - 1
    step = d_range / levels
    d_q = ((d - d_min) / step).round().clamp(0, levels) * step + d_min
    d_q = d_q / d_q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return (n * d_q).to(dtype)


# =============================================================
# === CELL V3 === KV cache compression + PPL (identical)
# =============================================================

def get_kv(past, li):
    if hasattr(past, 'layers'):
        return past.layers[li].keys, past.layers[li].values
    return past[li][0], past[li][1]

def set_kv(past, li, k, v):
    if hasattr(past, 'layers'):
        past.layers[li].keys = k; past.layers[li].values = v
    else:
        past[li] = (k, v)

def n_cache_layers(past):
    return len(past.layers) if hasattr(past, 'layers') else len(past)

def compress_cache(past, method_fn):
    if method_fn is None:
        return
    for li in range(n_cache_layers(past)):
        ok, ov = get_kv(past, li)
        ok = ok.clone(); ov = ov.clone()
        nk = torch.zeros_like(ok); nv = torch.zeros_like(ov)
        for h in range(ok.shape[1]):
            nk[0, h] = method_fn(ok[0, h]).to(ok.dtype)
            nv[0, h] = method_fn(ov[0, h]).to(ov.dtype)
        set_kv(past, li, nk, nv)


ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
wtext = "\n\n".join([x for x in ds["text"] if x.strip()])[:48000]
print(f"WikiText-2 text length: {len(wtext)} chars")


def ppl_with(method_fn, ctx_len=1024, stride=512, max_windows=10):
    ids = tok.encode(wtext, return_tensors="pt").to("cuda")
    T = ids.shape[1]
    total_nll = 0.0; total_tokens = 0; windows_done = 0

    with torch.inference_mode():
        for begin in range(0, T - ctx_len, stride):
            if windows_done >= max_windows: break
            window = ids[:, begin:begin + ctx_len]
            split = ctx_len // 2
            context = window[:, :split]
            targets = window[:, split:]

            out_ctx = mdl(context, use_cache=True)
            past = out_ctx.past_key_values
            first_logits = out_ctx.logits[:, -1:, :]
            compress_cache(past, method_fn)
            out_tgt = mdl(targets, past_key_values=past, use_cache=False)
            rest_logits = out_tgt.logits[:, :-1, :]
            all_logits = torch.cat([first_logits, rest_logits], dim=1)
            log_probs = F.log_softmax(all_logits[0].float(), dim=-1)
            nll = -log_probs.gather(-1, targets[0].unsqueeze(-1)).squeeze(-1)
            total_nll += nll.sum().item()
            total_tokens += targets.shape[1]
            windows_done += 1

            del out_ctx, out_tgt, past
            gc.collect(); torch.cuda.empty_cache()

    mean_nll = total_nll / max(total_tokens, 1)
    return float(np.exp(mean_nll)), total_tokens


METHODS = [
    ("baseline (FP16)",              None,                                   "16"),
    ("nsep+pchan INT3 symmetric",    lambda x: qa_nsep_pchan_sym(x, 3),      "3"),
    ("nsep+pchan INT3 asymmetric",   lambda x: qa_nsep_pchan_asym(x, 3),     "3"),
    ("nsep+pchan INT4 symmetric",    lambda x: qa_nsep_pchan_sym(x, 4),      "4"),
]


def run_model(model_id, label, extra_kwargs=None):
    global mdl, tok
    try:
        del mdl, tok
    except NameError:
        pass
    gc.collect(); torch.cuda.empty_cache()

    kwargs = dict(torch_dtype=torch.float16, device_map="auto",
                  trust_remote_code=True)
    if extra_kwargs:
        kwargs.update(extra_kwargs)

    print(f"\n{'='*80}")
    print(f"Loading {label} ({model_id}) ...")
    print(f"{'='*80}")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    mdl.eval()
    print(f"VRAM used: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # Report architecture snapshot so we can spot anomalies (head_dim,
    # n_kv_heads, n_layers) without crashing later.
    try:
        cfg = mdl.config
        hd = getattr(cfg, "head_dim",
                     cfg.hidden_size // cfg.num_attention_heads)
        nkv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
        print(f"  arch: n_layers={cfg.num_hidden_layers}, "
              f"n_attn_heads={cfg.num_attention_heads}, "
              f"n_kv_heads={nkv}, head_dim={hd}")
    except Exception as e:
        print(f"  (could not introspect config: {e})")

    print(f"\n  {'Method':<32} | {'bits':>5} | {'PPL':>10} | {'ΔPPL':>10}")
    print(f"  " + "-"*64)

    results = {}
    baseline_ppl = None
    for name, fn, bits in METHODS:
        print(f"  Running {name} ...", flush=True)
        ppl, nt = ppl_with(fn)
        if baseline_ppl is None:
            baseline_ppl = ppl
        delta = ppl - baseline_ppl
        results[name] = {"ppl": ppl, "bits": bits, "delta": delta, "tokens": nt}
        dstr = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
        print(f"  {name:<32} | {bits:>5} | {ppl:>10.4f} | {dstr:>10}")

    sym_d  = results["nsep+pchan INT3 symmetric"]["delta"]
    asym_d = results["nsep+pchan INT3 asymmetric"]["delta"]
    int4_d = results["nsep+pchan INT4 symmetric"]["delta"]
    reduction_pct = (1 - asym_d / max(sym_d, 1e-9)) * 100 if sym_d > 0 else 0

    print(f"\n  sym INT3  -> asym INT3: {sym_d:.4f} -> {asym_d:.4f} "
          f"({reduction_pct:+.1f}% reduction)")
    print(f"  sym INT4 reference: {int4_d:.4f}")

    return {
        "label":              label,
        "model":              model_id,
        "baseline_ppl":       baseline_ppl,
        "results":            results,
        "sym_int3_delta":     sym_d,
        "asym_int3_delta":    asym_d,
        "sym_int4_delta":     int4_d,
        "asym_reduction_pct": reduction_pct,
    }


# =============================================================
# === CELL V4 === Qwen3.5-9B (2026 flagship, dense base)
# =============================================================

qwen35_res = run_model("Qwen/Qwen3.5-9B", "Qwen3.5-9B")


# =============================================================
# === CELL V5 === Gemma 4-E4B (2026 flagship, efficient dense)
# =============================================================

# gemma-4-E4B is multimodal-capable but we feed text only. No image
# input, no audio -- just text tokens through the base model.
gemma4_res = run_model("google/gemma-4-E4B", "gemma-4-E4B")


# =============================================================
# === CELL V6 === Combined 8-model summary
# =============================================================

# Phase 14 + Phase 16 priors
PRIOR = [
    {"label": "Qwen2-7B",        "mode": "Mode 1", "baseline_ppl": 8.3693,
     "sym_int3_delta": 0.4021, "asym_int3_delta": 0.1034, "sym_int4_delta": 0.0544,
     "asym_reduction_pct": 74.3},
    {"label": "Llama-3.1-8B",    "mode": "Mode 2", "baseline_ppl": 7.3840,
     "sym_int3_delta": 0.2541, "asym_int3_delta": 0.1059, "sym_int4_delta": 0.0133,
     "asym_reduction_pct": 58.3},
    {"label": "Pythia-6.9B",     "mode": "Mode 2", "baseline_ppl": 10.7594,
     "sym_int3_delta": 2.3781, "asym_int3_delta": 0.0994, "sym_int4_delta": 0.8355,
     "asym_reduction_pct": 95.8},
    {"label": "Qwen2.5-14B",     "mode": "Mode 2", "baseline_ppl": 1.9575,
     "sym_int3_delta": 0.2608, "asym_int3_delta": 0.0909, "sym_int4_delta": 0.0274,
     "asym_reduction_pct": 65.1},
    {"label": "Mistral-7B-v0.1", "mode": "Mode 2", "baseline_ppl": 6.2983,
     "sym_int3_delta": 0.1243, "asym_int3_delta": 0.0547, "sym_int4_delta": 0.0183,
     "asym_reduction_pct": 56.0},
    {"label": "GPT-J-6B",        "mode": "Mode 2", "baseline_ppl": 10.2922,
     "sym_int3_delta": 0.1873, "asym_int3_delta": 0.0703, "sym_int4_delta": 0.0182,
     "asym_reduction_pct": 62.5},
]

all_results = list(PRIOR)
for res in [qwen35_res, gemma4_res]:
    all_results.append({
        "label":              res["label"],
        "mode":               "TBD",  # Phase 17 didn't measure the L0 K outlier
                                      # ratio; mode inferred from behavior only.
        "baseline_ppl":       res["baseline_ppl"],
        "sym_int3_delta":     res["sym_int3_delta"],
        "asym_int3_delta":    res["asym_int3_delta"],
        "sym_int4_delta":     res["sym_int4_delta"],
        "asym_reduction_pct": res["asym_reduction_pct"],
    })

print("\n" + "=" * 92)
print(f"Phase 17 + 16 + 14 synthesis "
      f"({datetime.now().isoformat(timespec='seconds')})")
print("=" * 92)
print(f"  {'Model':<22} {'mode':<7} {'base':>7} "
      f"{'sym Δ':>8} {'asym Δ':>8} {'reduction':>10} "
      f"{'sym INT4 Δ':>11} {'pct_base':>9}")
print(f"  {'-'*22} {'-'*7} {'-'*7} {'-'*8} {'-'*8} {'-'*10} {'-'*11} {'-'*9}")
for r in all_results:
    pct_base = r["asym_int3_delta"] / r["baseline_ppl"] * 100
    print(f"  {r['label']:<22} {r['mode']:<7} {r['baseline_ppl']:>7.3f} "
          f"+{r['sym_int3_delta']:>6.4f} +{r['asym_int3_delta']:>6.4f} "
          f"{r['asym_reduction_pct']:>+8.1f}%  "
          f"+{r['sym_int4_delta']:>8.4f}  {pct_base:>7.2f}%")

asym_wins = sum(1 for r in all_results if r["asym_int3_delta"] < r["sym_int3_delta"])
mean_red  = sum(r["asym_reduction_pct"] for r in all_results) / len(all_results)
min_red   = min(r["asym_reduction_pct"] for r in all_results)
max_red   = max(r["asym_reduction_pct"] for r in all_results)
asym_prod = sum(1 for r in all_results
                if r["asym_int3_delta"] / r["baseline_ppl"] < 0.05)

print(f"\n  asym beats sym:                     {asym_wins}/{len(all_results)} models")
print(f"  reduction:   mean {mean_red:+6.1f}%   min {min_red:+6.1f}%   max {max_red:+6.1f}%")
print(f"  asym INT3 under 5% of baseline:     {asym_prod}/{len(all_results)} models")

out_path = "/content/phase17_2026_flagships.json"
combined = {
    "date":         datetime.now().isoformat(timespec='seconds'),
    "prior_6_model": PRIOR,
    "phase17_runs": {
        "Qwen3.5-9B":  qwen35_res,
        "gemma-4-E4B": gemma4_res,
    },
    "eight_model_summary": all_results,
}
with open(out_path, "w") as f:
    json.dump(combined, f, indent=2, default=str)
print(f"\nSaved -> {out_path}")

print("\n" + "=" * 92)
print("VERDICT")
print("=" * 92)
if asym_wins == len(all_results) and min_red > 0:
    print(f"  -> v0.3 asym INT3 generalizes UNIVERSALLY: "
          f"{asym_wins}/{len(all_results)} models, 2023-2026 coverage.")
    print(f"     Ship v0.3.3 with 'validated on 2026 flagships' claim.")
elif asym_wins >= len(all_results) - 1:
    print(f"  -> asym INT3 wins on {asym_wins}/{len(all_results)} models.")
    print(f"     Investigate the outlier and decide whether to ship.")
else:
    print(f"  -> MIXED: asym wins on {asym_wins}/{len(all_results)}.")
    print(f"     Add a caveat to the v0.3.3 README table.")
