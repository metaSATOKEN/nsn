# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Phase 16: Asymmetric INT3 generalization on Phase 8's 5 models
# ============================================================
# Phase 14 measured asymmetric INT3 vs the symmetric baseline on
# two canonical models:
#   Qwen2-7B       (Mode 1): ΔPPL +0.40 -> +0.10  (-74.3%)
#   Llama-3.1-8B   (Mode 2): ΔPPL +0.25 -> +0.11  (-58.3%)
#
# This phase extends the comparison to the five additional models
# from Phase 8, covering different architectures, head_dim values,
# and baseline PPL regimes. Combined with the Phase 14 pair, this
# gives a 7-model generalization check for the v0.3 claim.
#
# Models:
#   Pythia-6.9B       (Mode 2, base model, noisy regime, head_dim=128)
#   Qwen2.5-14B       (Mode 2, post-Qwen2 recipe, low baseline)
#   Mistral-7B-v0.1   (Mode 2, classic baseline)
#   GPT-J-6B          (Mode 2, 2021 arch, head_dim=256)
#   Phi-3-medium-4k   (Mode 2, Microsoft recipe, low baseline)
#
# Methods (same 4 per model; ctx=1024, sliding-window 10 windows):
#   baseline FP16
#   nsep+pchan INT3 symmetric   (v0.2 reference)
#   nsep+pchan INT3 asymmetric  (v0.3 hero)
#   nsep+pchan INT4 symmetric   (1-bit-higher reference)
#
# Expected outcome (if v0.3 generalizes):
#   asym INT3 cuts sym INT3 ΔPPL by 30-80% on all 5 models,
#   never regresses, and on low-baseline models (Qwen2.5-14B,
#   Phi-3-medium) may bring ΔPPL under 5% of baseline.
#
# Runtime: ~60-90 min on Colab A100 (5 models × 4 methods × 10
# windows + model loads).
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
# === CELL V2 === Quantizers (symmetric + asymmetric, inline)
# =============================================================

def qa_nsep_pchan_sym(x, bits):
    """Symmetric per-channel absmax (v0.2 baseline)."""
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
    """Asymmetric per-channel (min + range) -- v0.3 hero."""
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
# === CELL V3 === KV cache compression + PPL evaluation
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
                  use_safetensors=True, trust_remote_code=True)
    if extra_kwargs:
        kwargs.update(extra_kwargs)

    print(f"\n{'='*80}")
    print(f"Loading {label} ({model_id}) ...")
    print(f"{'='*80}")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    mdl.eval()
    print(f"VRAM used: {torch.cuda.memory_allocated()/1e9:.1f} GB")

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

    # Claim check: asym INT3 cuts sym INT3 ΔPPL
    sym  = results["nsep+pchan INT3 symmetric"]["delta"]
    asym = results["nsep+pchan INT3 asymmetric"]["delta"]
    int4 = results["nsep+pchan INT4 symmetric"]["delta"]
    reduction_pct = (1 - asym / max(sym, 1e-9)) * 100 if sym > 0 else 0

    print(f"\n  sym INT3  -> asym INT3: {sym:.4f} -> {asym:.4f} "
          f"({reduction_pct:+.1f}% reduction)")
    print(f"  sym INT4 reference: {int4:.4f}")

    return {
        "label":        label,
        "model":        model_id,
        "baseline_ppl": baseline_ppl,
        "results":      results,
        "sym_int3_delta":  sym,
        "asym_int3_delta": asym,
        "sym_int4_delta":  int4,
        "asym_reduction_pct": reduction_pct,
    }


# =============================================================
# === CELL V4 === Pythia-6.9B (Mode 2, base, noisy)
# =============================================================

pythia_res = run_model("EleutherAI/pythia-6.9b", "Pythia-6.9B")


# =============================================================
# === CELL V5 === Qwen2.5-14B (Mode 2, post-Qwen2 recipe)
# =============================================================

qwen25_res = run_model("Qwen/Qwen2.5-14B", "Qwen2.5-14B")


# =============================================================
# === CELL V6 === Mistral-7B-v0.1 (Mode 2, classic)
# =============================================================

mistral_res = run_model("mistralai/Mistral-7B-v0.1", "Mistral-7B-v0.1")


# =============================================================
# === CELL V7 === GPT-J-6B (Mode 2, head_dim=256)
# =============================================================

gptj_res = run_model("EleutherAI/gpt-j-6b", "GPT-J-6B")


# =============================================================
# === CELL V8 === Phi-3-medium-4k-instruct (Mode 2)
# =============================================================

phi3_res = run_model("microsoft/Phi-3-medium-4k-instruct",
                     "Phi-3-medium-4k-instruct")


# =============================================================
# === CELL V9 === 7-model synthesis (Phase 14 + Phase 16 combined)
# =============================================================

# Phase 14 priors (from results/phase14_asym_wikitext2_ppl.json)
PRIOR = [
    {"label": "Qwen2-7B",        "mode": "Mode 1",
     "baseline_ppl": 8.3693,
     "sym_int3_delta": 0.4021, "asym_int3_delta": 0.1034,
     "sym_int4_delta": 0.0544,
     "asym_reduction_pct": 74.3},
    {"label": "Llama-3.1-8B",    "mode": "Mode 2",
     "baseline_ppl": 7.3840,
     "sym_int3_delta": 0.2541, "asym_int3_delta": 0.1059,
     "sym_int4_delta": 0.0133,
     "asym_reduction_pct": 58.3},
]

all_results = list(PRIOR)
for res in [pythia_res, qwen25_res, mistral_res, gptj_res, phi3_res]:
    all_results.append({
        "label":              res["label"],
        "mode":               "Mode 2",
        "baseline_ppl":       res["baseline_ppl"],
        "sym_int3_delta":     res["sym_int3_delta"],
        "asym_int3_delta":    res["asym_int3_delta"],
        "sym_int4_delta":     res["sym_int4_delta"],
        "asym_reduction_pct": res["asym_reduction_pct"],
    })

print("\n" + "=" * 88)
print(f"Phase 16 + 14 synthesis: asym INT3 generalization across 7 models "
      f"({datetime.now().isoformat(timespec='seconds')})")
print("=" * 88)
print(f"  {'Model':<22} {'mode':<7} {'base':>7} "
      f"{'sym Δ':>8} {'asym Δ':>8} {'reduction':>10} "
      f"{'sym INT4 Δ':>11}")
print(f"  {'-'*22} {'-'*7} {'-'*7} {'-'*8} {'-'*8} {'-'*10} {'-'*11}")
for r in all_results:
    print(f"  {r['label']:<22} {r['mode']:<7} {r['baseline_ppl']:>7.3f} "
          f"+{r['sym_int3_delta']:>6.4f} +{r['asym_int3_delta']:>6.4f} "
          f"{r['asym_reduction_pct']:>+8.1f}%  "
          f"+{r['sym_int4_delta']:>8.4f}")

# Aggregate
asym_wins = sum(1 for r in all_results if r["asym_int3_delta"] < r["sym_int3_delta"])
mean_red = sum(r["asym_reduction_pct"] for r in all_results) / len(all_results)
min_red  = min(r["asym_reduction_pct"] for r in all_results)
max_red  = max(r["asym_reduction_pct"] for r in all_results)

# "Production" (< 5% of baseline at asym INT3) count
asym_prod = sum(
    1 for r in all_results
    if r["asym_int3_delta"] / r["baseline_ppl"] < 0.05
)

print(f"\n  asym beats sym: {asym_wins}/{len(all_results)} models")
print(f"  reduction: mean {mean_red:+.1f}%  min {min_red:+.1f}%  max {max_red:+.1f}%")
print(f"  asym INT3 under 5% of baseline: {asym_prod}/{len(all_results)} models")

# Save
out_path = "/content/phase16_asym_seven_model_sweep.json"
combined = {
    "date":          datetime.now().isoformat(timespec='seconds'),
    "prior_phase14": PRIOR,
    "phase16_runs":  {
        "Pythia-6.9B":              pythia_res,
        "Qwen2.5-14B":              qwen25_res,
        "Mistral-7B-v0.1":          mistral_res,
        "GPT-J-6B":                 gptj_res,
        "Phi-3-medium-4k-instruct": phi3_res,
    },
    "seven_model_summary": all_results,
}
with open(out_path, "w") as f:
    json.dump(combined, f, indent=2, default=str)
print(f"\nSaved -> {out_path}")

# Final verdict
print("\n" + "=" * 88)
print("VERDICT")
print("=" * 88)
if asym_wins == len(all_results) and min_red > 0:
    print(f"  -> v0.3 asym INT3 GENERALIZES UNIVERSALLY on our 7-model set.")
    print(f"     Release v0.3.2 with '7-model validated' claim.")
elif asym_wins >= len(all_results) - 1:
    print(f"  -> v0.3 asym INT3 generalizes to {asym_wins}/{len(all_results)} models.")
    print(f"     Investigate the outlier case before claiming universality.")
else:
    print(f"  -> MIXED: asym wins on {asym_wins}/{len(all_results)}.")
    print(f"     The generalization claim needs a caveat table.")
