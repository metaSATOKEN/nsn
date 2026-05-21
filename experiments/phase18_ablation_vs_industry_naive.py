# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Phase 18: Ablation -- what baseline is v0.3 actually beating?
# ============================================================
# Phases 14/16/17 compared nsn's "symmetric INT3" to nsn's
# "asymmetric INT3" and reported 58-96% ΔPPL reductions. But
# BOTH of those are already nsn's nsep+pchan pipeline with
# different quantization windows -- neither is the naive
# industry baseline a reader might assume when they see
# "symmetric INT3".
#
# This phase runs four INT3 methods side by side on three
# canonical models so we can separate three questions:
#
#   Q1: How much does per-channel absmax buy over naive per-row
#       absmax? (the classical quantization-recipe gain)
#   Q2: Is norm separation (nsep) necessary, or does
#       asymmetric per-channel alone already capture most of
#       the v0.3 gain?
#   Q3: Is v0.3 asymmetric universally best, or does it lose
#       to one of the simpler recipes on any model?
#
# Methods (each applied to the full KV cache):
#   (A) naive  per-row   symmetric  INT3  -- industry baseline
#   (B)        per-ch    asymmetric INT3  -- asym WITHOUT nsep
#   (C) nsep + per-ch    symmetric  INT3  -- nsn v0.2 default
#   (D) nsep + per-ch    asymmetric INT3  -- nsn v0.3 default
#
# Models (chosen for maximum mechanistic diversity within a
# reasonable runtime):
#   Qwen2-7B       (Mode 1, the canonical INT4-valley case;
#                   where nsep+pchan matters most)
#   Llama-3.1-8B   (Mode 2, well-behaved baseline)
#   Pythia-6.9B    (Mode 2 with high noise; where asym
#                   produced -95.8% in Phase 16 and the biggest
#                   absolute sym-INT3 drift among our 8-model set)
#
# Runtime: ~50-70 min on Colab A100 (3 models × 5 runs × 10 windows).
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
# === CELL V2 === Four-method quantizer zoo
# =============================================================

def qa_naive_perrow_sym(x, bits):
    """(A) Industry baseline: per-row symmetric absmax.

    No norm separation, no per-channel scales. The simplest thing
    a practitioner might try. This is what the top-of-README
    'naive INT4 +1382 ΔPPL' table uses.
    """
    dtype = x.dtype
    xf = x.float()
    qm = 2**(bits-1) - 1
    s = xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / qm
    return (((xf / s).round().clamp(-qm, qm)) * s).to(dtype)


def qa_asym_pchan_no_nsep(x, bits):
    """(B) Asymmetric per-channel, NO norm separation.

    This is the direct ablation partner of qa_nsep_pchan_asym:
    same [min, max]-per-channel quantization, but applied to the
    raw x rather than to the unit-direction vector. Tests whether
    norm separation is essential or whether asymmetric per-channel
    already captures most of the gain.
    """
    dtype = x.dtype
    xf = x.float()
    d_min = xf.amin(dim=0, keepdim=True)
    d_max = xf.amax(dim=0, keepdim=True)
    d_range = (d_max - d_min).clamp(min=1e-12)
    levels = 2**bits - 1
    step = d_range / levels
    dq = ((xf - d_min) / step).round().clamp(0, levels) * step + d_min
    return dq.to(dtype)


def qa_nsep_pchan_sym(x, bits):
    """(C) nsn v0.2 default: nsep + per-channel absmax (symmetric)."""
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
    """(D) nsn v0.3 default: nsep + per-channel min/max (asymmetric)."""
    dtype = x.dtype
    xf = x.float()
    n = xf.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    d = xf / n
    d_min = d.amin(dim=0, keepdim=True)
    d_max = d.amax(dim=0, keepdim=True)
    d_range = (d_max - d_min).clamp(min=1e-12)
    levels = 2**bits - 1
    step = d_range / levels
    dq = ((d - d_min) / step).round().clamp(0, levels) * step + d_min
    dq = dq / dq.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return (n * dq).to(dtype)


# =============================================================
# === CELL V3 === KV cache compression + PPL (with hybrid-skip)
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

def _layer_has_kv(past, li):
    if hasattr(past, 'layers'):
        layer = past.layers[li]
        return (hasattr(layer, 'keys') and hasattr(layer, 'values')
                and layer.keys is not None and layer.values is not None)
    return past[li] is not None and past[li][0] is not None

def compress_cache(past, method_fn):
    if method_fn is None:
        return
    for li in range(n_cache_layers(past)):
        if not _layer_has_kv(past, li):
            continue
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
    ("baseline (FP16)",                 None,                                  "16"),
    ("(A) naive per-row sym INT3",      lambda x: qa_naive_perrow_sym(x, 3),   "3"),
    ("(B) asym per-channel, no nsep",   lambda x: qa_asym_pchan_no_nsep(x, 3), "3"),
    ("(C) nsep+pchan sym INT3 (v0.2)",  lambda x: qa_nsep_pchan_sym(x, 3),     "3"),
    ("(D) nsep+pchan asym INT3 (v0.3)", lambda x: qa_nsep_pchan_asym(x, 3),    "3"),
]


def run_model(model_id, label):
    global mdl, tok
    try:
        del mdl, tok
    except NameError:
        pass
    gc.collect(); torch.cuda.empty_cache()

    print(f"\n{'='*80}")
    print(f"Loading {label} ({model_id}) ...")
    print(f"{'='*80}")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto",
        use_safetensors=True, trust_remote_code=True,
    )
    mdl.eval()
    print(f"VRAM used: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    print(f"\n  {'Method':<36} | {'bits':>5} | {'PPL':>10} | {'ΔPPL':>10}")
    print(f"  " + "-"*68)

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
        print(f"  {name:<36} | {bits:>5} | {ppl:>10.4f} | {dstr:>10}")

    # Extract the four ablation deltas.
    naive = results["(A) naive per-row sym INT3"]["delta"]
    asym_only = results["(B) asym per-channel, no nsep"]["delta"]
    sym_full = results["(C) nsep+pchan sym INT3 (v0.2)"]["delta"]
    asym_full = results["(D) nsep+pchan asym INT3 (v0.3)"]["delta"]

    def red(a, b):
        return (1 - b / max(a, 1e-9)) * 100 if a > 0 else 0

    print(f"\n  Ablation deltas:")
    print(f"    v0.3  vs naive (A->D):       "
          f"{naive:+.4f} -> {asym_full:+.4f}   "
          f"({red(naive, asym_full):+6.1f}%, "
          f"{naive/max(asym_full,1e-9):>5.1f}x)")
    print(f"    asym-only vs naive (A->B):   "
          f"{naive:+.4f} -> {asym_only:+.4f}   "
          f"({red(naive, asym_only):+6.1f}%)")
    print(f"    v0.3  vs asym-only (B->D):   "
          f"{asym_only:+.4f} -> {asym_full:+.4f}   "
          f"({red(asym_only, asym_full):+6.1f}%)")
    print(f"    v0.3  vs v0.2   (C->D):      "
          f"{sym_full:+.4f} -> {asym_full:+.4f}   "
          f"({red(sym_full, asym_full):+6.1f}%)")

    return {
        "label":             label,
        "model":             model_id,
        "baseline_ppl":      baseline_ppl,
        "results":           results,
        "delta_naive":       naive,
        "delta_asym_only":   asym_only,
        "delta_sym_full":    sym_full,
        "delta_asym_full":   asym_full,
        "v03_vs_naive_pct":  red(naive,     asym_full),
        "v03_vs_asym_only_pct": red(asym_only, asym_full),
        "v03_vs_v02_pct":    red(sym_full,  asym_full),
    }


# =============================================================
# === CELL V4 === Qwen2-7B (Mode 1, INT4 valley)
# =============================================================

qwen2_res = run_model("Qwen/Qwen2-7B", "Qwen2-7B (Mode 1)")


# =============================================================
# === CELL V5 === Llama-3.1-8B (Mode 2, well-behaved)
# =============================================================

llama_res = run_model("meta-llama/Llama-3.1-8B", "Llama-3.1-8B (Mode 2)")


# =============================================================
# === CELL V6 === Pythia-6.9B (Mode 2, high-noise; biggest
#                               Phase 16 asym win)
# =============================================================

pythia_res = run_model("EleutherAI/pythia-6.9b", "Pythia-6.9B (Mode 2)")


# =============================================================
# === CELL V7 === Ablation table synthesis
# =============================================================

def _pct_base(delta, base):
    return (delta / max(base, 1e-9)) * 100

all_runs = [qwen2_res, llama_res, pythia_res]

print("\n" + "=" * 102)
print(f"Phase 18 ablation  ({datetime.now().isoformat(timespec='seconds')})")
print("=" * 102)
print(f"  {'Model':<22} {'base':>7}  {'(A) naive':>12} "
      f"{'(B) asym-only':>16} {'(C) v0.2 sym':>14} {'(D) v0.3 asym':>15}")
print(f"  {'-'*22} {'-'*7}  {'-'*12} {'-'*16} {'-'*14} {'-'*15}")
for r in all_runs:
    b = r["baseline_ppl"]
    print(f"  {r['label']:<22} {b:>7.3f}  "
          f"+{r['delta_naive']:>10.4f}  "
          f"+{r['delta_asym_only']:>14.4f}  "
          f"+{r['delta_sym_full']:>12.4f}  "
          f"+{r['delta_asym_full']:>13.4f}")

print(f"\n  % of baseline PPL (production threshold < 5.0%):")
print(f"  {'Model':<22} {'(A)':>10} {'(B)':>10} {'(C)':>10} {'(D)':>10}")
print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
for r in all_runs:
    b = r["baseline_ppl"]
    print(f"  {r['label']:<22} "
          f"{_pct_base(r['delta_naive'],     b):>9.2f}% "
          f"{_pct_base(r['delta_asym_only'], b):>9.2f}% "
          f"{_pct_base(r['delta_sym_full'],  b):>9.2f}% "
          f"{_pct_base(r['delta_asym_full'], b):>9.2f}%")

print(f"\n  v0.3 (D) improvement ratios:")
print(f"  {'Model':<22} {'vs naive (A)':>15} {'vs asym-only (B)':>18} {'vs v0.2 sym (C)':>17}")
print(f"  {'-'*22} {'-'*15} {'-'*18} {'-'*17}")
for r in all_runs:
    def ratio(a, b):
        return a / max(b, 1e-9)
    print(f"  {r['label']:<22} "
          f"{ratio(r['delta_naive'],     r['delta_asym_full']):>12.1f}x  "
          f"{ratio(r['delta_asym_only'], r['delta_asym_full']):>15.1f}x  "
          f"{ratio(r['delta_sym_full'],  r['delta_asym_full']):>14.1f}x")

# Save
out_path = "/content/phase18_ablation_vs_industry_naive.json"
combined = {
    "date":         datetime.now().isoformat(timespec='seconds'),
    "runs":         {r["label"]: r for r in all_runs},
    "method_legend": {
        "A": "naive per-row symmetric INT3 (industry baseline)",
        "B": "asymmetric per-channel INT3, no norm separation",
        "C": "nsep+pchan symmetric INT3 (nsn v0.2 default)",
        "D": "nsep+pchan asymmetric INT3 (nsn v0.3 default)",
    },
    "ablation_questions": {
        "Q1_pchan_gain":      "does per-channel absmax help over naive? (A -> C)",
        "Q2_nsep_necessity":  "does norm-sep help on top of asym pchan? (B -> D)",
        "Q3_asym_necessity":  "does asym help on top of nsep+pchan? (C -> D)",
    },
}
with open(out_path, "w") as f:
    json.dump(combined, f, indent=2, default=str)
print(f"\nSaved -> {out_path}")

# Narrative-pointing verdict
print("\n" + "=" * 102)
print("NARRATIVE CHECK")
print("=" * 102)
for r in all_runs:
    print(f"\n  {r['label']}:")
    print(f"    v0.3 beats NAIVE industry baseline by: "
          f"{r['delta_naive']/max(r['delta_asym_full'],1e-9):.1f}x")
    print(f"    Does nsep still help on top of asym pchan (B -> D)? "
          f"{'YES' if r['delta_asym_full'] < r['delta_asym_only'] else 'NO'} "
          f"({r['v03_vs_asym_only_pct']:+.1f}% reduction)")
    print(f"    Does asym still help on top of nsep+pchan (C -> D)? "
          f"{'YES' if r['delta_asym_full'] < r['delta_sym_full'] else 'NO'} "
          f"({r['v03_vs_v02_pct']:+.1f}% reduction)")
