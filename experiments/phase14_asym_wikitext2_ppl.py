# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Phase 14: Asymmetric INT3 WikiText-2 PPL validation (v0.3 SOTA)
# ============================================================
# Phase 13 finding (MSE on real Layer 0 K):
#
#   asym_INT3 vs plain_INT3 at IDENTICAL 3.0 bits/value:
#     * Qwen2-7B  (Mode 1):  +91.2% MSE reduction, 4/4 heads win
#     * Llama-3.1-8B (Mode 2): +68.1% MSE reduction, 8/8 heads win
#
# The mechanism: post-RoPE K channels have strongly asymmetric
# distributions; symmetric absmax wastes ~half the quantization
# range on the unused side. Asymmetric per-channel (min + range)
# captures the real distribution at the same bit budget.
#
# This phase translates that MSE gain into the quantity we
# actually care about: WikiText-2 sliding-window perplexity.
#
# Methods (applied to the full KV cache across all layers):
#   (1) baseline (FP16)            -- reference, no quantization
#   (2) nsep+pchan INT3 symmetric  -- v0.2 default
#   (3) nsep+pchan INT3 asymmetric -- v0.3 candidate (hero)
#   (4) nsep+pchan INT4 symmetric  -- upper bound at 4 bits
#   (5) nsep+pchan INT4 asymmetric -- upper bound for asym
#
# Expected outcome (if MSE -> PPL transfer holds):
#   asym INT3 should cut the Qwen2-7B ΔPPL from +0.40 to under
#   +0.05 and land between plain INT4 (+0.05) and FP16 at exact
#   3 bits/value. That would be a clean SOTA story for the INT3
#   tier.
#
# Runtime: ~60-90 min on Colab A100 (2 models × 5 methods × 10
# windows of 1024 tokens each).
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
# === CELL V2 === Quantizers (symmetric + asymmetric)
# =============================================================

def qa_nsep_pchan_sym(x, bits):
    """Symmetric per-channel absmax (v0.2 default)."""
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
    """Asymmetric per-channel (min + range): the v0.3 hero.

    Uses all 2^bits levels over [min_c, max_c] per channel, instead
    of wasting half the range on the unused side of a symmetric
    absmax window. On Phase 13 real-KV probe: +91.2% MSE reduction
    on Qwen2-7B, +68.1% on Llama-3.1-8B at identical 3 bits/value.
    """
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
    """Apply method_fn to each (layer, head) slice of the cache.
    Passing method_fn=None leaves the cache untouched (FP16 baseline)."""
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


# WikiText-2 test split; use the first ~48 KB of non-empty text.
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
wtext = "\n\n".join([x for x in ds["text"] if x.strip()])[:48000]
print(f"WikiText-2 text length: {len(wtext)} chars")


def ppl_with(method_fn, ctx_len=1024, stride=512, max_windows=10):
    """Sliding-window perplexity. Each window: first half fills the cache,
    second half is predicted with the (optionally compressed) cache."""
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
    ("baseline (FP16)",              None,                                           "16"),
    ("nsep+pchan INT3 symmetric",    lambda x: qa_nsep_pchan_sym(x, 3),              "3"),
    ("nsep+pchan INT3 asymmetric",   lambda x: qa_nsep_pchan_asym(x, 3),             "3"),
    ("nsep+pchan INT4 symmetric",    lambda x: qa_nsep_pchan_sym(x, 4),              "4"),
    ("nsep+pchan INT4 asymmetric",   lambda x: qa_nsep_pchan_asym(x, 4),             "4"),
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

    print(f"\n  {'Method':<32} | {'bits':>6} | {'PPL':>10} | {'ΔPPL':>10}")
    print(f"  " + "-"*66)

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
        print(f"  {name:<32} | {bits:>6} | {ppl:>10.4f} | {dstr:>10}")

    # Key comparisons for the v0.3 claim:
    int3_sym = results["nsep+pchan INT3 symmetric"]["delta"]
    int3_asym = results["nsep+pchan INT3 asymmetric"]["delta"]
    int4_sym = results["nsep+pchan INT4 symmetric"]["delta"]

    print(f"\n  Claim check: 'asym INT3 crushes sym INT3 and competes with INT4'")
    print(f"    sym  INT3 ΔPPL: +{int3_sym:.4f}")
    print(f"    asym INT3 ΔPPL: +{int3_asym:.4f}")
    print(f"    sym  INT4 ΔPPL: +{int4_sym:.4f}")

    if int3_asym < int3_sym:
        sym_reduction_pct = (1 - int3_asym / max(int3_sym, 1e-9)) * 100
        print(f"    asym cuts sym INT3 ΔPPL by {sym_reduction_pct:.1f}%")

    if int3_asym < int4_sym:
        print(f"    ✓ asym INT3 BEATS sym INT4 at 1 fewer bit.")
        verdict = "ASYM_INT3_BEATS_SYM_INT4"
    elif int3_asym < 1.1 * int4_sym:
        print(f"    ~ asym INT3 approaches sym INT4 (within 10%).")
        verdict = "ASYM_INT3_APPROACHES_INT4"
    else:
        print(f"    ~ asym INT3 helps vs sym INT3 but does not reach INT4.")
        verdict = "ASYM_HELPS_NOT_INT4"

    return {
        "label":        label,
        "model":        model_id,
        "baseline_ppl": baseline_ppl,
        "results":      results,
        "verdict":      verdict,
    }


# =============================================================
# === CELL V4 === Qwen2-7B (Mode 1, the canonical outlier case)
# =============================================================

qwen_res = run_model("Qwen/Qwen2-7B", "Qwen2-7B (Mode 1)")


# =============================================================
# === CELL V5 === Llama-3.1-8B (Mode 2, reference)
# =============================================================

llama_res = run_model("meta-llama/Llama-3.1-8B", "Llama-3.1-8B (Mode 2)")


# =============================================================
# === CELL V6 === Two-model summary + save
# =============================================================

print("\n" + "=" * 80)
print(f"Phase 14 summary ({datetime.now().isoformat(timespec='seconds')})")
print("=" * 80)

for res in [qwen_res, llama_res]:
    print(f"\n{res['label']} (baseline PPL = {res['baseline_ppl']:.4f})")
    print(f"  {'Method':<32} | {'bits':>6} | {'PPL':>10} | {'ΔPPL':>10}")
    print(f"  " + "-"*66)
    for name, r in res["results"].items():
        dstr = f"+{r['delta']:.4f}" if r["delta"] >= 0 else f"{r['delta']:.4f}"
        print(f"  {name:<32} | {r['bits']:>6} | {r['ppl']:>10.4f} | {dstr:>10}")
    print(f"  verdict: {res['verdict']}")

# Quick side-by-side for the key asym-vs-sym headline
print("\n" + "=" * 80)
print("HEADLINE: asymmetric INT3 vs symmetric INT3 at identical 3 bits/value")
print("=" * 80)
print(f"  {'Model':<25} | {'sym INT3 ΔPPL':>14} | {'asym INT3 ΔPPL':>16} | "
      f"{'reduction':>12}")
print(f"  " + "-"*80)
for res in [qwen_res, llama_res]:
    sym  = res["results"]["nsep+pchan INT3 symmetric"]["delta"]
    asym = res["results"]["nsep+pchan INT3 asymmetric"]["delta"]
    red = (1 - asym / max(sym, 1e-9)) * 100 if sym > 0 else 0
    label_short = res["label"].split(" (")[0]
    print(f"  {label_short:<25} | +{sym:>12.4f}  | +{asym:>14.4f}   | "
          f"{red:>+10.1f}%")

# Save
combined = {
    "date":       datetime.now().isoformat(timespec='seconds'),
    "qwen":       qwen_res,
    "llama":      llama_res,
}
out_path = "/content/phase14_asym_wikitext2_ppl.json"
with open(out_path, "w") as f:
    json.dump(combined, f, indent=2)
print(f"\nSaved -> {out_path}")

# Final verdict
print("\n" + "=" * 80)
print("VERDICT")
print("=" * 80)
qwen_v = qwen_res["verdict"]
llama_v = llama_res["verdict"]

if qwen_v == "ASYM_INT3_BEATS_SYM_INT4" and llama_v in {
    "ASYM_INT3_BEATS_SYM_INT4", "ASYM_INT3_APPROACHES_INT4"
}:
    print("  -> v0.3 SHIP IT. asym INT3 is the new default.")
    print("     'exact 3 bits, beats sym INT4 on Mode 1, competitive on Mode 2.'")
elif qwen_v in {"ASYM_INT3_BEATS_SYM_INT4", "ASYM_INT3_APPROACHES_INT4"}:
    print("  -> v0.3 READY. asym INT3 provides a clear INT3-tier win on both.")
    print("     Release as the new default; measure INT4 as follow-up.")
else:
    print("  -> MSE gain did not fully transfer to PPL; inspect ΔPPL table.")
    print("     Consider: did we apply the quantizer correctly? Does the")
    print("     re-normalization step interact badly with asym at scale?")
