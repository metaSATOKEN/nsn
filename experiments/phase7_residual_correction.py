# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Phase 7: Composition Experiment -- 1-bit Residual Correction
# ============================================================
# Question: can a TurboQuant-style 1-bit residual correction
# rescue nsep+pchan's INT2 floor, and at what effective bit cost?
#
# Hypotheses:
#   (H1) nsep+pchan INT2 alone is catastrophic on Mode 1 (+326
#        ΔPPL on Qwen2-7B, +10 on Llama-3.1-8B per Phase 1/3).
#        Adding a 1-bit sign-based residual should rescue both.
#   (H2) At "3 bits effective" (INT2 base + 1-bit residual),
#        quality should approach nsep+pchan INT3 (3 bits alone).
#        If yes, we have a composable extension of nsep+pchan
#        that reaches TurboQuant's 6x compression territory
#        (minus the residual's sign storage).
#   (H3) On mild Mode 2 models (Llama-3.1-8B), residual
#        correction on INT3 should also reduce ΔPPL further.
#
# Storage accounting:
#   baseline INT3         : 3 bits/value                 -> 5.33x
#   INT2 + 1-bit residual : 3 bits/value (same as INT3)  -> 5.33x
#   INT3 + 1-bit residual : 4 bits/value (same as INT4)  -> 4x
#
# Residual quantization: sign(residual) * 0.25
#   rationale: for uniform residuals on [-0.5, +0.5], 0.25
#   is the RMS/mean-absolute value and the optimal 1-bit
#   point estimate. More sophisticated (per-channel RMS, etc.)
#   is possible but we start with this simple choice.
#
# Runtime: ~30 min (2 models × ~6 methods × 2 min each + loads)
# ============================================================


# =============================================================
# === CELL C1 === Setup + residual-correction quantizer
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


# ── Baseline nsep+pchan (no residual) ────────────────────────
def qa_nsep_pchan(x, bits):
    dtype = x.dtype
    xf = x.float()
    n = xf.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    d = xf / n
    qm = 2**(bits-1) - 1
    s = d.abs().amax(dim=0, keepdim=True).clamp(min=1e-12) / qm
    dq = ((d / s).round().clamp(-qm, qm)) * s
    dq = dq / dq.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return (n * dq).to(dtype)


# ── nsep+pchan + 1-bit residual correction (simple) ──────────
def qa_nsep_pchan_resid_simple(x, bits_base):
    """
    Adds a 1-bit sign-based residual correction with a fixed
    magnitude of 0.25 (the optimal constant for uniform
    residuals on [-0.5, 0.5]).
    Storage cost: +1 bit per value.
    """
    dtype = x.dtype
    xf = x.float()
    n = xf.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    d = xf / n

    qm = 2**(bits_base-1) - 1
    s = d.abs().amax(dim=0, keepdim=True).clamp(min=1e-12) / qm
    d_scaled = d / s
    d_q = d_scaled.round().clamp(-qm, qm)

    # 1-bit residual: sign of the unclamped residual, fixed magnitude 0.25
    residual = d_scaled - d_q
    r_1bit = 0.25 * residual.sign()

    d_corrected = (d_q + r_1bit) * s
    d_corrected = d_corrected / d_corrected.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return (n * d_corrected).to(dtype)


# ── nsep+pchan + 1-bit residual correction (per-channel RMS) ─
def qa_nsep_pchan_resid_rms(x, bits_base):
    """
    More adaptive variant: per-channel RMS of residuals as the
    1-bit magnitude. Storage cost: +1 bit per value + D FP16
    scalars per head (negligible).
    """
    dtype = x.dtype
    xf = x.float()
    n = xf.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    d = xf / n

    qm = 2**(bits_base-1) - 1
    s = d.abs().amax(dim=0, keepdim=True).clamp(min=1e-12) / qm
    d_scaled = d / s
    d_q = d_scaled.round().clamp(-qm, qm)

    residual = d_scaled - d_q
    # Per-channel RMS
    r_rms = (residual ** 2).mean(dim=0, keepdim=True).sqrt().clamp(min=1e-12)
    r_1bit = r_rms * residual.sign()

    d_corrected = (d_q + r_1bit) * s
    d_corrected = d_corrected / d_corrected.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return (n * d_corrected).to(dtype)


# ── KV cache helpers ────────────────────────────────────────
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
    """method_fn: callable taking a tensor, returning compressed tensor of same shape/dtype."""
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


# ── WikiText-2 PPL ──────────────────────────────────────────
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
    ("baseline (FP16)",              None,                                           "16 bits"),
    ("nsep+pchan INT2",              lambda x: qa_nsep_pchan(x, 2),                  "2 bits"),
    ("nsep+pchan INT2 + resid-0.25", lambda x: qa_nsep_pchan_resid_simple(x, 2),     "2+1 = 3 bits"),
    ("nsep+pchan INT2 + resid-RMS",  lambda x: qa_nsep_pchan_resid_rms(x, 2),        "2+1 = 3 bits*"),
    ("nsep+pchan INT3",              lambda x: qa_nsep_pchan(x, 3),                  "3 bits"),
    ("nsep+pchan INT3 + resid-0.25", lambda x: qa_nsep_pchan_resid_simple(x, 3),     "3+1 = 4 bits"),
    ("nsep+pchan INT3 + resid-RMS",  lambda x: qa_nsep_pchan_resid_rms(x, 3),        "3+1 = 4 bits*"),
    ("nsep+pchan INT4",              lambda x: qa_nsep_pchan(x, 4),                  "4 bits"),
]


def run_all(model_id, label):
    global mdl, tok
    try:
        del mdl, tok
    except NameError:
        pass
    gc.collect(); torch.cuda.empty_cache()

    print(f"\n{'='*80}")
    print(f"Loading {label} ({model_id}) ...")
    print(f"{'='*80}")
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto",
        use_safetensors=True,
    )
    mdl.eval()
    print(f"VRAM used: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    print(f"\n  {'Method':<32} | {'bits':<14} | {'PPL':>10} | {'ΔPPL':>10}")
    print(f"  " + "-"*75)

    results = {}
    baseline_ppl = None
    for name, fn, bits_desc in METHODS:
        print(f"  Running {name} ...", flush=True)
        ppl, nt = ppl_with(fn)
        if baseline_ppl is None:
            baseline_ppl = ppl
        delta = ppl - baseline_ppl
        results[name] = {"ppl": ppl, "bits_desc": bits_desc, "delta": delta, "tokens": nt}
        dstr = f"+{delta:.3f}" if delta >= 0 else f"{delta:.3f}"
        print(f"  {name:<32} | {bits_desc:<14} | {ppl:>10.3f} | {dstr:>10}")

    return {"label": label, "model": model_id, "baseline_ppl": baseline_ppl, "results": results}


# =============================================================
# === CELL C2 === Qwen2-7B (Mode 1)
# =============================================================

qwen_res = run_all("Qwen/Qwen2-7B", "Qwen2-7B (Mode 1)")


# =============================================================
# === CELL C3 === Llama-3.1-8B-Instruct (Mode 2)
# =============================================================

llama_res = run_all("meta-llama/Llama-3.1-8B-Instruct", "Llama-3.1-8B (Mode 2)")


# =============================================================
# === CELL C4 === Composition analysis + verdict
# =============================================================

print(f"\n\n{'='*80}")
print(f"COMPOSITION ANALYSIS: 1-bit Residual Correction")
print(f"{'='*80}")

for res in [qwen_res, llama_res]:
    label = res["label"]
    baseline = res["baseline_ppl"]
    r = res["results"]

    int2_alone   = r["nsep+pchan INT2"]["delta"]
    int2_r025    = r["nsep+pchan INT2 + resid-0.25"]["delta"]
    int2_rms     = r["nsep+pchan INT2 + resid-RMS"]["delta"]
    int3_alone   = r["nsep+pchan INT3"]["delta"]
    int3_r025    = r["nsep+pchan INT3 + resid-0.25"]["delta"]
    int3_rms     = r["nsep+pchan INT3 + resid-RMS"]["delta"]
    int4_alone   = r["nsep+pchan INT4"]["delta"]

    print(f"\n  {label} (baseline PPL = {baseline:.3f}):")
    print(f"    INT2 alone          : +{int2_alone:>8.3f}")
    print(f"    INT2 + residual-0.25: +{int2_r025:>8.3f}  [3 eff bits, expect INT3 quality]")
    print(f"    INT2 + residual-RMS : +{int2_rms:>8.3f}  [3+D eff bits, expect best INT3]")
    print(f"    INT3 alone          : +{int3_alone:>8.3f}  [reference for 3 bits]")
    print(f"    INT3 + residual-0.25: +{int3_r025:>8.3f}  [4 eff bits, expect INT4 quality]")
    print(f"    INT3 + residual-RMS : +{int3_rms:>8.3f}  [4+D eff bits, expect best INT4]")
    print(f"    INT4 alone          : +{int4_alone:>8.3f}  [reference for 4 bits]")

    # Rescue ratio for INT2 case
    if int2_alone > 1:
        rescue_025 = 1 - (int2_r025 / int2_alone)
        rescue_rms = 1 - (int2_rms / int2_alone)
        print(f"\n    Residual rescue of INT2:")
        print(f"      0.25 const: {rescue_025*100:>6.1f}% damage removed")
        print(f"      RMS per-ch: {rescue_rms*100:>6.1f}% damage removed")

    # Competitiveness vs INT3 alone at 3-bit cost
    if int3_alone > 0.01:
        ratio_025 = int2_r025 / int3_alone
        ratio_rms = int2_rms / int3_alone
        print(f"\n    3-bit comparison (INT2+resid vs INT3 alone):")
        print(f"      0.25 const / INT3: {ratio_025:>6.2f}x baseline  " +
              ("(BEATS INT3!)" if ratio_025 < 0.9 else
               "(matches INT3)" if ratio_025 < 1.1 else
               "(loses to INT3)"))
        print(f"      RMS per-ch / INT3: {ratio_rms:>6.2f}x baseline  " +
              ("(BEATS INT3!)" if ratio_rms < 0.9 else
               "(matches INT3)" if ratio_rms < 1.1 else
               "(loses to INT3)"))

# Save
import json
final = {
    "experiment": "phase7_residual_correction",
    "timestamp": datetime.now().isoformat(),
    "qwen2_7b": qwen_res,
    "llama_3_1_8b": llama_res,
}
print(f"\n{'='*80}\nFINAL JSON\n{'='*80}")
print(json.dumps(final, indent=2, default=str))
