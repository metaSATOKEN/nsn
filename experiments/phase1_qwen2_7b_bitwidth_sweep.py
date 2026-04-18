# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Phase 1: Bit-Width Sweep on Qwen2-7B (Mode 1 pathological)
# ============================================================
# Hypothesis: naive INT quantization shows a U-SHAPED ΔPPL curve
# vs bit-width on Mode 1 (Layer-0 outlier-driven) models. Low
# precision zeroes normal values cleanly; high precision gives
# accurate normal values; the middle (INT4-INT5) is the worst
# case because normal values quantize to noisy non-zero levels
# under outlier-dominated per-row scales.
#
# Prediction:
#   naive:       U-shape, worst around INT4-INT5
#   nsep+pchan:  monotonic (bigger bits = better, smoothly)
#
# Protocol: WikiText-2 sliding-window PPL
# (ctx=1024, stride=512, 10 windows, 5120 target tokens)
#
# Runtime: ~30-40 min (12-14 PPL runs on Qwen2-7B at ~2 min each)
# ============================================================


# =============================================================
# === CELL B1 === Setup + load Qwen2-7B
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
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2-7B"

print(f"Loading {MODEL_ID} ...")
tok = AutoTokenizer.from_pretrained(MODEL_ID)
mdl = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, device_map="auto",
    use_safetensors=True,
)
mdl.eval()
print(f"VRAM used: {torch.cuda.memory_allocated()/1e9:.1f} GB")


# =============================================================
# === CELL B2 === Bit-width-parameterized quantization
# =============================================================

def qa_perrow(x, bits):
    """Naive absmax per-row quantization at arbitrary bit width."""
    x = x.float()
    qm = 2**(bits-1) - 1  # INT8: 127, INT4: 7, INT3: 3, INT2: 1
    s = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / qm
    return ((x/s).round().clamp(-qm, qm)) * s

def qa_perchan(x, bits):
    """Per-channel absmax quantization at arbitrary bit width."""
    x = x.float()
    qm = 2**(bits-1) - 1
    s = x.abs().amax(dim=0, keepdim=True).clamp(min=1e-12) / qm
    return ((x/s).round().clamp(-qm, qm)) * s

def qa_nsep_pchan(x, bits):
    """nsep+pchan at arbitrary bit width."""
    x = x.float()
    n = x.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    d = x / n
    dq = qa_perchan(d, bits)
    dq = dq / dq.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return n * dq

def apply_method(x, method, bits):
    if method == "baseline" or method == "fp16":
        return x
    if method == "naive":
        return qa_perrow(x, bits)
    if method == "nsep+pchan":
        return qa_nsep_pchan(x, bits)
    raise ValueError(method)

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

def compress_cache_uniform(past, method, bits):
    """Apply (method, bits) to every layer."""
    if method == "baseline":
        return
    for li in range(n_cache_layers(past)):
        ok, ov = get_kv(past, li)
        ok = ok.clone(); ov = ov.clone()
        nk = torch.zeros_like(ok); nv = torch.zeros_like(ov)
        for h in range(ok.shape[1]):
            nk[0, h] = apply_method(ok[0, h], method, bits).to(ok.dtype)
            nv[0, h] = apply_method(ov[0, h], method, bits).to(ov.dtype)
        set_kv(past, li, nk, nv)


# =============================================================
# === CELL B3 === WikiText-2 PPL evaluation
# =============================================================

from datasets import load_dataset

ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
wtext = "\n\n".join([x for x in ds["text"] if x.strip()])[:48000]
print(f"WikiText-2 text length: {len(wtext)} chars")


def ppl_with(method, bits, ctx_len=1024, stride=512, max_windows=10):
    """Batched teacher-forced PPL on WikiText-2."""
    ids = tok.encode(wtext, return_tensors="pt").to("cuda")
    T = ids.shape[1]
    total_nll = 0.0
    total_tokens = 0
    windows_done = 0

    with torch.inference_mode():
        for begin in range(0, T - ctx_len, stride):
            if windows_done >= max_windows:
                break
            window = ids[:, begin:begin + ctx_len]
            split = ctx_len // 2
            context = window[:, :split]
            targets = window[:, split:]

            out_ctx = mdl(context, use_cache=True)
            past = out_ctx.past_key_values
            first_logits = out_ctx.logits[:, -1:, :]
            compress_cache_uniform(past, method, bits)
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
    ppl = float(np.exp(mean_nll))
    return ppl, total_tokens


# =============================================================
# === CELL B4 === Bit-width sweep
# =============================================================

BITS = [2, 3, 4, 5, 6, 8]
METHODS = ["naive", "nsep+pchan"]

print(f"\n{'='*80}")
print(f"Qwen2-7B: Bit-Width Sweep on WikiText-2")
print(f"{'='*80}\n")

# Baseline first
print("Running baseline (FP16) ...", flush=True)
baseline_ppl, nt = ppl_with("baseline", 0)
print(f"  baseline (FP16)  PPL = {baseline_ppl:.3f}  ({nt} tokens)\n")

# Main sweep
print(f"  {'Method':>14} | {'bits':>4} | {'PPL':>10} | {'ΔPPL':>10}")
print(f"  {'-'*46}")

results = {"baseline": {"ppl": baseline_ppl, "tokens": nt}}
for method in METHODS:
    results[method] = {}
    for bits in BITS:
        print(f"  Running {method} INT{bits} ...", flush=True)
        ppl, nt = ppl_with(method, bits)
        dppl = ppl - baseline_ppl
        results[method][f"int{bits}"] = {
            "ppl": ppl, "delta": dppl, "tokens": nt
        }
        print(f"  {method:>14} | INT{bits:<2} | {ppl:>10.3f} | {'+' if dppl>=0 else ''}{dppl:>9.3f}")


# =============================================================
# === CELL B5 === Analysis + U-shape verdict
# =============================================================

print(f"\n{'='*80}")
print(f"U-SHAPE HYPOTHESIS VERDICT")
print(f"{'='*80}")

print(f"\n  Bit-width table ({MODEL_ID}, baseline PPL={baseline_ppl:.2f}):")
print(f"  {'':>10} | {'INT2':>8} | {'INT3':>8} | {'INT4':>8} | {'INT5':>8} | {'INT6':>8} | {'INT8':>8}")
print(f"  {'-'*10} + {'-'*8} + {'-'*8} + {'-'*8} + {'-'*8} + {'-'*8} + {'-'*8}")
for method in METHODS:
    row = [f"+{results[method][f'int{b}']['delta']:.2f}" for b in BITS]
    print(f"  {method:>10} | " + " | ".join(f"{r:>8}" for r in row))

# U-shape detection for naive
naive_deltas = [results["naive"][f"int{b}"]["delta"] for b in BITS]
naive_argmax = BITS[int(np.argmax(naive_deltas))]
is_u_shape = (
    naive_deltas[0] < naive_deltas[2] and  # INT2 < INT4
    naive_deltas[-1] < naive_deltas[2]     # INT8 < INT4
)
print(f"\n  Naive ΔPPL by bit width:")
for b, d in zip(BITS, naive_deltas):
    marker = "  ← WORST" if b == naive_argmax else ""
    print(f"    INT{b}: +{d:.2f}{marker}")

if is_u_shape:
    print(f"\n  🎯 U-SHAPE CONFIRMED for naive quantization on Qwen2-7B")
    print(f"  >>> INT2 (+{naive_deltas[0]:.2f}) < INT4 (+{naive_deltas[2]:.2f}) > INT8 (+{naive_deltas[-1]:.2f})")
    print(f"  >>> Worst at INT{naive_argmax} -- confirms 'noisy non-zero middle precision' hypothesis")
    verdict = "U_SHAPE_CONFIRMED"
else:
    print(f"\n  ✗ No clear U-shape detected")
    print(f"  >>> Monotonic or otherwise; mechanism hypothesis needs revision")
    verdict = "NO_U_SHAPE"

# nsep+pchan monotonicity
nsep_deltas = [results["nsep+pchan"][f"int{b}"]["delta"] for b in BITS]
is_monotonic = all(nsep_deltas[i] >= nsep_deltas[i+1] - 0.5
                   for i in range(len(nsep_deltas)-1))
print(f"\n  nsep+pchan ΔPPL by bit width:")
for b, d in zip(BITS, nsep_deltas):
    print(f"    INT{b}: +{d:.2f}")
print(f"  {'✓ Monotonic' if is_monotonic else '~ Mixed'}: nsep+pchan " +
      ("improves smoothly with bits" if is_monotonic else "shows non-monotonic behavior"))

# Separation factor
if max(naive_deltas) > 1:
    print(f"\n  Improvement factor at each bit width (naive / nsep+pchan):")
    for b in BITS:
        n = results["naive"][f"int{b}"]["delta"]
        s = max(results["nsep+pchan"][f"int{b}"]["delta"], 0.01)
        print(f"    INT{b}: {n/s:>8.1f}x")

# Save
final = {
    "experiment": "phase1_qwen2_7b_bitwidth_sweep",
    "model": MODEL_ID,
    "timestamp": datetime.now().isoformat(),
    "baseline_ppl": baseline_ppl,
    "bits_swept": BITS,
    "methods": METHODS,
    "results": results,
    "verdict": verdict,
    "naive_worst_bit": int(naive_argmax),
    "is_u_shape": bool(is_u_shape),
    "nsep_pchan_monotonic": bool(is_monotonic),
}

print(f"\n{'='*80}")
print("FINAL JSON")
print(f"{'='*80}")
print(json.dumps(final, indent=2, ensure_ascii=False))
