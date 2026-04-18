# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Phase 2: Bit-Width Sweep on Pythia-6.9B (Mode 2 distributed)
# ============================================================
# Phase 1 confirmed on Qwen2-7B (Mode 1):
#   - naive: U-shape with INT4 catastrophe valley (+1382)
#   - nsep+pchan: monotonic, INT3 sweet spot
#
# Phase 2 tests Mode 2 (distributed norm-variation):
#   - Hypothesis: naive is MONOTONIC (more bits = better, smoothly)
#   - Hypothesis: nsep+pchan is also monotonic
#   - Hypothesis: NO catastrophic INT4 valley
#
# If confirmed, we have a clean dichotomy:
#   Mode 1 models: naive shows INT4 valley of death, use INT3 or INT5+
#   Mode 2 models: both methods monotonic, use smallest safe bits
#
# Runtime: ~30-40 min (Qwen2-7B speed; Pythia-6.9B is smaller)
# ============================================================


# =============================================================
# === CELL B1 === Setup + load Pythia-6.9B
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

MODEL_ID = "EleutherAI/pythia-6.9b"

print(f"Loading {MODEL_ID} ...")
tok = AutoTokenizer.from_pretrained(MODEL_ID)
mdl = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, device_map="auto",
    use_safetensors=True,
)
mdl.eval()
print(f"VRAM used: {torch.cuda.memory_allocated()/1e9:.1f} GB")


# =============================================================
# === CELL B2 === Quantization helpers (same as Phase 1)
# =============================================================

def qa_perrow(x, bits):
    x = x.float()
    qm = 2**(bits-1) - 1
    s = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / qm
    return ((x/s).round().clamp(-qm, qm)) * s

def qa_perchan(x, bits):
    x = x.float()
    qm = 2**(bits-1) - 1
    s = x.abs().amax(dim=0, keepdim=True).clamp(min=1e-12) / qm
    return ((x/s).round().clamp(-qm, qm)) * s

def qa_nsep_pchan(x, bits):
    x = x.float()
    n = x.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    d = x / n
    dq = qa_perchan(d, bits)
    dq = dq / dq.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return n * dq

def apply_method(x, method, bits):
    if method == "baseline":
        return x
    if method == "naive":
        return qa_perrow(x, bits)
    if method == "nsep+pchan":
        return qa_nsep_pchan(x, bits)
    raise ValueError(method)

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
# === CELL B3 === WikiText-2 PPL
# =============================================================

from datasets import load_dataset
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
wtext = "\n\n".join([x for x in ds["text"] if x.strip()])[:48000]
print(f"WikiText-2 text length: {len(wtext)} chars")

def ppl_with(method, bits, ctx_len=1024, stride=512, max_windows=10):
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
print(f"Pythia-6.9B: Bit-Width Sweep on WikiText-2")
print(f"{'='*80}\n")

print("Running baseline (FP16) ...", flush=True)
baseline_ppl, nt = ppl_with("baseline", 0)
print(f"  baseline (FP16)  PPL = {baseline_ppl:.3f}  ({nt} tokens)\n")

print(f"  {'Method':>14} | {'bits':>4} | {'PPL':>10} | {'ΔPPL':>10}")
print(f"  {'-'*46}")

results = {"baseline": {"ppl": baseline_ppl, "tokens": nt}}
for method in METHODS:
    results[method] = {}
    for bits in BITS:
        print(f"  Running {method} INT{bits} ...", flush=True)
        ppl, nt = ppl_with(method, bits)
        dppl = ppl - baseline_ppl
        results[method][f"int{bits}"] = {"ppl": ppl, "delta": dppl, "tokens": nt}
        print(f"  {method:>14} | INT{bits:<2} | {ppl:>10.3f} | {'+' if dppl>=0 else ''}{dppl:>9.3f}")


# =============================================================
# === CELL B5 === Mode-2 monotonicity verdict
# =============================================================

print(f"\n{'='*80}")
print(f"MODE 2 MONOTONICITY VERDICT (Pythia-6.9B)")
print(f"{'='*80}")

print(f"\n  Bit-width table (baseline PPL={baseline_ppl:.2f}):")
print(f"  {'':>10} | {'INT2':>10} | {'INT3':>10} | {'INT4':>10} | {'INT5':>10} | {'INT6':>10} | {'INT8':>10}")
print(f"  {'-'*10} + {'-'*10} + {'-'*10} + {'-'*10} + {'-'*10} + {'-'*10} + {'-'*10}")
for method in METHODS:
    row = [f"+{results[method][f'int{b}']['delta']:.3f}" for b in BITS]
    print(f"  {method:>10} | " + " | ".join(f"{r:>10}" for r in row))

# Compare with Qwen2-7B (Mode 1) from Phase 1
print(f"\n  Reference (Qwen2-7B, Mode 1, from Phase 1):")
qwen_naive = {"int2": 122.62, "int3": 11.92, "int4": 1382.07,
              "int5": 4.69, "int6": 11.67, "int8": 0.00}
print(f"      naive | " + " | ".join(f"+{qwen_naive[f'int{b}']:>9.2f}" for b in BITS))

# Check monotonicity for naive on Pythia
naive_deltas = [results["naive"][f"int{b}"]["delta"] for b in BITS]
naive_argmax = BITS[int(np.argmax(naive_deltas))]

# Definition of "monotonic decreasing" as bits increase (allowing minor noise)
strictly_monotonic = all(naive_deltas[i] >= naive_deltas[i+1] - 0.5
                          for i in range(len(naive_deltas)-1))
has_int4_valley = (naive_deltas[2] > naive_deltas[1] and
                   naive_deltas[2] > naive_deltas[3])

# Relative INT4 severity vs Mode 1
rel_int4 = naive_deltas[2] / max(sum(naive_deltas)/len(naive_deltas), 0.01)
print(f"\n  Naive ΔPPL by bit width (Pythia-6.9B):")
for b, d in zip(BITS, naive_deltas):
    marker = "  ← MAX" if b == naive_argmax else ""
    print(f"    INT{b}: +{d:.3f}{marker}")

if has_int4_valley and naive_deltas[2] > 10 * np.median(naive_deltas):
    print(f"\n  ⚠️  Pythia-6.9B ALSO shows INT4 valley -- hypothesis unexpectedly broken")
    verdict = "MODE_2_ALSO_U_SHAPED"
elif strictly_monotonic:
    print(f"\n  🎯 MONOTONIC confirmed for Pythia-6.9B naive")
    print(f"  >>> Mode 2 is qualitatively different from Mode 1")
    print(f"  >>> INT4 valley is a Mode-1-specific phenomenon!")
    verdict = "MODE_2_MONOTONIC"
elif not has_int4_valley:
    print(f"\n  ✓ No INT4 catastrophic valley (~monotonic, mild noise)")
    print(f"  >>> Supports Mode 1 specificity of the INT4 pathology")
    verdict = "MODE_2_NO_INT4_VALLEY"
else:
    print(f"\n  ~ Mixed pattern, needs deeper analysis")
    verdict = "MIXED"

# nsep+pchan check
nsep_deltas = [results["nsep+pchan"][f"int{b}"]["delta"] for b in BITS]
print(f"\n  nsep+pchan ΔPPL by bit width:")
for b, d in zip(BITS, nsep_deltas):
    print(f"    INT{b}: +{d:.3f}")

# Combined mode classification
print(f"\n{'─'*80}")
print(f"COMBINED MODE COMPARISON (Qwen2-7B Mode 1 vs Pythia-6.9B Mode 2)")
print(f"{'─'*80}")
print(f"\n  Ratio of INT4 naive ΔPPL to INT5 naive ΔPPL:")
qwen_int4_int5_ratio = 1382.07 / max(4.69, 0.01)
pythia_int4_int5_ratio = naive_deltas[2] / max(naive_deltas[3], 0.01)
print(f"    Qwen2-7B   (Mode 1): {qwen_int4_int5_ratio:>8.1f}x  (INT4 massively worse than INT5)")
print(f"    Pythia-6.9B (Mode 2): {pythia_int4_int5_ratio:>8.1f}x  (INT4 " +
      ("much worse" if pythia_int4_int5_ratio > 10 else
       "comparable" if pythia_int4_int5_ratio > 2 else "not especially worse") + " than INT5)")

# Save
final = {
    "experiment": "phase2_pythia_69b_bitwidth_sweep",
    "model": MODEL_ID,
    "timestamp": datetime.now().isoformat(),
    "baseline_ppl": baseline_ppl,
    "bits_swept": BITS,
    "methods": METHODS,
    "results": results,
    "verdict": verdict,
    "naive_worst_bit": int(naive_argmax),
    "has_int4_valley": bool(has_int4_valley),
    "is_monotonic": bool(strictly_monotonic),
    "qwen2_7b_int4_int5_ratio": qwen_int4_int5_ratio,
    "pythia_69b_int4_int5_ratio": float(pythia_int4_int5_ratio),
}

print(f"\n{'='*80}")
print("FINAL JSON")
print(f"{'='*80}")
print(json.dumps(final, indent=2, ensure_ascii=False))
