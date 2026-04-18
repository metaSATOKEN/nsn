# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Phase 3: Bit-Width Sweep on Latest Popular Models
# ============================================================
# Phase 1 (Qwen2-7B, Mode 1): INT4 valley of death confirmed.
# Phase 2 (Pythia-6.9B, Mode 2): monotonic confirmed.
#
# Phase 3 extends to two modern/popular models for impact:
#   - Llama-3.1-8B-Instruct (Meta, cleanest profile: K=4.3, V=4.3)
#   - Gemma 4 31B-it       (Google, largest dense tested: V=39.6)
#
# Both are predicted Mode 2 from prior K/V study (Layer 0 K low).
# If confirmed, we have:
#   - Mode 1: 1 case (Qwen2-7B)
#   - Mode 2: 3 cases (Pythia-6.9B + Llama + Gemma4)
# which is a strong empirical separation across vendors.
#
# Runtime: ~60-75 min total
#   Llama 3.1 8B: DL ~3min + 12 runs ~20min = ~25min
#   Gemma 4 31B:  DL ~15min + 12 runs ~30min = ~45min
# ============================================================


# =============================================================
# === CELL Z1 === Setup + run_bit_sweep helper
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
# >>>>>  PASTE YOUR HF TOKEN (Llama gated, Gemma public)  <<<<<
HF_TOKEN = "hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
# ─────────────────────────────────────────────────────────────
from huggingface_hub import login
login(token=HF_TOKEN)

import json, numpy as np
import torch.nn.functional as F
from datetime import datetime
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Quantization helpers ─────────────────────────────────────
def qa_perrow(x, bits):
    x = x.float(); qm = 2**(bits-1)-1
    s = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / qm
    return ((x/s).round().clamp(-qm, qm)) * s

def qa_perchan(x, bits):
    x = x.float(); qm = 2**(bits-1)-1
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

# ── WikiText-2 ──────────────────────────────────────────────
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
wtext = "\n\n".join([x for x in ds["text"] if x.strip()])[:48000]
print(f"WikiText-2 text length: {len(wtext)} chars")

def ppl_with(mdl_local, tok_local, method, bits,
             ctx_len=1024, stride=512, max_windows=10):
    ids = tok_local.encode(wtext, return_tensors="pt").to("cuda")
    T = ids.shape[1]
    total_nll = 0.0; total_tokens = 0; windows_done = 0

    with torch.inference_mode():
        for begin in range(0, T - ctx_len, stride):
            if windows_done >= max_windows: break
            window = ids[:, begin:begin + ctx_len]
            split = ctx_len // 2
            context = window[:, :split]
            targets = window[:, split:]

            out_ctx = mdl_local(context, use_cache=True)
            past = out_ctx.past_key_values
            first_logits = out_ctx.logits[:, -1:, :]
            compress_cache_uniform(past, method, bits)
            out_tgt = mdl_local(targets, past_key_values=past, use_cache=False)
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


def run_bit_sweep(hf_id, label):
    """Load model, run full bit-width sweep, free memory, return results."""
    print(f"\n{'='*80}")
    print(f"Loading {label} ({hf_id}) ...")
    print(f"{'='*80}")

    tok_local = AutoTokenizer.from_pretrained(hf_id)
    mdl_local = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=torch.float16, device_map="auto",
        use_safetensors=True,
    )
    mdl_local.eval()
    print(f"VRAM used: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    BITS = [2, 3, 4, 5, 6, 8]
    METHODS = ["naive", "nsep+pchan"]

    print(f"\n{label}: Bit-Width Sweep on WikiText-2")
    print(f"  Running baseline (FP16) ...", flush=True)
    baseline_ppl, nt = ppl_with(mdl_local, tok_local, "baseline", 0)
    print(f"    baseline  PPL = {baseline_ppl:.3f}\n")

    print(f"  {'Method':>14} | {'bits':>4} | {'PPL':>10} | {'ΔPPL':>10}")
    print(f"  {'-'*46}")

    results = {"baseline": {"ppl": baseline_ppl, "tokens": nt}}
    for method in METHODS:
        results[method] = {}
        for bits in BITS:
            print(f"  Running {method} INT{bits} ...", flush=True)
            ppl, nt = ppl_with(mdl_local, tok_local, method, bits)
            dppl = ppl - baseline_ppl
            results[method][f"int{bits}"] = {"ppl": ppl, "delta": dppl, "tokens": nt}
            print(f"  {method:>14} | INT{bits:<2} | {ppl:>10.3f} | {'+' if dppl>=0 else ''}{dppl:>9.3f}")

    # Detect pattern
    naive_deltas = [results["naive"][f"int{b}"]["delta"] for b in BITS]
    has_int4_valley = (naive_deltas[2] > naive_deltas[1] and
                       naive_deltas[2] > naive_deltas[3])
    strictly_monotonic = all(naive_deltas[i] >= naive_deltas[i+1] - 0.5
                              for i in range(len(naive_deltas)-1))
    int4_int5_ratio = naive_deltas[2] / max(naive_deltas[3], 0.001)

    print(f"\n  Pattern: ", end="")
    if has_int4_valley and int4_int5_ratio > 10:
        print(f"🎯 MODE 1 (INT4 valley, ratio {int4_int5_ratio:.1f}x)")
        pattern = "MODE_1_INT4_VALLEY"
    elif strictly_monotonic:
        print(f"✓ MODE 2 (monotonic, INT4/INT5 ratio {int4_int5_ratio:.1f}x)")
        pattern = "MODE_2_MONOTONIC"
    else:
        print(f"~ mixed, INT4/INT5 ratio {int4_int5_ratio:.1f}x")
        pattern = "MIXED"

    del mdl_local, tok_local
    gc.collect(); torch.cuda.empty_cache()
    return {
        "label": label, "model": hf_id,
        "baseline_ppl": baseline_ppl,
        "results": results,
        "pattern": pattern,
        "int4_int5_ratio": int4_int5_ratio,
    }


# =============================================================
# === CELL Z2 === Llama-3.1-8B-Instruct
# =============================================================

llama_results = run_bit_sweep(
    "meta-llama/Llama-3.1-8B-Instruct",
    "Llama-3.1-8B-Instruct"
)


# =============================================================
# === CELL Z3 === Gemma 4 31B-it
# =============================================================

gemma_results = run_bit_sweep(
    "google/gemma-4-31B-it",
    "Gemma 4 31B-it"
)


# =============================================================
# === CELL Z4 === 4-model summary + pattern synthesis
# =============================================================

# Prior results from Phase 1 + Phase 2
PRIOR = {
    "Qwen2-7B (Mode 1)": {
        "label": "Qwen2-7B",
        "mode_prior": "Mode 1",
        "baseline_ppl": 8.37,
        "naive":      {"int2": 122.62, "int3": 11.92, "int4": 1382.07, "int5": 4.69, "int6": 11.67, "int8": 0.00},
        "nsep+pchan": {"int2": 325.96, "int3":  0.40, "int4":    0.05, "int5": 0.01, "int6":  0.00, "int8": 0.00},
        "int4_int5_ratio": 294.7,
    },
    "Pythia-6.9B (Mode 2)": {
        "label": "Pythia-6.9B",
        "mode_prior": "Mode 2",
        "baseline_ppl": 10.76,
        "naive":      {"int2": 164.37, "int3": 22.46, "int4":  7.48, "int5": 2.02, "int6": 0.53, "int8": 0.00},
        "nsep+pchan": {"int2":  10.15, "int3":  2.38, "int4":  0.84, "int5": 0.36, "int6": 0.04, "int8": 0.00},
        "int4_int5_ratio": 3.7,
    }
}

BITS = [2, 3, 4, 5, 6, 8]

print(f"\n{'='*100}")
print(f"4-MODEL BIT-WIDTH DICHOTOMY SUMMARY")
print(f"{'='*100}")

print(f"\n  NAIVE quantization ΔPPL:")
print(f"  {'Model':<30} | {'INT2':>8} | {'INT3':>8} | {'INT4':>8} | {'INT5':>8} | {'INT6':>8} | {'INT8':>8} | I4/I5")
print(f"  {'-'*30} + {'-'*8} + {'-'*8} + {'-'*8} + {'-'*8} + {'-'*8} + {'-'*8} + {'-'*5}")

for key, r in PRIOR.items():
    vals = [r["naive"][f"int{b}"] for b in BITS]
    print(f"  {r['label']:<30} | " + " | ".join(f"+{v:>7.2f}" for v in vals) +
          f" | {r['int4_int5_ratio']:>5.1f}")

for r in [llama_results, gemma_results]:
    vals = [r["results"]["naive"][f"int{b}"]["delta"] for b in BITS]
    print(f"  {r['label']:<30} | " + " | ".join(f"+{v:>7.2f}" for v in vals) +
          f" | {r['int4_int5_ratio']:>5.1f}")

print(f"\n  nsep+pchan ΔPPL:")
print(f"  {'Model':<30} | {'INT2':>8} | {'INT3':>8} | {'INT4':>8} | {'INT5':>8} | {'INT6':>8} | {'INT8':>8}")
print(f"  {'-'*30} + {'-'*8} + {'-'*8} + {'-'*8} + {'-'*8} + {'-'*8} + {'-'*8}")

for key, r in PRIOR.items():
    vals = [r["nsep+pchan"][f"int{b}"] for b in BITS]
    print(f"  {r['label']:<30} | " + " | ".join(f"+{v:>7.3f}" for v in vals))

for r in [llama_results, gemma_results]:
    vals = [r["results"]["nsep+pchan"][f"int{b}"]["delta"] for b in BITS]
    print(f"  {r['label']:<30} | " + " | ".join(f"+{v:>7.3f}" for v in vals))

# Mode classification summary
print(f"\n{'─'*100}")
print(f"MODE CLASSIFICATION (via INT4/INT5 ratio and curve shape)")
print(f"{'─'*100}")
print(f"\n  Mode 1 (INT4 valley of death):")
print(f"    Qwen2-7B           INT4/INT5 ratio: 294.7x")
print(f"\n  Mode 2 (monotonic):")
print(f"    Pythia-6.9B        INT4/INT5 ratio:   3.7x")
print(f"    {llama_results['label']:<18} INT4/INT5 ratio: {llama_results['int4_int5_ratio']:>5.1f}x")
print(f"    {gemma_results['label']:<18} INT4/INT5 ratio: {gemma_results['int4_int5_ratio']:>5.1f}x")

# Pragmatic recommendations
print(f"\n{'─'*100}")
print(f"INT3 nsep+pchan as universal 5.33x-compression sweet spot")
print(f"{'─'*100}")
for key, r in PRIOR.items():
    v = r["nsep+pchan"]["int3"]
    baseline = r["baseline_ppl"]
    pct = 100 * v / baseline
    print(f"  {r['label']:<18}: ΔPPL +{v:.3f} ({pct:.1f}% of baseline PPL)")
for r in [llama_results, gemma_results]:
    v = r["results"]["nsep+pchan"]["int3"]["delta"]
    baseline = r["baseline_ppl"]
    pct = 100 * v / baseline
    print(f"  {r['label']:<18}: ΔPPL +{v:.3f} ({pct:.1f}% of baseline PPL)")

# Save
final = {
    "experiment": "phase3_llama_gemma_bitwidth_sweep",
    "timestamp": datetime.now().isoformat(),
    "llama_3_1_8b": llama_results,
    "gemma_4_31b": gemma_results,
    "prior_results": PRIOR,
}
print(f"\n{'='*100}")
print("FINAL JSON")
print(f"{'='*100}")
print(json.dumps(final, indent=2, ensure_ascii=False, default=str))
