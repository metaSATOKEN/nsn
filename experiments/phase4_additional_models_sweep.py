# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Phase 4: Additional Models Bit-Width Sweep
# ============================================================
# After Phase 1-3 established the 4-model dichotomy
# (Qwen2-7B Mode 1, Pythia-6.9B/Llama-3.1-8B/Qwen2.5-14B Mode 2),
# add more models to strengthen generalization claims:
#
#   1. Mistral-7B-v0.1 (iconic OSS baseline, Mode 2 expected)
#   2. GPT-J-6B (natural control, K=18.15 @ L1, head_dim=256)
#   3. Phi-3-medium-4k-instruct (Microsoft diversity, V=34.99)
#
# All are predicted Mode 2 (none has K>=15 at Layer 0), which
# would bring Mode 2 count to 6 vs Mode 1's 1.
#
# Runtime: ~70-90 min (all three)
# ============================================================


# =============================================================
# === CELL P1 === Setup + run_bit_sweep helper
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


def run_bit_sweep(hf_id, label, trust_remote_code=False):
    """Load model, run full bit-width sweep, free memory, return results."""
    print(f"\n{'='*80}")
    print(f"Loading {label} ({hf_id}) ...")
    print(f"{'='*80}")

    tok_local = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=trust_remote_code)
    mdl_local = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=torch.float16, device_map="auto",
        use_safetensors=True, trust_remote_code=trust_remote_code,
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
# === CELL P2 === Mistral-7B-v0.1 (iconic baseline)
# =============================================================

mistral_results = run_bit_sweep(
    "mistralai/Mistral-7B-v0.1",
    "Mistral-7B-v0.1"
)


# =============================================================
# === CELL P3 === GPT-J-6B (natural control, head_dim=256)
# =============================================================

gptj_results = run_bit_sweep(
    "EleutherAI/gpt-j-6b",
    "GPT-J-6B"
)


# =============================================================
# === CELL P4 === Phi-3-medium-4k-instruct (Microsoft)
# =============================================================

phi3_results = run_bit_sweep(
    "microsoft/Phi-3-medium-4k-instruct",
    "Phi-3-medium-4k-instruct",
    trust_remote_code=False  # native transformers impl
)


# =============================================================
# === CELL P5 === 7-Model Final Summary
# =============================================================

BITS = [2, 3, 4, 5, 6, 8]

# Prior Phase 1-3
PRIOR = [
    {"label": "Qwen2-7B",      "mode": "Mode 1", "baseline": 8.37,
     "naive":      [122.62, 11.92, 1382.07, 4.69, 11.67, 0.00],
     "nsep+pchan": [325.96, 0.40, 0.05, 0.01, 0.00, 0.00],
     "int4_int5_ratio": 294.7},
    {"label": "Pythia-6.9B",   "mode": "Mode 2", "baseline": 10.76,
     "naive":      [164.37, 22.46, 7.48, 2.02, 0.53, 0.00],
     "nsep+pchan": [10.15, 2.38, 0.84, 0.36, 0.04, 0.00],
     "int4_int5_ratio": 3.7},
    {"label": "Llama-3.1-8B",  "mode": "Mode 2", "baseline": 8.94,
     "naive":      [87.89, 0.42, 0.05, 0.01, 0.00, 0.00],
     "nsep+pchan": [22.84, 0.27, 0.04, 0.00, 0.00, 0.00],
     "int4_int5_ratio": 8.9},
    {"label": "Qwen2.5-14B",   "mode": "Mode 2", "baseline": 1.96,
     "naive":      [7.27, 0.86, 0.19, 0.03, 0.01, 0.00],
     "nsep+pchan": [5.84, 0.26, 0.03, 0.00, 0.00, 0.00],
     "int4_int5_ratio": 7.4},
]

print("\n" + "="*110)
print("FINAL 7-MODEL BIT-WIDTH DICHOTOMY")
print("="*110)

print(f"\n  NAIVE ΔPPL:")
print(f"  {'Model':<28} | {'Mode':<7} | {'base':>5} | {'INT2':>7} | {'INT3':>7} | {'INT4':>9} | {'INT5':>7} | {'INT6':>7} | {'INT8':>6} | I4/I5")
print(f"  " + "-"*112)

for r in PRIOR:
    vals = r["naive"]
    print(f"  {r['label']:<28} | {r['mode']:<7} | {r['baseline']:>5.2f} | " +
          " | ".join(f"+{v:>6.2f}" if i != 2 else f"+{v:>8.2f}" for i, v in enumerate(vals)) +
          f" | {r['int4_int5_ratio']:>5.1f}")

# New results
for r in [mistral_results, gptj_results, phi3_results]:
    vals = [r["results"]["naive"][f"int{b}"]["delta"] for b in BITS]
    mode = "Mode 1" if r["pattern"] == "MODE_1_INT4_VALLEY" else "Mode 2"
    print(f"  {r['label']:<28} | {mode:<7} | {r['baseline_ppl']:>5.2f} | " +
          " | ".join(f"+{v:>6.2f}" if i != 2 else f"+{v:>8.2f}" for i, v in enumerate(vals)) +
          f" | {r['int4_int5_ratio']:>5.1f}")

print(f"\n  nsep+pchan ΔPPL:")
print(f"  {'Model':<28} | {'Mode':<7} | {'INT2':>7} | {'INT3':>7} | {'INT4':>7} | {'INT5':>7} | {'INT6':>7} | {'INT8':>7}")
print(f"  " + "-"*100)

for r in PRIOR:
    vals = r["nsep+pchan"]
    print(f"  {r['label']:<28} | {r['mode']:<7} | " + " | ".join(f"+{v:>6.3f}" for v in vals))

for r in [mistral_results, gptj_results, phi3_results]:
    vals = [r["results"]["nsep+pchan"][f"int{b}"]["delta"] for b in BITS]
    mode = "Mode 1" if r["pattern"] == "MODE_1_INT4_VALLEY" else "Mode 2"
    print(f"  {r['label']:<28} | {mode:<7} | " + " | ".join(f"+{v:>6.3f}" for v in vals))

# Mode classification summary
mode_1 = sum(1 for r in PRIOR if r["mode"] == "Mode 1")
mode_2 = sum(1 for r in PRIOR if r["mode"] == "Mode 2")
for r in [mistral_results, gptj_results, phi3_results]:
    if r["pattern"] == "MODE_1_INT4_VALLEY": mode_1 += 1
    else: mode_2 += 1

print(f"\n  🎯 Mode distribution: Mode 1 = {mode_1}, Mode 2 = {mode_2}")
print(f"  📊 INT3 nsep+pchan universal sweet spot:")
for r in PRIOR:
    v = r["nsep+pchan"][1]  # INT3 index
    pct = 100 * v / r["baseline"]
    print(f"    {r['label']:<28}: ΔPPL +{v:.3f} ({pct:.1f}%)")
for r in [mistral_results, gptj_results, phi3_results]:
    v = r["results"]["nsep+pchan"]["int3"]["delta"]
    pct = 100 * v / r["baseline_ppl"]
    print(f"    {r['label']:<28}: ΔPPL +{v:.3f} ({pct:.1f}%)")

# Save
final = {
    "experiment": "phase4_additional_models_sweep",
    "timestamp": datetime.now().isoformat(),
    "mistral_7b": mistral_results,
    "gpt_j_6b": gptj_results,
    "phi_3_medium": phi3_results,
    "prior_phases_1_3": PRIOR,
    "mode_distribution": {"mode_1": mode_1, "mode_2": mode_2},
}
print(f"\n{'='*110}")
print("FINAL JSON")
print(f"{'='*110}")
print(json.dumps(final, indent=2, default=str))
