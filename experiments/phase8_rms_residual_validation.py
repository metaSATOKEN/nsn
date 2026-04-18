# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Phase 8: RMS Residual Validation across 5 Additional Models
# ============================================================
# Phase 7 found on 2 models (Qwen2-7B, Llama-3.1-8B):
#   INT3 + per-channel RMS residual correction reaches Pareto
#   front at 4+D/head bits, surpassing INT4 alone on Mode 1.
#   Fixed-0.25 constant magnitude BACKFIRES on Mode 1.
#
# This phase validates the finding on the remaining 5 models
# from our bit-width study, to confirm universality of
# "INT3 + RMS beats INT4 alone" (or identify where it fails).
#
# Models:
#   Pythia-6.9B    (Mode 2, base model, noisy PPL regime)
#   Qwen2.5-14B    (Mode 2, post-Qwen2 recipe, low baseline)
#   Mistral-7B-v0.1(Mode 2, classic baseline)
#   GPT-J-6B       (Mode 2, 2021 architecture, head_dim=256)
#   Phi-3-medium   (Mode 2, Microsoft recipe, low baseline)
#
# Focused methods (to keep runtime under 1h):
#   - baseline (FP16)
#   - nsep+pchan INT3          [3 bits, reference]
#   - nsep+pchan INT3 + R-RMS  [4+D bits, new Pareto candidate]
#   - nsep+pchan INT4          [4 bits, upper-bit reference]
#
# Runtime: ~60 min total (5 models × ~10 min each + model loads)
# ============================================================


# =============================================================
# === CELL V1 === Setup + quantizers
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


# ── Quantizers (same as Phase 7) ─────────────────────────────
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


def qa_nsep_pchan_resid_rms(x, bits_base):
    dtype = x.dtype
    xf = x.float()
    n = xf.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    d = xf / n
    qm = 2**(bits_base-1) - 1
    s = d.abs().amax(dim=0, keepdim=True).clamp(min=1e-12) / qm
    d_scaled = d / s
    d_q = d_scaled.round().clamp(-qm, qm)
    residual = d_scaled - d_q
    r_rms = (residual ** 2).mean(dim=0, keepdim=True).sqrt().clamp(min=1e-12)
    r_1bit = r_rms * residual.sign()
    d_corrected = (d_q + r_1bit) * s
    d_corrected = d_corrected / d_corrected.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return (n * d_corrected).to(dtype)


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


# ── WikiText-2 ──────────────────────────────────────────────
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
    ("baseline (FP16)",            None,                                       "16"),
    ("nsep+pchan INT3",            lambda x: qa_nsep_pchan(x, 3),              "3"),
    ("nsep+pchan INT3 + R-RMS",    lambda x: qa_nsep_pchan_resid_rms(x, 3),    "4+D"),
    ("nsep+pchan INT4",            lambda x: qa_nsep_pchan(x, 4),              "4"),
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
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto",
        use_safetensors=True,
    )
    mdl.eval()
    print(f"VRAM used: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    print(f"\n  {'Method':<28} | {'bits':>6} | {'PPL':>10} | {'ΔPPL':>10}")
    print(f"  " + "-"*62)

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
        print(f"  {name:<28} | {bits:>6} | {ppl:>10.4f} | {dstr:>10}")

    # Does INT3+RMS beat INT4 alone?
    int3_rms = results["nsep+pchan INT3 + R-RMS"]["delta"]
    int4_alone = results["nsep+pchan INT4"]["delta"]
    int3_alone = results["nsep+pchan INT3"]["delta"]

    print(f"\n  Claim check: 'INT3 + RMS beats INT4 alone'")
    print(f"    INT3 alone    ΔPPL: +{int3_alone:.4f}")
    print(f"    INT3 + R-RMS  ΔPPL: +{int3_rms:.4f}")
    print(f"    INT4 alone    ΔPPL: +{int4_alone:.4f}")
    if int3_rms < int4_alone and int3_rms < int3_alone:
        print(f"    ✓ INT3+RMS WINS: better than both INT3 alone AND INT4 alone")
        verdict = "WINS"
    elif int3_rms < int4_alone:
        print(f"    ~ INT3+RMS beats INT4 but not INT3 alone (unusual)")
        verdict = "BEATS_INT4"
    elif int3_rms < int3_alone:
        print(f"    ~ INT3+RMS improves INT3 but loses to INT4 alone")
        verdict = "BEATS_INT3_ONLY"
    else:
        print(f"    ✗ INT3+RMS loses to both alone variants")
        verdict = "LOSES"

    return {
        "label": label, "model": model_id,
        "baseline_ppl": baseline_ppl,
        "results": results,
        "verdict": verdict,
    }


# =============================================================
# === CELL V2 === Pythia-6.9B (Mode 2, base, noisy)
# =============================================================

pythia_res = run_model("EleutherAI/pythia-6.9b", "Pythia-6.9B")


# =============================================================
# === CELL V3 === Qwen2.5-14B (Mode 2, post-Qwen2 recipe)
# =============================================================

qwen25_res = run_model("Qwen/Qwen2.5-14B", "Qwen2.5-14B")


# =============================================================
# === CELL V4 === Mistral-7B-v0.1 (Mode 2, classic)
# =============================================================

mistral_res = run_model("mistralai/Mistral-7B-v0.1", "Mistral-7B-v0.1")


# =============================================================
# === CELL V5 === GPT-J-6B (Mode 2, head_dim=256)
# =============================================================

gptj_res = run_model("EleutherAI/gpt-j-6b", "GPT-J-6B")


# =============================================================
# === CELL V6 === Phi-3-medium-4k-instruct (Mode 2, MS)
# =============================================================

phi3_res = run_model("microsoft/Phi-3-medium-4k-instruct", "Phi-3-medium-4k-instruct")


# =============================================================
# === CELL V7 === 7-model synthesis (Phase 7 + 8 combined)
# =============================================================

# Phase 7 prior
PRIOR = [
    {"label": "Qwen2-7B (Mode 1)",    "model": "Qwen/Qwen2-7B",
     "baseline_ppl": 8.369,
     "int3_alone":  0.402,
     "int3_rms":    0.025,
     "int4_alone":  0.054,
     "verdict":     "WINS"},
    {"label": "Llama-3.1-8B (Mode 2)", "model": "meta-llama/Llama-3.1-8B-Instruct",
     "baseline_ppl": 8.942,
     "int3_alone":  0.271,
     "int3_rms":    0.075,
     "int4_alone":  0.036,
     "verdict":     "BEATS_INT3_ONLY"},
]

all_results = [{"label": r["label"], "mode": r["label"].split("(")[-1].rstrip(")"),
                "baseline": r["baseline_ppl"],
                "int3_alone": r["int3_alone"], "int3_rms": r["int3_rms"],
                "int4_alone": r["int4_alone"], "verdict": r["verdict"]}
               for r in PRIOR]

for res in [pythia_res, qwen25_res, mistral_res, gptj_res, phi3_res]:
    r = res["results"]
    all_results.append({
        "label": res["label"],
        "mode": "Mode 2",
        "baseline": res["baseline_ppl"],
        "int3_alone": r["nsep+pchan INT3"]["delta"],
        "int3_rms":   r["nsep+pchan INT3 + R-RMS"]["delta"],
        "int4_alone": r["nsep+pchan INT4"]["delta"],
        "verdict":    res["verdict"],
    })

print("\n" + "="*100)
print("7-MODEL VALIDATION: INT3 + RMS Residual vs INT4 alone")
print("="*100)
print(f"\n  {'Model':<28} | {'baseline':>8} | {'INT3':>8} | {'INT3+RMS':>9} | {'INT4':>8} | {'Verdict':<16}")
print(f"  " + "-"*92)
for r in all_results:
    print(f"  {r['label']:<28} | {r['baseline']:>8.3f} | +{r['int3_alone']:>6.4f} | +{r['int3_rms']:>7.4f} | +{r['int4_alone']:>6.4f} | {r['verdict']:<16}")

# Counts
wins       = sum(1 for r in all_results if r["verdict"] == "WINS")
beats_int4 = sum(1 for r in all_results if r["verdict"] == "BEATS_INT4")
beats_int3 = sum(1 for r in all_results if r["verdict"] == "BEATS_INT3_ONLY")
loses      = sum(1 for r in all_results if r["verdict"] == "LOSES")

print(f"\n  Verdict distribution (7 models):")
print(f"    WINS (beats both alone variants)   : {wins}")
print(f"    BEATS_INT4 (but loses to INT3)     : {beats_int4}")
print(f"    BEATS_INT3_ONLY (loses to INT4)    : {beats_int3}")
print(f"    LOSES                              : {loses}")

# Save
final = {
    "experiment": "phase8_rms_residual_validation",
    "timestamp": datetime.now().isoformat(),
    "phase7_prior": PRIOR,
    "new_phase8_measurements": {
        "pythia_6_9b":    pythia_res,
        "qwen2_5_14b":    qwen25_res,
        "mistral_7b_v01": mistral_res,
        "gpt_j_6b":       gptj_res,
        "phi_3_medium":   phi3_res,
    },
    "combined_7_model_summary": all_results,
    "verdict_counts": {"WINS": wins, "BEATS_INT4": beats_int4,
                       "BEATS_INT3_ONLY": beats_int3, "LOSES": loses},
}
print(f"\n{'='*100}\nFINAL JSON\n{'='*100}")
print(json.dumps(final, indent=2, default=str))
