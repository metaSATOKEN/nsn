# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Phase 20: bf16 baseline disambiguation for 2026 flagships
# ============================================================
# Phase 19 produced a surprising result on the two 2026 flagship
# models: both Qwen3.5-9B and gemma-4-E4B showed NEGATIVE ΔPPL
# on the asym-only ablation (method B). Specifically:
#
#   Qwen3.5-9B:   B-ΔPPL = -0.004  (within noise)
#   gemma-4-E4B:  B-ΔPPL = -0.440  (~2% improvement over FP16
#                                   baseline, outside noise)
#
# Phase 19's discussion flagged two competing hypotheses:
#
#   H1 (FP16-eval artifact):   Both models are bf16-trained.
#       Loading them in FP16 for inference introduces minor
#       numerical instability in attention logits. Quantization
#       noise happens to partially mask that instability, making
#       the quantized PPL look marginally better than the
#       "inflated" FP16 baseline. If we re-run with a bf16
#       baseline, the negative ΔPPL should shrink or disappear.
#
#   H2 (real regularization): Quantization noise is genuinely
#       acting as beneficial regularization on well-trained
#       2026 models. A bf16 baseline would still produce
#       negative ΔPPL.
#
# This phase loads each 2026 model once in FP16 and once in BF16,
# runs the same four-method ablation under each, and compares the
# baselines and the B-deltas.
#
# Decision rule:
#   - If bf16 baseline PPL < fp16 baseline PPL by > 0.1
#     AND asym-only ΔPPL goes less-negative / positive under bf16
#     -> H1 is supported; negative ΔPPL is an FP16 eval artifact.
#   - If bf16 baseline ≈ fp16 baseline
#     AND B-delta stays negative even under bf16
#     -> H2 is supported; there's real regularization going on.
#   - Mixed signals -> neither hypothesis clean; mark as open
#     question in the paper.
#
# Models (2 of the 8 ablation set, specifically the two that
# showed the anomaly):
#   Qwen3.5-9B     (2026, hybrid attention; B-ΔPPL = -0.004 fp16)
#   gemma-4-E4B    (2026, multimodal; B-ΔPPL = -0.440 fp16)
#
# Runtime: ~40-60 min on Colab A100 (2 models × 2 dtypes × 5 runs).
# ============================================================


# =============================================================
# === CELL V1 === Setup (fresh runtime)
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
# === CELL V2 === Quantizer zoo (dtype-preserving)
# =============================================================
# These quantizers cast to FP32 internally for numerical stability
# but preserve the input dtype on output, so they work correctly
# on both FP16 and BF16 input tensors.

def qa_naive_perrow_sym(x, bits):
    """(A) naive per-row symmetric absmax INT3."""
    dtype = x.dtype
    xf = x.float()
    qm = 2**(bits-1) - 1
    s = xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / qm
    return (((xf / s).round().clamp(-qm, qm)) * s).to(dtype)


def qa_asym_pchan_no_nsep(x, bits):
    """(B) asymmetric per-channel INT3, NO norm separation."""
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
    """(C) nsep + per-channel absmax (symmetric) = modeq v0.2."""
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
    """(D) nsep + per-channel min/max (asymmetric) = modeq v0.3."""
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
# === CELL V3 === Cache compression + PPL (hybrid-skip aware)
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
    ("baseline (no quant)",             None,                                  "16"),
    ("(A) naive per-row sym INT3",      lambda x: qa_naive_perrow_sym(x, 3),   "3"),
    ("(B) asym per-channel, no nsep",   lambda x: qa_asym_pchan_no_nsep(x, 3), "3"),
    ("(C) nsep+pchan sym INT3 (v0.2)",  lambda x: qa_nsep_pchan_sym(x, 3),     "3"),
    ("(D) nsep+pchan asym INT3 (v0.3)", lambda x: qa_nsep_pchan_asym(x, 3),    "3"),
]


def run_model(model_id, label, dtype, dtype_label):
    """Load model in the specified dtype and run the 5-method sweep."""
    global mdl, tok
    try:
        del mdl, tok
    except NameError:
        pass
    gc.collect(); torch.cuda.empty_cache()

    print(f"\n{'='*80}")
    print(f"Loading {label}  DTYPE = {dtype_label}  ({model_id}) ...")
    print(f"{'='*80}")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
        use_safetensors=True,
        trust_remote_code=True,
    )
    mdl.eval()
    print(f"  VRAM used: {torch.cuda.memory_allocated()/1e9:.1f} GB")
    print(f"  model param dtype (sample): {next(iter(mdl.parameters())).dtype}")

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

    return {
        "label":              label,
        "model":              model_id,
        "dtype":              dtype_label,
        "baseline_ppl":       baseline_ppl,
        "results":            results,
        "delta_A":            results["(A) naive per-row sym INT3"]["delta"],
        "delta_B":            results["(B) asym per-channel, no nsep"]["delta"],
        "delta_C":            results["(C) nsep+pchan sym INT3 (v0.2)"]["delta"],
        "delta_D":            results["(D) nsep+pchan asym INT3 (v0.3)"]["delta"],
    }


# =============================================================
# === CELL V4 === Qwen3.5-9B: FP16 run, then BF16 run
# =============================================================

qwen35_fp16 = run_model("Qwen/Qwen3.5-9B",    "Qwen3.5-9B", torch.float16, "fp16")
qwen35_bf16 = run_model("Qwen/Qwen3.5-9B",    "Qwen3.5-9B", torch.bfloat16, "bf16")


# =============================================================
# === CELL V5 === Gemma 4-E4B: FP16 run, then BF16 run
# =============================================================

gemma4_fp16 = run_model("google/gemma-4-E4B", "gemma-4-E4B", torch.float16, "fp16")
gemma4_bf16 = run_model("google/gemma-4-E4B", "gemma-4-E4B", torch.bfloat16, "bf16")


# =============================================================
# === CELL V6 === H1 vs H2 verdict
# =============================================================

def summarise(fp16_res, bf16_res):
    base_fp16 = fp16_res["baseline_ppl"]
    base_bf16 = bf16_res["baseline_ppl"]
    base_shift = base_fp16 - base_bf16

    delta_B_fp16 = fp16_res["delta_B"]
    delta_B_bf16 = bf16_res["delta_B"]

    delta_D_fp16 = fp16_res["delta_D"]
    delta_D_bf16 = bf16_res["delta_D"]

    return {
        "label":         fp16_res["label"],
        "base_fp16":     base_fp16,
        "base_bf16":     base_bf16,
        "base_shift":    base_shift,
        "delta_B_fp16":  delta_B_fp16,
        "delta_B_bf16":  delta_B_bf16,
        "B_goes_positive_under_bf16": (delta_B_bf16 > 0 and delta_B_fp16 < 0),
        "B_shrinks_under_bf16":       (abs(delta_B_bf16) < abs(delta_B_fp16)),
        "delta_D_fp16":  delta_D_fp16,
        "delta_D_bf16":  delta_D_bf16,
    }


qwen35_summary = summarise(qwen35_fp16, qwen35_bf16)
gemma4_summary = summarise(gemma4_fp16, gemma4_bf16)

print("\n" + "=" * 88)
print(f"Phase 20 H1 vs H2 disambiguation  ({datetime.now().isoformat(timespec='seconds')})")
print("=" * 88)

for s in [qwen35_summary, gemma4_summary]:
    print(f"\n  ### {s['label']} ###")
    print(f"    Baseline PPL: fp16 = {s['base_fp16']:.4f}  bf16 = {s['base_bf16']:.4f}  "
          f"shift (fp16 - bf16) = {s['base_shift']:+.4f}")
    print(f"    Asym-only ΔPPL: fp16 = {s['delta_B_fp16']:+.4f}  bf16 = {s['delta_B_bf16']:+.4f}")
    print(f"    v0.3     ΔPPL: fp16 = {s['delta_D_fp16']:+.4f}  bf16 = {s['delta_D_bf16']:+.4f}")
    if s["B_goes_positive_under_bf16"]:
        print(f"    -> B crosses zero (was negative under fp16, positive under bf16).")
    elif s["B_shrinks_under_bf16"]:
        print(f"    -> B shrinks in magnitude under bf16 but doesn't flip sign.")
    else:
        print(f"    -> B does NOT shrink under bf16.")


def verdict(s):
    """Return 'H1', 'H2', or 'mixed' per decision rule."""
    strong_base_shift = s["base_shift"] > 0.1
    if strong_base_shift and s["B_goes_positive_under_bf16"]:
        return "H1_supported_strongly"
    if strong_base_shift and s["B_shrinks_under_bf16"]:
        return "H1_supported_mildly"
    if (not strong_base_shift) and (s["delta_B_bf16"] < 0):
        return "H2_supported"
    return "mixed"

print("\n  VERDICTS:")
for s in [qwen35_summary, gemma4_summary]:
    v = verdict(s)
    print(f"    {s['label']:<16}: {v}")

# Save
out_path = "/content/phase20_bf16_disambiguation.json"
combined = {
    "date":           datetime.now().isoformat(timespec='seconds'),
    "hypothesis_H1":  "Negative ΔPPL on 2026 models is an FP16-eval artifact; bf16 should flatten it.",
    "hypothesis_H2":  "Quantization noise is genuinely regularizing these models; bf16 should preserve negative ΔPPL.",
    "runs": {
        "Qwen3.5-9B/fp16":   qwen35_fp16,
        "Qwen3.5-9B/bf16":   qwen35_bf16,
        "gemma-4-E4B/fp16":  gemma4_fp16,
        "gemma-4-E4B/bf16":  gemma4_bf16,
    },
    "summary": {
        "Qwen3.5-9B":  qwen35_summary,
        "gemma-4-E4B": gemma4_summary,
    },
    "verdict": {
        "Qwen3.5-9B":  verdict(qwen35_summary),
        "gemma-4-E4B": verdict(gemma4_summary),
    },
}
with open(out_path, "w") as f:
    json.dump(combined, f, indent=2, default=str)
print(f"\nSaved -> {out_path}")

print("\n" + "=" * 88)
print("INTERPRETATION KEY")
print("=" * 88)
print("""
  H1 (FP16-eval artifact) supported if BOTH are true:
    - bf16 baseline PPL is < fp16 baseline PPL by > 0.1
    - asym-only ΔPPL shrinks toward zero (or flips sign) under bf16

  H2 (real regularization) supported if:
    - bf16 baseline PPL ≈ fp16 baseline PPL (within 0.05)
    - asym-only ΔPPL stays negative under bf16

  Mixed / inconclusive if neither holds cleanly -- flag as open
  question in the paper.
""")
