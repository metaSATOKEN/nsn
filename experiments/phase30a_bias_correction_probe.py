# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Phase 30a: Bias-correction probe on attention K/V (Paper C scope)
# ============================================================
# EXPLORATORY / sandbox. NOT for the NSN paper.
# Forward-looking probe for a possible Paper C extension.
#
# Background. An external analysis (Gemini, 2026-04) proposed an
# "NSSD" (Norm-Separated State Duality) recipe for hybrid SSM +
# Transformer architectures. Of NSSD's three components --
#
#   (i)   nsep on input projections (already what nsn does on
#         attention K/V)
#   (ii)  per-channel bias correction on the recursive state to
#         counter accumulated drift, modelled on SSDi8 (Kim et al.,
#         ICLR 2026 Poster, openreview pjMDZJd4rT)
#   (iii) hybrid filtering (attention's softmax as a low-pass on
#         SSM-side noise)
#
# -- only (i) is already in nsn's shipped recipe, and (iii) is
# not directly implementable in our cache-only framework. (ii),
# however, IS implementable as a calibration step on top of the
# v0.3 attention K/V quantiser. Phase 30a tests just (ii) on a
# standard transformer to answer one specific question:
#
#   On a model where attention is FIR and quantisation errors do
#   NOT accumulate across time, does adding per-channel bias
#   correction (calibrated on a held-out text) reduce ΔPPL below
#   what the calibration-free v0.3 recipe achieves?
#
# Predictions (informative either way):
#
#   - No-op (|Δ ΔPPL| < 0.5 % of baseline): the FIR/IIR
#     distinction holds empirically. Bias correction is
#     unnecessary on attention; reserve it for SSM blocks where
#     errors actually accumulate. Sharpens the paper's
#     "calibration-free" claim from "we chose not to" to
#     "calibration provides no benefit on our substrate".
#
#   - Helps (Δ ΔPPL < -0.5 % of baseline): the renormalisation
#     step in qa_nsep_pchan_asym does not fully remove a
#     systematic per-channel bias. A future Paper-C-tier
#     recipe could add an optional one-pass calibration
#     refinement step.
#
#   - Hurts (Δ ΔPPL > +0.5 %): test-time distribution drift
#     from the calibration text overshoots the correction.
#     Direct evidence that calibration-free is locally robust;
#     paper's design choice is empirically optimal.
#
# Protocol:
#   - Model:        Qwen2.5-7B (Phase 28c/28d baseline; PPL +3.47 %
#                   under v0.3 asym INT3 known)
#   - Calibration:  WikiText-2 train split, single 1024-token window
#                   (different from test data; standard practice)
#   - Test:         WikiText-2 test split, 10 sliding windows at
#                   ctx=1024 stride=512 (same as Phase 28c)
#   - Conditions:
#     1. FP16 baseline
#     2. asym INT3 (paper recipe)
#     3. asym INT3 + bias correction
#     4. asym INT8 (control - small noise regime)
#     5. asym INT8 + bias correction (control)
#
# Runtime: ~10-12 min on A100 80 GB (5 PPL runs over 10 windows
# plus one calibration pass).
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

import json, math
import torch.nn.functional as F
from datetime import datetime
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# =============================================================
# === CELL V2 === Quantizer (v0.3 shipped recipe)
# =============================================================

def qa_nsep_pchan_asym(x, bits):
    dtype = x.dtype
    xf = x.float()
    n = xf.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    d = xf / n
    d_min = d.amin(dim=0, keepdim=True)
    d_max = d.amax(dim=0, keepdim=True)
    d_range = (d_max - d_min).clamp(min=1e-12)
    levels = 2 ** bits - 1
    step = d_range / levels
    dq = ((d - d_min) / step).round().clamp(0, levels) * step + d_min
    dq = dq / dq.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return (n * dq).to(dtype)


# =============================================================
# === CELL V3 === KV cache plumbing
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


def _layer_is_transformer_kv(past, li, expected_ctx_len):
    if not _layer_has_kv(past, li):
        return False
    ok, ov = get_kv(past, li)
    if ok is None or ov is None:
        return False
    if not (torch.is_tensor(ok) and torch.is_tensor(ov)):
        return False
    if ok.ndim != 4 or ov.ndim != 4:
        return False
    if ok.shape != ov.shape:
        return False
    if ok.shape[-2] != expected_ctx_len:
        return False
    return True


# =============================================================
# === CELL V4 === Bias calibration
# =============================================================

@torch.inference_mode()
def calibrate_bias(model, calib_ids, attn_method):
    """
    Run a single forward pass on calib_ids, capture each transformer
    layer's K/V tensor, and compute the per-(head, channel) mean of
    (K_fp16 - K_quantised) and (V_fp16 - V_quantised). The mean is
    taken over the time axis T, giving a [B, H, 1, D] tensor that
    can be added to a quantised cache to subtract the systematic
    per-channel bias.

    Returns: dict {layer_idx: {"k_bias": Tensor[B,H,1,D],
                                "v_bias": Tensor[B,H,1,D]}}
    """
    biases = {}
    out = model(calib_ids, use_cache=True)
    past = out.past_key_values
    expected_ctx_len = calib_ids.shape[1]

    for li in range(n_cache_layers(past)):
        if not _layer_is_transformer_kv(past, li, expected_ctx_len):
            continue
        ok, ov = get_kv(past, li)
        ok_q = torch.zeros_like(ok)
        ov_q = torch.zeros_like(ov)
        for h in range(ok.shape[1]):
            ok_q[0, h] = attn_method(ok[0, h]).to(ok.dtype)
            ov_q[0, h] = attn_method(ov[0, h]).to(ov.dtype)
        # Mean over time axis T -> per-channel bias
        k_bias = (ok.float() - ok_q.float()).mean(dim=-2, keepdim=True)
        v_bias = (ov.float() - ov_q.float()).mean(dim=-2, keepdim=True)
        biases[li] = {"k_bias": k_bias, "v_bias": v_bias}
    return biases


def compress_with_optional_bias(past, attn_method, biases,
                                  expected_ctx_len):
    """
    Apply attn_method to every transformer-KV layer, then optionally
    add the per-channel bias correction from `biases`. If `biases`
    is None or this layer is missing, no correction is applied
    (equivalent to the standard v0.3 recipe).
    """
    if attn_method is None:
        return
    for li in range(n_cache_layers(past)):
        if not _layer_is_transformer_kv(past, li, expected_ctx_len):
            continue
        ok, ov = get_kv(past, li)
        ok_q = torch.zeros_like(ok)
        ov_q = torch.zeros_like(ov)
        for h in range(ok.shape[1]):
            ok_q[0, h] = attn_method(ok[0, h]).to(ok.dtype)
            ov_q[0, h] = attn_method(ov[0, h]).to(ov.dtype)
        if biases is not None and li in biases:
            # Cast bias to the cache dtype and add. The bias was
            # calibrated in fp32 for numerical safety; the broadcast
            # is shape [B, H, 1, D] -> [B, H, T, D].
            ok_q = ok_q + biases[li]["k_bias"].to(ok_q.dtype)
            ov_q = ov_q + biases[li]["v_bias"].to(ov_q.dtype)
        set_kv(past, li, ok_q, ov_q)


# =============================================================
# === CELL V5 === WikiText-2 PPL with optional bias correction
# =============================================================

def ppl_with(model, tok, attn_method, biases, eval_text,
              ctx_len=1024, stride=512, max_windows=10):
    ids = tok.encode(eval_text, return_tensors="pt").to("cuda")
    T = ids.shape[1]
    total_nll = 0.0; total_tokens = 0; windows_done = 0

    with torch.inference_mode():
        for begin in range(0, T - ctx_len, stride):
            if windows_done >= max_windows:
                break
            window = ids[:, begin:begin + ctx_len]
            split = ctx_len // 2
            context = window[:, :split]
            targets = window[:, split:]

            out_ctx = model(context, use_cache=True)
            past = out_ctx.past_key_values
            first_logits = out_ctx.logits[:, -1:, :]
            compress_with_optional_bias(
                past, attn_method, biases,
                expected_ctx_len=split,
            )
            out_tgt = model(targets, past_key_values=past, use_cache=False)
            rest_logits = out_tgt.logits[:, :-1, :]
            all_logits = torch.cat([first_logits, rest_logits], dim=1)
            log_probs = F.log_softmax(all_logits[0].float(), dim=-1)
            nll = -log_probs.gather(-1,
                                     targets[0].unsqueeze(-1)).squeeze(-1)
            total_nll += nll.sum().item()
            total_tokens += targets.shape[1]
            windows_done += 1
    return math.exp(total_nll / total_tokens), windows_done


# =============================================================
# === CELL V6 === Load model + datasets
# =============================================================

MODEL_ID    = "Qwen/Qwen2.5-7B"
MODEL_LABEL = "Qwen2.5-7B"

CTX_LEN          = 1024
STRIDE           = 512
N_TEST_WINDOWS   = 10
CALIB_TOKENS     = 1024  # match the half-window context length

print(f"\n{'=' * 80}")
print(f"Loading {MODEL_LABEL}  ({MODEL_ID})")
print(f"{'=' * 80}")

tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
mdl = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    use_safetensors=True,
    trust_remote_code=True,
)
mdl.eval()
print(f"  VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# Test text: WikiText-2 test (same as Phase 28c)
ds_test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
test_text = "\n\n".join([x for x in ds_test["text"]
                          if x.strip()])[:48000]
print(f"  Test text:        {len(test_text)} chars")

# Calibration text: WikiText-2 TRAIN (different from test, no
# contamination)
ds_train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
calib_text = "\n\n".join([x for x in ds_train["text"]
                           if x.strip()])[:8000]
print(f"  Calibration text: {len(calib_text)} chars (train split)")

# Calibration token IDs at CTX_LEN length (so the shape guard
# `ok.shape[-2] == expected_ctx_len` is satisfied)
calib_ids = tok.encode(calib_text, return_tensors="pt").to("cuda")
calib_ids = calib_ids[:, :CALIB_TOKENS]
print(f"  Calibration tokens used: {calib_ids.shape[1]}")


# =============================================================
# === CELL V7 === Calibrate biases for INT3 and INT8
# =============================================================

print(f"\n{'-' * 80}")
print("Calibrating per-channel biases on a 1024-token train window")
print(f"{'-' * 80}")

print("  asym INT3 calibration ...", flush=True)
bias_int3 = calibrate_bias(
    mdl, calib_ids,
    attn_method=lambda x: qa_nsep_pchan_asym(x, 3),
)
print(f"    captured biases for {len(bias_int3)} transformer layers")

print("  asym INT8 calibration ...", flush=True)
bias_int8 = calibrate_bias(
    mdl, calib_ids,
    attn_method=lambda x: qa_nsep_pchan_asym(x, 8),
)
print(f"    captured biases for {len(bias_int8)} transformer layers")

# Quick sanity numbers: the magnitude of the calibrated bias
# (per channel) and the per-token L2 norm of the bias tensor.
def _bias_stats(biases):
    if not biases:
        return 0.0, 0.0
    k_norms, v_norms = [], []
    for li, d in biases.items():
        k_norms.append(d["k_bias"].norm().item())
        v_norms.append(d["v_bias"].norm().item())
    import statistics
    return statistics.mean(k_norms), statistics.mean(v_norms)

k3, v3 = _bias_stats(bias_int3)
k8, v8 = _bias_stats(bias_int8)
print(f"\n  Mean across-layer ||k_bias||  : INT3={k3:.4g}  INT8={k8:.4g}")
print(f"  Mean across-layer ||v_bias||  : INT3={v3:.4g}  INT8={v8:.4g}")


# =============================================================
# === CELL V8 === Run the 5-condition PPL sweep
# =============================================================

CONDITIONS = [
    ("1. FP16 baseline",
     None, None),
    ("2. asym INT3 (no bias)",
     lambda x: qa_nsep_pchan_asym(x, 3), None),
    ("3. asym INT3 + bias correction",
     lambda x: qa_nsep_pchan_asym(x, 3), bias_int3),
    ("4. asym INT8 (no bias, control)",
     lambda x: qa_nsep_pchan_asym(x, 8), None),
    ("5. asym INT8 + bias (control)",
     lambda x: qa_nsep_pchan_asym(x, 8), bias_int8),
]

print(f"\n{'=' * 80}")
print(f"WikiText-2 sliding-window PPL ({N_TEST_WINDOWS} windows, "
      f"ctx={CTX_LEN})")
print(f"{'=' * 80}")

results = {}
fp16_ppl = None
for label, fn, biases in CONDITIONS:
    ppl, n_win = ppl_with(
        mdl, tok, fn, biases, test_text,
        ctx_len=CTX_LEN, stride=STRIDE,
        max_windows=N_TEST_WINDOWS,
    )
    if fn is None:
        fp16_ppl = ppl
    dppl = ppl - fp16_ppl
    pct  = 100.0 * dppl / fp16_ppl
    results[label] = {
        "ppl":            ppl,
        "delta_ppl":      dppl,
        "pct_of_base":    pct,
        "n_windows":      n_win,
    }
    print(f"  {label:<38s}  ppl={ppl:>8.4f}  "
          f"ΔPPL={dppl:+8.4f}  ({pct:+7.3f} % of base)")


# =============================================================
# === CELL V9 === Hypothesis readout
# =============================================================

int3_no_bias  = results["2. asym INT3 (no bias)"]["pct_of_base"]
int3_w_bias   = results["3. asym INT3 + bias correction"]["pct_of_base"]
int8_no_bias  = results["4. asym INT8 (no bias, control)"]["pct_of_base"]
int8_w_bias   = results["5. asym INT8 + bias (control)"]["pct_of_base"]

delta_int3 = int3_w_bias - int3_no_bias  # negative -> bias HELPS
delta_int8 = int8_w_bias - int8_no_bias

print(f"\n{'=' * 80}")
print("Bias-correction effect readout")
print(f"{'=' * 80}")
print(f"  INT3:  no-bias = {int3_no_bias:+7.3f} %  "
      f"|  + bias = {int3_w_bias:+7.3f} %  "
      f"|  Δ = {delta_int3:+7.3f} %  "
      f"({'helps' if delta_int3 < -0.5 else 'no-op' if abs(delta_int3) <= 0.5 else 'hurts'})")
print(f"  INT8:  no-bias = {int8_no_bias:+7.3f} %  "
      f"|  + bias = {int8_w_bias:+7.3f} %  "
      f"|  Δ = {delta_int8:+7.3f} %  "
      f"({'helps' if delta_int8 < -0.5 else 'no-op' if abs(delta_int8) <= 0.5 else 'hurts'})")

print("\n  Verdict:")
if delta_int3 < -0.5:
    print("    🏆 BIAS CORRECTION HELPS at INT3.")
    print("    The renormalisation step in qa_nsep_pchan_asym does not")
    print("    fully remove a systematic per-channel bias on this")
    print("    substrate. A future Paper-C-tier recipe could expose an")
    print("    optional one-pass calibration refinement.")
elif abs(delta_int3) <= 0.5:
    print("    ⚖️  NO-OP at INT3, consistent with the FIR/IIR")
    print("    framing of attention's quantisation tolerance: errors")
    print("    do not accumulate across the 1-step attention window,")
    print("    so subtracting their mean is irrelevant. This sharpens")
    print("    the paper's calibration-free claim from a design choice")
    print("    to an empirical optimum on this substrate.")
else:
    print("    ❌ BIAS CORRECTION HURTS at INT3.")
    print("    Test-time distribution drift overshoots the calibrated")
    print("    correction; the calibration-free recipe is locally")
    print("    robust, and this is direct evidence that adding")
    print("    calibration cost would not buy quality.")


# =============================================================
# === CELL V10 === Save
# =============================================================

output = {
    "date":           datetime.utcnow().isoformat(timespec="seconds"),
    "phase":          "30a",
    "purpose":        ("Bias-correction probe: tests whether adding "
                       "per-(layer, head, channel) calibrated mean-"
                       "error correction to the v0.3 recipe reduces "
                       "ΔPPL on a standard transformer where errors "
                       "do not accumulate across time. Paper-C scope "
                       "(NSSD inspiration; not for the current NSN "
                       "paper)."),
    "model":          MODEL_ID,
    "label":          MODEL_LABEL,
    "protocol": {
        "ctx_len":          CTX_LEN,
        "stride":           STRIDE,
        "n_test_windows":   N_TEST_WINDOWS,
        "calib_tokens":     CALIB_TOKENS,
        "calib_split":      "WikiText-2 train",
        "test_split":       "WikiText-2 test",
        "conditions":       [c[0] for c in CONDITIONS],
    },
    "fp16_baseline_ppl":   fp16_ppl,
    "results":             results,
    "bias_norm_summary": {
        "int3_mean_k_norm": k3, "int3_mean_v_norm": v3,
        "int8_mean_k_norm": k8, "int8_mean_v_norm": v8,
    },
    "deltas": {
        "int3_with_minus_without_bias_pct": delta_int3,
        "int8_with_minus_without_bias_pct": delta_int8,
    },
}

out_path = "/content/phase30a_bias_correction_probe.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\nSaved -> {out_path}")


# =============================================================
# === READING ===
# =============================================================
#
# The read on the result is the verdict in CELL V9. Three branches:
#
#   A. Bias correction helps INT3 ΔPPL by more than 0.5 % of
#      baseline. There is a meaningful systematic per-channel bias
#      that the v0.3 renormalisation does not absorb. Paper C
#      should consider a calibrated extension as an optional path.
#
#   B. Bias correction is a no-op (|Δ ΔPPL| ≤ 0.5 % of baseline).
#      Consistent with FIR/IIR theory: attention's 1-step window
#      means error means cancel out and there is nothing for
#      calibration to remove. The v0.3 calibration-free design
#      is at the empirical optimum on this substrate. Reserve
#      bias correction for IIR (SSM) substrates where errors
#      genuinely accumulate -- which is exactly NSSD's domain.
#
#   C. Bias correction hurts INT3 ΔPPL by more than 0.5 % of
#      baseline. The calibrated bias trained on a held-out text
#      transfers poorly. Either: (a) the renormalisation step
#      already removes most of the bias, so adding back any
#      static correction creates a small overshoot; or (b)
#      per-channel bias is not stationary across text domains
#      and needs adaptive estimation. Both cases are evidence
#      AGAINST a calibrated extension on attention.
#
# The INT8 row is a low-noise control. Under the FIR/IIR theory
# we expect roughly the same Δ at INT8 as at INT3 -- bias
# correction either helps both or neither. If INT8 + bias hurts
# while INT3 + bias helps, that is structural evidence for the
# bias correction being a noise-floor effect rather than an
# accumulating-error fix.
# ============================================================
