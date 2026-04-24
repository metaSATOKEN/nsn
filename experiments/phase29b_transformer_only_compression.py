# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Phase 29b: Transformer-only KV compression on hybrid models
# ============================================================
# EXPLORATORY / sandbox. Not for the NSN paper.
#
# Phase 29 ran the full-layer compress_cache on Qwen3.6-27B and
# every quantized condition looped, including the shipped v0.3
# asym INT3 / INT4 / INT6 / INT8 recipes. Root cause (hypothesis):
# Qwen3.6-27B is a hybrid architecture (flash-linear-attention +
# causal-conv1d warnings in the HF load log), so part of the
# cache object represents SSM / linear-attention state rather
# than standard [B, H, T, D] softmax-attention K/V. The Phase 29
# compress_cache iterated ALL cache layers and silently applied
# nsep+pchan+asym to whatever was in .keys / .values, which for
# the SSM layers is not a post-RoPE direction distribution --
# it is a recurrent state that gets destroyed by per-token L2
# normalization.
#
# Phase 29b improves compress_cache with a shape-based sanity
# check before compression: only compress layers whose K tensor
# is 4D with time dimension == prefill context length and whose
# K and V shapes match. SSM state tensors fail this check (they
# have a state-size dimension unrelated to prefill length, or
# a lower rank). This should leave SSM state alone and compress
# only the transformer-attention sub-layers.
#
# Also fixes the Phase 29 Prefix B bug: Prefix B was a 50-token
# short prompt that never triggered the 4 K compression probe.
# Phase 29b uses two 4 K WikiText-2 slices from different
# offsets (domain diversity without the length bug).
#
# Protocol:
#   - Model:     Qwen3.6-27B (same as Phase 29)
#   - Context:   4096-token prefix, two offsets
#   - Methods:   FP16 baseline
#                naive sym INT {3, 4, 6, 8}
#                v0.3 asym  INT {3, 4, 6, 8}
#   - Decode:    greedy, 500 new tokens
#   - New:       compress_cache returns (compressed, skipped, total)
#                to show how many layers are transformer (compressed)
#                vs SSM-like (skipped)
#
# What we are looking for:
#   (a) compressed_layers / total_layers < 1.0. If every layer is
#       compressed, the shape check missed something and the
#       Phase 29 looping story is still unexplained. If only
#       some layers are compressed, Qwen3.6-27B is indeed
#       hybrid and the skipped layers are SSM/linear-attention.
#   (b) v0.3 asym conditions recover from LOOP -> ok on Prefix A
#       once SSM layers are skipped. This would confirm the
#       mechanism hypothesis.
#   (c) Naive conditions may still loop at INT3/INT4/INT6 since
#       Mode 1-style outlier channels live in the transformer
#       attention sub-layers; this is expected and consistent
#       with Phase 28c on Qwen2.5-7B.
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

# Install the hybrid-model dependencies so we don't fall back to
# the torch implementation (which may itself be a loop source).
!pip install -q flash-linear-attention causal-conv1d || true

import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# ─────────────────────────────────────────────────────────────
# >>>>>  PASTE YOUR HF TOKEN  <<<<<
HF_TOKEN = "hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
# ─────────────────────────────────────────────────────────────
from huggingface_hub import login
login(token=HF_TOKEN)

import json
from datetime import datetime
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# =============================================================
# === CELL V2 === Quantizers (same as Phase 28d / 29)
# =============================================================

def qa_naive_perrow_sym(x, bits):
    dtype = x.dtype
    xf = x.float()
    qm = 2 ** (bits - 1) - 1
    s = xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / qm
    q = (xf / s).round().clamp(-qm, qm) * s
    return q.to(dtype)


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
# === CELL V3 === KV cache plumbing with transformer-only guard
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
    """
    Return True iff layer `li` looks like a standard transformer-
    attention K/V cache: both tensors are 4D with shape
    [B, H, T, D] and T == expected_ctx_len (prefill length).

    SSM / linear-attention state tensors fail this check because
    they are typically 3D with a state-size dimension instead
    of a time dimension, or 4D with a state-size axis that does
    not match prefill length.
    """
    if not _layer_has_kv(past, li):
        return False
    ok, ov = get_kv(past, li)
    if ok is None or ov is None:
        return False
    if not (torch.is_tensor(ok) and torch.is_tensor(ov)):
        return False
    if ok.ndim != 4 or ov.ndim != 4:
        return False
    if ok.shape != ov.shape:   # GQA keeps K/V shape-matched
        return False
    if ok.shape[-2] != expected_ctx_len:
        return False
    return True


def compress_cache_transformer_only(past, method_fn, expected_ctx_len):
    """
    Apply `method_fn` to every transformer-attention K/V layer in
    `past`. Skip layers whose cache does not look like a
    transformer K/V (likely SSM / linear-attention state).
    Returns a dict with per-run diagnostics.
    """
    total   = n_cache_layers(past)
    compressed_ids = []
    skipped_ids    = []

    if method_fn is None:
        return {
            "total_layers":       total,
            "compressed_layers":  0,
            "skipped_layers":     0,
            "compressed_indices": [],
            "skipped_indices":    list(range(total)),
        }

    for li in range(total):
        if not _layer_is_transformer_kv(past, li, expected_ctx_len):
            skipped_ids.append(li)
            continue
        ok, ov = get_kv(past, li)
        ok = ok.clone(); ov = ov.clone()
        nk = torch.zeros_like(ok); nv = torch.zeros_like(ov)
        for h in range(ok.shape[1]):
            nk[0, h] = method_fn(ok[0, h]).to(ok.dtype)
            nv[0, h] = method_fn(ov[0, h]).to(ov.dtype)
        set_kv(past, li, nk, nv)
        compressed_ids.append(li)

    return {
        "total_layers":       total,
        "compressed_layers":  len(compressed_ids),
        "skipped_layers":     len(skipped_ids),
        "compressed_indices": compressed_ids,
        "skipped_indices":    skipped_ids,
    }


# =============================================================
# === CELL V4 === Loop / repetition metrics (same as Phase 29)
# =============================================================

def longest_repeated_substring(text, max_scan_len=200):
    n = len(text)
    best = 0
    for i in range(n):
        for j in range(i + 1, min(n, i + max_scan_len * 2)):
            k = 0
            while (j + k < n and text[i + k] == text[j + k]
                   and k < max_scan_len):
                k += 1
            if k > best:
                best = k
    return best

def ngram_repetition_ratio(tokens, n):
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return 1.0 - len(set(ngrams)) / len(ngrams)

def distinct_n(tokens, n):
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(ngrams)) / len(ngrams)

def loop_verdict(metrics):
    if metrics["longest_repeat_chars"] >= 60:
        return "LOOP"
    if metrics["repetition_4gram"] >= 0.50:
        return "LOOP"
    if metrics["distinct_2"] < 0.25:
        return "LOOP"
    return "ok"


# =============================================================
# === CELL V5 === Greedy generation with transformer-only KV
# =============================================================

@torch.inference_mode()
def generate_with_compressed_cache(
    prefix_ids, method_fn, mdl, n_new=500
):
    out = mdl(prefix_ids, use_cache=True)
    past = out.past_key_values
    info = compress_cache_transformer_only(
        past, method_fn, expected_ctx_len=prefix_ids.shape[1]
    )

    next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated_ids = [next_id.item()]

    for _ in range(n_new - 1):
        out = mdl(next_id, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_ids.append(next_id.item())

    return generated_ids, info


# =============================================================
# === CELL V6 === Prefixes (both 4K tokens, different offsets)
# =============================================================

ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
wtext = "\n\n".join([x for x in ds["text"] if x.strip()])
print(f"WikiText-2 total chars: {len(wtext)}")

# Two 4K-token windows from non-overlapping offsets.
# Offsets in chars; the tokenizer will trim to 4096 tokens.
PREFIXES = [
    ("wikitext_offset_0",     wtext[0:25000]),
    ("wikitext_offset_200k",  wtext[200000:225000]),
]


# =============================================================
# === CELL V7 === Run the sweep
# =============================================================

# ─────────────────────────────────────────────────────────────
# >>>>>  Adjust if the HF path differs  <<<<<
MODEL_ID    = "Qwen/Qwen3.6-27B"
MODEL_LABEL = "Qwen3.6-27B"
# ─────────────────────────────────────────────────────────────

CTX_TARGET_TOKENS = 4096
N_NEW_TOKENS      = 500

METHODS = [
    ("FP16 baseline",          None),
    ("(A) naive INT8",         lambda x: qa_naive_perrow_sym(x, 8)),
    ("(A) naive INT6",         lambda x: qa_naive_perrow_sym(x, 6)),
    ("(A) naive INT4",         lambda x: qa_naive_perrow_sym(x, 4)),
    ("(A) naive INT3",         lambda x: qa_naive_perrow_sym(x, 3)),
    ("(D) asym  INT8",         lambda x: qa_nsep_pchan_asym(x, 8)),
    ("(D) asym  INT6",         lambda x: qa_nsep_pchan_asym(x, 6)),
    ("(D) asym  INT4",         lambda x: qa_nsep_pchan_asym(x, 4)),
    ("(D) asym  INT3",         lambda x: qa_nsep_pchan_asym(x, 3)),
]

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


all_results = {}
layer_breakdown_reported = False

for prefix_name, prefix_text in PREFIXES:
    prefix_ids_full = tok.encode(
        prefix_text, return_tensors="pt"
    ).to("cuda")
    prefix_ids = prefix_ids_full[:, :CTX_TARGET_TOKENS]
    actual_ctx = prefix_ids.shape[1]

    print(f"\n{'=' * 80}")
    print(f"Prefix: {prefix_name}  (ctx = {actual_ctx} tokens)")
    print(f"{'=' * 80}")

    per_method = {}
    for method_name, fn in METHODS:
        gen_ids, info = generate_with_compressed_cache(
            prefix_ids, fn, mdl, n_new=N_NEW_TOKENS
        )
        gen_text = tok.decode(gen_ids, skip_special_tokens=True)

        metrics = {
            "longest_repeat_chars":  longest_repeated_substring(gen_text),
            "repetition_3gram":      ngram_repetition_ratio(gen_ids, 3),
            "repetition_4gram":      ngram_repetition_ratio(gen_ids, 4),
            "distinct_1":            distinct_n(gen_ids, 1),
            "distinct_2":            distinct_n(gen_ids, 2),
            "tail_200_chars":        gen_text[-200:],
            "n_chars":               len(gen_text),
            "n_tokens":              len(gen_ids),
            "compression_info":      info,
        }
        metrics["verdict"] = loop_verdict(metrics)
        per_method[method_name] = metrics

        tag = ""
        if fn is not None:
            tag = (f" [compressed {info['compressed_layers']}/"
                   f"{info['total_layers']} layers]")

        print(f"  {method_name:<22s}  "
              f"lrs={metrics['longest_repeat_chars']:>4d}ch  "
              f"r4={metrics['repetition_4gram']:.2f}  "
              f"d2={metrics['distinct_2']:.2f}  "
              f"[{metrics['verdict']}]{tag}")

        # Report layer breakdown once for the first compressed run
        if (not layer_breakdown_reported and fn is not None
                and info["total_layers"] > 0):
            print(f"\n  Layer breakdown (first compressed method):")
            print(f"    total layers:       {info['total_layers']}")
            print(f"    compressed layers:  {info['compressed_layers']}")
            print(f"    skipped layers:     {info['skipped_layers']}")
            n_to_show = min(20, len(info["compressed_indices"]))
            print(f"    compressed indices: "
                  f"{info['compressed_indices'][:n_to_show]}"
                  f"{' ...' if len(info['compressed_indices']) > n_to_show else ''}")
            n_to_show = min(20, len(info["skipped_indices"]))
            print(f"    skipped indices:    "
                  f"{info['skipped_indices'][:n_to_show]}"
                  f"{' ...' if len(info['skipped_indices']) > n_to_show else ''}")
            print()
            layer_breakdown_reported = True

    all_results[prefix_name] = {
        "ctx_tokens":  actual_ctx,
        "per_method":  per_method,
    }


# =============================================================
# === CELL V8 === Summary + save
# =============================================================

print(f"\n\n{'=' * 80}")
print(f"Loop-rate summary on {MODEL_LABEL}  (transformer-only)")
print(f"{'=' * 80}")
print(f"  {'method':<22s}  {'wiki_0':>10s}  {'wiki_200k':>10s}  "
      f"{'both-loop':>10s}")
print(f"  {'-' * 22}  {'-' * 10}  {'-' * 10}  {'-' * 10}")
for method_name, _ in METHODS:
    v_a = all_results["wikitext_offset_0"]["per_method"][method_name]["verdict"]
    v_b = all_results["wikitext_offset_200k"]["per_method"][method_name]["verdict"]
    both = "YES" if (v_a == "LOOP" and v_b == "LOOP") else ""
    print(f"  {method_name:<22s}  {v_a:>10s}  {v_b:>10s}  {both:>10s}")


output = {
    "date":           datetime.utcnow().isoformat(timespec="seconds"),
    "phase":          "29b",
    "purpose":        ("Transformer-only KV compression on hybrid "
                       "(linear-attention / SSM) models. Skip cache "
                       "layers whose K/V shape does not look like "
                       "[B, H, prefill_T, D]."),
    "model":          MODEL_ID,
    "label":          MODEL_LABEL,
    "protocol": {
        "ctx_target_tokens": CTX_TARGET_TOKENS,
        "n_new_tokens":      N_NEW_TOKENS,
        "decode":            "greedy",
        "compress_policy":   ("transformer-only: once at prefill, "
                              "skip layers where K.ndim != 4 or "
                              "K.shape[-2] != prefill_ctx_len or "
                              "K.shape != V.shape"),
        "prefix_names":      [p[0] for p in PREFIXES],
        "methods":           [m[0] for m in METHODS],
    },
    "loop_thresholds": {
        "longest_repeated_substring_chars": 60,
        "repetition_4gram_ratio":           0.50,
        "distinct_2_below":                 0.25,
    },
    "results":         all_results,
}

out_path = "/content/phase29b_transformer_only_compression.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\nSaved -> {out_path}")


# =============================================================
# === READING ===
# =============================================================
#
# Layer breakdown is the primary diagnostic. Expected outcomes:
#
#   Case X. compressed_layers == total_layers (all compressed).
#       The shape check did not separate transformer from SSM;
#       either Qwen3.6-27B is all-transformer (then Phase 29's
#       looping was NOT an SSM artifact and we need a different
#       explanation), or the SSM layers' cache tensors happen
#       to have [B, H, prefill_T, D] shape (worth inspecting).
#
#   Case Y. compressed_layers < total_layers (the expected case).
#       Hybrid confirmed. Now check the loop verdicts:
#
#       Y1. All v0.3 asym conditions recover to 'ok' while some
#           naive conditions still LOOP. This is the shipped
#           recipe's expected behaviour; confirms that Phase 29's
#           LOOP under asym was from over-applying nsep to SSM
#           state.
#
#       Y2. Some v0.3 asym conditions still LOOP. Either the
#           transformer attention layers themselves are Mode 1
#           at the specific bit width (look at Phase 28c priors),
#           or a second scope-outside interaction remains.
#
# Report: whichever case we land in, this run lets us either
# retract Phase 29's "scope-outside" claim (Case X) or refine
# it (Case Y) with concrete per-layer evidence.
# ============================================================
