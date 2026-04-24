# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Phase 29: Long-context generation-loop detection on
#           Qwen 3.6 27B under naive-vs-v0.3 KV quantization
# ============================================================
# EXPLORATORY / sandbox. Not for the NSN paper.
#
# A Reddit post (Mac Studio M2 Ultra, 128 GB unified memory) reports
# severe looping / "Bart Simpson writing on chalkboard" behaviour
# under llama.cpp with CACHE_TYPE_K = CACHE_TYPE_V = q8_0 on
# Qwen 3.6 27 B and 35 B at 131 072-token context.
#
# q8_0 in llama.cpp is block-wise naive symmetric absmax INT8. That
# is very close in spirit to our Phase 28c/28d naive per-row sym
# condition. Our data on Qwen2.5-7B shows naive INT8 is clean
# (+0.3 % ΔPPL at ctx=1024) but naive INT4 is catastrophic
# (+3663 %). Looping under autoregressive decode is a downstream
# symptom of KV quality collapse: with the attention distribution
# corrupted by per-token V-cache norm dispersion, the model falls
# into a repetition attractor.
#
# This phase does NOT attempt to reproduce the exact Reddit config
# (no llama.cpp, no 128 K). It runs a controlled probe on the same
# model family to answer two questions:
#
#   Q1. Does naive per-row quantization at various bit widths make
#       Qwen 3.6 27B loop under greedy generation?
#
#   Q2. Does the v0.3 asym nsep+pchan recipe suppress the looping?
#
# Protocol:
#   - Model:   Qwen 3.6 27B (adjust MODEL_ID below if the HF path
#              name differs from what this script guesses)
#   - Context: 4096-token WikiText-2 prefix (+ a "short story"
#              prompt as a second prefix, for one long-form task)
#   - Methods: FP16 baseline
#              naive sym INT {3, 4, 6, 8}
#              v0.3  asym INT {3, 4, 6, 8}
#   - Decode:  greedy, 500 new tokens, KV cache compressed once
#              after prefill (matches the paper's Phase 24 NIH
#              protocol, same compress-once-then-decode pattern
#              as a realistic deployment)
#
# Loop metrics (per generation):
#   - Longest exactly-repeated substring (char-level)
#   - 3-gram / 4-gram repetition ratio (1 - unique/total)
#   - Distinct-1 / Distinct-2 (standard generation-quality metric)
#   - Tail 200 chars of the decoded output (human-eyeball check)
#
# A "looping" condition is flagged when:
#   - longest repeated substring  >= 60 chars, OR
#   - 4-gram repetition ratio     >= 0.50
#
# Runtime budget (A100 80 GB estimated):
#   - Weights:  ~54 GB in fp16 (27 B params)
#   - KV cache: ~0.6 GB at 4096-token context (GQA, 8 KV heads)
#   - Generation: 500 tok × 9 methods × 2 prefixes = 18 runs
#     at ~30-50 ms/tok, ~5-8 min/run, ~2-2.5 h total
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

import json, math, re
from datetime import datetime
from collections import Counter
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# =============================================================
# === CELL V2 === Two quantizers (matches Phase 28d)
# =============================================================

def qa_naive_perrow_sym(x, bits):
    """llama.cpp q8_0-style naive per-row symmetric absmax."""
    dtype = x.dtype
    xf = x.float()
    qm = 2 ** (bits - 1) - 1
    s = xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / qm
    q = (xf / s).round().clamp(-qm, qm) * s
    return q.to(dtype)


def qa_nsep_pchan_asym(x, bits):
    """v0.3 shipped recipe: norm-sep + per-channel + asym window."""
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
# === CELL V3 === KV cache plumbing (matches Phase 28 series)
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


# =============================================================
# === CELL V4 === Loop / repetition metrics
# =============================================================

def longest_repeated_substring(text, max_scan_len=200):
    """Longest exactly-repeated substring in `text` (char-level).
    O(n^2) brute force, bounded by max_scan_len inner loop for
    practicality. Sufficient for 500-token outputs."""
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
    """Fraction of n-grams that are duplicates (1 - unique/total)."""
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    total = len(ngrams)
    unique = len(set(ngrams))
    return 1.0 - unique / total


def distinct_n(tokens, n):
    """Distinct-n: unique n-grams / total n-grams. Higher = more
    diverse. Common generation-quality metric."""
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(ngrams)) / len(ngrams)


def loop_verdict(metrics):
    """Flag condition as 'loop' if generation looks collapsed."""
    if metrics["longest_repeat_chars"] >= 60:
        return "LOOP"
    if metrics["repetition_4gram"] >= 0.50:
        return "LOOP"
    if metrics["distinct_2"] < 0.25:
        return "LOOP"
    return "ok"


# =============================================================
# === CELL V5 === Greedy generation with compressed KV cache
# =============================================================

@torch.inference_mode()
def generate_with_compressed_cache(
    prefix_ids, method_fn, mdl, n_new=500
):
    """Prefill the prefix, compress the resulting KV cache once,
    then greedy-decode n_new new tokens. New tokens' KV entries
    accumulate uncompressed at the tail -- matches a realistic
    deployment where only the long prefix needs compression."""
    device = prefix_ids.device

    # Prefill
    out = mdl(prefix_ids, use_cache=True)
    past = out.past_key_values
    compress_cache(past, method_fn)

    next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated_ids = [next_id.item()]

    # Greedy decode
    for _ in range(n_new - 1):
        out = mdl(next_id, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_ids.append(next_id.item())

    return generated_ids


# =============================================================
# === CELL V6 === Prefixes
# =============================================================

ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
wtext = "\n\n".join([x for x in ds["text"] if x.strip()])
print(f"WikiText-2 total chars: {len(wtext)}")

PREFIX_A_TEXT = wtext[:20000]  # ~4-5k tokens depending on tokenizer
PREFIX_B_TEXT = (
    "Write a short story about a robot who learns to paint. "
    "Describe the robot's first visit to an art museum, the "
    "moment it discovers color, and the first painting it "
    "finishes. Keep the tone gentle and reflective.\n\n"
    "Here is the story:\n\n"
)


# =============================================================
# === CELL V7 === Run the sweep
# =============================================================

# ─────────────────────────────────────────────────────────────
# >>>>>  Adjust if the HF path differs  <<<<<
MODEL_ID = "Qwen/Qwen3.6-27B"
MODEL_LABEL = "Qwen3.6-27B"
# ─────────────────────────────────────────────────────────────

CTX_TARGET_TOKENS = 4096
N_NEW_TOKENS = 500

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

PREFIXES = [
    ("wikitext_continuation", PREFIX_A_TEXT),
    ("short_story_prompt",    PREFIX_B_TEXT),
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
        gen_ids = generate_with_compressed_cache(
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
        }
        metrics["verdict"] = loop_verdict(metrics)
        per_method[method_name] = metrics

        print(f"  {method_name:<22s}  "
              f"lrs={metrics['longest_repeat_chars']:>4d}ch  "
              f"r4={metrics['repetition_4gram']:.2f}  "
              f"d2={metrics['distinct_2']:.2f}  "
              f"[{metrics['verdict']}]")

    all_results[prefix_name] = {
        "ctx_tokens":  actual_ctx,
        "per_method":  per_method,
    }


# =============================================================
# === CELL V8 === Summary + save
# =============================================================

print(f"\n\n{'=' * 80}")
print(f"Loop-rate summary on {MODEL_LABEL}")
print(f"{'=' * 80}")
print(f"  {'method':<22s}  {'wikitext':>10s}  {'story':>10s}  {'both-loop':>10s}")
print(f"  {'-' * 22}  {'-' * 10}  {'-' * 10}  {'-' * 10}")
for method_name, _ in METHODS:
    v_wiki = all_results["wikitext_continuation"]["per_method"][method_name]["verdict"]
    v_story = all_results["short_story_prompt"]["per_method"][method_name]["verdict"]
    both = "YES" if (v_wiki == "LOOP" and v_story == "LOOP") else ""
    print(f"  {method_name:<22s}  {v_wiki:>10s}  {v_story:>10s}  {both:>10s}")


output = {
    "date":           datetime.utcnow().isoformat(timespec="seconds"),
    "phase":          "29",
    "purpose":        ("Replicate Reddit-reported 'generation looping' "
                       "symptom of naive KV quantization at long "
                       "context on Qwen3.6-27B, and measure "
                       "suppression under v0.3 asym recipe."),
    "model":          MODEL_ID,
    "label":          MODEL_LABEL,
    "protocol": {
        "ctx_target_tokens": CTX_TARGET_TOKENS,
        "n_new_tokens":      N_NEW_TOKENS,
        "decode":            "greedy",
        "compress_policy":   "once at prefill; new tokens uncompressed",
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

out_path = "/content/phase29_qwen_loop_detection.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\nSaved -> {out_path}")


# =============================================================
# === READING ===
# =============================================================
#
# Expected patterns based on Phase 28c/28d data for Qwen2.5-7B
# (Qwen3.6-27B may differ; this is the priors file):
#
#   naive INT8 : likely 'ok' (Phase 28c: +0.3 % of baseline)
#   naive INT6 : possibly LOOP (Phase 28c: +3498 % -- catastrophic)
#   naive INT4 : likely LOOP  (Phase 28c: +3663 % -- catastrophic)
#   naive INT3 : mixed        (Phase 28c: +176 %; may degrade text
#                             quality without hard-looping)
#
#   asym  INT8 : ok           (Phase 28d: +0.02 % of baseline)
#   asym  INT6 : ok           (Phase 28d: +0.07 %)
#   asym  INT4 : ok           (Phase 28d: +1.03 %)
#   asym  INT3 : ok           (Phase 28d: +3.47 %, shipped recipe)
#
# If naive INT4 AND INT6 both LOOP but all asym conditions 'ok',
# the Reddit user's llama.cpp-q8_0-at-128K looping story is
# consistent with a Qwen-family outlier-channel pathology that
# the v0.3 recipe would have suppressed end-to-end.
#
# If naive INT8 also LOOPs, the story is more interesting:
# something in llama.cpp's specific q8_0 implementation (block
# structure, group-wise scaling) or the 128 K context length
# interacts with attention in a way that a simpler 4 K probe
# misses. Worth a follow-up.
#
# Post-run: consider running at CTX_TARGET_TOKENS = 16384 or
# 32768 if the 4 K sweep is inconclusive -- longer context amplifies
# V-cache per-token norm dispersion, which is exactly the
# mechanism §5.3 and §5.7 trace NSN's bit-mixing-hallucination
# phenomenon to.
# ============================================================
