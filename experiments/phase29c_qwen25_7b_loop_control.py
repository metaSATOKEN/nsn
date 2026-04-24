# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Phase 29c: Loop-detection control experiment on Qwen2.5-7B
#            (standard-transformer baseline for Phase 29b)
# ============================================================
# EXPLORATORY / sandbox. Not for the NSN paper.
#
# Phase 29b ran a loop-detection sweep on Qwen3.6-27B (a hybrid
# SSM + softmax-attention model, 16 / 64 transformer layers
# interleaved 1-in-4 with linear / SSM layers) and found:
#
#   - The transformer-only compression guard worked as designed
#     (48 / 64 layers correctly skipped as non-transformer).
#   - Every v0.3 asym condition still LOOPed.
#   - Crucially, FP16 baseline also LOOPed on one of the two
#     prefixes, which means the looping is NOT caused by
#     quantization: Qwen3.6-27B has a greedy-decode prefix-
#     dependent instability that exists at full precision.
#
# The open question is: does the same sweep on a standard
# softmax-attention model (our Phase 28 benchmark Qwen2.5-7B)
# show the expected Phase 28c / 28d pattern --
#
#   naive INT4 / INT6 -> LOOP  (multi-peak naive catastrophe)
#   v0.3 asym all bits -> ok   (shipped recipe rescues)
#
# -- on the same long-context generation protocol? If yes, the
# two experiments together constitute a clean "standard vs
# hybrid" contrast: on standard transformers the v0.3 recipe
# suppresses naive-quant-induced looping; on hybrid models the
# looping has a different root cause and the recipe is neither
# the right lever nor the right scope.
#
# Protocol (same as Phase 29b, modulo the model):
#   - Model:   Qwen2.5-7B (all-transformer, no fla / causal-conv1d)
#   - Context: 4096-token WikiText-2 prefix, two offsets
#   - Methods: FP16 baseline
#              naive sym INT {3, 4, 6, 8}
#              v0.3  asym INT {3, 4, 6, 8}
#   - Decode:  greedy, 500 new tokens, transformer-only compress
#
# Expected outcomes (priors from Phase 28c/28d):
#   Layer breakdown: 28 / 28 compressed (all layers transformer)
#   FP16 baseline:   ok on both prefixes
#   naive INT4/INT6: LOOP (we know these are +3663 % / +3498 % dPPL)
#   naive INT3:      borderline (+176 % dPPL, may or may not LOOP)
#   naive INT8:      ok (+0.3 % dPPL)
#   asym INT3-INT8:  ok (+3.47 / +1.03 / +0.07 / +0.02 % dPPL,
#                    all well below the text-quality breakdown point)
#
# Runtime (A100 80 GB):
#   - Weights:  ~14 GB in fp16 (7 B params)
#   - Total:    ~40-60 min for 9 methods × 2 prefixes
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

# Qwen2.5-7B is pure softmax-attention -- no fla / causal-conv1d
# required (unlike Phase 29b).

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
# === CELL V2 === Quantizers (same as Phase 28d / 29 / 29b)
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
# === CELL V3 === KV cache plumbing (same shape guard as 29b)
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


def compress_cache_transformer_only(past, method_fn, expected_ctx_len):
    total = n_cache_layers(past)
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
# === CELL V4 === Loop / repetition metrics (same as 29 / 29b)
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
# === CELL V5 === Greedy generation with compressed KV cache
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
# === CELL V6 === Prefixes (matches Phase 29b exactly)
# =============================================================

ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
wtext = "\n\n".join([x for x in ds["text"] if x.strip()])
print(f"WikiText-2 total chars: {len(wtext)}")

PREFIXES = [
    ("wikitext_offset_0",     wtext[0:25000]),
    ("wikitext_offset_200k",  wtext[200000:225000]),
]


# =============================================================
# === CELL V7 === Run the sweep
# =============================================================

MODEL_ID    = "Qwen/Qwen2.5-7B"
MODEL_LABEL = "Qwen2.5-7B"

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

        if (not layer_breakdown_reported and fn is not None
                and info["total_layers"] > 0):
            print(f"\n  Layer breakdown (first compressed method):")
            print(f"    total layers:       {info['total_layers']}")
            print(f"    compressed layers:  {info['compressed_layers']}")
            print(f"    skipped layers:     {info['skipped_layers']}")
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
print(f"Loop-rate summary on {MODEL_LABEL}  (standard transformer)")
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
    "phase":          "29c",
    "purpose":        ("Control experiment for Phase 29b: run the "
                       "same loop-detection sweep on Qwen2.5-7B "
                       "(standard softmax attention, already "
                       "benchmarked in Phase 28c/28d) to isolate "
                       "what is model-architecture vs what is "
                       "quantization-driven looping."),
    "model":          MODEL_ID,
    "label":          MODEL_LABEL,
    "protocol": {
        "ctx_target_tokens": CTX_TARGET_TOKENS,
        "n_new_tokens":      N_NEW_TOKENS,
        "decode":            "greedy",
        "compress_policy":   "transformer-only (same shape guard as 29b)",
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

out_path = "/content/phase29c_qwen25_7b_loop_control.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\nSaved -> {out_path}")


# =============================================================
# === READING ===
# =============================================================
#
# The contrast matrix with Phase 29b is the point:
#
#   Expected on Qwen2.5-7B (this run):
#     FP16 baseline      : ok on both prefixes
#     naive INT4, INT6   : LOOP (multi-peak naive catastrophe,
#                          Phase 28c: +3663 % / +3498 % dPPL)
#     naive INT3         : possibly LOOP (Phase 28c: +176 %)
#     naive INT8         : ok (Phase 28c: +0.3 %)
#     asym INT3-INT8     : ok (Phase 28d: +3.47 / +1.03 / +0.07 /
#                          +0.02 % dPPL; outlier rescue)
#
#   Observed on Qwen3.6-27B (Phase 29b):
#     FP16 baseline      : ok on one prefix, LOOP on the other
#     Every quantized    : LOOP (with the 16/64 transformer-only
#                          guard already applied)
#
#   Reading:
#
#     - If Qwen2.5-7B shows the expected pattern above, then
#       v0.3 asym IS an effective loop-suppressor on standard
#       transformers. Phase 29b's uniform LOOP is attributable
#       to the model's own greedy-decode instability, not to
#       any v0.3 failure.
#
#     - If Qwen2.5-7B also shows uniform LOOP, something is
#       wrong with the compress-once-then-greedy-decode protocol
#       itself (maybe interacting badly with the cache object
#       at long context). Re-examine the protocol.
# ============================================================
