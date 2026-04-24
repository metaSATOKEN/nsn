# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Phase 29f: Per-layer fragility probe on Qwen3.6-27B
#            (which of the 16 transformer-attention layers
#            is the most fragile to INT3 compression?)
# ============================================================
# EXPLORATORY / sandbox. Not for the NSN paper.
#
# Phase 29b found 16 transformer-attention layers at indices
# {3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63}
# interleaved 1-in-4 between SSM / linear-attention layers.
#
# Phase 29d showed compressing all 16 with asym INT3 K+V pushes
# r4 from 0.16 -> 0.52; K-only and V-only are both ~0.44-0.46.
#
# Phase 29e located the fragility threshold at per-token L2
# noise magnitude ~0.002 (any per-element magnitude above breaks).
#
# Phase 29f asks: within those 16 layers, is the fragility
# uniform (every layer contributes equally) or concentrated in
# a few specific layers (early, late, or some specific index)?
#
# Protocol:
#   - Model:     Qwen3.6-27B
#   - Prefix:    wiki_offset_0, 4096 tokens
#   - Baseline:  FP16 and "all-16-layers asym INT3 K+V"
#   - Sweep:     for each of the 16 transformer layers, compress
#                ONLY that single layer with asym INT3 K+V; leave
#                the other 15 transformer layers (and all 48 SSM
#                layers) uncompressed. Measure r4.
#   - Groups:    also test four quartile groups (4 layers each)
#                for a coarser view of position-in-stack effects.
#
# Runtime: ~25-30 min for 1 + 16 + 4 + 1 = 22 generations on
# A100 80 GB, 500 new tokens each.
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

import json
from datetime import datetime
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# =============================================================
# === CELL V2 === Quantizer
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
# === CELL V3 === KV cache plumbing with per-layer targeting
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


def compress_cache_targeted(past, method_fn, expected_ctx_len,
                             target_layer_ids):
    """
    Compress only the layers whose index is in `target_layer_ids`
    AND which pass the transformer-KV shape guard. Every other
    layer is left untouched.
    """
    total = n_cache_layers(past)
    compressed_ids = []
    for li in range(total):
        if li not in target_layer_ids:
            continue
        if not _layer_is_transformer_kv(past, li, expected_ctx_len):
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
        "requested_layers":   sorted(target_layer_ids),
        "compressed_layers":  len(compressed_ids),
        "compressed_indices": compressed_ids,
    }


# =============================================================
# === CELL V4 === Metrics
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


# =============================================================
# === CELL V5 === Generation
# =============================================================

@torch.inference_mode()
def generate_with_layer_filter(
    prefix_ids, method_fn, target_ids, mdl, n_new=500
):
    out = mdl(prefix_ids, use_cache=True)
    past = out.past_key_values
    info = compress_cache_targeted(
        past, method_fn,
        expected_ctx_len=prefix_ids.shape[1],
        target_layer_ids=set(target_ids),
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
# === CELL V6 === Prefix (wiki_0, same as 29d/e)
# =============================================================

ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
wtext = "\n\n".join([x for x in ds["text"] if x.strip()])
print(f"WikiText-2 total chars: {len(wtext)}")

PREFIX_TEXT = wtext[0:25000]
PREFIX_NAME = "wikitext_offset_0"


# =============================================================
# === CELL V7 === Transformer layer indices + groupings
# =============================================================

# Phase 29b confirmed these 16 are the transformer-attention layers.
ATTN_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31,
               35, 39, 43, 47, 51, 55, 59, 63]

# Four quartile groupings (4 layers each)
QUARTILES = [
    ("Q1 early  (layers 3,7,11,15)",       [3, 7, 11, 15]),
    ("Q2 mid-early (19,23,27,31)",         [19, 23, 27, 31]),
    ("Q3 mid-late  (35,39,43,47)",         [35, 39, 43, 47]),
    ("Q4 late   (51,55,59,63)",            [51, 55, 59, 63]),
]


# =============================================================
# === CELL V8 === Run the per-layer sweep
# =============================================================

MODEL_ID    = "Qwen/Qwen3.6-27B"
MODEL_LABEL = "Qwen3.6-27B"

CTX_TARGET_TOKENS = 4096
N_NEW_TOKENS      = 500

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


prefix_ids_full = tok.encode(
    PREFIX_TEXT, return_tensors="pt"
).to("cuda")
prefix_ids = prefix_ids_full[:, :CTX_TARGET_TOKENS]
actual_ctx = prefix_ids.shape[1]
print(f"\nPrefix: {PREFIX_NAME}  (ctx = {actual_ctx} tokens)")


def run_probe(label, method_fn, target_ids):
    gen_ids, info = generate_with_layer_filter(
        prefix_ids, method_fn, target_ids, mdl, n_new=N_NEW_TOKENS
    )
    gen_text = tok.decode(gen_ids, skip_special_tokens=True)
    r4 = ngram_repetition_ratio(gen_ids, 4)
    d2 = distinct_n(gen_ids, 2)
    lrs = longest_repeated_substring(gen_text)
    return {
        "label":                label,
        "target_layer_ids":     sorted(target_ids),
        "n_target_layers":      len(target_ids),
        "longest_repeat_chars": lrs,
        "repetition_4gram":     r4,
        "distinct_2":           d2,
        "tail_200_chars":       gen_text[-200:],
        "n_tokens":             len(gen_ids),
        "compression_info":     info,
    }


all_probes = {}

# Reference 1: FP16 baseline (no compression)
print(f"\n{'-' * 80}")
print("Reference probes")
print(f"{'-' * 80}")
r = run_probe("FP16 baseline", method_fn=None, target_ids=set())
# run_probe calls compress_cache_targeted with method_fn=None;
# compress path does not mutate anything then. Defensive: pass None
# directly through generate. Re-run with explicit None-path:
with torch.inference_mode():
    out = mdl(prefix_ids, use_cache=True)
    past = out.past_key_values  # leave untouched
    next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    gen_ids = [next_id.item()]
    for _ in range(N_NEW_TOKENS - 1):
        out = mdl(next_id, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        gen_ids.append(next_id.item())
    gen_text = tok.decode(gen_ids, skip_special_tokens=True)
fp16_r4 = ngram_repetition_ratio(gen_ids, 4)
fp16_d2 = distinct_n(gen_ids, 2)
fp16_lrs = longest_repeated_substring(gen_text)
all_probes["FP16 baseline"] = {
    "label":                "FP16 baseline",
    "target_layer_ids":     [],
    "n_target_layers":      0,
    "longest_repeat_chars": fp16_lrs,
    "repetition_4gram":     fp16_r4,
    "distinct_2":           fp16_d2,
    "tail_200_chars":       gen_text[-200:],
    "n_tokens":             len(gen_ids),
}
print(f"  FP16 baseline                        "
      f"r4={fp16_r4:.3f}            d2={fp16_d2:.2f}")

# Reference 2: all 16 transformer layers (should reproduce 29d's 0.52)
r = run_probe("asym INT3 K+V, all 16 attn layers",
              lambda x: qa_nsep_pchan_asym(x, 3),
              set(ATTN_LAYERS))
r["delta_r4_vs_fp16"] = r["repetition_4gram"] - fp16_r4
all_probes["all_16_layers"] = r
print(f"  asym INT3 K+V, all 16 attn layers    "
      f"r4={r['repetition_4gram']:.3f}  Δr4={r['delta_r4_vs_fp16']:+.3f}  "
      f"d2={r['distinct_2']:.2f}")

# Per-quartile group probes
print(f"\n{'-' * 80}")
print("Quartile-group probes (4 layers each)")
print(f"{'-' * 80}")
for qlabel, qids in QUARTILES:
    r = run_probe(qlabel,
                  lambda x: qa_nsep_pchan_asym(x, 3),
                  set(qids))
    r["delta_r4_vs_fp16"] = r["repetition_4gram"] - fp16_r4
    all_probes[qlabel] = r
    print(f"  {qlabel:<36s} r4={r['repetition_4gram']:.3f}  "
          f"Δr4={r['delta_r4_vs_fp16']:+.3f}  "
          f"d2={r['distinct_2']:.2f}")

# Per-layer single probes
print(f"\n{'-' * 80}")
print("Per-layer single-layer probes (16 layers)")
print(f"{'-' * 80}")
for li in ATTN_LAYERS:
    label = f"single layer {li:>2d}"
    r = run_probe(label,
                  lambda x: qa_nsep_pchan_asym(x, 3),
                  {li})
    r["delta_r4_vs_fp16"] = r["repetition_4gram"] - fp16_r4
    all_probes[label] = r
    print(f"  layer {li:>2d}  "
          f"r4={r['repetition_4gram']:.3f}  "
          f"Δr4={r['delta_r4_vs_fp16']:+.3f}  "
          f"d2={r['distinct_2']:.2f}")


# =============================================================
# === CELL V9 === Fragility ranking
# =============================================================

print(f"\n\n{'=' * 80}")
print(f"Per-layer fragility ranking on {MODEL_LABEL}")
print(f"{'=' * 80}")
print(f"  FP16 baseline r4 = {fp16_r4:.3f}")
print(f"  all-16 layers  r4 = {all_probes['all_16_layers']['repetition_4gram']:.3f}")
print()

per_layer = [
    (li, all_probes[f"single layer {li:>2d}"]["repetition_4gram"],
         all_probes[f"single layer {li:>2d}"]["delta_r4_vs_fp16"])
    for li in ATTN_LAYERS
]
# Sort by Δr4 descending (most fragile first)
per_layer_sorted = sorted(per_layer, key=lambda t: -t[2])

print("  Sorted by fragility (most fragile first):")
for li, r4, dr4 in per_layer_sorted:
    tag = "💀" if dr4 > 0.15 else ("⚠️" if dr4 > 0.05 else "✅")
    print(f"    layer {li:>2d}   r4={r4:.3f}   Δr4={dr4:+.3f}   {tag}")

n_fragile    = sum(1 for _, _, d in per_layer if d > 0.15)
n_borderline = sum(1 for _, _, d in per_layer if 0.05 < d <= 0.15)
n_safe       = sum(1 for _, _, d in per_layer if d <= 0.05)
print(f"\n  💀 fragile (Δr4 > 0.15):      {n_fragile} / 16 layers")
print(f"  ⚠️  borderline (0.05-0.15):    {n_borderline} / 16 layers")
print(f"  ✅ safe (Δr4 <= 0.05):         {n_safe} / 16 layers")

print(f"\n  Quartile-group r4 summary:")
for qlabel, qids in QUARTILES:
    r4 = all_probes[qlabel]["repetition_4gram"]
    dr4 = all_probes[qlabel]["delta_r4_vs_fp16"]
    print(f"    {qlabel:<36s} r4={r4:.3f}  Δr4={dr4:+.3f}")


# =============================================================
# === CELL V10 === Save
# =============================================================

output = {
    "date":           datetime.utcnow().isoformat(timespec="seconds"),
    "phase":          "29f",
    "purpose":        ("Per-layer fragility ranking on Qwen3.6-27B "
                       "transformer-attention layers. For each of "
                       "the 16 attention layers, measure r4 when "
                       "ONLY that layer is asym-INT3-compressed."),
    "model":          MODEL_ID,
    "label":          MODEL_LABEL,
    "protocol": {
        "ctx_target_tokens": CTX_TARGET_TOKENS,
        "n_new_tokens":      N_NEW_TOKENS,
        "decode":            "greedy",
        "prefix":            PREFIX_NAME,
        "attn_layer_ids":    ATTN_LAYERS,
        "quartile_groups":   {qlabel: qids for qlabel, qids in QUARTILES},
    },
    "fp16_r4_baseline": fp16_r4,
    "per_probe":        all_probes,
    "fragility_counts": {
        "fragile_dr4_gt_0p15":      n_fragile,
        "borderline_dr4_0p05_0p15": n_borderline,
        "safe_dr4_le_0p05":         n_safe,
    },
}

out_path = "/content/phase29f_per_layer_fragility.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\nSaved -> {out_path}")


# =============================================================
# === READING ===
# =============================================================
#
# Possible patterns we might see:
#
#   (U) Uniform: all 16 layers have similar Δr4 near
#       (all-16 Δr4) / 16 ~ 0.02. Compressing any single layer
#       contributes ~ linearly to the full fragility. Means the
#       16 layers share the routing load evenly.
#
#   (E) Early-heavy: layer 3 alone produces the bulk of the
#       Δr4. Early attention sets the overall token mixing
#       direction and perturbing it propagates through the rest.
#
#   (L) Late-heavy: layer 63 alone produces the bulk. Late
#       attention is the "output decision" layer.
#
#   (F) Few specific: 2-3 layers are fragile, the rest are safe.
#       Identifying those is immediately actionable: a "mixed
#       precision" variant could compress the 13 safe layers
#       and leave the 3 fragile ones in FP16.
#
#   (B) Step-function: compressing any single layer already
#       breaks it at near-0.5 r4, meaning the model has no
#       redundancy and every transformer layer is mission-
#       critical. Pessimistic but clear: hybrid attention is
#       scope-outside for per-layer compression.
#
# Pattern (F) would be the most actionable; patterns (U), (E),
# (L) suggest architectural descriptions; (B) closes the door.
# ============================================================
