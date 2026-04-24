# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Phase 29d: Diagnostic probe of hybrid-model fragility on
#            Qwen3.6-27B transformer-attention layers
# ============================================================
# EXPLORATORY / sandbox. Not for the NSN paper.
#
# Phase 29b established: on Qwen3.6-27B, compressing just the
# 16 of 64 transformer-attention layers with v0.3 asym INT3
# pushes the generation r4 from 0.16 (FP16 baseline) to 0.52
# (loop). The question left open: WHICH part of that compression
# is doing the damage?
#
# Hypotheses to distinguish:
#
#   H1. Cache-mechanics bug. The compress_cache clone + replace
#       pattern itself breaks something in the hybrid HF cache
#       object (internal bookkeeping, counter, etc.), not the
#       quantization math.
#
#   H2. Magnitude fragility. Any non-zero perturbation -- even
#       a 0.1 % rescaling -- drives r4 up. Implies architectural
#       sensitivity, not a quantization-noise specific problem.
#
#   H3. Asymmetric K / V sensitivity. Only one of K, V is the
#       fragile side. Which would map to "attention pattern is
#       fragile" (K) vs "value mixing feeds SSM state and is
#       fragile" (V).
#
# Six conditions on the wiki_0 4 K prefix (omit wiki_200k; it
# was saturated in 29b and carries no additional signal):
#
#   1. FP16 baseline                 -- reference
#   2. Identity  (x -> x)            -- tests H1: does clone &
#                                        replace of K/V break
#                                        anything at all?
#   3. Tiny perturbation (x -> x * 1.001)
#                                    -- tests H2: is the
#                                        problem "any perturbation"
#                                        or "quantization-amount"
#                                        perturbation?
#   4. asym INT3 K-only              -- tests H3a
#   5. asym INT3 V-only              -- tests H3b
#   6. asym INT3 K+V (Phase 29b ref) -- reference
#
# Runtime estimate: 6 conditions x 500 tokens on Qwen3.6-27B
# fp16 via torch fallback == ~15 min on A100 80 GB.
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

# Deliberately NOT installing flash-linear-attention / causal-conv1d
# here. Phase 29b observed they fail to build on Colab anyway, and
# the torch-fallback path is the regime we are characterising. A
# consistent fallback regime means 29b and 29d measurements are
# directly comparable.

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
# === CELL V2 === Quantizers + probe methods
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


def method_identity(x):
    """Clone / replace with no change. Probes H1: does the
    compress_cache mechanics itself break the hybrid cache?"""
    return x.clone()


def method_tiny_perturbation(x):
    """Multiply by 1.001 -- a 0.1 % rescaling, far below INT8
    quantization noise. Probes H2: any-perturbation vs
    quant-magnitude."""
    return (x.float() * 1.001).to(x.dtype)


# =============================================================
# === CELL V3 === KV cache plumbing with K-only / V-only
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


def compress_cache_sided(
    past, method_fn, expected_ctx_len, side="both"
):
    """
    Apply `method_fn` to K, V, or both on transformer-attention
    layers. `side` in {"both", "k", "v"}. Returns per-run
    diagnostic dict (same schema as compress_cache_transformer_only
    in Phase 29b/29c plus a 'side' field).
    """
    assert side in ("both", "k", "v")
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
            "side":               side,
        }

    for li in range(total):
        if not _layer_is_transformer_kv(past, li, expected_ctx_len):
            skipped_ids.append(li)
            continue
        ok, ov = get_kv(past, li)
        ok_orig = ok.clone(); ov_orig = ov.clone()
        nk = torch.zeros_like(ok_orig); nv = torch.zeros_like(ov_orig)

        # K side
        if side in ("both", "k"):
            for h in range(ok_orig.shape[1]):
                nk[0, h] = method_fn(ok_orig[0, h]).to(ok_orig.dtype)
        else:
            nk = ok_orig  # leave K untouched

        # V side
        if side in ("both", "v"):
            for h in range(ov_orig.shape[1]):
                nv[0, h] = method_fn(ov_orig[0, h]).to(ov_orig.dtype)
        else:
            nv = ov_orig  # leave V untouched

        set_kv(past, li, nk, nv)
        compressed_ids.append(li)

    return {
        "total_layers":       total,
        "compressed_layers":  len(compressed_ids),
        "skipped_layers":     len(skipped_ids),
        "compressed_indices": compressed_ids,
        "skipped_indices":    skipped_ids,
        "side":               side,
    }


# =============================================================
# === CELL V4 === Loop metrics (same as Phase 29 series)
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
# === CELL V5 === Generation with sided compression
# =============================================================

@torch.inference_mode()
def generate_with_probe(
    prefix_ids, method_fn, side, mdl, n_new=500
):
    out = mdl(prefix_ids, use_cache=True)
    past = out.past_key_values
    info = compress_cache_sided(
        past, method_fn,
        expected_ctx_len=prefix_ids.shape[1],
        side=side,
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
# === CELL V6 === Prefix (wiki_0 only; wiki_200k was saturated)
# =============================================================

ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
wtext = "\n\n".join([x for x in ds["text"] if x.strip()])
print(f"WikiText-2 total chars: {len(wtext)}")

PREFIX_TEXT = wtext[0:25000]
PREFIX_NAME = "wikitext_offset_0"


# =============================================================
# === CELL V7 === Run the diagnostic sweep
# =============================================================

MODEL_ID    = "Qwen/Qwen3.6-27B"
MODEL_LABEL = "Qwen3.6-27B"

CTX_TARGET_TOKENS = 4096
N_NEW_TOKENS      = 500

# (label, method_fn, side) tuples
# method_fn = None -> FP16 baseline (no compression)
# side is ignored when method_fn is None
PROBES = [
    ("1. FP16 baseline",           None,                          "both"),
    ("2. Identity (clone only)",   method_identity,                "both"),
    ("3. Tiny perturb x * 1.001",  method_tiny_perturbation,       "both"),
    ("4. asym INT3  K-only",       lambda x: qa_nsep_pchan_asym(x, 3),  "k"),
    ("5. asym INT3  V-only",       lambda x: qa_nsep_pchan_asym(x, 3),  "v"),
    ("6. asym INT3  K+V (ref)",    lambda x: qa_nsep_pchan_asym(x, 3),  "both"),
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


prefix_ids_full = tok.encode(
    PREFIX_TEXT, return_tensors="pt"
).to("cuda")
prefix_ids = prefix_ids_full[:, :CTX_TARGET_TOKENS]
actual_ctx = prefix_ids.shape[1]

print(f"\n{'=' * 80}")
print(f"Prefix: {PREFIX_NAME}  (ctx = {actual_ctx} tokens)")
print(f"{'=' * 80}")

per_probe = {}
fp16_r4 = None

for label, fn, side in PROBES:
    gen_ids, info = generate_with_probe(
        prefix_ids, fn, side, mdl, n_new=N_NEW_TOKENS
    )
    gen_text = tok.decode(gen_ids, skip_special_tokens=True)

    r3 = ngram_repetition_ratio(gen_ids, 3)
    r4 = ngram_repetition_ratio(gen_ids, 4)
    d2 = distinct_n(gen_ids, 2)
    lrs = longest_repeated_substring(gen_text)
    if fn is None:
        fp16_r4 = r4
    delta_r4 = (r4 - fp16_r4) if fp16_r4 is not None else 0.0

    metrics = {
        "side":                 side,
        "longest_repeat_chars": lrs,
        "repetition_3gram":     r3,
        "repetition_4gram":     r4,
        "delta_r4_vs_fp16":     delta_r4,
        "distinct_2":           d2,
        "tail_200_chars":       gen_text[-200:],
        "n_chars":              len(gen_text),
        "n_tokens":             len(gen_ids),
        "compression_info":     info,
    }
    per_probe[label] = metrics

    print(f"  {label:<30s}  "
          f"r4={r4:.2f}  Δr4={delta_r4:+.2f}  "
          f"d2={d2:.2f}  lrs={lrs:>4d}ch")


# =============================================================
# === CELL V8 === Diagnostic readout
# =============================================================

print(f"\n\n{'=' * 80}")
print(f"Hybrid-fragility diagnostic on {MODEL_LABEL}")
print(f"{'=' * 80}")

r4 = {label: per_probe[label]["repetition_4gram"] for label, _, _ in PROBES}

print("\n  Raw r4 (4-gram repetition):")
for label, _, _ in PROBES:
    print(f"    {label:<30s}  r4 = {r4[label]:.3f}")

# Hypothesis assertions
print("\n  Hypothesis readout:")

identity_broke = (r4["2. Identity (clone only)"] -
                  r4["1. FP16 baseline"]) > 0.15
if identity_broke:
    print("    ✅ H1 SUPPORTED: Identity clone/replace alone drives r4 up."
          " The compress_cache mechanics itself is breaking the hybrid"
          " cache, not the quantization math. Phase 29b's uniform LOOP"
          " would then be a bug, not an architectural fragility.")
else:
    print("    ❌ H1 REJECTED: Identity clone/replace is safe (Δr4 < 0.15)."
          " The cache-replacement mechanics is fine; the problem is"
          " value-level, not structural.")

tiny_broke = (r4["3. Tiny perturb x * 1.001"] -
              r4["1. FP16 baseline"]) > 0.15
int3_broke = (r4["6. asym INT3  K+V (ref)"] -
              r4["1. FP16 baseline"]) > 0.15
if not identity_broke and tiny_broke:
    print("    ✅ H2 SUPPORTED: even a 0.1 %% rescaling of K/V breaks it."
          " Sparse hybrid attention layers are magnitude-sensitive;"
          " more bit precision will not fix this.")
elif not identity_broke and not tiny_broke and int3_broke:
    print("    ✅ H2 REJECTED in favor of quantization-noise specific:"
          " tiny perturbation is fine, but INT3-level quantization is"
          " not. Bit-width-dependent; higher bits may be usable.")

k_gap = r4["4. asym INT3  K-only"] - r4["1. FP16 baseline"]
v_gap = r4["5. asym INT3  V-only"] - r4["1. FP16 baseline"]
if not identity_broke and not tiny_broke:
    if k_gap > 0.15 and v_gap < 0.05:
        print(f"    ✅ H3a SUPPORTED: K-only breaks (Δr4={k_gap:+.2f});"
              f" V-only safe (Δr4={v_gap:+.2f})."
              " The attention-pattern side is the fragile routing.")
    elif v_gap > 0.15 and k_gap < 0.05:
        print(f"    ✅ H3b SUPPORTED: V-only breaks (Δr4={v_gap:+.2f});"
              f" K-only safe (Δr4={k_gap:+.2f})."
              " The value side feeds downstream SSM state fragility.")
    elif k_gap > 0.15 and v_gap > 0.15:
        print(f"    ⚠️  Both sides fragile (K Δr4={k_gap:+.2f},"
              f" V Δr4={v_gap:+.2f}). Architectural, not asymmetric.")
    else:
        print(f"    🤔 Neither side alone breaks (K Δr4={k_gap:+.2f},"
              f" V Δr4={v_gap:+.2f}) but the joint K+V does"
              f" (Δr4={r4['6. asym INT3  K+V (ref)']-r4['1. FP16 baseline']:+.2f})."
              " Suggests a K-V coupling effect.")


# =============================================================
# === CELL V9 === Save
# =============================================================

output = {
    "date":           datetime.utcnow().isoformat(timespec="seconds"),
    "phase":          "29d",
    "purpose":        ("Diagnose which part of v0.3 compression "
                       "breaks Qwen3.6-27B: cache mechanics (H1), "
                       "any-perturbation magnitude fragility (H2), "
                       "or K vs V asymmetric sensitivity (H3)."),
    "model":          MODEL_ID,
    "label":          MODEL_LABEL,
    "protocol": {
        "ctx_target_tokens": CTX_TARGET_TOKENS,
        "n_new_tokens":      N_NEW_TOKENS,
        "decode":            "greedy",
        "prefix":            PREFIX_NAME,
        "probes":            [p[0] for p in PROBES],
    },
    "fp16_r4_baseline": fp16_r4,
    "per_probe":        per_probe,
}

out_path = "/content/phase29d_hybrid_fragility_diagnostic.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\nSaved -> {out_path}")


# =============================================================
# === READING ===
# =============================================================
#
# Priors from Phase 29 / 29b on the same prefix (wiki_0):
#   FP16 r4 ~ 0.16
#   asym INT3 K+V r4 ~ 0.52 (the "uniform LOOP" in 29b)
#
# Readout paths:
#
#   If (2) Identity breaks: H1 wins. Phase 29b's story is not
#   "hybrid fragility" but "our compress_cache mutates the cache
#   object in a way HF DynamicCache / hybrid cache doesn't
#   handle well". Opens a code fix, not a scope caveat.
#
#   If (2) is fine but (3) Tiny breaks: H2 wins. Sparse attention
#   layers in this hybrid are genuinely fragile to any float
#   perturbation. Higher bit widths will not fix this; v0.3
#   is scope-outside on principle for this architecture.
#
#   If (2) and (3) are both fine but (6) K+V breaks: the damage
#   is specifically in the quantization noise magnitude, and
#   the K-only / V-only rows pinpoint which side.
#
#   If (4) K-only breaks but (5) V-only is fine: attention
#   routing fragility. A shipped "asym-only-on-V" variant for
#   hybrid models would be safe.
#
#   If (5) V-only breaks but (4) K-only is fine: value mixing
#   fragility (probably feeding SSM state downstream).
#   A shipped "asym-only-on-K" variant would be safe.
#
#   Any single "X breaks the model" outcome is a concrete
#   scope-outside / mitigation recommendation for hybrid
#   deployments.
# ============================================================
