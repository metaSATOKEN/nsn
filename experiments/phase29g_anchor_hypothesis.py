# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Phase 29g: Anchor-layer hypothesis test on Qwen3.6-27B
# ============================================================
# EXPLORATORY / sandbox. Not for the NSN paper.
#
# Phase 29f discovered:
#   - 3 of 16 attention layers (3, 15, 35) are individually safe
#     to compress (delta r4 ~ 0.00).
#   - The other 13 are individually fragile (delta r4 > 0.15).
#   - BUT: compressing 4 fragile layers together (groups Q2, Q3)
#     is LESS fragile than any single layer within them.
#
# Hypothesis for this phase (user's conjecture):
#   The safe layers {3, 15, 35} might act as "anchors". When
#   compressed together with fragile layers, they may provide
#   a stable reference that lets the downstream SSM / attention
#   flow adapt to the noise, pulling the fragile layers along.
#
# If the anchor hypothesis is true, we should see a non-trivial
# dilution curve: adding more fragile layers to the set {3, 15,
# 35} does NOT monotonically increase r4 the way one would
# naively expect.
#
# Six conditions on wiki_0 prefix:
#
#   1. FP16 baseline                         -- reference
#   2. {3, 15, 35} only (3 safe layers)      -- anchor-only
#   3. {3, 15, 35} + {43}                    -- +1 worst fragile
#                                               (does anchor save 43?)
#   4. {3, 15, 35} + {43, 39, 11}            -- +3 worst fragile
#   5. {3, 15, 35} + 13 fragile (all 16)     -- same as Phase 29d
#                                               all-16 reference
#   6. 13 fragile ONLY (no anchors)          -- counterfactual:
#                                               is the full-16
#                                               r4=0.52 driven by
#                                               the 13 or the 3?
#
# Runtime: ~8-10 min (6 generations * 500 tokens on A100).
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
# === CELL V2 === Quantizer + plumbing (same as 29f)
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
# === CELL V3 === Metrics (same as 29 series)
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
# === CELL V4 === Generation
# =============================================================

@torch.inference_mode()
def generate_with_layer_filter(
    prefix_ids, method_fn, target_ids, mdl, n_new=500
):
    out = mdl(prefix_ids, use_cache=True)
    past = out.past_key_values
    if method_fn is not None and target_ids:
        info = compress_cache_targeted(
            past, method_fn,
            expected_ctx_len=prefix_ids.shape[1],
            target_layer_ids=set(target_ids),
        )
    else:
        info = {"compressed_layers": 0, "requested_layers": []}
    next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated_ids = [next_id.item()]
    for _ in range(n_new - 1):
        out = mdl(next_id, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_ids.append(next_id.item())
    return generated_ids, info


# =============================================================
# === CELL V5 === Prefix (wiki_0, same as 29d/e/f)
# =============================================================

ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
wtext = "\n\n".join([x for x in ds["text"] if x.strip()])
print(f"WikiText-2 total chars: {len(wtext)}")

PREFIX_TEXT = wtext[0:25000]
PREFIX_NAME = "wikitext_offset_0"


# =============================================================
# === CELL V6 === Anchor sets
# =============================================================

# From Phase 29f per-layer probe:
SAFE_LAYERS   = [3, 15, 35]          # delta r4 <= 0.05 individually
FRAGILE_TOP3  = [43, 39, 11]         # individually worst 3 (r4 > 0.49)
FRAGILE_ALL13 = [7, 11, 19, 23, 27, 31, 39, 43, 47, 51, 55, 59, 63]

ALL_16        = sorted(SAFE_LAYERS + FRAGILE_ALL13)
assert set(ALL_16) == {3, 7, 11, 15, 19, 23, 27, 31,
                        35, 39, 43, 47, 51, 55, 59, 63}

PROBES = [
    ("1. FP16 baseline",                           None,   []),
    ("2. {3,15,35} only (anchors)",                "asym3", SAFE_LAYERS),
    ("3. {3,15,35} + {43}",                        "asym3", SAFE_LAYERS + [43]),
    ("4. {3,15,35} + top-3 fragile",               "asym3", SAFE_LAYERS + FRAGILE_TOP3),
    ("5. {3,15,35} + all 13 fragile (all 16)",     "asym3", ALL_16),
    ("6. 13 fragile only (no anchors)",            "asym3", FRAGILE_ALL13),
]


# =============================================================
# === CELL V7 === Run the sweep
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

per_probe = {}
fp16_r4 = None

def asym3(x):
    return qa_nsep_pchan_asym(x, 3)

for label, tag, target_ids in PROBES:
    fn = asym3 if tag == "asym3" else None
    gen_ids, info = generate_with_layer_filter(
        prefix_ids, fn, target_ids, mdl, n_new=N_NEW_TOKENS
    )
    gen_text = tok.decode(gen_ids, skip_special_tokens=True)

    r4 = ngram_repetition_ratio(gen_ids, 4)
    d2 = distinct_n(gen_ids, 2)
    lrs = longest_repeated_substring(gen_text)
    if fn is None:
        fp16_r4 = r4
    delta_r4 = (r4 - fp16_r4) if fp16_r4 is not None else 0.0

    per_probe[label] = {
        "target_layer_ids":     sorted(target_ids),
        "n_target_layers":      len(target_ids),
        "longest_repeat_chars": lrs,
        "repetition_4gram":     r4,
        "delta_r4_vs_fp16":     delta_r4,
        "distinct_2":           d2,
        "tail_200_chars":       gen_text[-200:],
        "n_tokens":             len(gen_ids),
        "compression_info":     info,
    }

    print(f"  {label:<42s}  "
          f"n={len(target_ids):>2d}  "
          f"r4={r4:.3f}  Δr4={delta_r4:+.3f}  "
          f"d2={d2:.2f}  lrs={lrs:>4d}ch")


# =============================================================
# === CELL V8 === Anchor-hypothesis readout
# =============================================================

def r4(label):
    return per_probe[label]["repetition_4gram"]

def dr4(label):
    return per_probe[label]["delta_r4_vs_fp16"]

print(f"\n\n{'=' * 80}")
print(f"Anchor-layer hypothesis readout on {MODEL_LABEL}")
print(f"{'=' * 80}")
print(f"\n  FP16 baseline r4           : {fp16_r4:.3f}")
print(f"  {{3, 15, 35}} only           : {r4('2. {3,15,35} only (anchors)'):.3f}  "
      f"(Δ {dr4('2. {3,15,35} only (anchors)'):+.3f})")
print(f"  + {{43}}                     : {r4('3. {3,15,35} + {43}'):.3f}  "
      f"(Δ {dr4('3. {3,15,35} + {43}'):+.3f})")
print(f"  + top 3 fragile             : {r4('4. {3,15,35} + top-3 fragile'):.3f}  "
      f"(Δ {dr4('4. {3,15,35} + top-3 fragile'):+.3f})")
print(f"  + all 13 fragile (all 16)  : {r4('5. {3,15,35} + all 13 fragile (all 16)'):.3f}  "
      f"(Δ {dr4('5. {3,15,35} + all 13 fragile (all 16)'):+.3f})")
print(f"  13 fragile only (counter)  : {r4('6. 13 fragile only (no anchors)'):.3f}  "
      f"(Δ {dr4('6. 13 fragile only (no anchors)'):+.3f})")

anchor_only_safe = dr4("2. {3,15,35} only (anchors)") < 0.10
anchor_plus_13   = dr4("5. {3,15,35} + all 13 fragile (all 16)")
no_anchor_13     = dr4("6. 13 fragile only (no anchors)")
anchor_helps     = no_anchor_13 - anchor_plus_13  # positive = anchors help

print("\n  Hypothesis readout:")
if anchor_only_safe:
    print(f"    ✅ Anchor-only condition is safe as expected"
          f" (Δr4 = {dr4('2. {3,15,35} only (anchors)'):+.3f}).")
else:
    print(f"    ⚠️  Anchor-only condition already breaks"
          f" (Δr4 = {dr4('2. {3,15,35} only (anchors)'):+.3f})."
          f" The 'safe individually' finding from 29f did not"
          f" cleanly generalise to the 3-layer joint case.")

print(f"\n    Anchor-effect magnitude:"
      f"\n      13 fragile alone         Δr4 = {no_anchor_13:+.3f}"
      f"\n      13 fragile + 3 anchors   Δr4 = {anchor_plus_13:+.3f}"
      f"\n      difference (positive = anchors help) = {anchor_helps:+.3f}")

if anchor_helps > 0.15:
    print(f"    🏆 ANCHOR HYPOTHESIS SUPPORTED: adding the 3 safe"
          f" layers to the 13 fragile set reduces Δr4 by"
          f" {anchor_helps:.2f}. Composition seems to stabilise the"
          f" fragile set.")
elif anchor_helps > 0.05:
    print(f"    ⚖️  PARTIAL SUPPORT: anchors provide a small"
          f" stabilising effect (Δ reduction = {anchor_helps:.2f}),"
          f" but probably within run-to-run noise.")
elif anchor_helps > -0.05:
    print(f"    ❌ ANCHOR HYPOTHESIS REJECTED: adding anchors has"
          f" roughly no effect on the 13-fragile set"
          f" (Δ difference = {anchor_helps:+.2f}). The non-"
          f" additive effect seen in 29f quartile groups is not"
          f" driven by the safe layers.")
else:
    print(f"    🤔 ANTI-ANCHOR: adding the 3 safe layers makes"
          f" things WORSE (Δ = {anchor_helps:+.2f}). Unexpected"
          f" destructive interaction.")


# =============================================================
# === CELL V9 === Save
# =============================================================

output = {
    "date":           datetime.utcnow().isoformat(timespec="seconds"),
    "phase":          "29g",
    "purpose":        ("Anchor-layer hypothesis test: do the 3 "
                       "individually-safe layers {3, 15, 35} "
                       "stabilise the 13 individually-fragile "
                       "layers when compressed together?"),
    "model":          MODEL_ID,
    "label":          MODEL_LABEL,
    "protocol": {
        "ctx_target_tokens": CTX_TARGET_TOKENS,
        "n_new_tokens":      N_NEW_TOKENS,
        "decode":            "greedy",
        "prefix":            PREFIX_NAME,
        "safe_layers":       SAFE_LAYERS,
        "fragile_top3":      FRAGILE_TOP3,
        "fragile_all13":     FRAGILE_ALL13,
        "probes":            [p[0] for p in PROBES],
    },
    "fp16_r4_baseline": fp16_r4,
    "per_probe":        per_probe,
    "anchor_effect":    {
        "dr4_13_fragile_alone":        no_anchor_13,
        "dr4_13_fragile_plus_anchors": anchor_plus_13,
        "dr4_reduction_by_anchors":    anchor_helps,
    },
}

out_path = "/content/phase29g_anchor_hypothesis.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\nSaved -> {out_path}")


# =============================================================
# === READING ===
# =============================================================
#
# Probe 6 is the counterfactual that makes the story clean:
#
#   If {13 fragile only} gives Δr4 ~ 0.35 (close to all-16's
#   +0.36) and {13 fragile + 3 anchors} also gives +0.36,
#   the 3 "safe" layers contribute nothing at joint level;
#   the fragility is entirely driven by the 13.
#
#   If {13 fragile only} gives Δr4 ~ 0.50 (much worse than
#   all-16's +0.36) and {13 + anchors} gives +0.36, the
#   anchors do meaningfully stabilise the whole cache.
#
#   If the anchor-only probe (#2) already has Δr4 > 0.15,
#   then layer 3, 15, 35's "safe individually" was a
#   Phase 29f artifact and the scope caveat tightens.
#
# Either way, this single 6-run probe lets us decide whether
# the user's "anchor" intuition points to a real mitigation
# recipe or to a false pattern.
# ============================================================
