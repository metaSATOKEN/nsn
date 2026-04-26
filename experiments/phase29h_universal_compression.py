# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Phase 29h: Two-track universal compression on Qwen3.6-27B
#            -- attention K/V via modeq, SSM state via naive
#            uniform quantizer. Tests the hypothesis that
#            "attention-only compression creates an asymmetric
#            noise distribution; uniformly compressing both
#            attention and SSM should be more stable".
# ============================================================
# EXPLORATORY / sandbox. Not for the NSN paper.
#
# Phase 29g found a W-shaped fragility-vs-density curve on
# Qwen3.6-27B's 16 transformer-attention layers: compressing
# 1 layer is catastrophic (r4 = 0.55), 6 layers is semi-ok
# (r4 = 0.29), and 13+ layers breaks again. The "single noisy
# layer in a sea of clean layers" was the worst regime, which
# suggested an unobvious extension: maybe the imbalance between
# "attention layers compressed" and "SSM layers untouched"
# is itself part of the fragility.
#
# Phase 29h tests this directly by adding a second compression
# track for the SSM-like layers (the 48 of 64 cache layers
# that the shape guard skips). Two questions:
#
#   Q1. What does the SSM cache actually look like? (shape
#       diagnostic on the first SSM layer)
#
#   Q2. Does adding naive uniform quantization to SSM state
#       (matching modeq's attention compression) reduce the
#       attention-only fragility, or compound it?
#
# Conditions on wiki_0 prefix:
#
#   1. FP16                      reference
#   2. attn modeq INT3           (Phase 29b reference; r4 ~ 0.52)
#   3. ssm  naive INT8           (control: SSM-only perturbation)
#   4. ssm  naive INT4           (control: stronger SSM-only)
#   5. attn modeq INT3 + ssm naive INT8     <-- hypothesis main test
#   6. attn modeq INT3 + ssm naive INT4     <-- stronger uniform noise
#
# Predictions:
#
#   - If (5) gives r4 << 0.52 (much better than condition 2):
#     hypothesis SUPPORTED. Adding SSM noise dilutes the
#     attention-only anomaly the way Phase 29g's mid-density
#     attention conditions did.
#
#   - If (5) gives r4 ~ 0.52: SSM noise is irrelevant; the
#     attention fragility stands alone.
#
#   - If (5) gives r4 > 0.52: SSM noise compounds the damage.
#     Hypothesis REJECTED.
#
#   - Condition 3 / 4 are useful even if 5 is inconclusive:
#     they tell us the standalone SSM tolerance.
#
# Runtime ~15 min on A100 80 GB (6 generations of 500 tokens).
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
# === CELL V2 === Quantizers
# =============================================================

def qa_naive_perrow_sym(x, bits):
    """Per-row symmetric absmax. Operates on the last dim, so it
    handles any ndim >= 1 tensor uniformly. Used here for the
    SSM-state track; not the recommended SSM recipe (Quamba would
    be), but it is the simplest 'apply-anywhere' uniform quantizer
    for the noise-uniformity hypothesis test."""
    dtype = x.dtype
    xf = x.float()
    qm = 2 ** (bits - 1) - 1
    s = xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / qm
    q = (xf / s).round().clamp(-qm, qm) * s
    return q.to(dtype)


def qa_nsep_pchan_asym(x, bits):
    """v0.3 shipped recipe. Used here on the attention K/V track
    (operates on a [T, D] head slice; not appropriate for SSM
    state which has different axis semantics)."""
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
# === CELL V3 === KV cache plumbing with two-track compression
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


_SSM_DIAGNOSTIC_PRINTED = False

def compress_cache_universal(past, attn_method, ssm_method,
                              expected_ctx_len):
    """
    Two-track compression on a hybrid cache:
      - Layers passing the transformer K/V shape guard ->
        attn_method (typically modeq asym INT3 on a [T, D] slice
        per head; iterated with the per-head loop our other
        scripts use).
      - All other layers (SSM-like; shape != [B, H, T, D] or
        K/V mismatch) -> ssm_method, applied as a single tensor
        operation on the whole K and V (naive sym quantizer
        which handles arbitrary ndim through per-last-dim
        absmax).

    Either method may be None to leave that track alone.
    """
    global _SSM_DIAGNOSTIC_PRINTED
    total = n_cache_layers(past)
    transformer_compressed = []
    ssm_compressed         = []
    ssm_skipped_no_kv      = []
    ssm_diagnostic         = None

    for li in range(total):
        if not _layer_has_kv(past, li):
            ssm_skipped_no_kv.append(li)
            continue
        ok, ov = get_kv(past, li)
        is_attn = _layer_is_transformer_kv(past, li, expected_ctx_len)

        if is_attn:
            if attn_method is not None:
                ok2 = ok.clone(); ov2 = ov.clone()
                nk = torch.zeros_like(ok2); nv = torch.zeros_like(ov2)
                for h in range(ok2.shape[1]):
                    nk[0, h] = attn_method(ok2[0, h]).to(ok2.dtype)
                    nv[0, h] = attn_method(ov2[0, h]).to(ov2.dtype)
                set_kv(past, li, nk, nv)
                transformer_compressed.append(li)
        else:
            # First-time diagnostic on SSM-like cache shape
            if ssm_diagnostic is None:
                ssm_diagnostic = {
                    "first_ssm_layer": li,
                    "k_shape": list(ok.shape) if torch.is_tensor(ok) else None,
                    "k_dtype": str(ok.dtype) if torch.is_tensor(ok) else None,
                    "v_shape": list(ov.shape) if torch.is_tensor(ov) else None,
                    "v_dtype": str(ov.dtype) if torch.is_tensor(ov) else None,
                }
                if not _SSM_DIAGNOSTIC_PRINTED:
                    print(f"\n  [SSM cache diagnostic, first SSM layer = "
                          f"{li}]")
                    print(f"    K: shape={ssm_diagnostic['k_shape']}  "
                          f"dtype={ssm_diagnostic['k_dtype']}")
                    print(f"    V: shape={ssm_diagnostic['v_shape']}  "
                          f"dtype={ssm_diagnostic['v_dtype']}")
                    _SSM_DIAGNOSTIC_PRINTED = True
            if ssm_method is not None and torch.is_tensor(ok) \
                    and torch.is_tensor(ov):
                try:
                    nk = ssm_method(ok).to(ok.dtype)
                    nv = ssm_method(ov).to(ov.dtype)
                    set_kv(past, li, nk, nv)
                    ssm_compressed.append(li)
                except Exception as e:
                    # If the SSM tensor shape breaks our quantizer,
                    # log and skip rather than crash the whole sweep.
                    print(f"    [warn] ssm_method raised on layer "
                          f"{li} (shape={list(ok.shape)}): {e}")

    return {
        "total_layers":            total,
        "transformer_compressed":  transformer_compressed,
        "n_transformer_compressed": len(transformer_compressed),
        "ssm_compressed":          ssm_compressed,
        "n_ssm_compressed":        len(ssm_compressed),
        "ssm_skipped_no_kv":       ssm_skipped_no_kv,
        "ssm_diagnostic":          ssm_diagnostic,
    }


# =============================================================
# === CELL V4 === Loop / repetition metrics (same as 29 series)
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
def generate_with_two_tracks(
    prefix_ids, attn_method, ssm_method, mdl, n_new=500
):
    out = mdl(prefix_ids, use_cache=True)
    past = out.past_key_values
    info = compress_cache_universal(
        past, attn_method, ssm_method,
        expected_ctx_len=prefix_ids.shape[1],
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
# === CELL V6 === Prefix
# =============================================================

ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
wtext = "\n\n".join([x for x in ds["text"] if x.strip()])
print(f"WikiText-2 total chars: {len(wtext)}")

PREFIX_TEXT = wtext[0:25000]
PREFIX_NAME = "wikitext_offset_0"


# =============================================================
# === CELL V7 === Run the sweep
# =============================================================

MODEL_ID    = "Qwen/Qwen3.6-27B"
MODEL_LABEL = "Qwen3.6-27B"

CTX_TARGET_TOKENS = 4096
N_NEW_TOKENS      = 500

# (label, attn_method, ssm_method)
PROBES = [
    ("1. FP16 baseline",
     None,
     None),
    ("2. attn modeq INT3 only (29b ref)",
     lambda x: qa_nsep_pchan_asym(x, 3),
     None),
    ("3. ssm naive INT8 only (control)",
     None,
     lambda x: qa_naive_perrow_sym(x, 8)),
    ("4. ssm naive INT4 only (control)",
     None,
     lambda x: qa_naive_perrow_sym(x, 4)),
    ("5. attn modeq INT3 + ssm naive INT8",
     lambda x: qa_nsep_pchan_asym(x, 3),
     lambda x: qa_naive_perrow_sym(x, 8)),
    ("6. attn modeq INT3 + ssm naive INT4",
     lambda x: qa_nsep_pchan_asym(x, 3),
     lambda x: qa_naive_perrow_sym(x, 4)),
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
print(f"\nPrefix: {PREFIX_NAME}  (ctx = {actual_ctx} tokens)")

per_probe = {}
fp16_r4 = None

for label, attn_fn, ssm_fn in PROBES:
    gen_ids, info = generate_with_two_tracks(
        prefix_ids, attn_fn, ssm_fn, mdl, n_new=N_NEW_TOKENS
    )
    gen_text = tok.decode(gen_ids, skip_special_tokens=True)

    r4 = ngram_repetition_ratio(gen_ids, 4)
    d2 = distinct_n(gen_ids, 2)
    lrs = longest_repeated_substring(gen_text)
    if attn_fn is None and ssm_fn is None:
        fp16_r4 = r4
    delta_r4 = r4 - fp16_r4 if fp16_r4 is not None else 0.0

    per_probe[label] = {
        "longest_repeat_chars": lrs,
        "repetition_4gram":     r4,
        "delta_r4_vs_fp16":     delta_r4,
        "distinct_2":           d2,
        "tail_200_chars":       gen_text[-200:],
        "n_tokens":             len(gen_ids),
        "compression_info":     {
            "n_transformer_compressed":
                info["n_transformer_compressed"],
            "n_ssm_compressed":
                info["n_ssm_compressed"],
            "ssm_diagnostic":
                info["ssm_diagnostic"],
        },
    }

    tag = (f"[attn {info['n_transformer_compressed']}/"
           f"16, ssm {info['n_ssm_compressed']}/48]")
    print(f"  {label:<40s}  r4={r4:.3f}  Δr4={delta_r4:+.3f}  "
          f"d2={d2:.2f}  lrs={lrs:>4d}ch  {tag}")


# =============================================================
# === CELL V8 === Hypothesis readout
# =============================================================

print(f"\n{'=' * 80}")
print(f"Universal-compression hypothesis readout on {MODEL_LABEL}")
print(f"{'=' * 80}")
print(f"  FP16 baseline r4 = {fp16_r4:.3f}")

attn_only      = per_probe["2. attn modeq INT3 only (29b ref)"]["repetition_4gram"]
ssm_only_int8  = per_probe["3. ssm naive INT8 only (control)"]["repetition_4gram"]
ssm_only_int4  = per_probe["4. ssm naive INT4 only (control)"]["repetition_4gram"]
both_int8      = per_probe["5. attn modeq INT3 + ssm naive INT8"]["repetition_4gram"]
both_int4      = per_probe["6. attn modeq INT3 + ssm naive INT4"]["repetition_4gram"]

print(f"\n  Reference rows:")
print(f"    attn modeq INT3 only             r4 = {attn_only:.3f}")
print(f"    ssm  naive INT8 only             r4 = {ssm_only_int8:.3f}")
print(f"    ssm  naive INT4 only             r4 = {ssm_only_int4:.3f}")
print(f"  Hypothesis test rows:")
print(f"    attn INT3 + ssm INT8             r4 = {both_int8:.3f}")
print(f"    attn INT3 + ssm INT4             r4 = {both_int4:.3f}")

# Hypothesis evaluation
delta_int8 = both_int8 - attn_only  # negative = adding SSM noise HELPS
delta_int4 = both_int4 - attn_only

print(f"\n  Δr4 of (attn+ssm) vs (attn only):")
print(f"    + ssm INT8: {delta_int8:+.3f}  "
      f"({'adds SSM noise HELPS' if delta_int8 < -0.05 else 'no clear effect' if abs(delta_int8) <= 0.05 else 'adds SSM noise HURTS'})")
print(f"    + ssm INT4: {delta_int4:+.3f}  "
      f"({'adds SSM noise HELPS' if delta_int4 < -0.05 else 'no clear effect' if abs(delta_int4) <= 0.05 else 'adds SSM noise HURTS'})")

print(f"\n  Verdict:")
if delta_int8 < -0.10:
    print(f"    🏆 HYPOTHESIS SUPPORTED: adding SSM noise (INT8) "
          f"reduces attention-only fragility by {-delta_int8:.2f}.")
    print(f"    Uniform two-track compression dilutes the attention-"
          f"only anomaly the way Phase 29g's mid-density attention "
          f"conditions did. Compression of both tracks may be a viable "
          f"hybrid recipe direction (with proper SSM-aware quantizers, "
          f"not the naive sym we used here).")
elif abs(delta_int8) <= 0.10 and abs(delta_int4) <= 0.10:
    print(f"    ⚖️  HYPOTHESIS NEUTRAL: SSM noise neither helps nor "
          f"hurts the attention-only fragility on this prefix. "
          f"Attention-side fragility stands alone.")
else:
    print(f"    ❌ HYPOTHESIS REJECTED: adding SSM noise compounds "
          f"the damage instead of diluting it. SSM tolerance is "
          f"tighter than attention; uniform compression is not the "
          f"right strategy for this architecture.")

if max(ssm_only_int8 - fp16_r4, ssm_only_int4 - fp16_r4) > 0.15:
    print(f"\n    Aside: SSM-only naive quantization breaks the model "
          f"on its own (INT8 Δr4={ssm_only_int8-fp16_r4:+.2f}, "
          f"INT4 Δr4={ssm_only_int4-fp16_r4:+.2f}). SSM tolerance "
          f"appears tighter than attention's ~0.2% from Phase 29e -- "
          f"any naive sym on SSM state is above its threshold. This "
          f"is consistent with the literature on Quamba (SSM-specific "
          f"recipes via Hadamard smoothing; arXiv:2410.13229).")


# =============================================================
# === CELL V9 === Save
# =============================================================

output = {
    "date":           datetime.utcnow().isoformat(timespec="seconds"),
    "phase":          "29h",
    "purpose":        ("Two-track compression: attn modeq INT3 + "
                       "ssm naive sym INT8/INT4 on Qwen3.6-27B. "
                       "Tests whether uniformly perturbing both "
                       "attention and SSM dilutes the attention-"
                       "only fragility observed in Phase 29b/g."),
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

out_path = "/content/phase29h_universal_compression.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\nSaved -> {out_path}")


# =============================================================
# === READING ===
# =============================================================
#
# Things this run can tell us:
#
#   (a) The shape and dtype of the SSM cache tensors. We have
#       not actually inspected these directly in any earlier
#       Phase 29 run; the diagnostic print on the first SSM-
#       compressed run will resolve it. Useful for future hybrid
#       work even if the hypothesis test is inconclusive.
#
#   (b) Whether SSM state is independently fragile to naive
#       sym quantization. If condition 3 (ssm INT8 only) breaks
#       the model with FP16 attention, we have a direct lower
#       bound on the SSM tolerance and can compare it to the
#       attention threshold from Phase 29e (~0.2% per-token-L2).
#
#   (c) Whether two-track uniform compression is a viable hybrid
#       direction at all. If condition 5 reduces r4 well below
#       attention-only (condition 2 ~ 0.52), it strengthens the
#       Phase 29g dilution-curve interpretation and points at a
#       concrete recipe space (modeq attn + Quamba-style SSM)
#       that could be worth a real implementation.
# ============================================================
