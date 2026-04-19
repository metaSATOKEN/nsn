# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Phase 23: per-token K-cache L2 norm dispersion measurement
# ============================================================
# Paper B §5.3 proposes a mechanism for why variant (B) --
# asymmetric per-channel INT3 WITHOUT norm separation -- fails
# catastrophically on Llama-family models while passing on
# Qwen3.5-9B / Gemma-4-E4B:
#
#     "Token-wise L2 norm variance contaminates per-channel
#      statistics when norm separation is omitted."
#
# In draft v0.8 we explicitly label this as a conjecture
# ("we conjecture -- but in this draft do not directly
# measure") and list "measure the per-token K-norm standard
# deviation on each of the 8 models" as future work.
#
# This phase IS that measurement. We take each of the 8 models
# in the main benchmark, run a fixed WikiText-2 prefix of
# 1024 tokens through it with ``use_cache=True``, and extract
# the per-token L2 norm distribution of the K (and V) cache
# at several layer depths. The primary statistic reported is
# the coefficient of variation
#
#     CV = std(||K_t||_2) / mean(||K_t||_2)
#
# pooled across K heads at a given layer, plus per-head CV
# and per-token-norm percentile ratios (p99 / p50) as a
# heavy-tail-robust companion. We also save the raw per-token
# norms for Layer 0 so that a histogram figure can be produced
# offline without re-running the model.
#
# Protocol per model:
#   Text:     WikiText-2 test split tokenized WITH EACH MODEL'S
#             OWN TOKENIZER, then sliced into 3 non-overlapping
#             1024-token segments at token indices
#                 [0 : 1024],
#                 [8192 : 8192+1024],
#                 [16384 : 16384+1024].
#             Token-based slicing guarantees each segment is
#             exactly PREFILL_TOKENS tokens regardless of how
#             many characters/byte-pair-units that takes; an
#             earlier character-based slicing dropped 7/8 models
#             because 4000 chars yielded <1024 tokens on most
#             tokenizers.
#   Prefill:  1024 tokens (same as the main benchmark context).
#   Layers:   L0 (the one §5.3 focuses on, based on the Phase 10
#             real-KV distribution study that localized the
#             outlier structure there) PLUS L mid and L last
#             to check whether the pattern is layer-0-specific.
#   Heads:    pool all K heads per layer; also store per-head CV.
#
# Expected outcome sketch (from the paper's hypothesis):
#   high CV (>= ~0.6) on {Llama-3.1-8B, Mistral-7B, Qwen2.5-14B}
#      -> correlates with variant (B) catastrophe (> 100% base)
#   low CV (<= ~0.4) on {Qwen3.5-9B, Gemma-4-E4B}
#      -> correlates with variant (B) stability / negative dPPL
#   GPT-J-6B / Qwen2-7B / Pythia-6.9B sit in between; the
#      intermediate prediction is less sharp, and we will read
#      whatever the data says rather than assume.
#
# If the rank-order matches the Table 2 B-failure magnitudes,
# §5.3 gets upgraded from "we conjecture -- but do not measure"
# to "a diagnostic measurement supports the hypothesis".
# If it does NOT match, §5.3 gets rewritten honestly and the
# mechanism claim is scoped to Llama-family only (or dropped).
# Either direction is informative.
#
# Runtime: ~20-30 min on Colab A100 (8 models x 3 segments x
# ~5 s per forward + model load).
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
import numpy as np
from datetime import datetime
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# =============================================================
# === CELL V2 === KV access (hybrid-attention aware)
# =============================================================

def get_kv(past, li):
    if hasattr(past, 'layers'):
        return past.layers[li].keys, past.layers[li].values
    return past[li][0], past[li][1]


def n_cache_layers(past):
    return len(past.layers) if hasattr(past, 'layers') else len(past)


def _layer_has_kv(past, li):
    """Return True if layer ``li`` exposes a K/V cache.

    Qwen3.5-9B hybrid attention mixes standard-attention layers
    (with keys/values) and linear-attention layers (recurrent
    state, no keys/values). We must skip the latter.
    """
    if hasattr(past, 'layers'):
        layer = past.layers[li]
        return (hasattr(layer, 'keys') and hasattr(layer, 'values')
                and layer.keys is not None and layer.values is not None)
    return past[li] is not None and past[li][0] is not None


def kv_layer_indices(past):
    """Return the sorted list of layer indices that carry a K/V
    cache. On dense models this is just ``range(n_cache_layers)``.
    On Qwen3.5-9B it is the 8 standard-attention layers only."""
    return [li for li in range(n_cache_layers(past)) if _layer_has_kv(past, li)]


# =============================================================
# === CELL V3 === Per-token norm statistics
# =============================================================

def norm_stats(tensor_HD_or_TD):
    """Given a 2D tensor (rows = tokens, cols = head dim), return
    per-row-L2-norm summary statistics pooled over rows.
    """
    t = tensor_HD_or_TD.float()
    norms = t.norm(dim=-1)                       # [rows]
    norms_flat = norms.reshape(-1)
    mu = norms_flat.mean().item()
    sd = norms_flat.std(unbiased=False).item()
    cv = sd / mu if mu > 1e-12 else float('nan')
    p50 = torch.quantile(norms_flat, 0.50).item()
    p99 = torch.quantile(norms_flat, 0.99).item()
    p01 = torch.quantile(norms_flat, 0.01).item()
    p99_over_p50 = p99 / p50 if p50 > 1e-12 else float('nan')
    return {
        "mean":         mu,
        "std":          sd,
        "cv":           cv,
        "p01":          p01,
        "p50":          p50,
        "p99":          p99,
        "p99_over_p50": p99_over_p50,
        "n_tokens":     int(norms_flat.shape[0]),
    }


def extract_layer_kv(past, li):
    """Extract K, V at layer ``li`` as flat [H*T, D] tensors
    (heads stacked into the token axis so per-token stats pool
    across heads within the layer).
    """
    k, v = get_kv(past, li)
    # k, v: [B=1, H, T, D]
    if k.dim() != 4:
        raise RuntimeError(f"unexpected K shape at layer {li}: {tuple(k.shape)}")
    k = k[0]                          # [H, T, D]
    v = v[0]
    # Per-head CV list
    per_head_k_cv = []
    per_head_v_cv = []
    H = k.shape[0]
    for h in range(H):
        per_head_k_cv.append(norm_stats(k[h])["cv"])
        per_head_v_cv.append(norm_stats(v[h])["cv"])
    # Pooled (H*T, D)
    k_pool = k.reshape(-1, k.shape[-1])
    v_pool = v.reshape(-1, v.shape[-1])
    return {
        "k_pooled":       norm_stats(k_pool),
        "v_pooled":       norm_stats(v_pool),
        "k_per_head_cv":  per_head_k_cv,
        "v_per_head_cv":  per_head_v_cv,
        "n_heads":        H,
        "D":              int(k.shape[-1]),
    }


# =============================================================
# === CELL V4 === WikiText-2 text source + segment indices
# =============================================================
#
# NOTE ON SLICING:
# We previously used character-based slicing (4000-char windows
# from the raw text) and got < 1024 tokens on 7/8 models --
# most tokenizers here average ~4 chars/token and individual
# segments dipped to ~900. This version tokenizes per model with
# each model's own tokenizer and slices by TOKEN index.

PREFILL_TOKENS = 1024

# Three non-overlapping token-index windows. Start indices are
# spaced by 8192 so we land well clear of chapter/section
# boundary artifacts in WikiText-2 test.
SEGMENT_TOKEN_STARTS = [0, 8192, 16384]

ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
wtext_full = "\n\n".join([x for x in ds["text"] if x.strip()])
print(f"WikiText-2 full text: {len(wtext_full)} chars; "
      f"will tokenize per model and slice 3 x {PREFILL_TOKENS} "
      f"token windows starting at {SEGMENT_TOKEN_STARTS}.")


# =============================================================
# === CELL V5 === Single-model measurement
# =============================================================

def measure_model(model_id, label, extra_kwargs=None):
    """Load model, run 3 segments through prefill, extract per-
    token K/V norm stats at L0 / L mid / L last. Returns a dict.
    """
    global mdl, tok
    try:
        del mdl, tok
    except NameError:
        pass
    gc.collect(); torch.cuda.empty_cache()

    print(f"\n{'='*80}")
    print(f"Loading {label}  ({model_id})")
    print(f"{'='*80}")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    kwargs = dict(
        torch_dtype=torch.float16,
        device_map="auto",
        use_safetensors=True,
        trust_remote_code=True,
    )
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    mdl = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    mdl.eval()
    print(f"VRAM used: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # Tokenize the full WikiText-2 text with this model's
    # tokenizer once; slice by token index below.
    full_ids = tok.encode(wtext_full, return_tensors="pt")[0]
    T_total = full_ids.shape[0]
    need = SEGMENT_TOKEN_STARTS[-1] + PREFILL_TOKENS
    if T_total < need:
        raise RuntimeError(
            f"{label}: tokenized WikiText-2 test is only {T_total} "
            f"tokens, need >= {need} for the 3-segment protocol."
        )
    print(f"  tokenized text length: {T_total} tokens "
          f"(need {need} for 3 segments)")

    per_segment = []
    for seg_idx, t0 in enumerate(SEGMENT_TOKEN_STARTS):
        ids = full_ids[t0 : t0 + PREFILL_TOKENS].unsqueeze(0).to("cuda")

        with torch.inference_mode():
            out = mdl(ids, use_cache=True)
            past = out.past_key_values

        kv_indices = kv_layer_indices(past)
        n_kv_layers = len(kv_indices)
        n_all_layers = n_cache_layers(past)
        sampled_layers = {
            "L0":    kv_indices[0],
            "Lmid":  kv_indices[n_kv_layers // 2],
            "Llast": kv_indices[-1],
        }

        layer_stats = {}
        raw_k_norms_L0 = None
        for tag, li in sampled_layers.items():
            stats = extract_layer_kv(past, li)
            stats["layer_index"] = li
            layer_stats[tag] = stats
            if tag == "L0":
                k, _ = get_kv(past, li)
                raw_k_norms_L0 = k[0].float().norm(dim=-1).cpu().numpy()  # [H, T]

        seg_record = {
            "segment_idx":       seg_idx,
            "token_start":       t0,
            "prefill_tokens":    PREFILL_TOKENS,
            "n_kv_layers":       n_kv_layers,
            "n_all_layers":      n_all_layers,
            "sampled_layers":    sampled_layers,
            "layer_stats":       layer_stats,
            "raw_L0_k_norms":    raw_k_norms_L0.tolist(),  # [H][T]
        }
        per_segment.append(seg_record)

        l0_k_cv = layer_stats["L0"]["k_pooled"]["cv"]
        l0_v_cv = layer_stats["L0"]["v_pooled"]["cv"]
        print(f"  seg {seg_idx} (t={t0}): L0 K-CV = {l0_k_cv:.3f}, "
              f"V-CV = {l0_v_cv:.3f}, KV layers = {n_kv_layers}/{n_all_layers}")

        del past, out
        gc.collect(); torch.cuda.empty_cache()

    def agg(key_chain):
        vals = []
        for seg in per_segment:
            x = seg["layer_stats"]
            for k in key_chain:
                x = x[k]
            vals.append(x)
        return {
            "values": vals,
            "mean":   float(np.mean(vals)),
            "std":    float(np.std(vals, ddof=0)),
        }

    summary = {
        "L0_k_cv":           agg(["L0", "k_pooled", "cv"]),
        "L0_v_cv":           agg(["L0", "v_pooled", "cv"]),
        "Lmid_k_cv":         agg(["Lmid", "k_pooled", "cv"]),
        "Llast_k_cv":        agg(["Llast", "k_pooled", "cv"]),
        "L0_k_p99_over_p50": agg(["L0", "k_pooled", "p99_over_p50"]),
    }

    print(f"  SUMMARY {label}:")
    print(f"    L0 K-CV            mean = {summary['L0_k_cv']['mean']:.3f} "
          f"(+/- {summary['L0_k_cv']['std']:.3f} across 3 segments)")
    print(f"    L0 V-CV            mean = {summary['L0_v_cv']['mean']:.3f} "
          f"(+/- {summary['L0_v_cv']['std']:.3f})")
    print(f"    L0 K p99/p50       mean = {summary['L0_k_p99_over_p50']['mean']:.3f} "
          f"(+/- {summary['L0_k_p99_over_p50']['std']:.3f})")
    print(f"    Lmid K-CV          mean = {summary['Lmid_k_cv']['mean']:.3f}")
    print(f"    Llast K-CV         mean = {summary['Llast_k_cv']['mean']:.3f}")

    return {
        "label":        label,
        "model":        model_id,
        "per_segment":  per_segment,
        "summary":      summary,
    }


# =============================================================
# === CELL V6 === Run all 8 models
# =============================================================
# Ordered (roughly) by (B) regression magnitude in Table 2, so
# the hypothesis check is visible early in the log.

results = {}

# Catastrophic B-failures (hypothesis: high CV)
results["Llama-3.1-8B"] = measure_model(
    "meta-llama/Llama-3.1-8B", "Llama-3.1-8B")

results["Mistral-7B-v0.1"] = measure_model(
    "mistralai/Mistral-7B-v0.1", "Mistral-7B-v0.1")

results["Qwen2.5-14B"] = measure_model(
    "Qwen/Qwen2.5-14B", "Qwen2.5-14B")

# Mild / OK (hypothesis: medium CV)
results["GPT-J-6B"] = measure_model(
    "EleutherAI/gpt-j-6b", "GPT-J-6B")

results["Qwen2-7B"] = measure_model(
    "Qwen/Qwen2-7B", "Qwen2-7B")

results["Pythia-6.9B"] = measure_model(
    "EleutherAI/pythia-6.9b", "Pythia-6.9B")

# B-negative (hypothesis: low CV)
results["Qwen3.5-9B"] = measure_model(
    "Qwen/Qwen3.5-9B", "Qwen3.5-9B")

results["Gemma-4-E4B"] = measure_model(
    "google/gemma-4-E4B", "Gemma-4-E4B")


# =============================================================
# === CELL V7 === Summary table + JSON save
# =============================================================

B_VARIANT_PCT_OF_BASE = {
    "GPT-J-6B":         6.27,
    "Pythia-6.9B":      5.88,
    "Mistral-7B-v0.1":  119.72,
    "Qwen2-7B":         4.29,
    "Qwen2.5-14B":      145.67,
    "Llama-3.1-8B":     147.91,
    "Qwen3.5-9B":       -0.04,
    "Gemma-4-E4B":      -2.13,
}

B_CATEGORY = {
    "GPT-J-6B":         "mild (threshold violation)",      # 6.27%  > 5% threshold
    "Pythia-6.9B":      "mild (threshold violation)",      # 5.88%  > 5% threshold
    "Mistral-7B-v0.1":  "catastrophic (>100% base)",
    "Qwen2-7B":         "OK (below threshold)",            # 4.29% < 5%
    "Qwen2.5-14B":      "catastrophic (>100% base)",
    "Llama-3.1-8B":     "catastrophic (>100% base)",
    "Qwen3.5-9B":       "negative dPPL (B-anomaly class)",
    "Gemma-4-E4B":      "negative dPPL (B-anomaly class)",
}

print("\n" + "=" * 96)
print(f"Phase 23 norm-dispersion summary "
      f"({datetime.now().isoformat(timespec='seconds')})")
print("=" * 96)
print(f"  {'Model':<18} {'L0 K-CV':>10} {'+/- SD':>8} {'L0 V-CV':>10} "
      f"{'p99/p50':>9} {'(B) %base':>10} {'(B) class':<28}")
print(f"  {'-'*18} {'-'*10} {'-'*8} {'-'*10} {'-'*9} {'-'*10} {'-'*28}")

rows_for_json = []
for model_key in [
    "Llama-3.1-8B", "Qwen2.5-14B", "Mistral-7B-v0.1",       # catastrophic
    "GPT-J-6B",                                             # mild
    "Qwen2-7B", "Pythia-6.9B",                              # OK
    "Qwen3.5-9B", "Gemma-4-E4B",                            # B-negative
]:
    r = results[model_key]["summary"]
    k_cv_mu = r["L0_k_cv"]["mean"]
    k_cv_sd = r["L0_k_cv"]["std"]
    v_cv_mu = r["L0_v_cv"]["mean"]
    k_p = r["L0_k_p99_over_p50"]["mean"]
    b_pct = B_VARIANT_PCT_OF_BASE[model_key]
    b_cat = B_CATEGORY[model_key]
    print(f"  {model_key:<18} {k_cv_mu:>10.3f} {k_cv_sd:>8.3f} {v_cv_mu:>10.3f} "
          f"{k_p:>9.3f} {b_pct:>9.2f}% {b_cat:<28}")
    rows_for_json.append({
        "model":                 model_key,
        "L0_K_CV_mean":          k_cv_mu,
        "L0_K_CV_std":           k_cv_sd,
        "L0_V_CV_mean":          v_cv_mu,
        "L0_K_p99_over_p50":     k_p,
        "B_variant_pct_of_base": b_pct,
        "B_variant_category":    b_cat,
    })


out_path = "/content/phase23_norm_dispersion.json"
combined = {
    "date":     datetime.now().isoformat(timespec='seconds'),
    "protocol": {
        "text_source":       "WikiText-2 test split",
        "segment_count":     len(SEGMENT_TOKEN_STARTS),
        "segment_token_starts": SEGMENT_TOKEN_STARTS,
        "prefill_tokens":    PREFILL_TOKENS,
        "slicing":           "per-model tokenizer, token-index slicing (not char)",
        "layers_sampled":    ["L0 (first KV)", "L mid", "L last"],
        "statistic":         "per-token L2 norm; CV = std/mean pooled across K heads",
        "companion_stat":    "p99/p50 norm ratio (heavy-tail robust)",
        "notes":             (
            "per-head CV also recorded. Raw L0 K per-token norms "
            "saved for histogram plotting. V-cache stats reported "
            "as companion; §5.3 mechanism focuses on K since Phase 21 "
            "showed Gemma B-anomaly is 87% K-driven."
        ),
    },
    "ablation_B_regressions_paper_table2": B_VARIANT_PCT_OF_BASE,
    "ablation_B_categories":               B_CATEGORY,
    "summary_table":                       rows_for_json,
    "per_model":                           results,
}
with open(out_path, "w") as f:
    json.dump(combined, f, indent=2, default=str)
print(f"\nSaved -> {out_path}")


print("\n" + "=" * 96)
print("READING")
print("=" * 96)
print("""
  HYPOTHESIS (Paper B v0.8 §5.3):
      Models where variant (B) catastrophically regresses will
      show substantially higher L0 K-CV than models where (B)
      stays within threshold or produces negative dPPL.

  WHAT TO LOOK FOR:
    1. Rank-order of L0 K-CV vs (B) regression magnitude.
       Ideally Spearman correlation > ~0.6 with the ordering
            catastrophic > mild > OK > B-negative.

    2. The SD across 3 WikiText-2 segments. If SD is small
       compared to the inter-model gap, the diagnostic is
       stable; if it is the same order of magnitude, we
       under-report the reliability.

    3. L mid / L last K-CV. If the L0 pattern is mirrored
       deeper in the network, the mechanism is general;
       if only L0 shows it, §5.3 needs a layer-0 qualifier.

  OUTCOMES:
    match  : §5.3 goes from "we conjecture -- but do not measure"
             to "a diagnostic measurement supports this hypothesis";
             Appendix E added with the full table + histogram.
    partial: §5.3 scoped to Llama-family, other models listed
             as unexplained.
    miss   : §5.3 rewritten; the mechanism claim is withdrawn
             or replaced by the observed correlation, whatever
             it is.
""")
