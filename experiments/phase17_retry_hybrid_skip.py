# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Phase 17 retry: hybrid-attention layers are skipped
# ============================================================
# Qwen3.5-9B uses a hybrid architecture: some layers are standard
# attention (with K/V cache), others are linear attention (with a
# recurrent state, no K/V). Our compress_cache() assumed every
# layer has .keys / .values, and crashed on the first linear-attn
# layer.
#
# Fix: skip layers that don't expose a K/V cache. The PPL we
# measure is then "the effect of compressing ALL standard-attn
# layers, with linear-attn layers untouched". For the v0.3 claim
# this is still a meaningful test -- we want to know whether asym
# INT3 beats sym INT3 on hybrid models where our method applies
# to a subset of layers.
#
# Paste this cell into the SAME Colab session that ran CELLs V1-V3
# of phase17_2026_flagships.py. The patched compress_cache shadows
# the original.
# ============================================================

import gc, torch
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()


# =============================================================
# === Patched cache compression (skip non-standard layers)
# =============================================================

def _layer_has_kv(past, li):
    """Return True if layer li exposes a K/V cache (not linear attn)."""
    if hasattr(past, 'layers'):
        layer = past.layers[li]
        return (hasattr(layer, 'keys') and hasattr(layer, 'values')
                and layer.keys is not None and layer.values is not None)
    # legacy tuple-of-tuples cache: every layer is standard attn by design
    return past[li] is not None and past[li][0] is not None


def compress_cache(past, method_fn):
    """Apply method_fn to each STANDARD-attention layer's (K, V) slice.
    Linear-attention layers (Qwen3.5 hybrid architecture) have no K/V
    cache; they are skipped silently."""
    if method_fn is None:
        return
    n_std = 0
    n_skipped = 0
    for li in range(n_cache_layers(past)):
        if not _layer_has_kv(past, li):
            n_skipped += 1
            continue
        ok, ov = get_kv(past, li)
        ok = ok.clone(); ov = ov.clone()
        nk = torch.zeros_like(ok); nv = torch.zeros_like(ov)
        for h in range(ok.shape[1]):
            nk[0, h] = method_fn(ok[0, h]).to(ok.dtype)
            nv[0, h] = method_fn(ov[0, h]).to(ov.dtype)
        set_kv(past, li, nk, nv)
        n_std += 1
    # Cache stats once per call so the user sees the hybrid breakdown.
    if not hasattr(compress_cache, "_reported"):
        print(f"    (cache has {n_std} standard-attn layers + "
              f"{n_skipped} non-KV layers)")
        compress_cache._reported = True


# Reset the one-shot reporter so it prints for each new model load.
if hasattr(compress_cache, "_reported"):
    del compress_cache._reported


# =============================================================
# === Re-run Qwen3.5-9B with the patched compress_cache
# =============================================================

qwen35_res = run_model("Qwen/Qwen3.5-9B", "Qwen3.5-9B")


# =============================================================
# === Reset reporter + run Gemma 4-E4B
# =============================================================

if hasattr(compress_cache, "_reported"):
    del compress_cache._reported

gemma4_res = run_model("google/gemma-4-E4B", "gemma-4-E4B")


# =============================================================
# === 8-model synthesis (Phase 14 + 16 + 17)
# =============================================================

import json
from datetime import datetime

PRIOR = [
    {"label": "Qwen2-7B",        "mode": "Mode 1", "baseline_ppl": 8.3693,
     "sym_int3_delta": 0.4021, "asym_int3_delta": 0.1034, "sym_int4_delta": 0.0544,
     "asym_reduction_pct": 74.3},
    {"label": "Llama-3.1-8B",    "mode": "Mode 2", "baseline_ppl": 7.3840,
     "sym_int3_delta": 0.2541, "asym_int3_delta": 0.1059, "sym_int4_delta": 0.0133,
     "asym_reduction_pct": 58.3},
    {"label": "Pythia-6.9B",     "mode": "Mode 2", "baseline_ppl": 10.7594,
     "sym_int3_delta": 2.3781, "asym_int3_delta": 0.0994, "sym_int4_delta": 0.8355,
     "asym_reduction_pct": 95.8},
    {"label": "Qwen2.5-14B",     "mode": "Mode 2", "baseline_ppl": 1.9575,
     "sym_int3_delta": 0.2608, "asym_int3_delta": 0.0909, "sym_int4_delta": 0.0274,
     "asym_reduction_pct": 65.1},
    {"label": "Mistral-7B-v0.1", "mode": "Mode 2", "baseline_ppl": 6.2983,
     "sym_int3_delta": 0.1243, "asym_int3_delta": 0.0547, "sym_int4_delta": 0.0183,
     "asym_reduction_pct": 56.0},
    {"label": "GPT-J-6B",        "mode": "Mode 2", "baseline_ppl": 10.2922,
     "sym_int3_delta": 0.1873, "asym_int3_delta": 0.0703, "sym_int4_delta": 0.0182,
     "asym_reduction_pct": 62.5},
]

all_results = list(PRIOR)
for res in [qwen35_res, gemma4_res]:
    all_results.append({
        "label":              res["label"],
        "mode":               "hybrid/2026" if "Qwen3.5" in res["label"] else "2026",
        "baseline_ppl":       res["baseline_ppl"],
        "sym_int3_delta":     res["sym_int3_delta"],
        "asym_int3_delta":    res["asym_int3_delta"],
        "sym_int4_delta":     res["sym_int4_delta"],
        "asym_reduction_pct": res["asym_reduction_pct"],
    })

print("\n" + "=" * 92)
print(f"Phase 17 + 16 + 14 synthesis "
      f"({datetime.now().isoformat(timespec='seconds')})")
print("=" * 92)
print(f"  {'Model':<22} {'mode':<12} {'base':>7} "
      f"{'sym Δ':>8} {'asym Δ':>8} {'reduction':>10} "
      f"{'sym INT4 Δ':>11} {'pct_base':>9}")
print(f"  {'-'*22} {'-'*12} {'-'*7} {'-'*8} {'-'*8} {'-'*10} {'-'*11} {'-'*9}")
for r in all_results:
    pct_base = r["asym_int3_delta"] / r["baseline_ppl"] * 100
    print(f"  {r['label']:<22} {r['mode']:<12} {r['baseline_ppl']:>7.3f} "
          f"+{r['sym_int3_delta']:>6.4f} +{r['asym_int3_delta']:>6.4f} "
          f"{r['asym_reduction_pct']:>+8.1f}%  "
          f"+{r['sym_int4_delta']:>8.4f}  {pct_base:>7.2f}%")

asym_wins = sum(1 for r in all_results if r["asym_int3_delta"] < r["sym_int3_delta"])
mean_red  = sum(r["asym_reduction_pct"] for r in all_results) / len(all_results)
min_red   = min(r["asym_reduction_pct"] for r in all_results)
max_red   = max(r["asym_reduction_pct"] for r in all_results)
asym_prod = sum(1 for r in all_results
                if r["asym_int3_delta"] / r["baseline_ppl"] < 0.05)

print(f"\n  asym beats sym:                     {asym_wins}/{len(all_results)} models")
print(f"  reduction:   mean {mean_red:+6.1f}%   min {min_red:+6.1f}%   max {max_red:+6.1f}%")
print(f"  asym INT3 under 5% of baseline:     {asym_prod}/{len(all_results)} models")

out_path = "/content/phase17_2026_flagships.json"
combined = {
    "date":         datetime.now().isoformat(timespec='seconds'),
    "prior_6_model": PRIOR,
    "phase17_runs": {
        "Qwen3.5-9B":  qwen35_res,
        "gemma-4-E4B": gemma4_res,
    },
    "eight_model_summary": all_results,
    "note":          ("Qwen3.5-9B uses hybrid attention (standard + linear). "
                      "compress_cache was patched to skip linear-attention "
                      "layers, which have no K/V cache to compress. The "
                      "reported PPL reflects compression of the standard "
                      "attention subset only."),
}
with open(out_path, "w") as f:
    json.dump(combined, f, indent=2, default=str)
print(f"\nSaved -> {out_path}")

print("\n" + "=" * 92)
print("VERDICT")
print("=" * 92)
if asym_wins == len(all_results) and min_red > 0:
    print(f"  -> UNIVERSAL: {asym_wins}/{len(all_results)} models, 2023-2026 coverage.")
    print(f"     Ship v0.3.3 with 'validated on 2026 flagships' claim.")
elif asym_wins >= len(all_results) - 1:
    print(f"  -> asym INT3 wins on {asym_wins}/{len(all_results)} models.")
else:
    print(f"  -> MIXED: asym wins on {asym_wins}/{len(all_results)}.")
