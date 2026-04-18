# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Phase 16 retry: Phi-3-medium with rope_scaling config patch
# ============================================================
# Paste this cell into the SAME Colab session where CELLs V1-V8 of
# phase16_asym_seven_model_sweep.py ran. It assumes the following
# are still in scope:
#
#   run_model, ppl_with, compress_cache         (CELL V3)
#   qa_nsep_pchan_sym, qa_nsep_pchan_asym       (CELL V2)
#   METHODS                                     (CELL V3)
#   pythia_res, qwen25_res, mistral_res,
#   gptj_res                                    (CELLs V4-V7)
#
# If the kernel was restarted, re-run CELLs V1-V7 first, or use the
# hardcoded-prior version at the bottom of this file instead.
#
# The failure that motivates this retry:
#   Phi-3-medium-4k-instruct's bundled modeling_phi3.py looks for
#   config.rope_scaling["type"], but recent transformers versions
#   renamed that key to "rope_type" when loading the config. We
#   manually backfill "type" before handing the config to the
#   model constructor.
# ============================================================


# =============================================================
# === Retry CELL === Phi-3-medium with rope_scaling patch
# =============================================================

import gc, torch
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
try:
    del mdl, tok
except NameError:
    pass

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

MODEL_ID = "microsoft/Phi-3-medium-4k-instruct"
LABEL    = "Phi-3-medium-4k-instruct"

# ---- Load & patch config before constructing the model ----
cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
if cfg.rope_scaling is not None and "type" not in cfg.rope_scaling:
    # transformers 4.45+ renamed the key to `rope_type`; Phi-3's
    # custom modeling_phi3.py still expects `type`. Backfill.
    cfg.rope_scaling["type"] = cfg.rope_scaling.get("rope_type", "longrope")
    print(f"  Patched rope_scaling['type'] = {cfg.rope_scaling['type']!r}")

print(f"\n{'='*80}")
print(f"Loading {LABEL} ({MODEL_ID}) ...")
print(f"{'='*80}")
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
mdl = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    config=cfg,
    torch_dtype=torch.float16,
    device_map="auto",
    use_safetensors=True,
    trust_remote_code=True,
)
mdl.eval()
print(f"VRAM used: {torch.cuda.memory_allocated()/1e9:.1f} GB")


# ---- Run the four methods (re-uses functions already in scope) ----
print(f"\n  {'Method':<32} | {'bits':>5} | {'PPL':>10} | {'ΔPPL':>10}")
print(f"  " + "-"*64)

results = {}
baseline_ppl = None
for name, fn, bits in METHODS:
    print(f"  Running {name} ...", flush=True)
    ppl, nt = ppl_with(fn)
    if baseline_ppl is None:
        baseline_ppl = ppl
    delta = ppl - baseline_ppl
    results[name] = {"ppl": ppl, "bits": bits, "delta": delta, "tokens": nt}
    dstr = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
    print(f"  {name:<32} | {bits:>5} | {ppl:>10.4f} | {dstr:>10}")

sym_d  = results["nsep+pchan INT3 symmetric"]["delta"]
asym_d = results["nsep+pchan INT3 asymmetric"]["delta"]
int4_d = results["nsep+pchan INT4 symmetric"]["delta"]
reduction_pct = (1 - asym_d / max(sym_d, 1e-9)) * 100 if sym_d > 0 else 0

print(f"\n  sym INT3  -> asym INT3: {sym_d:.4f} -> {asym_d:.4f} "
      f"({reduction_pct:+.1f}% reduction)")
print(f"  sym INT4 reference: {int4_d:.4f}")

phi3_res = {
    "label":        LABEL,
    "model":        MODEL_ID,
    "baseline_ppl": baseline_ppl,
    "results":      results,
    "sym_int3_delta":     sym_d,
    "asym_int3_delta":    asym_d,
    "sym_int4_delta":     int4_d,
    "asym_reduction_pct": reduction_pct,
}


# =============================================================
# === CELL V9 (re-run) === 7-model synthesis with Phi-3
# =============================================================

import json
from datetime import datetime

PRIOR = [
    {"label": "Qwen2-7B",        "mode": "Mode 1",
     "baseline_ppl": 8.3693,
     "sym_int3_delta": 0.4021, "asym_int3_delta": 0.1034,
     "sym_int4_delta": 0.0544,
     "asym_reduction_pct": 74.3},
    {"label": "Llama-3.1-8B",    "mode": "Mode 2",
     "baseline_ppl": 7.3840,
     "sym_int3_delta": 0.2541, "asym_int3_delta": 0.1059,
     "sym_int4_delta": 0.0133,
     "asym_reduction_pct": 58.3},
]

all_results = list(PRIOR)
for res in [pythia_res, qwen25_res, mistral_res, gptj_res, phi3_res]:
    all_results.append({
        "label":              res["label"],
        "mode":               "Mode 2",
        "baseline_ppl":       res["baseline_ppl"],
        "sym_int3_delta":     res["sym_int3_delta"],
        "asym_int3_delta":    res["asym_int3_delta"],
        "sym_int4_delta":     res["sym_int4_delta"],
        "asym_reduction_pct": res["asym_reduction_pct"],
    })

print("\n" + "=" * 88)
print(f"Phase 16 + 14 synthesis ({datetime.now().isoformat(timespec='seconds')})")
print("=" * 88)
print(f"  {'Model':<22} {'mode':<7} {'base':>7} "
      f"{'sym Δ':>8} {'asym Δ':>8} {'reduction':>10} "
      f"{'sym INT4 Δ':>11}")
print(f"  {'-'*22} {'-'*7} {'-'*7} {'-'*8} {'-'*8} {'-'*10} {'-'*11}")
for r in all_results:
    print(f"  {r['label']:<22} {r['mode']:<7} {r['baseline_ppl']:>7.3f} "
          f"+{r['sym_int3_delta']:>6.4f} +{r['asym_int3_delta']:>6.4f} "
          f"{r['asym_reduction_pct']:>+8.1f}%  "
          f"+{r['sym_int4_delta']:>8.4f}")

asym_wins = sum(1 for r in all_results if r["asym_int3_delta"] < r["sym_int3_delta"])
mean_red  = sum(r["asym_reduction_pct"] for r in all_results) / len(all_results)
min_red   = min(r["asym_reduction_pct"] for r in all_results)
max_red   = max(r["asym_reduction_pct"] for r in all_results)
asym_prod = sum(
    1 for r in all_results
    if r["asym_int3_delta"] / r["baseline_ppl"] < 0.05
)

print(f"\n  asym beats sym:                     {asym_wins}/{len(all_results)} models")
print(f"  reduction:   mean {mean_red:+6.1f}%   min {min_red:+6.1f}%   max {max_red:+6.1f}%")
print(f"  asym INT3 under 5% of baseline:     {asym_prod}/{len(all_results)} models")

out_path = "/content/phase16_asym_seven_model_sweep.json"
combined = {
    "date":          datetime.now().isoformat(timespec='seconds'),
    "prior_phase14": PRIOR,
    "phase16_runs":  {
        "Pythia-6.9B":              pythia_res,
        "Qwen2.5-14B":              qwen25_res,
        "Mistral-7B-v0.1":          mistral_res,
        "GPT-J-6B":                 gptj_res,
        "Phi-3-medium-4k-instruct": phi3_res,
    },
    "seven_model_summary": all_results,
}
with open(out_path, "w") as f:
    json.dump(combined, f, indent=2, default=str)
print(f"\nSaved -> {out_path}")

print("\n" + "=" * 88)
print("VERDICT")
print("=" * 88)
if asym_wins == len(all_results) and min_red > 0:
    print(f"  -> v0.3 asym INT3 GENERALIZES UNIVERSALLY on our 7-model set.")
    print(f"     Release v0.3.2 with '7-model validated' claim.")
elif asym_wins >= len(all_results) - 1:
    print(f"  -> v0.3 asym INT3 generalizes to {asym_wins}/{len(all_results)} models.")
    print(f"     Investigate the outlier case before claiming universality.")
else:
    print(f"  -> MIXED: asym wins on {asym_wins}/{len(all_results)}.")
    print(f"     The generalization claim needs a caveat table.")
