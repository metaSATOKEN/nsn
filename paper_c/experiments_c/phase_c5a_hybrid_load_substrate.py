# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Phase C-5a: Hybrid model load + substrate detection
# ============================================================
# Minimum-scope precondition for Phase C-5b / C-5c. Single
# goal: verify that we can load a hybrid SSM+Attention model
# and identify its SSM and attention-KV substrates. modeq
# integration and quantisation logic are NOT in scope here —
# they enter in C-5b / C-5c.
#
# Success criteria (all five must pass):
#   1. Hybrid model loads on A100 80 GB
#   2. SSM mixer modules are detected (count + names + shapes)
#   3. Attention K / V projection layers are detected
#   4. FP16 forward runs, FP16 WikiText-2 PPL is reported
#   5. A no-op monkey-patch on the SSM forward path leaves the
#      PPL identical (within fp16 tolerance)
#
# Loading strategies tried in order (cost low → cost high):
#
#   Strategy C: AutoModelForCausalLM.from_pretrained(
#                   "Zyphra/Zamba2-2.7B",
#                   trust_remote_code=True,
#                   ignore_mismatched_sizes=True,
#                   low_cpu_mem_usage=True,
#                   ...)
#               (single line, no env changes)
#
#   Strategy A: same as C but explicit low_cpu_mem_usage; in
#               practice usually no different — kept for the
#               failure-isolation log.
#
#   Strategy B: smaller sibling Zyphra/Zamba2-1.2B (same arch,
#               smaller; sometimes a tied-weights bug in a
#               specific size is absent in another size).
#
#   Strategy D: pseudo-hybrid fallback (Mamba-2.8B-hf body +
#               GPT2 attention, manually composed). Only as a
#               diagnostic; not paper-grade. Marked clearly in
#               the JSON if reached.
# ============================================================


# =============================================================
# === CELL V1 === Setup
# =============================================================

import gc
import math
import json
import torch
import torch.nn.functional as F
from datetime import datetime

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

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers as _hf_transformers
print(f"  transformers version: {_hf_transformers.__version__}")


# =============================================================
# === CELL V2 === Try loading hybrid model (4 strategies)
# =============================================================

LOAD_STRATEGIES = [
    {
        "name":    "C: Zamba2-2.7B + trust_remote_code + ignore_mismatched",
        "label":   "Zamba2-2.7B-strategyC",
        "model_id": "Zyphra/Zamba2-2.7B",
        "kwargs": dict(
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            ignore_mismatched_sizes=True,
            low_cpu_mem_usage=True,
        ),
    },
    {
        "name":    "A: Zamba2-2.7B baseline (no ignore_mismatched)",
        "label":   "Zamba2-2.7B-strategyA",
        "model_id": "Zyphra/Zamba2-2.7B",
        "kwargs": dict(
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ),
    },
    {
        "name":    "B: Zamba2-1.2B smaller sibling",
        "label":   "Zamba2-1.2B-strategyB",
        "model_id": "Zyphra/Zamba2-1.2B",
        "kwargs": dict(
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            ignore_mismatched_sizes=True,
            low_cpu_mem_usage=True,
        ),
    },
    {
        "name":    "D: pseudo-hybrid (Mamba-2.8B-hf, no real attn)",
        "label":   "Mamba-2.8B-hf-pseudo-hybrid",
        "model_id": "state-spaces/mamba-2.8b-hf",
        "kwargs": dict(
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        ),
        "is_fallback": True,
    },
]

mdl       = None
tok       = None
strategy_used  = None
strategy_log   = []

for strat in LOAD_STRATEGIES:
    print(f"\n{'=' * 80}")
    print(f"Trying load strategy: {strat['name']}")
    print(f"{'=' * 80}")
    try:
        # tokenizer first (cheap, will fail early if model_id is wrong)
        try:
            tok_try = AutoTokenizer.from_pretrained(
                strat["model_id"], trust_remote_code=True
            )
        except Exception:
            # Fall back to a generic tokenizer for pseudo-hybrid
            tok_try = AutoTokenizer.from_pretrained(
                "EleutherAI/gpt-neox-20b", trust_remote_code=True
            )
        mdl_try = AutoModelForCausalLM.from_pretrained(
            strat["model_id"], **strat["kwargs"]
        )
        mdl_try.eval()
        mdl, tok = mdl_try, tok_try
        strategy_used = strat
        print(f"  ✓ LOADED. VRAM = "
              f"{torch.cuda.memory_allocated() / 1e9:.2f} GB")
        strategy_log.append({"name": strat["name"], "result": "success"})
        break
    except Exception as e:
        err = f"{type(e).__name__}: {str(e)[:300]}"
        print(f"  ✗ FAILED. {err}")
        strategy_log.append({"name": strat["name"], "result": "failed",
                              "error": err})
        # Clean up any partial allocation before next try
        try:
            del mdl_try
        except NameError:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if mdl is None:
    raise RuntimeError(
        "All four loading strategies failed. See strategy_log "
        "for diagnostics. Likely needed: pip install "
        "transformers==<x> for a stable Zamba2 version, then "
        "restart kernel and re-run."
    )

print(f"\n{'─' * 80}")
print(f"Strategy used: {strategy_used['name']}")
print(f"Model:         {strategy_used['model_id']}")
print(f"is_fallback:   {strategy_used.get('is_fallback', False)}")
print(f"{'─' * 80}")


# =============================================================
# === CELL V3 === Substrate detection
# =============================================================

def detect_ssm_mixers(model):
    """Find all modules with an A_log parameter (Mamba1 / Mamba2
    family fingerprint)."""
    found = []
    for name, mod in model.named_modules():
        A_log = getattr(mod, "A_log", None)
        if isinstance(A_log, torch.nn.Parameter):
            entry = {
                "name":          name,
                "class":         type(mod).__name__,
                "A_log_shape":   list(A_log.shape),
            }
            for attr in ("intermediate_size", "ssm_state_size",
                          "time_step_rank", "d_inner", "d_state",
                          "head_dim", "n_heads", "n_groups"):
                if hasattr(mod, attr):
                    entry[attr] = int(getattr(mod, attr))
            found.append(entry)
    return found


def detect_attention_kv(model):
    """Find all Linear layers whose name suggests an attention
    K / V / KV / QKV projection."""
    found = []
    suffixes = ("k_proj", "v_proj", "kv_proj", "qkv_proj",
                 "Wk", "Wv", "Wqkv")
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        if any(name.endswith(s) for s in suffixes):
            found.append({
                "name":          name,
                "class":         type(mod).__name__,
                "in_features":   mod.in_features,
                "out_features":  mod.out_features,
                "has_bias":      mod.bias is not None,
            })
    return found


ssm_mixers = detect_ssm_mixers(mdl)
attn_kv    = detect_attention_kv(mdl)

print(f"\n{'=' * 80}")
print(f"Substrate detection")
print(f"{'=' * 80}")
print(f"  SSM mixers detected:    {len(ssm_mixers)}")
print(f"  Attention K/V layers:   {len(attn_kv)}")

if ssm_mixers:
    print(f"\n  First SSM mixer (representative):")
    for k, v in ssm_mixers[0].items():
        print(f"    {k:<24s}  {v}")
    if len(ssm_mixers) > 1:
        print(f"  Last SSM mixer:")
        for k, v in ssm_mixers[-1].items():
            print(f"    {k:<24s}  {v}")
    classes = sorted(set(m["class"] for m in ssm_mixers))
    print(f"  Distinct SSM mixer classes: {classes}")
else:
    print(f"  ⚠ NO SSM mixers detected. C-5b will need a different "
          f"detection heuristic.")

if attn_kv:
    print(f"\n  First 3 attention K/V layers:")
    for entry in attn_kv[:3]:
        print(f"    {entry['name']:<60s}  in={entry['in_features']:<5d}"
              f"  out={entry['out_features']:<5d}")
    if len(attn_kv) > 3:
        print(f"    ... and {len(attn_kv) - 3} more")
else:
    if not strategy_used.get("is_fallback", False):
        print(f"  ⚠ NO attention K/V layers detected on a non-fallback "
              f"model. C-5c integration with modeq will need a "
              f"different detection heuristic.")
    else:
        print(f"  (Pseudo-hybrid fallback has no attention layers — "
              f"expected.)")


# =============================================================
# === CELL V4 === FP16 PPL sanity
# =============================================================

ds_test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
test_text = "\n\n".join([x for x in ds_test["text"] if x.strip()])[:48000]
test_ids = tok.encode(test_text, return_tensors="pt").to("cuda")

CTX_LEN   = 1024
STRIDE    = 512
N_WINDOWS = 5     # short — sanity check, not a full benchmark


def ppl_simple(model, ids, ctx_len=1024, stride=512, n_windows=5):
    nll_total = 0.0
    tok_total = 0
    windows_done = 0
    with torch.inference_mode():
        for begin in range(0, ids.shape[1] - ctx_len, stride):
            if windows_done >= n_windows:
                break
            window = ids[:, begin:begin + ctx_len]
            out = model(window, use_cache=False)
            log_probs = F.log_softmax(out.logits[0, :-1].float(), dim=-1)
            targets = window[0, 1:]
            nll = -log_probs.gather(-1,
                                      targets.unsqueeze(-1)).squeeze(-1)
            nll_total += nll.sum().item()
            tok_total += targets.numel()
            windows_done += 1
    return math.exp(nll_total / tok_total), windows_done


print(f"\n{'=' * 80}")
print(f"FP16 PPL sanity (WikiText-2 test, ctx={CTX_LEN}, "
      f"n_windows={N_WINDOWS})")
print(f"{'=' * 80}")

fp16_ppl, n_win = ppl_simple(mdl, test_ids,
                               ctx_len=CTX_LEN, stride=STRIDE,
                               n_windows=N_WINDOWS)
print(f"  FP16 PPL = {fp16_ppl:.4f}  ({n_win} windows)")


# =============================================================
# === CELL V5 === No-op monkey-patch sanity
# =============================================================
#
# Apply an identity monkey-patch to each SSM mixer's forward
# function. Rerun PPL. The two should match within fp16
# tolerance, confirming that monkey-patching the substrate is
# safe and does not break the model — the precondition for
# C-5b's actual quantisation patch.

def find_forward_attr(mod):
    """Pick the first method on `mod` that exists and looks like
    a reference / slow forward."""
    for name in ("slow_forward", "torch_forward",
                  "reference_forward"):
        if hasattr(mod, name) and callable(getattr(mod, name)):
            return name
    return None


def apply_noop_patches(ssm_mixers_metadata, model):
    """Replace each SSM mixer's slow forward (if any) with a
    direct passthrough call to the original method. Returns a
    list of (module, attr_name, original) tuples for restore."""
    records = []
    seen_classes = {}
    # We patch the first available slow forward on each distinct
    # class, since slow_forward is typically defined at the class
    # level and shared across instances.
    for entry in ssm_mixers_metadata:
        # Walk down the model to find the actual module.
        mod = model
        for part in entry["name"].split("."):
            if part:
                mod = getattr(mod, part)
        cls = type(mod)
        if cls in seen_classes:
            continue
        attr = find_forward_attr(cls)
        if attr is None:
            seen_classes[cls] = None
            continue
        orig = getattr(cls, attr)
        # Pass-through wrapper: identical behaviour, but proves
        # that the monkey-patch surface works.
        def make_passthrough(original_fn):
            def patched(self, *args, **kwargs):
                return original_fn(self, *args, **kwargs)
            return patched
        setattr(cls, attr, make_passthrough(orig))
        records.append((cls, attr, orig))
        seen_classes[cls] = attr
        print(f"  patched {cls.__module__}.{cls.__name__}.{attr}")
    return records


def restore_patches(records):
    for cls, attr, orig in records:
        setattr(cls, attr, orig)


print(f"\n{'=' * 80}")
print(f"No-op monkey-patch sanity")
print(f"{'=' * 80}")
patch_records = apply_noop_patches(ssm_mixers, mdl)
if not patch_records:
    print(f"  ⚠ No slow / reference forward found on any SSM mixer. "
          f"C-5b will need a different patch surface (probably "
          f"hooks, or wrapping `forward` directly).")
    patched_ppl = None
else:
    patched_ppl, n_win2 = ppl_simple(
        mdl, test_ids,
        ctx_len=CTX_LEN, stride=STRIDE, n_windows=N_WINDOWS,
    )
    print(f"  PPL after no-op patch = {patched_ppl:.4f}  "
          f"({n_win2} windows)")
    delta = abs(patched_ppl - fp16_ppl)
    rel   = delta / fp16_ppl
    print(f"  |Δ|     = {delta:.6f}")
    print(f"  rel    = {rel:.2e}")
    if rel < 1e-4:
        print(f"  ✓ Pass-through monkey-patch is identity within fp16 "
              f"noise. C-5b can safely insert a quantising patch on "
              f"this surface.")
    else:
        print(f"  ⚠ Pass-through patch shifted PPL noticeably "
              f"(rel = {rel:.2e}). Investigate before C-5b.")
restore_patches(patch_records)


# =============================================================
# === CELL V6 === Save
# =============================================================

success_criteria = {
    "1_model_loaded":               mdl is not None,
    "2_ssm_mixers_detected":        len(ssm_mixers) > 0,
    "3_attention_kv_detected":      len(attn_kv) > 0,
    "4_fp16_ppl_acquired":          fp16_ppl is not None,
    "5_noop_patch_pass":            (patched_ppl is not None and
                                       abs(patched_ppl - fp16_ppl)
                                          / fp16_ppl < 1e-4),
}
all_pass = all(success_criteria.values())

print(f"\n{'=' * 80}")
print(f"Phase C-5a success criteria")
print(f"{'=' * 80}")
for k, v in success_criteria.items():
    mark = "✓" if v else "✗"
    print(f"  {mark}  {k}")
print(f"\n  All criteria pass: {'YES' if all_pass else 'NO'}")
if all_pass:
    print(f"\n  C-5b can proceed.")
elif not success_criteria["3_attention_kv_detected"] and \
        strategy_used.get("is_fallback", False):
    print(f"\n  Fallback (pseudo-hybrid) was used. No attention K/V "
          f"layers is expected. C-5b can still run as an SSM-only "
          f"hybrid-context test, but C-5c (composition) will need "
          f"a real hybrid model.")
else:
    print(f"\n  Investigate failed criteria before proceeding to "
          f"C-5b.")

output = {
    "date":           datetime.utcnow().isoformat(timespec="seconds"),
    "phase":          "C-5a",
    "purpose":        ("Hybrid model load + substrate detection. "
                       "Verifies that a hybrid SSM+Attention "
                       "model loads, identifies its SSM mixers "
                       "and attention K/V projections, recovers "
                       "FP16 PPL, and survives a no-op monkey-"
                       "patch on the SSM forward surface."),
    "transformers_version": _hf_transformers.__version__,
    "strategy_log":   strategy_log,
    "strategy_used":  strategy_used["name"] if strategy_used else None,
    "model_id":       strategy_used["model_id"] if strategy_used else None,
    "is_fallback":    strategy_used.get("is_fallback", False)
                       if strategy_used else None,
    "ssm_mixers": {
        "count":        len(ssm_mixers),
        "classes":      sorted(set(m["class"] for m in ssm_mixers)),
        "first":        ssm_mixers[0] if ssm_mixers else None,
        "last":         ssm_mixers[-1] if ssm_mixers else None,
        "all_names":    [m["name"] for m in ssm_mixers],
    },
    "attention_kv": {
        "count":          len(attn_kv),
        "classes":        sorted(set(e["class"] for e in attn_kv)),
        "first_three":    attn_kv[:3],
        "all_names":      [e["name"] for e in attn_kv],
    },
    "ppl": {
        "fp16":           fp16_ppl,
        "noop_patched":   patched_ppl,
        "abs_delta":      (abs(patched_ppl - fp16_ppl)
                            if patched_ppl is not None else None),
        "rel_delta":      ((abs(patched_ppl - fp16_ppl) / fp16_ppl)
                            if patched_ppl is not None else None),
    },
    "success_criteria":   success_criteria,
    "all_pass":           all_pass,
}

out_path = "/content/phase_c5a_hybrid_load_substrate.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False, default=str)
print(f"\nSaved -> {out_path}")


# =============================================================
# === READING ===
# =============================================================
#
# All 5 criteria pass (and is_fallback = False):
#   Zamba2 (2.7B or 1.2B) loaded, SSM + attention substrates
#   detected, FP16 PPL acquired, monkey-patch surface verified.
#   C-5b can proceed: write a quantising slow_forward replacement
#   for the detected SSM mixer class and verify `full` > `uniform`
#   inside the hybrid.
#
# All 5 criteria pass but is_fallback = True:
#   Zamba2 failed all three strategies; Mamba-2.8B-hf used as
#   pseudo-hybrid. C-5b can still run (it would be a regression
#   test of C-3 / C-4 inside the same flow, useful as a sanity
#   check), but C-5c is blocked until a real hybrid loads.
#   Action: investigate Zamba2 errors logged in strategy_log
#   (often a transformers-version pin to ~4.42 - 4.45 fixes the
#   tied_weights bug; this requires a kernel restart).
#
# Criterion 1 fails:
#   No hybrid model loadable on this transformers version.
#   Pin transformers, restart kernel, re-run.
#
# Criterion 2 fails:
#   Substrate detector did not find any A_log parameters. Either
#   the model isn't an SSM (impossible for Mamba family) or
#   Mamba2 uses a different parameter name. Inspect
#   `mdl.named_parameters()` for SSM-like tensors and update
#   `detect_ssm_mixers`.
#
# Criterion 3 fails:
#   No attention K/V Linear layers found by suffix match. Inspect
#   the model's named_modules() for the actual attention layer
#   naming convention (Zyphra may use `attention.0.k_proj` or a
#   single fused QKV; either is fine, just update the detector).
#
# Criterion 4 fails:
#   FP16 forward errored. Likely a shape / dtype mismatch in the
#   model itself; usually fixed by re-loading without
#   ignore_mismatched_sizes (move from strategy C to A).
#
# Criterion 5 fails:
#   The pass-through monkey-patch shifted PPL. Most likely cause:
#   the slow_forward signature differs in a way the pass-through
#   wrapper drops (e.g., a kwarg the patched method needs but our
#   `*args, **kwargs` doesn't forward correctly). Fix the wrapper
#   in C-5b before adding quantisation logic.
# ============================================================
