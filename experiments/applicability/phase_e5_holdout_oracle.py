# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Phase E-5 (phase 2) -- hold-out generalisation oracle
# ============================================================
# Pre-registered (experiments/holdout_preregistration.md, lock #2 SHA 1ebf5d9):
# brute-force the K-recipe grid on the four hold-out models under the N=30
# random-window bootstrap, then test whether each frozen-compiler prediction
# (results/e5_holdout_predictions.json) is the empirically-best recipe at its
# compression tier. This is the oracle the predictions are scored against; it
# was run AFTER the predictions were locked.
#
# Grid (per pre-reg section 5), measured on every model so the comparison is
# symmetric whether or not the model routes to hotspot:
#   nsn_int3 / nsn_int4 / nsn_int5            (norm-separated K family)
#   plain_asym_int3 / plain_asym_int4 / int5  (KIVI-core K family)
#   three_tier_mask                           (NSN per-layer mask from diagnostic)
#
# Recipe family also sets V handling (matching phase_e4): NSN family -> V = NSN
# at the same bits; plain-asym family -> V = per-(head, token) asym at the same
# bits; the 3-tier mask is an NSN-family masked recipe. E / M / S untouched.
#
# Quant is vendored bit-identical from phase_e1 (nsn_k / _asym_qdq_over_tokens /
# v_per_token; A0 full-window scale, the in-harness protocol shared with
# phase_e3 / e4). Per-window paired CIs are computed on the same windows.
#
# Metric (analysed in analyze_e5.py): the matched-budget family / structure
# comparison (phase_e3 "3-tier vs best single", phase_e4 "plain-asym vs NSN at
# matched bits"). INT5 is the conservative tier and wins on raw PPL by spending
# bits, so the H1 verdict uses the matched-budget comparison, not best-overall.
#
# Output: /content/e5_holdout_oracle.json (save-first after each recipe).
# ============================================================

import gc
import json
import math
import os
import re
import shutil
import traceback
from datetime import datetime, timezone

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import google.colab  # noqa: F401
    _IN_COLAB = True
except Exception:
    _IN_COLAB = False

if _IN_COLAB:
    os.system("pip install -q -U transformers accelerate hf_transfer datasets")
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    from google.colab import userdata
    from huggingface_hub import login
    _tok = userdata.get("HF_TOKEN")   # NAME of a Colab Secret, not the value
    if _tok:
        login(token=_tok)

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

CTX = 1024
N_RANDOM_WINDOWS = 30
N_BOOTSTRAP = 5000
RANDOM_SEED = 20260510          # seed-aligned to e1/e2/e3/e4 so windows match
EPS = 1e-12
OUT_PATH = "/content/e5_holdout_oracle.json" if _IN_COLAB else "e5_holdout_oracle.json"

# Hold-out models + their diagnostic 3-tier mask (K_good -> INT3, K_bad -> FP16,
# rest -> INT4), taken from results/e5_holdout_predictions.json (frozen-compiler
# output, committed at lock #2). Uniform models have K_good = K_bad = [] so their
# 3-tier cell is all-INT4 (a self-consistency check against nsn_int4). Each
# model also records its frozen-compiler prediction (the recipe under test).
MODELS = [
    {"id": "ibm-ai-platform/Bamba-9B-v2", "label": "Bamba-9B-v2",
     "trust_remote_code": False, "K_good": [9], "K_bad": [18, 27],
     "K_localization": "hotspot", "prediction": "three_tier_mask"},
    {"id": "allenai/OLMoE-1B-7B-0924", "label": "OLMoE-1B-7B",
     "trust_remote_code": False, "K_good": [], "K_bad": [],
     "K_localization": "uniform", "prediction": "plain_asym_int4"},
    {"id": "ibm-granite/granite-3.0-3b-a800m-base", "label": "Granite-3.0-3b-a800m",
     "trust_remote_code": False, "K_good": [5, 7, 8, 10], "K_bad": [21],
     "K_localization": "hotspot", "prediction": "three_tier_mask"},
    {"id": "Qwen/Qwen3-30B-A3B-Base", "label": "Qwen3-30B-A3B",
     "trust_remote_code": False, "K_good": [], "K_bad": [],
     "K_localization": "uniform", "prediction": "plain_asym_int4"},
]

# Recipe grid (family sets V handling). use_mask draws the per-model 3-tier mask.
RECIPES = [
    {"key": "nsn_int3", "family": "nsn", "k_bits": 3, "use_mask": False},
    {"key": "nsn_int4", "family": "nsn", "k_bits": 4, "use_mask": False},
    {"key": "nsn_int5", "family": "nsn", "k_bits": 5, "use_mask": False},
    {"key": "plain_asym_int3", "family": "plain_asym", "k_bits": 3, "use_mask": False},
    {"key": "plain_asym_int4", "family": "plain_asym", "k_bits": 4, "use_mask": False},
    {"key": "plain_asym_int5", "family": "plain_asym", "k_bits": 5, "use_mask": False},
    {"key": "three_tier_mask", "family": "nsn", "k_bits": 4, "use_mask": True},
]


# --- quant (vendored bit-identical from phase_e1 A0) ---
def _nsn_qdq_over_tokens(d, bits):
    levels = (1 << bits) - 1
    d_min = d.amin(dim=1, keepdim=True)
    d_max = d.amax(dim=1, keepdim=True)
    step = ((d_max - d_min).float() / levels).clamp(min=EPS).to(d.dtype)
    return ((d - d_min).float() / step.float()).round().clamp(0, levels).to(d.dtype) * step + d_min


def nsn_k(k, bits, n_kv, head_dim):
    """NSN norm-separated per-(head, channel) asym K, full-window scale."""
    if k.dim() != 3:
        return k
    B, T, hidden = k.shape
    if hidden != n_kv * head_dim:
        return k
    x = k.reshape(B, T, n_kv, head_dim).float()
    norm = x.norm(dim=-1, keepdim=True).clamp(min=EPS)
    d = x / norm
    dq = _nsn_qdq_over_tokens(d, bits)
    dq = dq / dq.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return (norm * dq).reshape(B, T, hidden).to(k.dtype)


def _asym_qdq_over_tokens(x, bits):
    """KIVI-core per-channel asym quant->dequant; scale over the token axis."""
    levels = (1 << bits) - 1
    x_min = x.amin(dim=1, keepdim=True)
    x_max = x.amax(dim=1, keepdim=True)
    step = ((x_max - x_min).float() / levels).clamp(min=EPS).to(x.dtype)
    q = ((x - x_min).float() / step.float()).round().clamp(0, levels)
    return q.to(x.dtype) * step + x_min


def plain_asym_k(k, bits, n_kv, head_dim):
    """KIVI-core plain per-channel asym K (no norm separation), full-window."""
    if k.dim() != 3:
        return k
    return _asym_qdq_over_tokens(k, bits)


def v_per_token(v, bits, n_kv, head_dim):
    """V cache per-(head, token) asym (scale over head_dim); already causal."""
    if v.dim() != 3:
        return v
    B, T, hidden = v.shape
    if hidden != n_kv * head_dim:
        return v
    vv = v.reshape(B, T, n_kv, head_dim)
    levels = (1 << bits) - 1
    v_min = vv.amin(dim=-1, keepdim=True)
    v_max = vv.amax(dim=-1, keepdim=True)
    step = ((v_max - v_min).float() / levels).clamp(min=EPS).to(vv.dtype)
    q = ((vv - v_min).float() / step.float()).round().clamp(0, levels)
    return (q.to(vv.dtype) * step + v_min).reshape(B, T, hidden)


# --- layer detection + geometry (reuse phase_e3) ---
LAYER_REGEXES = [
    re.compile(r"^model\.layers\.(\d+)\b"),
    re.compile(r"^model\.language_model\.layers\.(\d+)\b"),
    re.compile(r"^language_model\.(?:model\.)?layers\.(\d+)\b"),
    re.compile(r"^backbone\.layers\.(\d+)\b"),
    re.compile(r"^layers\.(\d+)\b"),
]


def match_layer(name):
    for rx in LAYER_REGEXES:
        m = rx.match(name)
        if m is not None:
            return int(m.group(1))
    return None


def find_attention_layers(model):
    found = {}
    for name, mod in model.named_modules():
        idx = match_layer(name)
        if idx is None or "self_attn" not in name or idx in found:
            continue
        kp, vp = getattr(mod, "k_proj", None), getattr(mod, "v_proj", None)
        if isinstance(kp, nn.Linear) and isinstance(vp, nn.Linear):
            found[idx] = {"self_attn_module": mod, "k_proj": kp, "v_proj": vp}
    return found


def read_kv_geometry(attn_mod):
    head_dim = (getattr(attn_mod, "head_dim", None)
                or getattr(attn_mod, "hidden_size_per_head", None))
    num_kv = (getattr(attn_mod, "num_key_value_heads", None)
              or getattr(attn_mod, "num_kv_heads", None)
              or getattr(attn_mod, "num_attention_heads", None)
              or getattr(attn_mod, "num_heads", None))
    if (num_kv is None or head_dim is None) and hasattr(attn_mod, "config"):
        cfg = attn_mod.config
        if num_kv is None:
            num_kv = (getattr(cfg, "num_key_value_heads", None)
                      or getattr(cfg, "num_attention_heads", None))
        if head_dim is None:
            head_dim = getattr(cfg, "head_dim", None)
            if head_dim is None:
                hidden = getattr(cfg, "hidden_size", None)
                n_attn = getattr(cfg, "num_attention_heads", None)
                if hidden and n_attn:
                    head_dim = hidden // n_attn
    if num_kv is None and head_dim is not None:
        kp = getattr(attn_mod, "k_proj", None)
        if isinstance(kp, nn.Linear) and kp.out_features % head_dim == 0:
            num_kv = kp.out_features // head_dim
    return int(num_kv), int(head_dim)


def build_mask(attn_layers, K_good, K_bad):
    """3-tier mask {layer: bits}: K_good -> 3, K_bad -> FP16 (omitted), else 4."""
    good, bad = set(K_good), set(K_bad)
    m = {}
    for idx in attn_layers:
        if idx in bad:
            continue          # FP16: no hook
        m[idx] = 3 if idx in good else 4
    return m


def install_hooks(attn_layers, recipe, mask):
    """Install K + V hooks for a recipe. NSN family -> K,V = nsn_k(bits);
    plain_asym family -> K = plain_asym_k(bits), V = v_per_token(bits).
    use_mask -> per-layer NSN bits from `mask` (K_bad omitted = FP16)."""
    handles = []
    if recipe["use_mask"]:
        plan = [(idx, mask[idx]) for idx in sorted(mask) if idx in attn_layers]
        kq = vq = nsn_k
    elif recipe["family"] == "nsn":
        plan = [(idx, recipe["k_bits"]) for idx in attn_layers]
        kq = vq = nsn_k
    else:  # plain_asym
        plan = [(idx, recipe["k_bits"]) for idx in attn_layers]
        kq, vq = plain_asym_k, v_per_token
    for idx, kb in plan:
        info = attn_layers[idx]
        n_kv, head_dim = read_kv_geometry(info["self_attn_module"])

        def mk(fn, b, nk, hd):
            def hook(mod, inp, out):
                return fn(out, b, nk, hd)
            return hook

        handles.append(info["k_proj"].register_forward_hook(mk(kq, kb, n_kv, head_dim)))
        handles.append(info["v_proj"].register_forward_hook(mk(vq, kb, n_kv, head_dim)))
    return handles


def remove_hooks(handles):
    for h in handles:
        h.remove()


# --- PPL + windows + bootstrap (reuse phase_e3) ---
def ppl_single_window(model, ids, begin):
    window = ids[:, begin:begin + CTX]
    with torch.inference_mode():
        out = model(window, use_cache=False)
        logits = out.logits if hasattr(out, "logits") else None
        if logits is None:
            return float("inf")
        logits_f = logits[0, :-1].float()
        targets = window[0, 1:]
        nll = -F.log_softmax(logits_f, dim=-1).gather(
            -1, targets.unsqueeze(-1)).squeeze(-1)
        del logits, logits_f
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return math.exp(float(nll.mean()))


def select_random_windows(n_tokens, n_windows, seed):
    g = torch.Generator().manual_seed(seed)
    max_start = n_tokens - CTX - 1
    if max_start <= 0:
        raise ValueError(f"corpus too short for ctx={CTX}")
    return torch.randint(0, max_start, (n_windows,), generator=g).tolist()


def bootstrap_ci(values, seed, ci_pct=95):
    arr = torch.tensor(values, dtype=torch.float64)
    g = torch.Generator().manual_seed(seed)
    n = len(values)
    means = torch.stack([arr[torch.randint(0, n, (n,), generator=g)].mean()
                         for _ in range(N_BOOTSTRAP)])
    lo = float(torch.quantile(means, (100 - ci_pct) / 200))
    hi = float(torch.quantile(means, 1 - (100 - ci_pct) / 200))
    return {"mean": float(arr.mean()),
            "std": float(arr.std(unbiased=True)) if n > 1 else 0.0,
            "ci_lo": lo, "ci_hi": hi, "ci_half_width": (hi - lo) / 2}


def free_hf_cache():
    for d in ("/root/.cache/huggingface/hub", os.path.expanduser("~/.cache/huggingface/hub")):
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _save(payload):
    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)


def run_model_oracle(spec, text, payload):
    label = spec["label"]
    print(f"\n{'=' * 74}\nOracle: {label}  ({spec['id']})\n{'=' * 74}")
    tok = AutoTokenizer.from_pretrained(spec["id"], trust_remote_code=spec["trust_remote_code"])
    model = AutoModelForCausalLM.from_pretrained(
        spec["id"], torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=spec["trust_remote_code"], low_cpu_mem_usage=True).eval()
    dev = next(model.parameters()).device
    ids = tok(text, return_tensors="pt").input_ids.to(dev)
    attn = find_attention_layers(model)
    mask = build_mask(attn, spec["K_good"], spec["K_bad"])
    print(f"  {len(attn)} attn layers; {ids.shape[1]} tokens; "
          f"mask INT3={sorted(spec['K_good'])} FP16={sorted(spec['K_bad'])} "
          f"INT4={len([l for l in attn if l not in set(spec['K_good']) | set(spec['K_bad'])])}")

    starts = select_random_windows(ids.shape[1], N_RANDOM_WINDOWS, RANDOM_SEED)
    fp16 = [ppl_single_window(model, ids, b) for b in starts]
    fp16_mean = sum(fp16) / len(fp16)
    print(f"  FP16 ref PPL mean = {fp16_mean:.4f}")

    mrec = {"model_id": spec["id"], "K_localization": spec["K_localization"],
            "prediction": spec["prediction"], "n_attn_layers": len(attn),
            "window_starts": starts, "fp16_ref_ppl_mean": fp16_mean,
            "mask": {str(k): v for k, v in mask.items()},
            "cells": {}, "paired": {}}

    for r in RECIPES:
        try:
            handles = install_hooks(attn, r, mask)
            q = [ppl_single_window(model, ids, b) for b in starts]
            remove_hooks(handles)
            dppl = [100.0 * (q[i] - fp16[i]) / fp16[i] for i in range(len(fp16))]
            ci = bootstrap_ci(dppl, RANDOM_SEED + 1)
            mrec["cells"][r["key"]] = {"family": r["family"], "k_bits": r["k_bits"],
                                       "use_mask": r["use_mask"],
                                       "per_window_dppl": dppl, **ci}
            print(f"  [{r['key']:16s}] dPPL {ci['mean']:+.4f}% [{ci['ci_lo']:+.3f}, {ci['ci_hi']:+.3f}]")
        except Exception as e:
            mrec["cells"].setdefault("_errors", {})[r["key"]] = f"{type(e).__name__}: {e}"
            print(f"  !! RECIPE ERROR [{r['key']}]: {type(e).__name__}: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        payload["models"][label] = mrec
        _save(payload)

    # paired CIs: prediction vs every other measured recipe (same windows)
    cells = mrec["cells"]
    pred = spec["prediction"]
    if pred in cells and "per_window_dppl" in cells[pred]:
        for k, c in cells.items():
            if k == pred or k == "_errors" or "per_window_dppl" not in c:
                continue
            diff = [cells[pred]["per_window_dppl"][i] - c["per_window_dppl"][i]
                    for i in range(N_RANDOM_WINDOWS)]
            ci = bootstrap_ci(diff, RANDOM_SEED + 2)
            ci["pred_cleaner_excludes_zero"] = bool(ci["ci_hi"] < 0)   # pred sig more negative
            ci["pred_not_worse"] = bool(ci["ci_lo"] <= 0)              # CI does not favour alt
            mrec["paired"][f"{pred}_vs_{k}"] = ci
    payload["models"][label] = mrec
    _save(payload)

    del model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    print(f"{'=' * 74}\nPhase E-5 phase 2: hold-out generalisation oracle")
    print(f"  models: {[m['label'] for m in MODELS]}")
    print(f"  recipes: {[r['key'] for r in RECIPES]}")
    print(f"  ctx={CTX} N={N_RANDOM_WINDOWS} N_boot={N_BOOTSTRAP} seed={RANDOM_SEED}\n{'=' * 74}")

    from datasets import load_dataset
    text = None
    for ds_id in ("Salesforce/wikitext", "wikitext"):
        try:
            ds = load_dataset(ds_id, "wikitext-2-raw-v1", split="test")
            text = "\n\n".join(x for x in ds["text"] if x.strip())
            print(f"  WikiText-2 test from '{ds_id}' ({len(text)} chars)")
            break
        except Exception as e:
            print(f"  (wikitext via '{ds_id}' failed: {type(e).__name__})")
    if text is None:
        raise RuntimeError("could not load WikiText-2 test")

    payload = {"phase": "E-5 phase 2: hold-out generalisation oracle",
               "frozen_compiler_sha": "98508ff", "preregistration_lock2": "1ebf5d9",
               "knobs": {"ctx": CTX, "n_windows": N_RANDOM_WINDOWS,
                         "n_bootstrap": N_BOOTSTRAP, "seed": RANDOM_SEED},
               "models": {}, "model_errors": {},
               "started": datetime.now(timezone.utc).isoformat()}

    for spec in MODELS:
        try:
            run_model_oracle(spec, text, payload)
        except Exception as e:
            payload["model_errors"][spec["label"]] = f"{type(e).__name__}: {e}"
            print(f"\n  [FAIL] {spec['label']}: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        _save(payload)
        free_hf_cache()

    payload["finished"] = datetime.now(timezone.utc).isoformat()
    _save(payload)
    print(f"\n{'=' * 74}\nDONE. {len(payload['models'])}/{len(MODELS)} models.")
    print(f"Errors: {list(payload['model_errors'])}")
    print(f"Saved: {OUT_PATH}\nNext: analyze_e5.py locally for the H1 verdict.")


if __name__ == "__main__":
    main()
