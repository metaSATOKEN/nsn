# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Phase W-1d -- causal-faithful re-evaluation of the long-context family split
# ============================================================
# Pre-registered: experiments/W1d_causal_faithful_preregistration.md (repo 949cb7a).
#
# Decides whether the section6.5 long-context recipe-family separation (NSN-INT5 in band,
# plain-asym-INT5 out of band) survives the production-faithful scale, or is an
# artefact of the in-harness full-window (A0) scale that section9 shows is pessimistic to
# the plain-asymmetric family by up to 2.0 pp on Gemma4.
#
# For each load-bearing cell -- Gemma4-26B-A4B at 8K and 32K, Gemma4-12B at 64K --
# measure NSN-INT5 and plain-asym (KIVI-core) INT5 under the four causal-faithful
# configs and report the cross-recipe gap (plain-asym minus NSN) per config:
#   A0  full-window scale          (the section6.5 measurement; the gap to re-test)
#   A1  causal per-group scale
#   A2  A1 + FP16 residual buffer
#   A3  A2 + post-RMSNorm hook      (production-faithful on QK-norm Gemma4)
# Verdict (pre-reg section4): the separation SURVIVES on a cell if at A3 the gap CI lower
# bound > 0 AND plain-asym-INT5 is still out of band (A3 CI lower bound > +0.5%).
#
# Quant + A3 post-norm hook vendored bit-identical from phase_e1; chunked NLL and the
# N=30 random-window bootstrap from phase_w1b; the multi-auto-class loader for the
# 12B unified model from phase_w1c. E/M/S held FP16.
#
# Output: results/w1d_causal_faithful_longctx.json (save-first after every recipe).
# GPU-free logic check: run `python3 phase_w1d_causal_faithful_longctx.py` OUTSIDE Colab.
# ============================================================

import gc
import json
import math
import os
import re
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
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    from google.colab import userdata
    from huggingface_hub import login
    _tok = userdata.get("HF_TOKEN")   # NAME of a Colab Secret, never the value
    if _tok:
        login(token=_tok)


# =============================================================
# Constants
# =============================================================

# (model_id, label, [contexts]); model loaded once, contexts run in order.
CELLS = [
    ("google/gemma-4-26B-A4B", "Gemma4-26B-A4B-base", [8192, 32768]),
    ("google/gemma-4-12B",     "Gemma4-12B",          [65536]),
]
RECIPES = ["nsn", "kivi"]            # NSN hero vs plain-asym (KIVI-core), both INT5
K_BITS = V_BITS = 5
GROUP_SIZE = 128
RESIDUAL_LEN = 128
N_RANDOM_WINDOWS = 30
N_BOOTSTRAP = 5000
RANDOM_SEED = 20260510
NLL_CHUNK = 2048
CLEAN_BAND_PP = 0.5
EPS = 1e-12

CONFIGS = [
    {"name": "A0_full_window",    "causal": False, "group": None,       "residual": 0,            "post_norm": False},
    {"name": "A1_causal_grouped", "causal": True,  "group": GROUP_SIZE, "residual": 0,            "post_norm": False},
    {"name": "A2_causal_residual","causal": True,  "group": GROUP_SIZE, "residual": RESIDUAL_LEN, "post_norm": False},
    {"name": "A3_post_norm",      "causal": True,  "group": GROUP_SIZE, "residual": RESIDUAL_LEN, "post_norm": True},
]

OUT_PATH = ("/content/w1d_causal_faithful_longctx.json" if _IN_COLAB
            else "results/w1d_causal_faithful_longctx.json")


# =============================================================
# Quant: A0 full-window vs causal-faithful (vendored bit-identical from phase_e1)
# =============================================================

def _asym_qdq_over_tokens(x, bits):
    levels = (1 << bits) - 1
    x_min = x.amin(dim=1, keepdim=True)
    x_max = x.amax(dim=1, keepdim=True)
    step = ((x_max - x_min).float() / levels).clamp(min=EPS).to(x.dtype)
    q = ((x - x_min).float() / step.float()).round().clamp(0, levels)
    return q.to(x.dtype) * step + x_min


def quant_k_kivi(k, bits, cfg):
    if k.dim() != 3:
        return k
    if not cfg["causal"]:
        return _asym_qdq_over_tokens(k, bits)
    T = k.shape[1]
    out = k.clone()
    n_resid = min(max(cfg["residual"], 0), T)
    quant_T = T - n_resid
    g = cfg["group"] or T
    for s in range(0, quant_T, g):
        e = min(s + g, quant_T)
        out[:, s:e, :] = _asym_qdq_over_tokens(k[:, s:e, :], bits)
    return out


def _nsn_qdq_over_tokens(d, bits):
    levels = (1 << bits) - 1
    d_min = d.amin(dim=1, keepdim=True)
    d_max = d.amax(dim=1, keepdim=True)
    step = ((d_max - d_min).float() / levels).clamp(min=EPS).to(d.dtype)
    return ((d - d_min).float() / step.float()).round().clamp(0, levels).to(d.dtype) * step + d_min


def quant_k_nsn(k, bits, cfg, n_kv, head_dim):
    if k.dim() != 3:
        return k
    B, T, hidden = k.shape
    if hidden != n_kv * head_dim:
        return k
    x = k.reshape(B, T, n_kv, head_dim).float()
    norm = x.norm(dim=-1, keepdim=True).clamp(min=EPS)     # per-(token, head): causal
    d = x / norm
    if not cfg["causal"]:
        dq = _nsn_qdq_over_tokens(d, bits)
    else:
        dq = d.clone()
        n_resid = min(max(cfg["residual"], 0), T)
        quant_T = T - n_resid
        g = cfg["group"] or T
        for s in range(0, quant_T, g):
            e = min(s + g, quant_T)
            dq[:, s:e] = _nsn_qdq_over_tokens(d[:, s:e], bits)
    dq = dq / dq.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return (norm * dq).reshape(B, T, hidden).to(k.dtype)


def quant_v_per_token(v, bits, n_kv, head_dim):
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


# =============================================================
# Layer detection + geometry + k_norm (vendored from phase_e1)
# =============================================================

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


def find_k_norm(attn_mod):
    for nm in ("k_norm", "key_norm", "k_layernorm", "key_layernorm", "k_rmsnorm"):
        m = getattr(attn_mod, nm, None)
        if isinstance(m, nn.Module):
            return m
    return None


def find_attention_layers(model):
    found = {}
    for name, mod in model.named_modules():
        idx = match_layer(name)
        if idx is None or "self_attn" not in name or idx in found:
            continue
        kp, vp = getattr(mod, "k_proj", None), getattr(mod, "v_proj", None)
        if isinstance(kp, nn.Linear) and isinstance(vp, nn.Linear):
            found[idx] = {"self_attn_module": mod, "k_proj": kp, "v_proj": vp,
                          "k_norm": find_k_norm(mod)}
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


# =============================================================
# A3 post-norm hook plumbing (vendored from phase_e1 / causal_faithful_quant)
# =============================================================

def postnorm_to_canonical(t, n_kv, head_dim):
    hidden = n_kv * head_dim
    if t.dim() == 3 and t.shape[-1] == hidden:
        return t, "BTH"
    if t.dim() == 4 and t.shape[-1] == head_dim:
        B, d1, d2, _ = t.shape
        if d1 == n_kv:
            return t.transpose(1, 2).reshape(B, d2, hidden), "BnTh"
        if d2 == n_kv:
            return t.reshape(B, d1, hidden), "BTnh"
    return None, "unknown" + str(tuple(t.shape))


def postnorm_from_canonical(c, ref, layout, n_kv, head_dim):
    if layout in ("BTH", "BTnh"):
        return c.reshape(ref.shape).to(ref.dtype)
    if layout == "BnTh":
        B, T = ref.shape[0], c.shape[1]
        return c.reshape(B, T, n_kv, head_dim).transpose(1, 2).reshape(ref.shape).to(ref.dtype)
    return ref


_A3_SEEN = set()


def install_kv_hooks(attn_layers, recipe, cfg):
    """K (recipe in {nsn, kivi}) + V hooks at uniform K_BITS/V_BITS, config-driven.
    cfg['post_norm'] (A3) moves the K hook from k_proj to k_norm."""
    post_norm = cfg.get("post_norm", False)
    handles = []
    for idx in sorted(attn_layers):
        info = attn_layers[idx]
        n_kv, head_dim = read_kv_geometry(info["self_attn_module"])

        def make_k(nk, hd):
            def hook(mod, inp, out):
                if recipe == "nsn":
                    return quant_k_nsn(out, K_BITS, cfg, nk, hd)
                return quant_k_kivi(out, K_BITS, cfg)
            return hook

        def make_k_postnorm(nk, hd):
            def hook(mod, inp, out):
                canon, layout = postnorm_to_canonical(out, nk, hd)
                if layout not in _A3_SEEN:
                    _A3_SEEN.add(layout)
                    print(f"    [A3] k_norm out {tuple(out.shape)} layout={layout}")
                if canon is None:
                    raise RuntimeError(f"A3: unrecognized k_norm layout {layout}")
                cq = (quant_k_nsn(canon, K_BITS, cfg, nk, hd) if recipe == "nsn"
                      else quant_k_kivi(canon, K_BITS, cfg))
                return postnorm_from_canonical(cq, out, layout, nk, hd)
            return hook

        def make_v(nk, hd):
            def hook(mod, inp, out):
                # match the family V handling (NSN family V = NSN; KIVI V = per-token)
                if recipe == "nsn":
                    return quant_k_nsn(out, V_BITS, cfg, nk, hd)
                return quant_v_per_token(out, V_BITS, nk, hd)
            return hook

        k_norm = info.get("k_norm")
        if post_norm and k_norm is not None:
            handles.append(k_norm.register_forward_hook(make_k_postnorm(n_kv, head_dim)))
        else:
            handles.append(info["k_proj"].register_forward_hook(make_k(n_kv, head_dim)))
        handles.append(info["v_proj"].register_forward_hook(make_v(n_kv, head_dim)))
    return handles


def remove_hooks(handles):
    for h in handles:
        h.remove()


# =============================================================
# Forward-path detection (vendored from phase_w1b)
# =============================================================

def _try_forward(model, ids, path_name):
    try:
        if path_name == "direct":
            out = model(ids, use_cache=False)
            return out.logits if hasattr(out, "logits") else None
        if path_name == "language_model":
            lm = getattr(model, "language_model", None)
            if lm is None:
                return None
            out = lm(ids, use_cache=False)
            if hasattr(out, "logits"):
                return out.logits
            hidden = (out.last_hidden_state if hasattr(out, "last_hidden_state")
                      else (out[0] if isinstance(out, tuple) else None))
            head = getattr(model, "lm_head", None) or getattr(lm, "lm_head", None)
            if hidden is not None and head is not None:
                return head(hidden)
        if path_name == "model.language_model":
            inner = getattr(getattr(model, "model", None), "language_model", None)
            if inner is None:
                return None
            out = inner(ids, use_cache=False)
            if hasattr(out, "logits"):
                return out.logits
            hidden = (out.last_hidden_state if hasattr(out, "last_hidden_state")
                      else (out[0] if isinstance(out, tuple) else None))
            head = getattr(model, "lm_head", None) or getattr(inner, "lm_head", None)
            if hidden is not None and head is not None:
                return head(hidden)
    except Exception as e:
        print(f"      [{path_name}] error: {type(e).__name__}: {str(e)[:80]}")
        return None
    return None


def select_forward_callable(model, ids_sample):
    print("  [forward-path detect] sanity PPL on a short sample:")
    best_path, best_ppl = None, float("inf")
    with torch.inference_mode():
        for p in ("direct", "language_model", "model.language_model"):
            logits = _try_forward(model, ids_sample, p)
            if logits is None:
                print(f"      [{p}] not available")
                continue
            lg = logits[0, :-1].float()
            tg = ids_sample[0, 1:]
            nll = -F.log_softmax(lg, dim=-1).gather(-1, tg.unsqueeze(-1)).squeeze(-1)
            ppl = math.exp(float(nll.mean()))
            print(f"      [{p}] sanity PPL = {ppl:.2f}")
            if ppl < best_ppl:
                best_ppl, best_path = ppl, p
            del logits, lg, nll
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if best_path is None:
        raise RuntimeError("all forward paths failed")
    print(f"  [forward-path detect] selected: {best_path} (sanity PPL {best_ppl:.2f})")
    return lambda m, x: _try_forward(m, x, best_path)


# =============================================================
# PPL (chunked NLL) + windows + bootstrap (vendored from phase_w1b)
# =============================================================

def ppl_single_window(forward_fn, model, ids, begin, ctx):
    window = ids[:, begin:begin + ctx]
    with torch.inference_mode():
        logits = forward_fn(model, window)
        if logits is None:
            return float("inf")
        L = logits[0, :-1]
        targets = window[0, 1:]
        nll_sum, n = 0.0, 0
        for i in range(0, targets.numel(), NLL_CHUNK):
            lg = L[i:i + NLL_CHUNK].float()
            tg = targets[i:i + NLL_CHUNK]
            nll = -F.log_softmax(lg, dim=-1).gather(-1, tg.unsqueeze(-1)).squeeze(-1)
            nll_sum += float(nll.sum())
            n += int(tg.numel())
            del lg, nll
        del logits, L
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return math.exp(nll_sum / max(n, 1))


def select_random_windows(n_tokens, n_windows, seed, ctx):
    g = torch.Generator().manual_seed(seed)
    max_start = n_tokens - ctx - 1
    if max_start <= 0:
        raise ValueError(f"corpus too short for ctx={ctx}")
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


def _save(payload):
    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)


def load_model(model_id):
    import transformers
    last = None
    for auto_name in ("AutoModelForCausalLM", "AutoModelForImageTextToText", "AutoModel"):
        AutoCls = getattr(transformers, auto_name, None)
        if AutoCls is None:
            continue
        try:
            m = AutoCls.from_pretrained(
                model_id, torch_dtype=torch.float16, device_map="auto",
                trust_remote_code=True, low_cpu_mem_usage=True).eval()
            print(f"  loaded via {auto_name}")
            return m
        except Exception as e:
            print(f"  {auto_name} failed: {type(e).__name__}: {str(e)[:90]}")
            last = e
    raise RuntimeError(f"all auto-classes failed: {last}")


# =============================================================
# Measure one (recipe) across A0-A3 at a context
# =============================================================

def run_recipe(forward_fn, model, ids, attn, ctx, recipe, starts, fp16, has_knorm):
    out = {}
    for cfg in CONFIGS:
        if cfg["post_norm"] and not has_knorm:
            out[cfg["name"]] = {"note": "no k_norm; A3 == A2; skipped"}
            continue
        handles = install_kv_hooks(attn, recipe, cfg)
        try:
            q = [ppl_single_window(forward_fn, model, ids, b, ctx) for b in starts]
        finally:
            remove_hooks(handles)
        dppl = [100.0 * (q[i] - fp16[i]) / fp16[i] for i in range(len(fp16))]
        ci = bootstrap_ci(dppl, RANDOM_SEED + 1)
        out[cfg["name"]] = {"per_window_dppl": dppl,
                            "out_of_band": bool(ci["ci_lo"] > CLEAN_BAND_PP),
                            "in_band": bool(ci["ci_hi"] < CLEAN_BAND_PP), **ci}
        print(f"    {recipe:>4} {cfg['name']:>20}: dPPL {ci['mean']:+.4f}% "
              f"[{ci['ci_lo']:+.3f}, {ci['ci_hi']:+.3f}]")
    return out


def cross_recipe_gaps(nsn, kivi, n):
    """Paired (kivi - nsn) gap per config, plus the A3 verdict (pre-reg section4)."""
    gaps = {}
    for name in [c["name"] for c in CONFIGS]:
        a, b = kivi.get(name), nsn.get(name)
        if not (a and b and "per_window_dppl" in a and "per_window_dppl" in b):
            continue
        diff = [a["per_window_dppl"][i] - b["per_window_dppl"][i] for i in range(n)]
        ci = bootstrap_ci(diff, RANDOM_SEED + 7)
        ci["plain_worse_excludes_zero"] = bool(ci["ci_lo"] > 0)
        gaps[name] = ci
    prod = "A3_post_norm" if "A3_post_norm" in gaps else "A2_causal_residual"
    g = gaps.get(prod)
    kivi_prod = kivi.get(prod, {})
    survive = bool(g and g["ci_lo"] > 0 and kivi_prod.get("out_of_band"))
    collapse = bool(g and (g["ci_lo"] <= 0 or kivi_prod.get("in_band")))
    verdict = "survive" if survive else ("collapse" if collapse else "marginal")
    return gaps, {"production_config": prod, "verdict": verdict,
                  "gap_at_prod": g, "plain_asym_out_of_band_at_prod": kivi_prod.get("out_of_band")}


# =============================================================
# Main
# =============================================================

def main():
    from datasets import load_dataset
    from transformers import AutoTokenizer
    import transformers as _hf

    print(f"{'=' * 74}\nPhase W-1d causal-faithful long-context family-split re-evaluation")
    print(f"  cells: {[(c[1], c[2]) for c in CELLS]}  recipes: {RECIPES} (INT{K_BITS})")
    print(f"  configs: {[c['name'] for c in CONFIGS]} (group={GROUP_SIZE}, residual={RESIDUAL_LEN})")
    print(f"  N={N_RANDOM_WINDOWS} seed={RANDOM_SEED}  transformers {_hf.__version__}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"{'=' * 74}")

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

    payload = {
        "phase": "W-1d causal-faithful long-context family-split re-evaluation",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": "phase_w1d_causal_faithful_longctx.py",
        "transformers_version": _hf.__version__,
        "preregistration": "experiments/W1d_causal_faithful_preregistration.md",
        "knobs": {"k_bits": K_BITS, "group_size": GROUP_SIZE, "residual_len": RESIDUAL_LEN,
                  "n_windows": N_RANDOM_WINDOWS, "n_bootstrap": N_BOOTSTRAP,
                  "seed": RANDOM_SEED, "nll_chunk": NLL_CHUNK, "clean_band_pp": CLEAN_BAND_PP},
        "cells": {}, "errors": {},
    }
    _save(payload)

    for model_id, label, contexts in CELLS:
        try:
            tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            mdl = load_model(model_id)
            dev = next(mdl.parameters()).device
            ids = tok.encode(text, return_tensors="pt").to(dev)
            attn = find_attention_layers(mdl)
            has_knorm = any(info.get("k_norm") is not None for info in attn.values())
            print(f"\n{label}: {len(attn)} attn layers; {ids.shape[1]} tokens; "
                  f"k_norm={'yes' if has_knorm else 'no (A3==A2)'}")
            forward_fn = select_forward_callable(mdl, ids[:, :256])
        except Exception as e:
            payload["errors"][label] = f"load: {type(e).__name__}: {e}"
            print(f"  !! {label} load FAILED: {type(e).__name__}: {e}")
            _save(payload)
            continue

        for ctx in contexts:
            key = f"{label}@{ctx}"
            print(f"\n{'-' * 74}\n{key}  (N={N_RANDOM_WINDOWS} random windows)")
            try:
                starts = select_random_windows(ids.shape[1], N_RANDOM_WINDOWS, RANDOM_SEED, ctx)
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                fp16 = [ppl_single_window(forward_fn, mdl, ids, b, ctx) for b in starts]
                rec = {"ctx": ctx, "window_starts": starts,
                       "fp16_ref_ppl_mean": sum(fp16) / len(fp16), "recipes": {}}
                for recipe in RECIPES:
                    rec["recipes"][recipe] = run_recipe(forward_fn, mdl, ids, attn, ctx,
                                                        recipe, starts, fp16, has_knorm)
                    payload["cells"][key] = rec
                    _save(payload)
                gaps, verdict = cross_recipe_gaps(rec["recipes"]["nsn"], rec["recipes"]["kivi"],
                                                  N_RANDOM_WINDOWS)
                rec["cross_recipe_gaps"] = gaps
                rec["verdict"] = verdict
                rec["vram_peak_gb"] = (torch.cuda.max_memory_allocated() / 1e9
                                       if torch.cuda.is_available() else 0.0)
                payload["cells"][key] = rec
                _save(payload)
                gp = verdict.get("gap_at_prod") or {}
                print(f"  GAP @ {verdict['production_config']}: "
                      f"{gp.get('mean', float('nan')):+.3f} pp "
                      f"[{gp.get('ci_lo', float('nan')):+.3f}, {gp.get('ci_hi', float('nan')):+.3f}]  "
                      f"-> VERDICT: {verdict['verdict'].upper()}")
            except Exception as e:
                payload["errors"][key] = f"{type(e).__name__}: {e}"
                print(f"  !! {key} FAILED: {type(e).__name__}: {e}\n{traceback.format_exc()}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            _save(payload)

        del mdl, tok, ids, attn
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n{'=' * 74}\nDONE. cells: {list(payload['cells'])}  errors: {list(payload['errors'])}")
    for key, rec in payload["cells"].items():
        v = rec.get("verdict", {})
        print(f"  {key:>26}: {v.get('verdict', '?').upper()}")
    print(f"Saved: {OUT_PATH}")
    print(f"{'=' * 74}")


# =============================================================
# GPU-free logic self-tests
# =============================================================

def _run_self_tests():
    torch.manual_seed(0)
    B, T, n_kv, hd = 1, 600, 4, 16
    k = torch.randn(B, T, n_kv * hd)
    A0 = {"causal": False, "group": None, "residual": 0, "post_norm": False}
    A1 = {"causal": True, "group": 128, "residual": 0, "post_norm": False}
    A2 = {"causal": True, "group": 128, "residual": 128, "post_norm": False}

    # 1. causal axis is a real change: A0 != A1 for both recipes.
    assert (quant_k_kivi(k, 5, A0) - quant_k_kivi(k, 5, A1)).abs().max() > 1e-4, "kivi A0==A1"
    assert (quant_k_nsn(k, 5, A0, n_kv, hd) - quant_k_nsn(k, 5, A1, n_kv, hd)).abs().max() > 1e-4, "nsn A0==A1"

    # 2. residual buffer (A2): the last 128 tokens are bit-exact FP16 for kivi.
    q2 = quant_k_kivi(k, 5, A2)
    assert torch.equal(q2[:, T - 128:, :], k[:, T - 128:, :]), "kivi A2 residual not exact"
    assert not torch.equal(q2[:, :T - 128, :], k[:, :T - 128, :]), "kivi A2 quantized nothing"

    # 3. NSN preserves per-token norm under every config (norm separation property).
    for cfg in (A0, A1, A2):
        xq = quant_k_nsn(k, 5, cfg, n_kv, hd).reshape(B, T, n_kv, hd)
        nq = xq.norm(dim=-1)
        nk = k.reshape(B, T, n_kv, hd).norm(dim=-1)
        assert ((nq - nk).abs() / nk.clamp(min=1e-6)).max() < 1e-2, "nsn norm not preserved"

    # 4. plain-asym (kivi) differs from nsn at the same config (the family axis under test).
    assert (quant_k_kivi(k, 5, A0) - quant_k_nsn(k, 5, A0, n_kv, hd)).abs().max() > 1e-3, "kivi==nsn"

    # 5. A3 post-norm layout round-trip (BnTh = Gemma transpose layout).
    x_bth = torch.randn(B, T, n_kv * hd)
    x_bnth = x_bth.reshape(B, T, n_kv, hd).transpose(1, 2).contiguous()
    c, tag = postnorm_to_canonical(x_bnth, n_kv, hd)
    assert tag == "BnTh" and (c - x_bth).abs().max() < 1e-6, "BnTh canonicalization"
    back = postnorm_from_canonical(c, x_bnth, "BnTh", n_kv, hd)
    assert back.shape == x_bnth.shape and (back - x_bnth).abs().max() < 1e-6, "BnTh round-trip"

    # 6. cross-recipe gap verdict logic: survive needs gap CI_lo>0 AND plain out of band.
    nsn = {"A3_post_norm": {"per_window_dppl": [0.1] * 30, "out_of_band": False, "in_band": True}}
    kivi_out = {"A3_post_norm": {"per_window_dppl": [1.6] * 30, "out_of_band": True, "in_band": False}}
    kivi_in = {"A3_post_norm": {"per_window_dppl": [0.2] * 30, "out_of_band": False, "in_band": True}}
    _, v_surv = cross_recipe_gaps(nsn, kivi_out, 30)
    _, v_coll = cross_recipe_gaps(nsn, kivi_in, 30)
    assert v_surv["verdict"] == "survive", f"expected survive, got {v_surv['verdict']}"
    assert v_coll["verdict"] == "collapse", f"expected collapse, got {v_coll['verdict']}"

    print("=== phase_w1d self-tests: ALL PASS ===")
    print("  causal axis real (A0!=A1) for kivi and nsn; A2 residual exact; nsn norm preserved")
    print("  A3 BnTh layout round-trip exact; kivi != nsn (family axis); verdict logic correct")
    print(f"  configs {[c['name'] for c in CONFIGS]}; cells {[(c[1], c[2]) for c in CELLS]}")


if __name__ == "__main__":
    if _IN_COLAB:
        main()
    else:
        _run_self_tests()
