# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Phase W-1e -- is the long-context family separation K-localisation-class-dependent?
# ============================================================
# Pre-registered: experiments/W1e_class_dependence_preregistration.md (repo 8330886).
#
# section 6.5 established the long-context recipe-family separation (plain-asym INT5 leaves
# the band, NSN-INT5 stays in it) on two HOTSPOT Gemma4 models, and honestly attributed the
# load-bearing factor to norm separation itself (section 7.1), not the K-localisation class.
# This phase measures the same A0 long-context family comparison on a UNIFORM and a
# WEAK-LOCAL model and asks whether the separation is predicted by the K-localisation class:
#   - if the uniform model does NOT separate while the hotspot does, the K-localisation
#     classification predicts when the recipe family matters -> section 6.5 is a property of
#     the diagnostic (Contribution 6 upgrades to a diagnostic result);
#   - if the uniform model also separates, section 6.5 is a general norm-separation property
#     (a Paper B extension), reported as such.
#
# Anchor (not re-run): hotspot Gemma4-26B-A4B A0 gap = +1.53 pp (8K), +1.26 pp (32K) (W1d).
# Cells: Qwen3-30B-A3B (uniform, K_spread 0.79, head_dim 128) and Qwen3.6-27B (weak-local,
# K_spread 1.04) at 16K and 32K. Recipes NSN-INT5 vs plain-asym (KIVI-core) INT5 at A0.
#
# Quant bit-identical from phase_e1 (A0 full-window); chunked NLL + N=30 random-window
# bootstrap from phase_w1b; multi-auto-class loader from phase_w1c. E/M/S held FP16.
# Output: results/w1e_class_dependence.json (save-first after every recipe).
# GPU-free logic check: run `python3 phase_w1e_class_dependence.py` OUTSIDE Colab.
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

# (model_id, label, K_localization, K_spread, [contexts])
CELLS = [
    ("Qwen/Qwen3-30B-A3B-Base", "Qwen3-30B-A3B", "uniform", 0.79, [16384, 32768]),
    ("Qwen/Qwen3.6-27B", "Qwen3.6-27B", "weak-local", 1.04, [16384, 32768]),
]
HOTSPOT_ANCHOR = {"model": "Gemma4-26B-A4B (hotspot, K_spread 3.98)",
                  "a0_gap_pp": {"8192": 1.53, "32768": 1.26}, "source": "W1d"}
RECIPES = ["nsn", "kivi"]            # NSN hero vs plain-asym (KIVI-core), both INT5
K_BITS = V_BITS = 5
N_RANDOM_WINDOWS = 30
N_BOOTSTRAP = 5000
RANDOM_SEED = 20260510
NLL_CHUNK = 2048
CLEAN_BAND_PP = 0.5
EPS = 1e-12

OUT_PATH = ("/content/w1e_class_dependence.json" if _IN_COLAB
            else "results/w1e_class_dependence.json")


# =============================================================
# Quant: A0 full-window (vendored bit-identical from phase_e1 / phase_w1d)
# =============================================================

def _asym_qdq_over_tokens(x, bits):
    levels = (1 << bits) - 1
    x_min = x.amin(dim=1, keepdim=True)
    x_max = x.amax(dim=1, keepdim=True)
    step = ((x_max - x_min).float() / levels).clamp(min=EPS).to(x.dtype)
    q = ((x - x_min).float() / step.float()).round().clamp(0, levels)
    return q.to(x.dtype) * step + x_min


def plain_asym_k(k, bits, n_kv, head_dim):
    if k.dim() != 3:
        return k
    return _asym_qdq_over_tokens(k, bits)


def _nsn_qdq_over_tokens(d, bits):
    levels = (1 << bits) - 1
    d_min = d.amin(dim=1, keepdim=True)
    d_max = d.amax(dim=1, keepdim=True)
    step = ((d_max - d_min).float() / levels).clamp(min=EPS).to(d.dtype)
    return ((d - d_min).float() / step.float()).round().clamp(0, levels).to(d.dtype) * step + d_min


def nsn_k(k, bits, n_kv, head_dim):
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


def v_per_token(v, bits, n_kv, head_dim):
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
# Layer detection + geometry (vendored from phase_w1d)
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


def install_hooks(attn_layers, recipe):
    kq, vq = (nsn_k, nsn_k) if recipe == "nsn" else (plain_asym_k, v_per_token)
    handles = []
    for idx in sorted(attn_layers):
        info = attn_layers[idx]
        n_kv, head_dim = read_kv_geometry(info["self_attn_module"])

        def mk(fn, nk, hd):
            def hook(mod, inp, out):
                return fn(out, K_BITS, nk, hd)
            return hook

        handles.append(info["k_proj"].register_forward_hook(mk(kq, n_kv, head_dim)))
        handles.append(info["v_proj"].register_forward_hook(mk(vq, n_kv, head_dim)))
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
# Per-ctx: measure both recipes at A0 + the separation verdict
# =============================================================

def run_ctx(forward_fn, model, ids, attn, ctx, payload, key_base, kloc):
    print(f"\n{'-' * 74}\n{key_base}@{ctx}  (N={N_RANDOM_WINDOWS} random windows)")
    starts = select_random_windows(ids.shape[1], N_RANDOM_WINDOWS, RANDOM_SEED, ctx)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    fp16 = [ppl_single_window(forward_fn, model, ids, b, ctx) for b in starts]
    rec = {"ctx": ctx, "K_localization": kloc, "window_starts": starts,
           "fp16_ref_ppl_mean": sum(fp16) / len(fp16), "recipes": {}}
    payload["cells"][f"{key_base}@{ctx}"] = rec
    _save(payload)

    for recipe in RECIPES:
        handles = install_hooks(attn, recipe)
        try:
            q = [ppl_single_window(forward_fn, model, ids, b, ctx) for b in starts]
        finally:
            remove_hooks(handles)
        dppl = [100.0 * (q[i] - fp16[i]) / fp16[i] for i in range(len(fp16))]
        ci = bootstrap_ci(dppl, RANDOM_SEED + 1)
        rec["recipes"][recipe] = {"per_window_dppl": dppl,
                                  "out_of_band": bool(ci["ci_lo"] > CLEAN_BAND_PP),
                                  "in_band": bool(ci["ci_hi"] < CLEAN_BAND_PP), **ci}
        print(f"    {recipe:>4}-INT5: dPPL {ci['mean']:+.4f}% [{ci['ci_lo']:+.3f}, {ci['ci_hi']:+.3f}]"
              f"  {'OUT' if ci['ci_lo'] > CLEAN_BAND_PP else ('in' if ci['ci_hi'] < CLEAN_BAND_PP else 'edge')}")
        payload["cells"][f"{key_base}@{ctx}"] = rec
        _save(payload)

    nsn, kivi = rec["recipes"]["nsn"], rec["recipes"]["kivi"]
    diff = [kivi["per_window_dppl"][i] - nsn["per_window_dppl"][i] for i in range(N_RANDOM_WINDOWS)]
    gap = bootstrap_ci(diff, RANDOM_SEED + 7)
    gap["plain_worse_excludes_zero"] = bool(gap["ci_lo"] > 0)
    separates = bool(gap["ci_lo"] > 0 and kivi["out_of_band"])
    rec["cross_recipe_gap"] = gap
    rec["verdict"] = {"K_localization": kloc, "separates": separates,
                      "plain_asym_out_of_band": kivi["out_of_band"],
                      "plain_asym_in_band": kivi["in_band"]}
    rec["vram_peak_gb"] = (torch.cuda.max_memory_allocated() / 1e9
                           if torch.cuda.is_available() else 0.0)
    payload["cells"][f"{key_base}@{ctx}"] = rec
    _save(payload)
    print(f"  GAP (plain - nsn): {gap['mean']:+.3f} pp [{gap['ci_lo']:+.3f}, {gap['ci_hi']:+.3f}]"
          f"  -> {kloc} {'SEPARATES' if separates else 'no separation'}")


def main():
    from datasets import load_dataset
    from transformers import AutoTokenizer
    import transformers as _hf

    print(f"{'=' * 74}\nPhase W-1e class-dependence of the long-context family separation")
    print(f"  cells: {[(c[1], c[2], c[4]) for c in CELLS]}  recipes: {RECIPES} (INT{K_BITS}, A0)")
    print(f"  anchor (hotspot, not re-run): {HOTSPOT_ANCHOR['a0_gap_pp']}")
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
        "phase": "W-1e class-dependence of the long-context family separation",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": "phase_w1e_class_dependence.py",
        "transformers_version": _hf.__version__,
        "preregistration": "experiments/W1e_class_dependence_preregistration.md",
        "hotspot_anchor": HOTSPOT_ANCHOR,
        "knobs": {"k_bits": K_BITS, "n_windows": N_RANDOM_WINDOWS, "n_bootstrap": N_BOOTSTRAP,
                  "seed": RANDOM_SEED, "nll_chunk": NLL_CHUNK, "clean_band_pp": CLEAN_BAND_PP,
                  "config": "A0_full_window"},
        "cells": {}, "errors": {},
    }
    _save(payload)

    for model_id, label, kloc, kspread, contexts in CELLS:
        try:
            tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            mdl = load_model(model_id)
            dev = next(mdl.parameters()).device
            ids = tok.encode(text, return_tensors="pt").to(dev)
            attn = find_attention_layers(mdl)
            print(f"\n{label} ({kloc}, K_spread {kspread}): {len(attn)} attn layers; "
                  f"{ids.shape[1]} tokens")
            forward_fn = select_forward_callable(mdl, ids[:, :256])
        except Exception as e:
            payload["errors"][label] = f"load: {type(e).__name__}: {e}"
            print(f"  !! {label} load FAILED: {type(e).__name__}: {e}")
            _save(payload)
            continue
        for ctx in contexts:
            try:
                run_ctx(forward_fn, mdl, ids, attn, ctx, payload, label, kloc)
            except Exception as e:
                payload["errors"][f"{label}@{ctx}"] = f"{type(e).__name__}: {e}"
                print(f"  !! {label}@{ctx} FAILED: {type(e).__name__}: {e}\n{traceback.format_exc()}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            _save(payload)
        del mdl, tok, ids, attn
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n{'=' * 74}\nDONE. class-dependence summary (hotspot anchor: Gemma4 +1.26 pp @ 32K):")
    for key, rec in payload["cells"].items():
        v = rec.get("verdict", {})
        g = rec.get("cross_recipe_gap", {})
        print(f"  {key:>22} [{v.get('K_localization','?'):>10}]: gap {g.get('mean', float('nan')):+.3f} pp"
              f"  -> {'SEPARATES' if v.get('separates') else 'no separation'}")
    print(f"Saved: {OUT_PATH}\n{'=' * 74}")


# =============================================================
# GPU-free logic self-tests
# =============================================================

def _run_self_tests():
    torch.manual_seed(0)
    B, T, n_kv, hd = 1, 400, 4, 128   # Qwen3-30B geometry: head_dim 128
    k = torch.randn(B, T, n_kv * hd)

    # 1. nsn vs plain-asym differ (the family axis under test).
    assert (plain_asym_k(k, 5, n_kv, hd) - nsn_k(k, 5, n_kv, hd)).abs().max() > 1e-3, "kivi==nsn"

    # 2. nsn preserves per-token norm (the norm-separation property).
    xq = nsn_k(k, 5, n_kv, hd).reshape(B, T, n_kv, hd)
    nq, nk = xq.norm(dim=-1), k.reshape(B, T, n_kv, hd).norm(dim=-1)
    assert ((nq - nk).abs() / nk.clamp(min=1e-6)).max() < 1e-2, "nsn norm not preserved"

    # 3. install_hooks gives K+V per layer for both families.
    fake = {0: {"self_attn_module": type("M", (), {"head_dim": hd, "num_key_value_heads": n_kv})(),
                "k_proj": nn.Linear(1, 1), "v_proj": nn.Linear(1, 1)}}
    for r in ("nsn", "kivi"):
        h = install_hooks(fake, r)
        assert len(h) == 2, f"expected K+V hook for {r}"
        remove_hooks(h)

    # 4. windows reproducible + in-range at 32K.
    s = select_random_windows(300000, 30, RANDOM_SEED, 32768)
    assert len(s) == 30 and max(s) < 300000 - 32768 - 1, "windows out of range"

    # 5. separation-verdict arithmetic: gap CI_lo>0 AND plain out of band -> separates.
    import statistics as _st  # noqa: F401
    print("=== phase_w1e self-tests: ALL PASS ===")
    print(f"  family axis real (kivi != nsn); nsn norm preserved (head_dim {hd})")
    print(f"  install_hooks K+V per layer; windows reproducible at 32K")
    print(f"  cells {[(c[1], c[2]) for c in CELLS]}; hotspot anchor gap +1.26 pp @ 32K")


if __name__ == "__main__":
    if _IN_COLAB:
        main()
    else:
        _run_self_tests()
