# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Phase 25: Fractional-bit KV cache quantization (EXPLORATORY)
# ============================================================
# NOTE: This phase is a sandbox probe and is NOT integrated into
# nsn/nsn.md. It may become a Paper C extension or a v0.5
# feature later, depending on what the data says.
#
# Motivation. The NSN paper §5.1 / §5.5 plots the 4-method INT3 ablation
# at exactly 3.0 bits/value. Phase 1-4 characterized the naive INT
# sweep at integer bit widths 2, 3, 4, 5, 6, 8. Nothing in the
# paper covers *fractional* bit widths -- in particular the obvious
# interpolation points INT2.5 and INT3.5. Both are reachable with a
# simple per-channel bit mix:
#
#     INT2.5  :=  first D/2 channels at 2 bits, rest at 3 bits
#     INT3.5  :=  first D/2 channels at 3 bits, rest at 4 bits
#
# The effective bit rate is (bits_low + bits_high) / 2.
#
# Why it's interesting to check:
#
#   1. The Pareto plot in fig04 has a knee at 3 bits under v0.3 asym.
#      INT2.5 and INT3.5 fill in the most-curved region of that knee
#      and should tell us whether the curve is smooth or has kinks
#      (e.g., another "valley" analogous to Qwen2-7B's INT4 anomaly
#      at a non-integer bit width).
#
#   2. INT3.5 on Qwen2-7B is of specific interest: naive INT4 gives
#      +1382 dPPL on that model, while naive INT3 gives +12 and naive
#      INT5 gives +5. Does INT3.5 inherit the INT4 valley (because
#      half the channels land at INT4), or does the asym nsep+pchan
#      machinery neutralize it the way it neutralizes naive INT4 at
#      pure b=3? If the valley is channel-structure-driven, INT3.5
#      should be fine; if it is bit-grid-driven, INT3.5 might inherit
#      some of the damage.
#
#   3. INT2.5 is the first sub-INT3 point we would measure with the
#      v0.3 recipe. The NSN paper explicitly does not claim anything below
#      INT3, so INT2.5 data tells us how steep the quality cliff is
#      beyond the recipe's advertised operating range.
#
# Mixed-bit implementation. Uses a minimal extension of
# qa_nsep_pchan_asym: after nsep + per-channel (min, max), we still
# compute one (m, M) pair per channel but scale the quantization
# grid differently per channel -- the first D/2 channels get
# levels = 2**bits_low - 1 and the rest get 2**bits_high - 1.
# Everything else (the nsep step, re-normalization) is identical
# to the v0.3 default.
#
# Models: Qwen2-7B (Mode 1, has INT4 valley), Llama-3.1-8B
# (Mode 2 canonical), Pythia-6.9B (high compression sensitivity).
#
# Bit widths: 2.0, 2.5, 3.0, 3.5, 4.0 (all under asym nsep+pchan).
#
# Protocol: WikiText-2 sliding window, ctx=1024, stride=512,
# 10 windows, same as Phases 14/16/17. The INT3 and INT4 data
# points should roughly match the published numbers; disagreement
# would flag an implementation bug.
#
# Runtime estimate: ~25-30 min on Colab A100 (3 models x 6 runs
# each -- FP16 baseline + 5 bit widths -- at ~1 min per PPL run
# plus ~3-5 min model load per model).
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
import torch.nn.functional as F
from datetime import datetime
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# =============================================================
# === CELL V2 === Quantizers: v0.3 asym + fractional-bit mixed
# =============================================================

def qa_nsep_pchan_asym(x, bits):
    """v0.3 asymmetric nsep+pchan (the NSN paper §3.5)."""
    dtype = x.dtype
    xf = x.float()
    n = xf.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    d = xf / n
    d_min = d.amin(dim=0, keepdim=True)
    d_max = d.amax(dim=0, keepdim=True)
    d_range = (d_max - d_min).clamp(min=1e-12)
    levels = 2**bits - 1
    step = d_range / levels
    dq = ((d - d_min) / step).round().clamp(0, levels) * step + d_min
    dq = dq / dq.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return (n * dq).to(dtype)


def qa_nsep_pchan_asym_mixed(x, bits_low, bits_high, split_idx=None):
    """Fractional-bit asymmetric nsep+pchan.

    Extends qa_nsep_pchan_asym by applying a per-channel bit budget:
    the first ``split_idx`` channels use ``bits_low``, the remaining
    channels use ``bits_high``. Everything else (nsep, per-channel
    (min, max), direction re-normalization) is identical to the
    v0.3 default.

    Args
    ----
    x : Tensor [..., T, D]
    bits_low, bits_high : int
        Bit widths for the low- and high-precision channel groups.
    split_idx : int or None
        Number of channels in the low-precision group.
        If None, defaults to D // 2 (even 50/50 split).

    The effective bit rate is
        (bits_low * split_idx + bits_high * (D - split_idx)) / D
    plus the usual FP16 norm + per-channel metadata overhead.
    """
    dtype = x.dtype
    xf = x.float()
    D = xf.shape[-1]
    if split_idx is None:
        split_idx = D // 2

    # nsep (identical to v0.3)
    n = xf.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    d = xf / n

    # per-channel (min, max) over token axis
    # For 2D [T, D] input, dim=0 is tokens; for >2D we still pool
    # token axis only, like the original qa_nsep_pchan_asym above.
    d_min = d.amin(dim=0, keepdim=True)     # [1, D]
    d_max = d.amax(dim=0, keepdim=True)     # [1, D]

    # Per-channel levels vector, shape [D]
    levels_low  = 2**bits_low  - 1
    levels_high = 2**bits_high - 1
    levels = torch.full((D,), float(levels_high),
                        device=xf.device, dtype=xf.dtype)
    levels[:split_idx] = float(levels_low)

    step = ((d_max - d_min) / levels).clamp(min=1e-12)  # [1, D]

    # Quantize with per-channel clamp bounds
    q = ((d - d_min) / step).round()
    q = torch.minimum(torch.maximum(q, torch.zeros_like(q)),
                      levels.expand_as(q))

    dq = d_min + q * step
    dq = dq / dq.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return (n * dq).to(dtype)


def _self_test_mixed():
    """Sanity: mixed(b, b) must equal plain asym at bits=b."""
    x = torch.randn(128, 64, dtype=torch.float16)
    for b in [2, 3, 4]:
        a = qa_nsep_pchan_asym(x, b)
        m = qa_nsep_pchan_asym_mixed(x, b, b)
        if not torch.allclose(a.float(), m.float(), atol=1e-4):
            max_diff = (a.float() - m.float()).abs().max().item()
            raise RuntimeError(
                f"mixed({b}, {b}) != plain asym({b}); max_diff={max_diff:.2e}")
    print("  self-test passed: mixed(b, b) == plain asym(b) for b in {2,3,4}")

_self_test_mixed()


# =============================================================
# === CELL V3 === KV cache plumbing
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

def compress_cache(past, method_fn):
    if method_fn is None:
        return
    for li in range(n_cache_layers(past)):
        if not _layer_has_kv(past, li):
            continue
        ok, ov = get_kv(past, li)
        ok = ok.clone(); ov = ov.clone()
        nk = torch.zeros_like(ok); nv = torch.zeros_like(ov)
        for h in range(ok.shape[1]):
            nk[0, h] = method_fn(ok[0, h]).to(ok.dtype)
            nv[0, h] = method_fn(ov[0, h]).to(ov.dtype)
        set_kv(past, li, nk, nv)


# =============================================================
# === CELL V4 === WikiText-2 sliding-window PPL runner
# =============================================================

ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
wtext = "\n\n".join([x for x in ds["text"] if x.strip()])[:48000]
print(f"WikiText-2 text length: {len(wtext)} chars")


def ppl_with(method_fn, mdl, tok, ctx_len=1024, stride=512, max_windows=10):
    ids = tok.encode(wtext, return_tensors="pt").to("cuda")
    T = ids.shape[1]
    total_nll = 0.0; total_tokens = 0; windows_done = 0

    with torch.inference_mode():
        for begin in range(0, T - ctx_len, stride):
            if windows_done >= max_windows:
                break
            window = ids[:, begin:begin + ctx_len]
            split = ctx_len // 2
            context = window[:, :split]
            targets = window[:, split:]

            out_ctx = mdl(context, use_cache=True)
            past = out_ctx.past_key_values
            first_logits = out_ctx.logits[:, -1:, :]
            compress_cache(past, method_fn)
            out_tgt = mdl(targets, past_key_values=past, use_cache=False)
            rest_logits = out_tgt.logits[:, :-1, :]
            all_logits = torch.cat([first_logits, rest_logits], dim=1)
            log_probs = F.log_softmax(all_logits[0].float(), dim=-1)
            nll = -log_probs.gather(-1, targets[0].unsqueeze(-1)).squeeze(-1)
            total_nll += nll.sum().item()
            total_tokens += targets.shape[1]
            windows_done += 1

    import math
    return math.exp(total_nll / total_tokens), windows_done


# =============================================================
# === CELL V5 === Bit-width configurations
# =============================================================

# Each config is (label, effective_bits, method_fn).
# method_fn is None => FP16 baseline (no cache compression).
#
# 2.5 := half INT2, half INT3
# 3.5 := half INT3, half INT4
BIT_CONFIGS = [
    ("FP16 baseline",   16.0, None),
    ("asym INT2",        2.0, lambda x: qa_nsep_pchan_asym(x, 2)),
    ("asym INT2.5",      2.5, lambda x: qa_nsep_pchan_asym_mixed(x, 2, 3)),
    ("asym INT3",        3.0, lambda x: qa_nsep_pchan_asym(x, 3)),
    ("asym INT3.5",      3.5, lambda x: qa_nsep_pchan_asym_mixed(x, 3, 4)),
    ("asym INT4",        4.0, lambda x: qa_nsep_pchan_asym(x, 4)),
]


MODELS = [
    ("Qwen/Qwen2-7B",            "Qwen2-7B (Mode 1)"),
    ("meta-llama/Llama-3.1-8B",  "Llama-3.1-8B (Mode 2)"),
    ("EleutherAI/pythia-6.9b",   "Pythia-6.9B"),
]


# =============================================================
# === CELL V6 === Run all (model, bit) cells
# =============================================================

results = {}

for model_id, label in MODELS:
    try:
        del mdl, tok
    except NameError:
        pass
    gc.collect(); torch.cuda.empty_cache()

    print(f"\n{'='*80}")
    print(f"Loading {label}  ({model_id})")
    print(f"{'='*80}")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        use_safetensors=True,
        trust_remote_code=True,
    )
    mdl.eval()
    print(f"  VRAM used: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    per_bit = {}
    baseline_ppl = None
    for cfg_label, bits_eff, method_fn in BIT_CONFIGS:
        ppl, nwin = ppl_with(method_fn, mdl, tok)
        if cfg_label == "FP16 baseline":
            baseline_ppl = ppl
            dppl = 0.0
            pct = 0.0
        else:
            dppl = ppl - baseline_ppl
            pct = 100.0 * dppl / baseline_ppl
        per_bit[cfg_label] = {
            "effective_bits": bits_eff,
            "ppl":            ppl,
            "delta_ppl":      dppl,
            "pct_of_base":    pct,
            "n_windows":      nwin,
        }
        print(f"  {cfg_label:<22s}  ({bits_eff:>4.1f} b)  "
              f"ppl = {ppl:>8.3f}   ΔPPL = {dppl:+.4f}   "
              f"({pct:+.2f}% of base)")

    results[label] = {
        "model":        model_id,
        "baseline_ppl": baseline_ppl,
        "per_bit":      per_bit,
    }


# =============================================================
# === CELL V7 === Summary + JSON save
# =============================================================

print("\n" + "=" * 96)
print(f"Phase 25 fractional-bit summary  "
      f"({datetime.now().isoformat(timespec='seconds')})")
print("=" * 96)

# Pivot table: rows=models, cols=bit configs (excluding baseline)
header = f"  {'Model':<24s}  "
header += "  ".join(f"{lab.replace('asym ',''):>12s}"
                    for lab, _, _ in BIT_CONFIGS if lab != "FP16 baseline")
print(header)
print("  " + "-" * (len(header) - 2))

for label in [m[1] for m in MODELS]:
    row = f"  {label:<24s}  "
    for cfg_label, _, _ in BIT_CONFIGS:
        if cfg_label == "FP16 baseline":
            continue
        pct = results[label]["per_bit"][cfg_label]["pct_of_base"]
        row += f"  {pct:>10.2f}%"
    print(row)


out_path = "/content/phase25_fractional_bits.json"
combined = {
    "date":     datetime.now().isoformat(timespec='seconds'),
    "protocol": {
        "task":             "WikiText-2 sliding-window perplexity",
        "ctx_len":          1024,
        "stride":           512,
        "max_windows":      10,
        "text_chars":       48000,
        "quantizer":        "qa_nsep_pchan_asym + fractional-bit extension",
        "fractional_mix":   ("channel-wise 50/50: first D/2 channels at "
                             "bits_low, remaining D/2 channels at bits_high; "
                             "split_idx = D // 2"),
        "bit_configs":      [lab for lab, _, _ in BIT_CONFIGS],
        "models":           [m[1] for m in MODELS],
    },
    "results": results,
    "status":  ("EXPLORATORY / sandbox. Not integrated into "
                "nsn/nsn.md. May become a Paper C extension or "
                "a v0.5 feature depending on what the Pareto curve shape "
                "says."),
}
with open(out_path, "w") as f:
    json.dump(combined, f, indent=2, default=str)
print(f"\nSaved -> {out_path}")


print("\n" + "=" * 96)
print("READING")
print("=" * 96)
print("""
  What to look for:

  1. Smoothness of the Pareto curve between INT2, INT3, INT4 under
     asym nsep+pchan. Smooth interpolation at INT2.5 / INT3.5
     suggests the recipe's quality is bit-width-monotone in the
     aggregate; kinks would indicate a bit-grid-structural effect.

  2. Qwen2-7B (Mode 1) INT3.5 specifically. Naive INT4 on this model
     is +1382 ΔPPL (INT4 valley of death). Under asym nsep+pchan the
     naive INT4 is rescued -- we do not expect INT4-valley behavior
     at INT3.5 either, but if we see it that would identify a
     residual valley structure the recipe does not fully eliminate.

  3. INT2.5 as the first sub-INT3 data point under the recipe. Paper
     B §6 states the quality cliff "below INT3" without characterizing
     it; INT2.5 tells us how steep the cliff actually is on each
     model. A graceful drop between INT2.5 and INT2 would suggest
     sub-INT3 deployment is possible with care; a sharp cliff
     confirms the paper's "INT3 bit floor" claim.

  4. Llama-3.1-8B and Pythia-6.9B as Mode-2 controls. If they show
     monotone interpolation while Qwen2-7B shows a kink, the kink
     is architectural rather than protocol-level.
""")
