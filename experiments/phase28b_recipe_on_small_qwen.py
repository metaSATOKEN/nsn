# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Phase 28b: Does the v0.3 recipe rescue the Qwen2.5-1.5B
#            INT3 valley discovered in Phase 28?
# ============================================================
# Phase 28 found that naive per-row INT quantization on
# Qwen2.5-1.5B produces a catastrophic spike at INT3:
#
#     bit:      2       3        4       5       6       8
#     dPPL: +73    +5961    +10    +15    +13    -0
#
# (INT3 at +60000 % of baseline PPL.) The spike is at INT3
# rather than INT4, but otherwise has the Qwen2-7B "INT4 valley"
# shape: one bit width is wildly worse than both neighbors.
# Qwen2.5-0.5B did NOT show this pattern and is Mode-2-like
# (monotone improvement with bit width).
#
# The NSN paper §5.3 argues that such failures are driven by
# V-cache per-token L2 norm dispersion and are rescued by the
# v0.3 recipe (qa_nsep_pchan_asym). But that claim has NOT been
# tested at Qwen2.5-1.5B's INT3 spike specifically -- the paper's
# main benchmark tests 8 different models, none of which is a
# small Qwen checkpoint.
#
# This phase is the direct test:
#
#   For Qwen2.5-0.5B and Qwen2.5-1.5B, at INT3 only:
#     (A) naive per-row symmetric absmax      (Phase 28 baseline)
#     (C) symmetric nsep+pchan (paper B v0.2)
#     (D) asymmetric nsep+pchan (paper B v0.3) <- the one to test
#
# Expected outcomes:
#
#   * Both models <5 % of baseline PPL under (D)  -> paper §6.1
#     update: Mode 1 family can be relaxed to n=2 Qwen
#     checkpoints (Qwen2-7B INT4, Qwen2.5-1.5B INT3), both
#     rescued by v0.3.
#   * Qwen2.5-1.5B fails under (D)                 -> significant
#     finding: V-CV mechanism has a blind spot; §5.3 needs a
#     scope caveat. Worth catching before the paper ships.
#   * Everything rescued except one model          -> most likely;
#     we characterize whichever fails and update §6.1 accordingly.
#
# Runtime: ~5-8 min on Colab A100. Small models, 3 methods x 2
# models at INT3 only.
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

import json, math
import torch.nn.functional as F
from datetime import datetime
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# =============================================================
# === CELL V2 === Three quantizers (naive / v0.2 sym / v0.3 asym)
# =============================================================

def qa_naive_perrow_sym(x, bits):
    """(A) Naive per-row symmetric absmax INT. Matches Phase 28."""
    dtype = x.dtype
    xf = x.float()
    qm = 2 ** (bits - 1) - 1
    s = xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / qm
    q = (xf / s).round().clamp(-qm, qm) * s
    return q.to(dtype)


def qa_nsep_pchan_sym(x, bits):
    """(C) Symmetric nsep+pchan (paper B v0.2)."""
    dtype = x.dtype
    xf = x.float()
    n = xf.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    d = xf / n
    qm = 2 ** (bits - 1) - 1
    s = d.abs().amax(dim=0, keepdim=True).clamp(min=1e-12) / qm
    dq = ((d / s).round().clamp(-qm, qm)) * s
    dq = dq / dq.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return (n * dq).to(dtype)


def qa_nsep_pchan_asym(x, bits):
    """(D) Asymmetric nsep+pchan (paper B v0.3, the shipped recipe)."""
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
# === CELL V4 === WikiText-2 PPL runner
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

    return math.exp(total_nll / total_tokens), windows_done


# =============================================================
# === CELL V5 === Run INT3 comparison
# =============================================================

MODELS = [
    ("Qwen/Qwen2.5-0.5B", "Qwen2.5-0.5B"),
    ("Qwen/Qwen2.5-1.5B", "Qwen2.5-1.5B"),
]

METHODS = [
    ("FP16 baseline",      None),
    ("(A) naive INT3",     lambda x: qa_naive_perrow_sym(x, 3)),
    ("(C) sym INT3 (v0.2)", lambda x: qa_nsep_pchan_sym(x, 3)),
    ("(D) asym INT3 (v0.3)", lambda x: qa_nsep_pchan_asym(x, 3)),
]

results = {}

for model_id, label in MODELS:
    try:
        del mdl, tok
    except NameError:
        pass
    gc.collect(); torch.cuda.empty_cache()

    print(f"\n{'=' * 80}")
    print(f"Loading {label}  ({model_id})")
    print(f"{'=' * 80}")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        use_safetensors=True,
        trust_remote_code=True,
    )
    mdl.eval()
    print(f"  VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    per_method = {}
    base_ppl = None
    for method_name, method_fn in METHODS:
        ppl, _ = ppl_with(method_fn, mdl, tok)
        if method_name == "FP16 baseline":
            base_ppl = ppl
            dppl = 0.0; pct = 0.0
        else:
            dppl = ppl - base_ppl
            pct = 100 * dppl / base_ppl
        per_method[method_name] = {
            "ppl": ppl, "delta_ppl": dppl, "pct_of_base": pct,
        }
        print(f"  {method_name:<24s}  ppl = {ppl:>10.4f}   "
              f"ΔPPL = {dppl:+10.4f}   ({pct:+8.2f}% of base)")

    asym_pct = per_method["(D) asym INT3 (v0.3)"]["pct_of_base"]
    naive_pct = per_method["(A) naive INT3"]["pct_of_base"]
    recipe_rescues = (asym_pct < 5.0)
    rescue_ratio = naive_pct / asym_pct if asym_pct > 1e-6 else float('inf')
    print(f"\n  Recipe rescue check (D < 5% of baseline?): "
          f"(D) = {asym_pct:.2f}% vs (A) = {naive_pct:.2f}%  "
          f"-> {'YES rescued' if recipe_rescues else 'NO blind spot'}  "
          f"(ratio A/D = {rescue_ratio:.1f}x)")

    results[label] = {
        "model": model_id,
        "baseline_ppl": base_ppl,
        "per_method": per_method,
        "recipe_rescues": recipe_rescues,
        "rescue_ratio":   rescue_ratio,
    }


# =============================================================
# === CELL V6 === Summary + JSON save
# =============================================================

print("\n" + "=" * 96)
print(f"Phase 28b recipe-rescue summary  "
      f"({datetime.now().isoformat(timespec='seconds')})")
print("=" * 96)

print(f"  {'Model':<14}  {'base':>7}  {'(A) naive':>12}  "
      f"{'(C) sym':>10}  {'(D) asym':>10}  {'rescue?':>8}")
print(f"  {'-' * 14}  {'-' * 7}  {'-' * 12}  {'-' * 10}  {'-' * 10}  {'-' * 8}")

for label in [m[1] for m in MODELS]:
    r = results[label]
    base = r["baseline_ppl"]
    a_pct = r["per_method"]["(A) naive INT3"]["pct_of_base"]
    c_pct = r["per_method"]["(C) sym INT3 (v0.2)"]["pct_of_base"]
    d_pct = r["per_method"]["(D) asym INT3 (v0.3)"]["pct_of_base"]
    rescued = "YES" if r["recipe_rescues"] else "NO"
    print(f"  {label:<14}  {base:>7.2f}  "
          f"{a_pct:>+11.1f}%  {c_pct:>+9.1f}%  {d_pct:>+9.1f}%  "
          f"{rescued:>8}")


out_path = "/content/phase28b_recipe_on_small_qwen.json"
combined = {
    "date":     datetime.now().isoformat(timespec='seconds'),
    "protocol": {
        "task":        "WikiText-2 sliding-window PPL at INT3 only",
        "ctx_len":     1024,
        "stride":      512,
        "max_windows": 10,
        "text_chars":  48000,
        "methods":     [m[0] for m in METHODS],
        "models":      [m[1] for m in MODELS],
        "rescue_rule": ("recipe_rescues = (asym INT3 pct_of_base < 5%). "
                        "Matches the paper B §5.1 usability threshold."),
    },
    "motivation_phase28_reminder": (
        "Phase 28 found Qwen2.5-1.5B has a non-monotone spike at INT3 "
        "under naive quantization: INT2 +73, INT3 +5961, INT4 +10, "
        "INT5 +15. This phase tests whether the v0.3 recipe rescues "
        "the INT3 spike, or whether it is a blind spot for the V-CV "
        "mechanism described in paper B §5.3."
    ),
    "results": results,
    "status":  ("Follow-up to Phase 28. If both Qwen2.5 checkpoints "
                "are rescued by (D), paper §6.1 can be updated to "
                "'Mode-1-style non-monotone failures observed on n=2 "
                "Qwen checkpoints; the v0.3 recipe rescues both'. If "
                "Qwen2.5-1.5B is NOT rescued, the paper has a "
                "previously-unknown blind spot and the V-CV mechanism "
                "claim needs scope-limiting."),
}
with open(out_path, "w") as f:
    json.dump(combined, f, indent=2, default=str)
print(f"\nSaved -> {out_path}")


print("\n" + "=" * 96)
print("READING")
print("=" * 96)
print("""
  Three possible outcomes:

  1. Both models rescued. (D) < 5 % of baseline on both.
     -> Paper §6.1 goes from "n = 1" to "n = 2 Qwen checkpoints,
        both rescued by v0.3". Strong update.

  2. Qwen2.5-1.5B NOT rescued by (D).
     -> Discovery: the V-CV mechanism has a blind spot at
        INT3-valley-type failures. Paper §5.3 / §6.1 / §8 would
        need scope adjustments. BEST caught before the paper
        ships; post-Zenodo would require an erratum.

  3. (C) sym v0.2 rescues but (D) asym v0.3 does not, or vice
     versa. Unlikely but informative: would tell us which
     ingredient (nsep+pchan vs the asym window) is doing the
     work at this specific bit width.
""")
