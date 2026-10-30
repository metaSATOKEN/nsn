# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Phase 19: Does the asym-only catastrophic failure generalize?
# ============================================================
# Phase 18 found a surprising pattern on 3 canonical models:
#
#   (A) naive  per-row sym   INT3:  Qwen2 +11.9, Llama +0.45, Pythia +22.5
#   (B) asym per-channel no nsep:    Qwen2 +0.36, Llama +10.9, Pythia +0.63
#   (C) nsep+pchan sym    INT3 (v0.2): Qwen2 +0.40, Llama +0.25, Pythia +2.38
#   (D) nsep+pchan asym   INT3 (v0.3): Qwen2 +0.10, Llama +0.11, Pythia +0.10
#
# The unexpected finding is (B): asymmetric-per-channel WITHOUT
# norm separation CATASTROPHICALLY fails on Llama-3.1-8B at
# +10.9 PPL, WORSE than the naive per-row baseline by 24x.
# Meanwhile Qwen2-7B (Mode 1) and Pythia-6.9B (Mode 2) both
# survive asym-only. If this is a genuine pattern rather than
# a Llama idiosyncrasy, it means norm separation is structurally
# essential for industry-standard modern Transformer recipes --
# a paper-grade finding.
#
# This phase tests 5 more models chosen to triangulate which
# architectural features correlate with the asym-only failure:
#
#   Mistral-7B-v0.1        (Llama-style arch, Mode 2; tests
#                            whether the pattern holds on the
#                            most Llama-like peer)
#   Qwen2.5-14B             (post-Qwen recipe, Mode 2)
#   GPT-J-6B                (2021 arch, head_dim=256; older,
#                            different; control case)
#   Qwen3.5-9B              (2026 hybrid attention; tests
#                            on-trend architecture)
#   gemma-4-E4B             (2026 multimodal; different
#                            training distribution)
#
# Each model runs the same 4-method ablation as Phase 18.
# Combined with Phase 18 this gives 8-model coverage for the
# ablation story.
#
# Runtime: ~70-90 min on Colab A100.
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

import json, numpy as np
import torch.nn.functional as F
from datetime import datetime
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# =============================================================
# === CELL V2 === Four-method quantizer zoo (identical to Phase 18)
# =============================================================

def qa_naive_perrow_sym(x, bits):
    """(A) Industry baseline: per-row symmetric absmax."""
    dtype = x.dtype
    xf = x.float()
    qm = 2**(bits-1) - 1
    s = xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / qm
    return (((xf / s).round().clamp(-qm, qm)) * s).to(dtype)


def qa_asym_pchan_no_nsep(x, bits):
    """(B) Asymmetric per-channel, NO norm separation."""
    dtype = x.dtype
    xf = x.float()
    d_min = xf.amin(dim=0, keepdim=True)
    d_max = xf.amax(dim=0, keepdim=True)
    d_range = (d_max - d_min).clamp(min=1e-12)
    levels = 2**bits - 1
    step = d_range / levels
    dq = ((xf - d_min) / step).round().clamp(0, levels) * step + d_min
    return dq.to(dtype)


def qa_nsep_pchan_sym(x, bits):
    """(C) nsn v0.2 default: nsep + per-channel absmax (symmetric)."""
    dtype = x.dtype
    xf = x.float()
    n = xf.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    d = xf / n
    qm = 2**(bits-1) - 1
    s = d.abs().amax(dim=0, keepdim=True).clamp(min=1e-12) / qm
    dq = ((d / s).round().clamp(-qm, qm)) * s
    dq = dq / dq.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return (n * dq).to(dtype)


def qa_nsep_pchan_asym(x, bits):
    """(D) nsn v0.3 default: nsep + per-channel min/max (asymmetric)."""
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


# =============================================================
# === CELL V3 === Cache + PPL (with hybrid-skip for Qwen3.5)
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


ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
wtext = "\n\n".join([x for x in ds["text"] if x.strip()])[:48000]
print(f"WikiText-2 text length: {len(wtext)} chars")


def ppl_with(method_fn, ctx_len=1024, stride=512, max_windows=10):
    ids = tok.encode(wtext, return_tensors="pt").to("cuda")
    T = ids.shape[1]
    total_nll = 0.0; total_tokens = 0; windows_done = 0
    with torch.inference_mode():
        for begin in range(0, T - ctx_len, stride):
            if windows_done >= max_windows: break
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
            del out_ctx, out_tgt, past
            gc.collect(); torch.cuda.empty_cache()
    mean_nll = total_nll / max(total_tokens, 1)
    return float(np.exp(mean_nll)), total_tokens


METHODS = [
    ("baseline (FP16)",                 None,                                  "16"),
    ("(A) naive per-row sym INT3",      lambda x: qa_naive_perrow_sym(x, 3),   "3"),
    ("(B) asym per-channel, no nsep",   lambda x: qa_asym_pchan_no_nsep(x, 3), "3"),
    ("(C) nsep+pchan sym INT3 (v0.2)",  lambda x: qa_nsep_pchan_sym(x, 3),     "3"),
    ("(D) nsep+pchan asym INT3 (v0.3)", lambda x: qa_nsep_pchan_asym(x, 3),    "3"),
]


def run_model(model_id, label, extra_note=None):
    global mdl, tok
    try:
        del mdl, tok
    except NameError:
        pass
    gc.collect(); torch.cuda.empty_cache()

    print(f"\n{'='*80}")
    print(f"Loading {label} ({model_id}) ...")
    print(f"{'='*80}")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto",
        use_safetensors=True, trust_remote_code=True,
    )
    mdl.eval()
    print(f"VRAM used: {torch.cuda.memory_allocated()/1e9:.1f} GB")
    if extra_note:
        print(f"  note: {extra_note}")

    print(f"\n  {'Method':<36} | {'bits':>5} | {'PPL':>10} | {'ΔPPL':>10}")
    print(f"  " + "-"*68)

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
        print(f"  {name:<36} | {bits:>5} | {ppl:>10.4f} | {dstr:>10}")

    naive     = results["(A) naive per-row sym INT3"]["delta"]
    asym_only = results["(B) asym per-channel, no nsep"]["delta"]
    sym_full  = results["(C) nsep+pchan sym INT3 (v0.2)"]["delta"]
    asym_full = results["(D) nsep+pchan asym INT3 (v0.3)"]["delta"]

    # Catastrophe flag: asym-only worse than naive?
    asym_only_catastrophic = asym_only > naive
    print(f"\n  asym-only WORSE than naive? "
          f"{'YES  ← norm-sep essential on this model' if asym_only_catastrophic else 'no'}")
    print(f"    naive     +{naive:.4f}  vs  asym-only +{asym_only:.4f}")

    return {
        "label":             label,
        "model":             model_id,
        "baseline_ppl":      baseline_ppl,
        "results":           results,
        "delta_naive":       naive,
        "delta_asym_only":   asym_only,
        "delta_sym_full":    sym_full,
        "delta_asym_full":   asym_full,
        "asym_only_catastrophic_vs_naive": asym_only_catastrophic,
    }


# =============================================================
# === CELL V4 === Mistral-7B-v0.1 (Llama-style Mode 2)
# =============================================================

mistral_res = run_model(
    "mistralai/Mistral-7B-v0.1", "Mistral-7B-v0.1",
    extra_note="Llama-style RoPE + SwiGLU; closest peer to Llama-3.1-8B in the sample",
)


# =============================================================
# === CELL V5 === Qwen2.5-14B (post-Qwen recipe Mode 2)
# =============================================================

qwen25_res = run_model(
    "Qwen/Qwen2.5-14B", "Qwen2.5-14B",
    extra_note="Post-Qwen2 recipe; low baseline PPL; Llama-family architecture",
)


# =============================================================
# === CELL V6 === GPT-J-6B (2021 control case)
# =============================================================

gptj_res = run_model(
    "EleutherAI/gpt-j-6b", "GPT-J-6B",
    extra_note="2021 arch, head_dim=256, no GQA, no SwiGLU; different from modern Llama-family",
)


# =============================================================
# === CELL V7 === Qwen3.5-9B (2026 hybrid attention)
# =============================================================

qwen35_res = run_model(
    "Qwen/Qwen3.5-9B", "Qwen3.5-9B",
    extra_note="2026 hybrid attn (8 standard + 24 linear layers); Llama-family recipe",
)


# =============================================================
# === CELL V8 === Gemma 4-E4B (2026 multimodal)
# =============================================================

gemma4_res = run_model(
    "google/gemma-4-E4B", "gemma-4-E4B",
    extra_note="2026 multimodal; fed text-only",
)


# =============================================================
# === CELL V9 === 8-model ablation synthesis (Phase 18 + 19)
# =============================================================

# Phase 18 priors (re-used so Phase 19 yields the full 8-model table
# in one place).
PHASE18 = [
    {"label": "Qwen2-7B",       "baseline_ppl":  8.3693,
     "delta_naive": 11.9238, "delta_asym_only": 0.3594,
     "delta_sym_full": 0.4021, "delta_asym_full": 0.1034,
     "asym_only_catastrophic_vs_naive": False,
     "arch_note": "Mode 1 (Layer-0 outlier); dense Llama-family"},
    {"label": "Llama-3.1-8B",   "baseline_ppl":  7.3840,
     "delta_naive": 0.4453, "delta_asym_only": 10.9219,
     "delta_sym_full": 0.2541, "delta_asym_full": 0.1059,
     "asym_only_catastrophic_vs_naive": True,
     "arch_note": "Mode 2 canonical; dense Llama-3 architecture"},
    {"label": "Pythia-6.9B",    "baseline_ppl": 10.7594,
     "delta_naive": 22.4631, "delta_asym_only": 0.6326,
     "delta_sym_full": 2.3781, "delta_asym_full": 0.0994,
     "asym_only_catastrophic_vs_naive": False,
     "arch_note": "Base model, high-noise Mode 2; older recipe"},
]

PHASE19 = []
for r in [mistral_res, qwen25_res, gptj_res, qwen35_res, gemma4_res]:
    PHASE19.append({
        "label":             r["label"],
        "baseline_ppl":      r["baseline_ppl"],
        "delta_naive":       r["delta_naive"],
        "delta_asym_only":   r["delta_asym_only"],
        "delta_sym_full":    r["delta_sym_full"],
        "delta_asym_full":   r["delta_asym_full"],
        "asym_only_catastrophic_vs_naive": r["asym_only_catastrophic_vs_naive"],
    })

all_rows = PHASE18 + PHASE19

print("\n" + "=" * 108)
print(f"Phase 18 + 19 ablation  ({datetime.now().isoformat(timespec='seconds')})")
print("=" * 108)
print(f"  {'Model':<22} {'base':>7}  "
      f"{'(A) naive':>11} {'(B) asym-only':>15} "
      f"{'(C) v0.2':>11} {'(D) v0.3':>11}  {'B cat?':>7}")
print(f"  {'-'*22} {'-'*7}  {'-'*11} {'-'*15} {'-'*11} {'-'*11}  {'-'*7}")
for r in all_rows:
    cat = "YES" if r["asym_only_catastrophic_vs_naive"] else "no"
    print(f"  {r['label']:<22} {r['baseline_ppl']:>7.3f}  "
          f"+{r['delta_naive']:>9.4f}  "
          f"+{r['delta_asym_only']:>13.4f}  "
          f"+{r['delta_sym_full']:>9.4f}  "
          f"+{r['delta_asym_full']:>9.4f}  {cat:>7}")

# B-catastrophe aggregate
b_cat_count = sum(1 for r in all_rows if r["asym_only_catastrophic_vs_naive"])
b_cat_models = [r["label"] for r in all_rows if r["asym_only_catastrophic_vs_naive"]]
print(f"\n  asym-only WORSE than naive on {b_cat_count}/{len(all_rows)} models: "
      f"{b_cat_models}")

# Production tier count per method
def prod(r, key):
    return (r[key] / r["baseline_ppl"]) < 0.05

methods = [("(A) naive", "delta_naive"),
           ("(B) asym-only", "delta_asym_only"),
           ("(C) v0.2 sym", "delta_sym_full"),
           ("(D) v0.3 asym", "delta_asym_full")]

print(f"\n  Production-tier count (<5% baseline) per method:")
for name, key in methods:
    count = sum(1 for r in all_rows if prod(r, key))
    print(f"    {name:<16}:  {count}/{len(all_rows)}")

# Save
out_path = "/content/phase19_ablation_generalization.json"
combined = {
    "date":         datetime.now().isoformat(timespec='seconds'),
    "phase18_prior": PHASE18,
    "phase19_runs": {
        "Mistral-7B-v0.1":           mistral_res,
        "Qwen2.5-14B":               qwen25_res,
        "GPT-J-6B":                  gptj_res,
        "Qwen3.5-9B":                qwen35_res,
        "gemma-4-E4B":               gemma4_res,
    },
    "eight_model_ablation": all_rows,
    "b_catastrophic_models": b_cat_models,
    "b_catastrophic_count":  b_cat_count,
}
with open(out_path, "w") as f:
    json.dump(combined, f, indent=2, default=str)
print(f"\nSaved -> {out_path}")

# Narrative
print("\n" + "=" * 108)
print("NARRATIVE CHECK")
print("=" * 108)
print(f"  8-model ablation coverage: 2021 (GPT-J) - 2026 (Qwen3.5, Gemma 4)")
print(f"  (A) naive INT3 failed catastrophically on: "
      f"{[r['label'] for r in all_rows if r['delta_naive'] / r['baseline_ppl'] > 0.5]}")
print(f"  (B) asym-only catastrophic WORSE than naive on: {b_cat_models}")
print(f"  (D) v0.3 is best on: "
      f"{sum(1 for r in all_rows if r['delta_asym_full'] <= min(r['delta_naive'], r['delta_asym_only'], r['delta_sym_full']))}/{len(all_rows)} models")
