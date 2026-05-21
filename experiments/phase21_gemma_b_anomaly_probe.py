# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Phase 21: Gemma 4-E4B B-anomaly follow-up (minimal probe)
# ============================================================
# Phase 20 ruled out FP16-eval artifact (H1) as the cause of
# gemma-4-E4B's negative ΔPPL under method (B) asymmetric
# per-channel INT3 without norm separation. The effect persists
# under native bf16 at -0.32 PPL, ~1.5% below baseline.
#
# Casual review recommended a two-pronged LIGHT probe (no deep
# investigation; just enough to either close the chapter or keep
# it alive as future-work material):
#
#   Axis 1 (reproducibility): re-measure B-delta with 3x more
#     windows and on a DIFFERENT WikiText-2 segment. If the
#     -0.3 effect vanishes, it was measurement-noise or segment-
#     specific and we close the book.
#
#   Axis 2 (K-only vs V-only): apply method (B) to only the Key
#     cache, only the Value cache, or both. If the effect is
#     K-only, it smells like attention-logit clipping /
#     entropy regularization. If V-only, value noise acting as
#     dropout. If both symmetrically, broad regularization.
#
# Scope constraint (explicit): this is a single-model 30-min
# probe, not a full ablation. We do NOT promote the negative-
# ΔPPL finding to a main-line claim regardless of outcome. The
# v0.3 recommended recipe (D) still produces normal small-
# positive ΔPPL (+0.93 / +1.16) on this model; that part is
# unchanged. We only want a clean footnote for the paper
# appendix.
#
# Methods (all at INT3):
#   baseline        - no quantization
#   B-KV            - asym per-channel no-nsep on BOTH K and V
#   B-K-only        - asym per-channel no-nsep on K, V untouched
#   B-V-only        - asym per-channel no-nsep on V, K untouched
#   D-KV            - nsep+pchan asym (v0.3) on both, as sanity
#
# Per method: 2 segments of WikiText-2 (chars [0:48k] and
# [48k:96k]) × 30 windows each, for ~6x the per-datapoint
# sample size of Phase 19/20.
#
# Runtime: ~30-45 min on Colab A100 (1 model × 5 methods ×
# 2 segments × 30 windows).
# ============================================================


# =============================================================
# === CELL V1 === Setup (fresh runtime)
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
# === CELL V2 === Quantizers
# =============================================================

def qa_asym_pchan_no_nsep(x, bits):
    """(B) asymmetric per-channel INT-bits, NO norm separation."""
    dtype = x.dtype
    xf = x.float()
    d_min = xf.amin(dim=0, keepdim=True)
    d_max = xf.amax(dim=0, keepdim=True)
    d_range = (d_max - d_min).clamp(min=1e-12)
    levels = 2**bits - 1
    step = d_range / levels
    dq = ((xf - d_min) / step).round().clamp(0, levels) * step + d_min
    return dq.to(dtype)


def qa_nsep_pchan_asym(x, bits):
    """(D) nsep + per-channel asym (nsn v0.3)."""
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
# === CELL V3 === K/V split cache compression
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


def compress_cache_split(past, fn_K, fn_V):
    """Apply fn_K to keys and fn_V to values independently. Pass
    None to leave that side untouched."""
    if fn_K is None and fn_V is None:
        return
    for li in range(n_cache_layers(past)):
        if not _layer_has_kv(past, li):
            continue
        ok, ov = get_kv(past, li)
        ok = ok.clone()
        ov = ov.clone()
        if fn_K is not None:
            nk = torch.zeros_like(ok)
            for h in range(ok.shape[1]):
                nk[0, h] = fn_K(ok[0, h]).to(ok.dtype)
            ok = nk
        if fn_V is not None:
            nv = torch.zeros_like(ov)
            for h in range(ov.shape[1]):
                nv[0, h] = fn_V(ov[0, h]).to(ov.dtype)
            ov = nv
        set_kv(past, li, ok, ov)


# =============================================================
# === CELL V4 === Load WikiText-2, two non-overlapping segments
# =============================================================

ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
wtext_full = "\n\n".join([x for x in ds["text"] if x.strip()])
print(f"WikiText-2 full test-split length: {len(wtext_full)} chars")

SEG1 = wtext_full[      0:  48000]   # same segment Phase 19/20 used
SEG2 = wtext_full[  48000:  96000]   # fresh, never-seen segment

print(f"Segment 1 length: {len(SEG1)} chars (matches Phase 19/20)")
print(f"Segment 2 length: {len(SEG2)} chars (reproducibility partner)")


def ppl_on_text(text, fn_K, fn_V, ctx_len=1024, stride=512, max_windows=30):
    ids = tok.encode(text, return_tensors="pt").to("cuda")
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
            compress_cache_split(past, fn_K, fn_V)
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
    # name,       fn_K,                                       fn_V
    ("baseline",  None,                                       None),
    ("B-KV",      lambda x: qa_asym_pchan_no_nsep(x, 3),       lambda x: qa_asym_pchan_no_nsep(x, 3)),
    ("B-K-only",  lambda x: qa_asym_pchan_no_nsep(x, 3),       None),
    ("B-V-only",  None,                                        lambda x: qa_asym_pchan_no_nsep(x, 3)),
    ("D-KV",      lambda x: qa_nsep_pchan_asym(x, 3),          lambda x: qa_nsep_pchan_asym(x, 3)),
]


# =============================================================
# === CELL V5 === Load Gemma 4-E4B (bf16, matches Phase 20)
# =============================================================

MODEL_ID = "google/gemma-4-E4B"
LABEL    = "gemma-4-E4B"

print(f"\n{'='*80}")
print(f"Loading {LABEL} ({MODEL_ID}) in bf16 ...")
print(f"{'='*80}")
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
mdl = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    use_safetensors=True,
    trust_remote_code=True,
)
mdl.eval()
print(f"VRAM used: {torch.cuda.memory_allocated()/1e9:.1f} GB")
print(f"model param dtype (sample): {next(iter(mdl.parameters())).dtype}")


# =============================================================
# === CELL V6 === Run on both segments at 30 windows each
# =============================================================

results = {}
for seg_name, seg_text in [("SEG1", SEG1), ("SEG2", SEG2)]:
    print(f"\n{'-'*80}")
    print(f"Running {seg_name} (len={len(seg_text)} chars, max_windows=30)")
    print(f"{'-'*80}")
    seg_out = {}
    baseline_ppl = None
    for name, fn_K, fn_V in METHODS:
        print(f"  Running {name} ...", flush=True)
        ppl, nt = ppl_on_text(seg_text, fn_K, fn_V, max_windows=30)
        if baseline_ppl is None:
            baseline_ppl = ppl
        delta = ppl - baseline_ppl
        seg_out[name] = {"ppl": ppl, "delta": delta, "tokens": nt}
        dstr = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
        print(f"    {name:<10}  PPL={ppl:>9.4f}   ΔPPL={dstr:>9}   tokens={nt}")
    results[seg_name] = {
        "baseline_ppl": baseline_ppl,
        "methods":      seg_out,
    }


# =============================================================
# === CELL V7 === Diagnosis
# =============================================================

print("\n" + "=" * 96)
print(f"Phase 21 diagnosis  ({datetime.now().isoformat(timespec='seconds')})")
print("=" * 96)

for seg_name in ["SEG1", "SEG2"]:
    r = results[seg_name]
    print(f"\n  {seg_name}  (baseline PPL = {r['baseline_ppl']:.4f})")
    for name in ["B-KV", "B-K-only", "B-V-only", "D-KV"]:
        d = r["methods"][name]["delta"]
        dstr = f"+{d:.4f}" if d >= 0 else f"{d:.4f}"
        pct = d / r["baseline_ppl"] * 100
        print(f"    {name:<10}  Δ={dstr:>10}  ({pct:+.2f}% of baseline)")

# Reproducibility verdict
b_kv_seg1 = results["SEG1"]["methods"]["B-KV"]["delta"]
b_kv_seg2 = results["SEG2"]["methods"]["B-KV"]["delta"]
phase19_b = -0.4393  # fp16 value from Phase 19

print(f"\n  B-KV reproducibility:")
print(f"    Phase 19 (SEG1, fp16, 10 windows):      {phase19_b:+.4f}")
print(f"    Phase 20 (SEG1, bf16, 10 windows):      -0.3200")
print(f"    Phase 21 (SEG1, bf16, 30 windows):      {b_kv_seg1:+.4f}")
print(f"    Phase 21 (SEG2, bf16, 30 windows):      {b_kv_seg2:+.4f}")

if b_kv_seg1 < -0.1 and b_kv_seg2 < -0.1:
    reproducibility = "CONFIRMED (both segments negative, effect ~0.3)"
elif b_kv_seg1 < -0.1 or b_kv_seg2 < -0.1:
    reproducibility = "PARTIAL (one segment shows effect, one does not)"
else:
    reproducibility = "NOT REPRODUCED (segments show near-zero effect)"
print(f"    -> {reproducibility}")

# K/V split interpretation (on SEG1, matching Phase 19/20 text)
print(f"\n  K/V split on SEG1 (what drives the effect?):")
b_k = results["SEG1"]["methods"]["B-K-only"]["delta"]
b_v = results["SEG1"]["methods"]["B-V-only"]["delta"]
b_full = results["SEG1"]["methods"]["B-KV"]["delta"]
print(f"    B-K-only    Δ = {b_k:+.4f}   (quantizing K only)")
print(f"    B-V-only    Δ = {b_v:+.4f}   (quantizing V only)")
print(f"    B-KV        Δ = {b_full:+.4f}   (both)")
if b_k < -0.1 and b_v > 0:
    split_verdict = "K-driven: attention-logit clipping / entropy regularization"
elif b_v < -0.1 and b_k > 0:
    split_verdict = "V-driven: value noise regularization"
elif b_k < -0.1 and b_v < -0.1:
    split_verdict = "symmetric: both K and V contribute (broad regularization)"
elif abs(b_k) < 0.05 and abs(b_v) < 0.05:
    split_verdict = "neither side alone reproduces; combined effect is constructive"
else:
    split_verdict = "mixed / inconclusive"
print(f"    -> {split_verdict}")

# Save
out_path = "/content/phase21_gemma_b_anomaly_probe.json"
combined = {
    "date":                  datetime.now().isoformat(timespec='seconds'),
    "model":                 MODEL_ID,
    "dtype":                 "bf16",
    "ctx_len":               1024,
    "stride":                512,
    "windows_per_method":    30,
    "segments": {
        "SEG1":  {"chars_range": [0,     48000], "text_start": SEG1[:80]},
        "SEG2":  {"chars_range": [48000, 96000], "text_start": SEG2[:80]},
    },
    "results":               results,
    "reproducibility":       reproducibility,
    "k_v_split_verdict":     split_verdict,
    "phase19_prior_delta_B": phase19_b,
    "phase20_prior_delta_B_bf16": -0.3200,
}
with open(out_path, "w") as f:
    json.dump(combined, f, indent=2, default=str)
print(f"\nSaved -> {out_path}")

print("\n" + "=" * 96)
print("DECISION RULE")
print("=" * 96)
print("""
  If reproducibility = NOT REPRODUCED:
    -> Close the chapter. Phase 19's -0.44 was measurement noise /
       segment-specific. Document in JSON and move on.

  If reproducibility = PARTIAL:
    -> Still a footnote, not a main claim. Interesting but fragile.

  If reproducibility = CONFIRMED and K/V split is informative:
    -> Write up as Appendix / lab note: "accidental clipping-type
       regularization on Gemma 4-E4B under asym-only ablation; does
       not affect v0.3 recommendation; mechanism open for future
       work." Keep the recommended method (D) as-is.
""")
