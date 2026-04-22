# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Phase 28c: Mid-scale Qwen coverage (3B, 7B)
# ============================================================
# Phase 28 / 28b found:
#
#   Qwen2.5-0.5B : Mode 2 (monotone), partial rescue (+10.75 %).
#   Qwen2.5-1.5B : Mode 1 INT3 valley, fully rescued to +2.48 %.
#
# Combined with the paper's existing Qwen coverage:
#
#   Qwen2-7B      : Mode 1 INT4 valley, rescued to +1.24 %.
#   Qwen2.5-14B   : Mode 2, rescued to +4.64 %.
#
# This leaves an awkward gap around 3B-7B in the Qwen2.5 series.
# This phase fills it in: Qwen2.5-3B and Qwen2.5-7B, each under
# the Phase-28 naive bit-width sweep (to detect Mode 1 behavior)
# AND the Phase-28b v0.2 / v0.3 recipe comparison at INT3.
#
# Four possible outcomes per model:
#   * Mode 1 valley at some bit width, rescued by v0.3    -> extend
#     the paper's Mode 1 family claim to n=3 or n=4.
#   * Mode 2 monotone, rescued by v0.3                    -> paper
#     claim stands; this model joins the "usable" set.
#   * Mode 2 monotone, NOT rescued below 5 % threshold    -> scope
#     caveat on this scale.
#   * Unexpected pattern (e.g., different bit-width valley, or
#     recipe blind spot)                                  -> new
#     observation for the paper.
#
# Runtime: ~20-25 min on Colab A100
#   - Qwen2.5-3B  : ~ 6 GB VRAM, 9 PPL runs x ~1 min + load
#   - Qwen2.5-7B  : ~15 GB VRAM, 9 PPL runs x ~1.5 min + load
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
# === CELL V2 === Three quantizers
# =============================================================

def qa_naive_perrow_sym(x, bits):
    """Paper B variant (A)."""
    dtype = x.dtype
    xf = x.float()
    qm = 2 ** (bits - 1) - 1
    s = xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / qm
    q = (xf / s).round().clamp(-qm, qm) * s
    return q.to(dtype)


def qa_nsep_pchan_sym(x, bits):
    """Paper B variant (C), v0.2."""
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
    """Paper B variant (D), v0.3 (the shipped recipe)."""
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
# === CELL V5 === Run sweep + recipe
# =============================================================

MODELS = [
    ("Qwen/Qwen2.5-3B", "Qwen2.5-3B"),
    ("Qwen/Qwen2.5-7B", "Qwen2.5-7B"),
]

NAIVE_BITS = [2, 3, 4, 5, 6, 8]

METHODS = (
    [("FP16 baseline", None)]
    + [(f"(A) naive INT{b}", (lambda b: lambda x: qa_naive_perrow_sym(x, b))(b))
       for b in NAIVE_BITS]
    + [
        ("(C) sym INT3 (v0.2)",   lambda x: qa_nsep_pchan_sym(x, 3)),
        ("(D) asym INT3 (v0.3)",  lambda x: qa_nsep_pchan_asym(x, 3)),
    ]
)

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

    # Mode 1 check on the naive sweep: non-monotone spike —
    # INT_b is massively worse than the next-higher bit AND
    # also worse than the next-lower bit (a local peak).
    # Uses list-index neighbors so NAIVE_BITS can skip values.
    int_dppls = {b: per_method[f"(A) naive INT{b}"]["delta_ppl"]
                 for b in NAIVE_BITS}
    sorted_bits = sorted(int_dppls.keys())
    spike_bit = None
    for i in range(1, len(sorted_bits) - 1):  # interior positions
        b = sorted_bits[i]
        lo = int_dppls[sorted_bits[i - 1]]
        hi = int_dppls[sorted_bits[i + 1]]
        # right-side drop by ≥5× (the "valley exit") AND local peak
        if int_dppls[b] > 5 * max(hi, 0.1) and int_dppls[b] > lo:
            spike_bit = b
            break

    recipe_pct = per_method["(D) asym INT3 (v0.3)"]["pct_of_base"]
    recipe_rescues = (recipe_pct < 5.0)

    print(f"\n  Mode check: spike_bit = {spike_bit if spike_bit else 'none (monotone)'}")
    print(f"  Recipe rescue (D < 5 % of baseline): "
          f"{recipe_pct:+.2f}% -> {'YES' if recipe_rescues else 'NO'}")

    results[label] = {
        "model":           model_id,
        "baseline_ppl":    base_ppl,
        "per_method":      per_method,
        "naive_dppl_by_bit": int_dppls,
        "mode_1_spike_bit":  spike_bit,
        "recipe_rescues":    recipe_rescues,
    }


# =============================================================
# === CELL V6 === Summary + JSON save
# =============================================================

print("\n" + "=" * 96)
print(f"Phase 28c mid-Qwen-scale summary  "
      f"({datetime.now().isoformat(timespec='seconds')})")
print("=" * 96)

header = f"  {'Model':<14}  {'base':>7}  "
header += "  ".join(f"{'INT' + str(b):>7}" for b in NAIVE_BITS)
header += f"  {'sym(C)':>8}  {'asym(D)':>8}  {'spike':>6}  {'rescue?':>7}"
print(header)

for label in [m[1] for m in MODELS]:
    r = results[label]
    base = r["baseline_ppl"]
    row = f"  {label:<14}  {base:>7.2f}  "
    row += "  ".join(
        f"{r['per_method'][f'(A) naive INT{b}']['pct_of_base']:>+6.1f}%"
        for b in NAIVE_BITS
    )
    c_pct = r["per_method"]["(C) sym INT3 (v0.2)"]["pct_of_base"]
    d_pct = r["per_method"]["(D) asym INT3 (v0.3)"]["pct_of_base"]
    spike = f"INT{r['mode_1_spike_bit']}" if r['mode_1_spike_bit'] else "—"
    rescued = "YES" if r["recipe_rescues"] else "NO"
    row += f"  {c_pct:>+7.1f}%  {d_pct:>+7.1f}%  {spike:>6}  {rescued:>7}"
    print(row)


out_path = "/content/phase28c_mid_qwen_scale.json"
combined = {
    "date":     datetime.now().isoformat(timespec='seconds'),
    "protocol": {
        "task":        "WikiText-2 sliding-window PPL",
        "ctx_len":     1024,
        "stride":      512,
        "max_windows": 10,
        "text_chars":  48000,
        "naive_bits":  NAIVE_BITS,
        "extra_methods": [
            "(C) sym INT3 (v0.2) nsep+pchan",
            "(D) asym INT3 (v0.3) nsep+pchan, shipped recipe",
        ],
        "models":        [m[1] for m in MODELS],
        "mode_1_rule":   ("naive INT-b dPPL > 5x max of INT-(b-1) dPPL and "
                          "INT-(b+1) dPPL, for interior bits b in {3, 4, 5, 6}. "
                          "Heuristic; matches the 'non-monotone single-bit "
                          "valley' definition used in Phase 28 and paper §6.1."),
        "rescue_rule":   "recipe_rescues = (asym INT3 pct_of_base < 5 %).",
    },
    "context": {
        "phase_28_prior_finding":  (
            "Qwen2.5-1.5B has a non-monotone INT3 valley (+5961 dPPL at "
            "INT3 vs +10 at INT4 and +15 at INT5 under naive per-row). "
            "Qwen2.5-0.5B is monotone Mode 2."
        ),
        "phase_28b_prior_finding": (
            "v0.3 recipe rescues Qwen2.5-1.5B's INT3 valley from "
            "+59675 % to +2.48 % of baseline (24026x improvement "
            "ratio). Qwen2.5-0.5B is partially rescued to +10.75 % "
            "but does not reach the 5 % threshold."
        ),
        "paper_existing_qwen_coverage": (
            "Qwen2-7B (INT4 valley, rescued to +1.24 %) and Qwen2.5-14B "
            "(Mode 2, rescued to +4.64 %) are already in the main 8-model "
            "benchmark."
        ),
    },
    "results": results,
    "status":  ("Phase 28c closes the Qwen2.5 scale coverage at 3B and 7B. "
                "Combined with Phase 28 / 28b and the paper's existing "
                "benchmark, the Qwen family is now characterized at "
                "0.5B / 1.5B / 3B / 7B (Qwen2-7B) / 7B (Qwen2.5) / 14B "
                "-- six Qwen checkpoints across roughly 30x parameter range."),
}
with open(out_path, "w") as f:
    json.dump(combined, f, indent=2, default=str)
print(f"\nSaved -> {out_path}")


print("\n" + "=" * 96)
print("READING")
print("=" * 96)
print("""
  What to look for:

  1. spike_bit column. If EITHER model shows a non-monotone spike at
     some bit width, the Mode 1 family grows from n=2 (Qwen2-7B INT4,
     Qwen2.5-1.5B INT3) to n=3 or n=4. Whichever bit width the spike
     lands at is informative: the paper hypothesis is that this is a
     bit-grid-alignment phenomenon dependent on the specific K-cache
     outlier distribution, so different models may spike at different
     bit widths.

  2. recipe_rescues column. v0.3 (D asym INT3) < 5 % of baseline PPL
     = the paper's "usable" threshold. If both rescued, Qwen2.5 scale
     coverage under v0.3 is 4 of 4 for {1.5B, 3B, 7B, 14B} (0.5B
     excluded as out-of-paper-scope). Strong story.

  3. Compare Qwen2.5-7B to Qwen2-7B at the same scale. If the same-
     scale, same-family but different-training-recipe models show
     different modes (one valley, one not), that quantifies how much
     training-pipeline variation matters within a fixed architecture.
""")
