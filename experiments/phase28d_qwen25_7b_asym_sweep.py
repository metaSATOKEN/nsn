# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Phase 28d: Does the v0.3 recipe rescue ALL bit widths on the
#            Qwen2.5-7B multi-peak case?
# ============================================================
# Exploratory / not for paper. Follow-up to Phase 28c.
#
# Phase 28c found that Qwen2.5-7B under *naive per-row symmetric*
# quantization is not a clean Mode-1 valley: it has three peaks at
# INT2 / INT4 / INT6 (all > +3000 % of baseline) and relatively
# mild failures at INT3 / INT5 (+176 % / +67 %). The v0.3 recipe
# (nsep + pchan + asym) at INT3 rescues it to +3.47 %.
#
# The question this phase answers:
#
#   Is v0.3 bit-width-universal on this broken model, or is there
#   something bit-grid-structural (a parity effect) that survives
#   the asymmetric window?
#
# Two hypotheses on the table:
#
#   (H1) v0.3 is bit-width universal. If nsep + pchan + asym kills
#        the outlier pathology independently of the grid step, then
#        at every bit width the asym result should be small (a
#        few % of baseline). The even-bit naive catastrophe is
#        purely a naive-per-row-sym artifact.
#
#   (H2) Bit parity survives asym. If the multi-peak pattern is
#        structural to the quantization grid (e.g., how the zero
#        level aligns with outlier positions), then even under
#        asym the even bits will remain broken. In that case the
#        asym-vs-naive rescue factor will be order-unity (not
#        order-1000) at the even bits.
#
# Protocol (Qwen2.5-7B only):
#   - Re-measure naive sym at { INT2, INT3, INT4, INT5, INT6, INT8 }
#     (matches Phase 28c; serves as a sanity check vs the saved
#     numbers too.)
#   - Measure v0.3 asym (nsep + pchan + asym) at the same bit
#     widths: { INT2, INT3, INT4, INT5, INT6, INT8 }.
#   - Report paired rescue factors (naive ΔPPL / asym ΔPPL) per bit.
#
# Interpretation:
#   - All 6 asym cells below ~10 % of baseline -> H1 supported,
#     the bit-parity pattern is a naive-quant artifact and the v0.3
#     recipe is a general outlier-suppressor, not INT3-specific.
#   - Even bits (INT2/4/6) asym remain catastrophic -> H2 supported,
#     there is a grid-parity phenomenon here beyond just outliers.
#   - Mixed pattern (e.g., INT2 broken, INT4/6 rescued, or vice
#     versa) -> more interesting structure, write it up as an
#     open observation.
#
# Runtime: ~10-15 min on Colab A100
#   - Qwen2.5-7B : ~15 GB VRAM, 13 PPL runs × ~1.5 min + load
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
# === CELL V2 === Two quantizers at arbitrary bit width
# =============================================================

def qa_naive_perrow_sym(x, bits):
    """Paper B variant (A): naive per-row symmetric absmax."""
    dtype = x.dtype
    xf = x.float()
    qm = 2 ** (bits - 1) - 1
    s = xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / qm
    q = (xf / s).round().clamp(-qm, qm) * s
    return q.to(dtype)


def qa_nsep_pchan_asym(x, bits):
    """Paper B variant (D), v0.3: norm-separated per-channel asym.
    Usually shipped at bits=3; here we probe all widths."""
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
# === CELL V4 === WikiText-2 PPL runner (same as Phase 28c)
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
# === CELL V5 === Run the 13-condition sweep on Qwen2.5-7B
# =============================================================

MODEL_ID = "Qwen/Qwen2.5-7B"
LABEL    = "Qwen2.5-7B"
BITS     = [2, 3, 4, 5, 6, 8]

METHODS = (
    [("FP16 baseline", None)]
    + [(f"(A) naive INT{b}", (lambda b: lambda x: qa_naive_perrow_sym(x, b))(b))
       for b in BITS]
    + [(f"(D) asym INT{b}",  (lambda b: lambda x: qa_nsep_pchan_asym(x, b))(b))
       for b in BITS]
)

try:
    del mdl, tok
except NameError:
    pass
gc.collect(); torch.cuda.empty_cache()

print(f"\n{'=' * 80}")
print(f"Loading {LABEL}  ({MODEL_ID})")
print(f"{'=' * 80}")
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
mdl = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    use_safetensors=True,
    trust_remote_code=True,
)
mdl.eval()
print(f"  VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")


per_method = {}
base_ppl = None

for method_name, fn in METHODS:
    ppl, n_win = ppl_with(fn, mdl, tok)
    if method_name == "FP16 baseline":
        base_ppl = ppl
        dppl = 0.0
        pct = 0.0
    else:
        dppl = ppl - base_ppl
        pct  = 100 * dppl / base_ppl
    per_method[method_name] = {
        "ppl":         ppl,
        "delta_ppl":   dppl,
        "pct_of_base": pct,
    }
    print(f"  {method_name:<22s}  ppl = {ppl:>10.4f}   "
          f"ΔPPL = {dppl:+10.4f}   ({pct:+8.2f}% of base)")


# =============================================================
# === CELL V6 === Per-bit rescue-factor analysis
# =============================================================

print(f"\n{'=' * 80}")
print(f"Per-bit rescue analysis on {LABEL}")
print(f"{'=' * 80}")
print(f"  Baseline FP16 PPL: {base_ppl:.4f}")
print()
print(f"  {'bit':>5s}  {'naive %':>10s}  {'asym %':>10s}  "
      f"{'rescue×':>10s}  {'verdict':<20s}")
print(f"  {'-' * 5}  {'-' * 10}  {'-' * 10}  {'-' * 10}  {'-' * 20}")

per_bit_rescue = {}
for b in BITS:
    naive_pct = per_method[f"(A) naive INT{b}"]["pct_of_base"]
    asym_pct  = per_method[f"(D) asym INT{b}"]["pct_of_base"]
    # Avoid div-by-zero; below 0.1 % treat as "already clean"
    if max(naive_pct, 0.0) < 0.5:
        rescue = float("nan")
        verdict = "(both clean)"
    else:
        denom = max(asym_pct, 1e-3)
        rescue = max(naive_pct, 0.0) / max(denom, 1e-3)
        if asym_pct < 5.0:
            verdict = "asym passes 5 %"
        elif asym_pct < 10.0:
            verdict = "asym marginal"
        else:
            verdict = "asym BROKEN too"
    print(f"  INT{b:<2d}  {naive_pct:>9.2f}%  {asym_pct:>9.2f}%  "
          f"{rescue:>9.2f}×  {verdict:<20s}")
    per_bit_rescue[b] = {
        "naive_pct":  naive_pct,
        "asym_pct":   asym_pct,
        "rescue_x":   rescue,
        "verdict":    verdict,
    }


# =============================================================
# === CELL V7 === H1 / H2 verdict
# =============================================================

even_bits_asym = [per_bit_rescue[b]["asym_pct"] for b in [2, 4, 6]]
odd_bits_asym  = [per_bit_rescue[b]["asym_pct"] for b in [3, 5]]

all_rescued = all(v < 5.0 for v in [per_bit_rescue[b]["asym_pct"] for b in BITS])
even_broken = any(v > 10.0 for v in even_bits_asym)

print(f"\n{'=' * 80}")
print(f"Hypothesis check")
print(f"{'=' * 80}")
print(f"  Even bits (INT2/4/6) asym %:  {even_bits_asym}")
print(f"  Odd  bits (INT3/5)   asym %:  {odd_bits_asym}")
print(f"  All 6 asym bits < 5 %:        {all_rescued}")
print(f"  Some even-bit asym > 10 %:    {even_broken}")

if all_rescued:
    verdict = "H1 supported -- v0.3 is bit-width universal on Qwen2.5-7B. Even-bit naive catastrophe is a per-row-sym artifact, not a grid-parity property."
elif even_broken and max(odd_bits_asym) < 10.0:
    verdict = "H2 supported -- bit parity survives asym. Even-bit failures are structural to the quantization grid, not purely outlier-driven."
else:
    verdict = "Mixed pattern -- neither H1 nor H2 cleanly. Worth writing up as an open observation."

print(f"\n  Verdict: {verdict}")


# =============================================================
# === CELL V8 === Save
# =============================================================

output = {
    "date": datetime.utcnow().isoformat(timespec="seconds"),
    "phase": "28d",
    "purpose": "Test whether v0.3 (nsep+pchan+asym) rescues Qwen2.5-7B at ALL bit widths or only at INT3, to disambiguate bit-parity-survives-asym (H2) from naive-artifact (H1).",
    "model":  MODEL_ID,
    "label":  LABEL,
    "protocol": {
        "task":     "WikiText-2 sliding-window PPL",
        "ctx_len":  1024,
        "stride":   512,
        "max_windows": 10,
        "bits_probed": BITS,
    },
    "baseline_ppl":    base_ppl,
    "per_method":      per_method,
    "per_bit_rescue":  per_bit_rescue,
    "summary": {
        "even_bits_asym_pct": even_bits_asym,
        "odd_bits_asym_pct":  odd_bits_asym,
        "all_rescued_under_5pct": all_rescued,
        "some_even_bit_asym_broken": even_broken,
        "verdict": verdict,
    },
}

out_path = "/content/phase28d_qwen25_7b_asym_sweep.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\nSaved -> {out_path}")


# =============================================================
# === READING ===
# =============================================================
#
# Rows to eyeball on the V6 table:
#
#   INT8: both naive and asym should be ~0 % (sanity check; if
#         either is > 1 % something is wrong with the runner).
#
#   INT3: asym should be ~3 % (matches Phase 28c's +3.47 % to
#         a fraction of a percent; confirms reproducibility).
#
#   INT2, INT4, INT6: naive is +3414 / +3663 / +3498 % in
#         Phase 28c. The question is whether asym drops these
#         to a single-digit % of baseline (H1) or leaves them
#         in double-or-triple-digit territory (H2).
#
#   INT5: naive is +66.6 % in Phase 28c; asym INT5 should land
#         somewhere between INT3's +3 % (if H1) and the even-bit
#         result (if H2 is bit-parity-specific rather than all-
#         low-bit-specific).
#
# Paper-body implications if we end up writing it up anyway:
#
#   * H1 -> §6.2 "INT3-tier recipe" claim is tighter than
#     necessary on broken models. The INT4 asym observation we
#     documented on Llama-3.1-8B (§6.3.5) is Llama-specific;
#     on outlier-broken models asym may win at every bit.
#
#   * H2 -> new mechanism section: bit-parity grid alignment is
#     a second failure mode distinct from the Layer-0 outlier
#     story in the main paper.
#
# ============================================================
