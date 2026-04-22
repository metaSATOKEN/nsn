# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Phase 28: Mode 1 detection on small Qwen2.5 models
# ============================================================
# The NSN paper §6.1 currently characterizes the Qwen2-7B "INT4
# valley of death" (+1382 naive-INT4 dPPL vs +12 at INT3 and +5
# at INT5) as an n = 1 observation in our 27-model broader
# survey. External signal from the TurboQuant (ICLR 2026) effort
# suggests that SMALL Qwen2.5 checkpoints may be the same
# failure mode at a different scale:
#
#   * Qwen2.5-1.5B has a K / V mean-norm ratio of 182x (the
#     highest reported across their tested models).
#   * Qwen-0.5B is explicitly flagged as a case where even
#     TurboQuant "may be insufficient", plausibly implying a
#     K / V ratio > 1000x.
#   * In both, "5-20% of K channels (especially Layer 0) have
#     10-100x larger RMS than median" -- the exact structural
#     signature §6.1 identifies for Mode 1 on Qwen2-7B.
#
# This phase runs the classic Phase 1-4 bit-width sweep
# (naive per-row symmetric absmax INT quantization, no nsep /
# pchan / asym) on Qwen2.5-0.5B and Qwen2.5-1.5B to see whether
# the non-monotonic INT4 ΔPPL spike reproduces at these scales.
# If either model shows the valley, the NSN paper's "n = 1"
# caveat can be relaxed to "n = 2 or 3 across Qwen family".
#
# Protocol (identical to Phases 1-4):
#   Task:     WikiText-2 sliding-window PPL, ctx = 1024,
#             stride = 512, 10 windows, first 48000 chars.
#   Models:   Qwen/Qwen2.5-0.5B, Qwen/Qwen2.5-1.5B.
#   Methods:  FP16 baseline + naive INT bit widths
#             {2, 3, 4, 5, 6, 8}.
#   Quantizer: per-row symmetric absmax. NOT the NSN recipe;
#              we want to see the NAIVE failure pattern here,
#              matching how §6.1 defines "Mode 1".
#
# Expected outcome: if Qwen2.5-0.5B and/or -1.5B show a non-
# monotonic spike at INT4 (much worse than INT3 and INT5),
# they replicate Qwen2-7B's Mode 1 signature and we can cite
# them in a paper v0.21 update to §6.1.
#
# Runtime: ~8-12 min on Colab A100 (both models fit in <4 GB
# VRAM; 6 bit widths x 2 models x PPL = 12 PPL runs + 2
# baselines).
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
# === CELL V2 === Naive per-row symmetric absmax INT quantizer
# (Paper B variant (A); Phase 1-4 uses the same form.)
# =============================================================

def qa_naive_perrow_sym(x, bits):
    """Per-row symmetric absmax INT quantization.

    This is the unmodified industry baseline: one absmax scale
    per row, symmetric [-qm, qm] grid, no nsep / pchan / asym.
    Paper B calls this "variant (A)" and §6.1 is defined w.r.t.
    its failure mode (+1382 dPPL on Qwen2-7B at INT4).
    """
    dtype = x.dtype
    xf = x.float()
    qm = 2 ** (bits - 1) - 1  # INT4 -> 7, INT3 -> 3, INT8 -> 127
    s = xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / qm
    q = (xf / s).round().clamp(-qm, qm) * s
    return q.to(dtype)


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
# === CELL V4 === WikiText-2 sliding-window PPL
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
# === CELL V5 === Run sweep
# =============================================================

MODELS = [
    ("Qwen/Qwen2.5-0.5B", "Qwen2.5-0.5B"),
    ("Qwen/Qwen2.5-1.5B", "Qwen2.5-1.5B"),
]

BIT_WIDTHS = [2, 3, 4, 5, 6, 8]

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

    # Baseline
    base_ppl, _ = ppl_with(None, mdl, tok)
    print(f"  FP16 baseline PPL: {base_ppl:.4f}")

    per_bit = {"FP16": {"bits": 16, "ppl": base_ppl,
                        "delta_ppl": 0.0, "pct_of_base": 0.0}}

    for b in BIT_WIDTHS:
        ppl, _ = ppl_with(lambda x: qa_naive_perrow_sym(x, b), mdl, tok)
        dppl = ppl - base_ppl
        pct = 100 * dppl / base_ppl
        per_bit[f"INT{b}"] = {
            "bits": b, "ppl": ppl, "delta_ppl": dppl, "pct_of_base": pct,
        }
        print(f"  INT{b} (naive)  : ppl = {ppl:>10.4f}   "
              f"ΔPPL = {dppl:+9.4f}   ({pct:+7.2f}% of base)")

    # Mode 1 detection: is INT4 non-monotonic? i.e. INT4 dPPL much
    # larger than both INT3 and INT5?
    int3_d = per_bit["INT3"]["delta_ppl"]
    int4_d = per_bit["INT4"]["delta_ppl"]
    int5_d = per_bit["INT5"]["delta_ppl"]
    mode_1 = (int4_d > 5 * max(int3_d, int5_d, 0.1))  # 5x heuristic
    print(f"\n  Mode 1 check (INT4 >> INT3 and INT4 >> INT5?): "
          f"INT3={int3_d:+.3f}  INT4={int4_d:+.3f}  INT5={int5_d:+.3f}  "
          f"-> {'Mode 1 DETECTED' if mode_1 else 'monotone / Mode 2'}")

    results[label] = {
        "model":        model_id,
        "baseline_ppl": base_ppl,
        "per_bit":      per_bit,
        "mode_1_detected": mode_1,
    }


# =============================================================
# === CELL V6 === Summary + JSON save
# =============================================================

print("\n" + "=" * 96)
print(f"Phase 28 Mode 1 detection summary  "
      f"({datetime.now().isoformat(timespec='seconds')})")
print("=" * 96)

print(f"  {'Model':<14}  {'base':>7}  "
      + "  ".join(f"{'INT' + str(b):>7}" for b in BIT_WIDTHS)
      + "  Mode 1?")
print(f"  {'-' * 14}  {'-' * 7}  "
      + "  ".join("-" * 7 for _ in BIT_WIDTHS)
      + "  -------")

for label in [m[1] for m in MODELS]:
    r = results[label]
    base = r["baseline_ppl"]
    row = f"  {label:<14}  {base:>7.2f}  "
    row += "  ".join(
        f"{r['per_bit'][f'INT{b}']['pct_of_base']:>+6.1f}%"
        for b in BIT_WIDTHS
    )
    row += f"  {'YES' if r['mode_1_detected'] else 'no'}"
    print(row)


out_path = "/content/phase28_mode1_detection_small_qwen.json"
combined = {
    "date":     datetime.now().isoformat(timespec='seconds'),
    "protocol": {
        "task":         "WikiText-2 sliding-window perplexity",
        "ctx_len":      1024,
        "stride":       512,
        "max_windows":  10,
        "text_chars":   48000,
        "quantizer":    ("qa_naive_perrow_sym: per-row symmetric absmax "
                         "INT quantization. Paper B variant (A); the "
                         "industry baseline against which Mode 1 is "
                         "defined in §6.1."),
        "bit_widths":   BIT_WIDTHS,
        "models":       [m[1] for m in MODELS],
        "mode_1_rule":  ("Non-monotonic INT4 spike: INT4 dPPL > 5x max("
                         "INT3 dPPL, INT5 dPPL, 0.1). Heuristic; paper "
                         "§6.1 defines Mode 1 qualitatively as 'naive "
                         "INT4 dramatically worse than its bit-width "
                         "neighbors on both sides'."),
    },
    "external_motivation": {
        "turboquant_2026_signal": (
            "Qwen2.5-1.5B has K/V mean-norm ratio = 182x (highest in "
            "the TurboQuant tested set). 5-20%% of K channels (especially "
            "Layer 0) have 10-100x larger RMS than median. Qwen-0.5B "
            "is flagged as a case where TurboQuant alone may be "
            "insufficient, plausibly implying even more extreme K/V "
            "structure."
        ),
        "source": "https://github.com/ggml-org/llama.cpp/discussions/20969",
    },
    "results": results,
    "status":  ("Detection run. If either Qwen2.5-0.5B or Qwen2.5-1.5B "
                "shows the Qwen2-7B-style INT4 valley (+1382 dPPL at "
                "INT4 with +12 at INT3 and +5 at INT5), the 'n = 1' "
                "caveat in NSN paper §6.1 can be relaxed to 'observed "
                "on 2-3 Qwen-family checkpoints'. If both are monotone, "
                "paper §6.1 stays as n = 1."),
}
with open(out_path, "w") as f:
    json.dump(combined, f, indent=2, default=str)
print(f"\nSaved -> {out_path}")


print("\n" + "=" * 96)
print("READING")
print("=" * 96)
print("""
  What to look for:

  1. Non-monotonic INT4 spike. The Qwen2-7B signature is INT3 ~ +12 dPPL,
     INT4 = +1382 dPPL, INT5 ~ +5 dPPL. If Qwen2.5-0.5B or Qwen2.5-1.5B
     shows anything similar (INT4 dPPL at least 5x larger than both
     INT3 and INT5 dPPL), they count as Mode 1 replications.

  2. Absolute magnitudes. Small models have lower baseline PPL
     (often < 10 for these two), so percent-of-baseline matters more
     than absolute dPPL. A +100 dPPL spike on a base of ~10 means the
     model is basically broken at that bit width.

  3. If the smaller models show the pattern but larger Qwen2.5
     checkpoints (e.g., Qwen2.5-14B in the main benchmark) do not,
     that is itself an interesting story: Mode 1 may be scale- and
     training-recipe-dependent within the Qwen family.

  Expected (from the external TurboQuant signal):
    - Qwen2.5-0.5B: very likely Mode 1 (extreme K/V imbalance).
    - Qwen2.5-1.5B: plausibly Mode 1 (182x K/V ratio).
    - Monotone on either would be itself informative.
""")
