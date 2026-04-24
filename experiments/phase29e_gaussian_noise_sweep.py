# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Phase 29e: Gaussian-noise sweep on Qwen3.6-27B to resolve
#            "noise structure vs noise magnitude" for the
#            hybrid-attention fragility
# ============================================================
# EXPLORATORY / sandbox. Not for the NSN paper.
#
# Phase 29d found that uniform rescaling (K, V -> K, V * 1.001,
# i.e. 0.1 % uniform scale change) produces bit-identical output
# to FP16 baseline on Qwen3.6-27B, while asym INT3 K+V breaks it
# (r4 0.16 -> 0.52). Two interpretations remain:
#
#   (S) Noise *structure* matters. Uniform rescaling is absorbed
#       by downstream renormalization; per-element non-uniform
#       noise (which is what quantization produces) is not. If
#       so, even a 0.1 %-magnitude per-element Gaussian will
#       break the model.
#
#   (M) Noise *magnitude* matters. Uniform 0.1 % is below the
#       model's tolerance; quantization at realistic bit widths
#       is above. The threshold lives somewhere between, and
#       Gaussian noise with small enough sigma will be absorbed.
#
# Phase 29e sweeps Gaussian noise at five magnitudes to locate
# the actual fragility threshold:
#
#   sigma_frac in {0.001, 0.003, 0.01, 0.03, 0.10}
#
# where sigma_frac is the ratio of per-token noise L2 norm to
# per-token input L2 norm. Concretely, noise is i.i.d. Gaussian
# per element with element-wise standard deviation
#
#   sigma = sigma_frac * ||x_token|| / sqrt(D)
#
# so that E[||noise_token||] = sigma_frac * ||x_token||.
#
# Reference conditions from Phase 29d included for direct
# comparison:
#
#   - FP16 baseline           (r4 should match 29d's 0.163)
#   - Uniform rescale 1.001   (r4 should match 29d's 0.163)
#   - asym INT3 K+V           (r4 should match 29d's 0.519)
#
# Applied to both K and V on the 16 transformer-attention
# layers (matches 29d's shape-guarded compress path).
#
# Runtime: 9 conditions x 500 tokens, roughly 20 min on A100.
# ============================================================


# =============================================================
# === CELL V1 === Setup
# =============================================================

import gc
import math
import torch

try:
    del mdl, tok
except NameError:
    pass
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

!pip install -q -U transformers accelerate hf_transfer datasets

# Same torch-fallback regime as Phase 29 / 29b / 29d: skip
# flash-linear-attention + causal-conv1d.

import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# ─────────────────────────────────────────────────────────────
# >>>>>  PASTE YOUR HF TOKEN  <<<<<
HF_TOKEN = "hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
# ─────────────────────────────────────────────────────────────
from huggingface_hub import login
login(token=HF_TOKEN)

import json
from datetime import datetime
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# =============================================================
# === CELL V2 === Quantizer + noise methods
# =============================================================

def qa_nsep_pchan_asym(x, bits):
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


def method_identity(x):
    return x.clone()


def method_uniform_rescale_0p1pct(x):
    return (x.float() * 1.001).to(x.dtype)


def make_gaussian_noise_method(sigma_frac, seed_salt=0):
    """Return a method that adds i.i.d. Gaussian noise per element.
    The noise stddev is chosen so that E[||noise_token||] equals
    sigma_frac * ||x_token|| (per-token-L2-norm calibration)."""
    def fn(x):
        dtype = x.dtype
        xf = x.float()
        # Per-token L2 norm, shape [..., T, 1]
        n = xf.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        D = xf.shape[-1]
        # Want E[||noise_token||] = sigma_frac * ||x_token||
        # noise ~ N(0, sigma_elem^2) per element, sum of D gives
        # E[||noise||^2] = D * sigma_elem^2, so
        # sigma_elem = sigma_frac * ||x|| / sqrt(D).
        sigma = sigma_frac * n / math.sqrt(D)
        noise = torch.randn_like(xf) * sigma
        return (xf + noise).to(dtype)
    fn._sigma_frac = sigma_frac
    return fn


# =============================================================
# === CELL V3 === KV cache plumbing (same shape guard as 29b/d)
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


def _layer_is_transformer_kv(past, li, expected_ctx_len):
    if not _layer_has_kv(past, li):
        return False
    ok, ov = get_kv(past, li)
    if ok is None or ov is None:
        return False
    if not (torch.is_tensor(ok) and torch.is_tensor(ov)):
        return False
    if ok.ndim != 4 or ov.ndim != 4:
        return False
    if ok.shape != ov.shape:
        return False
    if ok.shape[-2] != expected_ctx_len:
        return False
    return True


def compress_cache_transformer_only(past, method_fn, expected_ctx_len):
    total = n_cache_layers(past)
    compressed_ids = []
    if method_fn is None:
        return {"total_layers": total, "compressed_layers": 0,
                "skipped_layers": total}
    for li in range(total):
        if not _layer_is_transformer_kv(past, li, expected_ctx_len):
            continue
        ok, ov = get_kv(past, li)
        ok = ok.clone(); ov = ov.clone()
        nk = torch.zeros_like(ok); nv = torch.zeros_like(ov)
        for h in range(ok.shape[1]):
            nk[0, h] = method_fn(ok[0, h]).to(ok.dtype)
            nv[0, h] = method_fn(ov[0, h]).to(ov.dtype)
        set_kv(past, li, nk, nv)
        compressed_ids.append(li)
    return {
        "total_layers":      total,
        "compressed_layers": len(compressed_ids),
        "skipped_layers":    total - len(compressed_ids),
    }


# =============================================================
# === CELL V4 === Metrics
# =============================================================

def longest_repeated_substring(text, max_scan_len=200):
    n = len(text)
    best = 0
    for i in range(n):
        for j in range(i + 1, min(n, i + max_scan_len * 2)):
            k = 0
            while (j + k < n and text[i + k] == text[j + k]
                   and k < max_scan_len):
                k += 1
            if k > best:
                best = k
    return best

def ngram_repetition_ratio(tokens, n):
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return 1.0 - len(set(ngrams)) / len(ngrams)

def distinct_n(tokens, n):
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(ngrams)) / len(ngrams)


# =============================================================
# === CELL V5 === Generation
# =============================================================

@torch.inference_mode()
def generate_with_probe(prefix_ids, method_fn, mdl, n_new=500):
    out = mdl(prefix_ids, use_cache=True)
    past = out.past_key_values
    info = compress_cache_transformer_only(
        past, method_fn, expected_ctx_len=prefix_ids.shape[1]
    )
    next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated_ids = [next_id.item()]
    for _ in range(n_new - 1):
        out = mdl(next_id, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_ids.append(next_id.item())
    return generated_ids, info


# =============================================================
# === CELL V6 === Prefix (wiki_0 only)
# =============================================================

ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
wtext = "\n\n".join([x for x in ds["text"] if x.strip()])
print(f"WikiText-2 total chars: {len(wtext)}")

PREFIX_TEXT = wtext[0:25000]
PREFIX_NAME = "wikitext_offset_0"


# =============================================================
# === CELL V7 === Run the Gaussian noise sweep
# =============================================================

MODEL_ID    = "Qwen/Qwen3.6-27B"
MODEL_LABEL = "Qwen3.6-27B"

CTX_TARGET_TOKENS = 4096
N_NEW_TOKENS      = 500

SIGMA_FRACS = [0.001, 0.003, 0.01, 0.03, 0.10]

PROBES = (
    [
        ("1. FP16 baseline",                 None),
        ("2. Identity clone",                method_identity),
        ("3. Uniform rescale x*1.001",       method_uniform_rescale_0p1pct),
    ]
    + [
        (f"{4+i}. Gaussian noise sigma={s:<5g}",
         make_gaussian_noise_method(s))
        for i, s in enumerate(SIGMA_FRACS)
    ]
    + [
        ("9. asym INT3 K+V (ref)",
         lambda x: qa_nsep_pchan_asym(x, 3)),
    ]
)

print(f"\n{'=' * 80}")
print(f"Loading {MODEL_LABEL}  ({MODEL_ID})")
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

prefix_ids_full = tok.encode(
    PREFIX_TEXT, return_tensors="pt"
).to("cuda")
prefix_ids = prefix_ids_full[:, :CTX_TARGET_TOKENS]
actual_ctx = prefix_ids.shape[1]

print(f"\n{'=' * 80}")
print(f"Prefix: {PREFIX_NAME}  (ctx = {actual_ctx} tokens)")
print(f"{'=' * 80}")

per_probe = {}
fp16_r4 = None

# Fix RNG once so the five Gaussian probes get deterministic but
# distinct noise patterns (RNG state advances between probes).
torch.manual_seed(42)

for label, fn in PROBES:
    gen_ids, info = generate_with_probe(
        prefix_ids, fn, mdl, n_new=N_NEW_TOKENS
    )
    gen_text = tok.decode(gen_ids, skip_special_tokens=True)

    r3 = ngram_repetition_ratio(gen_ids, 3)
    r4 = ngram_repetition_ratio(gen_ids, 4)
    d2 = distinct_n(gen_ids, 2)
    lrs = longest_repeated_substring(gen_text)
    if fn is None:
        fp16_r4 = r4
    delta_r4 = (r4 - fp16_r4) if fp16_r4 is not None else 0.0

    per_probe[label] = {
        "sigma_frac":           getattr(fn, "_sigma_frac", None),
        "longest_repeat_chars": lrs,
        "repetition_3gram":     r3,
        "repetition_4gram":     r4,
        "delta_r4_vs_fp16":     delta_r4,
        "distinct_2":           d2,
        "tail_200_chars":       gen_text[-200:],
        "n_tokens":             len(gen_ids),
        "compression_info":     info,
    }

    print(f"  {label:<34s}  "
          f"r4={r4:.3f}  Δr4={delta_r4:+.3f}  "
          f"d2={d2:.2f}  lrs={lrs:>4d}ch")


# =============================================================
# === CELL V8 === Threshold readout
# =============================================================

print(f"\n\n{'=' * 80}")
print(f"Gaussian-noise fragility threshold on {MODEL_LABEL}")
print(f"{'=' * 80}")

# Pull the r4 sequence for the Gaussian probes
g_rows = []
for label, _ in PROBES:
    m = per_probe[label]
    if m["sigma_frac"] is not None:
        g_rows.append((m["sigma_frac"], m["repetition_4gram"]))
g_rows.sort()

print("\n  sigma_frac      r4      Δr4 vs FP16    verdict")
print(f"  {'FP16':<10s}  {fp16_r4:.3f}         --       reference")
print(f"  {'uniform 0.001':<10s}  "
      f"{per_probe['3. Uniform rescale x*1.001']['repetition_4gram']:.3f}     "
      f"{per_probe['3. Uniform rescale x*1.001']['delta_r4_vs_fp16']:+.3f}       "
      "(control: uniform is absorbed)")
for sig, r4 in g_rows:
    d = r4 - fp16_r4
    tag = "SAFE" if d < 0.15 else ("breaking" if d < 0.30 else "BROKEN")
    print(f"  {sig:<10g}  {r4:.3f}     {d:+.3f}       {tag}")

# Structural readout
def r4_of(label):
    return per_probe[label]["repetition_4gram"]

smallest_gauss_breaks = (
    r4_of(f"4. Gaussian noise sigma={SIGMA_FRACS[0]:<5g}") - fp16_r4 > 0.15
)

print("\n  Readout:")
if smallest_gauss_breaks:
    print(f"    ✅ STRUCTURE dominates: Gaussian sigma=0.001 (same"
          f" magnitude as the uniform 0.001 rescale, but PER-ELEMENT"
          f" rather than UNIFORM) already breaks the model, while"
          f" the uniform rescale at the same magnitude is absorbed."
          f" The fragility is about noise structure (per-element,"
          f" direction-rotating), not magnitude.")
else:
    # Find the threshold sigma
    threshold = None
    for sig, r4 in g_rows:
        if r4 - fp16_r4 > 0.15:
            threshold = sig
            break
    if threshold is not None:
        print(f"    ⚖️  MAGNITUDE matters: Gaussian below sigma="
              f"{threshold} is absorbed; at or above, the model breaks."
              f" The fragility has a magnitude threshold near"
              f" sigma={threshold}.")
    else:
        print(f"    🤔 Gaussian at all tested magnitudes is safe (up"
              f" to sigma={SIGMA_FRACS[-1]}), yet asym INT3 K+V"
              f" (r4={r4_of('9. asym INT3 K+V (ref)'):.3f}) still"
              f" breaks. Something about quantization-specific"
              f" structure (channel correlation? direction-only"
              f" rounding? per-channel asym min/max bias?) is the"
              f" culprit, and neither uniform rescale nor i.i.d."
              f" Gaussian reproduces it.")


# =============================================================
# === CELL V9 === Save
# =============================================================

output = {
    "date":           datetime.utcnow().isoformat(timespec="seconds"),
    "phase":          "29e",
    "purpose":        ("Gaussian-noise sweep on Qwen3.6-27B "
                       "transformer-attention layers to locate the "
                       "fragility threshold and resolve the "
                       "structure-vs-magnitude question left open "
                       "by Phase 29d."),
    "model":          MODEL_ID,
    "label":          MODEL_LABEL,
    "protocol": {
        "ctx_target_tokens": CTX_TARGET_TOKENS,
        "n_new_tokens":      N_NEW_TOKENS,
        "decode":            "greedy",
        "prefix":            PREFIX_NAME,
        "sigma_fracs":       SIGMA_FRACS,
        "probes":            [p[0] for p in PROBES],
        "rng_seed":          42,
    },
    "fp16_r4_baseline": fp16_r4,
    "per_probe":        per_probe,
}

out_path = "/content/phase29e_gaussian_noise_sweep.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\nSaved -> {out_path}")


# =============================================================
# === READING ===
# =============================================================
#
# What this run answers and what it does not:
#
#   - If sigma=0.001 Gaussian already breaks it while the 0.001
#     uniform rescale is bit-identical to FP16, STRUCTURE is the
#     knob. Downstream recipe implication: any per-element-ish
#     quantization on hybrid attention is scope-outside; only
#     uniform-only transforms are safe.
#
#   - If there is a clean sigma threshold (e.g., 0.003 safe, 0.01
#     broken), MAGNITUDE matters and there is a tolerance budget.
#     Downstream recipe implication: a sufficiently high-bit
#     quantizer may be usable on hybrid attention; the minimum
#     bit width can be computed from the threshold.
#
#   - If all Gaussian sigmas are safe while asym INT3 still breaks,
#     there is a quantization-SPECIFIC structure (channel
#     correlation of errors, asymmetric bias in per-channel
#     min/max, or the per-row vs per-channel step mismatch) that
#     pure i.i.d. Gaussian does not model. That outcome would
#     motivate a Phase 29f per-channel-vs-per-row noise-structure
#     probe.
# ============================================================
