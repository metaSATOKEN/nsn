# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Phase 27: Decode-step latency overhead of INT3 cache compression
# ============================================================
# The NSN paper §6.2 currently states that the Python-only
# Int3AsymNsepPchanCache adds roughly "+21 % decode latency on
# 7B-class models", and that a fused CUDA kernel would remove
# most of that overhead. This "+21 %" figure is a rough estimate
# carried over from the src/modeq/pack.py docstring; it has not
# been directly measured on the full decode pipeline. A human
# reviewer reasonably flagged this as a weak spot -- without a
# measured number, the 21 % is just a hand-wave.
#
# This phase replaces that hand-wave with an actual measurement.
# We prefill a realistic context, then time per-token decode
# under three regimes on two canonical models (Qwen2-7B,
# Llama-3.1-8B):
#
#   (A) FP16 baseline. Standard HF Transformers decode loop, no
#       cache compression. This is the reference latency.
#
#   (B) Roundtrip overhead. Before each decode step we call
#       compress_cache(past, real_cache_quant) which runs
#       from_fp16() + to_fp16() on every layer's every K/V head
#       -- both pack and unpack. This is the WORST case; real
#       deployment would not re-pack on every step.
#
#   (C) Unpack-only overhead. We pack the prefill cache ONCE
#       via from_fp16() (not timed), then each decode step
#       decompresses via to_fp16() into an FP16 cache that the
#       model reads. This is the REAL deployment cost.
#
# The two overheads (B) and (C) bracket the expected operational
# regime: (B) is the naive-implementation worst case and (C) is
# the kernel-free best case.
#
# Notes on methodology:
#
#   - Wall-clock timing via time.perf_counter() plus
#     torch.cuda.synchronize() to flush pending CUDA work.
#   - 5-step warmup before each measurement; 50 steps measured.
#   - Repeat 3 times; report mean / median of the three runs.
#   - KV cache is populated by a 1024-token prefill; decode then
#     appends one token at a time. The compressed prefix portion
#     remains the target of (C)'s to_fp16() call on every step,
#     while new tokens accumulate in FP16 on top -- matching how
#     a real packed-cache deployment would work.
#   - We do NOT claim model-family generality: 2 models is
#     indicative, not comprehensive. A fuller sweep would repeat
#     on the 8-model benchmark set.
#
# The Int3AsymNsepPchanCache and pack / unpack primitives are
# INLINED below (verbatim copy of src/modeq/cache.py and
# src/modeq/pack.py at commit f089e7a) so this script runs
# standalone on Colab without needing the modeq repository to
# be pip-installable (the repo is private at the time of this
# run).
#
# Runtime: ~15-20 min on Colab A100 (2 models x 3 regimes x
# 3 repeats x 50 steps + model load).
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
import time
import statistics
from datetime import datetime
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# =============================================================
# === CELL V2 === Inlined pack / unpack (verbatim from modeq)
# =============================================================

def pack_int3_unsigned(x_int8, strict=True):
    """Pack unsigned INT3 values (range [0, 7]) into compact uint8 bytes."""
    *prefix, N = x_int8.shape
    assert N % 8 == 0, f"Last dim must be divisible by 8, got {N}"
    if strict:
        xmin = int(x_int8.min().item())
        xmax = int(x_int8.max().item())
        if xmin < 0 or xmax > 7:
            raise ValueError(f"got [{xmin}, {xmax}]; need [0, 7]")
    u = x_int8.to(torch.int64).clamp(0, 7)
    u = u.view(*prefix, N // 8, 8)
    shifts = torch.arange(8, device=u.device, dtype=torch.int64) * 3
    packed64 = (u << shifts).sum(dim=-1)
    b0 = (packed64 & 0xFF).to(torch.uint8)
    b1 = ((packed64 >> 8) & 0xFF).to(torch.uint8)
    b2 = ((packed64 >> 16) & 0xFF).to(torch.uint8)
    return torch.stack([b0, b1, b2], dim=-1).view(*prefix, 3 * N // 8)


def unpack_int3_unsigned(bytes_in, N):
    """Unpack compact bytes back to unsigned INT3 values in [0, 7]."""
    *prefix, P = bytes_in.shape
    assert P == 3 * N // 8
    grouped = bytes_in.view(*prefix, N // 8, 3)
    b0, b1, b2 = grouped[..., 0], grouped[..., 1], grouped[..., 2]
    packed64 = (
        b0.to(torch.int64)
        | (b1.to(torch.int64) << 8)
        | (b2.to(torch.int64) << 16)
    )
    shifts = torch.arange(8, device=packed64.device, dtype=torch.int64) * 3
    mask = 7
    u = (packed64.unsqueeze(-1) >> shifts) & mask
    return u.view(*prefix, N).to(torch.int8)


# =============================================================
# === CELL V3 === Int3AsymNsepPchanCache (inlined)
# =============================================================

class Int3AsymNsepPchanCache:
    """Real-packed INT3 asymmetric nsep+pchan cache."""

    def __init__(self):
        self.packed_d = None
        self.norm = None
        self.d_min = None
        self.d_step = None

    @classmethod
    def from_fp16(cls, x, bits=3):
        assert bits == 3
        assert x.shape[-1] % 8 == 0
        obj = cls()
        xf = x.float()
        n = xf.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        d = xf / n
        levels = 2 ** bits - 1
        d_min = d.amin(dim=-2, keepdim=True)
        d_max = d.amax(dim=-2, keepdim=True)
        d_range = (d_max - d_min).clamp(min=1e-12)
        d_step = d_range / levels
        d_idx = ((d - d_min) / d_step).round().clamp(0, levels).to(torch.int8)
        obj.packed_d = pack_int3_unsigned(d_idx)
        obj.norm = n.half()
        obj.d_min = d_min.half()
        obj.d_step = d_step.half()
        return obj

    def to_fp16(self):
        *prefix, P = self.packed_d.shape
        D = P * 8 // 3
        d_idx = unpack_int3_unsigned(self.packed_d, N=D)
        d_q = d_idx.float() * self.d_step.float() + self.d_min.float()
        d_renorm = d_q / d_q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return (self.norm.float() * d_renorm).half()


def real_cache_quant(x, bits=3):
    """from_fp16 + to_fp16 round-trip; used in regime (B)."""
    cached = Int3AsymNsepPchanCache.from_fp16(x, bits=bits)
    return cached.to_fp16().to(x.dtype)


# =============================================================
# === CELL V3b === torch.compile version of to_fp16 (Regime E)
# =============================================================
# Functional rewrite of Int3AsymNsepPchanCache.to_fp16() + inlined
# unpack_int3_unsigned, so the whole pipeline (bit-unpack +
# reconstruct + renormalize + norm-mul + fp16 cast) is visible to
# torch.compile as one contiguous graph. torch.compile should fuse
# the 10+ small CUDA kernels into 1-2 compiled kernels, closing
# most of the remaining per-layer launch overhead in regime (D).

def _to_fp16_functional(packed_d, norm, d_min, d_step):
    """Pure-functional equivalent of Int3AsymNsepPchanCache.to_fp16()
    suitable for torch.compile. Accepts the four tensors that live
    inside the cache object and returns the reconstructed fp16."""
    *prefix, P = packed_d.shape
    N = P * 8 // 3

    # Inlined unpack_int3_unsigned
    grouped = packed_d.view(*prefix, N // 8, 3)
    b0 = grouped[..., 0]
    b1 = grouped[..., 1]
    b2 = grouped[..., 2]
    packed64 = (
        b0.to(torch.int64)
        | (b1.to(torch.int64) << 8)
        | (b2.to(torch.int64) << 16)
    )
    shifts = torch.arange(8, device=packed_d.device, dtype=torch.int64) * 3
    u = (packed64.unsqueeze(-1) >> shifts) & 7
    d_idx = u.view(*prefix, N).to(torch.int8)

    # Reconstruct
    d_q = d_idx.float() * d_step.float() + d_min.float()
    d_renorm = d_q / d_q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return (norm.float() * d_renorm).half()


# Compile once at module load. dynamic=False specializes on the
# observed [H, T, D] shape; all 32 layers of Llama-3.1-8B share
# the same shape so no recompiles occur after the first layer.
_to_fp16_compiled = torch.compile(_to_fp16_functional, dynamic=False)


# =============================================================
# === CELL V4 === KV cache plumbing
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


def compress_cache_roundtrip(past):
    """Regime (B): re-pack-then-unpack every K/V head in place."""
    for li in range(n_cache_layers(past)):
        if not _layer_has_kv(past, li):
            continue
        ok, ov = get_kv(past, li)
        ok = ok.clone(); ov = ov.clone()
        for h in range(ok.shape[1]):
            ok[0, h] = real_cache_quant(ok[0, h]).to(ok.dtype)
            ov[0, h] = real_cache_quant(ov[0, h]).to(ov.dtype)
        set_kv(past, li, ok, ov)


def pack_prefill_cache(past):
    """Regime (C) setup: pack every K/V head ONCE. Returns the
    structure of Int3AsymNsepPchanCache objects, indexed by
    [layer_idx][k_or_v][head_idx]."""
    packed = []
    for li in range(n_cache_layers(past)):
        if not _layer_has_kv(past, li):
            packed.append(None)
            continue
        k, v = get_kv(past, li)
        k_caches = [Int3AsymNsepPchanCache.from_fp16(k[0, h])
                    for h in range(k.shape[1])]
        v_caches = [Int3AsymNsepPchanCache.from_fp16(v[0, h])
                    for h in range(v.shape[1])]
        packed.append((k_caches, v_caches))
    return packed


def unpack_into_past(past, packed, prefix_len):
    """Regime (C): decompress packed prefix cache back into the
    first ``prefix_len`` positions of the live past tensors.
    New tokens beyond prefix_len are left untouched (they
    accumulated in FP16 during decode)."""
    for li in range(n_cache_layers(past)):
        if not _layer_has_kv(past, li):
            continue
        entry = packed[li]
        if entry is None:
            continue
        k_caches, v_caches = entry
        k, v = get_kv(past, li)
        for h in range(k.shape[1]):
            k[0, h, :prefix_len] = k_caches[h].to_fp16().to(k.dtype)
            v[0, h, :prefix_len] = v_caches[h].to_fp16().to(v.dtype)


def pack_prefill_cache_batched(past):
    """Regime (D) setup: pack each layer's K and V as a SINGLE
    batched tensor over all heads, leveraging the fact that
    Int3AsymNsepPchanCache.from_fp16() naturally supports leading
    dims (it operates on [..., T, D]). Replaces the per-head
    Python loop of regime (C) with a per-layer call -- an H-fold
    reduction in CUDA kernel launches per decode step."""
    packed = []
    for li in range(n_cache_layers(past)):
        if not _layer_has_kv(past, li):
            packed.append(None)
            continue
        k, v = get_kv(past, li)                            # [1, H, T, D]
        k_cache = Int3AsymNsepPchanCache.from_fp16(k[0])   # pack [H, T, D] in one call
        v_cache = Int3AsymNsepPchanCache.from_fp16(v[0])
        packed.append((k_cache, v_cache))
    return packed


def unpack_into_past_batched(past, packed, prefix_len):
    """Regime (D): decompress all H heads of each layer with ONE
    to_fp16() call per K or V tensor, instead of a per-head loop.
    On Llama-3.1-8B (H=8) this is an 8x reduction in the number
    of CUDA kernel launches for the unpack path."""
    for li in range(n_cache_layers(past)):
        if not _layer_has_kv(past, li):
            continue
        entry = packed[li]
        if entry is None:
            continue
        k_cache, v_cache = entry
        k, v = get_kv(past, li)                            # [1, H, T, D]
        k[0, :, :prefix_len] = k_cache.to_fp16().to(k.dtype)
        v[0, :, :prefix_len] = v_cache.to_fp16().to(v.dtype)


def unpack_into_past_compiled(past, packed, prefix_len):
    """Regime (E): like (D), but the to_fp16() equivalent is a
    torch.compile()'d functional form. torch.compile should fuse
    the 10+ small CUDA kernels inside to_fp16 into 1-2 kernels,
    reducing launch overhead beyond what (D) achieves."""
    for li in range(n_cache_layers(past)):
        if not _layer_has_kv(past, li):
            continue
        entry = packed[li]
        if entry is None:
            continue
        k_cache, v_cache = entry
        k, v = get_kv(past, li)
        k_recon = _to_fp16_compiled(
            k_cache.packed_d, k_cache.norm, k_cache.d_min, k_cache.d_step
        )
        v_recon = _to_fp16_compiled(
            v_cache.packed_d, v_cache.norm, v_cache.d_min, v_cache.d_step
        )
        k[0, :, :prefix_len] = k_recon.to(k.dtype)
        v[0, :, :prefix_len] = v_recon.to(v.dtype)


# =============================================================
# === CELL V5 === Timing utilities
# =============================================================

N_STEPS = 50
N_WARMUP = 5
N_REPEAT = 3
PREFIX_LEN = 1024


def _sync(): torch.cuda.synchronize()


ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
wtext = "\n\n".join([x for x in ds["text"] if x.strip()])[:48000]


def make_prefix_ids(tok):
    ids = tok.encode(wtext, return_tensors="pt").to("cuda")
    return ids[:, :PREFIX_LEN]


def prefill(mdl, prefix_ids):
    """Run the prefill pass and return the populated cache + the
    next-token for decode seed."""
    with torch.inference_mode():
        out = mdl(prefix_ids, use_cache=True)
    _sync()
    next_tok = out.logits[0, -1, :].argmax().view(1, 1)
    return out.past_key_values, next_tok


def time_decode_baseline(mdl, prefix_ids):
    """Regime (A): plain HF decode loop, no compression."""
    past, next_tok = prefill(mdl, prefix_ids)
    # Warmup
    with torch.inference_mode():
        for _ in range(N_WARMUP):
            out = mdl(next_tok, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_tok = out.logits[0, -1, :].argmax().view(1, 1)
    _sync()
    # Timed
    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(N_STEPS):
            out = mdl(next_tok, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_tok = out.logits[0, -1, :].argmax().view(1, 1)
    _sync()
    t1 = time.perf_counter()
    return (t1 - t0) / N_STEPS * 1000.0  # ms / step


def time_decode_roundtrip(mdl, prefix_ids):
    """Regime (B): compress_cache(roundtrip) before each decode step."""
    past, next_tok = prefill(mdl, prefix_ids)
    # Warmup
    with torch.inference_mode():
        for _ in range(N_WARMUP):
            compress_cache_roundtrip(past)
            out = mdl(next_tok, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_tok = out.logits[0, -1, :].argmax().view(1, 1)
    _sync()
    # Timed
    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(N_STEPS):
            compress_cache_roundtrip(past)
            out = mdl(next_tok, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_tok = out.logits[0, -1, :].argmax().view(1, 1)
    _sync()
    t1 = time.perf_counter()
    return (t1 - t0) / N_STEPS * 1000.0


def time_decode_unpack_only(mdl, prefix_ids):
    """Regime (C): pack prefill ONCE, then unpack into past before each step."""
    past, next_tok = prefill(mdl, prefix_ids)
    # One-time pack of the prefill cache
    packed = pack_prefill_cache(past)
    # Warmup
    with torch.inference_mode():
        for _ in range(N_WARMUP):
            unpack_into_past(past, packed, PREFIX_LEN)
            out = mdl(next_tok, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_tok = out.logits[0, -1, :].argmax().view(1, 1)
    _sync()
    # Timed
    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(N_STEPS):
            unpack_into_past(past, packed, PREFIX_LEN)
            out = mdl(next_tok, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_tok = out.logits[0, -1, :].argmax().view(1, 1)
    _sync()
    t1 = time.perf_counter()
    return (t1 - t0) / N_STEPS * 1000.0


def time_decode_unpack_batched(mdl, prefix_ids):
    """Regime (D): like (C), but pack / unpack per LAYER (not per head).
    Uses Int3AsymNsepPchanCache's native batched-leading-dim support
    to process all H heads of each layer in one call."""
    past, next_tok = prefill(mdl, prefix_ids)
    packed = pack_prefill_cache_batched(past)
    # Warmup
    with torch.inference_mode():
        for _ in range(N_WARMUP):
            unpack_into_past_batched(past, packed, PREFIX_LEN)
            out = mdl(next_tok, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_tok = out.logits[0, -1, :].argmax().view(1, 1)
    _sync()
    # Timed
    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(N_STEPS):
            unpack_into_past_batched(past, packed, PREFIX_LEN)
            out = mdl(next_tok, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_tok = out.logits[0, -1, :].argmax().view(1, 1)
    _sync()
    t1 = time.perf_counter()
    return (t1 - t0) / N_STEPS * 1000.0


def time_decode_unpack_compiled(mdl, prefix_ids):
    """Regime (E): like (D), but uses torch.compile()'d to_fp16().
    Extra warmup steps allow the torch.compile JIT to finish
    before timing begins (first call can take several seconds)."""
    past, next_tok = prefill(mdl, prefix_ids)
    packed = pack_prefill_cache_batched(past)
    # Extended warmup for torch.compile (first call triggers compilation).
    with torch.inference_mode():
        for _ in range(max(N_WARMUP, 10)):
            unpack_into_past_compiled(past, packed, PREFIX_LEN)
            out = mdl(next_tok, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_tok = out.logits[0, -1, :].argmax().view(1, 1)
    _sync()
    # Timed
    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(N_STEPS):
            unpack_into_past_compiled(past, packed, PREFIX_LEN)
            out = mdl(next_tok, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_tok = out.logits[0, -1, :].argmax().view(1, 1)
    _sync()
    t1 = time.perf_counter()
    return (t1 - t0) / N_STEPS * 1000.0


# =============================================================
# === CELL V6 === Model loop
# =============================================================

MODELS = [
    ("Qwen/Qwen2-7B",            "Qwen2-7B"),
    ("meta-llama/Llama-3.1-8B",  "Llama-3.1-8B"),
]

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

    prefix_ids = make_prefix_ids(tok)

    # Five regimes, each repeated N_REPEAT times
    runs = {"baseline": [], "roundtrip": [], "unpack_only": [],
            "unpack_batched": [], "unpack_compiled": []}

    for rep in range(N_REPEAT):
        print(f"  repeat {rep + 1}/{N_REPEAT}:")
        b = time_decode_baseline(mdl, prefix_ids)
        runs["baseline"].append(b)
        print(f"    baseline              : {b:7.2f} ms/step")

        r = time_decode_roundtrip(mdl, prefix_ids)
        runs["roundtrip"].append(r)
        print(f"    roundtrip (B)         : {r:7.2f} ms/step   (+{(r/b - 1)*100:6.1f} %)")

        u = time_decode_unpack_only(mdl, prefix_ids)
        runs["unpack_only"].append(u)
        print(f"    unpack-only (C)       : {u:7.2f} ms/step   (+{(u/b - 1)*100:6.1f} %)")

        d = time_decode_unpack_batched(mdl, prefix_ids)
        runs["unpack_batched"].append(d)
        print(f"    unpack-batched (D)    : {d:7.2f} ms/step   (+{(d/b - 1)*100:6.1f} %)")

        e = time_decode_unpack_compiled(mdl, prefix_ids)
        runs["unpack_compiled"].append(e)
        print(f"    unpack-compiled (E)   : {e:7.2f} ms/step   (+{(e/b - 1)*100:6.1f} %)")

    # Aggregate: median across repeats
    med = {k: statistics.median(v) for k, v in runs.items()}
    overhead_B = (med["roundtrip"] / med["baseline"] - 1) * 100
    overhead_C = (med["unpack_only"] / med["baseline"] - 1) * 100
    overhead_D = (med["unpack_batched"] / med["baseline"] - 1) * 100
    overhead_E = (med["unpack_compiled"] / med["baseline"] - 1) * 100

    print(f"\n  MEDIAN (n={N_REPEAT}):")
    print(f"    baseline             : {med['baseline']:7.2f} ms/step")
    print(f"    roundtrip (B)        : {med['roundtrip']:7.2f} ms/step   (+{overhead_B:6.1f} %)")
    print(f"    unpack-only (C)      : {med['unpack_only']:7.2f} ms/step   (+{overhead_C:6.1f} %)")
    print(f"    unpack-batched (D)   : {med['unpack_batched']:7.2f} ms/step   (+{overhead_D:6.1f} %)")
    print(f"    unpack-compiled (E)  : {med['unpack_compiled']:7.2f} ms/step   (+{overhead_E:6.1f} %)")

    results[label] = {
        "model":     model_id,
        "prefix_len": PREFIX_LEN,
        "n_steps":   N_STEPS,
        "n_warmup":  N_WARMUP,
        "n_repeat":  N_REPEAT,
        "raw_runs":  runs,
        "median_ms_per_step": med,
        "overhead_pct": {
            "roundtrip (B, worst-case)":          overhead_B,
            "unpack_only (C, per-head loop)":     overhead_C,
            "unpack_batched (D, per-layer)":      overhead_D,
            "unpack_compiled (E, torch.compile)": overhead_E,
        },
    }


# =============================================================
# === CELL V7 === Summary + JSON save
# =============================================================

print("\n" + "=" * 96)
print(f"Phase 27 decode-latency summary  "
      f"({datetime.now().isoformat(timespec='seconds')})")
print("=" * 96)

print(f"  {'Model':<16}  {'base(ms)':>9}  "
      f"{'(B)':>8}  {'(C)':>8}  {'(D)':>8}  {'(E)':>8}")
print(f"  {'-'*16}  {'-'*9}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")

for label in [m[1] for m in MODELS]:
    r = results[label]
    base = r["median_ms_per_step"]["baseline"]
    ob = r["overhead_pct"]["roundtrip (B, worst-case)"]
    oc = r["overhead_pct"]["unpack_only (C, per-head loop)"]
    od = r["overhead_pct"]["unpack_batched (D, per-layer)"]
    oe = r["overhead_pct"]["unpack_compiled (E, torch.compile)"]
    print(f"  {label:<16}  {base:>7.2f}   "
          f"{ob:>+6.1f}%  {oc:>+6.1f}%  {od:>+6.1f}%  {oe:>+6.1f}%")

out_path = "/content/phase27_decode_latency.json"
combined = {
    "date":     datetime.now().isoformat(timespec='seconds'),
    "protocol": {
        "task":         "Per-token decode latency with INT3 cache compression",
        "prefix_len":   PREFIX_LEN,
        "n_steps":      N_STEPS,
        "n_warmup":     N_WARMUP,
        "n_repeat":     N_REPEAT,
        "regimes": {
            "A_baseline":     "Standard HF decode loop, FP16 cache, no compression",
            "B_roundtrip":    ("from_fp16 + to_fp16 per K/V head per layer on "
                               "EVERY decode step. Worst case; re-packs redundantly."),
            "C_unpack_only":  ("Pack the prefill cache ONCE (not timed); decompress "
                               "via to_fp16 on every decode step. PER-HEAD loop -- "
                               "H calls per K tensor per step. Dominated by CUDA "
                               "kernel-launch overhead on small [T, D] tensors."),
            "D_unpack_batched": ("Same as C, but per-LAYER: one to_fp16() call "
                                 "processes all H heads of a layer at once by "
                                 "leveraging Int3AsymNsepPchanCache's native "
                                 "leading-dim support. H-fold reduction in kernel "
                                 "launches vs C."),
            "E_unpack_compiled": ("Same as D, but to_fp16() is a torch.compile()'d "
                                  "functional form (dynamic=False). torch.compile "
                                  "should fuse the 10+ small CUDA kernels inside "
                                  "to_fp16 into 1-2 compiled kernels. Closest "
                                  "no-hand-written-CUDA real-deployment cost."),
        },
        "models":       [m[1] for m in MODELS],
        "measurement":  ("wall-clock via time.perf_counter() + "
                         "torch.cuda.synchronize(), median of N_REPEAT repeats"),
    },
    "results": results,
    "status":  ("Verification run. Replaces the estimated '+21 % decode "
                "latency' figure in the NSN paper §6.2 with five measured "
                "numbers. Regime (E) is the lowest-overhead Python-only "
                "path: per-layer batching + torch.compile()-fused to_fp16. "
                "Regime (D) is the same batching without torch.compile. "
                "Regime (C) is the per-head loop used in Phases 22-26 "
                "compress_cache. Regime (B) is the naive worst case. "
                "The spread (B) -> (C) -> (D) -> (E) quantifies how much "
                "of the observed overhead is CUDA-kernel-launch latency "
                "(removable by batching + fusion) vs. genuine unpack work "
                "(not removable without a hand-written CUDA kernel)."),
}
with open(out_path, "w") as f:
    json.dump(combined, f, indent=2, default=str)
print(f"\nSaved -> {out_path}")


print("\n" + "=" * 96)
print("READING")
print("=" * 96)
print("""
  What to look for:

  1. unpack-only (C) overhead is the paper-relevant number. If it
     is substantially less than the paper's existing "+21 %" claim,
     §6.2 should be updated downward (and the kernel-free case
     becomes more attractive). If it matches, the estimate was
     correct. If it is substantially more, the paper's throughput
     claim was optimistic and needs caveating.

  2. roundtrip (B) is included only to cap the worst case: a user
     who naively wraps their decode step with compress_cache(past,
     real_cache_quant) would pay this overhead. It doubles as a
     sanity check on the measurement apparatus.

  3. baseline absolute value sanity check: Qwen2-7B and
     Llama-3.1-8B on A100 typically run ~15-30 ms / step at
     1024-token prefill without custom kernels. If our baseline
     is outside this range, model load / warmup is misconfigured.
""")
