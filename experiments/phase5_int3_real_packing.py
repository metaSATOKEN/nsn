# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Phase 5: INT3 Real Packing PoC
# ============================================================
# Goal: prove that INT3 nsep+pchan can deliver true 5.33x
# memory compression with production quality on Mode 2 models.
#
# Thesis claims to validate:
#   1. LOSSLESS PACKING: simulated INT3 vs packed INT3 produce
#      identical PPL (within floating-point noise).
#   2. REAL COMPRESSION: packed KV cache uses exactly 3/16 of
#      FP16 bytes, i.e., 5.33x compression, measured in real
#      tensor storage.
#   3. PRODUCTION QUALITY: Mode 2 models stay within 5% of
#      baseline PPL, even with real packing.
#
# Target models (confirmed Mode 2, instruction-tuned):
#   - Qwen2.5-14B (mid-size production model)
#   - Llama-3.1-8B-Instruct (ubiquitous baseline)
#
# Implementation: pack 8 INT3 values (24 bits) into 3 uint8
# bytes. Bit layout:
#   byte 0: [v0 bits 0-2][v1 bits 0-2][v2 bits 0-1]
#   byte 1: [v2 bit 2][v3 bits 0-2][v4 bits 0-2][v5 bit 0]
#   byte 2: [v5 bits 1-2][v6 bits 0-2][v7 bits 0-2]
#
# Total: 24 bits = 3 bytes for 8 values = 3.0 bits/value.
# (vs FP16's 16 bits/value, i.e., 5.33x compression)
# ============================================================


# =============================================================
# === CELL R1 === Setup + INT3 pack/unpack primitives
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


# ─── INT3 packing primitives ─────────────────────────────────
# Strategy: shift [-3, 3] → [0, 6] as 3-bit unsigned, then pack
# 8 consecutive 3-bit values into 3 bytes (24 bits) via bitwise.
# Implementation uses int64 intermediate arithmetic for clarity;
# CUDA kernels could do this with uint8 bitops directly.

def pack_int3_signed(x_int8):
    """
    Pack signed INT3 values (range [-3, 3]) into packed uint8 bytes.
    Input  : int8 tensor with last dim N, where N is divisible by 8.
    Output : uint8 tensor with last dim 3*N/8.
    """
    *prefix, N = x_int8.shape
    assert N % 8 == 0, f"Last dim must be divisible by 8, got {N}"

    # Shift [-3, 3] to [0, 6] (unsigned 3-bit, one value -- 7 -- unused)
    u = (x_int8.to(torch.int64) + 3).clamp(0, 7)

    # Reshape to groups of 8
    u = u.view(*prefix, N // 8, 8)

    # Pack 8 values into a single int64 (24 bits used)
    shifts = torch.arange(8, device=u.device, dtype=torch.int64) * 3
    packed64 = (u << shifts).sum(dim=-1)  # [..., N/8]

    # Split into 3 bytes (LSB first)
    b0 = (packed64 & 0xFF).to(torch.uint8)
    b1 = ((packed64 >> 8) & 0xFF).to(torch.uint8)
    b2 = ((packed64 >> 16) & 0xFF).to(torch.uint8)

    # Stack to [..., N/8, 3] then reshape to [..., 3*N/8]
    bytes_out = torch.stack([b0, b1, b2], dim=-1).view(*prefix, 3 * N // 8)
    return bytes_out


def unpack_int3_signed(bytes_in, N):
    """
    Reverse of pack_int3_signed. Input shape: [..., 3*N/8] uint8.
    Output shape: [..., N] int8 with values in [-3, 3].
    """
    *prefix, P = bytes_in.shape
    assert P == 3 * N // 8, f"Expected {3*N//8} bytes for N={N}, got {P}"

    grouped = bytes_in.view(*prefix, N // 8, 3)
    b0, b1, b2 = grouped[..., 0], grouped[..., 1], grouped[..., 2]
    packed64 = (b0.to(torch.int64) |
                (b1.to(torch.int64) << 8) |
                (b2.to(torch.int64) << 16))

    shifts = torch.arange(8, device=packed64.device, dtype=torch.int64) * 3
    mask = 7  # 3-bit mask
    u = (packed64.unsqueeze(-1) >> shifts) & mask  # [..., N/8, 8]

    # Shift [0, 6] back to [-3, 3]
    s = u - 3
    return s.view(*prefix, N).to(torch.int8)


# ─── Correctness check ───────────────────────────────────────
def roundtrip_test():
    torch.manual_seed(0)
    x = torch.randint(-3, 4, (2, 3, 64), dtype=torch.int8, device="cuda")
    packed = pack_int3_signed(x)
    unpacked = unpack_int3_signed(packed, N=64)
    assert torch.equal(x, unpacked), "Roundtrip FAILED"
    print(f"  Roundtrip test: PASSED ({x.numel()} values)")
    print(f"  Input  shape: {tuple(x.shape)}  dtype: {x.dtype}")
    print(f"  Packed shape: {tuple(packed.shape)}  dtype: {packed.dtype}")
    print(f"  Compression: {x.numel() * 2 / packed.numel():.2f}x (vs FP16)")
    print(f"              = {x.numel() / packed.numel() * 8 / 3:.2f}x theoretical")
    return True

print("INT3 packing primitives loaded.")
roundtrip_test()


# =============================================================
# === CELL R2 === Real INT3 nsep+pchan class + PPL eval
# =============================================================

class Int3NsepPchanCache:
    """
    Real-packed INT3 nsep+pchan cache.
    Stores: per-token norm (FP16), per-channel scale (FP16),
            direction as packed INT3.
    Reconstructs via unpack → dequantize → renormalize.
    """
    def __init__(self):
        self.packed_d = None      # uint8, shape [B, H, T, 3*D/8]
        self.norm = None          # fp16, shape [B, H, T, 1]
        self.ch_scale = None      # fp16, shape [B, H, 1, D]

    @classmethod
    def from_fp16(cls, x):
        """
        x: fp16 tensor [B, H, T, D]. D must be divisible by 8.
        """
        assert x.shape[-1] % 8 == 0, f"Last dim must be divisible by 8, got {x.shape[-1]}"
        obj = cls()
        xf = x.float()
        n = xf.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        d = xf / n
        ch_scale = d.abs().amax(dim=-2, keepdim=True).clamp(min=1e-12) / 3  # INT3 qm=3
        d_int3 = (d / ch_scale).round().clamp(-3, 3).to(torch.int8)
        obj.packed_d = pack_int3_signed(d_int3)
        obj.norm = n.half()
        obj.ch_scale = ch_scale.half()
        return obj

    def to_fp16(self):
        """Reconstruct approximate fp16 tensor."""
        *prefix, P = self.packed_d.shape
        D = P * 8 // 3
        d_int3 = unpack_int3_signed(self.packed_d, N=D)
        d_q = d_int3.float() * self.ch_scale.float()
        d_renorm = d_q / d_q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return (self.norm.float() * d_renorm).half()

    def size_bytes(self):
        """Total bytes in storage (packed direction + norm + scales)."""
        return (self.packed_d.numel() * 1 +
                self.norm.numel() * 2 +
                self.ch_scale.numel() * 2)


def simulated_nsep_pchan3(x):
    """Simulated (fake) INT3 nsep+pchan — for quality comparison."""
    xf = x.float()
    n = xf.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    d = xf / n
    qm = 3  # INT3
    s = d.abs().amax(dim=-2, keepdim=True).clamp(min=1e-12) / qm
    dq = (d / s).round().clamp(-qm, qm) * s
    dq = dq / dq.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return (n * dq).to(x.dtype)


# =============================================================
# === CELL R3 === Load target model and eval
# =============================================================

# Target a Mode 2, instruction-tuned, production-grade model.
# Llama-3.1-8B-Instruct chosen because:
#   - already cached from Phase 3 (fast load)
#   - 8B fits easily in Colab Pro VRAM
#   - INT3 nsep+pchan = +0.27 ΔPPL (3.0% of baseline) in Phase 3 -- clean PRODUCTION reference
#   - Meta recipe = cleanest activation profile in our sample
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

print(f"\nLoading {MODEL_ID} ...")
tok = AutoTokenizer.from_pretrained(MODEL_ID)
mdl = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, device_map="auto",
    use_safetensors=True,
)
mdl.eval()
print(f"VRAM used: {torch.cuda.memory_allocated()/1e9:.1f} GB")


# ─── WikiText-2 PPL helpers (reuse from phase3/4) ────────────
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
wtext = "\n\n".join([x for x in ds["text"] if x.strip()])[:48000]

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

def compress_cache(past, mode):
    """mode: 'baseline' | 'simulated' | 'real_packed'"""
    if mode == "baseline":
        return
    for li in range(n_cache_layers(past)):
        ok, ov = get_kv(past, li)
        ok = ok.clone(); ov = ov.clone()
        if mode == "simulated":
            nk = simulated_nsep_pchan3(ok)
            nv = simulated_nsep_pchan3(ov)
        elif mode == "real_packed":
            nk = Int3NsepPchanCache.from_fp16(ok).to_fp16()
            nv = Int3NsepPchanCache.from_fp16(ov).to_fp16()
        set_kv(past, li, nk, nv)


def ppl_with(mode, ctx_len=1024, stride=512, max_windows=10):
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
            compress_cache(past, mode)
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


# ─── Run three evaluations ───────────────────────────────────
print(f"\n{'='*70}")
print(f"Quality test: simulated vs real-packed INT3 nsep+pchan")
print(f"{'='*70}")
print(f"  {'Mode':<20} | {'PPL':>10} | {'ΔPPL':>10}")
print(f"  {'-'*47}")

results = {}
for mode in ["baseline", "simulated", "real_packed"]:
    print(f"  Running {mode} ...", flush=True)
    ppl, nt = ppl_with(mode)
    delta = ppl - results.get("baseline", {}).get("ppl", ppl) if mode != "baseline" else 0.0
    results[mode] = {"ppl": ppl, "tokens": nt, "delta": delta}
    dstr = f"+{delta:.4f}" if mode != "baseline" else "-"
    print(f"  {mode:<20} | {ppl:>10.4f} | {dstr:>10}")

# Lossless packing check
sim_ppl = results["simulated"]["ppl"]
real_ppl = results["real_packed"]["ppl"]
ppl_diff = abs(sim_ppl - real_ppl)
print(f"\n  |simulated - real_packed| = {ppl_diff:.6f}")
if ppl_diff < 0.001:
    print(f"  ✓ LOSSLESS packing (PPL match within 1e-3)")
else:
    print(f"  ⚠️  PPL difference detected, investigate")


# =============================================================
# === CELL R4 === Real memory measurement
# =============================================================

print(f"\n{'='*70}")
print(f"Memory measurement: FP16 vs INT3-packed KV cache")
print(f"{'='*70}")

# Build a 2048-token context cache, measure bytes
SEQ_LEN = 2048
probe = "The scientific method represents a fundamental approach " * 100
ids = tok.encode(probe, return_tensors="pt")[:, :SEQ_LEN].to("cuda")

with torch.inference_mode():
    out = mdl(ids, use_cache=True)
    past = out.past_key_values

# FP16 size: sum over all cache layers
fp16_bytes = 0
packed_bytes = 0
n_layers = n_cache_layers(past)

for li in range(n_layers):
    k, v = get_kv(past, li)
    fp16_bytes += k.element_size() * k.numel()
    fp16_bytes += v.element_size() * v.numel()
    # Ensure head_dim divisible by 8
    if k.shape[-1] % 8 != 0:
        print(f"  ⚠️  Layer {li}: head_dim {k.shape[-1]} not divisible by 8, skipping")
        continue
    pk = Int3NsepPchanCache.from_fp16(k)
    pv = Int3NsepPchanCache.from_fp16(v)
    packed_bytes += pk.size_bytes() + pv.size_bytes()

fp16_mb = fp16_bytes / 1e6
packed_mb = packed_bytes / 1e6
ratio = fp16_bytes / packed_bytes if packed_bytes > 0 else float('nan')

print(f"\n  Context: {SEQ_LEN} tokens, {n_layers} cache layers")
print(f"  FP16 KV cache:        {fp16_mb:>8.2f} MB")
print(f"  INT3-packed KV cache: {packed_mb:>8.2f} MB")
print(f"  Compression ratio:    {ratio:>8.2f}x")

# Theoretical
print(f"\n  Theoretical limits:")
print(f"    Perfect 3 bits/value:   {16/3:.2f}x compression")
print(f"    Our overhead (norm + scale) adds ~{(ratio - 16/3)/ratio * 100:.1f}% to ideal")

# Save
final = {
    "experiment": "phase5_int3_real_packing",
    "model": MODEL_ID,
    "timestamp": datetime.now().isoformat(),
    "ppl_results": results,
    "lossless_packing_verified": ppl_diff < 0.001,
    "ppl_difference_sim_vs_real": ppl_diff,
    "memory_fp16_mb": fp16_mb,
    "memory_packed_mb": packed_mb,
    "compression_ratio": ratio,
    "seq_len": SEQ_LEN,
    "n_cache_layers": n_layers,
}
print(f"\n{'='*70}\nFINAL JSON\n{'='*70}")
print(json.dumps(final, indent=2, default=str))
