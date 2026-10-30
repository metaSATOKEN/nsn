"""phase_k1_production_kernel_benchmark.py

K1: production-kernel runtime benchmark (NSN vs a production KIVI kernel),
Deliverable 1 = decode-step attention microbenchmark.
Pre-registration: experiments/K1_production_kernel_preregistration.md
(locked at 23f694d, before this harness existed).

GPU-FREE-VERIFIED  (run `python phase_k1_production_kernel_benchmark.py --selftest`
on any machine, CPU only):
  * reference quant/dequant for KIVI and NSN K (bit-identical vendor of
    phase_e1_causal_faithful_ci.py),
  * the store-time `effective_norm` folding, proven EXACT against the two-step
    phase_e1 re-normalisation: the fairness keystone (pre-reg Section 2), i.e.
    NSN decode == KIVI decode + ONE per-(token,head) multiply, no extra reduction,
  * the INT bit-packing round-trip and the bytes/token accounting
    (NSN minus KIVI == exactly 16/head_dim bpv, the per-token norm).

NEEDS A GPU  (Colab A100 80GB, `--bench`):
  * the operating-point grid (pre-reg Section 4), reported in full,
  * a torch GPU path that runs anywhere with CUDA and gives the NSN-vs-KIVI
    dequant-overhead delta (the core fairness number) plus a pessimistic absolute,
  * a fused Triton dequant->score kernel (FP16 / KIVI / NSN through one IS_NSN
    flag) that is bandwidth-faithful; THIS is the Colab-iteration target and is
    not covered by the CPU self-tests (no CPU Triton). KIVI's published CUDA
    kernel is the secondary absolute reference, loaded on Colab if present.

PASTE-AND-RUN (no repo / no git clone assumed): this file is self-contained
(torch + stdlib + optional triton, no repo imports). Run it with no flags, or
paste the whole file into a Colab GPU cell: it self-tests and then, if a GPU is
present, runs the grid. `--selftest` / `--bench` force a single action.

Fairness (pre-reg Section 2): NSN and KIVI share ONE kernel base; NSN is KIVI
plus a single `acc *= eff_norm` guarded by `if IS_NSN`. Every result is written
save-first to results/k1_production_kernel.json (the W1c disconnect lesson).
"""
import argparse
import json
import math
import os
import statistics
import subprocess
import time

import torch

SEED = 20260510
EPS = 1e-12        # matches phase_e1 (step / norm clamp)
RENORM_EPS = 1e-8  # matches phase_e1 (direction re-normalisation clamp)

# KIVI published kernel-runtime defaults (secondary absolute reference): group
# G=32, full-precision residual R=128 most-recent tokens. The shared-base primary
# uses the SAME group/residual for both recipes, so any runtime difference cannot
# be an optimisation-effort gap. Storage accounting follows Appendix B (bpv_K),
# whose convention is group-infinite / residual-empty; the production group and
# residual add a recipe-independent term that cancels in the NSN-minus-KIVI delta.
KIVI_GROUP = 32
KIVI_RESIDUAL = 128

# Operating-point grid (pre-reg Section 4), Cartesian product reported in full.
GRID_BATCH = [1, 8, 32, 64]
GRID_CTX = [2048, 8192, 32768, 65536]
# (recipe, bits): matched-bit primary (nsn5 vs kivi5), KIVI native (kivi4),
# aggressive tier where Section 7.1 quality advantage lives (nsn4), FP16 ceiling.
GRID_RECIPES = [("nsn", 5), ("kivi", 5), ("kivi", 4), ("nsn", 4), ("fp16", 16)]
# Two head_dim regimes (NSN overhead scales as 16/head_dim). n_kv / n_q are
# representative GQA values; the Colab bench overrides them from the real HF
# config when a model is loaded. The NSN-vs-KIVI delta is geometry-robust.
GRID_MODELS = [
    {"name": "Gemma4-26B-A4B", "n_kv": 8, "head_dim": 256, "n_q": 32},  # hotspot hero
    {"name": "Qwen3-30B-A3B", "n_kv": 8, "head_dim": 128, "n_q": 32},   # non-Gemma generality (W1e)
]


# =============================================================================
# Section 1: reference quant/dequant (CPU), bit-identical vendor of phase_e1.
# =============================================================================
def _asym_qdq_over_tokens(x, bits):
    """phase_e1 _asym_qdq_over_tokens: per-channel asym q->dq, scale over dim=1."""
    levels = (1 << bits) - 1
    x_min = x.amin(dim=1, keepdim=True)
    x_max = x.amax(dim=1, keepdim=True)
    step = ((x_max - x_min).float() / levels).clamp(min=EPS).to(x.dtype)
    q = ((x - x_min).float() / step.float()).round().clamp(0, levels)
    return q.to(x.dtype) * step + x_min


def _nsn_qdq_over_tokens(d, bits):
    """phase_e1 _nsn_qdq_over_tokens: per-channel asym q->dq of a direction block."""
    levels = (1 << bits) - 1
    d_min = d.amin(dim=1, keepdim=True)
    d_max = d.amax(dim=1, keepdim=True)
    step = ((d_max - d_min).float() / levels).clamp(min=EPS).to(d.dtype)
    return ((d - d_min).float() / step.float()).round().clamp(0, levels).to(d.dtype) * step + d_min


def ref_kivi_qdq(k, bits):
    """phase_e1 quant_k_kivi, A0 full-window (the in-harness scale)."""
    return _asym_qdq_over_tokens(k, bits)


def ref_nsn_qdq(k, bits, n_kv, head_dim):
    """phase_e1 quant_k_nsn, A0 full-window: the two-step re-normalised reference."""
    B, T, hidden = k.shape
    x = k.reshape(B, T, n_kv, head_dim).float()
    norm = x.norm(dim=-1, keepdim=True).clamp(min=EPS)
    d = x / norm
    dq = _nsn_qdq_over_tokens(d, bits)
    dq = dq / dq.norm(dim=-1, keepdim=True).clamp(min=RENORM_EPS)
    return (norm * dq).reshape(B, T, hidden).to(k.dtype)


# =============================================================================
# Section 2: kernel storage representation + the effective_norm folding.
# The kernel reads CODES; these produce what is stored and let the CPU self-test
# prove the one-multiply decode equals the two-step reference exactly.
# =============================================================================
def _quant_codes_over_tokens(x, bits):
    """Per-channel asym quant; return (code in [0,levels], x_min, step), scale over dim=1."""
    levels = (1 << bits) - 1
    x_min = x.amin(dim=1, keepdim=True)
    x_max = x.amax(dim=1, keepdim=True)
    step = ((x_max - x_min).float() / levels).clamp(min=EPS).to(x.dtype)
    code = ((x - x_min).float() / step.float()).round().clamp(0, levels).to(x.dtype)
    return code, x_min, step


def kivi_store(k, bits):
    """KIVI K kernel storage (A0). code/min/step over the token axis, per channel."""
    code, k_min, step = _quant_codes_over_tokens(k, bits)
    return {"code": code, "min": k_min, "step": step}


def kivi_decode(store):
    """KIVI decode: one per-channel affine. This is the shared kernel base."""
    return store["code"] * store["step"] + store["min"]


def nsn_store(k, bits, n_kv, head_dim):
    """NSN K kernel storage (A0): KIVI's storage applied to the unit direction,
    PLUS one per-(token,head) effective_norm that folds the re-normalisation
    `norm / ||dequant(direction)||`, computed at store time from the stored codes."""
    B, T, hidden = k.shape
    x = k.reshape(B, T, n_kv, head_dim).float()
    norm = x.norm(dim=-1, keepdim=True).clamp(min=EPS)              # (B,T,n_kv,1)
    d = x / norm
    code, d_min, step = _quant_codes_over_tokens(d, bits)          # scale over tokens (dim=1)
    dq = code * step + d_min                                        # the kernel's KIVI-path output
    eff_norm = norm / dq.norm(dim=-1, keepdim=True).clamp(min=RENORM_EPS)
    return {
        "code": code.reshape(B, T, hidden),
        "min": d_min.reshape(B, 1, hidden),
        "step": step.reshape(B, 1, hidden),
        "eff_norm": eff_norm.reshape(B, T, n_kv),
        "n_kv": n_kv,
        "head_dim": head_dim,
    }


def nsn_decode(store):
    """NSN decode == KIVI decode of the direction, then ONE multiply by eff_norm.
    No reduction at decode (the norm of the dequantised direction was folded in)."""
    B, T, hidden = store["code"].shape
    n_kv, head_dim = store["n_kv"], store["head_dim"]
    dq = store["code"] * store["step"] + store["min"]              # KIVI-path dequant (B,T,hidden)
    dq = dq.reshape(B, T, n_kv, head_dim)
    out = store["eff_norm"].unsqueeze(-1) * dq                     # the single extra op
    return out.reshape(B, T, hidden)


# =============================================================================
# Section 3: INT bit-packing (real storage) + bytes/token memory accounting.
# =============================================================================
def pack_codes(codes_int, bits):
    """Pack b-bit codes (values in [0, 2^b-1]) into a uint8 stream, LSB-first."""
    codes_int = codes_int.to(torch.int64).reshape(-1)
    n = codes_int.numel()
    bit_pos = torch.arange(bits, dtype=torch.int64)
    bits_mat = (codes_int.unsqueeze(1) >> bit_pos) & 1             # (n, bits) LSB-first
    stream = bits_mat.reshape(-1)
    pad = (-stream.numel()) % 8
    if pad:
        stream = torch.cat([stream, torch.zeros(pad, dtype=torch.int64)])
    stream = stream.reshape(-1, 8)
    weights = (1 << torch.arange(8, dtype=torch.int64))
    return (stream * weights).sum(dim=1).to(torch.uint8)


def unpack_codes(packed, bits, n):
    """Inverse of pack_codes for the first n codes."""
    bitstream = ((packed.to(torch.int64).unsqueeze(1) >> torch.arange(8, dtype=torch.int64)) & 1).reshape(-1)
    bitstream = bitstream[: n * bits].reshape(n, bits)
    weights = (1 << torch.arange(bits, dtype=torch.int64))
    return (bitstream * weights).sum(dim=1)


def bpv_K(bits, head_dim, is_nsn, T):
    """Appendix B effective bits-per-value for the K cache (line 399):
        KIVI: bits + 32/T                  (per-channel min+step amortised over tokens)
        NSN:  bits + 16/head_dim + 32/T    (+ the per-token FP16 norm)
    This is the paper's storage convention (group effectively infinite, residual
    buffer empty) that the compression claims rest on. A production kernel's
    finite group and residual add a recipe-INDEPENDENT term, so the NSN-minus-KIVI
    delta is exactly 16/head_dim regardless of T or grouping (== 2*n_kv bytes/token)."""
    base = bits + 32.0 / T
    if is_nsn:
        base += 16.0 / head_dim
    return base


def bytes_per_token_K(bits, n_kv, head_dim, T, is_nsn):
    """Stored K bytes per token, cross-checking Appendix B (5.1 bpv NSN-INT5, ~3x)."""
    return bpv_K(bits, head_dim, is_nsn, T) * (n_kv * head_dim) / 8.0


# =============================================================================
# Section 4: GPU benchmark paths.
# torch path (runs on any CUDA device): dequant with torch, score with matmul.
# Not bandwidth-faithful (materialises FP16 K), so it measures the dequant-
# overhead delta (NSN vs KIVI, the core fairness number) and a PESSIMISTIC
# absolute; the bandwidth win vs FP16 is the Triton path.
# =============================================================================
def _make_inputs(model, B, T, bits, recipe, device, dtype=torch.float16):
    """Build a query and a quantised K cache store for one cell."""
    n_kv, head_dim, n_q = model["n_kv"], model["head_dim"], model["n_q"]
    hidden = n_kv * head_dim
    g = torch.Generator(device="cpu").manual_seed(SEED)
    q = torch.randn(B, n_q, head_dim, generator=g, dtype=torch.float32).to(device, dtype)
    k = torch.randn(B, T, hidden, generator=g, dtype=torch.float32)
    if recipe == "fp16":
        store = {"kfp16": k.to(device, dtype), "n_kv": n_kv, "head_dim": head_dim}
    elif recipe == "kivi":
        s = kivi_store(k, bits)
        store = {kk: v.to(device, dtype) for kk, v in s.items()}
        store.update({"n_kv": n_kv, "head_dim": head_dim})
    elif recipe == "nsn":
        s = nsn_store(k, bits, n_kv, head_dim)
        store = {kk: (v.to(device, dtype) if torch.is_tensor(v) else v) for kk, v in s.items()}
    else:
        raise ValueError(recipe)
    return q, store, n_q, n_kv, head_dim


def _torch_decode_scores(q, store, recipe, n_q, n_kv, head_dim):
    """One decode step's K-side score (B, n_q, T), dequant included in the timed path."""
    B = q.shape[0]
    if recipe == "fp16":
        kdq = store["kfp16"]                                       # (B,T,hidden)
    elif recipe == "kivi":
        kdq = kivi_decode(store)
    else:
        kdq = nsn_decode(store)
    T = kdq.shape[1]
    kdq = kdq.reshape(B, T, n_kv, head_dim)
    rep = n_q // n_kv                                              # GQA group factor
    qx = q.reshape(B, n_kv, rep, head_dim)                         # group queries by KV head
    # scores[b, n_kv*rep, t] WITHOUT materialising the rep-expanded K cache
    scores = torch.einsum("bnrd,btnd->bnrt", qx, kdq) / math.sqrt(head_dim)
    return scores.reshape(B, n_q, T)


def _time_region(fn, warmup, iters):
    """Median and samples of per-iter ms on CUDA via events; CPU fallback uses perf_counter."""
    if torch.cuda.is_available():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        samples = []
        for _ in range(iters):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            samples.append(s.elapsed_time(e))
        return samples
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)
    return samples


def _bootstrap_ci(samples, n_boot=5000, alpha=0.05):
    """Median with a percentile bootstrap CI (same machinery as the quality side)."""
    x = torch.tensor(samples, dtype=torch.float64)
    n = x.numel()
    g = torch.Generator().manual_seed(SEED)
    meds = torch.stack([x[torch.randint(0, n, (n,), generator=g)].median() for _ in range(n_boot)])
    lo = torch.quantile(meds, alpha / 2).item()
    hi = torch.quantile(meds, 1 - alpha / 2).item()
    return float(x.median().item()), lo, hi


def _sample_power_w():
    """Best-effort instantaneous GPU power draw (W) via nvidia-smi; None if absent."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=5).decode().strip().splitlines()
        return float(out[0])
    except Exception:
        return None


def bench_cell_torch(model, B, T, recipe, bits, device, warmup=30, iters=100):
    """Time one cell on the torch path. Returns a result dict (save-first by caller)."""
    q, store, n_q, n_kv, head_dim = _make_inputs(model, B, T, bits, recipe, device)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    fn = lambda: _torch_decode_scores(q, store, recipe, n_q, n_kv, head_dim)
    samples = _time_region(fn, warmup, iters)
    med, lo, hi = _bootstrap_ci(samples)
    peak_mb = (torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else None
    return {
        "model": model["name"], "batch": B, "ctx": T, "recipe": recipe, "bits": bits,
        "path": "torch", "n_kv": n_kv, "head_dim": head_dim, "n_q": n_q,
        "latency_ms_median": med, "latency_ms_ci": [lo, hi], "n_iters": iters,
        "peak_mem_mb": peak_mb, "power_w": _sample_power_w(),
        "bytes_per_token_K": None if recipe == "fp16"
        else bytes_per_token_K(bits, n_kv, head_dim, T, is_nsn=(recipe == "nsn")),
        "bytes_per_token_K_fp16": 2.0 * n_kv * head_dim,
    }


# --- Triton fused dequant->score kernel (Colab-iteration target; not CPU-tested) ---
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:
    HAS_TRITON = False

if HAS_TRITON:

    @triton.jit
    def _fused_dequant_score_kernel(
        q_ptr, code_ptr, min_ptr, step_ptr, norm_ptr, s_ptr,
        B, Hq, Hkv, T, D,
        sqb, sqh, sqd, scb, sct, scc, ssb, ssh, sst, snb, snt,
        IS_NSN: tl.constexpr, IS_FP16: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        # One program computes scores for (batch pid_b, query-head pid_h, token-block pid_t).
        # K is stored as per-channel codes (B,T,hidden) with per-channel (min,step) over
        # channels, and (NSN) a per-(token,head) effective_norm. NSN differs from KIVI by
        # exactly the `kdq *= enorm` line under IS_NSN. THIS kernel is the Colab-iteration
        # target: the pointer strides and GQA mapping below need on-device verification.
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_t = tl.program_id(2)
        hkv = pid_h * Hkv // Hq                                    # GQA head mapping
        offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        offs_d = tl.arange(0, BLOCK_D)
        d_mask = offs_d < D
        t_mask = offs_t < T
        q = tl.load(q_ptr + pid_b * sqb + pid_h * sqh + offs_d * sqd, mask=d_mask, other=0.0)
        ch = hkv * D + offs_d                                      # channel index into hidden
        code = tl.load(code_ptr + pid_b * scb + offs_t[:, None] * sct + ch[None, :] * scc,
                       mask=t_mask[:, None] & d_mask[None, :], other=0.0)
        if IS_FP16:
            kdq = code                                            # code_ptr already holds FP16 K
        else:
            kmin = tl.load(min_ptr + ch, mask=d_mask, other=0.0)
            kstep = tl.load(step_ptr + ch, mask=d_mask, other=0.0)
            kdq = code * kstep[None, :] + kmin[None, :]
            if IS_NSN:
                enorm = tl.load(norm_ptr + pid_b * snb + offs_t * snt + hkv, mask=t_mask, other=0.0)
                kdq = kdq * enorm[:, None]                         # the single extra op
        s = tl.sum(kdq * q[None, :], axis=1) * (1.0 / tl.sqrt(D.to(tl.float32)))
        tl.store(s_ptr + pid_b * ssb + pid_h * ssh + offs_t * sst, s, mask=t_mask)

# A device-side runner that wires the grid to the Triton kernel will be added on
# Colab once `_fused_dequant_score_kernel` is verified against `_torch_decode_scores`
# (numerical-equivalence gate, pre-reg Section 10) on the target GPU.


# =============================================================================
# Section 5: grid driver (save-first JSON).
# =============================================================================
def run_bench(out_path, device="cuda", warmup=30, iters=100):
    results = []
    payload = {"meta": {"seed": SEED, "warmup": warmup, "iters": iters,
                        "torch": torch.__version__, "cuda": torch.version.cuda,
                        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                        "has_triton": HAS_TRITON, "kivi_group": KIVI_GROUP, "kivi_residual": KIVI_RESIDUAL,
                        "note": "torch path (dequant materialised, pessimistic absolute); "
                                "fused Triton bandwidth path is the Colab-iteration refinement"},
               "cells": results}
    for model in GRID_MODELS:
        for B in GRID_BATCH:
            for T in GRID_CTX:
                for recipe, bits in GRID_RECIPES:
                    try:
                        r = bench_cell_torch(model, B, T, recipe, bits, device, warmup, iters)
                    except RuntimeError as ex:                     # OOM etc.: log, do not skip silently
                        r = {"model": model["name"], "batch": B, "ctx": T, "recipe": recipe,
                             "bits": bits, "path": "torch", "error": str(ex)[:200]}
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    results.append(r)
                    with open(out_path, "w") as f:                 # save-first after every cell
                        json.dump(payload, f, indent=2)
                    tag = r.get("latency_ms_median")
                    print(f"[k1] {model['name']:>16} B={B:>2} T={T:>5} {recipe}{bits} "
                          + (f"{tag:.4f} ms" if tag is not None else f"ERR {r.get('error','')[:60]}"))
    print(f"[k1] wrote {out_path} ({len(results)} cells)")
    return payload


# =============================================================================
# Section 6: GPU-free self-tests (the verified foundation).
# =============================================================================
def _rand_k(B=2, T=257, n_kv=4, head_dim=128, seed=SEED):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, T, n_kv * head_dim, generator=g, dtype=torch.float32)


def test_kivi_decode_matches_ref():
    k = _rand_k()
    for bits in (4, 5, 6):
        got = kivi_decode(kivi_store(k, bits))
        ref = ref_kivi_qdq(k, bits)
        assert torch.allclose(got, ref, atol=1e-6, rtol=0), f"KIVI decode != ref at {bits}b"
    print("  ok  test_kivi_decode_matches_ref (split storage == phase_e1 KIVI A0)")


def test_nsn_folding_exact():
    """The fairness keystone: NSN one-multiply decode == two-step phase_e1 reference."""
    k = _rand_k()
    for n_kv, head_dim in ((4, 128), (8, 256)):
        kk = _rand_k(n_kv=n_kv, head_dim=head_dim)
        for bits in (4, 5):
            got = nsn_decode(nsn_store(kk, bits, n_kv, head_dim))
            ref = ref_nsn_qdq(kk, bits, n_kv, head_dim)
            rel = (got - ref).abs().max() / ref.abs().max().clamp(min=EPS)
            assert rel < 1e-6, f"NSN folding rel err {rel:.2e} at {bits}b hd{head_dim}"
    print("  ok  test_nsn_folding_exact (decode = KIVI-path + 1 multiply, rel < 1e-6)")


def test_nsn_is_kivi_plus_one_multiply():
    """With eff_norm forced to 1, NSN decode reduces to the direction's KIVI dequant."""
    kk = _rand_k(n_kv=4, head_dim=128)
    s = nsn_store(kk, 5, 4, 128)
    base = (s["code"] * s["step"] + s["min"])                     # KIVI-path on the direction
    s1 = dict(s); s1["eff_norm"] = torch.ones_like(s["eff_norm"])
    assert torch.allclose(nsn_decode(s1), base, atol=1e-6, rtol=0)
    print("  ok  test_nsn_is_kivi_plus_one_multiply (only op beyond KIVI is x eff_norm)")


def test_packing_roundtrip():
    g = torch.Generator().manual_seed(SEED)
    for bits in (4, 5, 6):
        n = 1000
        codes = torch.randint(0, 1 << bits, (n,), generator=g)
        back = unpack_codes(pack_codes(codes, bits), bits, n)
        assert torch.equal(codes.to(torch.int64), back), f"packing roundtrip failed at {bits}b"
    print("  ok  test_packing_roundtrip (INT4/5/6 pack->unpack lossless)")


def test_memory_delta_is_norm_only():
    for n_kv, head_dim in ((4, 128), (8, 256)):
        for T in (2048, 32768):
            kivi = bytes_per_token_K(5, n_kv, head_dim, T, is_nsn=False)
            nsn = bytes_per_token_K(5, n_kv, head_dim, T, is_nsn=True)
            delta = nsn - kivi
            assert abs(delta - 2.0 * n_kv) < 1e-9, f"delta {delta} != 2*n_kv {2*n_kv}"
            delta_bpv = delta * 8 / (n_kv * head_dim)
            assert abs(delta_bpv - 16.0 / head_dim) < 1e-9, f"delta_bpv {delta_bpv} != 16/hd"
    print("  ok  test_memory_delta_is_norm_only (NSN - KIVI == 2*n_kv B/tok == 16/head_dim bpv)")


def run_selftests():
    torch.manual_seed(SEED)
    print("[k1] GPU-free self-tests (the verified foundation):")
    test_kivi_decode_matches_ref()
    test_nsn_folding_exact()
    test_nsn_is_kivi_plus_one_multiply()
    test_packing_roundtrip()
    test_memory_delta_is_norm_only()
    print("[k1] all CPU self-tests passed. The fused Triton kernel is the Colab-iteration "
          "target (numerical-equivalence gate runs on the GPU, pre-reg Section 10).")


if __name__ == "__main__":
    # Paste-and-run friendly (no repo / no git clone assumed): with no flags this
    # self-tests and then, if a GPU is present, runs the grid. Pasting the whole
    # file into a Colab GPU cell therefore runs the full benchmark. Explicit flags
    # force a single action.
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="run only the GPU-free self-tests")
    ap.add_argument("--bench", action="store_true", help="run only the operating-point grid (needs CUDA)")
    ap.add_argument("--out", default="results/k1_production_kernel.json")
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--iters", type=int, default=100)
    args, _ = ap.parse_known_args()  # ignore a notebook kernel's -f kernel.json argv
    no_flags = not args.selftest and not args.bench
    if args.selftest or no_flags:
        run_selftests()
    if args.bench or (no_flags and torch.cuda.is_available()):
        outdir = os.path.dirname(args.out)
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        run_bench(args.out, warmup=args.warmup, iters=args.iters)
