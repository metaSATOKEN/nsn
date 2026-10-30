"""phase_k1c_triton_fused_kernel.py

K1c: the fused-kernel gate-closer for the K1 production-kernel benchmark.
Pre-registration: experiments/K1_production_kernel_preregistration.md (locked 23f694d).
Third in the K1 line, kept as its OWN self-contained file for traceability:
  phase_k1  (eager torch path)        -> results/k1_production_kernel.json
  phase_k1b (torch.compile)           -> results/k1b_compile_corroboration.json
  phase_k1c (Triton fused kernel)     -> results/k1c_triton_fused.json   (this file)

WHY. The eager and torch.compile paths both MATERIALISE K, which forces NSN's
per-token eff_norm to be a separate memory pass over the whole K cache: NSN runs
about +20 percent over KIVI at scale, and torch.compile does not remove it (k1b).
A real production kernel never materialises K: it reads the quantised cache and
dequantises IN-REGISTER inside the attention dot. There, because eff_norm is
per-token it factors OUT of the dot, so NSN is KIVI plus one per-token scalar
multiply on the score (not a per-element multiply) and one small extra HBM read
(eff_norm, +16/head_dim bpv). This file is that kernel: a fused dequant-then-score
Triton kernel for the decode step (the K side, where the recipes differ), with
FP16 / KIVI / NSN selected by one IS_NSN / IS_FP16 flag, fair by construction.

FIRST-CUT SCOPE. Codes are stored as int8 (one byte) rather than bit-packed to
2^bits, so the vs-FP16 absolute it shows is a LOWER BOUND on the bandwidth win
(real INT5 packing reads fewer bytes still); the memory accounting reports the
packed Appendix-B size separately. The NSN-vs-KIVI delta (the gate's core claim)
is exact regardless of packing. Bit-packed INT loads are the next refinement.

NUMERICAL-EQUIVALENCE GATE (pre-reg Section 10). Before any cell is timed, the
kernel's scores are checked against the reference `_torch_decode_scores` in fp32;
a cell is timed only if it passes, and the relative error is recorded. A fast but
wrong kernel is disqualified before it is timed. This is the safety net for the
on-GPU iteration: a gross stride / GQA / dequant bug shows up as a large rel here.

PASTE-AND-RUN (no repo / no git clone). Self-contained. Paste into a Colab GPU
cell: it self-tests (CPU), then if CUDA + triton are present runs the gate and the
grid to results/k1c_triton_fused.json (save-first) and auto-downloads it.
"""
import json
import math
import os
import subprocess
import time

import torch

SEED = 20260510
EPS = 1e-12
RENORM_EPS = 1e-8

GRID_BATCH = [1, 8, 32, 64]
GRID_CTX = [2048, 8192, 32768, 65536]
GRID_RECIPES = [("nsn", 5), ("kivi", 5), ("kivi", 4), ("nsn", 4), ("fp16", 16)]
GRID_MODELS = [
    {"name": "Gemma4-26B-A4B", "n_kv": 8, "head_dim": 256, "n_q": 32},
    {"name": "Qwen3-30B-A3B", "n_kv": 8, "head_dim": 128, "n_q": 32},
]

GATE_TOL = 1e-2   # rel-max gate; a correct kernel sits near 1e-6, a gross bug near 1
BLOCK_T = 32      # token block per program (tunable on GPU)


# =============================================================================
# Foundation (vendored bit-identical from phase_k1).
# =============================================================================
def _quant_codes_over_tokens(x, bits):
    levels = (1 << bits) - 1
    x_min = x.amin(dim=1, keepdim=True)
    x_max = x.amax(dim=1, keepdim=True)
    step = ((x_max - x_min).float() / levels).clamp(min=EPS).to(x.dtype)
    code = ((x - x_min).float() / step.float()).round().clamp(0, levels).to(x.dtype)
    return code, x_min, step


def _nsn_qdq_over_tokens(d, bits):
    levels = (1 << bits) - 1
    d_min = d.amin(dim=1, keepdim=True)
    d_max = d.amax(dim=1, keepdim=True)
    step = ((d_max - d_min).float() / levels).clamp(min=EPS).to(d.dtype)
    return ((d - d_min).float() / step.float()).round().clamp(0, levels).to(d.dtype) * step + d_min


def ref_nsn_qdq(k, bits, n_kv, head_dim):
    B, T, hidden = k.shape
    x = k.reshape(B, T, n_kv, head_dim).float()
    norm = x.norm(dim=-1, keepdim=True).clamp(min=EPS)
    d = x / norm
    dq = _nsn_qdq_over_tokens(d, bits)
    dq = dq / dq.norm(dim=-1, keepdim=True).clamp(min=RENORM_EPS)
    return (norm * dq).reshape(B, T, hidden).to(k.dtype)


def kivi_store(k, bits):
    code, k_min, step = _quant_codes_over_tokens(k, bits)
    return {"code": code, "min": k_min, "step": step}


def nsn_store(k, bits, n_kv, head_dim):
    B, T, hidden = k.shape
    x = k.reshape(B, T, n_kv, head_dim).float()
    norm = x.norm(dim=-1, keepdim=True).clamp(min=EPS)
    d = x / norm
    code, d_min, step = _quant_codes_over_tokens(d, bits)
    dq = code * step + d_min
    eff_norm = norm / dq.norm(dim=-1, keepdim=True).clamp(min=RENORM_EPS)
    return {"code": code.reshape(B, T, hidden), "min": d_min.reshape(B, 1, hidden),
            "step": step.reshape(B, 1, hidden), "eff_norm": eff_norm.reshape(B, T, n_kv),
            "n_kv": n_kv, "head_dim": head_dim}


def nsn_decode(store):
    B, T, hidden = store["code"].shape
    n_kv, head_dim = store["n_kv"], store["head_dim"]
    dq = store["code"] * store["step"] + store["min"]
    dq = dq.reshape(B, T, n_kv, head_dim)
    out = store["eff_norm"].unsqueeze(-1) * dq
    return out.reshape(B, T, hidden)


def kivi_decode(store):
    return store["code"] * store["step"] + store["min"]


def bpv_K(bits, head_dim, is_nsn, T):
    base = bits + 32.0 / T
    if is_nsn:
        base += 16.0 / head_dim
    return base


def bytes_per_token_K(bits, n_kv, head_dim, T, is_nsn):
    return bpv_K(bits, head_dim, is_nsn, T) * (n_kv * head_dim) / 8.0


def _make_inputs(model, B, T, bits, recipe, device, dtype=torch.float16):
    n_kv, head_dim, n_q = model["n_kv"], model["head_dim"], model["n_q"]
    g = torch.Generator(device="cpu").manual_seed(SEED)
    q = torch.randn(B, n_q, head_dim, generator=g, dtype=torch.float32).to(device, dtype)
    k = torch.randn(B, T, n_kv * head_dim, generator=g, dtype=torch.float32)
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
    """Reference: dequant (materialised) then score. The gate compares against this."""
    B = q.shape[0]
    if recipe == "fp16":
        kdq = store["kfp16"]
    elif recipe == "kivi":
        kdq = kivi_decode(store)
    else:
        kdq = nsn_decode(store)
    T = kdq.shape[1]
    kdq = kdq.reshape(B, T, n_kv, head_dim)
    rep = n_q // n_kv
    qx = q.reshape(B, n_kv, rep, head_dim)
    scores = torch.einsum("bnrd,btnd->bnrt", qx, kdq) / math.sqrt(head_dim)
    return scores.reshape(B, n_q, T)


# =============================================================================
# int8 cache layout for the bandwidth-faithful kernel (codes -> 1 byte).
# =============================================================================
def make_triton_store(model, B, T, bits, recipe, device):
    """Return (q, ref_store, tstore). ref_store is the materialised reference store
    (for the gate); tstore holds int8 codes + fp16 scales + fp16 eff_norm that the
    kernel reads. Codes are int8 (first cut); the packed size is reported via
    bytes_per_token_K. tstore tensors are contiguous on `device`."""
    q, ref_store, n_q, n_kv, hd = _make_inputs(model, B, T, bits, recipe, device)
    if recipe == "fp16":
        tstore = {"code": ref_store["kfp16"].contiguous(), "is_fp16": True, "is_nsn": False,
                  "T": T, "n_kv": n_kv, "head_dim": hd}
    else:
        code_u8 = ref_store["code"].round().clamp(0, 255).to(torch.uint8).contiguous()  # codes in [0, 2^bits-1]
        tstore = {"code": code_u8, "is_fp16": False, "is_nsn": (recipe == "nsn"),
                  "min": ref_store["min"].squeeze(1).contiguous(),     # (B, hidden)
                  "step": ref_store["step"].squeeze(1).contiguous(),   # (B, hidden)
                  "T": T, "n_kv": n_kv, "head_dim": hd}
        if recipe == "nsn":
            # store (B, n_kv, T) so the kernel's per-(head, token-block) load is contiguous
            # over tokens; the strided (B, T, n_kv) load was the ~6-7pct NSN-vs-KIVI residual
            tstore["eff_norm"] = ref_store["eff_norm"].transpose(1, 2).contiguous()  # (B, n_kv, T)
    return q, ref_store, tstore, n_q, n_kv, hd


# =============================================================================
# The fused dequant-then-score Triton kernel (Colab; CPU-guarded import).
# =============================================================================
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:
    HAS_TRITON = False

if HAS_TRITON:

    @triton.jit
    def _fused_score_kernel(
        q_ptr, code_ptr, smin_ptr, sstep_ptr, enorm_ptr, out_ptr,
        Hq, Hkv, T, D,
        sq_b, sq_h, sq_d,
        sc_b, sc_t, sc_c,
        ss_b, ss_c,
        se_b, se_t, se_h,
        so_b, so_h, so_t,
        inv_sqrt_d,
        IS_NSN: tl.constexpr, IS_FP16: tl.constexpr, BLK_T: tl.constexpr, BLK_D: tl.constexpr,
    ):
        # One program -> (batch pid_b, query-head pid_h, token-block pid_t).
        # score[t] = inv_sqrt_d * eff_norm[t] * sum_d q[d] * dir_dequant_K[t, hkv, d].
        # Pointers are int64 (B*T*hidden exceeds 2^31 at long ctx x large batch).
        # eff_norm is per-token, so it factors OUT of the dot: NSN is KIVI plus one
        # per-token scalar multiply on the score (not a per-element multiply).
        pid_b = tl.program_id(0).to(tl.int64)
        pid_h = tl.program_id(1).to(tl.int64)
        pid_t = tl.program_id(2)
        hkv = pid_h // (Hq // Hkv)                       # GQA: query-head -> kv-head (int64)
        offs_d = tl.arange(0, BLK_D)
        offs_t = pid_t * BLK_T + tl.arange(0, BLK_T)     # int32 token ids (< T <= 65536)
        d_mask = offs_d < D
        t_mask = offs_t < T
        offs_d64 = offs_d.to(tl.int64)
        offs_t64 = offs_t.to(tl.int64)
        q = tl.load(q_ptr + pid_b * sq_b + pid_h * sq_h + offs_d64 * sq_d,
                    mask=d_mask, other=0.0).to(tl.float32)
        ch = hkv * D + offs_d64                          # (BLK_D,) int64 channel into hidden
        code = tl.load(code_ptr + pid_b * sc_b + offs_t64[:, None] * sc_t + ch[None, :] * sc_c,
                       mask=t_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        if IS_FP16:
            kdq = code                                   # code_ptr already holds FP16 K
        else:
            kmin = tl.load(smin_ptr + pid_b * ss_b + ch * ss_c, mask=d_mask, other=0.0).to(tl.float32)
            kstep = tl.load(sstep_ptr + pid_b * ss_b + ch * ss_c, mask=d_mask, other=0.0).to(tl.float32)
            kdq = code * kstep[None, :] + kmin[None, :]  # dequant the DIRECTION (eff_norm not yet)
        s = tl.sum(kdq * q[None, :], axis=1) * inv_sqrt_d
        if IS_NSN:                                       # eff_norm pulled out of the dot: per-token
            en = tl.load(enorm_ptr + pid_b * se_b + offs_t64 * se_t + hkv * se_h,
                         mask=t_mask, other=0.0).to(tl.float32)
            s = s * en
        tl.store(out_ptr + pid_b * so_b + pid_h * so_h + offs_t64 * so_t, s, mask=t_mask)


def triton_decode_scores(q, tstore, n_q, n_kv, head_dim):
    """Host launcher. Returns scores (B, n_q, T) fp32. Requires CUDA + triton."""
    B = q.shape[0]
    T = tstore["T"]
    code = tstore["code"]
    out = torch.empty((B, n_q, T), device=q.device, dtype=torch.float32)
    dummy = torch.empty(1, device=q.device, dtype=torch.float16)
    smin = tstore.get("min", dummy)
    sstep = tstore.get("step", dummy)
    enorm = tstore.get("eff_norm", dummy)
    BLK_D = triton.next_power_of_2(head_dim)
    grid = (B, n_q, triton.cdiv(T, BLOCK_T))
    ss_b = smin.stride(0) if smin.dim() == 2 else 0
    ss_c = smin.stride(1) if smin.dim() == 2 else 0
    se_b = enorm.stride(0) if enorm.dim() == 3 else 0   # eff_norm is (B, n_kv, T)
    se_h = enorm.stride(1) if enorm.dim() == 3 else 0   # n_kv (head) is dim 1
    se_t = enorm.stride(2) if enorm.dim() == 3 else 0   # T (token) is dim 2, contiguous
    _fused_score_kernel[grid](
        q, code, smin, sstep, enorm, out,
        n_q, n_kv, T, head_dim,
        q.stride(0), q.stride(1), q.stride(2),
        code.stride(0), code.stride(1), code.stride(2),
        ss_b, ss_c,
        se_b, se_t, se_h,
        out.stride(0), out.stride(1), out.stride(2),
        1.0 / math.sqrt(head_dim),
        IS_NSN=tstore["is_nsn"], IS_FP16=tstore["is_fp16"], BLK_T=BLOCK_T, BLK_D=BLK_D,
    )
    return out


# =============================================================================
# Timing + the grid (with the numerical-equivalence gate).
# =============================================================================
def _time_region(fn, warmup, iters):
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


def _bootstrap_ci(samples, n_boot=5000, alpha=0.05):
    x = torch.tensor(samples, dtype=torch.float64)
    n = x.numel()
    g = torch.Generator().manual_seed(SEED)
    meds = torch.stack([x[torch.randint(0, n, (n,), generator=g)].median() for _ in range(n_boot)])
    return float(x.median().item()), torch.quantile(meds, alpha / 2).item(), torch.quantile(meds, 1 - alpha / 2).item()


def _sample_power_w():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=5).decode().strip().splitlines()
        return float(out[0])
    except Exception:
        return None


def _gate_rel(q, ref_store, tstore, recipe, n_q, n_kv, hd):
    """Relative-max error of the kernel vs the fp32 reference (pre-reg Section 10)."""
    ref32 = _torch_decode_scores(
        q.float(), {k: (v.float() if torch.is_tensor(v) else v) for k, v in ref_store.items()},
        recipe, n_q, n_kv, hd)
    tri = triton_decode_scores(q, tstore, n_q, n_kv, hd).float()
    return ((tri - ref32).abs().max() / ref32.abs().max().clamp(min=1e-6)).item()


def run_bench_triton(out_path="results/k1c_triton_fused.json", warmup=30, iters=100):
    results = []
    payload = {"meta": {"seed": SEED, "warmup": warmup, "iters": iters, "path": "triton_fused",
                        "torch": torch.__version__, "cuda": torch.version.cuda,
                        "triton": getattr(triton, "__version__", "?"),
                        "device_name": torch.cuda.get_device_name(0),
                        "block_t": BLOCK_T, "gate_tol": GATE_TOL,
                        "note": "fused dequant-then-score, no K materialisation; int64 pointers; "
                                "int8 codes (first cut, vs-FP16 is a bandwidth lower bound); NSN = "
                                "KIVI + 1 per-token scalar on the score; every timed cell passed the gate"},
               "cells": results}
    for model in GRID_MODELS:
        for B in GRID_BATCH:
            for T in GRID_CTX:
                for recipe, bits in GRID_RECIPES:
                    rec = {"model": model["name"], "batch": B, "ctx": T, "recipe": recipe,
                           "bits": bits, "path": "triton_fused", "n_kv": model["n_kv"],
                           "head_dim": model["head_dim"], "n_q": model["n_q"]}
                    try:
                        q, ref_store, tstore, n_q, n_kv, hd = make_triton_store(model, B, T, bits, recipe, "cuda")
                        rel = _gate_rel(q, ref_store, tstore, recipe, n_q, n_kv, hd)
                        rec["gate_rel"] = rel
                        if rel < GATE_TOL:                          # time only if correct
                            torch.cuda.reset_peak_memory_stats()
                            fn = lambda: triton_decode_scores(q, tstore, n_q, n_kv, hd)
                            med, lo, hi = _bootstrap_ci(_time_region(fn, warmup, iters))
                            rec.update({"latency_ms_median": med, "latency_ms_ci": [lo, hi], "n_iters": iters,
                                        "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20,
                                        "power_w": _sample_power_w(),
                                        "bytes_per_token_K": None if recipe == "fp16"
                                        else bytes_per_token_K(bits, n_kv, hd, T, is_nsn=(recipe == "nsn")),
                                        "bytes_per_token_K_fp16": 2.0 * n_kv * hd})
                        else:
                            rec["error"] = f"gate failed rel={rel:.3e} (kernel bug; not timed)"
                    except Exception as ex:
                        rec["error"] = f"{type(ex).__name__}: {str(ex)[:160]}"
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    results.append(rec)
                    with open(out_path, "w") as f:
                        json.dump(payload, f, indent=2)
                    tag = rec.get("latency_ms_median")
                    g = rec.get("gate_rel")
                    print(f"[k1c] {model['name']:>16} B={B:>2} T={T:>5} {recipe}{bits} "
                          + (f"{tag:.4f} ms (gate {g:.1e})" if tag is not None
                             else f"SKIP {rec.get('error', '')[:64]}"))
    print(f"[k1c] wrote {out_path} ({len(results)} cells)")
    return payload


# =============================================================================
# GPU-free self-tests (foundation + int8 code roundtrip).
# =============================================================================
def _rand_k(B=2, T=257, n_kv=4, head_dim=128, seed=SEED):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, T, n_kv * head_dim, generator=g, dtype=torch.float32)


def run_selftests():
    torch.manual_seed(SEED)
    print("[k1c] GPU-free self-tests:")
    for n_kv, head_dim in ((4, 128), (8, 256)):
        kk = _rand_k(n_kv=n_kv, head_dim=head_dim)
        for bits in (4, 5):
            got = nsn_decode(nsn_store(kk, bits, n_kv, head_dim))
            ref = ref_nsn_qdq(kk, bits, n_kv, head_dim)
            rel = (got - ref).abs().max() / ref.abs().max().clamp(min=EPS)
            assert rel < 1e-6, f"NSN folding rel {rel:.2e}"
    print("  ok  NSN folding exact (decode = KIVI + 1 multiply, rel < 1e-6)")
    # int8 code storage is lossless for the dequant: dequant(int8 codes) == reference
    for recipe in ("kivi", "nsn"):
        kk = _rand_k(n_kv=8, head_dim=256)
        s = kivi_store(kk, 5) if recipe == "kivi" else nsn_store(kk, 5, 8, 256)
        code_u8 = s["code"].round().clamp(0, 255).to(torch.uint8)
        assert torch.equal(code_u8.to(torch.float32), s["code"].float()), f"{recipe} int8 code lossy"
    print("  ok  int8 code storage lossless (codes in [0, 2^bits-1] <= 31)")
    for n_kv, head_dim in ((4, 128), (8, 256)):
        delta = bytes_per_token_K(5, n_kv, head_dim, 32768, True) - bytes_per_token_K(5, n_kv, head_dim, 32768, False)
        assert abs(delta - 2.0 * n_kv) < 1e-9
    print("  ok  memory delta = norm only (NSN - KIVI == 2*n_kv B/tok)")
    print(f"[k1c] foundation matches phase_k1; HAS_TRITON={HAS_TRITON}. The fused kernel + "
          "fp32 gate run on a CUDA+triton GPU.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="run only the GPU-free self-tests")
    ap.add_argument("--out", default="results/k1c_triton_fused.json")
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--iters", type=int, default=100)
    args, _ = ap.parse_known_args()
    run_selftests()
    if not args.selftest and torch.cuda.is_available() and HAS_TRITON:
        outdir = os.path.dirname(args.out)
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        run_bench_triton(args.out, warmup=args.warmup, iters=args.iters)
        try:
            from google.colab import files
            files.download(args.out)
        except Exception:
            pass
    elif not args.selftest and not (torch.cuda.is_available() and HAS_TRITON):
        print("[k1c] need a CUDA + triton runtime for the kernel step (Colab GPU); self-tests only here.")
