"""phase_k1b_compile_corroboration.py

K1b: torch.compile corroboration for the K1 production-kernel benchmark.
Pre-registration: experiments/K1_production_kernel_preregistration.md (locked 23f694d).
Companion to phase_k1_production_kernel_benchmark.py (the eager torch path, which
produced results/k1_production_kernel.json). Kept as its OWN self-contained file
for traceability: one experiment, one pinned script, one result.

QUESTION. The eager torch path measured NSN-vs-KIVI at about +20 percent at scale,
the cost of NSN's per-token eff_norm multiply when the dequant runs as a separate
materialised memory pass. Does that overhead collapse once the dequant is fused?
This runs the SAME operating-point grid with `torch.compile` (inductor fuses the
elementwise dequant into a single pass) and reports the fused NSN-vs-KIVI delta.

SCOPE. Both recipes go through the same compile, so the comparison stays fair.
The einsum still materialises K, so this is NOT a vs-FP16 absolute (the bandwidth
win is the Triton step). It isolates the fused NSN-vs-KIVI delta only.

PASTE-AND-RUN (no repo / no git clone). Self-contained (torch + stdlib). Paste the
whole file into a Colab GPU cell, or run it with no flags: it self-tests, then if a
GPU is present runs the compiled grid to results/k1b_compile_corroboration.json
(save-first per cell). `--selftest` runs only the GPU-free self-tests.
"""
import json
import math
import os
import subprocess
import time

import torch

SEED = 20260510
EPS = 1e-12        # matches phase_e1 / phase_k1 (step / norm clamp)
RENORM_EPS = 1e-8  # matches phase_e1 / phase_k1 (direction re-normalisation clamp)

# Operating-point grid, identical to phase_k1 (pre-reg Section 4), reported in full.
GRID_BATCH = [1, 8, 32, 64]
GRID_CTX = [2048, 8192, 32768, 65536]
GRID_RECIPES = [("nsn", 5), ("kivi", 5), ("kivi", 4), ("nsn", 4), ("fp16", 16)]
GRID_MODELS = [
    {"name": "Gemma4-26B-A4B", "n_kv": 8, "head_dim": 256, "n_q": 32},  # hotspot hero
    {"name": "Qwen3-30B-A3B", "n_kv": 8, "head_dim": 128, "n_q": 32},   # non-Gemma (W1e)
]


# =============================================================================
# Quant/dequant + the effective_norm folding (vendored bit-identical from phase_k1).
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
    """phase_e1 quant_k_nsn, A0: the two-step re-normalised reference."""
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


def kivi_decode(store):
    return store["code"] * store["step"] + store["min"]


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


def bpv_K(bits, head_dim, is_nsn, T):
    """Appendix B effective bits-per-value: KIVI bits+32/T; NSN +16/head_dim."""
    base = bits + 32.0 / T
    if is_nsn:
        base += 16.0 / head_dim
    return base


def bytes_per_token_K(bits, n_kv, head_dim, T, is_nsn):
    return bpv_K(bits, head_dim, is_nsn, T) * (n_kv * head_dim) / 8.0


# =============================================================================
# Decode-scores path (identical to phase_k1) + the timing/CI machinery.
# =============================================================================
def _make_inputs(model, B, T, bits, recipe, device, dtype=torch.float16):
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


def _time_region(fn, warmup, iters):
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


# =============================================================================
# The torch.compile corroboration grid.
# =============================================================================
_COMPILED = {}


def _compiled_decode():
    """torch.compile the decode path once; inductor fuses the elementwise dequant
    into a single pass, so NSN's extra eff_norm multiply stops being a separate
    memory pass. Both recipes go through this same compile (fair)."""
    if "fn" not in _COMPILED:
        _COMPILED["fn"] = torch.compile(_torch_decode_scores)
    return _COMPILED["fn"]


def run_bench_compiled(out_path="results/k1b_compile_corroboration.json", warmup=30, iters=100):
    results = []
    payload = {"meta": {"seed": SEED, "warmup": warmup, "iters": iters, "path": "torch_compiled",
                        "torch": torch.__version__, "cuda": torch.version.cuda,
                        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                        "note": "torch.compile-fused dequant; isolates the fused NSN-vs-KIVI delta; "
                                "still materialises K, so not a vs-FP16 absolute (that is the Triton step)"},
               "cells": results}
    cfn = _compiled_decode()
    for model in GRID_MODELS:
        for B in GRID_BATCH:
            for T in GRID_CTX:
                for recipe, bits in GRID_RECIPES:
                    try:
                        q, store, n_q, n_kv, hd = _make_inputs(model, B, T, bits, recipe, "cuda")
                        if torch.cuda.is_available():
                            torch.cuda.reset_peak_memory_stats()
                        fn = lambda: cfn(q, store, recipe, n_q, n_kv, hd)
                        samples = _time_region(fn, warmup, iters)
                        med, lo, hi = _bootstrap_ci(samples)
                        r = {"model": model["name"], "batch": B, "ctx": T, "recipe": recipe, "bits": bits,
                             "path": "torch_compiled", "n_kv": n_kv, "head_dim": hd, "n_q": n_q,
                             "latency_ms_median": med, "latency_ms_ci": [lo, hi], "n_iters": iters,
                             "peak_mem_mb": (torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else None,
                             "power_w": _sample_power_w(),
                             "bytes_per_token_K": None if recipe == "fp16"
                             else bytes_per_token_K(bits, n_kv, hd, T, is_nsn=(recipe == "nsn")),
                             "bytes_per_token_K_fp16": 2.0 * n_kv * hd}
                    except RuntimeError as ex:                     # OOM etc.: log, do not skip silently
                        r = {"model": model["name"], "batch": B, "ctx": T, "recipe": recipe,
                             "bits": bits, "path": "torch_compiled", "error": str(ex)[:200]}
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    results.append(r)
                    with open(out_path, "w") as f:                 # save-first after every cell
                        json.dump(payload, f, indent=2)
                    tag = r.get("latency_ms_median")
                    print(f"[k1b] {model['name']:>16} B={B:>2} T={T:>5} {recipe}{bits} "
                          + (f"{tag:.4f} ms" if tag is not None else f"ERR {r.get('error', '')[:60]}"))
    print(f"[k1b] wrote {out_path} ({len(results)} cells)")
    return payload


# =============================================================================
# GPU-free self-tests (the same foundation phase_k1 verifies, re-asserted here).
# =============================================================================
def _rand_k(B=2, T=257, n_kv=4, head_dim=128, seed=SEED):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, T, n_kv * head_dim, generator=g, dtype=torch.float32)


def run_selftests():
    torch.manual_seed(SEED)
    print("[k1b] GPU-free self-tests:")
    for n_kv, head_dim in ((4, 128), (8, 256)):
        kk = _rand_k(n_kv=n_kv, head_dim=head_dim)
        for bits in (4, 5):
            got = nsn_decode(nsn_store(kk, bits, n_kv, head_dim))
            ref = ref_nsn_qdq(kk, bits, n_kv, head_dim)
            rel = (got - ref).abs().max() / ref.abs().max().clamp(min=EPS)
            assert rel < 1e-6, f"NSN folding rel {rel:.2e}"
    print("  ok  NSN folding exact (decode = KIVI + 1 multiply, rel < 1e-6)")
    for n_kv, head_dim in ((4, 128), (8, 256)):
        for T in (2048, 32768):
            delta = bytes_per_token_K(5, n_kv, head_dim, T, True) - bytes_per_token_K(5, n_kv, head_dim, T, False)
            assert abs(delta - 2.0 * n_kv) < 1e-9
    print("  ok  memory delta = norm only (NSN - KIVI == 2*n_kv B/tok == 16/head_dim bpv)")
    print("[k1b] foundation matches phase_k1; compiled grid is the GPU step.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="run only the GPU-free self-tests")
    ap.add_argument("--out", default="results/k1b_compile_corroboration.json")
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--iters", type=int, default=100)
    args, _ = ap.parse_known_args()  # ignore a notebook kernel's -f kernel.json argv
    run_selftests()
    if not args.selftest and torch.cuda.is_available():
        outdir = os.path.dirname(args.out)
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        run_bench_compiled(args.out, warmup=args.warmup, iters=args.iters)
        try:                                   # auto-download on Colab; no-op elsewhere
            from google.colab import files
            files.download(args.out)
        except Exception:
            pass
