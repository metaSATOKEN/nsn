# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Phase C-12: Half-slope vs full-slope integer allocator
# ============================================================
# Closes the bit-space / sigma-space mismatch identified by
# the v0.2 pseudo-review. The deployed integer allocator in
# Section 3.5 uses
#
#     extras_g = ceil(log_2 R_g)+
#
# (slope 1 on log_2 R_g), while Section 3.4 derives the strict
# bit-budget optimum
#
#     b_g* = b_0 + (1/2) log_2 R_g + log_2 r_g
#
# (slope 1/2 on log_2 R_g, plus a per-group r_g term). The
# Section 4.5 sigma2-scaling experiment validated the
# continuous-noise profile sigma_g proportional to
# sqrt((1-rho_g)/Gamma_g), which under sigma proportional to
# 2^{-b} corresponds to the slope-1/2 integer construction,
# NOT the slope-1 heuristic that is actually deployed in
# Section 3.5 / Section 4.6 / Sections 5 to 7.
#
# Phase C-12 fixes this by directly comparing the slope-1
# heuristic and the slope-1/2 strict optimum AS INTEGER
# allocators (i.e., quantising the SSM state), at matched
# actual mean bit width, on Mamba-2.8B-hf, WikiText-2 PPL.
# It also tests the slope-1/2 + log_2 r_g variant per the
# Section 3.4 derivation.
#
# Profiles (4 total, all at integer-bit quant):
#   - uniform              b_g = b_0 (control)
#   - full_slope           b_g = b_0 + ceil(log_2 R_g)+ + shift  (current heuristic)
#   - half_slope           b_g = b_0 + ceil((1/2) log_2 R_g)+ + shift  (strict optimum)
#   - half_slope_rg        b_g = b_0 + ceil((1/2) log_2 R_g + log_2 r_g - mean_log_rg)+ + shift
#                            (full strict optimum including r_g term, with mean-zero r_g shift)
#
# Sweeps:
#   b_0 in {4, 6}          two bit budgets matching C-4 / C-8a
#   range_k = 7.0          matching C-4 saturation point
#   n_windows = 5          matching C-4
#   ctx = 1024             matching C-4
#
# Pre-registered hypotheses:
#
#   H1 (slope question):
#     If half_slope beats full_slope at matched mean bit
#     width, the heuristic in Section 3.5 is suboptimal and
#     the v0.3 paper should adopt half_slope as the deployed
#     allocator.
#
#   H2 (r_g term question):
#     If half_slope_rg beats half_slope at matched mean bit
#     width, the r_g term dropped in Section 3.4 carries
#     measurable benefit and should be reintroduced.
#
#   H3 (matched mean):
#     The shift normalisation should produce mean bits within
#     +/- 0.1 bits of b_0 across all profiles. We log the
#     actual mean bits per profile per b_0 to make the
#     comparison auditable.
#
# Outcome decision rule (pre-registered):
#
#   Case A: full_slope wins (delta_full < delta_half by > 0.5
#           percentage points at b_0 = 4 and b_0 = 6).
#     -> Heuristic full-slope is empirically tighter than the
#        strict optimum on the integer-bit channel. The v0.3
#        paper's Section 3.5 framing stands; the slope-1
#        heuristic is the right deployed allocator.
#
#   Case B: half_slope wins (delta_half < delta_full by > 0.5
#           pp at both b_0).
#     -> Strict slope-1/2 is empirically tighter. The v0.3
#        paper should adopt half_slope as the deployed
#        recipe, and re-run the C-4 / C-8a / C-6 / C-7 / C-9
#        sweeps with half_slope. Major recalibration needed.
#
#   Case C: half_slope_rg wins beyond half_slope.
#     -> The r_g term is empirically meaningful. The v0.3
#        paper should adopt half_slope_rg as the deployed
#        recipe.
#
#   Case D: All three (full / half / half+rg) within sampling
#           noise (< 0.5 pp gap).
#     -> Slope choice and r_g term are empirically irrelevant
#        on this evaluation. Section 3.5's heuristic framing
#        is fine; the choice is taste.
#
# Runtime estimate (A100 95 GB):
#   per-token state quant per condition at ctx=1024:
#     ~5 min per (b_0, profile)
#   4 profiles x 2 b_0 = 8 conditions, plus calibration
#   Total ~45 - 60 min.
# ============================================================


# =============================================================
# === CELL V1 === Setup
# =============================================================

import gc
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime

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

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers as _hf_transformers
print(f"  transformers version: {_hf_transformers.__version__}")
print(f"  GPU memory: "
      f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# =============================================================
# === CELL V2 === Load Mamba-2.8B-hf
# =============================================================

MODEL_ID    = "state-spaces/mamba-2.8b-hf"
MODEL_LABEL = "Mamba-2.8B-hf"
TOKENIZER_FALLBACKS = [MODEL_ID, "EleutherAI/gpt-neox-20b"]

print(f"\n{'=' * 80}")
print(f"Loading {MODEL_LABEL}")
print(f"{'=' * 80}")

tok = None
for tid in TOKENIZER_FALLBACKS:
    try:
        tok = AutoTokenizer.from_pretrained(tid, trust_remote_code=True)
        print(f"  Tokenizer:  {tid}")
        break
    except Exception:
        pass
assert tok is not None

mdl = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16,
    device_map="auto", trust_remote_code=True,
)
mdl.eval()
print(f"  VRAM after load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")


def find_mixers(model):
    found = []
    for name, mod in model.named_modules():
        if (hasattr(mod, "A_log") and isinstance(
                getattr(mod, "A_log", None), torch.nn.Parameter)
            and hasattr(mod, "dt_proj") and isinstance(
                getattr(mod, "dt_proj", None), torch.nn.Linear)
            and hasattr(mod, "x_proj") and isinstance(
                getattr(mod, "x_proj", None), torch.nn.Linear)):
            found.append((name, mod))
    return found

mixers = find_mixers(mdl)
print(f"  Found {len(mixers)} Mamba-1 mixers")
LAYER_NAME_BY_ID = {id(mix): name for name, mix in mixers}


# =============================================================
# === CELL V3 === Patched slow_forward (integer-bit state quant)
# =============================================================
# Same patched function as C-4 / C-8a: per-(c, s) INT-b
# symmetric mid-tread quantiser, range r_g = k * RMS(h_g),
# bit allocation b_g per layer. The point of C-12 is to vary
# WHICH b_g profile is used.

_C12 = {
    "mode":           "off",       # "off" | "calibrate" | "quantise"
    "b_per_layer":    None,
    "r_per_layer":    None,
    "calib_accum":    None,
}

EPS = 1e-6


def _quantize_h_per_unit(h, b_per_unit, r_per_unit):
    """INT-b symmetric quantiser with zero level."""
    b_f      = b_per_unit.float().clamp(min=2.0, max=16.0)
    n_neg    = 2.0 ** (b_f - 1.0)
    n_pos    = n_neg - 1.0
    delta    = (r_per_unit / n_neg).clamp(min=EPS)
    delta_b  = delta.unsqueeze(0)
    n_neg_b  = n_neg.unsqueeze(0)
    n_pos_b  = n_pos.unsqueeze(0)
    q_int    = torch.round(h / delta_b)
    q_int    = torch.maximum(q_int, -n_neg_b)
    q_int    = torch.minimum(q_int,  n_pos_b)
    return q_int * delta_b


def _c12_mamba_forward(mix, input_states):
    layer_name = LAYER_NAME_BY_ID.get(id(mix))
    B, T, _ = input_states.shape
    d_inner = mix.intermediate_size
    d_state = mix.ssm_state_size
    dt_rank = mix.time_step_rank

    proj = mix.in_proj(input_states)
    x_pre, gate = proj.chunk(2, dim=-1)

    x_pre_c = x_pre.transpose(1, 2).contiguous()
    x_pre_c = mix.conv1d(x_pre_c)[:, :, :T]
    x_pre = F.silu(x_pre_c).transpose(1, 2)

    xbc = mix.x_proj(x_pre)
    dt_raw, B_proj, C_proj = xbc.split(
        [dt_rank, d_state, d_state], dim=-1
    )

    dt = F.softplus(mix.dt_proj(dt_raw)).float()
    A_neg = -torch.exp(mix.A_log.float())

    A_bar = torch.exp(
        dt[..., None] * A_neg.unsqueeze(0).unsqueeze(0)
    )
    B_bar = (
        dt[..., None] * B_proj.float()[:, :, None, :]
    )
    deltaB_x = B_bar * x_pre.float()[..., None]
    C_f = C_proj.float()

    b_layer = None
    r_layer = None
    if _C12["mode"] == "quantise":
        b = _C12["b_per_layer"].get(layer_name)
        r = _C12["r_per_layer"].get(layer_name)
        if b is not None and r is not None:
            b_layer = b.to(x_pre.device)
            r_layer = r.to(x_pre.device, dtype=torch.float32)

    if _C12["mode"] == "calibrate":
        rho_sum     = A_bar.sum(dim=(0, 1))
        rho_count   = B * T
        gamma_sum   = (C_f ** 2).sum(dim=(0, 1))
        gamma_count = B * T
        rms_sumsq   = torch.zeros((d_inner, d_state),
                                    dtype=torch.float32,
                                    device=x_pre.device)

    h = torch.zeros((B, d_inner, d_state),
                     dtype=torch.float32, device=x_pre.device)
    ys = []
    for t in range(T):
        h = A_bar[:, t] * h + deltaB_x[:, t]
        if b_layer is not None and r_layer is not None:
            h = _quantize_h_per_unit(h, b_layer, r_layer)
        if _C12["mode"] == "calibrate":
            rms_sumsq += (h ** 2).sum(dim=0)
        y_t = (h * C_f[:, t].unsqueeze(1)).sum(dim=-1)
        ys.append(y_t)
    y = torch.stack(ys, dim=1)

    y = y + mix.D.float().unsqueeze(0).unsqueeze(0) * x_pre.float()
    y = y * F.silu(gate.float())
    out = mix.out_proj(y.to(mix.out_proj.weight.dtype))

    if _C12["mode"] == "calibrate":
        prev = _C12["calib_accum"].get(layer_name)
        if prev is None:
            _C12["calib_accum"][layer_name] = {
                "rho_sum":     rho_sum.detach().cpu(),
                "rho_count":   rho_count,
                "gamma_sum":   gamma_sum.detach().cpu(),
                "gamma_count": gamma_count,
                "rms_sumsq":   rms_sumsq.detach().cpu(),
                "rms_count":   B * T,
            }
        else:
            prev["rho_sum"]     += rho_sum.detach().cpu()
            prev["rho_count"]   += rho_count
            prev["gamma_sum"]   += gamma_sum.detach().cpu()
            prev["gamma_count"] += gamma_count
            prev["rms_sumsq"]   += rms_sumsq.detach().cpu()
            prev["rms_count"]   += B * T

    return out


def _c12_slow_forward_patch(self, input_states, *args, **kwargs):
    return _c12_mamba_forward(self, input_states)


_MixerClass = type(mixers[0][1])
_orig_slow_forward = _MixerClass.slow_forward
_MixerClass.slow_forward = _c12_slow_forward_patch
print(f"  Monkey-patched {_MixerClass.__module__}."
      f"{_MixerClass.__name__}.slow_forward")


def _force_slow_path(model):
    for _, mod in model.named_modules():
        for attr in ("use_fast_path", "_use_fast_path",
                      "use_mambapy", "_uses_cuda_kernels"):
            if hasattr(mod, attr):
                try:
                    setattr(mod, attr, False)
                except Exception:
                    pass

_force_slow_path(mdl)


# =============================================================
# === CELL V4 === Calibration on WikiText-2 train
# =============================================================

print(f"\n{'-' * 80}")
print(f"Calibrating ρ_emp + Γ_s + RMS(h)")
print(f"{'-' * 80}")

calib_text = "\n\n".join(
    [x for x in load_dataset("wikitext", "wikitext-2-raw-v1",
                              split="train")["text"]
     if x.strip()])[:8000]
calib_ids = tok.encode(calib_text, return_tensors="pt").to("cuda")[:, :1024]

_C12["mode"]        = "calibrate"
_C12["calib_accum"] = {}
with torch.inference_mode():
    _ = mdl(calib_ids, use_cache=False)
_C12["mode"] = "off"

captured = {}
for name, accum in _C12["calib_accum"].items():
    rho   = accum["rho_sum"]   / accum["rho_count"]
    gamma = accum["gamma_sum"] / accum["gamma_count"]
    rms   = (accum["rms_sumsq"] /
              accum["rms_count"]).clamp(min=1e-12).sqrt()
    captured[name] = {
        "rho_emp": rho,    # [d_inner, d_state]
        "gamma":   gamma,  # [d_state]
        "rms_h":   rms,    # [d_inner, d_state]
    }
print(f"  captured rho/gamma/rms for {len(captured)} layers")


# =============================================================
# === CELL V5 === Bit profile builders (4 profiles)
# =============================================================
# All profiles are integer-bit allocations on per-(c, s)
# granularity, with shift normalisation to match nominal b_0
# and clamp to [B_MIN, 16].
#
# uniform:      extras_g = 0
# full_slope:   extras_g = ceil(log_2 R_g)+
# half_slope:   extras_g = ceil((1/2) log_2 R_g)+
# half_slope_rg:extras_g = ceil((1/2) log_2 R_g + (log_2 r_g - mean log_2 r_g))+

B_MIN, RANGE_K = 2, 7.0


def _extras_uniform(captured):
    return {n: torch.zeros_like(d["rho_emp"]) for n, d in captured.items()}


def _extras_full_slope(captured):
    """slope 1 on log_2 R_g (the deployed heuristic)."""
    out = {}
    for n, d in captured.items():
        rho   = d["rho_emp"]
        gamma = d["gamma"].clamp(min=EPS)
        log_R = torch.log2(
            gamma.unsqueeze(0) / (1.0 - rho).clamp(min=EPS)
        )
        out[n] = torch.ceil(log_R).clamp(min=0)
    return out


def _extras_half_slope(captured):
    """slope 1/2 on log_2 R_g (the strict bit-budget optimum)."""
    out = {}
    for n, d in captured.items():
        rho   = d["rho_emp"]
        gamma = d["gamma"].clamp(min=EPS)
        log_R = torch.log2(
            gamma.unsqueeze(0) / (1.0 - rho).clamp(min=EPS)
        )
        out[n] = torch.ceil(0.5 * log_R).clamp(min=0)
    return out


def _extras_half_slope_rg(captured):
    """slope 1/2 + log_2(r_g) - mean log_2(r_g).
    This includes the r_g term from Section 3.4 derivation,
    centred at zero so it doesn't shift the global mean.
    """
    # First compute global mean of log_2 r_g
    rms_all = []
    for n, d in captured.items():
        rms_all.append(d["rms_h"].flatten())
    rms_cat = torch.cat(rms_all)
    log_rg_global_mean = float(torch.log2((RANGE_K * rms_cat).clamp(min=EPS)).mean())

    out = {}
    for n, d in captured.items():
        rho   = d["rho_emp"]
        gamma = d["gamma"].clamp(min=EPS)
        rms   = d["rms_h"]
        log_R = torch.log2(
            gamma.unsqueeze(0) / (1.0 - rho).clamp(min=EPS)
        )
        log_rg = torch.log2((RANGE_K * rms).clamp(min=EPS))
        log_rg_centered = log_rg - log_rg_global_mean
        # extras = ceil((1/2) log R + log r_g - mean log r_g)+
        out[n] = torch.ceil(0.5 * log_R + log_rg_centered).clamp(min=0)
    return out


def _build_b_per_unit(extras_dict, b_0):
    """Apply shift normalisation to match nominal b_0,
    clamp to [B_MIN, 16]."""
    raw_b = {n: b_0 + e for n, e in extras_dict.items()}
    sum_b = sum(float(b.sum()) for b in raw_b.values())
    n_total = sum(b.numel() for b in raw_b.values())
    mean_b = sum_b / n_total
    shift = b_0 - mean_b
    out = {}
    for n, b in raw_b.items():
        b_shift = b + shift
        b_int = torch.round(b_shift).long().clamp(min=B_MIN, max=16)
        out[n] = b_int
    return out


def _actual_mean_bits(b_per_layer):
    flat = torch.cat([b.flatten() for b in b_per_layer.values()])
    return float(flat.float().mean())


# Per-layer range (shared across profiles)
r_per_layer = {n: RANGE_K * d["rms_h"] for n, d in captured.items()}


# =============================================================
# === CELL V6 === Build all profiles for both b_0 values
# =============================================================

PROFILE_BUILDERS = [
    ("uniform",        _extras_uniform),
    ("full_slope",     _extras_full_slope),
    ("half_slope",     _extras_half_slope),
    ("half_slope_rg",  _extras_half_slope_rg),
]

B0_VALUES = [4, 6]

profiles_by_b0 = {}
for b_0 in B0_VALUES:
    profiles_by_b0[b_0] = {}
    print(f"\n  b_0 = {b_0}: profile mean(b_g) values")
    for label, builder in PROFILE_BUILDERS:
        extras = builder(captured)
        b_per_layer = _build_b_per_unit(extras, b_0)
        mean_b = _actual_mean_bits(b_per_layer)
        profiles_by_b0[b_0][label] = {
            "b_per_layer": b_per_layer,
            "mean_b":      mean_b,
        }
        print(f"    {label:<18s} mean(b_g) = {mean_b:.3f}")


# =============================================================
# === CELL V7 === Sweep
# =============================================================

CTX = 1024
N_WINDOWS = 5

# WikiText-2 test
test_text = "\n\n".join(
    [x for x in load_dataset("wikitext", "wikitext-2-raw-v1",
                              split="test")["text"]
     if x.strip()])[:200_000]
test_ids = tok.encode(test_text, return_tensors="pt").to("cuda")
print(f"\n  Test corpus: WikiText-2 test, {test_ids.shape[1]:,} tokens")


def ppl_with_quant(model, ids, b_per_layer, r_per_layer,
                    ctx=1024, n_windows=5):
    """Compute sliding-window PPL under integer-bit quantisation."""
    stride = ctx // 2
    _C12["mode"]         = "quantise"
    _C12["b_per_layer"]  = b_per_layer
    _C12["r_per_layer"]  = r_per_layer
    nll_total = 0.0
    tok_total = 0
    windows_done = 0
    try:
        with torch.inference_mode():
            for begin in range(0, ids.shape[1] - ctx, stride):
                if windows_done >= n_windows:
                    break
                window = ids[:, begin:begin + ctx]
                out = model(window, use_cache=False)
                log_probs = F.log_softmax(out.logits[0, :-1].float(),
                                            dim=-1)
                targets = window[0, 1:]
                nll = -log_probs.gather(-1,
                                          targets.unsqueeze(-1)).squeeze(-1)
                nll_total += nll.sum().item()
                tok_total += targets.numel()
                windows_done += 1
                del out, log_probs, targets, nll
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        _C12["mode"]         = "off"
        _C12["b_per_layer"]  = None
        _C12["r_per_layer"]  = None
    return math.exp(nll_total / tok_total), windows_done


# Baseline (no quant) — for ΔPPL reference
print(f"\n{'-' * 80}")
print(f"FP16 baseline (no quantisation)")
print(f"{'-' * 80}")

_C12["mode"] = "off"
fp16_ppl_total, n_win = 0.0, 0
nll_total = 0.0
tok_total = 0
windows_done = 0
stride = CTX // 2
with torch.inference_mode():
    for begin in range(0, test_ids.shape[1] - CTX, stride):
        if windows_done >= N_WINDOWS:
            break
        window = test_ids[:, begin:begin + CTX]
        out = mdl(window, use_cache=False)
        log_probs = F.log_softmax(out.logits[0, :-1].float(), dim=-1)
        targets = window[0, 1:]
        nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        nll_total += nll.sum().item()
        tok_total += targets.numel()
        windows_done += 1
        del out, log_probs, targets, nll
        gc.collect()
        torch.cuda.empty_cache()
fp16_ppl = math.exp(nll_total / tok_total)
print(f"  FP16 PPL = {fp16_ppl:.4f} (n_windows={windows_done})")


# Sweep
print(f"\n{'-' * 80}")
print(f"Phase C-12 integer-bit slope sweep")
print(f"  4 profiles x 2 b_0 = 8 conditions, n_windows={N_WINDOWS}")
print(f"{'-' * 80}")

sweep_results = {}

for b_0 in B0_VALUES:
    print(f"\n  b_0 = {b_0}")
    sweep_results[b_0] = {
        "FP16_baseline": {
            "ppl": fp16_ppl, "mean_b": float("nan"), "delta_pct": 0.0,
        }
    }
    for label, builder in PROFILE_BUILDERS:
        info = profiles_by_b0[b_0][label]
        b_per_layer = info["b_per_layer"]
        mean_b = info["mean_b"]
        ppl, n_win = ppl_with_quant(
            mdl, test_ids, b_per_layer, r_per_layer,
            ctx=CTX, n_windows=N_WINDOWS,
        )
        delta_pct = 100.0 * (ppl - fp16_ppl) / fp16_ppl
        sweep_results[b_0][label] = {
            "ppl":        ppl,
            "mean_b":     mean_b,
            "delta_pct":  delta_pct,
            "n_windows":  n_win,
        }
        print(f"    {label:<18s} mean_b={mean_b:.3f}  "
              f"ppl={ppl:>8.4f}  delta={delta_pct:+8.3f} %")


# =============================================================
# === CELL V8 === Verdict
# =============================================================

print(f"\n{'=' * 80}")
print(f"Phase C-12 verdict (slope question)")
print(f"{'=' * 80}")
print(f"\n  FP16 baseline PPL: {fp16_ppl:.4f}")
print(f"\n  ΔPPL (% of FP16) by profile and b_0:")
print(f"\n  {'Profile':<18s}  {'b_0=4':>10s}  {'b_0=6':>10s}")
print(f"  {'─' * 18}  {'─' * 10}  {'─' * 10}")
for label, _ in PROFILE_BUILDERS:
    d4 = sweep_results[4][label]["delta_pct"]
    d6 = sweep_results[6][label]["delta_pct"]
    mb4 = sweep_results[4][label]["mean_b"]
    mb6 = sweep_results[6][label]["mean_b"]
    print(f"  {label:<18s}  "
          f"{d4:+7.3f} % ({mb4:.2f}b)  "
          f"{d6:+7.3f} % ({mb6:.2f}b)")

# H1: full vs half slope at matched mean_b
print(f"\n  H1: full_slope vs half_slope")
for b_0 in B0_VALUES:
    full_d = sweep_results[b_0]["full_slope"]["delta_pct"]
    half_d = sweep_results[b_0]["half_slope"]["delta_pct"]
    full_mb = sweep_results[b_0]["full_slope"]["mean_b"]
    half_mb = sweep_results[b_0]["half_slope"]["mean_b"]
    diff = full_d - half_d
    winner = ("full_slope" if full_d < half_d
              else "half_slope" if half_d < full_d else "tie")
    matched = "✓" if abs(full_mb - half_mb) < 0.1 else "≠"
    print(f"    b_0={b_0}: full={full_d:+.3f}% (mean_b={full_mb:.2f})  "
          f"half={half_d:+.3f}% (mean_b={half_mb:.2f})  "
          f"diff={diff:+.3f}pp  winner={winner}  matched={matched}")

# H2: half_slope vs half_slope_rg
print(f"\n  H2: half_slope vs half_slope_rg")
for b_0 in B0_VALUES:
    half_d = sweep_results[b_0]["half_slope"]["delta_pct"]
    halfrg_d = sweep_results[b_0]["half_slope_rg"]["delta_pct"]
    half_mb = sweep_results[b_0]["half_slope"]["mean_b"]
    halfrg_mb = sweep_results[b_0]["half_slope_rg"]["mean_b"]
    diff = half_d - halfrg_d
    winner = ("half_slope" if half_d < halfrg_d
              else "half_slope_rg" if halfrg_d < half_d else "tie")
    matched = "✓" if abs(half_mb - halfrg_mb) < 0.1 else "≠"
    print(f"    b_0={b_0}: half={half_d:+.3f}% (mean_b={half_mb:.2f})  "
          f"half_rg={halfrg_d:+.3f}% (mean_b={halfrg_mb:.2f})  "
          f"diff={diff:+.3f}pp  winner={winner}  matched={matched}")

# Verdict class (Case A/B/C/D)
all_threshold = 0.5  # percentage points
case = "D"  # default: indistinguishable
for b_0 in B0_VALUES:
    full_d = sweep_results[b_0]["full_slope"]["delta_pct"]
    half_d = sweep_results[b_0]["half_slope"]["delta_pct"]
    halfrg_d = sweep_results[b_0]["half_slope_rg"]["delta_pct"]
    if full_d + all_threshold < half_d:
        case = "A"  # full wins
    elif half_d + all_threshold < full_d:
        case = "B"  # half wins
    if halfrg_d + all_threshold < half_d:
        case = "C"  # half_rg wins beyond half

VERDICTS = {
    "A": ("FULL_SLOPE_WINS",
          "The deployed heuristic full-slope allocator empirically beats the "
          "strict slope-1/2 optimum on the integer-bit channel by more than "
          "0.5 percentage points at matched mean bit width. The v0.3 paper's "
          "Section 3.5 framing stands; the slope-1 heuristic is empirically "
          "tighter than the strict slope-1/2 on this evaluation. The "
          "tail-risk-heuristic justification in Section 3.5 is empirically "
          "supported."),
    "B": ("HALF_SLOPE_WINS",
          "The strict slope-1/2 allocator empirically beats the deployed "
          "heuristic full-slope by more than 0.5 percentage points at "
          "matched mean bit width. The v0.3 paper should adopt half_slope "
          "as the deployed allocator and re-run the C-4 / C-8a / C-6 / "
          "C-7 / C-9 sweeps. Section 3.5's heuristic framing should be "
          "replaced with the strict optimum."),
    "C": ("HALF_SLOPE_RG_WINS",
          "The strict slope-1/2 allocator with the r_g term empirically "
          "beats half_slope (without r_g) by more than 0.5 percentage "
          "points. The r_g term dropped in Section 3.4 carries measurable "
          "benefit and should be reintroduced in v0.3."),
    "D": ("INDISTINGUISHABLE",
          "All three of full_slope, half_slope, half_slope_rg are within "
          "0.5 percentage points of each other at both b_0 values. The "
          "slope choice and r_g term are empirically irrelevant on this "
          "evaluation; Section 3.5's heuristic framing is fine, and the "
          "choice between slope variants is taste."),
}
verdict_class, verdict_text = VERDICTS[case]
print(f"\n  Verdict class: {verdict_class}")
print(f"  Verdict: {verdict_text}")


# =============================================================
# === CELL V9 === Save
# =============================================================

calib_summary = []
for name, d in captured.items():
    calib_summary.append({
        "layer_name":   name,
        "rho_p50":      float(d["rho_emp"].flatten().quantile(0.50)),
        "gamma_p50":    float(d["gamma"].quantile(0.50)),
        "rms_h_p50":    float(d["rms_h"].quantile(0.50)),
    })

# Per-(b_0, profile) bit-tier histogram for auditing
bit_histograms = {}
for b_0 in B0_VALUES:
    bit_histograms[str(b_0)] = {}
    for label, _ in PROFILE_BUILDERS:
        b_pl = profiles_by_b0[b_0][label]["b_per_layer"]
        flat = torch.cat([b.flatten() for b in b_pl.values()])
        hist = torch.bincount(flat, minlength=17)
        bit_histograms[str(b_0)][label] = {
            str(b): int(c) for b, c in enumerate(hist.tolist()) if c > 0
        }

output = {
    "date":            datetime.utcnow().isoformat(timespec="seconds"),
    "phase":           "C-12",
    "purpose":         ("Direct integer-bit comparison of the deployed "
                        "slope-1 heuristic full_slope allocator against "
                        "the strict slope-1/2 optimum (and its r_g-corrected "
                        "variant), at matched actual mean bit width on "
                        "Mamba-2.8B-hf, WikiText-2 PPL. Closes the bit-space "
                        "/ sigma-space narrative gap identified by the v0.2 "
                        "pseudo-review on Section 3.5 / Section 4.5."),
    "transformers_version": _hf_transformers.__version__,
    "model":           MODEL_ID,
    "label":           MODEL_LABEL,
    "protocol": {
        "ctx":           CTX,
        "n_windows":     N_WINDOWS,
        "b_0_values":    B0_VALUES,
        "b_min":         B_MIN,
        "range_k":       RANGE_K,
        "calib_split":   "WikiText-2 train (1024 tokens)",
        "test_split":    "WikiText-2 test",
        "profiles":      [p[0] for p in PROFILE_BUILDERS],
        "quant_scheme":  "per-(c, s) INT-b symmetric mid-tread, range = k * RMS(h)",
    },
    "fp16_baseline_ppl": fp16_ppl,
    "sweep_results":   {str(k): v for k, v in sweep_results.items()},
    "bit_histograms":  bit_histograms,
    "rho_gamma_rms_summary": calib_summary,
    "verdict_class":   verdict_class,
    "verdict":         verdict_text,
    "notes": (
        "Each profile is built with shift normalisation to match the "
        "nominal b_0; mean_b is the actual measured mean after shift + "
        "clamp. Profiles are matched within ~0.1 bits when shift "
        "normalisation works correctly; mean_b is reported per (b_0, "
        "profile) so that reviewers can verify matched comparison."
    ),
}

out_path = "/content/phase_c12_strict_optimum_full.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False, default=str)
print(f"\nSaved -> {out_path}")


# =============================================================
# === READING ===
# =============================================================
#
# Outcome cases (pre-registered):
#
#   FULL_SLOPE_WINS:
#     Heuristic deployed allocator is empirically tighter on
#     the integer channel. Section 3.5's tail-risk justification
#     is supported. Stay with the current recipe.
#
#   HALF_SLOPE_WINS:
#     Strict bit-budget optimum is tighter on the integer channel.
#     Section 3.5 must adopt half_slope. Major rebuild of C-4 /
#     C-6 / C-7 / C-8 / C-9 with the new allocator.
#
#   HALF_SLOPE_RG_WINS:
#     r_g term carries measurable benefit. Adopt half_slope_rg
#     and rebuild.
#
#   INDISTINGUISHABLE:
#     Slope choice is empirically irrelevant; current heuristic
#     framing is fine and reviewer-defensible.
#
# Mean-bit matching:
#   The shift normalisation matches all profiles to the nominal
#   b_0 within ~0.1 bits. If a profile's mean_b is more than 0.1
#   from b_0, the comparison is not apples-to-apples and the
#   verdict needs a corresponding caveat.
# ============================================================
