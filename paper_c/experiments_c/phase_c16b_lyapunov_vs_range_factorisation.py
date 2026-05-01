# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Phase C-16b: Lyapunov-vs-range factorisation
# ============================================================
# C-16 established that half_slope_rg wins at matched mean bit
# width on every (model, b_0) cell across the state-spaces
# Mamba-1 family. C-16 also established that the slope question
# (slope-1 vs slope-1/2 without r_g) is approximately a wash, so
# the dominant structural factor is the log_2 r_g term.
#
# C-16b closes the open factorisation question that v0.7 / v0.8
# pseudo-review flagged: how much of half_slope_rg's advantage is
# Lyapunov-aware (per-group recurrence statistics rho_g, Gamma_g)
# versus simply range-aware (per-group state magnitude r_g)?
#
# Three new profiles factorise the contributions:
#
#   range_only:        extras = ceil(log_2 r_g - mean(log_2 r_g))+
#     bits go to high-magnitude groups only, no R_g signal.
#
#   shuffled_R_plus_r: shuffle R_g values across (c, s) groups
#     before adding log_2 r_g. Preserves r_g placement, randomises
#     R_g placement. Tests whether per-group R_g placement does
#     structural work or only its global distribution matters.
#
#   R_plus_shuffled_r: shuffle r_g values across (c, s) groups
#     before adding to log_2 R_g. Preserves R_g placement,
#     randomises r_g placement. Tests whether per-group r_g
#     placement does structural work.
#
# Reference profiles (carried over from C-16 for context):
#
#   uniform, full_slope, half_slope_rg
#
# Single model: Mamba-2.8B-hf (the largest state-spaces Mamba-1
# size and the C-7 / C-9 long-context anchor). b_0 in {4, 6}.
#
# Pre-registered reading rules:
#
#   If range_only matches half_slope_rg within +/- 1 percentage
#   point at matched mean: the Lyapunov-aware structure is
#   empirically secondary on this evaluation, and the title /
#   abstract should explicitly acknowledge that the dominant
#   working ingredient is range-aware bit allocation. The recipe
#   would still derive from a Lyapunov bound (the derivation in
#   section 3 stands), but the empirical lever is range placement.
#
#   If range_only collapses toward the uniform regime: the
#   Lyapunov-aware structure is doing real work and the title /
#   abstract framing is empirically supported.
#
#   If R_plus_shuffled_r approaches the uniform / full_slope
#   regime while shuffled_R_plus_r approaches half_slope_rg:
#   the per-group r_g placement carries the bulk of the structural
#   advantage. This matches the C-16 finding that the log_2 r_g
#   term, not the slope on log_2 R_g, is the dominant factor.
#
# Runtime estimate (A100 80-95 GB):
#   Mamba-2.8B: ~10 to 12 minutes (6 quant conditions x 2 b_0 +
#   1 FP16 baseline = 13 PPL evals at ctx=1024, n_windows=5)
#
# Output: /content/phase_c16b_lyapunov_vs_range_factorisation.json
# ============================================================


# =============================================================
# === CELL V1 === Setup
# =============================================================

import gc
import math
import json
import os
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
# === CELL V2 === Constants and quantiser
# =============================================================

EPS = 1e-6
B_MIN, B_MAX, RANGE_K = 2, 16, 7.0
B0_VALUES = [4, 6]
SHUFFLE_SEED = 20260501

CTX = 1024
N_WINDOWS = 5

# Single-model factorisation. The C-16 result that half_slope_rg
# wins on every cell across the 5-size family already establishes
# the structural advantage; C-16b only needs the largest size to
# isolate the Lyapunov-vs-range contributions.
MODEL_LIST = [
    ("state-spaces/mamba-2.8b-hf", "Mamba-2.8B"),
]

TOKENIZER_FALLBACKS = [
    "state-spaces/mamba-2.8b-hf",
    "EleutherAI/gpt-neox-20b",
]

PARTIAL_DIR = "/content"

# Profile order: references first, then factorisation profiles.
PROFILE_NAMES = [
    "uniform",
    "full_slope",
    "half_slope_rg",
    "range_only",
    "shuffled_R_plus_r",
    "R_plus_shuffled_r",
]


# =============================================================
# === CELL V3 === Patched slow_forward (per-token state quant)
# =============================================================
# Same patched function pattern as C-4 / C-8a / C-12 / C-16.
# Mode dispatch:
#   "off"        unpatched semantics (no quant, no calibration)
#   "calibrate"  records rho_emp / Gamma / RMS(h) per (c, s)
#   "quantise"   per-(c, s) INT-b mid-tread quantisation

_C16B = {
    "mode":           "off",
    "b_per_layer":    None,
    "r_per_layer":    None,
    "calib_accum":    None,
}


def _quantize_h_per_unit(h, b_per_unit, r_per_unit):
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


def _make_c16b_forward(mixers, layer_name_by_id):
    """Build a forward function bound to the given mixer set."""

    def _c16b_mamba_forward(mix, input_states):
        layer_name = layer_name_by_id.get(id(mix))
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
        if _C16B["mode"] == "quantise":
            b = _C16B["b_per_layer"].get(layer_name)
            r = _C16B["r_per_layer"].get(layer_name)
            if b is not None and r is not None:
                b_layer = b.to(x_pre.device)
                r_layer = r.to(x_pre.device, dtype=torch.float32)

        if _C16B["mode"] == "calibrate":
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
            if _C16B["mode"] == "calibrate":
                rms_sumsq += (h ** 2).sum(dim=0)
            y_t = (h * C_f[:, t].unsqueeze(1)).sum(dim=-1)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)

        y = y + mix.D.float().unsqueeze(0).unsqueeze(0) * x_pre.float()
        y = y * F.silu(gate.float())
        out = mix.out_proj(y.to(mix.out_proj.weight.dtype))

        if _C16B["mode"] == "calibrate":
            prev = _C16B["calib_accum"].get(layer_name)
            if prev is None:
                _C16B["calib_accum"][layer_name] = {
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

    def _patch(self, input_states, *args, **kwargs):
        return _c16b_mamba_forward(self, input_states)

    return _patch


def _force_slow_path(model):
    for _, mod in model.named_modules():
        for attr in ("use_fast_path", "_use_fast_path",
                      "use_mambapy", "_uses_cuda_kernels"):
            if hasattr(mod, attr):
                try:
                    setattr(mod, attr, False)
                except Exception:
                    pass


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


# =============================================================
# === CELL V4 === Profile builders (raw extras pre-shift)
# =============================================================
# Each builder returns a dict {layer_name: tensor of integer
# extras} BEFORE the matched-mean shift normalisation. The
# matched-mean shift is applied uniformly via binary search in
# CELL V5.
#
# Reference builders (uniform, full_slope, half_slope_rg) are
# carried over from C-16. Factorisation builders (range_only,
# shuffled_R_plus_r, R_plus_shuffled_r) are new in C-16b.

def _extras_uniform(captured):
    return {n: torch.zeros_like(d["rho_emp"]) for n, d in captured.items()}


def _extras_full_slope(captured):
    """slope 1 on log_2 R_g (the deployed simplification, scale-side)."""
    out = {}
    for n, d in captured.items():
        rho   = d["rho_emp"]
        gamma = d["gamma"].clamp(min=EPS)
        log_R = torch.log2(
            gamma.unsqueeze(0) / (1.0 - rho).clamp(min=EPS)
        )
        out[n] = torch.ceil(log_R).clamp(min=0)
    return out


def _global_log_rg_mean(captured):
    """Pooled mean of log_2 r_g across all (c, s) units."""
    rms_all = []
    for n, d in captured.items():
        rms_all.append(d["rms_h"].flatten())
    rms_cat = torch.cat(rms_all)
    return float(
        torch.log2((RANGE_K * rms_cat).clamp(min=EPS)).mean()
    )


def _extras_half_slope_rg(captured):
    """slope 1/2 + log_2 r_g - mean log_2 r_g (recommended recipe)."""
    log_rg_global_mean = _global_log_rg_mean(captured)

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
        out[n] = torch.ceil(0.5 * log_R + log_rg_centered).clamp(min=0)
    return out


def _extras_range_only(captured):
    """Range-only: bits go to high-magnitude groups, no R_g signal.

        extras_g = ceil(log_2 r_g - mean(log_2 r_g))_+
    """
    log_rg_global_mean = _global_log_rg_mean(captured)

    out = {}
    for n, d in captured.items():
        rms = d["rms_h"]
        log_rg = torch.log2((RANGE_K * rms).clamp(min=EPS))
        log_rg_centered = log_rg - log_rg_global_mean
        # Broadcast to (d_inner, d_state) shape — log_rg already has it.
        out[n] = torch.ceil(log_rg_centered).clamp(min=0)
    return out


# ---------- shuffle helpers ----------
# Shuffles operate on the FULL (c, s) flattened vector across the
# whole model, then reshape back per layer. This randomises only
# the per-group placement; the global distribution of values is
# preserved exactly.

def _flatten_per_layer(captured, key, transform=None):
    """Concatenate a per-layer field across all layers.

    Returns the concatenated 1D tensor, and a list of (layer_name,
    shape, length) tuples used for reshaping back.
    """
    parts = []
    layout = []
    for n, d in captured.items():
        v = d[key]
        if transform is not None:
            v = transform(v)
        flat = v.flatten()
        parts.append(flat)
        layout.append((n, v.shape, flat.numel()))
    return torch.cat(parts), layout


def _unflatten_per_layer(flat, layout):
    """Inverse of _flatten_per_layer."""
    out = {}
    cursor = 0
    for n, shape, length in layout:
        chunk = flat[cursor:cursor + length].view(shape)
        out[n] = chunk
        cursor += length
    return out


def _build_log_R_per_layer(captured):
    """Compute per-layer log_2 R_g = log_2(gamma / (1 - rho))."""
    out = {}
    for n, d in captured.items():
        rho   = d["rho_emp"]
        gamma = d["gamma"].clamp(min=EPS)
        log_R = torch.log2(
            gamma.unsqueeze(0) / (1.0 - rho).clamp(min=EPS)
        )
        out[n] = log_R
    return out


def _build_log_rg_centered_per_layer(captured, log_rg_global_mean):
    """Compute per-layer (log_2 r_g - global mean log_2 r_g)."""
    out = {}
    for n, d in captured.items():
        rms = d["rms_h"]
        log_rg = torch.log2((RANGE_K * rms).clamp(min=EPS))
        out[n] = log_rg - log_rg_global_mean
    return out


def _extras_shuffled_R_plus_r(captured, seed=SHUFFLE_SEED):
    """Shuffle log_2 R_g across (c, s) groups before adding log_2 r_g.

        extras_g = ceil(0.5 * log_2 R_shuffled[g] + log_2 r_g
                        - mean(log_2 r_g))_+

    Preserves r_g placement, randomises R_g placement. Tests
    whether per-group R_g placement does structural work or only
    its global distribution matters.
    """
    log_rg_global_mean = _global_log_rg_mean(captured)
    log_R_per_layer = _build_log_R_per_layer(captured)
    log_rg_centered = _build_log_rg_centered_per_layer(
        captured, log_rg_global_mean
    )

    # Flatten log_R, shuffle, reshape back.
    log_R_flat = torch.cat(
        [v.flatten() for v in log_R_per_layer.values()]
    )
    layout = [(n, v.shape, v.numel())
                for n, v in log_R_per_layer.items()]

    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(log_R_flat.numel(), generator=g)
    log_R_shuffled_flat = log_R_flat[perm]
    log_R_shuffled = _unflatten_per_layer(log_R_shuffled_flat, layout)

    out = {}
    for n in captured.keys():
        ext = 0.5 * log_R_shuffled[n] + log_rg_centered[n]
        out[n] = torch.ceil(ext).clamp(min=0)
    return out


def _extras_R_plus_shuffled_r(captured, seed=SHUFFLE_SEED):
    """Shuffle log_2 r_g across (c, s) groups before adding to log_2 R_g.

        extras_g = ceil(0.5 * log_2 R_g + log_2 r_shuffled[g]
                        - mean(log_2 r_g))_+

    Preserves R_g placement, randomises r_g placement. Tests
    whether per-group r_g placement does structural work.
    """
    log_rg_global_mean = _global_log_rg_mean(captured)
    log_R_per_layer = _build_log_R_per_layer(captured)
    log_rg_centered = _build_log_rg_centered_per_layer(
        captured, log_rg_global_mean
    )

    # Flatten log_rg_centered, shuffle, reshape back.
    log_rg_flat = torch.cat(
        [v.flatten() for v in log_rg_centered.values()]
    )
    layout = [(n, v.shape, v.numel())
                for n, v in log_rg_centered.items()]

    g = torch.Generator()
    # Use a different seed offset so the two shuffles are independent
    g.manual_seed(seed + 1)
    perm = torch.randperm(log_rg_flat.numel(), generator=g)
    log_rg_shuffled_flat = log_rg_flat[perm]
    log_rg_shuffled = _unflatten_per_layer(log_rg_shuffled_flat, layout)

    out = {}
    for n in captured.keys():
        ext = 0.5 * log_R_per_layer[n] + log_rg_shuffled[n]
        out[n] = torch.ceil(ext).clamp(min=0)
    return out


PROFILE_BUILDERS = {
    "uniform":            _extras_uniform,
    "full_slope":         _extras_full_slope,
    "half_slope_rg":      _extras_half_slope_rg,
    "range_only":         _extras_range_only,
    "shuffled_R_plus_r":  _extras_shuffled_R_plus_r,
    "R_plus_shuffled_r":  _extras_R_plus_shuffled_r,
}


# =============================================================
# === CELL V5 === Matched-mean binary-search shift (reused C-16)
# =============================================================
# Identical to C-16: binary search over additive shift such that
# the post-clamp mean bit width equals target_mean within tol.

def _build_b_per_unit_matched_mean(extras_dict, target_mean,
                                    b_min=B_MIN, b_max=B_MAX,
                                    tol=0.005, max_iter=80):
    """Binary search for an additive shift such that the post-
    clamp mean bit width equals target_mean within tol.

        b_g = clip(round(target_mean + extras_g + shift),
                   b_min, b_max).

    Returns (b_per_layer, info_dict).
    """
    def mean_b_at_shift(shift):
        ss, nn = 0.0, 0
        for n, e in extras_dict.items():
            raw = float(target_mean) + e.float() + shift
            b_int = torch.round(raw).clamp(min=b_min, max=b_max)
            ss += float(b_int.sum())
            nn += b_int.numel()
        return ss / max(nn, 1)

    s_lo, s_hi = -10.0, 10.0
    iters = 0
    for it in range(max_iter):
        iters = it + 1
        s_mid = (s_lo + s_hi) / 2
        m = mean_b_at_shift(s_mid)
        if abs(m - target_mean) < tol:
            break
        if m < target_mean:
            s_lo = s_mid
        else:
            s_hi = s_mid

    final_shift = (s_lo + s_hi) / 2

    out = {}
    for n, e in extras_dict.items():
        raw = float(target_mean) + e.float() + final_shift
        out[n] = torch.round(raw).clamp(min=b_min, max=b_max).long()

    flat = torch.cat([b.flatten().float() for b in out.values()])
    info = {
        "shift":         float(final_shift),
        "iterations":    iters,
        "achieved_mean": float(flat.mean()),
        "target_mean":   float(target_mean),
        "diff":          abs(float(flat.mean()) - float(target_mean)),
        "b_min":         b_min,
        "b_max":         b_max,
    }
    return out, info


# =============================================================
# === CELL V6 === PPL eval helper (reused C-16)
# =============================================================

def ppl_with_quant(model, ids, b_per_layer=None, r_per_layer=None,
                    ctx=CTX, n_windows=N_WINDOWS):
    """Sliding-window PPL. If b_per_layer/r_per_layer are None,
    runs in 'off' mode (no quant). Otherwise quantise with the
    provided per-layer specs."""
    stride = ctx // 2
    if b_per_layer is None or r_per_layer is None:
        _C16B["mode"]        = "off"
        _C16B["b_per_layer"] = None
        _C16B["r_per_layer"] = None
    else:
        _C16B["mode"]        = "quantise"
        _C16B["b_per_layer"] = b_per_layer
        _C16B["r_per_layer"] = r_per_layer

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
        _C16B["mode"]        = "off"
        _C16B["b_per_layer"] = None
        _C16B["r_per_layer"] = None
    return math.exp(nll_total / tok_total), windows_done


# =============================================================
# === CELL V7 === Per-model run
# =============================================================

def run_one_model(model_id, model_label, tok, test_ids):
    """Load the model, calibrate, and run all (b_0, profile)
    conditions. Returns the per-model result dict and writes a
    partial JSON checkpoint to PARTIAL_DIR."""
    print(f"\n{'=' * 80}")
    print(f"Running {model_label} ({model_id})")
    print(f"{'=' * 80}")

    mdl = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16,
        device_map="auto", trust_remote_code=True,
    )
    mdl.eval()
    print(f"  VRAM after load: "
          f"{torch.cuda.memory_allocated() / 1e9:.2f} GB")

    mixers = find_mixers(mdl)
    print(f"  Found {len(mixers)} Mamba-1 mixers")
    layer_name_by_id = {id(mix): name for name, mix in mixers}

    _MixerClass = type(mixers[0][1])
    _orig_slow_forward = _MixerClass.slow_forward
    _MixerClass.slow_forward = _make_c16b_forward(mixers, layer_name_by_id)
    _force_slow_path(mdl)

    # Calibrate
    print(f"  Calibrating on WikiText-2 train (1024 tokens)...")
    calib_text = "\n\n".join(
        [x for x in load_dataset("wikitext", "wikitext-2-raw-v1",
                                  split="train")["text"]
         if x.strip()])[:8000]
    calib_ids = tok.encode(calib_text, return_tensors="pt").to("cuda")[:, :1024]

    _C16B["mode"]        = "calibrate"
    _C16B["calib_accum"] = {}
    with torch.inference_mode():
        _ = mdl(calib_ids, use_cache=False)
    _C16B["mode"] = "off"

    captured = {}
    for name, accum in _C16B["calib_accum"].items():
        rho   = accum["rho_sum"]   / accum["rho_count"]
        gamma = accum["gamma_sum"] / accum["gamma_count"]
        rms   = (accum["rms_sumsq"] /
                  accum["rms_count"]).clamp(min=1e-12).sqrt()
        captured[name] = {
            "rho_emp": rho,
            "gamma":   gamma,
            "rms_h":   rms,
        }
    print(f"  captured rho/gamma/rms for {len(captured)} layers")

    # log_2 r_g distribution diagnostic (helps verify that the
    # shuffles preserve global statistics).
    rg_all = torch.cat([RANGE_K * d["rms_h"].flatten()
                          for d in captured.values()])
    log_rg_all = torch.log2(rg_all.clamp(min=EPS))
    log_rg_diag = {
        "log_rg_mean":   float(log_rg_all.mean()),
        "log_rg_std":    float(log_rg_all.std()),
        "log_rg_min":    float(log_rg_all.min()),
        "log_rg_max":    float(log_rg_all.max()),
        "log_rg_p10":    float(log_rg_all.quantile(0.10)),
        "log_rg_p50":    float(log_rg_all.quantile(0.50)),
        "log_rg_p90":    float(log_rg_all.quantile(0.90)),
    }
    print(f"  log_2 r_g distribution: "
          f"mean={log_rg_diag['log_rg_mean']:+.3f} "
          f"std={log_rg_diag['log_rg_std']:.3f} "
          f"p10/p50/p90={log_rg_diag['log_rg_p10']:+.2f}/"
          f"{log_rg_diag['log_rg_p50']:+.2f}/"
          f"{log_rg_diag['log_rg_p90']:+.2f}")

    r_per_layer = {n: RANGE_K * d["rms_h"] for n, d in captured.items()}

    # FP16 baseline
    print(f"\n  FP16 baseline (no quant)...")
    fp16_ppl, n_win = ppl_with_quant(mdl, test_ids,
                                       ctx=CTX, n_windows=N_WINDOWS)
    print(f"    FP16 PPL = {fp16_ppl:.4f} (n_windows={n_win})")

    # Per-(b_0, profile) sweep
    sweep = {"FP16_baseline": {"ppl": fp16_ppl, "n_windows": n_win,
                                "achieved_mean": None,
                                "shift": None,
                                "delta_pct": 0.0}}

    for b_0 in B0_VALUES:
        print(f"\n  b_0 = {b_0}")
        for prof_name in PROFILE_NAMES:
            extras_builder = PROFILE_BUILDERS[prof_name]
            extras = extras_builder(captured)
            b_per_layer, shift_info = _build_b_per_unit_matched_mean(
                extras, target_mean=b_0,
            )
            ppl, n_win = ppl_with_quant(
                mdl, test_ids, b_per_layer, r_per_layer,
                ctx=CTX, n_windows=N_WINDOWS,
            )
            delta_pct = 100.0 * (ppl - fp16_ppl) / fp16_ppl
            key = f"b{b_0}_{prof_name}"
            sweep[key] = {
                "b_0":           b_0,
                "profile":       prof_name,
                "ppl":           ppl,
                "delta_pct":     delta_pct,
                "n_windows":     n_win,
                "achieved_mean": shift_info["achieved_mean"],
                "shift":         shift_info["shift"],
                "iterations":    shift_info["iterations"],
                "diff_target":   shift_info["diff"],
            }
            print(f"    {prof_name:<20s} mean_b={shift_info['achieved_mean']:.3f} "
                  f"(shift={shift_info['shift']:+.3f}, "
                  f"iter={shift_info['iterations']}, "
                  f"diff={shift_info['diff']:.4f}) "
                  f"PPL={ppl:>8.4f} ΔPPL={delta_pct:+.3f}%")

    # Per-model calibration summary
    calib_summary = []
    for name, d in captured.items():
        calib_summary.append({
            "layer_name":   name,
            "rho_p50":      float(d["rho_emp"].flatten().quantile(0.50)),
            "gamma_p50":    float(d["gamma"].quantile(0.50)),
            "rms_h_p50":    float(d["rms_h"].quantile(0.50)),
        })

    model_result = {
        "model_id":              model_id,
        "model_label":           model_label,
        "n_mixers":              len(mixers),
        "fp16_baseline_ppl":     fp16_ppl,
        "sweep":                 sweep,
        "log_rg_distribution":   log_rg_diag,
        "calib_summary":         calib_summary,
        "shuffle_seed":          SHUFFLE_SEED,
        "date":                  datetime.utcnow().isoformat(timespec="seconds"),
    }

    # Save partial JSON checkpoint
    safe_label = model_label.replace("/", "_").replace(" ", "_")
    partial_path = os.path.join(
        PARTIAL_DIR, f"phase_c16b_partial_{safe_label}.json"
    )
    with open(partial_path, "w") as f:
        json.dump(model_result, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Saved partial -> {partial_path}")

    # Restore original forward and unload
    _MixerClass.slow_forward = _orig_slow_forward
    del mdl, mixers
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"  VRAM after unload: "
          f"{torch.cuda.memory_allocated() / 1e9:.2f} GB")

    return model_result


# =============================================================
# === CELL V8 === Main loop
# =============================================================

# Load tokenizer
tok = None
for tid in TOKENIZER_FALLBACKS:
    try:
        tok = AutoTokenizer.from_pretrained(tid, trust_remote_code=True)
        print(f"  Tokenizer:  {tid}")
        break
    except Exception:
        pass
assert tok is not None

# WikiText-2 test text
test_text = "\n\n".join(
    [x for x in load_dataset("wikitext", "wikitext-2-raw-v1",
                              split="test")["text"]
     if x.strip()])[:200_000]
test_ids = tok.encode(test_text, return_tensors="pt").to("cuda")
print(f"  Test corpus: WikiText-2 test, {test_ids.shape[1]:,} tokens")


# Sweep all models
all_results = {}
for model_id, model_label in MODEL_LIST:
    try:
        all_results[model_label] = run_one_model(
            model_id, model_label, tok, test_ids,
        )
    except Exception as e:
        print(f"\n  [ERROR] {model_label} failed: "
              f"{type(e).__name__}: {str(e)[:200]}")
        all_results[model_label] = {
            "model_id":   model_id,
            "model_label": model_label,
            "error":      f"{type(e).__name__}: {str(e)[:500]}",
        }


# =============================================================
# === CELL V9 === Verdict and final save
# =============================================================

print(f"\n{'=' * 80}")
print(f"Phase C-16b verdict: Lyapunov-vs-range factorisation")
print(f"{'=' * 80}")

# Headline table: ΔPPL by (model, profile, b_0) at matched mean
print(f"\nHeadline ΔPPL (% of FP16 baseline) at matched mean bit width:\n")

for b_0 in B0_VALUES:
    print(f"\n  b_0 = {b_0}")
    print(f"  {'profile':<20s}  {'ΔPPL':>10s}  {'achieved mean':>14s}  "
          f"{'shift':>8s}")
    print(f"  {'-' * 20}  {'-' * 10}  {'-' * 14}  {'-' * 8}")
    for _, model_label in MODEL_LIST:
        r = all_results.get(model_label, {})
        if "error" in r:
            print(f"  {model_label:<20s}  [ERROR]")
            continue
        for prof in PROFILE_NAMES:
            key = f"b{b_0}_{prof}"
            d = r.get("sweep", {}).get(key, {})
            dpct = d.get("delta_pct", float("nan"))
            mean_b = d.get("achieved_mean", float("nan"))
            shift = d.get("shift", float("nan"))
            print(f"  {prof:<20s}  {dpct:>+9.3f}%  "
                  f"{mean_b:>14.3f}  {shift:>+8.3f}")

# Factorisation reading: range_only vs half_slope_rg gap
print(f"\n\n  Factorisation reading (Mamba-2.8B):")
for _, model_label in MODEL_LIST:
    r = all_results.get(model_label, {})
    if "error" in r:
        continue
    for b_0 in B0_VALUES:
        rg = r["sweep"].get(f"b{b_0}_half_slope_rg", {}).get("delta_pct")
        ronly = r["sweep"].get(f"b{b_0}_range_only", {}).get("delta_pct")
        full = r["sweep"].get(f"b{b_0}_full_slope", {}).get("delta_pct")
        unif = r["sweep"].get(f"b{b_0}_uniform", {}).get("delta_pct")
        sR = r["sweep"].get(f"b{b_0}_shuffled_R_plus_r", {}).get("delta_pct")
        sr = r["sweep"].get(f"b{b_0}_R_plus_shuffled_r", {}).get("delta_pct")
        print(f"\n    b_0 = {b_0}:")
        print(f"      uniform           = {unif:+7.3f}%   [no signal]")
        print(f"      full_slope        = {full:+7.3f}%   [R_g, no r_g]")
        print(f"      half_slope_rg     = {rg:+7.3f}%   [R_g + r_g, reference]")
        print(f"      range_only        = {ronly:+7.3f}%   [r_g only, no R_g]")
        print(f"      shuffled_R_plus_r = {sR:+7.3f}%   [R_g randomised, r_g placed]")
        print(f"      R_plus_shuffled_r = {sr:+7.3f}%   [R_g placed, r_g randomised]")

        if ronly is not None and rg is not None:
            gap = ronly - rg
            print(f"\n      range_only vs half_slope_rg gap: {gap:+.3f}pp")
            if abs(gap) <= 1.0:
                print(f"        -> Lyapunov-aware structure is empirically "
                      f"secondary (gap within +/- 1 pp).")
            elif gap > 1.0 and unif is not None and ronly < unif:
                print(f"        -> Lyapunov-aware structure adds material "
                      f"benefit beyond range alone.")

        if sR is not None and sr is not None and rg is not None:
            print(f"\n      shuffle reading:")
            print(f"        shuffled_R_plus_r vs reference: "
                  f"{sR - rg:+.3f}pp")
            print(f"        R_plus_shuffled_r vs reference: "
                  f"{sr - rg:+.3f}pp")
            if abs(sR - rg) < abs(sr - rg):
                print(f"        -> r_g placement carries more structural "
                      f"work than R_g placement.")
            else:
                print(f"        -> R_g placement carries more structural "
                      f"work than r_g placement.")


# Final merged JSON
output = {
    "date":            datetime.utcnow().isoformat(timespec="seconds"),
    "phase":           "C-16b",
    "purpose":         ("Lyapunov-vs-range factorisation: isolate the "
                        "contribution of the per-group recurrence "
                        "statistics (rho_g, Gamma_g) versus the per-"
                        "group state magnitude (r_g) within the "
                        "half_slope_rg recipe. Adds three baselines "
                        "(range_only, shuffled_R_plus_r, "
                        "R_plus_shuffled_r) on top of three reference "
                        "profiles (uniform, full_slope, half_slope_rg)."),
    "transformers_version": _hf_transformers.__version__,
    "model_list":      [{"id": m[0], "label": m[1]} for m in MODEL_LIST],
    "protocol": {
        "ctx":              CTX,
        "n_windows":        N_WINDOWS,
        "b_0_values":       B0_VALUES,
        "b_min":            B_MIN,
        "b_max":            B_MAX,
        "range_k":          RANGE_K,
        "shuffle_seed":     SHUFFLE_SEED,
        "calib_split":      "WikiText-2 train (1024 tokens)",
        "test_split":       "WikiText-2 test",
        "profiles":         PROFILE_NAMES,
        "shift_method":     "binary search, tol=0.005, max_iter=80",
        "quant_scheme":     "per-(c, s) INT-b symmetric mid-tread, range = k * RMS(h)",
    },
    "results":         all_results,
}

out_path = "/content/phase_c16b_lyapunov_vs_range_factorisation.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False, default=str)
print(f"\nFinal merged JSON saved -> {out_path}")


# =============================================================
# === READING ===
# =============================================================
#
# Outcome interpretation:
#
#   Case A: range_only is within +/- 1 pp of half_slope_rg
#     The Lyapunov-aware R_g surface is empirically a secondary
#     factor at integer-bit granularity on this evaluation. The
#     dominant lever is range placement. v0.9 / v1.0 should
#     acknowledge this explicitly: title may shift toward "Range-
#     and Lyapunov-Aware" or similar, and the abstract should
#     note that the recipe is dominantly range-aware in practice
#     while remaining motivated by the Lyapunov derivation.
#
#   Case B: range_only is materially worse than half_slope_rg
#     (more than 1 pp gap toward uniform)
#     The Lyapunov-aware R_g surface contributes structurally
#     beyond range alone. The current title is empirically
#     supported.
#
#   Case C: shuffled_R_plus_r ~ half_slope_rg AND
#           R_plus_shuffled_r >> half_slope_rg
#     Per-group r_g placement is the dominant structural factor;
#     R_g placement matters only in aggregate (its global
#     distribution). This is the most interesting outcome
#     because it would say the Lyapunov derivation is providing
#     the right global bit-budget, not the right per-group
#     placement. Any reading along these lines should be flagged
#     in section 6.7 closing paragraph.
#
#   Case D: shuffled_R_plus_r >> half_slope_rg AND
#           R_plus_shuffled_r ~ half_slope_rg
#     Per-group R_g placement is the dominant structural factor.
#     This is the outcome that makes the original Lyapunov-aware
#     framing strongest.
#
# Whatever the outcome, this experiment closes the v0.7 / v0.8
# pseudo-review critique that section 6.7's matched-mean victory
# leaves the Lyapunov-vs-range question unfactorised.
# ============================================================
