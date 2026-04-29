# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Phase C-5b: Chunk-boundary state quantisation in Zamba2
# ============================================================
# Diagnostic stage. Single question: does the Phase C-4 bit
# allocator (`full` beats `uniform` / `proxy_C2`) survive when
# embedded in a hybrid Mamba2 + Attention model (Zamba2-1.2B),
# with the attention side held at FP16?
#
# Implementation note (Y-route after the per-token-loop pivot):
# Mamba2 SSD's torch_forward computes states with a chunked
# parallel algorithm. The first attempt at C-5b replaced SSD
# with an explicit per-token recurrence; this drifted by ~3 %
# from the original SSD on FP16 PPL, casting doubt on whether
# the quantisation results were measuring quantisation effects
# or a math mismatch in the loop. We pivoted to **chunk-
# boundary state quantisation**: keep the original SSD math
# bit-for-bit, but inject the quantiser on the chunk-boundary
# states that SSD already computes (`states[:, c]` = state at
# the start of chunk c). This corresponds to a deployment
# regime in which the SSM state is cached / packed every
# chunk_size = 256 tokens and is restored at chunk start. The
# noise is therefore "read out" through Y_off (the off-
# diagonal-block contribution), preserving the linear-response
# structure that Paper C's bit allocator is derived for.
#
# Conditions:
#   1. FP16 (mode = off, calls original torch_forward)
#   2. uniform        (mode = quantise)
#   3. proxy_C2       (mode = quantise)
#   4. proxy_correct  (mode = quantise)
#   5. full           (mode = quantise)
#   6. anti_full      (mode = quantise)
#
# Because mode = "off" calls the unmodified `torch_forward`,
# there is no SSD-vs-loop sanity gap — the FP16 baseline is
# bit-for-bit the original SSD path.
#
# Pre-registered hypothesis:
#   `full` beats `uniform`, `proxy_C2`, `proxy_correct`, and
#   `anti_full` on ΔPPL at b_0 = 6 inside the Zamba2-1.2B
#   hybrid context. If this fails, the bit allocator's per-
#   (c, s) Mamba1 validation does not transfer directly to the
#   per-(h, s) Mamba2 chunk-boundary setting, and C-5c
#   composition is blocked until the discrepancy is understood.
#
# Granularity adaptation (Mamba2):
#   - ρ is per head (A_log shape [n_heads], dt per head)
#   - Γ is per (head, state-dim) — extracted from C
#   - RMS(state) is per (head, head_dim, state-dim) measured on
#     the chunk-boundary states
#   - Bit allocation b_g is per (h, s) and broadcast over
#     head_dim during quantisation
#   - Range r_g = k · RMS_state(h, d, s) is per (h, d, s)
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


# =============================================================
# === CELL V2 === Load Zamba2-1.2B (Strategy B from C-5a)
# =============================================================

MODEL_ID    = "Zyphra/Zamba2-1.2B"
MODEL_LABEL = "Zamba2-1.2B"

print(f"\n{'=' * 80}")
print(f"Loading {MODEL_LABEL}  ({MODEL_ID})")
print(f"{'=' * 80}")

tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
mdl = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    ignore_mismatched_sizes=True,
    low_cpu_mem_usage=True,
)
mdl.eval()
print(f"  VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")


def find_mixers(model):
    found = []
    for name, mod in model.named_modules():
        A_log = getattr(mod, "A_log", None)
        if isinstance(A_log, torch.nn.Parameter):
            found.append((name, mod))
    return found

mixers = find_mixers(mdl)
print(f"  Found {len(mixers)} Zamba2MambaMixer layers")
LAYER_NAME_BY_ID = {id(mix): name for name, mix in mixers}
mixer_class = type(mixers[0][1])
print(f"  Mixer class: {mixer_class.__module__}.{mixer_class.__name__}")

# Pull the helpers used by SSD from the model module so we can
# replicate the math exactly.
from transformers.models.zamba2.modeling_zamba2 import (
    pad_tensor_by_size, reshape_into_chunks, segment_sum,
)


# =============================================================
# === CELL V3 === Patched torch_forward
#                 (SSD math + chunk-boundary quantisation)
# =============================================================

_C5B = {
    "mode":          "off",       # "off" | "calibrate" | "quantise"
    "b_per_layer":   None,        # dict {name: tensor [n_heads, d_state] long}
    "r_per_layer":   None,        # dict {name: tensor [n_heads, head_dim, d_state] float}
    "calib_accum":   None,
}

EPS = 1e-6


def _quantize_h_per_unit_mamba2(states, b_per_hs, r_per_hds):
    """Per-(h, d, s) INT-b symmetric mid-tread quantiser with
    zero level (matches Phase C-4). Bit count b is shared over
    head_dim within each (h, s) pair; range r is per (h, d, s).

    states:      [..., n_heads, head_dim, d_state] float
                  (any leading shape; typically [B, n_chunks, ...])
    b_per_hs:    [n_heads, d_state]                long
    r_per_hds:   [n_heads, head_dim, d_state]      float32, > 0
    """
    nh, hd, ds = r_per_hds.shape
    b_expanded = b_per_hs.unsqueeze(1).expand(nh, hd, ds).float().clamp(
        min=2.0, max=16.0
    )
    n_neg = 2.0 ** (b_expanded - 1.0)
    n_pos = n_neg - 1.0
    delta = (r_per_hds / n_neg).clamp(min=EPS)

    # Broadcast over leading dims of states
    while delta.dim() < states.dim():
        delta = delta.unsqueeze(0)
    while n_neg.dim() < states.dim():
        n_neg = n_neg.unsqueeze(0)
    while n_pos.dim() < states.dim():
        n_pos = n_pos.unsqueeze(0)

    q_int = torch.round(states / delta)
    q_int = torch.maximum(q_int, -n_neg)
    q_int = torch.minimum(q_int,  n_pos)
    return q_int * delta


# Save original BEFORE patching
_orig_torch_forward = mixer_class.torch_forward


def _c5b_torch_forward(self, input_states, cache_params=None,
                        attention_mask=None):
    """SSD math (bit-for-bit copy of the original torch_forward)
    with optional calibration accumulation and chunk-boundary
    state quantisation. mode = 'off' shortcuts to the original
    function, guaranteeing FP16 baseline equality."""

    if _C5B["mode"] == "off":
        return _orig_torch_forward(self, input_states,
                                      cache_params, attention_mask)

    layer_name = LAYER_NAME_BY_ID.get(id(self))
    batch_size, seq_len, _ = input_states.shape
    dtype = input_states.dtype

    # ── in_proj split ────────────────────────────────────
    if cache_params is not None and cache_params.has_previous_state(self.layer_idx):
        projected_states = self.in_proj(input_states)
    else:
        if attention_mask is not None:
            input_states = (input_states * attention_mask[:, :, None]).to(dtype)
        projected_states = self.in_proj(input_states)
    d_mlp = (projected_states.shape[-1] - 2 * self.intermediate_size
             - 2 * self.n_groups * self.ssm_state_size
             - self.num_heads) // 2
    _, _, gate, hidden_states, dt = projected_states.split(
        [d_mlp, d_mlp, self.intermediate_size,
         self.conv_dim, self.num_heads],
        dim=-1,
    )
    hidden_states = hidden_states.transpose(1, 2)

    # ── conv1d ──────────────────────────────────────────
    if cache_params is not None:
        # not exercised in our PPL eval (cache_params=None)
        conv_state = nn.functional.pad(
            hidden_states,
            (self.conv_kernel_size - hidden_states.shape[-1], 0),
        )
        conv_state = cache_params.update_conv_state(
            conv_state, self.layer_idx
        )
    hidden_states = self.act(
        self.conv1d(hidden_states)[..., :seq_len].transpose(1, 2)
    )
    if attention_mask is not None:
        hidden_states = (
            hidden_states * attention_mask[:, :, None]
        ).to(hidden_states.dtype)

    # ── split [hidden_states, B, C] ────────────────────
    hidden_states, B_proj, C_proj = torch.split(
        hidden_states,
        [self.intermediate_size,
         self.n_groups * self.ssm_state_size,
         self.n_groups * self.ssm_state_size],
        dim=-1,
    )
    A = -torch.exp(self.A_log.float())                # [n_heads]

    # ── SSD math (verbatim from upstream) ──────────────
    dt = nn.functional.softplus(dt + self.dt_bias)
    dt = torch.clamp(dt, self.time_step_min)
    hidden_states = hidden_states.reshape(
        batch_size, seq_len, -1, self.head_dim
    ).float()                                          # [B, T, nh, hd]
    B_proj = B_proj.reshape(
        batch_size, seq_len, -1, self.ssm_state_size
    ).float()
    C_proj = C_proj.reshape(
        batch_size, seq_len, -1, self.ssm_state_size
    ).float()
    B_proj = B_proj.repeat_interleave(
        self.num_heads // self.n_groups,
        dim=2, output_size=self.num_heads,
    )
    C_proj = C_proj.repeat_interleave(
        self.num_heads // self.n_groups,
        dim=2, output_size=self.num_heads,
    )
    pad_size = (self.chunk_size
                - seq_len % self.chunk_size) % self.chunk_size

    D_residual = self.D[..., None] * pad_tensor_by_size(
        hidden_states, pad_size
    )

    # ── Calibration: rho, gamma (cheap, pre-discretisation) ─
    if _C5B["mode"] == "calibrate":
        rho_t_full = torch.exp(
            A.unsqueeze(0).unsqueeze(0) * dt.float()
        )                                               # [B, T, nh]
        rho_sum_layer   = rho_t_full.sum(dim=(0, 1))    # [nh]
        rho_count_layer = batch_size * seq_len
        gamma_sum_layer   = (C_proj ** 2).sum(dim=(0, 1))  # [nh, ds]
        gamma_count_layer = batch_size * seq_len

    # ── Discretise x and A ────────────────────────────
    hidden_states = hidden_states * dt[..., None]
    A = A.to(hidden_states.dtype) * dt

    # ── Chunked reshape ───────────────────────────────
    hidden_states, A, B_proj, C_proj = [
        reshape_into_chunks(t, pad_size, self.chunk_size)
        for t in (hidden_states, A, B_proj, C_proj)
    ]
    A = A.permute(0, 3, 1, 2)
    A_cumsum = torch.cumsum(A, dim=-1)

    # ── Y_diag (intra-chunk) ──────────────────────────
    L = torch.exp(segment_sum(A))
    G_intermediate = (
        C_proj[:, :, :, None, :, :] * B_proj[:, :, None, :, :, :]
    )
    G = G_intermediate.sum(dim=-1)
    M_intermediate = (
        G[..., None] * L.permute(0, 2, 3, 4, 1)[..., None]
    )
    M = M_intermediate.sum(dim=-1)
    Y_diag = (
        M[..., None] * hidden_states[:, :, None]
    ).sum(3)

    # ── Compute chunk-boundary states ────────────────
    decay_states = torch.exp(
        A_cumsum[:, :, :, -1:] - A_cumsum
    )
    B_decay_contraction = (
        B_proj * decay_states.permute(0, 2, 3, 1)[..., None]
    )
    states = (
        B_decay_contraction.permute(0, 1, 3, 2, 4)[..., None]
        * hidden_states.permute(0, 1, 3, 2, 4)[..., None, :]
    ).sum(dim=3).permute(0, 1, 2, 4, 3)
    previous_states = torch.zeros_like(states[:, :1])
    states = torch.cat([previous_states, states], dim=1)
    decay_chunk = torch.exp(
        segment_sum(nn.functional.pad(
            A_cumsum[:, :, :, -1], (1, 0)
        ))
    )
    states_permuted = states.permute(0, 2, 1, 3, 4)
    result = (
        decay_chunk[..., None, None]
        * states_permuted[:, :, None, ...]
    ).sum(dim=2)
    new_states = result.permute(0, 2, 1, 3, 4)
    states, ssm_state = new_states[:, :-1], new_states[:, -1]
    # states shape:    [B, n_chunks, n_heads, head_dim, d_state]
    # ssm_state shape: [B, n_heads, head_dim, d_state]

    # ── INJECTION POINT: chunk-boundary state quantisation ─
    if _C5B["mode"] == "quantise":
        b_layer = _C5B["b_per_layer"].get(layer_name)
        r_layer = _C5B["r_per_layer"].get(layer_name)
        if b_layer is not None and r_layer is not None:
            b_layer = b_layer.to(states.device)
            r_layer = r_layer.to(states.device, dtype=torch.float32)
            states = _quantize_h_per_unit_mamba2(
                states, b_layer, r_layer
            )

    # Calibration: chunk-boundary state RMS (per (h, d, s))
    if _C5B["mode"] == "calibrate":
        rms_sumsq_layer = (states ** 2).sum(dim=(0, 1))
        rms_count_layer = batch_size * states.shape[1]   # B * n_chunks

    # ── Y_off (inter-chunk; uses possibly-quantised states) ─
    state_decay_out = torch.exp(A_cumsum)
    C_times_states = (
        C_proj[..., None, :] * states[:, :, None, ...]
    )
    state_decay_out_permuted = state_decay_out.permute(0, 2, 3, 1)
    Y_off = (
        C_times_states.sum(-1)
        * state_decay_out_permuted[..., None]
    )

    y = Y_diag + Y_off
    y = y.reshape(batch_size, -1, self.num_heads, self.head_dim)
    y = y + D_residual
    if pad_size > 0:
        y = y[:, :seq_len, :, :]
    y = y.reshape(batch_size, seq_len, -1)

    # Save calibration accumulators
    if _C5B["mode"] == "calibrate":
        prev = _C5B["calib_accum"].get(layer_name)
        if prev is None:
            _C5B["calib_accum"][layer_name] = {
                "rho_sum":     rho_sum_layer.detach().cpu(),
                "rho_count":   rho_count_layer,
                "gamma_sum":   gamma_sum_layer.detach().cpu(),
                "gamma_count": gamma_count_layer,
                "rms_sumsq":   rms_sumsq_layer.detach().cpu(),
                "rms_count":   rms_count_layer,
            }
        else:
            prev["rho_sum"]     += rho_sum_layer.detach().cpu()
            prev["rho_count"]   += rho_count_layer
            prev["gamma_sum"]   += gamma_sum_layer.detach().cpu()
            prev["gamma_count"] += gamma_count_layer
            prev["rms_sumsq"]   += rms_sumsq_layer.detach().cpu()
            prev["rms_count"]   += rms_count_layer

    # Norm + gate fusion + out_proj
    scan_output = self.norm(y, gate)
    contextualized_states = self.out_proj(scan_output.to(dtype))
    return contextualized_states


def _patch_on():
    mixer_class.torch_forward = _c5b_torch_forward


def _patch_off():
    mixer_class.torch_forward = _orig_torch_forward


# =============================================================
# === CELL V4 === Calibration: ρ_h + Γ_(h,s) + RMS(state)
# =============================================================

print(f"\n{'-' * 80}")
print(f"Calibrating ρ_h + Γ_(h,s) + RMS(state) at chunk boundaries")
print(f"{'-' * 80}")

ds_test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
test_text = "\n\n".join([x for x in ds_test["text"]
                          if x.strip()])[:48000]
calib_text = "\n\n".join(
    [x for x in load_dataset("wikitext", "wikitext-2-raw-v1",
                              split="train")["text"]
     if x.strip()])[:8000]
calib_ids = tok.encode(calib_text, return_tensors="pt").to("cuda")[:, :1024]

_patch_on()
_C5B["mode"]        = "calibrate"
_C5B["calib_accum"] = {}
with torch.inference_mode():
    _ = mdl(calib_ids, use_cache=False)
_C5B["mode"] = "off"

captured = {}
for name, accum in _C5B["calib_accum"].items():
    rho_h = accum["rho_sum"]   / accum["rho_count"]
    gamma = accum["gamma_sum"] / accum["gamma_count"]
    rms   = (accum["rms_sumsq"] /
              accum["rms_count"]).clamp(min=1e-12).sqrt()
    captured[name] = {
        "rho_h":    rho_h,
        "gamma":    gamma,
        "rms_h":    rms,
    }
print(f"  captured rho_h / gamma / rms_h for {len(captured)} layers")


# =============================================================
# === CELL V5 === Bit allocation profiles (per (h, s))
# =============================================================

B_MIN   = 2
RANGE_K = 7.0


def _extras_uniform(captured):
    return {n: torch.zeros_like(d["gamma"]) for n, d in captured.items()}


def _extras_proxy_C2(captured):
    out = {}
    for n, d in captured.items():
        rho = d["rho_h"].clamp(max=1.0 - EPS)
        log_R = torch.log2(1.0 / (1.0 - rho))
        log_R_hs = log_R.unsqueeze(-1).expand_as(d["gamma"])
        out[n] = torch.ceil(log_R_hs).clamp(min=0)
    return out


def _extras_proxy_correct(captured):
    out = {}
    for n, d in captured.items():
        rho = d["rho_h"].clamp(max=1.0 - EPS)
        log_R = 0.5 * torch.log2(1.0 / (1.0 - rho))
        log_R_hs = log_R.unsqueeze(-1).expand_as(d["gamma"])
        out[n] = torch.ceil(log_R_hs).clamp(min=0)
    return out


def _extras_full(captured):
    out = {}
    for n, d in captured.items():
        rho = d["rho_h"].clamp(max=1.0 - EPS)
        gamma = d["gamma"].clamp(min=EPS)
        log_R = torch.log2(
            gamma / (1.0 - rho).unsqueeze(-1)
        )
        out[n] = torch.ceil(log_R).clamp(min=0)
    return out


def _extras_anti_full(captured):
    out = {}
    for n, d in captured.items():
        rho = d["rho_h"].clamp(max=1.0 - EPS)
        gamma = d["gamma"].clamp(min=EPS)
        log_R = torch.log2(
            (1.0 - rho).unsqueeze(-1) / gamma
        )
        out[n] = torch.ceil(log_R)
    return out


def _build_b_per_unit(extras_dict, b_0):
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


def _r_per_layer(captured, k=RANGE_K):
    return {n: k * d["rms_h"] for n, d in captured.items()}


PROFILE_LABELS = ["uniform", "proxy_C2", "proxy_correct",
                   "full", "anti_full"]
EXTRAS_FNS = {
    "uniform":       _extras_uniform,
    "proxy_C2":      _extras_proxy_C2,
    "proxy_correct": _extras_proxy_correct,
    "full":          _extras_full,
    "anti_full":     _extras_anti_full,
}


def build_profiles(captured, b_0):
    return {
        label: _build_b_per_unit(EXTRAS_FNS[label](captured), b_0)
        for label in PROFILE_LABELS
    }


B_0_MAIN = 6
profiles = build_profiles(captured, B_0_MAIN)
r_per_layer = _r_per_layer(captured, k=RANGE_K)


# =============================================================
# === CELL V6 === Sliding-window PPL
# =============================================================

def ppl_eval(model, ids, b_per_layer=None, r_per_layer=None,
              ctx_len=1024, stride=512, n_windows=5):
    if b_per_layer is None:
        # FP16 mode = patch off (calls original SSD bit-for-bit)
        _patch_off()
        _C5B["mode"]         = "off"
        _C5B["b_per_layer"]  = None
        _C5B["r_per_layer"]  = None
    else:
        _patch_on()
        _C5B["mode"]         = "quantise"
        _C5B["b_per_layer"]  = b_per_layer
        _C5B["r_per_layer"]  = r_per_layer
    nll_total = 0.0
    tok_total = 0
    windows_done = 0
    try:
        with torch.inference_mode():
            for begin in range(0, ids.shape[1] - ctx_len, stride):
                if windows_done >= n_windows:
                    break
                window = ids[:, begin:begin + ctx_len]
                out = model(window, use_cache=False)
                log_probs = F.log_softmax(out.logits[0, :-1].float(),
                                            dim=-1)
                targets = window[0, 1:]
                nll = -log_probs.gather(-1,
                                          targets.unsqueeze(-1)).squeeze(-1)
                nll_total += nll.sum().item()
                tok_total += targets.numel()
                windows_done += 1
    finally:
        _C5B["mode"]        = "off"
        _C5B["b_per_layer"] = None
        _C5B["r_per_layer"] = None
        _patch_off()
    return math.exp(nll_total / tok_total), windows_done


# =============================================================
# === CELL V7 === Run sweep (6 conditions @ b_0 = 6)
# =============================================================

CTX_LEN   = 1024
STRIDE    = 512
N_WINDOWS = 5

test_ids = tok.encode(test_text, return_tensors="pt").to("cuda")
print(f"\n{'-' * 80}")
print(f"Zamba2-1.2B PPL  (ctx={CTX_LEN}, stride={STRIDE}, "
      f"n_windows={N_WINDOWS})  b_0={B_0_MAIN}, range_k={RANGE_K}")
print(f"  Quant inject point: chunk-boundary `states` (every "
      f"chunk_size tokens)")
print(f"{'-' * 80}")

results = {}

# 1. FP16 baseline (= unmodified SSD via patch=off shortcut)
print(f"\n  1. FP16 (mode=off, original SSD)...")
fp16_ppl, _ = ppl_eval(mdl, test_ids, None, None,
                          ctx_len=CTX_LEN, stride=STRIDE,
                          n_windows=N_WINDOWS)
print(f"     ppl = {fp16_ppl:.4f}")
results["fp16"] = {"ppl": fp16_ppl}

# 2 - 6. Quantised conditions
for label in PROFILE_LABELS:
    print(f"\n  Profile: {label}  (b_0 = {B_0_MAIN}, k = {RANGE_K})")
    b_dict = profiles[label]
    flat   = torch.cat([b.flatten() for b in b_dict.values()])
    mean_b = float(flat.float().mean())
    std_b  = float(flat.float().std())
    ppl, n_win = ppl_eval(mdl, test_ids, b_dict, r_per_layer,
                            ctx_len=CTX_LEN, stride=STRIDE,
                            n_windows=N_WINDOWS)
    dppl = ppl - fp16_ppl
    pct  = 100.0 * dppl / fp16_ppl
    print(f"     ppl = {ppl:>9.4f}  ΔPPL = {dppl:+8.4f}  "
          f"({pct:+8.3f} % of FP16)  "
          f"mean_b={mean_b:5.2f}  std_b={std_b:4.2f}")
    results[f"profile_{label}"] = {
        "ppl":          ppl,
        "delta_ppl":    dppl,
        "pct_of_base":  pct,
        "n_windows":    n_win,
        "mean_b":       mean_b,
        "std_b":        std_b,
    }


# =============================================================
# === CELL V8 === Verdict
# =============================================================

full_pct      = results["profile_full"]["pct_of_base"]
proxy_C2_pct  = results["profile_proxy_C2"]["pct_of_base"]
proxy_sq_pct  = results["profile_proxy_correct"]["pct_of_base"]
unif_pct      = results["profile_uniform"]["pct_of_base"]
anti_pct      = results["profile_anti_full"]["pct_of_base"]

print(f"\n{'=' * 80}")
print(f"Phase C-5b verdict @ b_0 = {B_0_MAIN}, range_k = {RANGE_K}")
print(f"  Quant injection: chunk-boundary state (Mamba2 SSD path)")
print(f"{'=' * 80}")
print(f"\n  {'profile':<16s}  ΔPPL %")
print(f"  {'─' * 16}  ──────")
for label in PROFILE_LABELS:
    p = results[f"profile_{label}"]["pct_of_base"]
    print(f"  {label:<16s}  {p:+7.3f}")

checks = {
    "full_beats_uniform": full_pct < unif_pct,
    "full_beats_proxy":   full_pct < proxy_C2_pct,
    "full_beats_anti":    full_pct < anti_pct,
    "full_is_best":       full_pct == min([full_pct, proxy_C2_pct,
                                              proxy_sq_pct, unif_pct,
                                              anti_pct]),
}
print(f"\n  Pre-registered checks:")
for k, v in checks.items():
    mark = "✓" if v else "✗"
    print(f"    {mark}  {k}")
all_pass = all(checks.values())
print(f"\n  All checks pass: {'YES' if all_pass else 'NO'}")

if all_pass:
    verdict_class = "BIT_ALLOCATOR_TRANSFERS"
    verdict = (f"Paper C bit allocator transfers from Mamba1 "
                f"(C-3 / C-3b / C-4) to Mamba2 hybrid context "
                f"(Zamba2-1.2B). `full` retains rank-1 over "
                f"uniform / proxy_C2 / anti_full at b_0 = "
                f"{B_0_MAIN}. C-5c (composition with attention "
                f"INT3) can proceed.")
elif checks["full_beats_uniform"] and checks["full_beats_anti"]:
    verdict_class = "ALLOCATOR_PARTIAL"
    verdict = (f"`full` beats `uniform` and `anti_full` but does "
                f"not beat the ρ-only proxy. Mamba2's per-head ρ "
                f"granularity may make Γ-aware allocation less "
                f"valuable than in Mamba1's per-(c, s) regime.")
else:
    verdict_class = "ALLOCATOR_FAILS"
    verdict = (f"`full` does not retain rank-1 in this hybrid "
                f"context. The bit allocator's transfer to Mamba2 "
                f"is not clean. Investigate before proceeding.")

print(f"\n  Class:  {verdict_class}")
print(f"  Verdict:")
print(f"    {verdict}")


# =============================================================
# === CELL V9 === Save
# =============================================================

summary = []
for name, d in captured.items():
    summary.append({
        "layer_name":     name,
        "rho_h_min":      float(d["rho_h"].min()),
        "rho_h_p50":      float(d["rho_h"].quantile(0.50)),
        "rho_h_max":      float(d["rho_h"].max()),
        "gamma_min":      float(d["gamma"].min()),
        "gamma_p50":      float(d["gamma"].quantile(0.50)),
        "gamma_max":      float(d["gamma"].max()),
        "rms_h_p50":      float(d["rms_h"].quantile(0.50)),
        "rms_h_max":      float(d["rms_h"].max()),
        "n_heads":        int(d["rho_h"].shape[0]),
        "n_state_dims":   int(d["gamma"].shape[1]),
        "n_head_dim":     int(d["rms_h"].shape[1]),
    })

output = {
    "date":            datetime.utcnow().isoformat(timespec="seconds"),
    "phase":           "C-5b",
    "purpose":         ("Chunk-boundary state quantisation of "
                        "Zamba2-1.2B's Mamba2 mixers (attention "
                        "held at FP16). Verifies that Phase C-4's "
                        "bit allocator transfers from Mamba1 per-"
                        "(c, s) to Mamba2 per-(h, s) granularity."),
    "implementation":  ("SSD math preserved bit-for-bit; quantiser "
                        "injected on chunk-boundary `states` "
                        "between SSD's state computation and Y_off "
                        "readout. mode='off' shortcuts to original "
                        "torch_forward, so FP16 baseline is "
                        "guaranteed identical."),
    "transformers_version": _hf_transformers.__version__,
    "model":           MODEL_ID,
    "label":           MODEL_LABEL,
    "protocol": {
        "ctx_len":         CTX_LEN,
        "stride":          STRIDE,
        "n_windows":       N_WINDOWS,
        "b_0":             B_0_MAIN,
        "b_min":           B_MIN,
        "range_k":         RANGE_K,
        "calib_split":     "WikiText-2 train",
        "test_split":      "WikiText-2 test",
        "quant_scheme":    ("per-(h, d, s) INT-b symmetric mid-tread "
                            "on chunk-boundary states, "
                            "b shared over head_dim, "
                            "range = k * RMS_state(h, d, s)"),
        "profiles":        PROFILE_LABELS,
    },
    "fp16_ppl":        fp16_ppl,
    "results":         results,
    "checks":          checks,
    "all_pass":        all_pass,
    "verdict_class":   verdict_class,
    "verdict":         verdict,
    "rho_gamma_rms_summary": summary,
}

out_path = "/content/phase_c5b_ssm_quant_in_hybrid.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False, default=str)
print(f"\nSaved -> {out_path}")
