# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Phase C-5c: Hybrid composition diagnostic
# ============================================================
# Tests whether the independently-validated Transformer-KV
# (modeq INT3, Paper B) and SSM-state (Paper C `full` allocator,
# C-4 / C-5b validated) quantisation recipes *compose* inside a
# hybrid Mamba2 + Attention architecture (Zamba2-1.2B), or
# whether cross-substrate coupling introduces non-additive
# degradation. This is a diagnostic experiment, NOT yet the
# final deployment / memory-accounting benchmark (that's C-6).
#
# Conditions (4):
#   1. FP16             no quantisation
#   2. attn-only        modeq INT3 K/V on attention, FP16 SSM
#   3. SSM-only         Paper C `full` SSM (b_0=6) chunk-bnd, FP16 attn
#   4. both             modeq INT3 K/V + Paper C `full` SSM
#
# Pre-registered hypothesis (additivity check):
#   ΔPPL(both) ≈ ΔPPL(attn-only) + ΔPPL(SSM-only)
#   - |interaction| < 0.5 percentage points → clean composition ✓
#   - interaction > 0.5 pp                  → non-additive degradation
#   - interaction < -0.5 pp                 → unexpected cancellation
#
# Implementation note:
# We use a single-forward eval (use_cache=False) rather than
# Paper B's prefill+decode pattern, because Zamba2's Mamba2
# torch_forward `use_precomputed_state` branch is a single-token
# decode path that does not accept the multi-token decode
# windows Paper B uses. To match Paper B's "K/V coming from a
# compressed cache" semantics, we monkey-patch
# `Zamba2Attention.forward` to apply `qa_nsep_pchan_asym(K, 3)`
# and `qa_nsep_pchan_asym(V, 3)` between the K/V projection
# (post-RoPE) and the attention computation. This is
# mathematically identical to the cache-compression scenario
# for fresh-cache forwards. SSM-side uses the same chunk-
# boundary `states` quantisation as C-5b.
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
# === CELL V2 === Load Zamba2-1.2B (from C-5a/b)
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


def find_attentions(model):
    """Return all module instances that have q_proj, k_proj, v_proj
    as Linear children (the Zamba2Attention shared-attention blocks)."""
    found = []
    seen = set()
    for name, mod in model.named_modules():
        if id(mod) in seen:
            continue
        if (isinstance(getattr(mod, "q_proj", None), nn.Linear)
            and isinstance(getattr(mod, "k_proj", None), nn.Linear)
            and isinstance(getattr(mod, "v_proj", None), nn.Linear)
            and isinstance(getattr(mod, "o_proj", None), nn.Linear)):
            found.append((name, mod))
            seen.add(id(mod))
    return found


mixers = find_mixers(mdl)
attentions = find_attentions(mdl)
print(f"  Found {len(mixers)} Zamba2MambaMixer layers")
print(f"  Found {len(attentions)} Zamba2Attention modules")
LAYER_NAME_BY_ID = {id(mix): name for name, mix in mixers}
mixer_class = type(mixers[0][1])
attn_class  = type(attentions[0][1])
print(f"  Mixer class:     "
      f"{mixer_class.__module__}.{mixer_class.__name__}")
print(f"  Attention class: "
      f"{attn_class.__module__}.{attn_class.__name__}")


# Pull SSD helpers + attention helpers from the Zamba2 module
from transformers.models.zamba2.modeling_zamba2 import (
    pad_tensor_by_size, reshape_into_chunks, segment_sum,
    apply_rotary_pos_emb, eager_attention_forward,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


# =============================================================
# === CELL V3 === Patches: SSM forward + Attention forward
# =============================================================

_C5C = {
    # SSM side
    "ssm_mode":      "off",       # "off" | "calibrate" | "quantise"
    "b_per_layer":   None,
    "r_per_layer":   None,
    "calib_accum":   None,
    # Attention side
    "attn_quant":    False,
    "attn_bits":     3,
}

EPS = 1e-6


def _quantize_h_per_unit_mamba2(states, b_per_hs, r_per_hds):
    """Per-(h, d, s) INT-b symmetric mid-tread quantiser with
    zero level (matches C-4 / C-5b)."""
    nh, hd, ds = r_per_hds.shape
    b_expanded = b_per_hs.unsqueeze(1).expand(nh, hd, ds).float().clamp(
        min=2.0, max=16.0
    )
    n_neg = 2.0 ** (b_expanded - 1.0)
    n_pos = n_neg - 1.0
    delta = (r_per_hds / n_neg).clamp(min=EPS)
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


def qa_nsep_pchan_asym(x, bits):
    """modeq v0.3 INT-b asymmetric per-channel norm-separated
    quantisation (Paper B). Inlined for self-contained Colab.
    Operates on shape [..., T, D] with D = head_dim as the
    channel axis. Reference: src/modeq/quantize.py."""
    dtype = x.dtype
    xf = x.float()
    n = xf.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    d = xf / n
    d_min = d.amin(dim=-2, keepdim=True)
    d_max = d.amax(dim=-2, keepdim=True)
    d_range = (d_max - d_min).clamp(min=1e-12)
    levels = 2 ** bits - 1
    step = d_range / levels
    d_q = ((d - d_min) / step).round().clamp(0, levels) * step + d_min
    d_q = d_q / d_q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return (n * d_q).to(dtype)


# ── SSM forward patch (chunk-boundary, copied from C-5b) ──
_orig_torch_forward = mixer_class.torch_forward


def _c5c_torch_forward(self, input_states, cache_params=None,
                        attention_mask=None):
    if _C5C["ssm_mode"] == "off":
        return _orig_torch_forward(self, input_states,
                                      cache_params, attention_mask)
    if cache_params is not None and cache_params.has_previous_state(self.layer_idx):
        # Decode-mode (per-token) is not patched. Defer to original.
        return _orig_torch_forward(self, input_states,
                                      cache_params, attention_mask)

    layer_name = LAYER_NAME_BY_ID.get(id(self))
    batch_size, seq_len, _ = input_states.shape
    dtype = input_states.dtype

    if attention_mask is not None:
        input_states = (input_states * attention_mask[:, :, None]).to(dtype)
    projected_states = self.in_proj(input_states)
    d_mlp = (projected_states.shape[-1] - 2 * self.intermediate_size
             - 2 * self.n_groups * self.ssm_state_size
             - self.num_heads) // 2
    _, _, gate, hidden_states, dt = projected_states.split(
        [d_mlp, d_mlp, self.intermediate_size,
         self.conv_dim, self.num_heads], dim=-1,
    )
    hidden_states = hidden_states.transpose(1, 2)
    hidden_states = self.act(
        self.conv1d(hidden_states)[..., :seq_len].transpose(1, 2)
    )
    if attention_mask is not None:
        hidden_states = (
            hidden_states * attention_mask[:, :, None]
        ).to(hidden_states.dtype)
    hidden_states, B_proj, C_proj = torch.split(
        hidden_states,
        [self.intermediate_size,
         self.n_groups * self.ssm_state_size,
         self.n_groups * self.ssm_state_size], dim=-1,
    )
    A = -torch.exp(self.A_log.float())

    dt = nn.functional.softplus(dt + self.dt_bias)
    dt = torch.clamp(dt, self.time_step_min)
    hidden_states = hidden_states.reshape(
        batch_size, seq_len, -1, self.head_dim
    ).float()
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

    if _C5C["ssm_mode"] == "calibrate":
        rho_t_full = torch.exp(
            A.unsqueeze(0).unsqueeze(0) * dt.float()
        )
        rho_sum_layer   = rho_t_full.sum(dim=(0, 1))
        rho_count_layer = batch_size * seq_len
        gamma_sum_layer   = (C_proj ** 2).sum(dim=(0, 1))
        gamma_count_layer = batch_size * seq_len

    hidden_states = hidden_states * dt[..., None]
    A = A.to(hidden_states.dtype) * dt
    hidden_states, A, B_proj, C_proj = [
        reshape_into_chunks(t, pad_size, self.chunk_size)
        for t in (hidden_states, A, B_proj, C_proj)
    ]
    A = A.permute(0, 3, 1, 2)
    A_cumsum = torch.cumsum(A, dim=-1)

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

    # Inject quantisation on chunk-boundary states
    if _C5C["ssm_mode"] == "quantise":
        b_layer = _C5C["b_per_layer"].get(layer_name)
        r_layer = _C5C["r_per_layer"].get(layer_name)
        if b_layer is not None and r_layer is not None:
            b_layer = b_layer.to(states.device)
            r_layer = r_layer.to(states.device, dtype=torch.float32)
            states = _quantize_h_per_unit_mamba2(states, b_layer, r_layer)

    if _C5C["ssm_mode"] == "calibrate":
        rms_sumsq_layer = (states ** 2).sum(dim=(0, 1))
        rms_count_layer = batch_size * states.shape[1]

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

    if _C5C["ssm_mode"] == "calibrate":
        prev = _C5C["calib_accum"].get(layer_name)
        if prev is None:
            _C5C["calib_accum"][layer_name] = {
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

    scan_output = self.norm(y, gate)
    return self.out_proj(scan_output.to(dtype))


# ── Attention forward patch ──
_orig_attn_forward = attn_class.forward


def _c5c_attention_forward(
    self, hidden_states, layer_idx,
    attention_mask=None, past_key_values=None,
    position_embeddings=None, **kwargs,
):
    """Faithful copy of Zamba2Attention.forward with optional
    modeq INT3 quantisation of K and V applied between the
    (post-RoPE) projection and the attention computation. This
    is mathematically identical to the cache-compression
    pattern (Paper B's `compress_cache`) for fresh-cache
    forwards (use_cache=False)."""
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states)
    key_states   = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)
    if self.config.use_shared_attention_adapter:
        adapter_layer_idx = self.layer_dic[layer_idx]
        query_states = (
            query_states
            + self.linear_q_adapter_list[adapter_layer_idx](hidden_states)
        )
        key_states = (
            key_states
            + self.linear_k_adapter_list[adapter_layer_idx](hidden_states)
        )
        value_states = (
            value_states
            + self.linear_v_adapter_list[adapter_layer_idx](hidden_states)
        )

    query_states = query_states.view(hidden_shape).transpose(1, 2)
    key_states   = key_states.view(hidden_shape).transpose(1, 2)
    value_states = value_states.view(hidden_shape).transpose(1, 2)

    if self.config.use_mem_rope:
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

    if past_key_values is not None:
        key_states, value_states = past_key_values.update(
            key_states, value_states, layer_idx
        )

    # *** modeq INT3 K/V quantisation ***
    if _C5C["attn_quant"]:
        # K, V shape: [B, n_kv_heads, T, head_dim]
        # qa_nsep_pchan_asym expects [..., T, D]
        key_states   = qa_nsep_pchan_asym(key_states,   _C5C["attn_bits"])
        value_states = qa_nsep_pchan_asym(value_states, _C5C["attn_bits"])

    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, eager_attention_forward
    )
    attn_output, attn_weights = attention_interface(
        self, query_states, key_states, value_states, attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling, **kwargs,
    )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def _patch_ssm_on():
    mixer_class.torch_forward = _c5c_torch_forward


def _patch_ssm_off():
    mixer_class.torch_forward = _orig_torch_forward


def _patch_attn_on():
    attn_class.forward = _c5c_attention_forward


def _patch_attn_off():
    attn_class.forward = _orig_attn_forward


# =============================================================
# === CELL V4 === Calibration (SSM side, for `full` profile)
# =============================================================

print(f"\n{'-' * 80}")
print(f"Calibrating ρ_h + Γ_(h,s) + RMS(state) on chunk boundaries")
print(f"{'-' * 80}")

ds_test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
test_text = "\n\n".join([x for x in ds_test["text"]
                          if x.strip()])[:48000]
calib_text = "\n\n".join(
    [x for x in load_dataset("wikitext", "wikitext-2-raw-v1",
                              split="train")["text"]
     if x.strip()])[:8000]
calib_ids = tok.encode(calib_text, return_tensors="pt").to("cuda")[:, :1024]

_patch_ssm_on()
_C5C["ssm_mode"]    = "calibrate"
_C5C["calib_accum"] = {}
with torch.inference_mode():
    _ = mdl(calib_ids, use_cache=False)
_C5C["ssm_mode"] = "off"
_patch_ssm_off()

captured = {}
for name, accum in _C5C["calib_accum"].items():
    rho_h = accum["rho_sum"]   / accum["rho_count"]
    gamma = accum["gamma_sum"] / accum["gamma_count"]
    rms   = (accum["rms_sumsq"] /
              accum["rms_count"]).clamp(min=1e-12).sqrt()
    captured[name] = {"rho_h": rho_h, "gamma": gamma, "rms_h": rms}
print(f"  captured rho_h / gamma / rms_h for {len(captured)} layers")


# =============================================================
# === CELL V5 === Build `full` profile @ b_0 = 6
# =============================================================

B_MIN   = 2
RANGE_K = 7.0
B_0     = 6


def _extras_full(captured):
    out = {}
    for n, d in captured.items():
        rho = d["rho_h"].clamp(max=1.0 - EPS)
        gamma = d["gamma"].clamp(min=EPS)
        log_R = torch.log2(gamma / (1.0 - rho).unsqueeze(-1))
        out[n] = torch.ceil(log_R).clamp(min=0)
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


b_per_layer_full = _build_b_per_unit(_extras_full(captured), B_0)
r_per_layer = {n: RANGE_K * d["rms_h"] for n, d in captured.items()}

flat = torch.cat([b.flatten() for b in b_per_layer_full.values()])
print(f"\n  `full` profile @ b_0={B_0}: mean_b={flat.float().mean():.2f}, "
      f"std_b={flat.float().std():.2f}")


# =============================================================
# === CELL V6 === PPL eval (4 conditions)
# =============================================================

def ppl_eval(model, ids, ssm_quant, attn_quant,
              ctx_len=1024, stride=512, n_windows=5):
    """Single-forward PPL. ssm_quant ∈ {False, True} toggles
    chunk-boundary state quantisation. attn_quant ∈ {False, True}
    toggles modeq INT3 K/V quantisation."""
    if ssm_quant:
        _patch_ssm_on()
        _C5C["ssm_mode"]    = "quantise"
        _C5C["b_per_layer"] = b_per_layer_full
        _C5C["r_per_layer"] = r_per_layer
    else:
        _C5C["ssm_mode"]    = "off"
        _patch_ssm_off()
    if attn_quant:
        _patch_attn_on()
        _C5C["attn_quant"]  = True
        _C5C["attn_bits"]   = 3
    else:
        _C5C["attn_quant"]  = False
        _patch_attn_off()

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
        _C5C["ssm_mode"]    = "off"
        _C5C["attn_quant"]  = False
        _patch_ssm_off()
        _patch_attn_off()
    return math.exp(nll_total / tok_total), windows_done


# =============================================================
# === CELL V7 === Run 4 conditions + additivity check
# =============================================================

CTX_LEN   = 1024
STRIDE    = 512
N_WINDOWS = 5

test_ids = tok.encode(test_text, return_tensors="pt").to("cuda")
print(f"\n{'-' * 80}")
print(f"Zamba2-1.2B PPL  (ctx={CTX_LEN}, stride={STRIDE}, "
      f"n_windows={N_WINDOWS})")
print(f"  SSM quant:  Paper C `full` @ b_0={B_0}, k={RANGE_K}, "
      f"chunk-boundary states")
print(f"  Attn quant: modeq INT3 asym (qa_nsep_pchan_asym, bits=3)")
print(f"{'-' * 80}")

CONDITIONS = [
    ("FP16",          False, False),
    ("attn_only",     False, True),
    ("SSM_only",      True,  False),
    ("both",          True,  True),
]

results = {}
for label, ssm_q, attn_q in CONDITIONS:
    print(f"\n  Condition: {label}  (SSM={ssm_q}, attn={attn_q})...")
    ppl, n_win = ppl_eval(
        mdl, test_ids, ssm_q, attn_q,
        ctx_len=CTX_LEN, stride=STRIDE, n_windows=N_WINDOWS,
    )
    results[label] = {"ppl": ppl, "n_windows": n_win}
    print(f"     ppl = {ppl:.4f}")

fp16_ppl = results["FP16"]["ppl"]
for label in ["attn_only", "SSM_only", "both"]:
    ppl = results[label]["ppl"]
    dppl = ppl - fp16_ppl
    pct  = 100.0 * dppl / fp16_ppl
    results[label]["delta_ppl"]    = dppl
    results[label]["pct_of_base"]  = pct

results["FP16"]["delta_ppl"]   = 0.0
results["FP16"]["pct_of_base"] = 0.0

# Additivity check
attn_pct  = results["attn_only"]["pct_of_base"]
ssm_pct   = results["SSM_only"]["pct_of_base"]
both_pct  = results["both"]["pct_of_base"]
expected_both_pct = attn_pct + ssm_pct
interaction_pp    = both_pct - expected_both_pct


# =============================================================
# === CELL V8 === Verdict
# =============================================================

print(f"\n{'=' * 80}")
print(f"Phase C-5c verdict — composition diagnostic")
print(f"{'=' * 80}")
print(f"\n  {'condition':<14s}  {'PPL':>9s}  {'ΔPPL':>9s}  {'% of FP16':>12s}")
print(f"  {'─' * 14}  {'─' * 9}  {'─' * 9}  {'─' * 12}")
for label in ["FP16", "attn_only", "SSM_only", "both"]:
    r = results[label]
    print(f"  {label:<14s}  {r['ppl']:>9.4f}  "
          f"{r['delta_ppl']:>+9.4f}  {r['pct_of_base']:>+11.3f}%")

print(f"\n  Additivity check:")
print(f"    attn_only  ΔPPL%  = {attn_pct:>+8.3f}")
print(f"    SSM_only   ΔPPL%  = {ssm_pct:>+8.3f}")
print(f"    expected   ΔPPL%  = {expected_both_pct:>+8.3f}  "
      f"(if perfectly additive)")
print(f"    observed   ΔPPL%  = {both_pct:>+8.3f}")
print(f"    interaction       = {interaction_pp:>+8.3f} pp  "
      f"(observed − expected)")

ABS_THRESHOLD = 0.5
if abs(interaction_pp) < ABS_THRESHOLD:
    verdict_class = "CLEAN_COMPOSITION"
    verdict = (f"|interaction| = {abs(interaction_pp):.3f} pp < "
                f"{ABS_THRESHOLD} pp threshold. The two recipes "
                f"compose additively in this hybrid context. "
                f"Paper B's attention recipe and Paper C's SSM "
                f"recipe can be combined deployment-side without "
                f"cross-substrate amplification. C-6 (real "
                f"packing + memory accounting) is unblocked.")
elif interaction_pp > 0:
    verdict_class = "NON_ADDITIVE_DEGRADATION"
    verdict = (f"interaction = +{interaction_pp:.3f} pp > "
                f"+{ABS_THRESHOLD} pp threshold. The combined "
                f"recipe degrades MORE than the sum of its "
                f"parts. Cross-substrate coupling exists. "
                f"Investigate whether (i) attention quantisation "
                f"distorts the residual that feeds the next SSM "
                f"layer, or (ii) SSM quantisation distorts the "
                f"context that feeds attention. Mitigations: "
                f"raise SSM b_0, raise attn bits, or accept this "
                f"as a non-additive cost in the paper.")
else:
    verdict_class = "UNEXPECTED_CANCELLATION"
    verdict = (f"interaction = {interaction_pp:.3f} pp < "
                f"-{ABS_THRESHOLD} pp threshold. The combined "
                f"recipe degrades LESS than the sum of its "
                f"parts (some cancellation). Possible: noise on "
                f"one substrate is being absorbed by the other "
                f"without amplification, or the FP16 baseline "
                f"itself has slack that quantisation collectively "
                f"removes (regularisation). Worth a paper note "
                f"but does not block C-6.")

print(f"\n  Class:    {verdict_class}")
print(f"  Verdict:  {verdict}")


# =============================================================
# === CELL V9 === Save
# =============================================================

output = {
    "date":           datetime.utcnow().isoformat(timespec="seconds"),
    "phase":          "C-5c",
    "purpose":        ("Composition diagnostic: does Paper B's "
                       "modeq INT3 K/V attention quantisation "
                       "compose additively with Paper C's `full` "
                       "SSM bit allocator inside Zamba2-1.2B?"),
    "implementation": ("Single-forward eval (use_cache=False). "
                       "SSM patched at chunk-boundary states "
                       "(C-5b path). Attention patched between "
                       "post-RoPE K/V projection and attention "
                       "computation, applying qa_nsep_pchan_asym "
                       "with bits=3. The two patches are "
                       "independently togglable."),
    "transformers_version": _hf_transformers.__version__,
    "model":          MODEL_ID,
    "label":          MODEL_LABEL,
    "protocol": {
        "ctx_len":     CTX_LEN,
        "stride":      STRIDE,
        "n_windows":   N_WINDOWS,
        "ssm_b_0":     B_0,
        "ssm_b_min":   B_MIN,
        "ssm_range_k": RANGE_K,
        "attn_bits":   3,
        "attn_method": "qa_nsep_pchan_asym (modeq v0.3)",
        "calib_split": "WikiText-2 train",
        "test_split":  "WikiText-2 test",
        "conditions":  [c[0] for c in CONDITIONS],
    },
    "fp16_ppl":             fp16_ppl,
    "results":              results,
    "additivity": {
        "attn_only_pct":     attn_pct,
        "ssm_only_pct":      ssm_pct,
        "expected_pct":      expected_both_pct,
        "observed_pct":      both_pct,
        "interaction_pp":    interaction_pp,
        "abs_threshold_pp":  ABS_THRESHOLD,
    },
    "verdict_class":         verdict_class,
    "verdict":               verdict,
}

out_path = "/content/phase_c5c_hybrid_composition.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False, default=str)
print(f"\nSaved -> {out_path}")


# =============================================================
# === READING ===
# =============================================================
#
# CLEAN_COMPOSITION:
#   The two recipes are deployment-friendly: combining them is
#   safe. Paper C's headline becomes "Paper B + C compose
#   without amplification, unlocking 1-2 bits on SSM in addition
#   to Paper B's 3 bits on attention K/V." Move to C-6 (real
#   packing).
#
# NON_ADDITIVE_DEGRADATION:
#   The naive composition leaks more than additive. Paper C
#   should report this as a deployment caveat. Two paths:
#     (a) raise SSM b_0 to 7 or 8 to reduce SSM-side noise; the
#         attention side is fixed at INT3 by Paper B's recipe.
#     (b) report the non-additive cost transparently and offer
#         a mitigation (e.g., interleaved fine-tuning, or per-
#         layer bit budget tuning that accounts for cross-
#         substrate coupling).
#
# UNEXPECTED_CANCELLATION:
#   The composition is BETTER than additive. Possible
#   regularisation. Investigate: (a) does this hold across more
#   windows? (b) does it hold for SSM b_0 ∈ {5, 7, 8}? If
#   robust, frame as a "free lunch from heterogeneous
#   quantisation" finding.
# ============================================================
