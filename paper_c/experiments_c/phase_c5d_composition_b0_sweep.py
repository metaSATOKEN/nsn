# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Phase C-5d: Composition diagnostic — b_0 sweep
# ============================================================
# Follow-up to Phase C-5c. C-5c showed CLEAN_COMPOSITION at a
# single SSM bit budget b_0 = 6. C-5d sweeps b_0 ∈ {3, 4, 5, 6,
# 8} and re-evaluates the additivity check at each level. The
# question is whether the additive-composition property is
# robust across the full deployment-relevant SSM bit range
# (where Paper C's `full` allocator's per-bit advantage is
# largest at low b_0 and shrinks at high b_0).
#
# Conditions per b_0:
#   1. SSM_only @ b_0  (Paper C `full`, attn FP16)
#   2. both     @ b_0  (Paper C `full` + modeq INT3)
# Plus two b_0-independent baselines (run once):
#   FP16
#   attn_only          (modeq INT3, SSM FP16)
#
# Total runs: 2 + 5 b_0 × 2 = 12 PPL evaluations. ~2 min on A100.
#
# Reading template:
#   Per b_0, compute interaction = observed_both - (attn + SSM).
#   - |interaction| < 0.5 pp → CLEAN at this b_0
#   - >  0.5 pp → DEGRADATION at this b_0
#   - < -0.5 pp → CANCELLATION at this b_0
# Look for whether the verdict class is stable across b_0.
# Particularly interesting points:
#   - b_0 = 3 (most aggressive): does SSM-side breakdown amplify
#                                 attention damage?
#   - b_0 = 8 (near-FP):         does interaction → 0 cleanly?
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
# === CELL V2 === Load Zamba2-1.2B
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
LAYER_NAME_BY_ID = {id(mix): name for name, mix in mixers}
mixer_class = type(mixers[0][1])
attn_class  = type(attentions[0][1])
print(f"  Found {len(mixers)} SSM mixers + {len(attentions)} "
      f"attention modules")

from transformers.models.zamba2.modeling_zamba2 import (
    pad_tensor_by_size, reshape_into_chunks, segment_sum,
    apply_rotary_pos_emb, eager_attention_forward,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


# =============================================================
# === CELL V3 === Patches (identical to C-5c)
# =============================================================

_C5D = {
    "ssm_mode":      "off",
    "b_per_layer":   None,
    "r_per_layer":   None,
    "calib_accum":   None,
    "attn_quant":    False,
    "attn_bits":     3,
}

EPS = 1e-6


def _quantize_h_per_unit_mamba2(states, b_per_hs, r_per_hds):
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


_orig_torch_forward = mixer_class.torch_forward


def _c5d_torch_forward(self, input_states, cache_params=None,
                        attention_mask=None):
    if _C5D["ssm_mode"] == "off":
        return _orig_torch_forward(self, input_states,
                                      cache_params, attention_mask)
    if cache_params is not None and cache_params.has_previous_state(self.layer_idx):
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

    if _C5D["ssm_mode"] == "calibrate":
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

    if _C5D["ssm_mode"] == "quantise":
        b_layer = _C5D["b_per_layer"].get(layer_name)
        r_layer = _C5D["r_per_layer"].get(layer_name)
        if b_layer is not None and r_layer is not None:
            b_layer = b_layer.to(states.device)
            r_layer = r_layer.to(states.device, dtype=torch.float32)
            states = _quantize_h_per_unit_mamba2(states, b_layer, r_layer)

    if _C5D["ssm_mode"] == "calibrate":
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

    if _C5D["ssm_mode"] == "calibrate":
        prev = _C5D["calib_accum"].get(layer_name)
        if prev is None:
            _C5D["calib_accum"][layer_name] = {
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


_orig_attn_forward = attn_class.forward


def _c5d_attention_forward(
    self, hidden_states, layer_idx,
    attention_mask=None, past_key_values=None,
    position_embeddings=None, **kwargs,
):
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
    if _C5D["attn_quant"]:
        key_states   = qa_nsep_pchan_asym(key_states,   _C5D["attn_bits"])
        value_states = qa_nsep_pchan_asym(value_states, _C5D["attn_bits"])
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


def _patch_ssm_on():     mixer_class.torch_forward = _c5d_torch_forward
def _patch_ssm_off():    mixer_class.torch_forward = _orig_torch_forward
def _patch_attn_on():    attn_class.forward = _c5d_attention_forward
def _patch_attn_off():   attn_class.forward = _orig_attn_forward


# =============================================================
# === CELL V4 === Calibration
# =============================================================

print(f"\n{'-' * 80}")
print(f"Calibrating ρ_h + Γ_(h,s) + RMS(state)")
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
_C5D["ssm_mode"]    = "calibrate"
_C5D["calib_accum"] = {}
with torch.inference_mode():
    _ = mdl(calib_ids, use_cache=False)
_C5D["ssm_mode"] = "off"
_patch_ssm_off()

captured = {}
for name, accum in _C5D["calib_accum"].items():
    rho_h = accum["rho_sum"]   / accum["rho_count"]
    gamma = accum["gamma_sum"] / accum["gamma_count"]
    rms   = (accum["rms_sumsq"] /
              accum["rms_count"]).clamp(min=1e-12).sqrt()
    captured[name] = {"rho_h": rho_h, "gamma": gamma, "rms_h": rms}
print(f"  captured for {len(captured)} layers")


# =============================================================
# === CELL V5 === Build `full` profiles for each b_0
# =============================================================

B_MIN     = 2
RANGE_K   = 7.0
B_0_LIST  = [3, 4, 5, 6, 8]


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


extras_full = _extras_full(captured)
profiles_by_b0 = {
    b_0: _build_b_per_unit(extras_full, b_0) for b_0 in B_0_LIST
}
r_per_layer = {n: RANGE_K * d["rms_h"] for n, d in captured.items()}


# =============================================================
# === CELL V6 === PPL eval
# =============================================================

def ppl_eval(model, ids, ssm_quant_b_dict, attn_quant,
              ctx_len=1024, stride=512, n_windows=5):
    if ssm_quant_b_dict is not None:
        _patch_ssm_on()
        _C5D["ssm_mode"]    = "quantise"
        _C5D["b_per_layer"] = ssm_quant_b_dict
        _C5D["r_per_layer"] = r_per_layer
    else:
        _C5D["ssm_mode"]    = "off"
        _patch_ssm_off()
    if attn_quant:
        _patch_attn_on()
        _C5D["attn_quant"]  = True
    else:
        _C5D["attn_quant"]  = False
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
        _C5D["ssm_mode"]    = "off"
        _C5D["attn_quant"]  = False
        _patch_ssm_off()
        _patch_attn_off()
    return math.exp(nll_total / tok_total), windows_done


# =============================================================
# === CELL V7 === Run sweep
# =============================================================

CTX_LEN   = 1024
STRIDE    = 512
N_WINDOWS = 20   # Bumped from 5 → 20 to settle the b_0=8 anomaly
                  # observed in the first run. With 5 windows the
                  # ±5 pp variance overshadows the genuine
                  # composition signal; 20 windows quadruples the
                  # token count (~20K tokens evaluated per condition)
                  # and should pin the interaction estimates within
                  # ±0.5 pp.

test_ids = tok.encode(test_text, return_tensors="pt").to("cuda")
print(f"\n{'-' * 80}")
print(f"Zamba2-1.2B PPL  (ctx={CTX_LEN}, n_windows={N_WINDOWS})")
print(f"  b_0 sweep: {B_0_LIST}")
print(f"  Attn:      modeq INT3 asym")
print(f"{'-' * 80}")

# 1. FP16 baseline (b_0-independent)
print(f"\n  FP16...")
fp16_ppl, _ = ppl_eval(mdl, test_ids, None, False,
                          ctx_len=CTX_LEN, stride=STRIDE,
                          n_windows=N_WINDOWS)
print(f"     ppl = {fp16_ppl:.4f}")

# 2. attn_only baseline (b_0-independent)
print(f"\n  attn_only (modeq INT3, SSM FP16)...")
attn_only_ppl, _ = ppl_eval(mdl, test_ids, None, True,
                                ctx_len=CTX_LEN, stride=STRIDE,
                                n_windows=N_WINDOWS)
attn_only_pct = 100.0 * (attn_only_ppl - fp16_ppl) / fp16_ppl
print(f"     ppl = {attn_only_ppl:.4f}  ΔPPL = "
      f"{attn_only_pct:+.3f}%")

# 3. Per-b_0: SSM_only and both
sweep_results = {}
for b_0 in B_0_LIST:
    print(f"\n  --- b_0 = {b_0} ---")
    b_dict = profiles_by_b0[b_0]
    flat = torch.cat([b.flatten() for b in b_dict.values()])
    mean_b = float(flat.float().mean())
    std_b  = float(flat.float().std())

    print(f"  SSM_only @ b_0={b_0}  (Paper C `full`, attn FP16)...")
    ssm_only_ppl, _ = ppl_eval(mdl, test_ids, b_dict, False,
                                  ctx_len=CTX_LEN, stride=STRIDE,
                                  n_windows=N_WINDOWS)
    ssm_only_pct = 100.0 * (ssm_only_ppl - fp16_ppl) / fp16_ppl
    print(f"     ppl = {ssm_only_ppl:.4f}  ΔPPL = "
          f"{ssm_only_pct:+.3f}%")

    print(f"  both     @ b_0={b_0}...")
    both_ppl, _ = ppl_eval(mdl, test_ids, b_dict, True,
                              ctx_len=CTX_LEN, stride=STRIDE,
                              n_windows=N_WINDOWS)
    both_pct = 100.0 * (both_ppl - fp16_ppl) / fp16_ppl
    expected_pct    = attn_only_pct + ssm_only_pct
    interaction_pp  = both_pct - expected_pct
    print(f"     ppl = {both_ppl:.4f}  ΔPPL = "
          f"{both_pct:+.3f}%  "
          f"(expected {expected_pct:+.3f}%, "
          f"interaction {interaction_pp:+.3f} pp)")

    sweep_results[str(b_0)] = {
        "b_0":            b_0,
        "mean_b":         mean_b,
        "std_b":          std_b,
        "ssm_only_ppl":   ssm_only_ppl,
        "ssm_only_pct":   ssm_only_pct,
        "both_ppl":       both_ppl,
        "both_pct":       both_pct,
        "expected_pct":   expected_pct,
        "interaction_pp": interaction_pp,
    }


# =============================================================
# === CELL V8 === Verdict + summary table
# =============================================================

ABS_THRESHOLD = 0.5

print(f"\n{'=' * 80}")
print(f"Phase C-5d sweep — composition diagnostic vs b_0")
print(f"{'=' * 80}")

print(f"\n  Reference:")
print(f"    FP16 PPL          = {fp16_ppl:>9.4f}")
print(f"    attn_only ΔPPL%   = {attn_only_pct:>+9.3f}  "
      f"(b_0-independent)")

print(f"\n  Sweep table (% of FP16):")
print(f"  {'b_0':>5s}  {'mean_b':>7s}  {'SSM_only':>10s}  "
      f"{'both':>10s}  {'expected':>10s}  "
      f"{'interaction':>14s}  {'class':>20s}")
print(f"  {'─' * 5}  {'─' * 7}  {'─' * 10}  {'─' * 10}  "
      f"{'─' * 10}  {'─' * 14}  {'─' * 20}")

for b_0 in B_0_LIST:
    r = sweep_results[str(b_0)]
    if abs(r["interaction_pp"]) < ABS_THRESHOLD:
        cls = "CLEAN"
    elif r["interaction_pp"] > 0:
        cls = "DEGRADATION"
    else:
        cls = "CANCELLATION"
    print(f"  {b_0:>5d}  {r['mean_b']:>7.2f}  "
          f"{r['ssm_only_pct']:>+9.3f}%  "
          f"{r['both_pct']:>+9.3f}%  "
          f"{r['expected_pct']:>+9.3f}%  "
          f"{r['interaction_pp']:>+12.3f} pp  "
          f"{cls:>20s}")

# Class stability
classes = []
for b_0 in B_0_LIST:
    r = sweep_results[str(b_0)]
    if abs(r["interaction_pp"]) < ABS_THRESHOLD:
        classes.append("CLEAN")
    elif r["interaction_pp"] > 0:
        classes.append("DEGRADATION")
    else:
        classes.append("CANCELLATION")

unique_classes = set(classes)
print(f"\n  Distinct classes across b_0 sweep: {sorted(unique_classes)}")

if len(unique_classes) == 1 and "CLEAN" in unique_classes:
    sweep_verdict = "CLEAN_AT_ALL_B_0"
    sweep_text = (f"Composition is additive at every b_0 ∈ "
                   f"{B_0_LIST}. The Paper B + Paper C combination "
                   f"is robust across the full SSM bit budget "
                   f"range, including the aggressive INT3 setting. "
                   f"This is paper-grade evidence for deployment-"
                   f"ready composition. Move to C-6.")
elif "DEGRADATION" not in unique_classes:
    sweep_verdict = "CLEAN_OR_BETTER"
    sweep_text = (f"No regime shows non-additive degradation. "
                   f"Some b_0 settings exhibit unexpected "
                   f"cancellation (slight free lunch). Worth "
                   f"investigating in paper but does not block "
                   f"C-6.")
elif len(unique_classes) > 1:
    sweep_verdict = "REGIME_DEPENDENT"
    sweep_text = (f"Composition class varies across b_0. "
                   f"Identify the boundary at which interaction "
                   f"becomes meaningful and pin the deployment "
                   f"recommendation accordingly.")
else:
    sweep_verdict = "DEGRADATION_AT_ALL_B_0"
    sweep_text = (f"Composition shows non-additive degradation at "
                   f"every b_0. Paper B + Paper C interaction is "
                   f"a real cost; report transparently and "
                   f"explore mitigation.")

print(f"\n  Sweep verdict: {sweep_verdict}")
print(f"  {sweep_text}")


# =============================================================
# === CELL V9 === Save
# =============================================================

# Per-layer summary (compact)
summary = []
for name, d in captured.items():
    summary.append({
        "layer_name":     name,
        "rho_h_p50":      float(d["rho_h"].quantile(0.50)),
        "gamma_p50":      float(d["gamma"].quantile(0.50)),
        "rms_h_p50":      float(d["rms_h"].quantile(0.50)),
    })

output = {
    "date":            datetime.utcnow().isoformat(timespec="seconds"),
    "phase":           "C-5d",
    "purpose":         ("Composition diagnostic with SSM b_0 "
                        "swept ∈ {3, 4, 5, 6, 8}. Verifies that "
                        "C-5c's CLEAN_COMPOSITION result at b_0=6 "
                        "is robust across the deployment-relevant "
                        "SSM bit range, including aggressive INT3."),
    "model":           MODEL_ID,
    "label":           MODEL_LABEL,
    "transformers_version": _hf_transformers.__version__,
    "protocol": {
        "ctx_len":      CTX_LEN,
        "stride":       STRIDE,
        "n_windows":    N_WINDOWS,
        "b_0_list":     B_0_LIST,
        "b_min":        B_MIN,
        "range_k":      RANGE_K,
        "attn_bits":    3,
        "calib_split":  "WikiText-2 train",
        "test_split":   "WikiText-2 test",
        "ssm_method":   "Paper C `full` chunk-boundary state quant",
        "attn_method":  "modeq qa_nsep_pchan_asym INT3",
    },
    "fp16_ppl":         fp16_ppl,
    "attn_only_ppl":    attn_only_ppl,
    "attn_only_pct":    attn_only_pct,
    "sweep_results":    sweep_results,
    "abs_threshold_pp": ABS_THRESHOLD,
    "sweep_verdict":    sweep_verdict,
    "sweep_text":       sweep_text,
    "rho_gamma_rms_summary": summary,
}

out_path = "/content/phase_c5d_composition_b0_sweep.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False, default=str)
print(f"\nSaved -> {out_path}")


# =============================================================
# === READING ===
# =============================================================
#
# Key questions the sweep answers:
#
# 1. Does CLEAN_COMPOSITION hold at the most aggressive
#    b_0 = 3 setting where SSM-side noise is large?
#       YES → composition is bulletproof for low-bit deployment
#       NO  → there is a low-bit regime where Paper B + C
#             interact non-additively
#
# 2. Does interaction → 0 at b_0 = 8 where SSM noise is tiny?
#       YES (expected)
#       NO  → odd, suggests something other than SSM noise
#             magnitude is driving interaction
#
# 3. Is the interaction sign consistent across b_0?
#       Same sign at all b_0 → systematic effect
#       Sign changes  → b_0-dependent regime shift
#
# Combined with C-4's Mamba1 deployment curve table, the
# Paper C manuscript can claim:
#     - Single-model: 1 - 2 bit savings (C-4)
#     - Hybrid composition: additive (C-5c, C-5d)
#     → multiplicative deployment story for Paper B + C
# ============================================================
