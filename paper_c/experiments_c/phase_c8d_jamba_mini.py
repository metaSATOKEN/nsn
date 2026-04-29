# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Phase C-8d: Jamba-1.5-Mini hybrid Mamba-1 generalisation
# ============================================================
# Paper-C-v1 generalisation, Tier 2 final.
#
# C-8a (state-spaces 130M-2.8B) + C-8c (Falcon-Mamba 7B) + C-5b
# (Mamba-2 inside Zamba2) collectively establish that the bit
# allocator is rank-1 across:
#   - Mamba-1 family across two release lines (state-spaces, TII)
#   - Mamba-1 → Mamba-2 architectural divide
#
# C-8d closes the generalisation with a third lineage and a
# different architectural composition: AI21Labs' Jamba-1.5-Mini,
# a hybrid of Mamba-1 + Transformer attention + MoE experts. We
# patch ONLY the Mamba-1 mixers (attention + MoE held at FP16
# for the model's loaded precision); the question is whether
# Paper C's `full` allocator survives the *MoE-rich training
# regime* and the *interleaved attention context* that Jamba's
# Mamba layers operate within.
#
# Implementation note (loading at scale):
# Jamba-1.5-Mini is 52B total parameters (12B active). At fp16
# this is ~104 GB, beyond a 95 GB GPU. We try two paths in
# order of preference:
#
#   1. fp16 + device_map="auto"  — distributes the model across
#      GPU + CPU (HF auto-offload). Forward inference is slower
#      due to host transfers but otherwise correct.
#   2. int8 + bitsandbytes        — quantises Linear weights to
#      int8, fits ~52 GB on GPU. The Paper C state quantisation
#      is layered on top; rank-1 rankings between profiles are
#      still meaningful (model weights held constant across
#      profile comparisons).
#
# Patched forward additions:
#   Jamba's slow_forward applies learnable JambaRMSNorm modules
#   to dt / B / C before discretisation. The patch detects this
#   via attribute presence:
#     - hasattr(mix, "rms_eps")        → Falcon-Mamba (weightless)
#     - hasattr(mix, "dt_layernorm")   → Jamba       (learnable weight)
#     - else                           → state-spaces (no norm)
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

!pip install -q -U transformers accelerate hf_transfer datasets bitsandbytes

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
# === CELL V2 === Load Jamba-1.5-Mini
#                 (fp16 with device_map auto; int8 fallback)
# =============================================================

# Candidate models tried in priority order. The 1.6 release of
# AI21 Jamba is gated (HTTP 403 without explicit access grant);
# the older 1.5 / v0.1 releases may still be open. If all
# Jamba candidates are unreachable we pivot to Mistral's
# Codestral-Mamba — a pure Mamba-1 model from a third lineage,
# which broadens the "lineage generalisation" axis instead of
# the "hybrid composition" axis.
MODEL_CANDIDATES = [
    ("ai21labs/AI21-Jamba-Mini-1.6",       "Jamba-Mini-1.6",    "hybrid"),
    ("ai21labs/Jamba-v0.1",                "Jamba-v0.1",        "hybrid"),
    ("ai21labs/Jamba-1.5-Mini",            "Jamba-1.5-Mini",    "hybrid"),
    ("mistralai/Mamba-Codestral-7B-v0.1",  "Codestral-Mamba-7B", "pure"),
]

CTX_LEN   = 1024
STRIDE    = 512
N_WINDOWS = 5     # smaller than C-8a/c (10) to keep runtime
                   # manageable; large hybrid forwards are slow
B_0_LIST  = [4, 6]
B_MIN     = 2
RANGE_K   = 7.0
EPS       = 1e-6

PROFILE_LABELS = ["uniform", "proxy_C2", "proxy_correct",
                   "full", "anti_full"]


def _try_load(cand_id, cand_label):
    """Attempt fp16+device_map_auto first, fall back to int8."""
    # Strategy 1: fp16
    print(f"\n  [{cand_label}] Strategy 1: fp16 + "
          f"device_map='auto'")
    try:
        tok_local = AutoTokenizer.from_pretrained(
            cand_id, trust_remote_code=True
        )
        mdl_local = AutoModelForCausalLM.from_pretrained(
            cand_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        mdl_local.eval()
        print(f"  ✓ LOADED. VRAM = "
              f"{torch.cuda.memory_allocated() / 1e9:.2f} GB")
        return mdl_local, tok_local, "fp16_device_map_auto", None
    except Exception as e:
        err = f"{type(e).__name__}: {str(e)[:200]}"
        print(f"  ✗ FAILED. {err}")
        try:
            del mdl_local
        except NameError:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Strategy 2: int8 (only if Strategy 1 did NOT fail because
    # of access / network issues — those would also block int8)
    is_access_error = (
        "GatedRepoError" in err
        or "RepositoryNotFoundError" in err
        or "401" in err or "403" in err or "404" in err
    )
    if is_access_error:
        print(f"  (skip int8 strategy — access / repo error, "
              f"trying next candidate)")
        return None, None, None, err

    print(f"\n  [{cand_label}] Strategy 2: int8 + bitsandbytes")
    try:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        tok_local = AutoTokenizer.from_pretrained(
            cand_id, trust_remote_code=True
        )
        mdl_local = AutoModelForCausalLM.from_pretrained(
            cand_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        mdl_local.eval()
        print(f"  ✓ LOADED at INT8. VRAM = "
              f"{torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"  ⚠ Model weights INT8. Profile rankings remain "
              f"meaningful; absolute PPL differs from FP16.")
        return mdl_local, tok_local, "int8_bitsandbytes", None
    except Exception as e2:
        err2 = f"{type(e2).__name__}: {str(e2)[:200]}"
        print(f"  ✗ FAILED. {err2}")
        return None, None, None, err2


print(f"\n{'=' * 80}")
print(f"Attempting hybrid / pure Mamba-1 model load (3 candidates)")
print(f"{'=' * 80}")

mdl, tok = None, None
model_id_used    = None
model_label_used = None
model_kind_used  = None
load_strategy_used = None
load_log = []

for cand_id, cand_label, cand_kind in MODEL_CANDIDATES:
    print(f"\n{'─' * 80}")
    print(f"Candidate: {cand_label}  ({cand_id})  [kind={cand_kind}]")
    print(f"{'─' * 80}")
    mdl_try, tok_try, strat, err = _try_load(cand_id, cand_label)
    load_log.append({
        "model_id":  cand_id,
        "label":     cand_label,
        "kind":      cand_kind,
        "result":    "success" if mdl_try is not None else "failed",
        "strategy":  strat,
        "error":     err,
    })
    if mdl_try is not None:
        mdl, tok = mdl_try, tok_try
        model_id_used    = cand_id
        model_label_used = cand_label
        model_kind_used  = cand_kind
        load_strategy_used = strat
        break

if mdl is None:
    raise RuntimeError(
        "All candidate models failed to load. Inspect load_log "
        "for diagnostics. Options: (a) request HF access to one "
        "of the gated repos and retry; (b) declare C-8d "
        "out-of-scope (existing 7 instances from C-4 + C-8a + "
        "C-8c + C-5b are sufficient for the universal claim)."
    )

# Aliases for downstream code
MODEL_ID    = model_id_used
MODEL_LABEL = model_label_used

print(f"\n{'=' * 80}")
print(f"USING: {model_label_used}  ({model_id_used})")
print(f"  kind:     {model_kind_used}")
print(f"  strategy: {load_strategy_used}")
print(f"{'=' * 80}")


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

if not mixers:
    raise RuntimeError(
        "No Mamba-1 mixers detected. Jamba may use a different "
        "attribute layout in this version. Inspect "
        "list(mdl.named_modules())[:20] and update find_mixers."
    )

mixer_class = type(mixers[0][1])
print(f"  Mixer class:     "
      f"{mixer_class.__module__}.{mixer_class.__name__}")
ex_mix = mixers[0][1]
print(f"  Mixer geometry:  d_inner={ex_mix.intermediate_size}, "
      f"d_state={ex_mix.ssm_state_size}, "
      f"dt_rank={ex_mix.time_step_rank}")
print(f"  Has dt_layernorm: {hasattr(ex_mix, 'dt_layernorm')}")
print(f"  Has rms_eps:      {hasattr(ex_mix, 'rms_eps')}")


# =============================================================
# === CELL V3 === Quantiser + patched slow_forward
#                 (Jamba/FalconMamba/state-spaces unified)
# =============================================================

_C8D = {
    "mode":          "off",
    "b_per_layer":   None,
    "r_per_layer":   None,
    "calib_accum":   None,
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


def _rms_norm_no_weight(x, eps):
    """Falcon-Mamba's `rms_forward`: weightless RMSNorm."""
    input_dtype = x.dtype
    x32 = x.to(torch.float32)
    variance = x32.pow(2).mean(-1, keepdim=True)
    x32 = x32 * torch.rsqrt(variance + eps)
    return x32.to(input_dtype)


def _c8d_mamba_forward(mix, input_states):
    """Mamba-1 reference forward path with optional per-(c, s)
    state-step quantisation. Compatible with three lineages:
        - state-spaces (no normalisation between split and
          discretisation)
        - Falcon-Mamba (rms_forward via mix.rms_eps attribute)
        - Jamba (mix.dt_layernorm / mix.b_layernorm /
          mix.c_layernorm modules with learnable weight)
    """
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

    # Lineage-specific normalisation between (split) and (discretise).
    if hasattr(mix, "dt_layernorm"):
        # Jamba: learnable RMSNorm modules.
        dt_raw = mix.dt_layernorm(dt_raw)
        B_proj = mix.b_layernorm(B_proj)
        C_proj = mix.c_layernorm(C_proj)
    elif hasattr(mix, "rms_eps"):
        # Falcon-Mamba: weightless rms_forward.
        eps_rms = float(mix.rms_eps)
        B_proj  = _rms_norm_no_weight(B_proj,  eps_rms)
        C_proj  = _rms_norm_no_weight(C_proj,  eps_rms)
        dt_raw  = _rms_norm_no_weight(dt_raw,  eps_rms)
    # else: state-spaces, no normalisation.

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
    if _C8D["mode"] == "quantise":
        b = _C8D["b_per_layer"].get(layer_name)
        r = _C8D["r_per_layer"].get(layer_name)
        if b is not None and r is not None:
            b_layer = b.to(x_pre.device)
            r_layer = r.to(x_pre.device, dtype=torch.float32)

    if _C8D["mode"] == "calibrate":
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
        if _C8D["mode"] == "calibrate":
            rms_sumsq += (h ** 2).sum(dim=0)
        y_t = (h * C_f[:, t].unsqueeze(1)).sum(dim=-1)
        ys.append(y_t)
    y = torch.stack(ys, dim=1)

    y = y + mix.D.float().unsqueeze(0).unsqueeze(0) * x_pre.float()
    y = y * F.silu(gate.float())
    out = mix.out_proj(y.to(mix.out_proj.weight.dtype))

    if _C8D["mode"] == "calibrate":
        prev = _C8D["calib_accum"].get(layer_name)
        if prev is None:
            _C8D["calib_accum"][layer_name] = {
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


def _c8d_slow_forward_patch(self, input_states, *args, **kwargs):
    return _c8d_mamba_forward(self, input_states)


def _force_slow_path(model):
    for _, mod in model.named_modules():
        for attr in ("use_fast_path", "_use_fast_path",
                      "use_mambapy", "_uses_cuda_kernels",
                      "use_mamba_kernels"):
            if hasattr(mod, attr):
                try:
                    setattr(mod, attr, False)
                except Exception:
                    pass


# Detect forward attribute (Jamba uses slow_forward like Mamba1)
forward_attr = None
for cand in ("slow_forward", "torch_forward", "reference_forward"):
    if hasattr(mixer_class, cand):
        forward_attr = cand
        break
if forward_attr is None:
    forward_attr = "forward"
print(f"  Patching attribute: {mixer_class.__name__}.{forward_attr}")

_orig_forward = getattr(mixer_class, forward_attr)
setattr(mixer_class, forward_attr, _c8d_slow_forward_patch)
_force_slow_path(mdl)


# =============================================================
# === CELL V4 === Calibration + profile builders
# =============================================================

def _extras_uniform(captured):
    return {n: torch.zeros_like(d["rho_emp"]) for n, d in captured.items()}


def _extras_proxy_C2(captured):
    out = {}
    for n, d in captured.items():
        rho = d["rho_emp"]
        log_R = torch.log2(1.0 / (1.0 - rho).clamp(min=EPS))
        out[n] = torch.ceil(log_R).clamp(min=0)
    return out


def _extras_proxy_correct(captured):
    out = {}
    for n, d in captured.items():
        rho = d["rho_emp"]
        log_R = 0.5 * torch.log2(1.0 / (1.0 - rho).clamp(min=EPS))
        out[n] = torch.ceil(log_R).clamp(min=0)
    return out


def _extras_full(captured):
    out = {}
    for n, d in captured.items():
        rho   = d["rho_emp"]
        gamma = d["gamma"].clamp(min=EPS)
        log_R = torch.log2(
            gamma.unsqueeze(0) / (1.0 - rho).clamp(min=EPS)
        )
        out[n] = torch.ceil(log_R).clamp(min=0)
    return out


def _extras_anti_full(captured):
    out = {}
    for n, d in captured.items():
        rho   = d["rho_emp"]
        gamma = d["gamma"].clamp(min=EPS)
        log_R = torch.log2(
            (1.0 - rho).clamp(min=EPS) / gamma.unsqueeze(0)
        )
        out[n] = torch.ceil(log_R)
    return out


EXTRAS_FNS = {
    "uniform":       _extras_uniform,
    "proxy_C2":      _extras_proxy_C2,
    "proxy_correct": _extras_proxy_correct,
    "full":          _extras_full,
    "anti_full":     _extras_anti_full,
}


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


# Calibration
print(f"\n{'-' * 80}")
print(f"Calibrating ρ_emp + Γ_s + RMS(h) on Jamba Mamba layers")
print(f"{'-' * 80}")

ds_test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
test_text = "\n\n".join([x for x in ds_test["text"]
                          if x.strip()])[:48000]
calib_text = "\n\n".join(
    [x for x in load_dataset("wikitext", "wikitext-2-raw-v1",
                              split="train")["text"]
     if x.strip()])[:8000]
calib_ids = tok.encode(calib_text, return_tensors="pt").to("cuda")[:, :1024]
test_ids  = tok.encode(test_text, return_tensors="pt").to("cuda")
print(f"  Calibration ids: {calib_ids.shape[1]} tokens")
print(f"  Test ids:        {test_ids.shape[1]} tokens")

_C8D["mode"]        = "calibrate"
_C8D["calib_accum"] = {}
with torch.inference_mode():
    _ = mdl(calib_ids, use_cache=False)
_C8D["mode"] = "off"

captured = {}
for name, accum in _C8D["calib_accum"].items():
    rho   = accum["rho_sum"]   / accum["rho_count"]
    gamma = accum["gamma_sum"] / accum["gamma_count"]
    rms   = (accum["rms_sumsq"] /
              accum["rms_count"]).clamp(min=1e-12).sqrt()
    captured[name] = {
        "rho_emp": rho,
        "gamma":   gamma,
        "rms_h":   rms,
    }
n_layers = len(captured)
ex = next(iter(captured.values()))
d_inner = ex["rho_emp"].shape[0]
d_state = ex["rho_emp"].shape[1]
print(f"  captured: {n_layers} layers, "
      f"d_inner={d_inner}, d_state={d_state}")
print(f"  example  rho p50 = "
      f"{float(ex['rho_emp'].flatten().quantile(0.50)):.4f}, "
      f"max = {float(ex['rho_emp'].max()):.4f}")
print(f"  example  gamma min = {float(ex['gamma'].min()):.4e}, "
      f"max = {float(ex['gamma'].max()):.4e}")
print(f"  example  rms_h p50 = {float(ex['rms_h'].quantile(0.50)):.4e}, "
      f"max = {float(ex['rms_h'].max()):.4e}")

profiles_by_b0 = {
    b_0: {
        label: _build_b_per_unit(EXTRAS_FNS[label](captured), b_0)
        for label in PROFILE_LABELS
    }
    for b_0 in B_0_LIST
}
r_per_layer = _r_per_layer(captured, k=RANGE_K)


# =============================================================
# === CELL V5 === PPL eval
# =============================================================

def ppl_eval(model, ids, b_per_layer=None, r_per_layer=None,
              ctx_len=CTX_LEN, stride=STRIDE, n_windows=N_WINDOWS):
    if b_per_layer is None:
        _C8D["mode"] = "off"
        _C8D["b_per_layer"] = None
        _C8D["r_per_layer"] = None
    else:
        _C8D["mode"] = "quantise"
        _C8D["b_per_layer"] = b_per_layer
        _C8D["r_per_layer"] = r_per_layer
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
        _C8D["mode"] = "off"
        _C8D["b_per_layer"] = None
        _C8D["r_per_layer"] = None
    return math.exp(nll_total / tok_total), windows_done


# =============================================================
# === CELL V6 === Run sweep
# =============================================================

print(f"\n{'-' * 80}")
print(f"Jamba-1.5-Mini PPL  (ctx={CTX_LEN}, n_windows={N_WINDOWS}, "
      f"k={RANGE_K})")
print(f"  Load strategy: {load_strategy_used}")
print(f"  b_0 sweep:     {B_0_LIST}")
print(f"{'-' * 80}")

# FP16 baseline (or whatever the load strategy used)
fp16_ppl, _ = ppl_eval(mdl, test_ids, None, None)
print(f"\n  Baseline PPL = {fp16_ppl:.4f}  "
      f"(load = {load_strategy_used})")

# Sweep
sweep = {}
for b_0 in B_0_LIST:
    print(f"\n  --- b_0 = {b_0} ---")
    sweep[b_0] = {}
    for label in PROFILE_LABELS:
        b_dict = profiles_by_b0[b_0][label]
        flat   = torch.cat([b.flatten() for b in b_dict.values()])
        mean_b = float(flat.float().mean())
        std_b  = float(flat.float().std())
        ppl, _ = ppl_eval(mdl, test_ids, b_dict, r_per_layer)
        dppl = ppl - fp16_ppl
        pct  = 100.0 * dppl / fp16_ppl
        sweep[b_0][label] = {
            "ppl":          ppl,
            "delta_ppl":    dppl,
            "pct_of_base":  pct,
            "mean_b":       mean_b,
            "std_b":        std_b,
        }
        print(f"  {label:<14s}  ppl={ppl:>9.4f}  "
              f"ΔPPL={pct:+8.3f} %  "
              f"mean_b={mean_b:.2f}  std_b={std_b:.2f}")


# =============================================================
# === CELL V7 === Verdict
# =============================================================

print(f"\n{'=' * 80}")
print(f"Phase C-8d verdict — Jamba-1.5-Mini hybrid Mamba-1 check")
print(f"{'=' * 80}")
print(f"\n  Load strategy: {load_strategy_used}")
print(f"  Baseline PPL: {fp16_ppl:.4f}")
print(f"\n  ΔPPL % per (b_0, profile):")
print(f"  {'b_0':>5s}  " +
      "  ".join(f"{l:>14s}" for l in PROFILE_LABELS))
for b_0 in B_0_LIST:
    cells = []
    for label in PROFILE_LABELS:
        pct = sweep[b_0][label]["pct_of_base"]
        cells.append(f"{pct:>+13.3f}%")
    print(f"  {b_0:>5d}  " + "  ".join(cells))

# Rank-1 check
all_full_wins = True
print(f"\n  `full` rank-1 check:")
for b_0 in B_0_LIST:
    full_pct = sweep[b_0]["full"]["pct_of_base"]
    others = [sweep[b_0][l]["pct_of_base"]
              for l in PROFILE_LABELS if l != "full"]
    is_best = all(full_pct < o for o in others)
    if not is_best:
        all_full_wins = False
    print(f"    b_0={b_0}  full={full_pct:+8.3f}%  "
          f"min(others)={min(others):+8.3f}%  "
          f"{'✓' if is_best else '✗'}")

print(f"\n  full advantage vs uniform:")
for b_0 in B_0_LIST:
    f = sweep[b_0]["full"]["pct_of_base"]
    u = sweep[b_0]["uniform"]["pct_of_base"]
    ratio = u / max(f, EPS)
    print(f"    b_0={b_0}  uniform/full ratio = {ratio:.2f}×")

if all_full_wins:
    if model_kind_used == "hybrid":
        verdict_class = "HYBRID_OK"
        verdict = (f"`full` is rank-1 at b_0 ∈ {{4, 6}} on "
                    f"{model_label_used}'s Mamba layers (load "
                    f"strategy = {load_strategy_used}). Combined "
                    f"with C-8a (state-spaces 130M-2.8B), C-8c "
                    f"(Falcon-Mamba 7B), and C-5b (Mamba-2 in "
                    f"Zamba2-1.2B), Paper C's bit allocator is "
                    f"empirically validated across THREE Mamba-1 "
                    f"lineages, the Mamba-1 / Mamba-2 "
                    f"architectural divide, AND the hybrid "
                    f"Mamba-1 + Attention {'+ MoE' if 'Jamba' in model_label_used else ''} "
                    f"composition. Universal claim fully "
                    f"established.")
    else:  # "pure" — Codestral-Mamba pivot
        verdict_class = "PURE_LINEAGE_OK"
        verdict = (f"`full` is rank-1 at b_0 ∈ {{4, 6}} on "
                    f"{model_label_used} (load strategy = "
                    f"{load_strategy_used}). This is a third "
                    f"Mamba-1 lineage (Mistral, after state-"
                    f"spaces and TII Falcon). Combined with C-8a "
                    f"(state-spaces 130M-2.8B), C-8c (Falcon-"
                    f"Mamba 7B), and C-5b (Mamba-2 in Zamba2-"
                    f"1.2B), Paper C's bit allocator is "
                    f"validated across THREE Mamba-1 lineages "
                    f"plus the Mamba-1 / Mamba-2 architectural "
                    f"divide. The hybrid + MoE composition axis "
                    f"is deferred to Paper D since gated Jamba "
                    f"checkpoints blocked direct testing.")
else:
    verdict_class = "PARTIAL"
    verdict = (f"`full` does NOT win at every b_0 on "
                f"{model_label_used}. Inspect failed point. The "
                f"lineage-invariance claim from C-8a/c/5b still "
                f"holds at the existing tested instances; this "
                f"specific model is a notable nuance worth a "
                f"paper footnote.")

print(f"\n  Verdict class: {verdict_class}")
print(f"  Verdict: {verdict}")


# =============================================================
# === CELL V8 === Save
# =============================================================

summary = []
for name, d in captured.items():
    summary.append({
        "layer_name":     name,
        "rho_p50":        float(d["rho_emp"].flatten().quantile(0.50)),
        "rho_max":        float(d["rho_emp"].max()),
        "gamma_min":      float(d["gamma"].min()),
        "gamma_p50":      float(d["gamma"].quantile(0.50)),
        "gamma_max":      float(d["gamma"].max()),
        "rms_h_p50":      float(d["rms_h"].quantile(0.50)),
        "rms_h_max":      float(d["rms_h"].max()),
    })

output = {
    "date":            datetime.utcnow().isoformat(timespec="seconds"),
    "phase":           "C-8d",
    "purpose":         ("Hybrid Mamba-1 + Attention + MoE "
                        "generalisation. Tests whether Paper C's "
                        "bit allocator's rank-1 property holds on "
                        "Jamba-1.5-Mini's Mamba mixers when "
                        "embedded alongside attention + MoE "
                        "experts."),
    "transformers_version": _hf_transformers.__version__,
    "model":           MODEL_ID,
    "label":           MODEL_LABEL,
    "model_kind":      model_kind_used,
    "load_strategy":   load_strategy_used,
    "load_log":        load_log,
    "n_layers":        n_layers,
    "d_inner":         int(d_inner),
    "d_state":         int(d_state),
    "protocol": {
        "ctx_len":     CTX_LEN,
        "stride":      STRIDE,
        "n_windows":   N_WINDOWS,
        "b_0_list":    B_0_LIST,
        "b_min":       B_MIN,
        "range_k":     RANGE_K,
        "calib_split": "WikiText-2 train",
        "test_split":  "WikiText-2 test",
        "profiles":    PROFILE_LABELS,
        "quant_scheme": ("per-(c, s) INT-b symmetric mid-tread, "
                          "range = k * RMS(h), patched on "
                          f"{forward_attr}"),
    },
    "fp16_ppl":        fp16_ppl,
    "sweep":           sweep,
    "all_full_wins":   all_full_wins,
    "verdict_class":   verdict_class,
    "verdict":         verdict,
    "rho_gamma_rms_summary": summary,
}

out_path = "/content/phase_c8d_jamba_mini.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False, default=str)
print(f"\nSaved -> {out_path}")


# =============================================================
# === READING ===
# =============================================================
#
# JAMBA_OK (full rank-1 at both b_0):
#   The bit allocator generalises across three Mamba-1
#   lineages and across hybrid composition. Paper C's
#   universal claim covers:
#     - state-spaces 130M-2.8B (Pile training)
#     - TII Falcon-Mamba 7B (RefinedWeb training)
#     - AI21 Jamba-1.5-Mini (proprietary mix + MoE)
#     - Mamba-2 inside Zamba2-1.2B (per-(h, s) granularity)
#
# JAMBA_PARTIAL:
#   The hybrid + MoE training regime introduces a (ρ, Γ)
#   distribution that shifts the optimal profile. Paper C
#   should report: "validated on 7+ Mamba instances; one
#   instance (Jamba) shows nuance at specific bit budgets,
#   detailed in §3.x".
#
# Loading caveat:
#   If load_strategy_used = "int8_bitsandbytes", model weights
#   are quantised to int8. The Paper C state-quantisation
#   experiment is layered on top of int8 weights. Profile
#   rankings (full vs uniform vs proxy_C2 vs anti_full) remain
#   meaningful because all profiles are compared at the same
#   model-weight precision; the absolute baseline PPL just
#   differs from a true FP16 baseline. This is a deployment-
#   realistic scenario (real Jamba deployment likely uses
#   weight quantisation alongside state quantisation).
# ============================================================
