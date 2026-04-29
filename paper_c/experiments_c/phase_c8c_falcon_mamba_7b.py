# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Phase C-8c: Falcon-Mamba-7B cross-lineage generalisation
# ============================================================
# Paper-C-v1 generalisation test, Tier 2.
#
# C-8a established that the `full` allocator is rank-1 across
# the state-spaces Mamba-1 family at scales 130M → 2.8B
# (10/10 across (model, b_0)). C-8c tests whether the same
# property transfers to **a different Mamba-1 lineage**: TII's
# Falcon-Mamba-7B (tiiuae/falcon-mamba-7b), which is trained
# on a separate corpus and at a larger scale (7B vs 2.8B).
#
# If `full` is also rank-1 here, the bit allocator's
# generalisation claim broadens from "scale-invariant within
# the state-spaces release line" to "lineage-invariant within
# the Mamba-1 architecture family" — a much stronger claim
# for the paper §3 generalisation section.
#
# Falcon-Mamba uses the same Mamba-1 mathematical structure
# (selective scan with per-(channel, state) ρ from A_log; per-
# state Γ from C_proj). The HF integration may use a custom
# `FalconMambaModel` class but the inner mixer is structurally
# Mamba-1, so the C-8a quantiser and patched slow_forward
# transfer directly. Substrate detection uses the same
# A_log + dt_proj + x_proj fingerprint as C-8a.
#
# Protocol (matched to C-4 / C-8a):
#   - ctx_len = 1024, stride = 512, n_windows = 10
#   - range_k = 7.0, b_min = 2
#   - b_0 ∈ {4, 6}
#   - 5 profiles: uniform, proxy_C2, proxy_correct, full, anti_full
#
# Pre-registered hypothesis: `full` is rank-1 over uniform /
# proxy_C2 / anti_full at b_0 = 4 AND b_0 = 6.
#
# Runtime: ~25 - 35 min on A100 80 GB.
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
# === CELL V2 === Load Falcon-Mamba-7B
# =============================================================

MODEL_ID    = "tiiuae/falcon-mamba-7b"
MODEL_LABEL = "Falcon-Mamba-7B"

CTX_LEN   = 1024
STRIDE    = 512
N_WINDOWS = 10
B_0_LIST  = [4, 6]
B_MIN     = 2
RANGE_K   = 7.0
EPS       = 1e-6

PROFILE_LABELS = ["uniform", "proxy_C2", "proxy_correct",
                   "full", "anti_full"]

print(f"\n{'=' * 80}")
print(f"Loading {MODEL_LABEL}  ({MODEL_ID})")
print(f"{'=' * 80}")

# Tokenizer with fallback (Falcon-Mamba uses its own; if it
# differs in encoding from gpt-neox we still produce valid IDs
# for this specific model).
tok = None
for tid in [MODEL_ID, "EleutherAI/gpt-neox-20b"]:
    try:
        tok = AutoTokenizer.from_pretrained(tid, trust_remote_code=True)
        print(f"  Tokenizer:  {tid}")
        break
    except Exception:
        pass
assert tok is not None

mdl = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
mdl.eval()
print(f"  VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")


def find_mixers(model):
    """Mamba-1 fingerprint (A_log Parameter + dt_proj Linear +
    x_proj Linear). Works for state-spaces/mamba-*-hf, Falcon-
    Mamba, and any other Mamba-1 release that follows HF's
    layout."""
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
        "No Mamba-1 mixers detected. Falcon-Mamba may use a "
        "different attribute layout. Inspect "
        "list(mdl.named_modules())[:20] and update find_mixers."
    )

mixer_class = type(mixers[0][1])
print(f"  Mixer class:     "
      f"{mixer_class.__module__}.{mixer_class.__name__}")
ex_mix = mixers[0][1]
print(f"  Mixer geometry:  d_inner={ex_mix.intermediate_size}, "
      f"d_state={ex_mix.ssm_state_size}, "
      f"dt_rank={ex_mix.time_step_rank}")


# =============================================================
# === CELL V3 === Quantiser + patched slow_forward
#                 (identical to C-8a / C-4)
# =============================================================

_C8C = {
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
    """Falcon-Mamba's `rms_forward`: RMSNorm without learnable
    weight, applied along the last dim. Returns same dtype as
    input."""
    input_dtype = x.dtype
    x32 = x.to(torch.float32)
    variance = x32.pow(2).mean(-1, keepdim=True)
    x32 = x32 * torch.rsqrt(variance + eps)
    return x32.to(input_dtype)


def _c8c_mamba_forward(mix, input_states):
    """Mamba-1 reference forward path (verbatim from C-4 / C-8a)
    with optional per-(c, s) state-step quantisation."""
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

    # Falcon-Mamba-specific: apply rms_forward (RMSNorm without
    # learnable weight) to B, C, and dt_raw before discretisation.
    # state-spaces/mamba-*-hf does NOT have this step (no
    # `rms_eps` attribute), so the branch is opt-in by
    # attribute presence — keeps the function compatible across
    # the entire Mamba-1 family.
    if hasattr(mix, "rms_eps"):
        eps_rms = float(mix.rms_eps)
        B_proj  = _rms_norm_no_weight(B_proj,  eps_rms)
        C_proj  = _rms_norm_no_weight(C_proj,  eps_rms)
        dt_raw  = _rms_norm_no_weight(dt_raw,  eps_rms)

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
    if _C8C["mode"] == "quantise":
        b = _C8C["b_per_layer"].get(layer_name)
        r = _C8C["r_per_layer"].get(layer_name)
        if b is not None and r is not None:
            b_layer = b.to(x_pre.device)
            r_layer = r.to(x_pre.device, dtype=torch.float32)

    if _C8C["mode"] == "calibrate":
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
        if _C8C["mode"] == "calibrate":
            rms_sumsq += (h ** 2).sum(dim=0)
        y_t = (h * C_f[:, t].unsqueeze(1)).sum(dim=-1)
        ys.append(y_t)
    y = torch.stack(ys, dim=1)

    y = y + mix.D.float().unsqueeze(0).unsqueeze(0) * x_pre.float()
    y = y * F.silu(gate.float())
    out = mix.out_proj(y.to(mix.out_proj.weight.dtype))

    if _C8C["mode"] == "calibrate":
        prev = _C8C["calib_accum"].get(layer_name)
        if prev is None:
            _C8C["calib_accum"][layer_name] = {
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


def _c8c_slow_forward_patch(self, input_states, *args, **kwargs):
    return _c8c_mamba_forward(self, input_states)


def _force_slow_path(model):
    for _, mod in model.named_modules():
        for attr in ("use_fast_path", "_use_fast_path",
                      "use_mambapy", "_uses_cuda_kernels"):
            if hasattr(mod, attr):
                try:
                    setattr(mod, attr, False)
                except Exception:
                    pass


# Find which forward attribute the model uses (slow_forward
# preferred; fall back to torch_forward / forward).
forward_attr = None
for cand in ("slow_forward", "torch_forward", "reference_forward"):
    if hasattr(mixer_class, cand):
        forward_attr = cand
        break
if forward_attr is None:
    forward_attr = "forward"
print(f"  Patching attribute: {mixer_class.__name__}.{forward_attr}")

_orig_forward = getattr(mixer_class, forward_attr)
setattr(mixer_class, forward_attr, _c8c_slow_forward_patch)
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
print(f"Calibrating ρ_emp + Γ_s + RMS(h) on Falcon-Mamba-7B")
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

_C8C["mode"]        = "calibrate"
_C8C["calib_accum"] = {}
with torch.inference_mode():
    _ = mdl(calib_ids, use_cache=False)
_C8C["mode"] = "off"

captured = {}
for name, accum in _C8C["calib_accum"].items():
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
print(f"  example  rho p50 = {float(ex['rho_emp'].flatten().quantile(0.50)):.4f}, "
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
        _C8C["mode"] = "off"
        _C8C["b_per_layer"] = None
        _C8C["r_per_layer"] = None
    else:
        _C8C["mode"] = "quantise"
        _C8C["b_per_layer"] = b_per_layer
        _C8C["r_per_layer"] = r_per_layer
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
        _C8C["mode"] = "off"
        _C8C["b_per_layer"] = None
        _C8C["r_per_layer"] = None
    return math.exp(nll_total / tok_total), windows_done


# =============================================================
# === CELL V6 === Run sweep
# =============================================================

print(f"\n{'-' * 80}")
print(f"Falcon-Mamba-7B PPL  (ctx={CTX_LEN}, n_windows={N_WINDOWS}, "
      f"k={RANGE_K})")
print(f"  b_0 sweep: {B_0_LIST}")
print(f"{'-' * 80}")

# FP16 baseline
fp16_ppl, _ = ppl_eval(mdl, test_ids, None, None)
print(f"\n  FP16 PPL = {fp16_ppl:.4f}")

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
print(f"Phase C-8c verdict — Falcon-Mamba-7B cross-lineage check")
print(f"{'=' * 80}")

print(f"\n  FP16 PPL: {fp16_ppl:.4f}")
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
print(f"\n  `full` rank-1 check:")
all_full_wins = True
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

# Cross-lineage comparison (rough): same scale (~7B is between
# 2.8B and the next state-spaces would have been)
print(f"\n  full advantage vs uniform:")
for b_0 in B_0_LIST:
    f = sweep[b_0]["full"]["pct_of_base"]
    u = sweep[b_0]["uniform"]["pct_of_base"]
    ratio = u / max(f, EPS)
    print(f"    b_0={b_0}  uniform/full ratio = {ratio:.2f}×")

if all_full_wins:
    verdict_class = "CROSS_LINEAGE_OK"
    verdict = (f"`full` is rank-1 at b_0 ∈ {{4, 6}} on "
                f"Falcon-Mamba-7B (TII lineage). Combined with "
                f"C-8a (state-spaces lineage, 130M → 2.8B), the "
                f"bit allocator is empirically validated as "
                f"lineage-invariant within the Mamba-1 family. "
                f"Paper C's universal claim broadens from "
                f"'scale-invariant within state-spaces release' "
                f"to 'lineage-invariant within Mamba-1 "
                f"architecture'. C-8d (Jamba-1.5-Mini) is the "
                f"next step.")
else:
    verdict_class = "PARTIAL"
    verdict = (f"`full` does NOT win at every b_0 on Falcon-"
                f"Mamba-7B. The lineage-invariance claim needs "
                f"qualification. Inspect failed point above; "
                f"likely the calibrated (ρ, Γ, RMS) distribution "
                f"on TII's training corpus differs in a way that "
                f"makes the proxy_C2 or proxy_correct profile "
                f"competitive at this specific budget.")

print(f"\n  Verdict class: {verdict_class}")
print(f"  Verdict: {verdict}")


# =============================================================
# === CELL V8 === Save
# =============================================================

# Compact per-layer summary
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
    "phase":           "C-8c",
    "purpose":         ("Cross-lineage Mamba-1 generalisation. "
                        "Tests whether the `full` bit allocator's "
                        "rank-1 property transfers from the "
                        "state-spaces release line (C-8a, 130M - "
                        "2.8B) to TII's Falcon-Mamba-7B lineage."),
    "transformers_version": _hf_transformers.__version__,
    "model":           MODEL_ID,
    "label":           MODEL_LABEL,
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

out_path = "/content/phase_c8c_falcon_mamba_7b.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False, default=str)
print(f"\nSaved -> {out_path}")


# =============================================================
# === READING ===
# =============================================================
#
# CROSS_LINEAGE_OK (full rank-1 at both b_0):
#   The bit allocator generalises across Mamba-1 lineages.
#   Paper C's §3 generalisation section can claim "tested on
#   six Mamba-1 instances spanning two release lines and three
#   orders of magnitude in scale (130M state-spaces, 370M, 790M,
#   1.4B, 2.8B state-spaces, 7B Falcon)". Move to C-8d.
#
# PARTIAL:
#   Investigate the failed point. Falcon-Mamba was trained on a
#   different corpus (RefinedWeb + multi-source) which may
#   produce a different (ρ, Γ) distribution than state-spaces'
#   Pile-trained checkpoints. Most likely cause: the calibrated
#   gamma distribution is more uniform (less tail), making the
#   `full` allocator's Γ-aware advantage smaller. In that case
#   the rank may shift between full and proxy_correct at the
#   borderline b_0, but the universal property "structure-aware
#   beats uniform" should still hold.
# ============================================================
