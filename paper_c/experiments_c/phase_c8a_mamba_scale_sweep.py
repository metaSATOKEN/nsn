# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Phase C-8a: Mamba family scale generalisation
# ============================================================
# Paper-C-v1 generalisation test. C-1 → C-4 validated the bit
# allocator on `state-spaces/mamba-2.8b-hf` only. C-8a sweeps
# the full Mamba-1 family (130M / 370M / 790M / 1.4B) at the
# C-4 deployment-realistic bit budgets b_0 ∈ {4, 6} to
# establish that the `full` allocator's rank-1 property is
# scale-invariant within a fixed architecture family.
#
# Together with C-8b (Mamba2 generalisation, reframed from
# C-5b's already-acquired data), this makes the Paper C
# universal claim:
#
#   "The bit allocator R_g = Γ_s / (1 - ρ_g) holds across the
#    Mamba family at scales 130M to 2.8B (Mamba-1) and across
#    the Mamba-1 / Mamba-2 architectural divide."
#
# Protocol (matched to C-4 for direct comparability):
#   - ctx_len = 1024, stride = 512, n_windows = 10
#   - range_k = 7.0  (post-C-4 saturation point)
#   - b_min = 2      (binary floor avoidance)
#   - b_0 ∈ {4, 6}   (deployment-realistic INT4 / INT6)
#   - 5 profiles: uniform, proxy_C2, proxy_correct, full, anti_full
#
# Mamba-2.8B-hf is NOT re-run here; its data lives in
# `phase_c4_state_int_quantisation.json` and is referenced for
# the aggregate summary at the end.
#
# Pre-registered hypothesis:
#   `full` is rank-1 over uniform / proxy_C2 / anti_full at
#   every (model, b_0) pair tested. If this holds for all 4
#   smaller models AND we can reference C-4 for the 2.8B point,
#   the bit allocator's scale-invariance property is empirically
#   established.
#
# Runtime estimate (A100 80 GB):
#     130M ~5 min   |  370M ~12 min  |  790M ~22 min
#     1.4B ~38 min  |                  total ~80 min
# (smaller models faster due to fewer layers + smaller intermediate_size)
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
# === CELL V2 === Models and protocol
# =============================================================

MODELS_TO_SWEEP = [
    "state-spaces/mamba-130m-hf",
    "state-spaces/mamba-370m-hf",
    "state-spaces/mamba-790m-hf",
    "state-spaces/mamba-1.4b-hf",
    # mamba-2.8b-hf NOT re-run; reuses Phase C-4 data
]
TOKENIZER_FALLBACKS = ["EleutherAI/gpt-neox-20b"]

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
print(f"Mamba family scale sweep: {len(MODELS_TO_SWEEP)} models × "
      f"{len(B_0_LIST)} b_0 × {len(PROFILE_LABELS)} profiles "
      f"+ {len(MODELS_TO_SWEEP)} FP16 baselines")
print(f"  Total evaluations: "
      f"{len(MODELS_TO_SWEEP) * (len(B_0_LIST) * len(PROFILE_LABELS) + 1)}")
print(f"  Protocol: ctx={CTX_LEN}, stride={STRIDE}, "
      f"n_windows={N_WINDOWS}, range_k={RANGE_K}")
print(f"{'=' * 80}")


# =============================================================
# === CELL V3 === Quantiser + patched slow_forward (Mamba1)
# =============================================================
#
# Bit-for-bit copy of the Phase C-4 patch (Mamba1 family). The
# math is the same across all Mamba-{130M, 370M, 790M, 1.4B,
# 2.8B}-hf checkpoints since they share `MambaMixer` with
# identical attribute conventions (A_log, dt_proj, x_proj,
# in_proj, conv1d, D, out_proj).

_C8A = {
    "mode":          "off",
    "b_per_layer":   None,
    "r_per_layer":   None,
    "calib_accum":   None,
    "layer_name_by_id": None,
}


def _quantize_h_per_unit(h, b_per_unit, r_per_unit):
    """Per-(c, s) INT-b symmetric mid-tread quantiser with zero
    level. Identical to C-4."""
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


def _c8a_mamba_forward(mix, input_states):
    """C-4's `_c4_mamba_forward` — verbatim. Mamba1 reference
    forward path with optional per-(c, s) quantisation
    post-state-update."""
    layer_name = _C8A["layer_name_by_id"].get(id(mix))
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
    if _C8A["mode"] == "quantise":
        b = _C8A["b_per_layer"].get(layer_name)
        r = _C8A["r_per_layer"].get(layer_name)
        if b is not None and r is not None:
            b_layer = b.to(x_pre.device)
            r_layer = r.to(x_pre.device, dtype=torch.float32)

    if _C8A["mode"] == "calibrate":
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
        if _C8A["mode"] == "calibrate":
            rms_sumsq += (h ** 2).sum(dim=0)
        y_t = (h * C_f[:, t].unsqueeze(1)).sum(dim=-1)
        ys.append(y_t)
    y = torch.stack(ys, dim=1)

    y = y + mix.D.float().unsqueeze(0).unsqueeze(0) * x_pre.float()
    y = y * F.silu(gate.float())
    out = mix.out_proj(y.to(mix.out_proj.weight.dtype))

    if _C8A["mode"] == "calibrate":
        prev = _C8A["calib_accum"].get(layer_name)
        if prev is None:
            _C8A["calib_accum"][layer_name] = {
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


def _c8a_slow_forward_patch(self, input_states, *args, **kwargs):
    return _c8a_mamba_forward(self, input_states)


# =============================================================
# === CELL V4 === Profile builders (per-(c, s), Mamba1)
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


def build_all_profiles(captured, b_0_list):
    profiles_by_b0 = {}
    for b_0 in b_0_list:
        profiles_by_b0[b_0] = {
            label: _build_b_per_unit(EXTRAS_FNS[label](captured), b_0)
            for label in PROFILE_LABELS
        }
    return profiles_by_b0


# =============================================================
# === CELL V5 === Per-model worker
# =============================================================

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


def _force_slow_path(model):
    for _, mod in model.named_modules():
        for attr in ("use_fast_path", "_use_fast_path",
                      "use_mambapy", "_uses_cuda_kernels"):
            if hasattr(mod, attr):
                try:
                    setattr(mod, attr, False)
                except Exception:
                    pass


def calibrate_model(mdl, calib_ids):
    _C8A["mode"]        = "calibrate"
    _C8A["calib_accum"] = {}
    with torch.inference_mode():
        _ = mdl(calib_ids, use_cache=False)
    _C8A["mode"] = "off"
    captured = {}
    for name, accum in _C8A["calib_accum"].items():
        rho   = accum["rho_sum"]   / accum["rho_count"]
        gamma = accum["gamma_sum"] / accum["gamma_count"]
        rms   = (accum["rms_sumsq"] /
                  accum["rms_count"]).clamp(min=1e-12).sqrt()
        captured[name] = {
            "rho_emp": rho,
            "gamma":   gamma,
            "rms_h":   rms,
        }
    return captured


def ppl_eval(model, ids, b_per_layer=None, r_per_layer=None,
              ctx_len=CTX_LEN, stride=STRIDE, n_windows=N_WINDOWS):
    if b_per_layer is None:
        _C8A["mode"] = "off"
        _C8A["b_per_layer"] = None
        _C8A["r_per_layer"] = None
    else:
        _C8A["mode"] = "quantise"
        _C8A["b_per_layer"] = b_per_layer
        _C8A["r_per_layer"] = r_per_layer
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
        _C8A["mode"] = "off"
        _C8A["b_per_layer"] = None
        _C8A["r_per_layer"] = None
    return math.exp(nll_total / tok_total), windows_done


def run_one_model(model_id, calib_ids, test_ids):
    """Load model, calibrate, sweep all (b_0, profile), unload."""
    print(f"\n{'─' * 80}")
    print(f"Model: {model_id}")
    print(f"{'─' * 80}")

    # Tokenizer (model-specific or fallback)
    tok_local = None
    for tid in [model_id] + TOKENIZER_FALLBACKS:
        try:
            tok_local = AutoTokenizer.from_pretrained(
                tid, trust_remote_code=True
            )
            break
        except Exception:
            pass
    assert tok_local is not None

    mdl_local = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16,
        device_map="auto", trust_remote_code=True,
    )
    mdl_local.eval()
    print(f"  VRAM after load: "
          f"{torch.cuda.memory_allocated() / 1e9:.2f} GB")

    mixers = find_mixers(mdl_local)
    print(f"  Found {len(mixers)} mixers")
    _C8A["layer_name_by_id"] = {id(mix): name for name, mix in mixers}
    mixer_class = type(mixers[0][1])

    # Patch
    orig_forward = mixer_class.slow_forward
    mixer_class.slow_forward = _c8a_slow_forward_patch
    _force_slow_path(mdl_local)

    try:
        # Calibrate
        captured = calibrate_model(mdl_local, calib_ids)
        n_layers = len(captured)
        ex = next(iter(captured.values()))
        d_inner = ex["rho_emp"].shape[0]
        d_state = ex["rho_emp"].shape[1]
        print(f"  calibrated: {n_layers} layers, "
              f"d_inner={d_inner}, d_state={d_state}")

        # Build profiles
        profiles_by_b0 = build_all_profiles(captured, B_0_LIST)
        r_per_layer = _r_per_layer(captured, k=RANGE_K)

        # FP16 baseline
        fp16_ppl, n_win = ppl_eval(mdl_local, test_ids,
                                       None, None)
        print(f"  FP16 ppl = {fp16_ppl:.4f}")

        # Sweep
        sweep = {}
        for b_0 in B_0_LIST:
            sweep[b_0] = {}
            for label in PROFILE_LABELS:
                b_dict = profiles_by_b0[b_0][label]
                flat   = torch.cat([b.flatten() for b in b_dict.values()])
                mean_b = float(flat.float().mean())
                std_b  = float(flat.float().std())
                ppl, _ = ppl_eval(mdl_local, test_ids,
                                   b_dict, r_per_layer)
                dppl = ppl - fp16_ppl
                pct  = 100.0 * dppl / fp16_ppl
                sweep[b_0][label] = {
                    "ppl":           ppl,
                    "delta_ppl":     dppl,
                    "pct_of_base":   pct,
                    "mean_b":        mean_b,
                    "std_b":         std_b,
                }
                print(f"  b_0={b_0}  {label:<14s}  "
                      f"ppl={ppl:>9.4f}  ΔPPL={pct:+8.3f} %  "
                      f"mean_b={mean_b:.2f}  std_b={std_b:.2f}")

        result = {
            "model_id":  model_id,
            "n_layers":  n_layers,
            "d_inner":   int(d_inner),
            "d_state":   int(d_state),
            "fp16_ppl":  fp16_ppl,
            "sweep":     sweep,
        }
    finally:
        mixer_class.slow_forward = orig_forward
        _C8A["layer_name_by_id"] = None
        del mdl_local, mixers
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return result


# =============================================================
# === CELL V6 === Run sweep
# =============================================================

# Calibration / test text (loaded once, reused across models)
ds_test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
test_text = "\n\n".join([x for x in ds_test["text"]
                          if x.strip()])[:48000]
calib_text = "\n\n".join(
    [x for x in load_dataset("wikitext", "wikitext-2-raw-v1",
                              split="train")["text"]
     if x.strip()])[:8000]

# Use a single tokenizer for encoding text into ids (the model
# tokenizers will all use the same gpt-neox vocabulary, but we
# encode once with the first available tokenizer).
ref_tok = None
for tid in [MODELS_TO_SWEEP[0]] + TOKENIZER_FALLBACKS:
    try:
        ref_tok = AutoTokenizer.from_pretrained(
            tid, trust_remote_code=True
        )
        break
    except Exception:
        pass
assert ref_tok is not None
calib_ids = ref_tok.encode(calib_text,
                              return_tensors="pt").to("cuda")[:, :1024]
test_ids  = ref_tok.encode(test_text,
                              return_tensors="pt").to("cuda")
print(f"\nCalibration ids: {calib_ids.shape[1]} tokens")
print(f"Test ids:        {test_ids.shape[1]} tokens")

aggregate_results = {}
for model_id in MODELS_TO_SWEEP:
    aggregate_results[model_id] = run_one_model(
        model_id, calib_ids, test_ids
    )


# =============================================================
# === CELL V7 === Inject Phase C-4 (Mamba-2.8B-hf) data point
# =============================================================
#
# Phase C-4's k=7 sweep already includes b_0 ∈ {3, 4, 5, 6, 8}
# at the same protocol. We extract b_0 ∈ {4, 6} and merge into
# the aggregate table for a single 5-model figure.

C4_PHASE_DATA = {
    "model_id":  "state-spaces/mamba-2.8b-hf",
    "n_layers":  64,
    "d_inner":   5120,
    "d_state":   16,
    "fp16_ppl":  12.3730,
    "sweep": {
        4: {
            "uniform":       {"pct_of_base": +221.299, "mean_b": 4.00, "std_b": 0.00,  "ppl": None},
            "proxy_C2":      {"pct_of_base": +301.815, "mean_b": 3.88, "std_b": 2.92,  "ppl": None},
            "proxy_correct": {"pct_of_base":  +73.776, "mean_b": 4.10, "std_b": 1.60,  "ppl": None},
            "full":          {"pct_of_base":  +26.582, "mean_b": 4.11, "std_b": 3.04,  "ppl": None},
            "anti_full":     {"pct_of_base":+1440.416, "mean_b": 5.15, "std_b": 2.37,  "ppl": None},
        },
        6: {
            "uniform":       {"pct_of_base":  +18.681, "mean_b": 6.00, "std_b": 0.00,  "ppl": None},
            "proxy_C2":      {"pct_of_base":  +31.309, "mean_b": 5.59, "std_b": 2.99,  "ppl": None},
            "proxy_correct": {"pct_of_base":   +8.319, "mean_b": 6.10, "std_b": 1.60,  "ppl": None},
            "full":          {"pct_of_base":   +6.694, "mean_b": 5.56, "std_b": 3.34,  "ppl": None},
            "anti_full":     {"pct_of_base": +164.576, "mean_b": 6.86, "std_b": 2.81,  "ppl": None},
        },
    },
    "source": "phase_c4_state_int_quantisation.json",
}
aggregate_results["state-spaces/mamba-2.8b-hf"] = C4_PHASE_DATA


# =============================================================
# === CELL V8 === Aggregate summary + verdict
# =============================================================

print(f"\n{'=' * 80}")
print(f"Phase C-8a aggregate sweep — Mamba family scale generalisation")
print(f"{'=' * 80}")

ALL_MODELS = MODELS_TO_SWEEP + ["state-spaces/mamba-2.8b-hf"]

for b_0 in B_0_LIST:
    print(f"\n  ── b_0 = {b_0} ──")
    print(f"  {'model':<28s}  " +
          "  ".join(f"{l:>10s}" for l in PROFILE_LABELS))
    for model_id in ALL_MODELS:
        cells = []
        for label in PROFILE_LABELS:
            pct = aggregate_results[model_id]["sweep"][b_0][label]["pct_of_base"]
            cells.append(f"{pct:>+9.3f}%")
        short = model_id.split("/")[-1]
        print(f"  {short:<28s}  " + "  ".join(cells))

# `full` rank check at every (model, b_0)
print(f"\n  `full` rank-1 at every (model, b_0)?")
all_full_wins = True
for model_id in ALL_MODELS:
    for b_0 in B_0_LIST:
        prof = aggregate_results[model_id]["sweep"][b_0]
        full_pct = prof["full"]["pct_of_base"]
        others = [prof[l]["pct_of_base"] for l in PROFILE_LABELS
                  if l != "full"]
        is_best = all(full_pct < o for o in others)
        if not is_best:
            all_full_wins = False
        short = model_id.split("/")[-1]
        print(f"    {short:<28s}  b_0={b_0}  full={full_pct:+8.3f}%  "
              f"min(others)={min(others):+8.3f}%  "
              f"{'✓' if is_best else '✗'}")

print(f"\n  ALL `full` rank-1: "
      f"{'YES (10/10)' if all_full_wins else 'NO'}")

# `full` advantage scaling with model size
print(f"\n  `full` advantage vs uniform (uniform_pct / full_pct ratio)")
print(f"  {'b_0':>5s}  " +
      "  ".join(f"{m.split('/')[-1]:>20s}" for m in ALL_MODELS))
for b_0 in B_0_LIST:
    cells = []
    for model_id in ALL_MODELS:
        prof = aggregate_results[model_id]["sweep"][b_0]
        f = prof["full"]["pct_of_base"]
        u = prof["uniform"]["pct_of_base"]
        ratio = u / max(f, EPS)
        cells.append(f"{ratio:>20.2f}")
    print(f"  {b_0:>5d}  " + "  ".join(cells))

if all_full_wins:
    verdict_class = "SCALE_INVARIANT"
    verdict = (f"`full` is rank-1 at every (model, b_0) tested "
                f"(5 models × 2 b_0 = 10/10). The bit allocator's "
                f"core property is scale-invariant within the "
                f"Mamba-1 family across two orders of magnitude "
                f"(130M → 2.8B). Combined with C-5b (Mamba-2 "
                f"generalisation), the `full` allocator is "
                f"empirically validated as architecture-agnostic "
                f"within the SSM substrate.")
else:
    verdict_class = "PARTIAL"
    verdict = (f"`full` does NOT win at every (model, b_0). "
                f"Inspect the failed points above. Possible "
                f"causes: small-model under-fitting of the "
                f"`full` recipe (smaller models may have "
                f"different ρ / Γ distributions where the "
                f"allocator's assumptions break), or numerical "
                f"edge cases in the small d_state regime.")

print(f"\n  Verdict class: {verdict_class}")
print(f"  Verdict: {verdict}")


# =============================================================
# === CELL V9 === Save
# =============================================================

# Trim aggregate_results for serialisation (drop tensors)
output = {
    "date":            datetime.utcnow().isoformat(timespec="seconds"),
    "phase":           "C-8a",
    "purpose":         ("Mamba-1 family scale generalisation of "
                        "the Paper C `full` bit allocator. Tests "
                        "the rank-1 property at b_0 ∈ {4, 6} for "
                        "models 130M / 370M / 790M / 1.4B; "
                        "Mamba-2.8B-hf data is reused from "
                        "Phase C-4."),
    "transformers_version": _hf_transformers.__version__,
    "models":          ALL_MODELS,
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
                          "range = k * RMS(h)"),
    },
    "results":         aggregate_results,
    "all_full_wins":   all_full_wins,
    "verdict_class":   verdict_class,
    "verdict":         verdict,
}

out_path = "/content/phase_c8a_mamba_scale_sweep.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False, default=str)
print(f"\nSaved -> {out_path}")


# =============================================================
# === READING ===
# =============================================================
#
# SCALE_INVARIANT (5/5 × 2/2 = 10/10):
#   The `full` allocator transfers across the Mamba-1 family
#   from 130M to 2.8B. Combined with C-5b (Mamba-2 in
#   Zamba2-1.2B), Paper C's universal claim holds: the
#   R_g = Γ_s / (1 - ρ_g) formula is architecture-agnostic
#   for SSM substrates within the Mamba family.
#
# PARTIAL (some models / b_0 fail):
#   Read the failed points carefully. If the smallest models
#   (130M, 370M) fail, the issue is likely the d_state being
#   too small for the bit allocator's coarse-grained granularity
#   to distinguish (R_g distribution may collapse). If a single
#   intermediate model fails at one b_0, may be sampling noise.
#
# Useful diagnostic: the `full` advantage vs uniform (ratio
# u / f). At b_0 = 4 we expect 6 - 10× advantage (matching C-4).
# At b_0 = 6 expect 2 - 4× advantage. If the ratio is much
# smaller for tiny models, the allocator's value diminishes at
# small scales — paper-worthy nuance.
# ============================================================
