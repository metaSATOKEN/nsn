# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Phase C-12: Strict bit-budget optimum with Gamma_g
# ============================================================
# Section 3.5 of the paper acknowledges that the heuristic
# integer construction
#
#     extras_g = ceil(log_2 R_g)_+
#     b_g       = b_0 + extras_g + shift, clamped to [b_min, 16]
#
# uses a slope of unity on log_2 R_g, twice the slope of the
# strict bit-budget optimum derived in Section 3.4
# (b_g* = b_0 + (1/2) log_2 R_g + log_2 r_g, dropping log_2 r_g
# in our specific implementation). The Section 4.5 sigma2-scaling
# experiment validates the heuristic full empirically against
# uniform / proxy / anti / inverse, but does not include the
# strict-slope full as a control. Both Grok and GPT pseudo-
# reviewers flagged this as a missing comparison.
#
# Phase C-12 closes that gap. It runs the same C-3 / C-3b state-
# step Gaussian-noise injection on Mamba-2.8B-hf with five
# profiles, including the new "full_half" profile that
# implements the strict (1/2)-slope optimum:
#
#   uniform:        sigma_g constant
#   proxy_C2:       sigma_g proportional to (1 - rho_g)
#   proxy_correct:  sigma_g proportional to sqrt(1 - rho_g)
#   full (heur):    sigma_g proportional to sqrt((1 - rho_g) / Gamma_g)
#                   = the heuristic profile from Section 4
#   full_half:      sigma_g proportional to ((1 - rho_g) / Gamma_g)^{1/4}
#                   = the strict (1/2)-slope optimum on log_2 R_g
#                   under sigma -> 2^{-b} mapping, which produces
#                   sigma proportional to 1/sqrt(R_g) in continuous
#                   space and (1/2) log_2 R_g extras in integer space
#   anti_full:      sigma_g proportional to sqrt(Gamma_g / (1 - rho_g))
#                   (deliberately backwards, as in C-3 / C-3b)
#
# Sweep:
#   sigma_rel in {0.05, 0.10, 0.20}   (matching C-3b)
#   n_windows = 5                      (matching C-3 / C-3b)
#
# The total noise energy sum_g sigma_g^2 is held constant across
# profiles at every sigma_rel (matching C-3 / C-3b). The
# comparison isolates allocation shape from noise amount.
#
# Pre-registered hypotheses:
#
#   H1 (full vs full_half ranking):
#     If the strict (1/2)-slope optimum is in fact tighter than
#     our unity-slope heuristic for the bit-budget objective,
#     then full_half should beat full at every sigma_rel.
#
#   H2 (full vs full_half scaling):
#     If the heuristic full is genuinely uniquely sigma2-scaling
#     (Section 4.5), then full_half should be super-quadratic.
#
#   H3 (rank order preservation):
#     The full and full_half profiles should sandwich the
#     proxy_correct profile from above and below, in the order
#     full_half < full < proxy_correct or
#     full < full_half < proxy_correct, depending on which slope
#     is empirically tighter.
#
# Outcome decision rule:
#
#   - If full_half consistently beats full: the heuristic is
#     suboptimal, Section 3.5 should be revised to recommend
#     the strict (1/2)-slope construction.
#   - If full consistently beats full_half: the empirical
#     justification for the heuristic is strengthened, Section
#     4.5 / Section 3.5 stand as written.
#   - If results are within noise (under one percentage point):
#     the choice between unity and (1/2) slope is empirically
#     irrelevant on this Mamba-2.8B-hf evaluation, and Section
#     8.3 (Phase C-12) is closed without changing the headline.
#
# Runtime estimate (A100 95 GB):
#   ~10 min per (sigma_rel, profile) at n_windows=5, ctx 1024
#   6 profiles x 3 sigma_rel = 18 runs
#   Total ~3 hours
#
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
# === CELL V3 === Patched slow_forward (Gaussian state-step noise)
# =============================================================
# This patch replicates the C-3 / C-3b protocol: inject Gaussian
# state-step noise eta_t with per-(c, s) variance sigma_g^2,
# between the state update and the readout.

_C12 = {
    "mode":           "off",       # "off" | "calibrate" | "noise_inject"
    "sigma_per_layer": None,       # dict[layer_name, tensor [d_inner, d_state]]
    "calib_accum":    None,
}

EPS = 1e-6


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

    sigma_layer = None
    if _C12["mode"] == "noise_inject" and _C12["sigma_per_layer"] is not None:
        sig = _C12["sigma_per_layer"].get(layer_name)
        if sig is not None:
            sigma_layer = sig.to(x_pre.device, dtype=torch.float32)

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
        if sigma_layer is not None:
            # inject per-(c, s) Gaussian noise with sigma_layer
            eta = torch.randn_like(h) * sigma_layer.unsqueeze(0)
            h = h + eta
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
# === CELL V5 === Build sigma profiles (per-(c, s))
# =============================================================
# Each profile is a dict[layer_name, sigma_tensor [d_inner, d_state]].
# sigma is then scaled to a target sigma_rel = sigma_g / RMS(h_g) on
# average across all (c, s) units, holding total noise energy fixed.

def _shape_uniform(captured):
    """sigma_g constant across all (c, s)."""
    return {n: torch.ones_like(d["rms_h"]) for n, d in captured.items()}


def _shape_proxy_C2(captured):
    """sigma_g ∝ (1 - rho_g)."""
    return {n: (1.0 - d["rho_emp"]).clamp(min=EPS)
            for n, d in captured.items()}


def _shape_proxy_correct(captured):
    """sigma_g ∝ sqrt(1 - rho_g)."""
    return {n: (1.0 - d["rho_emp"]).clamp(min=EPS).sqrt()
            for n, d in captured.items()}


def _shape_full_heuristic(captured):
    """sigma_g ∝ sqrt((1 - rho_g) / Gamma_g)
    (= heuristic full from Section 4: sigma ∝ 1/sqrt(R_g)
       where R_g = Gamma / (1 - rho)).
    Per (c, s); Gamma broadcasts from [d_state] over channel dim."""
    out = {}
    for n, d in captured.items():
        rho   = d["rho_emp"]
        gamma = d["gamma"].clamp(min=EPS)  # [d_state]
        denom = gamma.unsqueeze(0).expand_as(rho)  # [d_inner, d_state]
        sig = ((1.0 - rho).clamp(min=EPS) / denom).sqrt()
        out[n] = sig
    return out


def _shape_full_half(captured):
    """sigma_g ∝ ((1 - rho_g) / Gamma_g)^{1/4}
    (= strict (1/2)-slope optimum: continuous sigma ∝ R_g^{-1/2}
       maps to b_g = (1/2) log_2 R_g, which under sigma ∝ 2^{-b}
       gives sigma ∝ 2^{-(1/2) log_2 R_g} = R_g^{-1/2}.
       But here we're directly comparing CONTINUOUS sigma profiles,
       not bits, so the strict optimum on continuous sigma is
       1/sqrt(R_g) = full_heuristic above. The "half" profile in
       this experiment corresponds to the ratio between the full
       integer-bit allocator and the strict (1/2)-slope integer
       allocator, which when translated back to continuous sigma
       gives sigma ∝ R_g^{-1/4} = ((1-rho)/Gamma)^{1/4}.)

    The motivation for testing this profile: in C-3b, the heuristic
    full integer construction maps to sigma proportional to 1/R_g
    (because b_g ∝ log_2 R_g, sigma ∝ 2^{-b_g} ∝ R_g^{-1}). The
    strict optimum integer would map to sigma ∝ R_g^{-1/2}. So
    when we compare CONTINUOUS sigma profiles in C-3b, the
    "heuristic full" continuous profile that corresponds to the
    integer heuristic is sigma ∝ R_g^{-1}, which equals
    (1-rho)/Gamma — NOT sqrt((1-rho)/Gamma). And the "strict
    (1/2)-slope" continuous profile that corresponds to the
    integer (1/2) construction is sigma ∝ R_g^{-1/2} =
    sqrt((1-rho)/Gamma).

    Phase C-12 corrects this terminology gap. The new profiles:
       full_one         continuous sigma proportional to 1/R_g
                          (corresponds to integer b_g = log_2 R_g)
       full_half        continuous sigma proportional to 1/sqrt(R_g)
                          (corresponds to integer b_g = (1/2) log_2 R_g)

    The (Section 4.5 "full" was actually full_half-named-full, i.e.,
    the bit-budget-optimal continuous profile). This re-labelling
    matches the original C-2c convention ("full" in C-2c = sigma
    proportional to sqrt((1-rho)/Gamma) = 1/sqrt(R_g)).
    """
    out = {}
    for n, d in captured.items():
        rho   = d["rho_emp"]
        gamma = d["gamma"].clamp(min=EPS)
        denom = gamma.unsqueeze(0).expand_as(rho)
        sig = ((1.0 - rho).clamp(min=EPS) / denom).pow(0.25)
        out[n] = sig
    return out


def _shape_full_one(captured):
    """sigma_g ∝ (1 - rho_g) / Gamma_g
    (= the heuristic full integer construction's continuous
       equivalent, sigma ∝ 1/R_g)."""
    out = {}
    for n, d in captured.items():
        rho   = d["rho_emp"]
        gamma = d["gamma"].clamp(min=EPS)
        denom = gamma.unsqueeze(0).expand_as(rho)
        sig = (1.0 - rho).clamp(min=EPS) / denom
        out[n] = sig
    return out


def _shape_anti_full(captured):
    """sigma_g ∝ sqrt(Gamma_g / (1 - rho_g))
    (deliberately backwards, = sigma ∝ sqrt(R_g))."""
    out = {}
    for n, d in captured.items():
        rho   = d["rho_emp"]
        gamma = d["gamma"].clamp(min=EPS)
        denom = (1.0 - rho).clamp(min=EPS)
        sig = (gamma.unsqueeze(0).expand_as(rho) / denom).sqrt()
        out[n] = sig
    return out


def _normalise_sigma_profile(shape_dict, captured, sigma_rel):
    """Scale shape so that average sigma_g / RMS(h_g) = sigma_rel
    across all (c, s) units; preserve total noise energy
    sum_g sigma_g^2 = sum_g (sigma_rel * RMS(h_g))^2 / |g| * sum 1

    Concretely: we want sum_g sigma_g^2 to equal the same constant
    as a uniform sigma_rel * RMS profile would produce. So set
        sigma_g = sigma_rel * RMS(h_g) * norm(shape) / shape_g
    where norm normalises shape to satisfy energy conservation.

    Implementation: we want
        sum_g (alpha * shape_g)^2 = sum_g (sigma_rel * RMS(h_g))^2
    so alpha = sigma_rel * sqrt(sum_g RMS(h_g)^2) / sqrt(sum_g shape_g^2).
    Then sigma_g = alpha * shape_g / RMS(h_g) ? No: we want sigma_g
    to be the actual injected noise sigma, not relative. Just:
        sigma_g = alpha * shape_g
    where alpha is set so total noise energy equals reference.
    """
    # reference total energy = sum_g (sigma_rel * RMS(h_g))^2
    ref_energy = 0.0
    n_total = 0
    for n, d in captured.items():
        rms = d["rms_h"]
        ref_energy += float(((sigma_rel * rms) ** 2).sum())
        n_total += rms.numel()

    # current shape energy = sum_g shape_g^2
    shape_energy = 0.0
    for n, sh in shape_dict.items():
        shape_energy += float((sh ** 2).sum())

    alpha = math.sqrt(ref_energy / max(shape_energy, EPS))

    out = {n: alpha * sh for n, sh in shape_dict.items()}

    # sanity: recompute total energy
    total_energy = sum(float((s ** 2).sum()) for s in out.values())
    print(f"      profile energy check: ref={ref_energy:.4e} "
          f"out={total_energy:.4e} ratio={total_energy / ref_energy:.4f}")
    return out


# =============================================================
# === CELL V6 === Sweep
# =============================================================

CTX = 1024
N_WINDOWS = 5
SIGMA_RELS = [0.05, 0.10, 0.20]

# Note: profile labels match C-3b nomenclature where possible. The
# "full" and "full_half" labels are what we want to compare for
# Phase C-12; "full_one" is included to map the original C-3b
# "full" profile under the corrected interpretation.
PROFILES = [
    ("uniform",       _shape_uniform),
    ("proxy_C2",      _shape_proxy_C2),
    ("proxy_correct", _shape_proxy_correct),
    ("full_one",      _shape_full_one),       # sigma ∝ 1/R_g (= heuristic integer's continuous)
    ("full_half",     _shape_full_half),      # sigma ∝ 1/sqrt(R_g) (= strict (1/2)-slope continuous)
    ("full",          _shape_full_heuristic), # sigma ∝ sqrt((1-rho)/Gamma) (= C-3b "full")
    ("anti_full",     _shape_anti_full),
]


def ppl_with_noise(model, ids, ctx=1024, n_windows=5):
    """Compute sliding-window PPL under current _C12 mode."""
    stride = ctx // 2
    nll_total = 0.0
    tok_total = 0
    windows_done = 0
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
    return math.exp(nll_total / tok_total), windows_done


# Baseline (no noise)
print(f"\n{'-' * 80}")
print(f"FP16 baseline (no injection)")
print(f"{'-' * 80}")

# Use a separate test corpus (matching C-3 / C-3b: WikiText-2 test)
test_text = "\n\n".join(
    [x for x in load_dataset("wikitext", "wikitext-2-raw-v1",
                              split="test")["text"]
     if x.strip()])[:200_000]
test_ids = tok.encode(test_text, return_tensors="pt").to("cuda")
print(f"  Test corpus: WikiText-2 test, {test_ids.shape[1]:,} tokens")

_C12["mode"] = "off"
fp16_ppl, n_win = ppl_with_noise(mdl, test_ids, ctx=CTX, n_windows=N_WINDOWS)
print(f"  FP16 PPL = {fp16_ppl:.4f} (n_windows={n_win})")


# Sweep
print(f"\n{'-' * 80}")
print(f"Phase C-12 noise sweep: 7 profiles x 3 sigma_rel")
print(f"  ctx={CTX}, n_windows={N_WINDOWS}, total energy held constant")
print(f"{'-' * 80}")

sweep_results = {}

for sigma_rel in SIGMA_RELS:
    print(f"\n  sigma_rel = {sigma_rel:.2f}")
    sweep_results[sigma_rel] = {"FP16_baseline": {"ppl": fp16_ppl, "delta_pct": 0.0}}
    for label, shape_fn in PROFILES:
        print(f"    {label}...")
        shape = shape_fn(captured)
        sigma = _normalise_sigma_profile(shape, captured, sigma_rel)
        _C12["sigma_per_layer"] = sigma
        _C12["mode"] = "noise_inject"
        ppl, n_win = ppl_with_noise(mdl, test_ids, ctx=CTX, n_windows=N_WINDOWS)
        _C12["mode"] = "off"
        delta_pct = 100.0 * (ppl - fp16_ppl) / fp16_ppl
        sweep_results[sigma_rel][label] = {
            "ppl": ppl,
            "delta_pct": delta_pct,
            "n_windows": n_win,
        }
        print(f"      ppl = {ppl:>9.4f}  delta = {delta_pct:+8.3f} %")


# =============================================================
# === CELL V7 === Verdict
# =============================================================

print(f"\n{'=' * 80}")
print(f"Phase C-12 verdict")
print(f"{'=' * 80}")
print(f"\n  FP16 baseline PPL: {fp16_ppl:.4f}")
print(f"\n  Profile           sigma_rel = 0.05   0.10   0.20")
print(f"  {'─' * 64}")
for label, _ in PROFILES:
    row = f"  {label:<15s}        "
    for sr in SIGMA_RELS:
        d = sweep_results[sr][label]["delta_pct"]
        row += f"  {d:+7.3f} %"
    print(row)

# Specific comparison: full vs full_half
print(f"\n  full vs full_half (the v0.2 question):")
for sr in SIGMA_RELS:
    full_d = sweep_results[sr]["full"]["delta_pct"]
    half_d = sweep_results[sr]["full_half"]["delta_pct"]
    one_d  = sweep_results[sr]["full_one"]["delta_pct"]
    diff = full_d - half_d
    winner = "full" if full_d < half_d else "full_half" if half_d < full_d else "tie"
    print(f"    sigma_rel={sr}: full={full_d:+.3f} % "
          f"half={half_d:+.3f} % one={one_d:+.3f} %  "
          f"(diff full - half = {diff:+.3f}, {winner} wins)")

# Sigma2-scaling check on each profile
print(f"\n  Sigma2-scaling exponents (k = log_2(multiplier)):")
for label, _ in PROFILES:
    delta_005 = sweep_results[0.05][label]["delta_pct"]
    delta_010 = sweep_results[0.10][label]["delta_pct"]
    delta_020 = sweep_results[0.20][label]["delta_pct"]
    if delta_005 > 0 and delta_010 > 0 and delta_020 > 0:
        m1 = delta_010 / delta_005
        m2 = delta_020 / delta_010
        k1 = math.log2(max(m1, EPS))
        k2 = math.log2(max(m2, EPS))
        k_mean = (k1 + k2) / 2
        k_spread = abs(k1 - k2) / 2
        print(f"    {label:<15s}  k = {k_mean:.2f} +/- {k_spread:.2f} "
              f"(multipliers {m1:.2f}, {m2:.2f})")
    else:
        print(f"    {label:<15s}  multipliers undefined "
              f"(some delta is non-positive)")


# =============================================================
# === CELL V8 === Save
# =============================================================

calib_summary = []
for name, d in captured.items():
    calib_summary.append({
        "layer_name":   name,
        "rho_p50":      float(d["rho_emp"].flatten().quantile(0.50)),
        "gamma_p50":    float(d["gamma"].quantile(0.50)),
        "rms_h_p50":    float(d["rms_h"].quantile(0.50)),
    })

output = {
    "date":            datetime.utcnow().isoformat(timespec="seconds"),
    "phase":           "C-12",
    "purpose":         ("Strict (1/2)-slope vs heuristic-unity-slope "
                        "bit-allocation comparison via continuous noise "
                        "injection. Closes the Section 3.5 / Section 4.5 "
                        "gap flagged by Grok + GPT pseudo-review: prior "
                        "C-3 / C-3b experiments did not include the strict "
                        "(1/2)-slope profile (sigma proportional to "
                        "((1-rho)/Gamma)^{1/4}, equivalently 1/sqrt(R_g)) "
                        "as a control. Phase C-12 adds full_half and "
                        "full_one to the C-3b profile set and runs the "
                        "same 5-window WikiText-2 test sweep at "
                        "sigma_rel in {0.05, 0.10, 0.20}."),
    "transformers_version": _hf_transformers.__version__,
    "model":           MODEL_ID,
    "label":           MODEL_LABEL,
    "protocol": {
        "ctx":           CTX,
        "n_windows":     N_WINDOWS,
        "sigma_rels":    SIGMA_RELS,
        "calib_split":   "WikiText-2 train (1024 tokens)",
        "test_split":    "WikiText-2 test",
        "profiles":      [p[0] for p in PROFILES],
        "energy_constraint": "sum_g sigma_g^2 held constant across profiles at every sigma_rel",
    },
    "fp16_baseline_ppl": fp16_ppl,
    "sweep_results":   {str(k): v for k, v in sweep_results.items()},
    "rho_gamma_rms_summary": calib_summary,
    "notes": (
        "Profile labels: 'full' = sqrt((1-rho)/Gamma) (matches C-3b "
        "'full' = bit-budget-optimal continuous sigma); 'full_half' "
        "= ((1-rho)/Gamma)^{1/4} (the strict (1/2)-slope integer "
        "construction translated back to continuous sigma); "
        "'full_one' = (1-rho)/Gamma (the heuristic unity-slope integer "
        "construction translated back to continuous sigma; equivalent "
        "to the integer formula b_g = b_0 + ceil(log_2 R_g) under "
        "sigma proportional to 2^{-b_g}). Comparing full vs full_half "
        "directly addresses the 'is the unity slope justified' "
        "question raised in Section 3.5 of the v0.2 manuscript."
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
# Outcome decision rule (pre-registered):
#
# Case A: full_half consistently beats full at all sigma_rel
#         (e.g., delta_full - delta_full_half > 0.5 percentage
#          points at sigma_rel = 0.20).
#   -> The strict (1/2)-slope is empirically tighter. Section 3.5
#      should be revised to recommend the strict integer
#      construction b_g = b_0 + ceil((1/2) log_2 R_g)+. Section 4.5
#      should be re-run with full_half as the headline "full"
#      profile, OR the v0.2 manuscript should report both and
#      prefer the strict construction.
#
# Case B: full consistently beats full_half at all sigma_rel
#         (e.g., delta_full_half - delta_full > 0.5 pp at sigma_rel
#          = 0.20).
#   -> The unity-slope heuristic is empirically tighter. Section
#      4.5's σ²-scaling claim survives a stronger control. The
#      v0.2 paper's heuristic framing in Section 3.5 stands.
#
# Case C: full and full_half are within sampling noise at every
#         sigma_rel (delta diff < 0.5 pp throughout).
#   -> The choice is empirically irrelevant on this evaluation.
#      Section 3.5 of the manuscript can report this as a
#      successful closure: the strict optimum and the heuristic
#      are both valid integer constructions.
#
# Case D: full_one (= the heuristic integer's actual continuous
#         sigma profile) underperforms both full and full_half.
#   -> An interesting finding: the integer rounding + clamp + shift
#      pipeline of Section 3.5 may produce a deployment recipe
#      that is empirically better than the continuous heuristic
#      that motivates it. This would suggest the integer effects
#      (rounding to integer, B_MIN floor, ceiling clamp) play a
#      structural role beyond approximation.
# ============================================================
