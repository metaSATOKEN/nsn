# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Phase C-3: State-step quantisation PPL test on Mamba-2.8B
# ============================================================
# Paper C end-to-end follow-up to Phase C-2c.
#
# Phase C-2c (A_log perturbation surrogate) confirmed the
# bit-allocator direction by 71 % relative ΔPPL reduction
# (full vs proxy_C2) but ran into a *surrogate ceiling*:
#
#   rho = exp(-dt * exp(A_log))
#
# is doubly-exponential in A_log, so high-rho units are
# essentially insensitive to A_log noise. The unintended
# consequence was that anti_full (which concentrates noise
# on high-rho × high-Gamma units) also "won" — the high-rho
# units it targeted are saturated and absorb the noise
# harmlessly, while the full profile spreads its noise across
# medium-rho units that *are* sensitive.
#
# Phase C-3 removes the surrogate by injecting Gaussian noise
# directly into the recurrent state h_t at every time step:
#
#   h_t  ←  A_bar_t * h_{t-1} + B_bar_t * x_t
#   ĥ_t  =  h_t + η_t          η_t ~ N(0, σ_g² I)
#
# where σ_g indexes per-(channel c, state s) just like in
# C-2c. The readout y_t = C_t · ĥ_t carries the error linearly,
# so high-rho units are now *more* sensitive to noise (long
# memory accumulates the perturbation), not less. This is
# the regime in which the bit allocator's prediction
#
#   σ_g  ∝  1 / sqrt(R_g)  =  sqrt( (1 - rho_g) / Gamma_s )
#
# should be tightly observable.
#
# Predicted ranking (C-3, state-step):
#
#   full  <  proxy_correct  <  proxy_C2  <  uniform  <  anti_full
#
# Importantly anti_full should now be *clearly* worst, in
# contrast to C-2c where it surprise-won. If the C-3 ranking
# matches the prediction (and anti_full is clearly worst),
# the C-2c anomaly is fully explained by the A_log-surrogate
# ceiling and the bit allocator is empirically validated as
# the right object to optimise.
#
# Implementation notes:
#
#   1. The HF Mamba reference Python forward path
#      (`MambaMixer.slow_forward`) is monkey-patched. We
#      reproduce its math step-by-step in a Python function
#      `_c3_mamba_forward` so we can hook a noise injection
#      between the state update and the readout.
#   2. The same monkey-patched forward runs in three modes:
#         - "off"        : no noise, no calibration. Used to
#                          compute the FP16 baseline PPL.
#         - "calibrate"  : no noise, but accumulate
#                          rho_emp (E_t[A_bar]),
#                          Gamma_s (E_t[C_t² ]) and
#                          RMS(h)  (E_t[h²]^(1/2))
#                          per layer.
#         - "noise"      : sample η_t ~ N(0, σ_g²) at each
#                          step and add to h_t after the
#                          state update. RNG seed pinned.
#   3. Noise scale is *relative*: σ_g = σ_rel × shape(g) ×
#      RMS_h(g), where shape(g) is one of the six profiles
#      (uniform / proxy_C2 / proxy_correct / full / anti_full)
#      normalised so that mean(shape²) = 1 over all units.
#      This means each profile injects the same average
#      relative noise σ/RMS_h = σ_rel; the comparison
#      isolates *allocation* from *amount*.
#   4. Single seed × six conditions, ~25 - 35 min on A100.
# ============================================================


# =============================================================
# === CELL V1 === Setup
# =============================================================

import gc
import math
import json
import torch
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


# =============================================================
# === CELL V2 === Load
# =============================================================

MODEL_ID    = "state-spaces/mamba-2.8b-hf"
MODEL_LABEL = "Mamba-2.8B-hf"
TOKENIZER_FALLBACKS = [MODEL_ID, "EleutherAI/gpt-neox-20b"]

print(f"\n{'=' * 80}")
print(f"Loading {MODEL_LABEL}  ({MODEL_ID})")
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
print(f"  VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")


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
# === CELL V3 === Monkey-patch slow_forward with state-step
#                 quantisation hook
# =============================================================
#
# Global controller for the patched forward. Three modes:
#   "off"        — pure FP16 baseline
#   "calibrate"  — accumulate rho_emp / Gamma_s / RMS(h)
#   "noise"      — inject N(0, σ_g²) into h_t after update
#
# All accumulation tensors live on GPU during a run and are
# moved to CPU at the end of the calibration pass.

_C3 = {
    "mode":           "off",
    "sigma_per_layer": None,      # dict {name: tensor [d_inner, d_state]}
    "calib_accum":     None,      # dict {name: dict of running sums}
}


def _c3_mamba_forward(mix, input_states):
    """
    Reproduces MambaMixer reference forward pass with optional
    state-step Gaussian noise. Returns contextualized_states
    [B, T, hidden_size]. No KV cache support (use_cache=False).
    """
    layer_name = LAYER_NAME_BY_ID.get(id(mix))
    B, T, _ = input_states.shape
    d_inner = mix.intermediate_size
    d_state = mix.ssm_state_size
    dt_rank = mix.time_step_rank

    # --- Input projection and gating split -----------------
    proj = mix.in_proj(input_states)            # [B, T, 2 d_inner]
    x_pre, gate = proj.chunk(2, dim=-1)         # each [B, T, d_inner]

    # --- Causal 1D conv on x branch ------------------------
    # mix.conv1d is configured with padding=kernel_size-1 already
    # (depthwise causal pad). HF's reference path feeds the un-
    # padded sequence and truncates trailing ksize-1 positions:
    #     x = self.conv1d(x)[..., :seq_len]
    # We MUST NOT add an extra F.pad((k-1, 0), ...) on top of
    # that — doing so double-pads the input and shifts every
    # output by ksize-1 positions, breaking the forward pass
    # (PPL explodes). Phase C-2c's calibration code had this
    # extra F.pad too, but the bug only mildly biased rho/gamma
    # statistics there; here it would corrupt the forward, so
    # we strictly mirror HF.
    x_pre_c = x_pre.transpose(1, 2).contiguous()
    x_pre_c = mix.conv1d(x_pre_c)[:, :, :T]
    x_pre = F.silu(x_pre_c).transpose(1, 2)     # [B, T, d_inner]

    # --- x_proj split into (dt_raw, B, C) ------------------
    xbc = mix.x_proj(x_pre)
    dt_raw, B_proj, C_proj = xbc.split(
        [dt_rank, d_state, d_state], dim=-1
    )

    # --- Discretisation ------------------------------------
    dt = F.softplus(mix.dt_proj(dt_raw)).float()  # [B, T, d_inner]
    A_neg = -torch.exp(mix.A_log.float())         # [d_inner, d_state]

    # Pre-compute A_bar and B_bar over all T at once (per layer
    # this is at most ~1.3 GB at fp32, well within budget).
    A_bar = torch.exp(
        dt[..., None] * A_neg.unsqueeze(0).unsqueeze(0)
    )                                              # [B, T, d_inner, d_state]
    B_bar = (
        dt[..., None] * B_proj.float()[:, :, None, :]
    )                                              # [B, T, d_inner, d_state]
    deltaB_x = B_bar * x_pre.float()[..., None]    # [B, T, d_inner, d_state]

    C_f = C_proj.float()                           # [B, T, d_state]

    # --- Optional precomputed sigma for this layer ---------
    sigma_layer = None
    if _C3["mode"] == "noise":
        s = _C3["sigma_per_layer"].get(layer_name)
        if s is not None:
            sigma_layer = s.to(x_pre.device, dtype=torch.float32)

    # --- Calibration accumulators -------------------------
    if _C3["mode"] == "calibrate":
        # rho: E_t[A_bar]  -> [d_inner, d_state]
        # gamma: E_t[C^2]  -> [d_state]
        # rms: E_t[h^2]    -> [d_inner, d_state]
        rho_sum   = A_bar.sum(dim=(0, 1))           # [d_inner, d_state]
        rho_count = B * T
        gamma_sum   = (C_f ** 2).sum(dim=(0, 1))    # [d_state]
        gamma_count = B * T
        rms_sumsq = torch.zeros((d_inner, d_state),
                                  dtype=torch.float32,
                                  device=x_pre.device)

    # --- Recurrent loop ------------------------------------
    h = torch.zeros((B, d_inner, d_state),
                     dtype=torch.float32, device=x_pre.device)
    ys = []
    for t in range(T):
        h = A_bar[:, t] * h + deltaB_x[:, t]
        if sigma_layer is not None:
            h = h + sigma_layer.unsqueeze(0) * torch.randn_like(h)
        if _C3["mode"] == "calibrate":
            rms_sumsq += (h ** 2).sum(dim=0)
        # readout
        y_t = (h * C_f[:, t].unsqueeze(1)).sum(dim=-1)  # [B, d_inner]
        ys.append(y_t)
    y = torch.stack(ys, dim=1)                    # [B, T, d_inner]

    # --- Skip connection (D) -------------------------------
    y = y + mix.D.float().unsqueeze(0).unsqueeze(0) * x_pre.float()

    # --- Gating + out projection ---------------------------
    y = y * F.silu(gate.float())
    out = mix.out_proj(y.to(mix.out_proj.weight.dtype))

    if _C3["mode"] == "calibrate":
        prev = _C3["calib_accum"].get(layer_name)
        if prev is None:
            _C3["calib_accum"][layer_name] = {
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


def _c3_slow_forward_patch(self, input_states, *args, **kwargs):
    """
    Drop-in replacement for MambaMixer.slow_forward. Ignores
    cache_params / cache_position / attention_mask kwargs (we
    only run with use_cache=False).
    """
    return _c3_mamba_forward(self, input_states)


# Apply the patch at the class level so all 64 mixers are
# covered without per-instance bookkeeping.
_MixerClass = type(mixers[0][1])
_orig_slow_forward = _MixerClass.slow_forward
_MixerClass.slow_forward = _c3_slow_forward_patch
print(f"  Monkey-patched {_MixerClass.__module__}."
      f"{_MixerClass.__name__}.slow_forward")


def _force_slow_path(model):
    """Disable any fast CUDA kernel attribute the HF Mamba code
    might use (covers transformers ≥ 4.39 conditionals)."""
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
# === CELL V4 === Calibration: rho_emp + Gamma_s + RMS(h)
# =============================================================

print(f"\n{'-' * 80}")
print("Calibrating rho_emp + Gamma_s + RMS(h)  (state-step machinery)")
print(f"{'-' * 80}")

ds_test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
test_text = "\n\n".join([x for x in ds_test["text"]
                          if x.strip()])[:48000]
calib_text = "\n\n".join(
    [x for x in load_dataset("wikitext", "wikitext-2-raw-v1",
                              split="train")["text"]
     if x.strip()])[:8000]
calib_ids = tok.encode(calib_text, return_tensors="pt").to("cuda")[:, :1024]

_C3["mode"]         = "calibrate"
_C3["calib_accum"]  = {}
with torch.inference_mode():
    _ = mdl(calib_ids, use_cache=False)
_C3["mode"] = "off"

# Reduce sums to (rho_emp, gamma_s, rms_h) per layer
captured = {}
for name, accum in _C3["calib_accum"].items():
    rho   = accum["rho_sum"]   / accum["rho_count"]      # [d_inner, d_state]
    gamma = accum["gamma_sum"] / accum["gamma_count"]    # [d_state]
    rms   = (accum["rms_sumsq"] /
              accum["rms_count"]).clamp(min=1e-12).sqrt()  # [d_inner, d_state]
    captured[name] = {
        "rho_emp": rho,
        "gamma":   gamma,
        "rms_h":   rms,
    }
print(f"  captured rho/gamma/rms for {len(captured)} layers")

# Quick sanity diagnostic: per-layer typical magnitudes
ex_name = next(iter(captured))
ex = captured[ex_name]
print(f"  example layer  {ex_name}")
print(f"    rho_emp   p50 = {float(ex['rho_emp'].quantile(0.50)):.4f}, "
      f"max = {float(ex['rho_emp'].max()):.4f}")
print(f"    gamma     min = {float(ex['gamma'].min()):.4e}, "
      f"max = {float(ex['gamma'].max()):.4e}")
print(f"    rms_h     p50 = {float(ex['rms_h'].quantile(0.50)):.4e}, "
      f"max = {float(ex['rms_h'].max()):.4e}")


# =============================================================
# === CELL V5 === Six sigma profiles, RMS-relative budget
# =============================================================
#
# σ_g = σ_rel × shape(g) × RMS_h(g)
#
# shape(g) is one of:
#   uniform        : 1
#   proxy_C2       : (1 - rho)
#   proxy_correct  : sqrt(1 - rho)
#   full           : sqrt((1 - rho) / Gamma)
#   anti_full      : sqrt(Gamma / (1 - rho))
#
# Each shape is normalised so that mean(shape²) = 1 over all
# (layer, c, s) units. This means every profile injects the
# same average relative noise σ/RMS_h = σ_rel; only the
# allocation across units differs.

EPS = 1e-6


def _normalize_shape(raw):
    sumsq = sum(float((s ** 2).sum()) for s in raw.values())
    n_total = sum(s.numel() for s in raw.values())
    norm = (n_total / sumsq) ** 0.5
    return {n: s * norm for n, s in raw.items()}


def _shape_uniform(captured):
    return {n: torch.ones_like(d["rho_emp"]) for n, d in captured.items()}


def _shape_proxy_C2(captured):
    return {n: (1.0 - d["rho_emp"]).clamp(min=EPS)
            for n, d in captured.items()}


def _shape_proxy_correct(captured):
    return {n: (1.0 - d["rho_emp"]).clamp(min=EPS).sqrt()
            for n, d in captured.items()}


def _shape_full(captured):
    out = {}
    for n, d in captured.items():
        rho   = d["rho_emp"]
        gamma = d["gamma"].clamp(min=EPS)
        out[n] = ((1.0 - rho).clamp(min=EPS) /
                   gamma.unsqueeze(0)).sqrt()
    return out


def _shape_anti_full(captured):
    out = {}
    for n, d in captured.items():
        rho   = d["rho_emp"]
        gamma = d["gamma"].clamp(min=EPS)
        out[n] = (gamma.unsqueeze(0) /
                   (1.0 - rho).clamp(min=EPS)).sqrt()
    return out


def shape_to_sigma(shape, captured, sigma_rel):
    normed = _normalize_shape(shape)
    return {n: sigma_rel * normed[n] * captured[n]["rms_h"]
            for n in normed}


SIGMA_REL = 0.10

uniform_s        = shape_to_sigma(_shape_uniform(captured),       captured, SIGMA_REL)
proxy_C2_s       = shape_to_sigma(_shape_proxy_C2(captured),      captured, SIGMA_REL)
proxy_correct_s  = shape_to_sigma(_shape_proxy_correct(captured), captured, SIGMA_REL)
full_s           = shape_to_sigma(_shape_full(captured),          captured, SIGMA_REL)
anti_full_s      = shape_to_sigma(_shape_anti_full(captured),     captured, SIGMA_REL)


def total_relative_energy(sigma_dict, captured):
    """Sum_g (sigma_g / RMS_h_g)^2 — should match across profiles."""
    s = 0.0
    for n, sd in sigma_dict.items():
        rms = captured[n]["rms_h"].clamp(min=1e-12)
        s += float(((sd / rms) ** 2).sum())
    return s


print(f"\n  Total *relative* noise energy  Σ_g (σ_g / RMS_h)² "
      f"(should be ~equal across profiles):")
for label, sd in [("uniform",        uniform_s),
                   ("proxy_C2",       proxy_C2_s),
                   ("proxy_correct",  proxy_correct_s),
                   ("full",           full_s),
                   ("anti_full",      anti_full_s)]:
    print(f"    {label:<16s}  {total_relative_energy(sd, captured):>12.4e}")

# Bucket diagnostic: mean σ / RMS_h in each rho bucket
buckets = [(0.0, 0.5), (0.5, 0.8), (0.8, 0.95),
           (0.95, 0.99), (0.99, 0.999), (0.999, 1.0)]
print(f"\n  Mean (σ_g / RMS_h_g) per rho bucket:")
print(f"  {'rho':<14s}  {'unif':>9s}  {'proxy_C2':>9s}  "
      f"{'proxy_sq':>9s}  {'full':>9s}  {'anti':>9s}")
for lo, hi in buckets:
    means = []
    for sd in [uniform_s, proxy_C2_s, proxy_correct_s,
                full_s, anti_full_s]:
        vals = []
        for name, d in captured.items():
            rho = d["rho_emp"]
            mask = (rho >= lo) & (rho < hi if hi < 1.0 else rho <= hi)
            if mask.any():
                rel = sd[name] / d["rms_h"].clamp(min=1e-12)
                vals.append(rel[mask].flatten())
        if vals:
            cat = torch.cat(vals)
            means.append(float(cat.mean()))
        else:
            means.append(0.0)
    print(f"  [{lo:.3f},{hi:.3f}){'':2s}  "
          f"{means[0]:>9.4f}  {means[1]:>9.4f}  "
          f"{means[2]:>9.4f}  {means[3]:>9.4f}  {means[4]:>9.4f}")


# =============================================================
# === CELL V6 === Sliding-window PPL with state-step noise
# =============================================================

def ppl_with_state_noise(model, ids, sigma_dict, seed,
                          ctx_len=1024, stride=512, n_windows=10):
    """
    sigma_dict is None  -> mode "off" (FP16 baseline).
    Otherwise           -> mode "noise" with the dict pinned
                           and seed pinned.
    """
    if sigma_dict is None:
        _C3["mode"] = "off"
        _C3["sigma_per_layer"] = None
    else:
        _C3["mode"] = "noise"
        _C3["sigma_per_layer"] = sigma_dict
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

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
                log_probs = F.log_softmax(out.logits[0, :-1].float(), dim=-1)
                targets = window[0, 1:]
                nll = -log_probs.gather(-1,
                                          targets.unsqueeze(-1)).squeeze(-1)
                nll_total += nll.sum().item()
                tok_total += targets.numel()
                windows_done += 1
    finally:
        _C3["mode"] = "off"
        _C3["sigma_per_layer"] = None
    return math.exp(nll_total / tok_total), windows_done


# =============================================================
# === CELL V7 === Run six-condition sweep
# =============================================================

CTX_LEN   = 1024
STRIDE    = 512
N_WINDOWS = 10
SEED      = 42

test_ids = tok.encode(test_text, return_tensors="pt").to("cuda")
print(f"\n{'-' * 80}")
print(f"WikiText-2 sliding-window PPL  "
      f"(ctx={CTX_LEN}, stride={STRIDE}, n_windows={N_WINDOWS})  "
      f"sigma_rel={SIGMA_REL}")
print(f"{'-' * 80}")

CONDITIONS = [
    ("1. FP16 baseline",                            None),
    ("2. uniform sigma",                            uniform_s),
    ("3. proxy_C2  (sigma ∝ (1-rho))",              proxy_C2_s),
    ("4. proxy_correct (sigma ∝ √(1-rho))",         proxy_correct_s),
    ("5. full      (sigma ∝ √((1-rho)/G))",         full_s),
    ("6. anti_full (sigma ∝ √(G/(1-rho)))",         anti_full_s),
]

results = {}
fp16_ppl = None

for label, sigma_dict in CONDITIONS:
    ppl, n_win = ppl_with_state_noise(mdl, test_ids, sigma_dict, SEED,
                                        ctx_len=CTX_LEN, stride=STRIDE,
                                        n_windows=N_WINDOWS)
    if sigma_dict is None:
        fp16_ppl = ppl
    dppl = ppl - fp16_ppl
    pct  = 100.0 * dppl / fp16_ppl
    results[label] = {
        "ppl":            ppl,
        "delta_ppl":      dppl,
        "pct_of_base":    pct,
        "n_windows":      n_win,
    }
    print(f"  {label:<42s}  ppl={ppl:>9.4f}  "
          f"ΔPPL={dppl:+8.4f}  ({pct:+7.3f} % of base)")


# =============================================================
# === CELL V8 === Verdict: state-step disambiguates C-2c
# =============================================================

uni_pct      = results["2. uniform sigma"]["pct_of_base"]
pC2_pct      = results["3. proxy_C2  (sigma ∝ (1-rho))"]["pct_of_base"]
psq_pct      = results["4. proxy_correct (sigma ∝ √(1-rho))"]["pct_of_base"]
full_pct     = results["5. full      (sigma ∝ √((1-rho)/G))"]["pct_of_base"]
anti_pct     = results["6. anti_full (sigma ∝ √(G/(1-rho)))"]["pct_of_base"]

print(f"\n{'=' * 80}")
print(f"Phase C-3 verdict at sigma_rel = {SIGMA_REL}")
print(f"{'=' * 80}")
print(f"\n  Predicted ranking:")
print(f"    full < proxy_correct < proxy_C2 < uniform < anti_full")
print(f"\n  Observed (% of FP16 baseline):")
print(f"    full           {full_pct:+7.3f}")
print(f"    proxy_correct  {psq_pct:+7.3f}")
print(f"    proxy_C2       {pC2_pct:+7.3f}")
print(f"    uniform        {uni_pct:+7.3f}")
print(f"    anti_full      {anti_pct:+7.3f}")

# Headline question for C-3: does anti_full now lose, removing
# the C-2c surrogate-ceiling artefact?
gap_anti_vs_full = anti_pct - full_pct
gap_anti_vs_uni  = anti_pct - uni_pct
gap_full_vs_pC2  = pC2_pct - full_pct
rel_full_vs_pC2  = 100.0 * gap_full_vs_pC2 / max(pC2_pct, EPS)

print(f"\n  Headline comparisons:")
print(f"    anti_full vs full:    ΔΔPPL = {gap_anti_vs_full:+7.3f} %")
print(f"    anti_full vs uniform: ΔΔPPL = {gap_anti_vs_uni:+7.3f} %")
print(f"    full vs proxy_C2:     ΔΔPPL = {gap_full_vs_pC2:+7.3f} %  "
      f"(rel {rel_full_vs_pC2:+6.1f} %)")

# Sanity checks
direction_ok = (full_pct < uni_pct) and (uni_pct < anti_pct)
strict_rank  = (full_pct < psq_pct < pC2_pct < uni_pct < anti_pct)
anti_loses   = (anti_pct > uni_pct)        # C-2c had this fail
full_wins    = (full_pct < pC2_pct)         # C-2c had this hold

print(f"\n  Direction sanity (full < uniform < anti_full):  "
      f"{'✓' if direction_ok else '✗'}")
print(f"  Strict ranking holds (full<psq<pC2<unif<anti):    "
      f"{'✓' if strict_rank else '✗'}")
print(f"  anti_full clearly worst (anti > uniform):         "
      f"{'✓' if anti_loses else '✗'}  "
      f"[C-2c failed this — surrogate ceiling]")
print(f"  full beats proxy_C2:                              "
      f"{'✓' if full_wins else '✗'}")

# Verdict text
if direction_ok and anti_loses and full_wins:
    verdict = (f"STATE-STEP CONFIRMS BIT ALLOCATOR. The C-2c "
                f"anomaly (anti_full surprise win) was an A_log-"
                f"surrogate ceiling artefact, fully resolved here. "
                f"Bit allocator direction validated end-to-end on "
                f"the noise channel that physically corresponds to "
                f"quantisation. Proceed to Phase C-5 (hybrid stack).")
elif full_wins and not anti_loses:
    verdict = (f"PARTIAL. full beats proxy_C2 as predicted, but "
                f"anti_full still does not fall below uniform. The "
                f"surrogate-ceiling story is incomplete — there is a "
                f"second mechanism protecting high-rho × high-Gamma "
                f"units beyond A_log saturation. Investigate before "
                f"Phase C-5.")
elif (not full_wins) and anti_loses:
    verdict = (f"INCONCLUSIVE. anti_full now loses (surrogate "
                f"ceiling resolved) but full no longer beats "
                f"proxy_C2 cleanly. Possible: state-step noise on "
                f"low-rho units dominates the budget and erases "
                f"the Gamma signal. Try lowering sigma_rel.")
else:
    verdict = (f"UNEXPECTED. Both core predictions failed. Check "
                f"calibration drift, RMS_h scale, or whether the "
                f"monkey-patch is correctly active (look at FP16 "
                f"baseline ppl vs. the C-2c FP16 ppl — they should "
                f"agree to 4 decimals).")

print(f"\n  Verdict:\n    {verdict}")


# =============================================================
# === CELL V9 === Save
# =============================================================

# Compact per-layer summary
summary = []
for name, d in captured.items():
    rho   = d["rho_emp"]
    gamma = d["gamma"]
    rms   = d["rms_h"]
    summary.append({
        "layer_name":            name,
        "n_units":               int(rho.numel()),
        "rho_p50":               float(rho.flatten().quantile(0.50)),
        "rho_max":               float(rho.max()),
        "gamma_min":             float(gamma.min()),
        "gamma_p50":             float(gamma.quantile(0.50)),
        "gamma_max":             float(gamma.max()),
        "gamma_max_min_ratio":   float(gamma.max() /
                                        gamma.min().clamp(min=EPS)),
        "rms_h_p50":             float(rms.quantile(0.50)),
        "rms_h_max":             float(rms.max()),
    })

output = {
    "date":           datetime.utcnow().isoformat(timespec="seconds"),
    "phase":          "C-3",
    "purpose":        ("State-step quantisation PPL test on "
                       "Mamba-2.8B-hf. Removes the A_log-surrogate "
                       "ceiling that affected C-2c by injecting "
                       "Gaussian noise directly into the recurrent "
                       "state h_t at each step. Tests the same six "
                       "allocation profiles (uniform / proxy_C2 / "
                       "proxy_correct / full / anti_full / FP16) "
                       "with RMS-relative noise budget held "
                       "constant across profiles."),
    "model":          MODEL_ID,
    "label":          MODEL_LABEL,
    "protocol": {
        "ctx_len":      CTX_LEN,
        "stride":       STRIDE,
        "n_windows":    N_WINDOWS,
        "sigma_rel":    SIGMA_REL,
        "rng_seed":     SEED,
        "calib_split":  "WikiText-2 train",
        "test_split":   "WikiText-2 test",
        "noise_model":  "post-state-update Gaussian, σ_g = σ_rel·shape(g)·RMS_h(g)",
        "conditions":   [c[0] for c in CONDITIONS],
    },
    "fp16_baseline_ppl":   fp16_ppl,
    "results":             results,
    "headline_gaps": {
        "anti_full_vs_full_pp":   gap_anti_vs_full,
        "anti_full_vs_uniform_pp": gap_anti_vs_uni,
        "full_vs_proxy_C2_pp":    gap_full_vs_pC2,
        "full_vs_proxy_C2_rel":   rel_full_vs_pC2,
    },
    "direction_ok":            direction_ok,
    "strict_ranking":          strict_rank,
    "anti_full_loses":         anti_loses,
    "full_beats_proxy_C2":     full_wins,
    "rho_gamma_rms_summary":   summary,
    "verdict":                 verdict,
}

out_path = "/content/phase_c3_state_step_quantisation.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\nSaved -> {out_path}")


# =============================================================
# === READING ===
# =============================================================
#
# What this run answers (vs. C-2c):
#
#   STATE-STEP CONFIRMS BIT ALLOCATOR  (anti_full > uniform AND
#                                        full   < proxy_C2):
#     The C-2c anti_full surprise win was an A_log-surrogate
#     ceiling artefact. Under physically-correct state-step
#     quantisation noise, the bit allocator's predicted ranking
#     is recovered. Paper C is empirically validated. Move to
#     Phase C-5 (hybrid stack on Qwen 3.6 27B / Zamba2 with
#     full allocator on SSM side + modeq INT3 on attention side).
#
#   PARTIAL  (full < proxy_C2 but anti_full still wins):
#     full direction is right, but a second mechanism beyond
#     A_log saturation is still protecting high-rho units.
#     Candidates: numerical underflow in (1-rho) for rho ≈ 1,
#     gradient-flow effects in h, or RMS_h itself being
#     correlated with rho in a way that makes anti_full
#     accidentally inject *less* total noise. Diagnose before
#     C-5.
#
#   INCONCLUSIVE  (anti_full loses but full doesn't beat
#                   proxy_C2):
#     The Gamma signal is being washed out by the noise budget
#     itself. Try sigma_rel = 0.05 to reduce total noise; the
#     allocation gap is most visible at moderate noise levels
#     (too little = no effect, too much = saturation again).
#
#   UNEXPECTED  (both fail):
#     Sanity-check the FP16 baseline against C-2c's FP16
#     baseline ppl. If they disagree, the monkey-patch has a
#     bug. If they agree, check sigma_per_layer scaling — the
#     RMS-relative trick might be amplifying numerical issues
#     at very high or very low rho units.
# ============================================================
