# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Phase C-3b: σ_rel sweep on state-step quantisation
# ============================================================
# Paper C end-to-end follow-up to Phase C-3.
#
# Phase C-3 (single point at σ_rel = 0.10) found:
#
#     full         +0.351 %     ← best
#     proxy_correct +2.850 %
#     proxy_C2     +3.531 %
#     anti_full    +6.839 %
#     uniform      +46.624 %    ← unexpectedly catastrophic
#
# Two C-3 surprises to characterise:
#
#  1. `full` beats `proxy_C2` by 90 % rel reduction (largest
#     margin observed across C-2 / C-2b / C-2c / C-3). Does
#     this margin hold across σ_rel ∈ {0.05, 0.10, 0.20}, or is
#     0.10 a sweet spot?
#  2. `uniform` is dramatically worse than `anti_full`
#     (46.6 % vs 6.8 %). Is uniform's pathology a high-noise
#     phenomenon (catastrophic only when σ_rel is large) or
#     does it persist at low noise too?
#
# Phase C-3b reuses the C-3 monkey-patch and runs the same six
# conditions at three σ_rel values. RNG seed is pinned per
# (σ_rel × condition) so the comparisons are reproducible.
#
# Predicted readings:
#
#  - At σ_rel = 0.05: uniform's pathology is mild. All five
#    structured profiles cluster within ~1 % of FP16. `full`
#    still wins among them but the absolute gap shrinks.
#  - At σ_rel = 0.10: matches C-3 exactly (sanity check).
#  - At σ_rel = 0.20: uniform's degradation grows
#    super-linearly because long-memory accumulation is
#    σ²-quadratic in σ. anti_full also worsens but the gap
#    between uniform and anti_full grows.
#
# Runtime ~30 - 35 min × 3 σ_rel values ≈ 90 - 105 min on A100
# 80 GB.
# ============================================================


# =============================================================
# === CELL V1 === Setup  (identical to C-3)
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
# === CELL V3 === Monkey-patch slow_forward (identical to C-3)
# =============================================================

_C3 = {
    "mode":           "off",
    "sigma_per_layer": None,
    "calib_accum":     None,
}


def _c3_mamba_forward(mix, input_states):
    layer_name = LAYER_NAME_BY_ID.get(id(mix))
    B, T, _ = input_states.shape
    d_inner = mix.intermediate_size
    d_state = mix.ssm_state_size
    dt_rank = mix.time_step_rank

    proj = mix.in_proj(input_states)
    x_pre, gate = proj.chunk(2, dim=-1)

    # mix.conv1d already has padding=ksize-1, do NOT pre-pad.
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
    if _C3["mode"] == "noise":
        s = _C3["sigma_per_layer"].get(layer_name)
        if s is not None:
            sigma_layer = s.to(x_pre.device, dtype=torch.float32)

    if _C3["mode"] == "calibrate":
        rho_sum   = A_bar.sum(dim=(0, 1))
        rho_count = B * T
        gamma_sum   = (C_f ** 2).sum(dim=(0, 1))
        gamma_count = B * T
        rms_sumsq = torch.zeros((d_inner, d_state),
                                  dtype=torch.float32,
                                  device=x_pre.device)

    h = torch.zeros((B, d_inner, d_state),
                     dtype=torch.float32, device=x_pre.device)
    ys = []
    for t in range(T):
        h = A_bar[:, t] * h + deltaB_x[:, t]
        if sigma_layer is not None:
            h = h + sigma_layer.unsqueeze(0) * torch.randn_like(h)
        if _C3["mode"] == "calibrate":
            rms_sumsq += (h ** 2).sum(dim=0)
        y_t = (h * C_f[:, t].unsqueeze(1)).sum(dim=-1)
        ys.append(y_t)
    y = torch.stack(ys, dim=1)

    y = y + mix.D.float().unsqueeze(0).unsqueeze(0) * x_pre.float()
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
    return _c3_mamba_forward(self, input_states)


_MixerClass = type(mixers[0][1])
_orig_slow_forward = _MixerClass.slow_forward
_MixerClass.slow_forward = _c3_slow_forward_patch
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
# === CELL V4 === Calibration: rho_emp + Gamma_s + RMS(h)
# =============================================================

print(f"\n{'-' * 80}")
print("Calibrating rho_emp + Gamma_s + RMS(h)")
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

captured = {}
for name, accum in _C3["calib_accum"].items():
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


# =============================================================
# === CELL V5 === Six sigma profiles (functions only;
#                 instantiated per-σ_rel below)
# =============================================================

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


def build_profiles(captured, sigma_rel):
    return {
        "uniform":        shape_to_sigma(_shape_uniform(captured),
                                            captured, sigma_rel),
        "proxy_C2":       shape_to_sigma(_shape_proxy_C2(captured),
                                            captured, sigma_rel),
        "proxy_correct":  shape_to_sigma(_shape_proxy_correct(captured),
                                            captured, sigma_rel),
        "full":           shape_to_sigma(_shape_full(captured),
                                            captured, sigma_rel),
        "anti_full":      shape_to_sigma(_shape_anti_full(captured),
                                            captured, sigma_rel),
    }


# =============================================================
# === CELL V6 === Sliding-window PPL with state-step noise
# =============================================================

def ppl_with_state_noise(model, ids, sigma_dict, seed,
                          ctx_len=1024, stride=512, n_windows=10):
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
# === CELL V7 === Sweep: σ_rel × six conditions
# =============================================================

CTX_LEN     = 1024
STRIDE      = 512
N_WINDOWS   = 10
SEED        = 42
SIGMA_RELS  = [0.05, 0.10, 0.20]

PROFILE_LABELS = [
    "uniform", "proxy_C2", "proxy_correct", "full", "anti_full",
]

test_ids = tok.encode(test_text, return_tensors="pt").to("cuda")
print(f"\n{'-' * 80}")
print(f"WikiText-2 sliding-window PPL  "
      f"(ctx={CTX_LEN}, stride={STRIDE}, n_windows={N_WINDOWS})  "
      f"sigma_rel sweep = {SIGMA_RELS}")
print(f"{'-' * 80}")

# FP16 baseline (single shot, σ-independent)
fp16_ppl, _ = ppl_with_state_noise(mdl, test_ids, None, SEED,
                                      ctx_len=CTX_LEN, stride=STRIDE,
                                      n_windows=N_WINDOWS)
print(f"\n  FP16 baseline ppl = {fp16_ppl:.4f}\n")

sweep_results = {}

for sigma_rel in SIGMA_RELS:
    profiles = build_profiles(captured, sigma_rel)
    print(f"  --- sigma_rel = {sigma_rel} ---")
    sweep_results[f"{sigma_rel}"] = {}
    for label in PROFILE_LABELS:
        ppl, n_win = ppl_with_state_noise(
            mdl, test_ids, profiles[label], SEED,
            ctx_len=CTX_LEN, stride=STRIDE, n_windows=N_WINDOWS,
        )
        dppl = ppl - fp16_ppl
        pct  = 100.0 * dppl / fp16_ppl
        sweep_results[f"{sigma_rel}"][label] = {
            "ppl":          ppl,
            "delta_ppl":    dppl,
            "pct_of_base":  pct,
            "n_windows":    n_win,
        }
        print(f"    {label:<16s}  ppl={ppl:>9.4f}  "
              f"ΔPPL={dppl:+8.4f}  ({pct:+8.3f} % of base)")
    print()


# =============================================================
# === CELL V8 === Sweep summary table + verdicts
# =============================================================

print(f"\n{'=' * 80}")
print(f"Phase C-3b sweep summary  (ΔPPL % of FP16 baseline)")
print(f"{'=' * 80}\n")

# Wide table: rows = profile, cols = σ_rel
print(f"  {'profile':<16s}  " +
      "  ".join(f"σ_rel={s:>4.2f}" for s in SIGMA_RELS))
for label in PROFILE_LABELS:
    cells = []
    for s in SIGMA_RELS:
        pct = sweep_results[f"{s}"][label]["pct_of_base"]
        cells.append(f"{pct:>+9.3f}")
    print(f"  {label:<16s}  " + "  ".join(cells))

# Headline gaps as a function of σ_rel
print(f"\n  Key gap ratios (vs structured allocator family):\n")
print(f"  {'metric':<32s}  " +
      "  ".join(f"σ_rel={s:>4.2f}" for s in SIGMA_RELS))
for metric, fn in [
    ("uniform / anti_full ratio",
     lambda r: r["uniform"]["pct_of_base"] /
                max(r["anti_full"]["pct_of_base"], EPS)),
    ("proxy_C2 / full ratio",
     lambda r: r["proxy_C2"]["pct_of_base"] /
                max(r["full"]["pct_of_base"], EPS)),
    ("anti_full / full ratio",
     lambda r: r["anti_full"]["pct_of_base"] /
                max(r["full"]["pct_of_base"], EPS)),
    ("(proxy_C2 - full) %-points",
     lambda r: r["proxy_C2"]["pct_of_base"] -
                r["full"]["pct_of_base"]),
]:
    cells = []
    for s in SIGMA_RELS:
        v = fn(sweep_results[f"{s}"])
        cells.append(f"{v:>+9.3f}")
    print(f"  {metric:<32s}  " + "  ".join(cells))

# Rank stability of `full`
print(f"\n  Is `full` the best profile at every σ_rel?")
full_wins_all = True
for s in SIGMA_RELS:
    r = sweep_results[f"{s}"]
    full_pct = r["full"]["pct_of_base"]
    others_pct = [r[l]["pct_of_base"] for l in PROFILE_LABELS
                    if l != "full"]
    is_best = all(full_pct < o for o in others_pct)
    print(f"    σ_rel={s}: full={full_pct:+.3f}  "
          f"min(others)={min(others_pct):+.3f}  "
          f"full_is_best={'✓' if is_best else '✗'}")
    if not is_best:
        full_wins_all = False

# Uniform pathology character
print(f"\n  uniform vs anti_full pattern:")
for s in SIGMA_RELS:
    r = sweep_results[f"{s}"]
    u = r["uniform"]["pct_of_base"]
    a = r["anti_full"]["pct_of_base"]
    ratio = u / max(a, EPS)
    if ratio > 3.0:
        descr = "uniform CATASTROPHIC vs anti_full"
    elif ratio > 1.5:
        descr = "uniform substantially worse"
    elif ratio > 1.05:
        descr = "uniform mildly worse"
    elif abs(ratio - 1) < 0.05:
        descr = "uniform ≈ anti_full"
    else:
        descr = "anti_full WORSE than uniform (predicted regime)"
    print(f"    σ_rel={s}: uniform={u:+.3f}  anti_full={a:+.3f}  "
          f"ratio={ratio:.2f}  →  {descr}")


# =============================================================
# === CELL V9 === Save
# =============================================================

# Compact per-layer summary (small)
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
        "rms_h_p50":             float(rms.quantile(0.50)),
        "rms_h_max":             float(rms.max()),
    })

output = {
    "date":           datetime.utcnow().isoformat(timespec="seconds"),
    "phase":          "C-3b",
    "purpose":        ("σ_rel sweep on the C-3 state-step "
                       "quantisation design. Five structured "
                       "profiles + uniform are run at "
                       "σ_rel ∈ {0.05, 0.10, 0.20} to "
                       "characterise (i) the noise-amplitude "
                       "regime where uniform's IIR-accumulation "
                       "pathology becomes catastrophic, and "
                       "(ii) the rank stability of `full` "
                       "across the curve."),
    "model":          MODEL_ID,
    "label":          MODEL_LABEL,
    "protocol": {
        "ctx_len":      CTX_LEN,
        "stride":       STRIDE,
        "n_windows":    N_WINDOWS,
        "sigma_rels":   SIGMA_RELS,
        "rng_seed":     SEED,
        "calib_split":  "WikiText-2 train",
        "test_split":   "WikiText-2 test",
        "noise_model":  "post-state-update Gaussian, σ_g = σ_rel·shape(g)·RMS_h(g)",
        "profiles":     PROFILE_LABELS,
    },
    "fp16_baseline_ppl":   fp16_ppl,
    "sweep_results":       sweep_results,
    "full_wins_all_sigma": full_wins_all,
    "rho_gamma_rms_summary": summary,
}

out_path = "/content/phase_c3b_sigma_rel_sweep.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\nSaved -> {out_path}")


# =============================================================
# === READING ===
# =============================================================
#
# Look for two storylines in the sweep:
#
#   (1) full's rank stability:
#         If `full` is the best profile at every σ_rel
#         (full_wins_all=True), the bit allocator is robust
#         across the noise-amplitude curve. Use this in the
#         paper as evidence that σ_rel = 0.10 wasn't a sweet
#         spot.
#
#   (2) uniform's pathology shape:
#
#       At σ_rel = 0.05:
#         If uniform / anti_full ratio is small (< 1.5),
#         uniform's pathology is amplitude-driven —
#         catastrophe only appears at moderate-to-high noise.
#         This is the most physically interpretable case
#         (long-memory accumulation grows with σ²).
#
#         If the ratio is already large (> 3) at σ_rel = 0.05,
#         uniform's pathology is structural — even tiny noise
#         applied uniformly to all long-memory units corrupts
#         them in concert. This would be a stronger paper
#         claim.
#
#       At σ_rel = 0.20:
#         Expect uniform to blow up super-linearly (σ²
#         scaling on the long-memory tail). proxy / full
#         family should grow more linearly. The gap should
#         widen relative to σ_rel = 0.10.
#
#   (3) full vs proxy_C2 margin:
#         If (proxy_C2 - full) %-points grows with σ_rel,
#         the Γ-aware allocator's advantage compounds at
#         higher noise. This is the natural read for a
#         quantisation paper since real low-bit deployment
#         operates at "high enough" σ.
#
#         If the margin shrinks at σ_rel = 0.20, the
#         allocator hits its own ceiling — possibly because
#         the high-Γ tail saturates the readout regardless of
#         which profile is used. Worth reporting either way.
# ============================================================
