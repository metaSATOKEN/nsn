# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Phase C-1c-bis: Calibration robustness of Γ_s, ρ_emp, RMS(h)
# ============================================================
# Foundation check before Phase C-4 (σ → b bridge) and C-5
# (hybrid stack). Phases C-1c / C-3 / C-3b all calibrated on
# WikiText-2 *train* and evaluated on *test*. The bit allocator
# uses Γ_s (output coupling) calibrated on train; if Γ_s drifts
# materially between texts, the per-state-unit bit allocation
# is calibration-text-overfit and the C-3 / C-3b results are
# tied to the calibration source.
#
# This script re-calibrates ρ, Γ, RMS(h) on WikiText-2
# *validation* and compares with WikiText-2 train.
#
# Decision rule (pre-registered):
#
#   (a) Pearson(log Γ_train, log Γ_val) ≥ 0.95
#       AND bit-tier-match (full R_g) ≥ 80 %:
#         → Calibration is stable. C-3 / C-3b calibration is
#           safe foundation for C-4 / C-5. Move on.
#
#   (b) Pearson 0.85 - 0.95 OR bit-tier-match 60 - 80 %:
#         → Calibration is "usable but flag". Paper C should
#           note that ~10 - 20 % of units shift bit tiers
#           across calibration sources, motivating either
#           (i) calibration ensembling or (ii) more bits as
#           safety margin in deployment. Move on with caveat.
#
#   (c) Pearson < 0.85 OR bit-tier-match < 60 %:
#         → Calibration robustness becomes its own Paper C
#           contribution. Pause C-4 / C-5 and design a
#           calibration-stable allocator (e.g., Γ averaged
#           across multiple text sources, or a held-out
#           tail-trimmed estimator).
#
# Runtime: ~5 - 10 min on A100 80 GB. The bulk is two
# calibration passes (~1 min each on Python slow path) plus
# JSON I/O.
# ============================================================


# =============================================================
# === CELL V1 === Setup  (mirrors C-3 / C-3b)
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
# === CELL V3 === Monkey-patch slow_forward — calibrate mode
#                 only (identical math to C-3, no noise path)
# =============================================================

_C1B = {
    "mode":         "off",       # "calibrate" | "off"
    "calib_accum":  None,
}


def _c1b_mamba_forward(mix, input_states):
    layer_name = LAYER_NAME_BY_ID.get(id(mix))
    B, T, _ = input_states.shape
    d_inner = mix.intermediate_size
    d_state = mix.ssm_state_size
    dt_rank = mix.time_step_rank

    proj = mix.in_proj(input_states)
    x_pre, gate = proj.chunk(2, dim=-1)

    # mix.conv1d already has padding=ksize-1; no extra F.pad.
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

    if _C1B["mode"] == "calibrate":
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
        if _C1B["mode"] == "calibrate":
            rms_sumsq += (h ** 2).sum(dim=0)
        y_t = (h * C_f[:, t].unsqueeze(1)).sum(dim=-1)
        ys.append(y_t)
    y = torch.stack(ys, dim=1)

    y = y + mix.D.float().unsqueeze(0).unsqueeze(0) * x_pre.float()
    y = y * F.silu(gate.float())
    out = mix.out_proj(y.to(mix.out_proj.weight.dtype))

    if _C1B["mode"] == "calibrate":
        prev = _C1B["calib_accum"].get(layer_name)
        if prev is None:
            _C1B["calib_accum"][layer_name] = {
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


def _c1b_slow_forward_patch(self, input_states, *args, **kwargs):
    return _c1b_mamba_forward(self, input_states)


_MixerClass = type(mixers[0][1])
_orig_slow_forward = _MixerClass.slow_forward
_MixerClass.slow_forward = _c1b_slow_forward_patch
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
# === CELL V4 === Two calibration passes — train vs validation
# =============================================================

def calibrate_on(text_str, label, max_chars=8000, max_tokens=1024):
    """Run one calibration pass on the given text. Returns
    {layer_name: {rho_emp [d_inner, d_state],
                  gamma   [d_state],
                  rms_h   [d_inner, d_state]}}."""
    snippet = text_str[:max_chars]
    ids = tok.encode(snippet, return_tensors="pt").to("cuda")[:, :max_tokens]
    print(f"\n  Calibrating on {label}  "
          f"(text {len(snippet)} chars, {ids.shape[1]} tokens)")

    _C1B["mode"]        = "calibrate"
    _C1B["calib_accum"] = {}
    with torch.inference_mode():
        _ = mdl(ids, use_cache=False)
    _C1B["mode"] = "off"

    out = {}
    for name, accum in _C1B["calib_accum"].items():
        rho   = accum["rho_sum"]   / accum["rho_count"]
        gamma = accum["gamma_sum"] / accum["gamma_count"]
        rms   = (accum["rms_sumsq"] /
                  accum["rms_count"]).clamp(min=1e-12).sqrt()
        out[name] = {"rho_emp": rho, "gamma": gamma, "rms_h": rms}
    print(f"    captured for {len(out)} layers")
    return out


print(f"\n{'-' * 80}")
print("Loading WikiText-2 train + validation splits")
print(f"{'-' * 80}")

ds_train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
ds_val   = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
train_text = "\n\n".join([x for x in ds_train["text"] if x.strip()])
val_text   = "\n\n".join([x for x in ds_val["text"]   if x.strip()])

cap_train = calibrate_on(train_text, "WikiText-2 train")
cap_val   = calibrate_on(val_text,   "WikiText-2 validation")


# =============================================================
# === CELL V5 === Compare per-layer and global statistics
# =============================================================

EPS = 1e-6


def pearson_log(a, b):
    """Pearson correlation between log(a) and log(b), with
    EPS clamp. Both flat 1-D tensors."""
    la = torch.log(a.flatten().clamp(min=EPS))
    lb = torch.log(b.flatten().clamp(min=EPS))
    la = la - la.mean()
    lb = lb - lb.mean()
    denom = (la.pow(2).sum() * lb.pow(2).sum()).sqrt().clamp(min=EPS)
    return float((la * lb).sum() / denom)


def bit_tier(rho, gamma, b_0=4, eps=EPS):
    """Compute b_g = b_0 + ceil(log2(Gamma_s / (1 - rho))).
    rho:   [d_inner, d_state]
    gamma: [d_state]
    returns bit tier per (c, s) as [d_inner, d_state] long tensor.
    """
    R = (gamma.unsqueeze(0).clamp(min=eps) /
          (1.0 - rho).clamp(min=eps))
    extra = torch.ceil(torch.log2(R.clamp(min=eps))).clamp(min=0)
    return (b_0 + extra).long()


# Per-layer comparison table
print(f"\n{'=' * 80}")
print(f"Per-layer Γ stability (train vs validation)")
print(f"{'=' * 80}\n")
print(f"  {'layer':<35s}  "
      f"{'Pearson(log Γ)':>15s}  {'Pearson(log ρ)':>15s}  "
      f"{'Pearson(log RMS)':>17s}  {'Γ_max ratio':>12s}")

per_layer_stats = []
for name in cap_train:
    g_t = cap_train[name]["gamma"]
    g_v = cap_val[name]["gamma"]
    r_t = cap_train[name]["rho_emp"]
    r_v = cap_val[name]["rho_emp"]
    h_t = cap_train[name]["rms_h"]
    h_v = cap_val[name]["rms_h"]

    p_g = pearson_log(g_t, g_v)
    p_r = pearson_log(r_t, r_v)
    p_h = pearson_log(h_t, h_v)
    gamma_max_ratio = float(
        (g_v.max().clamp(min=EPS) / g_t.max().clamp(min=EPS))
    )
    per_layer_stats.append({
        "layer_name":         name,
        "pearson_log_gamma":  p_g,
        "pearson_log_rho":    p_r,
        "pearson_log_rms":    p_h,
        "gamma_max_ratio":    gamma_max_ratio,
    })
    short = name.replace("backbone.", "").replace(".mixer", "")
    print(f"  {short:<35s}  "
          f"{p_g:>15.4f}  {p_r:>15.4f}  {p_h:>17.4f}  "
          f"{gamma_max_ratio:>12.4f}")


# =============================================================
# === CELL V6 === Global aggregate + bit-tier match
# =============================================================

# Concatenate across all layers
all_g_t = torch.cat([cap_train[n]["gamma"] for n in cap_train])
all_g_v = torch.cat([cap_val[n]["gamma"]   for n in cap_train])
all_r_t = torch.cat([cap_train[n]["rho_emp"].flatten() for n in cap_train])
all_r_v = torch.cat([cap_val[n]["rho_emp"].flatten()   for n in cap_train])
all_h_t = torch.cat([cap_train[n]["rms_h"].flatten() for n in cap_train])
all_h_v = torch.cat([cap_val[n]["rms_h"].flatten()   for n in cap_train])

global_p_g = pearson_log(all_g_t, all_g_v)
global_p_r = pearson_log(all_r_t, all_r_v)
global_p_h = pearson_log(all_h_t, all_h_v)

print(f"\n{'=' * 80}")
print(f"Global stability (concatenated across all 64 layers)")
print(f"{'=' * 80}")
print(f"  Pearson(log Γ_s):    {global_p_g:.4f}")
print(f"  Pearson(log ρ_emp):  {global_p_r:.4f}  (sanity: should be very high)")
print(f"  Pearson(log RMS_h):  {global_p_h:.4f}")

# Bit-tier match (full R formula)
print(f"\n  Bit-tier match using b_g = b_0 + ⌈log₂(Γ/(1-ρ))⌉:")
match_per_layer = []
overall_match = 0
overall_total = 0
shift_hist = {-3: 0, -2: 0, -1: 0, 0: 0, 1: 0, 2: 0, 3: 0}
for name in cap_train:
    g_t = cap_train[name]["gamma"]
    g_v = cap_val[name]["gamma"]
    r_t = cap_train[name]["rho_emp"]
    r_v = cap_val[name]["rho_emp"]
    bt_t = bit_tier(r_t, g_t)
    bt_v = bit_tier(r_v, g_v)
    n = bt_t.numel()
    n_match = int((bt_t == bt_v).sum())
    overall_match += n_match
    overall_total += n
    diff = (bt_v - bt_t).flatten()
    for d in [-3, -2, -1, 0, 1, 2, 3]:
        if d == -3:
            shift_hist[d] += int((diff <= -3).sum())
        elif d == 3:
            shift_hist[d] += int((diff >= 3).sum())
        else:
            shift_hist[d] += int((diff == d).sum())
    match_per_layer.append({
        "layer_name":   name,
        "match_rate":   n_match / n,
        "n_units":      n,
    })
overall_match_rate = overall_match / overall_total
print(f"    overall match rate:  "
      f"{100.0 * overall_match_rate:.2f} %  "
      f"({overall_match} / {overall_total})")

print(f"\n  Bit-tier shift distribution (val − train, per (c, s) unit):")
for d in [-3, -2, -1, 0, 1, 2, 3]:
    pct = 100.0 * shift_hist[d] / overall_total
    label = (f"≤ −3" if d == -3 else f"≥ +3" if d == 3
             else f"{d:+d}")
    print(f"    {label:<5s}  "
          f"{shift_hist[d]:>10d}  ({pct:>6.3f} %)")


# =============================================================
# === CELL V7 === Verdict
# =============================================================

print(f"\n{'=' * 80}")
print(f"Phase C-1c-bis verdict")
print(f"{'=' * 80}")

if global_p_g >= 0.95 and overall_match_rate >= 0.80:
    verdict_class = "STABLE"
    verdict = (f"Calibration is stable. C-3 / C-3b's foundation "
                f"is safe for C-4 / C-5. Pearson(log Γ) = "
                f"{global_p_g:.3f}, bit-tier match = "
                f"{100*overall_match_rate:.1f} %, both above the "
                f"pre-registered 0.95 / 80 % thresholds. Move on.")
elif global_p_g >= 0.85 and overall_match_rate >= 0.60:
    verdict_class = "USABLE_WITH_FLAG"
    verdict = (f"Calibration is usable but should be flagged in "
                f"the paper. Pearson(log Γ) = {global_p_g:.3f}, "
                f"bit-tier match = {100*overall_match_rate:.1f} %. "
                f"~{100*(1-overall_match_rate):.0f} % of units "
                f"shift bit tiers across calibration sources. "
                f"Paper C should note this as a deployment "
                f"caveat (calibration ensembling or a +1 bit "
                f"safety margin) but the core results stand.")
else:
    verdict_class = "REQUIRES_OWN_PHASE"
    verdict = (f"Calibration is unstable. Pearson(log Γ) = "
                f"{global_p_g:.3f}, bit-tier match = "
                f"{100*overall_match_rate:.1f} %. Below pre-"
                f"registered thresholds. Calibration robustness "
                f"becomes its own Paper C contribution. Pause "
                f"C-4 / C-5 design and either (i) average Γ over "
                f"multiple text sources or (ii) tail-trim the "
                f"Γ estimator before taking log.")

print(f"\n  Class:  {verdict_class}")
print(f"\n  {verdict}")


# =============================================================
# === CELL V8 === Save
# =============================================================

# Layer summary (compact)
summary = []
for stat in per_layer_stats:
    n = stat["layer_name"]
    g_t = cap_train[n]["gamma"]
    g_v = cap_val[n]["gamma"]
    summary.append({
        **stat,
        "gamma_train_p50":  float(g_t.quantile(0.50)),
        "gamma_val_p50":    float(g_v.quantile(0.50)),
        "gamma_train_max":  float(g_t.max()),
        "gamma_val_max":    float(g_v.max()),
    })

output = {
    "date":           datetime.utcnow().isoformat(timespec="seconds"),
    "phase":          "C-1c-bis",
    "purpose":        ("Calibration robustness check for Γ_s, ρ, "
                       "RMS(h) on Mamba-2.8B-hf. Compares "
                       "WikiText-2 train (the source used by "
                       "C-1c / C-3 / C-3b) against WikiText-2 "
                       "validation. Decides whether bit "
                       "allocation tied to a single calibration "
                       "source is safe for Phase C-4 / C-5."),
    "model":          MODEL_ID,
    "label":          MODEL_LABEL,
    "protocol": {
        "calib_train_chars":  8000,
        "calib_val_chars":    8000,
        "calib_max_tokens":   1024,
        "bit_allocation_b0":  4,
        "bit_allocation_formula": "b_g = b_0 + ceil(log2(Gamma/(1-rho)))",
    },
    "global_pearson_log_gamma":   global_p_g,
    "global_pearson_log_rho":     global_p_r,
    "global_pearson_log_rms":     global_p_h,
    "overall_bit_tier_match":     overall_match_rate,
    "bit_tier_shift_histogram":   shift_hist,
    "per_layer_stats":            summary,
    "match_per_layer":            match_per_layer,
    "verdict_class":              verdict_class,
    "verdict":                    verdict,
}

out_path = "/content/phase_c1c_bis_calibration_robustness.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\nSaved -> {out_path}")


# =============================================================
# === READING ===
# =============================================================
#
# What the per-layer Γ Pearson tells you:
#
#   Per-layer Pearson > 0.95 in (almost) every layer:
#     Γ is text-stable. The 174× - 1572× per-layer max/min
#     ratio reported in C-1c is a *structural* property of the
#     model, not a calibration artefact.
#
#   A few layers with lower Pearson (< 0.9):
#     Those layers' Γ tail is calibration-text-driven. Likely
#     the deepest layers (32-40, the long-context core) since
#     they pick up text-genre features. Worth a per-layer note
#     in the paper.
#
#   Many layers with low Pearson (<< 0.9):
#     Γ extraction itself is unreliable. Calibration becomes
#     a Paper C contribution.
#
# What ρ Pearson tells you:
#
#   Should be very close to 1.0 (think 0.999+). ρ is determined
#   by A_log (a static parameter) modulated by dt (which weakly
#   depends on input). If ρ Pearson is materially below 1.0,
#   the slow_forward replay is noisier than expected and the
#   calibration step has a numerical bug.
#
# What RMS(h) Pearson tells you:
#
#   Should be high (> 0.95) since RMS(h) is the recurrent
#   integration of B_bar * x, and most layers' RMS scale is set
#   by static parameters. Lower Pearson means the dynamic range
#   of h is text-driven, which would affect σ_g = σ_rel * shape
#   * RMS scaling in C-3 / C-3b.
#
# Bit-tier shift distribution:
#
#   Most units in the 0 column: stable
#   Significant ±1 mass: 1-bit safety margin needed in
#                         deployment
#   Significant ±2 or larger: serious calibration drift
# ============================================================
