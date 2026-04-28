# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Phase C-1c: Output-coupling Γ extraction + full R_g check
# ============================================================
# Paper C follow-up. Completes the prospectus's R_g formula on
# Mamba-2.8B-hf and answers a single concrete question for the
# bit allocator design:
#
#   Is the cheap rho-only proxy
#       R_proxy(c, s) = 1 / (1 - rho_emp(c, s))
#   highly correlated (per state unit) with the full Paper C
#   formula
#       R_full(c, s)  = || Gamma_s || / (1 - rho_emp(c, s))
#   such that the bit allocations differ by at most 1 bit on
#   most state units?
#
# If yes, the cheap allocator is a sufficient practical
# approximation, and Phase C-2's "sigma_g ∝ (1 - rho_g)" profile
# was an essentially valid implementation of the Paper C bit
# allocator. If no, we need to extract Gamma during calibration
# and use the full formula.
#
# Mamba-1 specifics. The output equation is
#       y_{t,c} = sum_s C_{t,s} * h_{t,c,s}
# so C is per-state (varies in s, not in c). The output coupling
# is therefore
#       Gamma_s = E_t [ C_{t,s}^2 ]    (one scalar per state,
#                                       d_state values per layer)
# and the full per-(c, s) R is
#       R_full(c, s) = Gamma_s / (1 - rho_emp(c, s))
#
# The proxy R differs from the full R only by the per-state
# scalar Gamma_s. If Gamma_s is roughly flat across s, the proxy
# is essentially proportional to the full and bit allocations
# match. If Gamma_s is heavy-tailed, specific state indices
# dominate and the proxy allocator is wrong.
#
# This is the cheapest possible test of the prospectus's "full
# R_g" claim.
#
# Runtime ~5 min on A100 80 GB once the model is loaded.
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


# =============================================================
# === CELL V3 === Calibration: capture rho_emp AND Gamma_s
# =============================================================
#
# Same machinery as C-1b but additionally extracts C from the
# x_proj split and computes Gamma_s = mean(C^2 over B and T).
# =============================================================

def capture_rho_and_gamma(mdl, mixers, calib_ids):
    captured = {}
    handles  = []

    def make_hook(layer_name):
        def pre_hook(module, args, kwargs):
            if args:
                x = args[0]
            elif "hidden_states" in kwargs:
                x = kwargs["hidden_states"]
            else:
                x = None
            if x is not None and torch.is_tensor(x):
                captured[layer_name] = x.detach()
        return pre_hook

    for name, mix in mixers:
        h = mix.register_forward_pre_hook(make_hook(name),
                                            with_kwargs=True)
        handles.append(h)
    with torch.inference_mode():
        _ = mdl(calib_ids, use_cache=False)
    for h in handles:
        h.remove()

    out = {}  # layer_name -> {"rho_emp": [d_inner, d_state],
              #                "gamma":   [d_state]}
    for name, mix in mixers:
        if name not in captured:
            continue
        x = captured[name]
        d_inner = mix.intermediate_size
        d_state = mix.ssm_state_size
        dt_rank = mix.time_step_rank

        with torch.inference_mode():
            T = x.shape[1]
            B = x.shape[0]
            y = mix.in_proj(x.to(mix.in_proj.weight.dtype))
            _, x_pre = y.chunk(2, dim=-1)
            x_pre_c = x_pre.transpose(1, 2).contiguous()
            ksize = mix.conv1d.weight.shape[-1]
            x_pre_c = F.pad(x_pre_c, (ksize - 1, 0))
            x_pre_c = mix.conv1d(x_pre_c)[:, :, :T]
            x_pre = F.silu(x_pre_c).transpose(1, 2)
            xbc = mix.x_proj(x_pre.to(mix.x_proj.weight.dtype))
            dt_raw, B_proj, C_proj = xbc.split(
                [dt_rank, d_state, d_state], dim=-1
            )
            # === NEW: per-state output coupling ===
            # C_proj shape: [B, T, d_state]
            # Gamma_s = E_t [ C_{t,s}^2 ]
            gamma_s = (C_proj.float() ** 2).mean(dim=(0, 1))  # [d_state]

            # === Same as C-1b: rho_emp per (channel, state) ===
            dt = mix.dt_proj(dt_raw.to(mix.dt_proj.weight.dtype))
            dt = F.softplus(dt.float())
            A_neg = -torch.exp(mix.A_log.detach().float())
            half = T // 2
            sum_A_bar = torch.zeros((d_inner, d_state),
                                     dtype=torch.float32,
                                     device=dt.device)
            count = 0
            for t_lo in (0, half):
                t_hi = min(t_lo + half, T)
                if t_hi <= t_lo:
                    continue
                A_bar_chunk = torch.exp(
                    dt[:, t_lo:t_hi, :, None] *
                    A_neg.unsqueeze(0).unsqueeze(0)
                )
                sum_A_bar += A_bar_chunk.sum(dim=(0, 1))
                count    += B * (t_hi - t_lo)
            rho_emp = (sum_A_bar / count).detach().cpu()

        out[name] = {
            "rho_emp": rho_emp,           # [d_inner, d_state]
            "gamma":   gamma_s.detach().cpu(),  # [d_state]
        }
    return out


print(f"\n{'-' * 80}")
print("Calibrating rho_emp + Gamma_s on a 1024-token window")
print(f"{'-' * 80}")
calib_text = "\n\n".join(
    [x for x in load_dataset("wikitext", "wikitext-2-raw-v1",
                              split="train")["text"]
     if x.strip()])[:8000]
calib_ids = tok.encode(calib_text, return_tensors="pt").to("cuda")[:, :1024]
captured = capture_rho_and_gamma(mdl, mixers, calib_ids)
print(f"  captured {len(captured)} layers")


# =============================================================
# === CELL V4 === Per-layer R_full vs R_proxy comparison
# =============================================================

EPS = 1e-4
B0  = 3.0  # baseline bit budget for the bit allocator

per_layer = []
all_rho   = []
all_R_proxy = []
all_R_full  = []
all_b_proxy = []
all_b_full  = []

for name, d in captured.items():
    rho   = d["rho_emp"]              # [d_inner, d_state]
    gamma = d["gamma"]                # [d_state]
    d_inner, d_state = rho.shape

    # rho-only proxy: R_proxy(c, s) = 1 / (1 - rho(c, s))
    R_proxy = 1.0 / (1.0 - rho.clamp(max=1.0 - EPS) + EPS)
    # full:        R_full(c, s)  = gamma_s / (1 - rho(c, s))
    # gamma_s broadcasts over c
    R_full  = gamma.unsqueeze(0) * R_proxy

    # bit allocation under each:
    # b = b_0 + ceil(log_2 R)+
    b_proxy = B0 + torch.clamp(
        torch.ceil(torch.log2(R_proxy.clamp(min=1.0))), min=0.0
    )
    b_full  = B0 + torch.clamp(
        torch.ceil(torch.log2(R_full.clamp(min=1.0))), min=0.0
    )

    # log-log Pearson correlation per layer (good summary even
    # with heavy-tailed R distributions)
    log_proxy = torch.log10(R_proxy.flatten() + EPS)
    log_full  = torch.log10(R_full.flatten()  + EPS)
    pcorr = float(torch.corrcoef(
        torch.stack([log_proxy, log_full])
    )[0, 1])

    # bit-tier disagreement: how many units get a different bit?
    b_diff   = (b_full - b_proxy).int()
    n_units  = int(rho.numel())
    n_same   = int((b_diff == 0).sum())
    n_off1   = int((b_diff.abs() == 1).sum())
    n_off2p  = int((b_diff.abs() >= 2).sum())

    info = {
        "layer_name":      name,
        "d_inner":         int(d_inner),
        "d_state":         int(d_state),
        "n_units":         n_units,
        "gamma":           gamma.tolist(),
        "gamma_min":       float(gamma.min()),
        "gamma_p50":       float(gamma.quantile(0.50)),
        "gamma_max":       float(gamma.max()),
        "gamma_max_minus_min_ratio": float(gamma.max()
                                            / gamma.min().clamp(min=1e-12)),
        "log_log_pearson_R_proxy_vs_R_full": pcorr,
        "bit_tier_match_pct":         100.0 * n_same  / n_units,
        "bit_tier_off_by_1_pct":      100.0 * n_off1  / n_units,
        "bit_tier_off_by_2_or_more_pct": 100.0 * n_off2p / n_units,
        "rho_p50":         float(rho.flatten().quantile(0.50)),
    }
    per_layer.append(info)

    all_rho.extend(rho.flatten().tolist())
    # Avoid blowing up the global lists with 5.24M floats: only
    # collect a downsampled slice for global corr / scatter.
    n_sample = min(20000, n_units)
    idx = torch.randperm(n_units)[:n_sample]
    all_R_proxy.extend(R_proxy.flatten()[idx].tolist())
    all_R_full.extend(R_full.flatten()[idx].tolist())
    all_b_proxy.extend(b_proxy.flatten()[idx].tolist())
    all_b_full.extend(b_full.flatten()[idx].tolist())


# =============================================================
# === CELL V5 === Aggregate readout
# =============================================================

print(f"\n{'=' * 80}")
print("Phase C-1c readout: rho-only proxy vs full R_g (with Gamma)")
print(f"{'=' * 80}")

# Per-layer Gamma summary
print(f"\n  Gamma_s heavy-tailedness per layer (max / min ratio):")
print(f"  {'layer':<55s}  {'g_min':>9s}  {'g_p50':>9s}  "
      f"{'g_max':>9s}  {'max/min':>8s}")
for info in per_layer[::8]:  # every 8th layer
    print(f"  {info['layer_name']:<55s}  "
          f"{info['gamma_min']:>9.4f}  "
          f"{info['gamma_p50']:>9.4f}  "
          f"{info['gamma_max']:>9.4f}  "
          f"{info['gamma_max_minus_min_ratio']:>7.2f}x")

# Global log-log pearson
all_R_proxy_t = torch.tensor(all_R_proxy)
all_R_full_t  = torch.tensor(all_R_full)
log_p = torch.log10(all_R_proxy_t + EPS)
log_f = torch.log10(all_R_full_t  + EPS)
global_pearson = float(torch.corrcoef(
    torch.stack([log_p, log_f])
)[0, 1])

print(f"\n  Global log-log Pearson(R_proxy, R_full)  "
      f"(downsampled n={len(all_R_proxy)}):  {global_pearson:.4f}")

# Per-layer pearson distribution
layer_pearsons = [info["log_log_pearson_R_proxy_vs_R_full"]
                   for info in per_layer]
layer_pearsons_t = torch.tensor(layer_pearsons)
print(f"  Per-layer log-log Pearson distribution (n={len(per_layer)} layers):")
print(f"    min   = {layer_pearsons_t.min().item():.4f}")
print(f"    p25   = {layer_pearsons_t.quantile(0.25).item():.4f}")
print(f"    p50   = {layer_pearsons_t.quantile(0.50).item():.4f}")
print(f"    p75   = {layer_pearsons_t.quantile(0.75).item():.4f}")
print(f"    max   = {layer_pearsons_t.max().item():.4f}")

# Bit-tier agreement
match_pcts = [info["bit_tier_match_pct"]   for info in per_layer]
off1_pcts  = [info["bit_tier_off_by_1_pct"]    for info in per_layer]
off2_pcts  = [info["bit_tier_off_by_2_or_more_pct"] for info in per_layer]

print(f"\n  Bit-tier agreement between b_proxy and b_full "
      f"(b_0 = {B0}):")
print(f"    {'layer':<55s}  {'same %':>8s}  {'±1 %':>8s}  "
      f"{'≥2 off %':>10s}")
for info in per_layer[::8]:
    print(f"    {info['layer_name']:<55s}  "
          f"{info['bit_tier_match_pct']:>7.2f}%  "
          f"{info['bit_tier_off_by_1_pct']:>7.2f}%  "
          f"{info['bit_tier_off_by_2_or_more_pct']:>9.2f}%")

import statistics as stats
print(f"\n  Bit-tier agreement aggregate across all layers:")
print(f"    mean same:    {stats.mean(match_pcts):>7.2f} %")
print(f"    mean ±1:      {stats.mean(off1_pcts):>7.2f} %")
print(f"    mean ≥2 off:  {stats.mean(off2_pcts):>7.2f} %")


# =============================================================
# === CELL V6 === Verdict
# =============================================================

print(f"\n{'=' * 80}")
print("Verdict")
print(f"{'=' * 80}")

verdict_threshold_pearson = 0.95
verdict_threshold_offby1  = 90.0  # at least this % of units within ±1 bit

mean_within_1bit = stats.mean(
    [m + o for m, o in zip(match_pcts, off1_pcts)]
)

if (global_pearson >= verdict_threshold_pearson and
        mean_within_1bit >= verdict_threshold_offby1):
    verdict = (f"RHO-ONLY PROXY IS SUFFICIENT. Global log-log "
                f"Pearson(R_proxy, R_full) = {global_pearson:.3f} "
                f">= {verdict_threshold_pearson}, and on average "
                f"{mean_within_1bit:.1f} % of state units agree to "
                f"within ±1 bit between proxy and full allocators. "
                f"Phase C-2's sigma_g ∝ (1 - rho_g) implementation "
                f"is a faithful approximation of the Paper C bit "
                f"allocator. Gamma_s extraction is not required for "
                f"the deployed allocator -- it can be reserved for "
                f"diagnostic refinement.")
elif global_pearson < 0.7:
    verdict = (f"FULL R_g IS REQUIRED. Global log-log "
                f"Pearson(R_proxy, R_full) = {global_pearson:.3f} "
                f"< 0.7. The Gamma_s coupling carries information "
                f"that the rho-only proxy throws away. Phase C-2's "
                f"approximation is materially wrong; future runs "
                f"must extract Gamma during calibration and use "
                f"the full sigma_g ∝ sqrt((1 - rho_g) / Gamma_s) "
                f"profile.")
else:
    verdict = (f"BORDERLINE / FULL R RECOMMENDED. Global log-log "
                f"Pearson(R_proxy, R_full) = {global_pearson:.3f}, "
                f"and {mean_within_1bit:.1f} % of state units agree "
                f"within ±1 bit. The proxy is approximately right "
                f"in direction but disagrees on bit allocation for a "
                f"non-trivial minority of units. For a Paper C-tier "
                f"allocator, computing Gamma during calibration is "
                f"recommended; for a quick / cheap allocator, the "
                f"proxy is acceptable.")

print(f"\n  {verdict}")


# =============================================================
# === CELL V7 === Save
# =============================================================

# Compact per-layer output (no per-state-unit lists);
# scatter sample is downsampled to 20K per layer in CELL V4.
sample = {
    "n":          len(all_R_proxy),
    "log_R_proxy": log_p.tolist()[:5000],   # downsample to 5K
    "log_R_full":  log_f.tolist()[:5000],
}

output = {
    "date":            datetime.utcnow().isoformat(timespec="seconds"),
    "phase":           "C-1c",
    "purpose":         ("Output-coupling Gamma_s extraction and "
                        "rho-only proxy vs full R_g comparison. "
                        "Tests whether Phase C-2's sigma_g ∝ "
                        "(1 - rho_g) profile is a faithful "
                        "approximation of the Paper C bit allocator "
                        "or whether Gamma_s extraction is required."),
    "model":           MODEL_ID,
    "label":           MODEL_LABEL,
    "calib_split":     "WikiText-2 train",
    "calib_tokens":    int(calib_ids.shape[1]),
    "n_layers":        len(per_layer),
    "b0":              B0,
    "global_log_log_pearson_R_proxy_vs_R_full": global_pearson,
    "per_layer":       per_layer,
    "scatter_sample":  sample,
    "verdict":         verdict,
    "verdict_thresholds": {
        "pearson_min":       verdict_threshold_pearson,
        "within_1bit_min":   verdict_threshold_offby1,
    },
}

out_path = "/content/phase_c1c_gamma_extraction.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\nSaved -> {out_path}")


# =============================================================
# === READING ===
# =============================================================
#
# This run resolves the proxy-vs-full question. Three branches:
#
#   RHO-ONLY PROXY IS SUFFICIENT:
#     The Phase C-2 / C-2b confirmations stand as-is. The
#     deployed Paper C bit allocator can use the cheap
#     sigma_g ∝ (1 - rho_g) profile without computing Gamma
#     during calibration. Gamma extraction reverts to a
#     diagnostic refinement, not a hard requirement.
#
#   FULL R_g IS REQUIRED:
#     Phase C-2 / C-2b are valid as direction tests but the
#     proxy implementation under-allocates bits to high-Gamma
#     state indices. Future allocator implementations must
#     extract Gamma during calibration. The Paper C narrative
#     gets a cleaner four-part story (rho captures the recurrent
#     side, Gamma captures the readout side).
#
#   BORDERLINE:
#     Hybrid recipe -- cheap allocator for memory-constrained
#     deployments, full allocator for quality-sensitive ones.
#
# Per-layer Gamma_s heavy-tailedness (max / min ratio) is the
# diagnostic quantity. If most layers have a ratio below ~3,
# Gamma is roughly flat across state indices and the proxy is
# sufficient. If many layers exhibit ratios above 10, certain
# state indices dominate the readout and the full formula is
# needed.
# ============================================================
