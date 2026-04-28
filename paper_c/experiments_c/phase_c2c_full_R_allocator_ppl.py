# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Phase C-2c: Full-R bit-allocator PPL test on Mamba-2.8B
# ============================================================
# Paper C end-to-end follow-up to Phase C-2b.
#
# Phase C-1c established that the cheap ρ-only proxy used in
# Phase C-2 / C-2b is materially different from the full Paper C
# R_g formula on Mamba-2.8B-hf:
#
#   R_proxy(c, s) = 1 / (1 - rho_emp(c, s))
#   R_full (c, s) = Gamma_s / (1 - rho_emp(c, s))
#
# Pearson(log R_proxy, log R_full) = 0.635, bit-tier same-bit
# agreement = 9.24 %. Phase C-1c verdict: "FULL R_g IS REQUIRED".
#
# Under the standard quantisation argument (uniform-rounding
# noise variance σ_g² ∝ 4^{-b_g}, total bit budget Σ b_g = const),
# minimising Σ R_g σ_g² subject to the budget gives
#
#   σ_g  ∝  1 / sqrt(R_g)
#
# i.e.
#
#   σ_proxy(c, s)  ∝  sqrt(1 - rho_emp(c, s))
#   σ_full (c, s)  ∝  sqrt( (1 - rho_emp(c, s)) / Gamma_s )
#
# Note this is sqrt(1 - rho), not (1 - rho) as Phase C-2 used.
# Both are in the right direction (more noise to short-memory
# units), but sqrt is the bit-budget-optimal scaling. Phase C-2
# used (1 - rho) directly because the ratio R formula it was
# implicitly inverting was
#
#   σ_g  ∝  1 / R_g  =  (1 - rho_g)
#
# which corresponds to a different objective (minimising Σ σ_g²
# subject to Σ √R_g σ_g = const, an unusual constraint). For
# C-2c we use the more standard
#
#   σ_full          ∝  sqrt( (1 - rho_emp) / Gamma_s )
#   σ_proxy_correct ∝  sqrt(1 - rho_emp)
#
# and report alongside the C-2 / C-2b "(1 - rho) directly"
# profile so the comparison is honest about which numbers come
# from which scaling choice.
#
# Conditions tested at σ₀ = 0.15, n_windows = 10:
#
#   1. FP16 baseline                         no perturbation
#   2. uniform                                σ_g = σ_0 const
#   3. proxy_C2  (Phase C-2 / C-2b shape)    σ ∝ (1 - rho)
#   4. proxy_correct                          σ ∝ sqrt(1 - rho)
#   5. **full**  (Paper C bit allocator)     σ ∝ sqrt((1 - rho)/Gamma_s)
#   6. anti_full (sanity check)               σ ∝ sqrt(Gamma_s/(1 - rho))
#
# All six conditions share total noise energy Σ σ_g² = const, so
# the comparison isolates *allocation* from *amount*.
#
# Predicted ranking:
#   full  <  proxy_correct  <  proxy_C2  <  uniform  <  anti_full
#
# If full < proxy_C2 (cleared with margin), Γ-awareness is
# materially valuable end-to-end. If full ≈ proxy_C2 despite
# Pearson 0.635, ρ-only allocation is a sufficient practical
# allocator even though Pearson said otherwise (= bits-vs-noise
# objective is more forgiving than rank correlation suggests).
#
# Runtime ~30 - 35 min on A100 80 GB (6 conditions, 10 windows).
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
# === CELL V3 === Calibration: rho_emp + Gamma_s (C-1c machinery)
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

    out = {}
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
            gamma_s = (C_proj.float() ** 2).mean(dim=(0, 1))  # [d_state]
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
print("Calibrating rho_emp + Gamma_s (C-1c machinery)")
print(f"{'-' * 80}")
ds_test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
test_text = "\n\n".join([x for x in ds_test["text"]
                          if x.strip()])[:48000]
calib_text = "\n\n".join(
    [x for x in load_dataset("wikitext", "wikitext-2-raw-v1",
                              split="train")["text"]
     if x.strip()])[:8000]
calib_ids = tok.encode(calib_text, return_tensors="pt").to("cuda")[:, :1024]
captured = capture_rho_and_gamma(mdl, mixers, calib_ids)
print(f"  captured rho + gamma for {len(captured)} layers")


# =============================================================
# === CELL V4 === Six sigma profiles, common total noise energy
# =============================================================

EPS = 1e-6

def total_energy(sigma_dict):
    return sum(float((s ** 2).sum()) for s in sigma_dict.values())

def make_uniform(captured, sigma_0):
    return {name: torch.full_like(d["rho_emp"], sigma_0)
            for name, d in captured.items()}

def make_proxy_C2(captured, target_E):
    """Phase C-2 / C-2b shape: σ ∝ (1 - rho)."""
    raw = {name: (1.0 - d["rho_emp"]).clamp(min=EPS)
           for name, d in captured.items()}
    raw_E = total_energy(raw)
    scale = (target_E / raw_E) ** 0.5
    return {name: r * scale for name, r in raw.items()}

def make_proxy_correct(captured, target_E):
    """Bit-budget-optimal proxy: σ ∝ sqrt(1 - rho)."""
    raw = {name: (1.0 - d["rho_emp"]).clamp(min=EPS).sqrt()
           for name, d in captured.items()}
    raw_E = total_energy(raw)
    scale = (target_E / raw_E) ** 0.5
    return {name: r * scale for name, r in raw.items()}

def make_full(captured, target_E):
    """Paper C bit allocator: σ ∝ sqrt((1 - rho) / Gamma_s).
    gamma broadcasts over channel dim."""
    raw = {}
    for name, d in captured.items():
        rho   = d["rho_emp"]               # [d_inner, d_state]
        gamma = d["gamma"].clamp(min=EPS)  # [d_state]
        # (1 - rho)[c, s] / gamma[s], then sqrt
        ratio = ((1.0 - rho).clamp(min=EPS) /
                  gamma.unsqueeze(0))
        raw[name] = ratio.sqrt()
    raw_E = total_energy(raw)
    scale = (target_E / raw_E) ** 0.5
    return {name: r * scale for name, r in raw.items()}

def make_anti_full(captured, target_E):
    """Anti of full: σ ∝ sqrt(Gamma_s / (1 - rho)). Sanity."""
    raw = {}
    for name, d in captured.items():
        rho   = d["rho_emp"]
        gamma = d["gamma"].clamp(min=EPS)
        ratio = (gamma.unsqueeze(0) /
                  (1.0 - rho).clamp(min=EPS))
        raw[name] = ratio.sqrt()
    raw_E = total_energy(raw)
    scale = (target_E / raw_E) ** 0.5
    return {name: r * scale for name, r in raw.items()}


SIGMA_0 = 0.15

uniform_s         = make_uniform(captured, SIGMA_0)
target_E          = total_energy(uniform_s)
proxy_C2_s        = make_proxy_C2(captured, target_E)
proxy_correct_s   = make_proxy_correct(captured, target_E)
full_s            = make_full(captured, target_E)
anti_full_s       = make_anti_full(captured, target_E)

print(f"\n  Total noise energy (sum sigma^2):")
for label, sd in [("uniform",         uniform_s),
                   ("proxy_C2",        proxy_C2_s),
                   ("proxy_correct",   proxy_correct_s),
                   ("full",            full_s),
                   ("anti_full",       anti_full_s)]:
    print(f"    {label:<16s}  {total_energy(sd):>12.4e}")

# Bucket diagnostic: mean σ in each rho bucket
buckets = [(0.0, 0.5), (0.5, 0.8), (0.8, 0.95),
           (0.95, 0.99), (0.99, 0.999), (0.999, 1.0)]
print(f"\n  Mean σ_g per rho bucket (read across to compare profiles):")
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
                vals.append(sd[name][mask].flatten())
        if vals:
            cat = torch.cat(vals)
            means.append(float(cat.mean()))
        else:
            means.append(0.0)
    print(f"  [{lo:.3f},{hi:.3f}){'':2s}  "
          f"{means[0]:>9.4f}  {means[1]:>9.4f}  "
          f"{means[2]:>9.4f}  {means[3]:>9.4f}  {means[4]:>9.4f}")


# =============================================================
# === CELL V5 === Apply / restore A_log perturbation
# =============================================================

def perturb_A_log(mixers, sigma_dict, seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    saved = []
    for name, mix in mixers:
        if name not in sigma_dict:
            continue
        sigma = sigma_dict[name].to(mix.A_log.device)
        saved.append((mix, mix.A_log.data.clone()))
        with torch.inference_mode():
            noise = torch.empty_like(mix.A_log.data, dtype=torch.float32)
            noise.normal_()
            mix.A_log.data = (mix.A_log.data.float()
                              + (noise * sigma)).to(mix.A_log.dtype)
    return saved

def restore_A_log(saved):
    for mix, orig in saved:
        mix.A_log.data = orig


# =============================================================
# === CELL V6 === Sliding-window PPL
# =============================================================

def ppl_with(model, ids, ctx_len=1024, stride=512, n_windows=10):
    nll_total = 0.0
    tok_total = 0
    windows_done = 0
    with torch.inference_mode():
        for begin in range(0, ids.shape[1] - ctx_len, stride):
            if windows_done >= n_windows:
                break
            window = ids[:, begin:begin + ctx_len]
            out = model(window, use_cache=False)
            log_probs = F.log_softmax(out.logits[0, :-1].float(), dim=-1)
            targets = window[0, 1:]
            nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            nll_total += nll.sum().item()
            tok_total += targets.numel()
            windows_done += 1
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
      f"sigma_0={SIGMA_0}")
print(f"{'-' * 80}")

CONDITIONS = [
    ("1. FP16 baseline",                   None),
    ("2. uniform sigma",                   uniform_s),
    ("3. proxy_C2  (sigma ∝ (1-rho))",     proxy_C2_s),
    ("4. proxy_correct (sigma ∝ √(1-rho))", proxy_correct_s),
    ("5. full      (sigma ∝ √((1-rho)/G))", full_s),
    ("6. anti_full (sigma ∝ √(G/(1-rho)))", anti_full_s),
]

results = {}
fp16_ppl = None

for label, sigma_dict in CONDITIONS:
    if sigma_dict is None:
        ppl, n_win = ppl_with(mdl, test_ids,
                               ctx_len=CTX_LEN, stride=STRIDE,
                               n_windows=N_WINDOWS)
    else:
        saved = perturb_A_log(mixers, sigma_dict, SEED)
        try:
            ppl, n_win = ppl_with(mdl, test_ids,
                                   ctx_len=CTX_LEN, stride=STRIDE,
                                   n_windows=N_WINDOWS)
        finally:
            restore_A_log(saved)
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
# === CELL V8 === Verdict: full vs proxy
# =============================================================

uni_pct      = results["2. uniform sigma"]["pct_of_base"]
pC2_pct      = results["3. proxy_C2  (sigma ∝ (1-rho))"]["pct_of_base"]
psq_pct      = results["4. proxy_correct (sigma ∝ √(1-rho))"]["pct_of_base"]
full_pct     = results["5. full      (sigma ∝ √((1-rho)/G))"]["pct_of_base"]
anti_pct     = results["6. anti_full (sigma ∝ √(G/(1-rho)))"]["pct_of_base"]

print(f"\n{'=' * 80}")
print(f"Phase C-2c verdict at sigma_0 = {SIGMA_0}")
print(f"{'=' * 80}")
print(f"\n  Predicted ranking:")
print(f"    full < proxy_correct < proxy_C2 < uniform < anti_full")
print(f"\n  Observed (% of FP16 baseline):")
print(f"    full           {full_pct:+7.3f}")
print(f"    proxy_correct  {psq_pct:+7.3f}")
print(f"    proxy_C2       {pC2_pct:+7.3f}")
print(f"    uniform        {uni_pct:+7.3f}")
print(f"    anti_full      {anti_pct:+7.3f}")

# Headline question: does full beat proxy_C2 (Phase C-2/C-2b's
# implementation) by a margin?
gap_full_vs_pC2 = pC2_pct - full_pct
rel_full_vs_pC2 = 100.0 * (pC2_pct - full_pct) / max(pC2_pct, EPS)
gap_full_vs_uniform = uni_pct - full_pct
rel_full_vs_uniform = 100.0 * (uni_pct - full_pct) / max(uni_pct, EPS)

print(f"\n  Headline comparisons:")
print(f"    full vs proxy_C2:  ΔΔPPL = {gap_full_vs_pC2:+7.3f} %  "
      f"(rel {rel_full_vs_pC2:+6.1f} %)")
print(f"    full vs uniform:   ΔΔPPL = {gap_full_vs_uniform:+7.3f} %  "
      f"(rel {rel_full_vs_uniform:+6.1f} %)")

# Verdicts (pre-registered)
THRESH_REL_OVER_PROXY = 10.0  # percentage relative reduction over proxy_C2
THRESH_REL_OVER_UNIF  = 50.0  # over uniform (was 40 % at proxy in C-2b)

if (full_pct < pC2_pct - 0.05 and
        rel_full_vs_pC2 >= THRESH_REL_OVER_PROXY):
    full_vs_proxy = (f"FULL clearly beats PROXY_C2 by {rel_full_vs_pC2:.1f} % "
                      f"relative ΔPPL reduction. Gamma-awareness is "
                      f"materially valuable end-to-end on PPL.")
elif full_pct < pC2_pct - 0.02:
    full_vs_proxy = (f"FULL beats PROXY_C2 marginally "
                      f"(rel {rel_full_vs_pC2:+.1f} %). Direction "
                      f"matches Phase C-1c's prediction but margin "
                      f"is below {THRESH_REL_OVER_PROXY:.0f} % threshold.")
elif abs(full_pct - pC2_pct) <= 0.05:
    full_vs_proxy = (f"FULL ≈ PROXY_C2 (gap < 0.05 %-points). The "
                      f"per-state-unit Pearson 0.635 disagreement seen "
                      f"in C-1c does NOT translate to PPL improvement "
                      f"under this allocation form. The bit-budget "
                      f"objective averages over disagreements that "
                      f"rank correlation reports.")
else:
    full_vs_proxy = (f"FULL is WORSE than PROXY_C2 by {-rel_full_vs_pC2:.1f} %. "
                      f"Either the calibrated Gamma is overfitting to "
                      f"the calibration text, or the bit-budget "
                      f"derivation does not match the actual A_log "
                      f"perturbation surrogate.")

# Direction sanity check
direction_ok = (full_pct < uni_pct < anti_pct)
ranking_full = (full_pct < psq_pct < pC2_pct < uni_pct < anti_pct)

print(f"\n  Direction sanity (full < uniform < anti): "
      f"{'✓' if direction_ok else '✗'}")
print(f"  Strict ranking holds:                       "
      f"{'✓' if ranking_full else '✗'}")

print(f"\n  Verdict (full vs proxy_C2):\n    {full_vs_proxy}")


# =============================================================
# === CELL V9 === Save
# =============================================================

# Compact per-layer summary
summary = []
for name, d in captured.items():
    rho   = d["rho_emp"]
    gamma = d["gamma"]
    summary.append({
        "layer_name":            name,
        "n_units":               int(rho.numel()),
        "rho_p50":               float(rho.flatten().quantile(0.50)),
        "gamma_min":             float(gamma.min()),
        "gamma_p50":             float(gamma.quantile(0.50)),
        "gamma_max":             float(gamma.max()),
        "gamma_max_min_ratio":   float(gamma.max() / gamma.min().clamp(min=EPS)),
    })

output = {
    "date":           datetime.utcnow().isoformat(timespec="seconds"),
    "phase":          "C-2c",
    "purpose":        ("Full-R bit-allocator end-to-end PPL test on "
                       "Mamba-2.8B-hf. Compares the Paper C "
                       "Gamma-aware allocator (sigma ∝ √((1-rho)/G)) "
                       "to the rho-only proxy used in Phase C-2 / "
                       "C-2b (sigma ∝ (1-rho)) and to the "
                       "bit-budget-correct proxy form "
                       "(sigma ∝ √(1-rho))."),
    "model":          MODEL_ID,
    "label":          MODEL_LABEL,
    "protocol": {
        "ctx_len":      CTX_LEN,
        "stride":       STRIDE,
        "n_windows":    N_WINDOWS,
        "sigma_0":      SIGMA_0,
        "rng_seed":     SEED,
        "calib_split":  "WikiText-2 train",
        "test_split":   "WikiText-2 test",
        "conditions":   [c[0] for c in CONDITIONS],
    },
    "fp16_baseline_ppl":   fp16_ppl,
    "results":             results,
    "headline_gaps": {
        "full_vs_proxy_C2_pp":    gap_full_vs_pC2,
        "full_vs_proxy_C2_rel":   rel_full_vs_pC2,
        "full_vs_uniform_pp":     gap_full_vs_uniform,
        "full_vs_uniform_rel":    rel_full_vs_uniform,
    },
    "direction_ok":        direction_ok,
    "strict_ranking":      ranking_full,
    "rho_gamma_summary":   summary,
    "verdict_full_vs_proxy_C2": full_vs_proxy,
}

out_path = "/content/phase_c2c_full_R_allocator_ppl.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\nSaved -> {out_path}")


# =============================================================
# === READING ===
# =============================================================
#
# What this run answers:
#
#   FULL beats PROXY_C2 by ≥ 10 % relative ΔPPL reduction:
#     Gamma-awareness is materially valuable end-to-end. The
#     C-1c diagnostic (Pearson 0.635, 9 % bit-tier match) does
#     transfer to PPL improvement. Paper C's prospectus formula
#     is empirically supported in its full Gamma-aware form, not
#     just in direction. Move to Phase C-3 (state-step
#     quantisation) and eventually C-5 (hybrid stack on
#     Qwen 3.6 27B / Zamba2 with the full allocator).
#
#   FULL beats PROXY_C2 marginally (< 10 % rel):
#     Gamma matters but the bit-budget objective is more
#     forgiving than rank correlation. Cheap proxy is a near-
#     optimal allocator in practice; the prospectus's full
#     formula is correct in principle but not necessary for the
#     deployed allocator. Pull-quote the marginal improvement
#     and explain that A_log is a soft surrogate.
#
#   FULL ≈ PROXY_C2:
#     The gap visible in C-1c per-unit (Pearson 0.635) does not
#     transfer to per-PPL gain. Could be that the bit budget
#     averages over the rank disagreements, or that A_log
#     surrogate is too coarse to detect the difference.
#     Phase C-3 (state-step quantisation, much sharper) is the
#     definitive test.
#
#   FULL is WORSE than PROXY_C2:
#     The calibrated Gamma is hurting; possible causes are
#     calibration text drift, overfitting, or a sign error in
#     the math. Re-run with a held-out calibration window
#     (Phase C-2c-bis) before drawing conclusions.
# ============================================================
