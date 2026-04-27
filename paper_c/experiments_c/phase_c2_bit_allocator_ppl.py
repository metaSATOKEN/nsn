# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Phase C-2: Bit-allocator end-to-end PPL test
# ============================================================
# Paper C end-to-end probe. NOT for the NSN paper.
#
# Background. Phase C-1 / C-1b measured rho_g per state unit on
# Mamba-2.8B-hf and showed that the empirical (input-dependent
# dt) rho distribution is dramatically broader than the static-
# dt one, with 18.6 % of state units in the long-memory tail
# (rho > 0.95) and a per-layer U-shape (median rho ranges 0.39 -
# 0.88 across layer depth).
#
# The Paper C bit allocator says: a state unit's quantisation
# bit budget should grow with its memory length. Concretely,
# under a uniform-rounding model,
#
#   b_g     = b_0 + ceil( log_2 R_g )+      with R_g = 1/(1-rho_g)
#   var_g   ∝  4^(-b_g)  ∝  (1 - rho_g)^2
#   sigma_g ∝  (1 - rho_g)
#
# i.e. high-rho (long-memory) units should carry small noise
# variance, low-rho (forgetting) units can carry larger noise.
#
# Phase C-2 tests this prediction end-to-end on PPL. Direct
# quantisation of the SSM state at every step would require
# monkey-patching the HF Mamba slow_forward, which is fragile.
# We use a *surrogate*: perturb each A_log entry with Gaussian
# noise of a chosen sigma_g, and compare four sigma profiles
# under a *fixed total-noise-energy* constraint sum(sigma_g^2):
#
#   1. FP16 baseline                    no perturbation
#   2. uniform                          sigma_g = sigma_0 (const)
#   3. adaptive  (Paper C predicted)    sigma_g ∝ (1 - rho_g^emp)
#   4. anti-adaptive  (sanity check)    sigma_g ∝ rho_g^emp
#
# A_log fully determines the per-unit forgetting rate, so noise
# on A_log directly perturbs the recurrence dynamics. This is a
# weaker test than full state-step quantisation -- it does not
# capture Mamba's selectivity of dt under perturbation -- but it
# is a clean empirical probe of "where in the spectrum does it
# matter to spend precision".
#
# Predicted ranking:  ΔPPL_adaptive  <  ΔPPL_uniform  <  ΔPPL_anti
#
# If the ranking holds, Paper C's bit allocator direction is
# empirically supported on a real ~2.8B SSM. If uniform beats
# adaptive, the FIR/IIR-based reasoning is wrong on this
# architecture.
#
# Runtime ~25 min on A100 80 GB:
#   - load + calibration (rho_emp capture):   ~5 min
#   - 4 conditions × 5 windows × ~50 s each:  ~17 min
#   - save / summary:                          ~1 min
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
# === CELL V2 === Load Mamba-1
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
assert tok is not None, "No tokenizer loaded"

mdl = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16,
    device_map="auto", trust_remote_code=True,
)
mdl.eval()
print(f"  VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")


def find_mixers(model):
    found = []
    for name, mod in model.named_modules():
        has_a    = (hasattr(mod, "A_log") and isinstance(
            getattr(mod, "A_log", None), torch.nn.Parameter))
        has_dt   = (hasattr(mod, "dt_proj") and
                     isinstance(getattr(mod, "dt_proj", None),
                                 torch.nn.Linear))
        has_xp   = (hasattr(mod, "x_proj") and
                     isinstance(getattr(mod, "x_proj", None),
                                 torch.nn.Linear))
        if has_a and has_dt and has_xp:
            found.append((name, mod))
    return found

mixers = find_mixers(mdl)
print(f"  Found {len(mixers)} Mamba-1 mixers")


# =============================================================
# === CELL V3 === Calibration: capture rho_emp per state unit
# =============================================================
# (Same machinery as Phase C-1b, but here we keep the full per-
#  unit tensor instead of bucket counts -- we need to allocate
#  per-unit sigma in V5.)

def capture_rho_emp(mdl, mixers, calib_ids):
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

    rho_emp = {}  # layer_name -> [d_inner, d_state] tensor on CPU fp32
    for name, mix in mixers:
        if name not in captured:
            continue
        x = captured[name]
        d_inner    = mix.intermediate_size
        d_state    = mix.ssm_state_size
        dt_rank    = mix.time_step_rank
        with torch.inference_mode():
            T = x.shape[1]
            B = x.shape[0]
            y = mix.in_proj(x.to(mix.in_proj.weight.dtype))
            _, x_pre = y.chunk(2, dim=-1)
            x_pre_c = x_pre.transpose(1, 2).contiguous()
            ksize   = mix.conv1d.weight.shape[-1]
            x_pre_c = F.pad(x_pre_c, (ksize - 1, 0))
            x_pre_c = mix.conv1d(x_pre_c)[:, :, :T]
            x_pre   = F.silu(x_pre_c).transpose(1, 2)
            xbc     = mix.x_proj(x_pre.to(mix.x_proj.weight.dtype))
            dt_raw, _, _ = xbc.split(
                [dt_rank, d_state, d_state], dim=-1
            )
            dt = mix.dt_proj(dt_raw.to(mix.dt_proj.weight.dtype))
            dt = F.softplus(dt.float())  # [B, T, d_inner]
            A_neg = -torch.exp(mix.A_log.detach().float())  # [d_inner, d_state]

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
                )  # [B, t_hi-t_lo, d_inner, d_state]
                sum_A_bar += A_bar_chunk.sum(dim=(0, 1))
                count    += B * (t_hi - t_lo)
            rho_emp[name] = (sum_A_bar / count).detach().cpu()
    return rho_emp


print(f"\n{'-' * 80}")
print("Calibrating rho_emp per state unit on a 1024-token window")
print(f"{'-' * 80}")
ds_test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
test_text = "\n\n".join([x for x in ds_test["text"]
                          if x.strip()])[:48000]
calib_text = "\n\n".join(
    [x for x in load_dataset("wikitext", "wikitext-2-raw-v1",
                              split="train")["text"]
     if x.strip()])[:8000]

calib_ids = tok.encode(calib_text, return_tensors="pt").to("cuda")[:, :1024]
rho_emp = capture_rho_emp(mdl, mixers, calib_ids)
print(f"  captured rho_emp for {len(rho_emp)} layers")


# =============================================================
# === CELL V4 === Sigma allocators (with fixed total energy)
# =============================================================

def total_energy(sigma_dict):
    return sum(float((s ** 2).sum()) for s in sigma_dict.values())

def make_uniform_sigma(rho_emp, sigma_0):
    return {name: torch.full_like(rho, sigma_0)
            for name, rho in rho_emp.items()}

def make_adaptive_sigma(rho_emp, total_target_energy):
    """Adaptive: sigma_g ∝ (1 - rho_g). High-memory units get
    small sigma. Rescaled to match total_target_energy."""
    raw = {name: (1.0 - rho).clamp(min=1e-6)
           for name, rho in rho_emp.items()}
    raw_energy = total_energy(raw)
    scale = (total_target_energy / raw_energy) ** 0.5
    return {name: r * scale for name, r in raw.items()}

def make_anti_adaptive_sigma(rho_emp, total_target_energy):
    """Anti-adaptive: sigma_g ∝ rho_g. The wrong direction by
    the bit allocator hypothesis. Sanity check for direction."""
    raw = {name: rho.clamp(min=1e-6)
           for name, rho in rho_emp.items()}
    raw_energy = total_energy(raw)
    scale = (total_target_energy / raw_energy) ** 0.5
    return {name: r * scale for name, r in raw.items()}


# Reference: how big is sigma_0 for the uniform profile?
# We pick sigma_0 such that the typical relative perturbation
# of A_log is non-trivial but well below the magnitude of A_log
# itself (which has mean ~ 0 and std ~ 1 in Mamba-1).
SIGMA_0 = 0.05

uniform_sigma = make_uniform_sigma(rho_emp, SIGMA_0)
total_E       = total_energy(uniform_sigma)
adaptive_sigma     = make_adaptive_sigma(rho_emp, total_E)
anti_adaptive_sigma = make_anti_adaptive_sigma(rho_emp, total_E)

# Diagnostic: how does sigma differ across profiles for the
# 5 buckets used in C-1b?
def bucket_sigma_summary(rho_emp, sigma_dict):
    """For each rho bucket, report mean sigma."""
    buckets = [
        (0.0, 0.5), (0.5, 0.8), (0.8, 0.95),
        (0.95, 0.99), (0.99, 0.999), (0.999, 1.0),
    ]
    summary = []
    for lo, hi in buckets:
        s_vals = []
        for name, rho in rho_emp.items():
            mask = (rho >= lo) & (rho < hi if hi < 1.0 else rho <= hi)
            if mask.any():
                s_vals.append(sigma_dict[name][mask].flatten())
        if s_vals:
            cat = torch.cat(s_vals)
            summary.append((lo, hi, int(cat.numel()),
                            float(cat.mean()), float(cat.std())))
        else:
            summary.append((lo, hi, 0, 0.0, 0.0))
    return summary

print("\n  Sigma allocation summary (mean σ_g per rho bucket):")
print(f"  {'rho range':<18s}  {'n units':>10s}  "
      f"{'uniform':>10s}  {'adaptive':>10s}  {'anti':>10s}")
for (lo, hi, n_u, _, _), \
    (_, _, _, m_u, _), \
    (_, _, _, m_a, _), \
    (_, _, _, m_n, _) in zip(
        bucket_sigma_summary(rho_emp, uniform_sigma),
        bucket_sigma_summary(rho_emp, uniform_sigma),
        bucket_sigma_summary(rho_emp, adaptive_sigma),
        bucket_sigma_summary(rho_emp, anti_adaptive_sigma)):
    print(f"  [{lo:.3f},{hi:.3f}){'':3s}  {n_u:>10d}  "
          f"{m_u:>10.4f}  {m_a:>10.4f}  {m_n:>10.4f}")

print(f"\n  Total noise energy (sum sigma^2) per profile:")
print(f"    uniform:       {total_energy(uniform_sigma):>12.4e}")
print(f"    adaptive:      {total_energy(adaptive_sigma):>12.4e}")
print(f"    anti-adaptive: {total_energy(anti_adaptive_sigma):>12.4e}")


# =============================================================
# === CELL V5 === Apply / restore A_log perturbation
# =============================================================

def perturb_A_log(mixers, sigma_dict, generator):
    """Add Gaussian noise N(0, sigma) per A_log unit. Returns a
    list of (mixer, original_A_log_data) pairs for restoration."""
    saved = []
    for name, mix in mixers:
        if name not in sigma_dict:
            continue
        sigma = sigma_dict[name].to(mix.A_log.device)
        saved.append((mix, mix.A_log.data.clone()))
        with torch.inference_mode():
            noise = torch.empty_like(mix.A_log.data, dtype=torch.float32)
            noise.normal_(generator=generator)
            mix.A_log.data = (mix.A_log.data.float()
                              + (noise * sigma)).to(mix.A_log.dtype)
    return saved

def restore_A_log(saved):
    for mix, orig in saved:
        mix.A_log.data = orig


# =============================================================
# === CELL V6 === WikiText-2 sliding-window PPL
# =============================================================

def ppl_with(model, ids, ctx_len=1024, stride=512, n_windows=5):
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
# === CELL V7 === Run the 4-condition sweep
# =============================================================

CTX_LEN   = 1024
STRIDE    = 512
N_WINDOWS = 5

# Same RNG seed for all noise-added conditions, so the difference
# between conditions is the sigma profile (allocation), not the
# random draw.
SEED = 42

test_ids = tok.encode(test_text, return_tensors="pt").to("cuda")
print(f"\n{'-' * 80}")
print(f"WikiText-2 sliding-window PPL  "
      f"(ctx={CTX_LEN}, stride={STRIDE}, n_windows={N_WINDOWS})")
print(f"{'-' * 80}")

CONDITIONS = [
    ("1. FP16 baseline",         None),
    ("2. uniform sigma",         uniform_sigma),
    ("3. adaptive sigma",        adaptive_sigma),
    ("4. anti-adaptive sigma",   anti_adaptive_sigma),
]

results = {}
fp16_ppl = None

for label, sigma_dict in CONDITIONS:
    if sigma_dict is None:
        # FP16 baseline
        ppl, n_win = ppl_with(mdl, test_ids,
                               ctx_len=CTX_LEN, stride=STRIDE,
                               n_windows=N_WINDOWS)
    else:
        gen   = torch.Generator(device="cpu").manual_seed(SEED)
        saved = perturb_A_log(mixers, sigma_dict, gen)
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
    print(f"  {label:<26s}  ppl={ppl:>8.4f}  "
          f"ΔPPL={dppl:+8.4f}  ({pct:+7.3f} % of base)")


# =============================================================
# === CELL V8 === Hypothesis ranking
# =============================================================

print(f"\n{'=' * 80}")
print("Bit-allocator direction test on Mamba-2.8B-hf")
print(f"{'=' * 80}")

uni_pct  = results["2. uniform sigma"]["pct_of_base"]
ada_pct  = results["3. adaptive sigma"]["pct_of_base"]
ant_pct  = results["4. anti-adaptive sigma"]["pct_of_base"]

print(f"\n  Same total noise energy in all three perturbation conditions.")
print(f"\n  Predicted (Paper C bit allocator):")
print(f"    adaptive  <  uniform  <  anti-adaptive")
print(f"\n  Observed:")
print(f"    adaptive       Δr4_pct = {ada_pct:+7.3f} %")
print(f"    uniform        Δr4_pct = {uni_pct:+7.3f} %")
print(f"    anti-adaptive  Δr4_pct = {ant_pct:+7.3f} %")

verdict = ""
if ada_pct < uni_pct - 0.5 and uni_pct < ant_pct - 0.5:
    verdict = ("PREDICTED RANKING CONFIRMED. The bit-allocator "
                "direction (more precision to high-rho units) is "
                "empirically supported on Mamba-2.8B end-to-end on "
                "WikiText-2 PPL.")
elif ada_pct > ant_pct + 0.5:
    verdict = ("PREDICTED RANKING REVERSED. The Paper C "
                "spectral-radius-based prior is wrong on this "
                "architecture; FIR/IIR-style allocation is "
                "actively miscalibrated.")
elif abs(ada_pct - uni_pct) < 0.5 and abs(uni_pct - ant_pct) < 0.5:
    verdict = ("NO MEASURABLE EFFECT. Either the noise level "
                "SIGMA_0 is too small to register, or A_log noise "
                "is the wrong surrogate for state quantisation. "
                "Try larger SIGMA_0 or implement state-step "
                "quantisation directly.")
else:
    verdict = ("PARTIAL CONFIRMATION. Direction tentatively in "
                "the predicted order but at least one comparison "
                "is within noise. Increase n_windows or SIGMA_0 to "
                "resolve.")

print(f"\n  Verdict: {verdict}")


# =============================================================
# === CELL V9 === Save
# =============================================================

# Compact rho_emp summary (per-layer aggregates only) so the
# JSON stays small.
rho_summary = []
for name, rho in rho_emp.items():
    flat = rho.flatten()
    rho_summary.append({
        "layer_name": name,
        "n_units":    int(flat.numel()),
        "rho_min":    float(flat.min()),
        "rho_p50":    float(flat.quantile(0.50)),
        "rho_p99":    float(flat.quantile(0.99)),
        "rho_max":    float(flat.max()),
    })

output = {
    "date":           datetime.utcnow().isoformat(timespec="seconds"),
    "phase":          "C-2",
    "purpose":        ("End-to-end PPL test of the Paper C bit "
                       "allocator direction. A_log perturbation is "
                       "used as a surrogate for state-step "
                       "quantisation: each A_log entry is hit with "
                       "Gaussian noise of a profile-specific sigma, "
                       "with three profiles sharing the same total "
                       "noise energy."),
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
    "rho_emp_summary":     rho_summary,
    "verdict":             verdict,
}

out_path = "/content/phase_c2_bit_allocator_ppl.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\nSaved -> {out_path}")


# =============================================================
# === READING ===
# =============================================================
#
# The hypothesis is that, at the same total noise energy:
#   - adaptive      sigma_g ∝ (1 - rho_g)     gives smallest ΔPPL
#   - uniform       sigma_g = const            gives middle ΔPPL
#   - anti-adaptive sigma_g ∝ rho_g            gives largest ΔPPL
#
# What this run can tell us:
#
#   PREDICTED RANKING CONFIRMED:
#     The Paper C bit allocator direction is empirically right
#     on a real ~2.8B SSM. Move to Phase C-3 (error-feedback
#     rounding) and Phase C-5 (modeq attention + state-aware
#     SSM hybrid stack on Qwen 3.6 27B or Zamba2).
#
#   PREDICTED RANKING REVERSED:
#     The spectral-radius prior is wrong on Mamba. Could be
#     because A_log perturbation is not a faithful surrogate
#     for state quantisation (state quantisation interacts with
#     selectivity dt in ways A_log perturbation doesn't), or
#     because the rho_emp distribution we measured in C-1b is
#     mis-attributed (long-memory units are not the ones that
#     should get more bits). Either way, the prospectus needs
#     a major revision.
#
#   NO MEASURABLE EFFECT:
#     SIGMA_0 = 0.05 is too small to perturb PPL meaningfully.
#     Bump to 0.1 or 0.2 and re-run; or implement state-step
#     quantisation in slow_forward (more invasive).
#
# Sigma profiles share total energy by construction, so the
# comparison is "where to spend the noise budget" rather than
# "how much total noise". This is the clean Paper-C test.
# ============================================================
