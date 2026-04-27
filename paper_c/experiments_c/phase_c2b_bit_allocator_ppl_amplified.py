# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Phase C-2b: amplified bit-allocator PPL test on Mamba-2.8B
# ============================================================
# Paper C follow-up to Phase C-2.
#
# Phase C-2 ran the same protocol at sigma_0 = 0.05, n_windows = 5
# and produced the predicted ranking
#
#   adaptive (0.167 %) < uniform (0.215 %) < anti-adaptive (0.233 %)
#
# but with absolute pairwise gaps of 0.048 % / 0.018 % of baseline
# PPL — below the pre-registered 0.5 % threshold for "PREDICTED
# RANKING CONFIRMED". Three competing explanations:
#
#   (1) sigma_0 = 0.05 is conservative; the model is operating in
#       a near-linear-response regime where allocation differences
#       are diluted. A larger sigma_0 should amplify the gaps
#       roughly in proportion to sigma_0^2.
#
#   (2) A_log perturbation is a soft surrogate for state-step
#       quantisation; selectivity dt may compensate.
#
#   (3) 5 windows is at the edge of statistical resolution.
#
# Phase C-2b tests (1) and (3) directly:
#
#   sigma_0:   0.05 -> 0.15           (3x; expected gap inflation 9x)
#   n_windows: 5    -> 10             (variance halved roughly)
#
# Predicted: at sigma_0 = 0.15 with 10 windows, the adaptive vs
# uniform gap should grow from 0.048 % to ~0.4 - 0.6 %, clearing
# the pre-registered threshold and producing a "PREDICTED RANKING
# CONFIRMED" verdict outright.
#
# If the gap fails to scale -- e.g., adaptive ≈ uniform even at
# sigma_0 = 0.15 -- hypothesis (2) wins and Phase C-3 (state-step
# quantisation directly in slow_forward) is the next move.
#
# Runtime ~30 min on A100 80 GB.
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
# === CELL V3 === Calibration: rho_emp per state unit
# =============================================================

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

    rho_emp = {}
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
            dt_raw, _, _ = xbc.split(
                [dt_rank, d_state, d_state], dim=-1
            )
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
            rho_emp[name] = (sum_A_bar / count).detach().cpu()
    return rho_emp


print(f"\n{'-' * 80}")
print("Calibrating rho_emp")
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
# === CELL V4 === Sigma allocators (same as C-2)
# =============================================================

def total_energy(sigma_dict):
    return sum(float((s ** 2).sum()) for s in sigma_dict.values())

def make_uniform_sigma(rho_emp, sigma_0):
    return {name: torch.full_like(rho, sigma_0)
            for name, rho in rho_emp.items()}

def make_adaptive_sigma(rho_emp, total_target_energy):
    raw = {name: (1.0 - rho).clamp(min=1e-6)
           for name, rho in rho_emp.items()}
    raw_energy = total_energy(raw)
    scale = (total_target_energy / raw_energy) ** 0.5
    return {name: r * scale for name, r in raw.items()}

def make_anti_adaptive_sigma(rho_emp, total_target_energy):
    raw = {name: rho.clamp(min=1e-6)
           for name, rho in rho_emp.items()}
    raw_energy = total_energy(raw)
    scale = (total_target_energy / raw_energy) ** 0.5
    return {name: r * scale for name, r in raw.items()}


# Amplified: sigma_0 3x larger than C-2 (0.05 -> 0.15).
SIGMA_0 = 0.15

uniform_sigma     = make_uniform_sigma(rho_emp, SIGMA_0)
total_E           = total_energy(uniform_sigma)
adaptive_sigma     = make_adaptive_sigma(rho_emp, total_E)
anti_adaptive_sigma = make_anti_adaptive_sigma(rho_emp, total_E)

print(f"\n  Sigma profiles built at sigma_0 = {SIGMA_0}")
print(f"  Total noise energy (sum sigma^2):")
for label, sd in [("uniform", uniform_sigma),
                   ("adaptive", adaptive_sigma),
                   ("anti", anti_adaptive_sigma)]:
    print(f"    {label:<10s}  {total_energy(sd):>12.4e}")


# =============================================================
# === CELL V5 === Apply / restore
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
# === CELL V7 === Run amplified sweep
# =============================================================

CTX_LEN   = 1024
STRIDE    = 512
N_WINDOWS = 10  # doubled from C-2
SEED      = 42

test_ids = tok.encode(test_text, return_tensors="pt").to("cuda")
print(f"\n{'-' * 80}")
print(f"WikiText-2 sliding-window PPL  "
      f"(ctx={CTX_LEN}, stride={STRIDE}, n_windows={N_WINDOWS})  "
      f"sigma_0={SIGMA_0}")
print(f"{'-' * 80}")

CONDITIONS = [
    ("1. FP16 baseline",                 None),
    ("2. uniform sigma  (sigma_0=0.15)", uniform_sigma),
    ("3. adaptive sigma (sigma_0=0.15)", adaptive_sigma),
    ("4. anti-adaptive  (sigma_0=0.15)", anti_adaptive_sigma),
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
    print(f"  {label:<40s}  ppl={ppl:>9.4f}  "
          f"ΔPPL={dppl:+8.4f}  ({pct:+7.3f} % of base)")


# =============================================================
# === CELL V8 === Direction verdict + scaling check
# =============================================================

uni_pct  = results["2. uniform sigma  (sigma_0=0.15)"]["pct_of_base"]
ada_pct  = results["3. adaptive sigma (sigma_0=0.15)"]["pct_of_base"]
ant_pct  = results["4. anti-adaptive  (sigma_0=0.15)"]["pct_of_base"]

# Phase C-2 (sigma_0 = 0.05) reference numbers, for amplification
# scaling check.
C2_REF = {
    "uniform":  0.215,
    "adaptive": 0.167,
    "anti":     0.233,
    "sigma_0":  0.05,
}

print(f"\n{'=' * 80}")
print(f"Phase C-2b verdict at sigma_0 = {SIGMA_0}")
print(f"{'=' * 80}")
print(f"\n  Predicted ranking: adaptive < uniform < anti-adaptive")
print(f"\n  Observed (sigma_0={SIGMA_0}):")
print(f"    adaptive       Δ%baseline = {ada_pct:+8.3f}")
print(f"    uniform        Δ%baseline = {uni_pct:+8.3f}")
print(f"    anti-adaptive  Δ%baseline = {ant_pct:+8.3f}")
print(f"\n  Phase C-2 reference (sigma_0={C2_REF['sigma_0']}):")
print(f"    adaptive       Δ%baseline = {C2_REF['adaptive']:+8.3f}")
print(f"    uniform        Δ%baseline = {C2_REF['uniform']:+8.3f}")
print(f"    anti-adaptive  Δ%baseline = {C2_REF['anti']:+8.3f}")

# Pairwise gaps
gap_au_c2  = C2_REF["uniform"] - C2_REF["adaptive"]   # +0.048
gap_ua_c2  = C2_REF["anti"]    - C2_REF["uniform"]    # +0.018
gap_au_c2b = uni_pct - ada_pct
gap_ua_c2b = ant_pct - uni_pct

# If the noise is operating in a linear-response regime,
# ΔPPL ∝ sigma_0^2; for sigma_0 going 0.05 -> 0.15 that is a
# 9x amplification in expected gap.
expected_amp = (SIGMA_0 / C2_REF["sigma_0"]) ** 2
print(f"\n  Gap scaling (predicted amplification = sigma_0^2 ratio = "
      f"{expected_amp:.1f}x):")
print(f"    adaptive vs uniform:  C-2 {gap_au_c2:+.3f} -> "
      f"C-2b {gap_au_c2b:+.3f}  "
      f"(actual {gap_au_c2b/max(gap_au_c2, 1e-6):.1f}x)")
print(f"    anti vs uniform:      C-2 {gap_ua_c2:+.3f} -> "
      f"C-2b {gap_ua_c2b:+.3f}  "
      f"(actual {gap_ua_c2b/max(gap_ua_c2, 1e-6):.1f}x)")

THRESHOLD = 0.5
if (ada_pct < uni_pct - THRESHOLD
    and uni_pct < ant_pct - THRESHOLD):
    verdict = (f"PREDICTED RANKING CONFIRMED at sigma_0 = {SIGMA_0}, "
                f"clearing the {THRESHOLD} % absolute-gap threshold. "
                f"The Paper C bit allocator direction is empirically "
                f"supported on Mamba-2.8B end-to-end.")
elif ada_pct < uni_pct and uni_pct < ant_pct:
    verdict = (f"DIRECTION CORRECT but gaps still below the "
                f"{THRESHOLD} % absolute threshold. Compare to C-2 "
                f"scaling: if observed amplification matches the "
                f"predicted sigma_0^2 ratio of {expected_amp:.1f}x, "
                f"the small magnitude is purely a sigma_0 "
                f"size effect rather than a problem with the "
                f"allocator. Either bump sigma_0 further or accept "
                f"the soft-confirmation framing.")
elif ada_pct > ant_pct:
    verdict = ("PREDICTED RANKING REVERSED. The bit-allocator "
                "spectral prior is wrong on Mamba; FIR/IIR "
                "framework needs revision.")
else:
    verdict = ("MIXED or NO RANKING. Inspect the per-condition "
                "PPL and consider Phase C-3 (state-step "
                "quantisation) as a more direct test.")

print(f"\n  Verdict: {verdict}")


# =============================================================
# === CELL V9 === Save
# =============================================================

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
    "phase":          "C-2b",
    "purpose":        ("Amplified bit-allocator PPL test on "
                       "Mamba-2.8B-hf. Replays Phase C-2 with "
                       "sigma_0 = 0.15 (3x larger than C-2's 0.05) "
                       "and 10 windows (2x larger than C-2's 5) to "
                       "test whether the small magnitude observed "
                       "in C-2 is a near-linear-response artifact."),
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
    },
    "fp16_baseline_ppl":   fp16_ppl,
    "results":             results,
    "phase_c2_reference":  C2_REF,
    "expected_amplification": expected_amp,
    "rho_emp_summary":     rho_summary,
    "verdict":             verdict,
}

out_path = "/content/phase_c2b_bit_allocator_ppl_amplified.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\nSaved -> {out_path}")


# =============================================================
# === READING ===
# =============================================================
#
# Three ways this run resolves Phase C-2's ambiguity:
#
#   PREDICTED RANKING CONFIRMED at the 0.5 % absolute threshold:
#     The small magnitude in C-2 was purely sigma_0 dilution.
#     The Paper C bit allocator direction is empirically right.
#     Update paper_c/README.md from "soft confirmation" to
#     "confirmed at sigma_0=0.15", and move on to Phase C-3
#     (state-step quantisation, error-feedback rounding).
#
#   DIRECTION CORRECT, gaps still small but scaled with sigma_0:
#     The allocator works as predicted in linear-response regime
#     and we just need ever-larger sigma_0 to register more
#     dramatic gaps. Either accept the relative-effect framing
#     ("22 % ΔPPL reduction at fixed total noise budget") or
#     bump sigma_0 to 0.25 / 0.30 in a Phase C-2c.
#
#   PREDICTED RANKING REVERSED or NO RANKING:
#     The A_log perturbation is an unreliable surrogate for state
#     quantisation, or the spectral prior is wrong. Skip to
#     Phase C-3 with state-step quantisation in slow_forward.
#
# Crucial new diagnostic added in this run: gap-scaling check.
# In a linear-response regime, gaps should grow as sigma_0^2,
# i.e. 9x going 0.05 -> 0.15. If observed amplification is much
# below 9x, we are saturating; if much above, dispersion of the
# allocation matters non-linearly.
# ============================================================
