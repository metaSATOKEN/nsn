# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Phase C-9: FP16 / BF16 / clip-only ablation for the C-7
#            long-context stabilisation finding
# ============================================================
# Phase C-7 found that on Mamba-2.8B-hf at PG-19 long context
# (8K / 16K / 32K), the unpatched FP16 baseline diverges
# (PPL 20 → 118 → 172) while INT6-quantised paths stay in the
# 9 - 14 PPL range. We labelled this "implicit numerical
# stabilisation by state quantisation" and considered it a
# candidate Paper C v1 hero finding.
#
# This phase tests whether that finding holds up as a
# *quantisation-specific* claim, by adding two ablations:
#
#   (a) BF16_orig        — same Mamba-2.8B-hf reloaded with
#                          torch_dtype=bfloat16, unpatched.
#                          Tests whether BF16's wider dynamic
#                          range alone fixes the FP16 divergence
#                          (= FP16 numerical bug hypothesis).
#
#   (b) FP16_clip_only   — patched forward path with the SAME
#                          calibrated range r_g = k · RMS(h_g)
#                          as FP16_full, but applied as a pure
#                          h.clamp(-r, r) without rounding to a
#                          quantisation grid. Tests whether
#                          clipping alone reproduces the
#                          stabilisation (= calibrated-clipping
#                          hypothesis).
#
# Together with the existing FP16_orig and FP16_full conditions,
# this gives a 4-condition decision matrix that disambiguates
# precision / clipping / quantisation contributions cleanly.
#
# Decision matrix (PPL trend at ctx = 32K):
#
#   BF16_orig | clip_only | full_INT6 | conclusion
#   ──────────┼───────────┼───────────┼─────────────────────────
#    stable   |   any     |   stable  | FP16 bug → claim collapses
#    diverg   |  stable   |   stable  | clipping suffices
#    diverg   |  diverg   |   stable  | quantisation grid matters ★
#    diverg   |  diverg   |   diverg  | bug — investigate
#
# Stage 1 (FP16 model loaded):
#   - calibrate ρ_emp / Γ / RMS(h) on WikiText-2 1024 tokens
#   - run FP16_orig, FP16_clip_only, FP16_full at 8K / 16K / 32K
#
# Stage 2 (FP16 model unloaded, BF16 model loaded):
#   - run BF16_orig at 8K / 16K / 32K
#
# Sweep:
#   ctx_len ∈ {8192, 16384, 32768}
#   n_windows = 4   (doubled from C-7's 2 for tighter statistics)
#
# Runtime estimate (A100 95 GB):
#   FP16 stage   : ~50 - 60 min   (3 conditions × 3 ctx × 4 windows)
#   BF16 reload  : ~3 min
#   BF16 stage   : ~25 - 30 min   (1 condition × 3 ctx × 4 windows)
#   Total        : ~80 - 95 min
#
# VRAM consideration at 32K context (matches C-7):
#   A_bar / B_bar ~ 10.7 GB per tensor in float32 promotion;
#   intermediates ~30 GB. Fits on A100 80 GB / 95 GB.
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
# === CELL V2 === Load Mamba-2.8B-hf in FP16 (Stage 1)
# =============================================================

MODEL_ID    = "state-spaces/mamba-2.8b-hf"
MODEL_LABEL = "Mamba-2.8B-hf"
TOKENIZER_FALLBACKS = [MODEL_ID, "EleutherAI/gpt-neox-20b"]

print(f"\n{'=' * 80}")
print(f"Loading {MODEL_LABEL}  ({MODEL_ID})  [FP16, Stage 1]")
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
print(f"  VRAM after FP16 load: "
      f"{torch.cuda.memory_allocated() / 1e9:.2f} GB")


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
# === CELL V3 === Patched slow_forward (extended from C-7)
# =============================================================
# New addition vs C-7: a "clip_only" mode that applies the same
# per-(c, s) calibrated range clamp as the quantise mode but
# WITHOUT rounding to a quantisation grid. This is the canonical
# ablation for the "calibrated clipping is sufficient" hypothesis.

_C9 = {
    "mode":          "off",      # "off" | "calibrate" | "quantise" | "clip_only"
    "b_per_layer":   None,
    "r_per_layer":   None,
    "calib_accum":   None,
}

EPS = 1e-6


def _quantize_h_per_unit(h, b_per_unit, r_per_unit):
    """INT-b symmetric quantiser with zero level (mid-tread)."""
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


def _clip_only_h_per_unit(h, r_per_unit):
    """Per-unit clamp to [-r_g, +r_g] without quantisation."""
    r_b = r_per_unit.unsqueeze(0)            # [1, d_inner, d_state]
    return torch.clamp(h, min=-r_b, max=r_b)


def _c9_mamba_forward(mix, input_states):
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

    # Resolve per-layer b / r (only used when mode requires them)
    b_layer = None
    r_layer = None
    if _C9["mode"] in ("quantise", "clip_only"):
        if _C9["r_per_layer"] is not None:
            r = _C9["r_per_layer"].get(layer_name)
            if r is not None:
                r_layer = r.to(x_pre.device, dtype=torch.float32)
        if _C9["mode"] == "quantise" and _C9["b_per_layer"] is not None:
            b = _C9["b_per_layer"].get(layer_name)
            if b is not None:
                b_layer = b.to(x_pre.device)

    if _C9["mode"] == "calibrate":
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
        # ── injection point (between state update and readout) ──
        if _C9["mode"] == "quantise" and (
                b_layer is not None and r_layer is not None):
            h = _quantize_h_per_unit(h, b_layer, r_layer)
        elif _C9["mode"] == "clip_only" and r_layer is not None:
            h = _clip_only_h_per_unit(h, r_layer)
        # ─────────────────────────────────────────────────────────
        if _C9["mode"] == "calibrate":
            rms_sumsq += (h ** 2).sum(dim=0)
        y_t = (h * C_f[:, t].unsqueeze(1)).sum(dim=-1)
        ys.append(y_t)
    y = torch.stack(ys, dim=1)

    y = y + mix.D.float().unsqueeze(0).unsqueeze(0) * x_pre.float()
    y = y * F.silu(gate.float())
    out = mix.out_proj(y.to(mix.out_proj.weight.dtype))

    if _C9["mode"] == "calibrate":
        prev = _C9["calib_accum"].get(layer_name)
        if prev is None:
            _C9["calib_accum"][layer_name] = {
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


def _c9_slow_forward_patch(self, input_states, *args, **kwargs):
    return _c9_mamba_forward(self, input_states)


_MixerClass = type(mixers[0][1])
_orig_slow_forward = _MixerClass.slow_forward
_MixerClass.slow_forward = _c9_slow_forward_patch
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
# === CELL V4 === Profile builders + Calibration (FP16 model)
# =============================================================

def _extras_uniform(captured):
    return {n: torch.zeros_like(d["rho_emp"]) for n, d in captured.items()}


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


B_MIN     = 2
RANGE_K   = 7.0
B_0       = 6


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


# Calibration on WikiText-2 train (1024 tokens, matches C-4 / C-7 / C-8a)
print(f"\n{'-' * 80}")
print(f"Calibrating ρ_emp + Γ_s + RMS(h) on FP16 Mamba-2.8B (1024 tokens)")
print(f"{'-' * 80}")

calib_text = "\n\n".join(
    [x for x in load_dataset("wikitext", "wikitext-2-raw-v1",
                              split="train")["text"]
     if x.strip()])[:8000]
calib_ids = tok.encode(calib_text, return_tensors="pt").to("cuda")[:, :1024]

_C9["mode"]        = "calibrate"
_C9["calib_accum"] = {}
with torch.inference_mode():
    _ = mdl(calib_ids, use_cache=False)
_C9["mode"] = "off"

captured = {}
for name, accum in _C9["calib_accum"].items():
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

# Build profiles for the FP16-side conditions
b_per_layer_full = _build_b_per_unit(_extras_full(captured), B_0)
r_per_layer      = {n: RANGE_K * d["rms_h"] for n, d in captured.items()}

flat_full = torch.cat([b.flatten() for b in b_per_layer_full.values()])
print(f"  full mean_b = {float(flat_full.float().mean()):.2f}, "
      f"std = {float(flat_full.float().std()):.2f}")

# Per-layer summary (saved alongside results, for inspection)
calib_summary = []
for name, d in captured.items():
    calib_summary.append({
        "layer_name":     name,
        "rho_p50":        float(d["rho_emp"].flatten().quantile(0.50)),
        "gamma_p50":      float(d["gamma"].quantile(0.50)),
        "rms_h_p50":      float(d["rms_h"].quantile(0.50)),
    })


# =============================================================
# === CELL V5 === Load PG-19 long-context test text
# =============================================================
# Same fall-through chain as C-7. PG-19 (parquet, community)
# is the canonical first choice; WikiText-103 / WikiText-2 are
# loadability fallbacks.

print(f"\n{'-' * 80}")
print(f"Loading long-context test corpus")
print(f"{'-' * 80}")

text             = None
test_corpus_used = None
sources_tried    = []

CORPUS_CANDIDATES = [
    ("emozilla/pg19",        "test",  None,                  "PG-19 (parquet, community)"),
    ("wikitext",             "train", "wikitext-103-raw-v1", "WikiText-103 train"),
    ("wikitext",             "train", "wikitext-2-raw-v1",   "WikiText-2 train (fallback)"),
]

for ds_name, split, config, label in CORPUS_CANDIDATES:
    try:
        if config is not None:
            ds = load_dataset(ds_name, config, split=split)
        else:
            ds = load_dataset(ds_name, split=split)
        chunks = []
        total_chars = 0
        for entry in ds:
            t = entry.get("text", "")
            if not t or not t.strip():
                continue
            chunks.append(t)
            total_chars += len(t)
            if total_chars > 1_200_000:   # bigger budget for n_windows=4
                break
        text = "\n\n".join(chunks)
        test_corpus_used = label
        sources_tried.append({"label": label, "result": "success",
                                "n_entries": len(chunks),
                                "n_chars":   total_chars})
        print(f"  ✓ {label}: {len(chunks)} entries, "
              f"{total_chars:,} chars")
        break
    except Exception as e:
        err = f"{type(e).__name__}: {str(e)[:200]}"
        print(f"  ✗ {label}: {err}")
        sources_tried.append({"label": label, "result": "failed",
                                "error": err})
        continue

if text is None:
    raise RuntimeError(
        "All long-context corpus candidates failed. "
        "Inspect sources_tried for details."
    )

test_ids = tok.encode(text, return_tensors="pt").to("cuda")
print(f"\n  Source: {test_corpus_used}")
print(f"  Tokenised: {test_ids.shape[1]:,} tokens")


# =============================================================
# === CELL V6 === Long-context PPL eval (mode dispatch)
# =============================================================

CTX_LENS  = [8192, 16384, 32768]
N_WINDOWS = 4   # doubled from C-7's 2

# Modes the eval can run in
MODE_ORIG_FORWARD = "orig_forward"   # unpatched (precision = whatever model dtype is)
MODE_FP32_PATCHED = "fp32_patched"   # patched, mode='off' (sanity check)
MODE_QUANTISE     = "quantise"       # patched, mode='quantise'
MODE_CLIP_ONLY    = "clip_only"      # patched, mode='clip_only' (NEW)


def _set_eval_mode(eval_mode, b_per_layer=None, r_per_layer=None):
    """Switch between unpatched / patched-off / patched-quant / patched-clip."""
    if eval_mode == MODE_ORIG_FORWARD:
        _MixerClass.slow_forward = _orig_slow_forward
        _C9["mode"]         = "off"
        _C9["b_per_layer"]  = None
        _C9["r_per_layer"]  = None
    elif eval_mode == MODE_FP32_PATCHED:
        _MixerClass.slow_forward = _c9_slow_forward_patch
        _C9["mode"]         = "off"
        _C9["b_per_layer"]  = None
        _C9["r_per_layer"]  = None
    elif eval_mode == MODE_QUANTISE:
        _MixerClass.slow_forward = _c9_slow_forward_patch
        _C9["mode"]         = "quantise"
        _C9["b_per_layer"]  = b_per_layer
        _C9["r_per_layer"]  = r_per_layer
    elif eval_mode == MODE_CLIP_ONLY:
        _MixerClass.slow_forward = _c9_slow_forward_patch
        _C9["mode"]         = "clip_only"
        _C9["b_per_layer"]  = None
        _C9["r_per_layer"]  = r_per_layer
    else:
        raise ValueError(f"Unknown eval_mode: {eval_mode}")


def ppl_long_ctx(model, ids, eval_mode,
                  b_per_layer=None, r_per_layer=None,
                  ctx_len=8192, n_windows=4):
    """Sliding-window PPL at long ctx_len. Stride = ctx_len // 2."""
    stride = ctx_len // 2
    _set_eval_mode(eval_mode, b_per_layer, r_per_layer)
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
                del out, log_probs, targets, nll
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        # Always restore patched + off (next call re-configures)
        _MixerClass.slow_forward = _c9_slow_forward_patch
        _C9["mode"]        = "off"
        _C9["b_per_layer"] = None
        _C9["r_per_layer"] = None
    return math.exp(nll_total / tok_total), windows_done


# =============================================================
# === CELL V7 === Stage 1 — FP16 conditions
# =============================================================

print(f"\n{'-' * 80}")
print(f"Stage 1: FP16 conditions  (n_windows={N_WINDOWS})")
print(f"  FP16_orig / FP16_clip_only / FP16_full b_0={B_0}")
print(f"  Contexts: {CTX_LENS}")
print(f"{'-' * 80}")

# (label, mode, b_dict, r_dict)
FP16_CONDITIONS = [
    ("FP16_orig",       MODE_ORIG_FORWARD, None,             None),
    ("FP16_clip_only",  MODE_CLIP_ONLY,    None,             r_per_layer),
    ("FP16_full",       MODE_QUANTISE,     b_per_layer_full, r_per_layer),
]

sweep_results = {ctx: {} for ctx in CTX_LENS}

for ctx_len in CTX_LENS:
    print(f"\n  ── ctx_len = {ctx_len:,} ──")
    for label, mode, b_dict, r_dict in FP16_CONDITIONS:
        print(f"  Running {label} at ctx={ctx_len}...")
        ppl, n_win = ppl_long_ctx(
            mdl, test_ids, mode, b_dict, r_dict,
            ctx_len=ctx_len, n_windows=N_WINDOWS,
        )
        sweep_results[ctx_len][label] = {
            "ppl":             ppl,
            "n_windows":       n_win,
            "mode":            mode,
            "model_dtype":     "fp16",
        }
        print(f"    ppl = {ppl:>9.4f}  (n_windows={n_win})")


# =============================================================
# === CELL V8 === Stage 2 — Unload FP16, load BF16
# =============================================================

print(f"\n{'-' * 80}")
print(f"Stage 2: Unloading FP16 and loading BF16 model")
print(f"{'-' * 80}")

# Restore upstream slow_forward before unloading (so the BF16
# model that loads later re-binds the SAME class with the
# original method — no patched leakage between models)
_MixerClass.slow_forward = _orig_slow_forward

del mdl, mixers
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print(f"  VRAM after unload: "
      f"{torch.cuda.memory_allocated() / 1e9:.2f} GB")

mdl = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16,
    device_map="auto", trust_remote_code=True,
)
mdl.eval()
print(f"  VRAM after BF16 load: "
      f"{torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"  Confirming dtype:  "
      f"{next(mdl.parameters()).dtype}")

# Re-find mixers on the BF16 model
mixers = find_mixers(mdl)
LAYER_NAME_BY_ID = {id(mix): name for name, mix in mixers}
_force_slow_path(mdl)

# `_MixerClass` is the same class object — slow_forward was
# already restored to original above. We don't need to re-patch
# for BF16_orig (it uses the unpatched path).


# =============================================================
# === CELL V9 === Stage 2 — BF16_orig
# =============================================================

print(f"\n{'-' * 80}")
print(f"Stage 2 eval: BF16_orig  (n_windows={N_WINDOWS})")
print(f"{'-' * 80}")

# BF16 uses MODE_ORIG_FORWARD — the same unpatched path as
# FP16_orig, but the model weights are BF16.
for ctx_len in CTX_LENS:
    print(f"\n  ── ctx_len = {ctx_len:,} ──")
    print(f"  Running BF16_orig at ctx={ctx_len}...")
    ppl, n_win = ppl_long_ctx(
        mdl, test_ids, MODE_ORIG_FORWARD, None, None,
        ctx_len=ctx_len, n_windows=N_WINDOWS,
    )
    sweep_results[ctx_len]["BF16_orig"] = {
        "ppl":             ppl,
        "n_windows":       n_win,
        "mode":            MODE_ORIG_FORWARD,
        "model_dtype":     "bf16",
    }
    print(f"    ppl = {ppl:>9.4f}  (n_windows={n_win})")


# =============================================================
# === CELL V10 === Verdict (4-condition decision matrix)
# =============================================================

print(f"\n{'=' * 80}")
print(f"Phase C-9 ablation verdict")
print(f"{'=' * 80}")

print(f"\n  PG-19 sliding-window PPL on Mamba-2.8B-hf, b_0 = {B_0}")
print(f"  (calibrated on WikiText-2 train, ctx 1024)")
print(f"  n_windows = {N_WINDOWS}")
print(f"\n  {'ctx':>7s}  {'FP16_orig':>10s}  {'BF16_orig':>10s}  "
      f"{'clip_only':>10s}  {'FP16_full':>10s}")
print(f"  {'─' * 7}  {'─' * 10}  {'─' * 10}  {'─' * 10}  {'─' * 10}")
for ctx_len in CTX_LENS:
    r = sweep_results[ctx_len]
    fp16  = r["FP16_orig"]["ppl"]
    bf16  = r["BF16_orig"]["ppl"]
    clip  = r["FP16_clip_only"]["ppl"]
    full  = r["FP16_full"]["ppl"]
    print(f"  {ctx_len:>7,}  {fp16:>10.4f}  {bf16:>10.4f}  "
          f"{clip:>10.4f}  {full:>10.4f}")

# Reference points from C-7 (n_windows=2)
print(f"\n  Reference: C-7 at n_windows=2 (same model, same setup):")
print(f"    ctx     FP16_orig    full_INT6")
print(f"    8K       20.46         9.79")
print(f"    16K     118.34         9.46")
print(f"    32K     172.70         9.26")

# Threshold for "stable" / "diverged" classification
# Working definition: "diverged" iff PPL > 2 × C-4 1024-ctx FP16 PPL = 2 × 12.37 ≈ 25.
# This is a conservative threshold — actual divergence at 32K
# in C-7 was ~14× the 1024-ctx baseline.
DIVERG_THRESHOLD_PPL = 25.0


def _classify(ppl_at_32k):
    return "stable" if ppl_at_32k <= DIVERG_THRESHOLD_PPL else "diverged"


fp16_32k = sweep_results[32768]["FP16_orig"]["ppl"]
bf16_32k = sweep_results[32768]["BF16_orig"]["ppl"]
clip_32k = sweep_results[32768]["FP16_clip_only"]["ppl"]
full_32k = sweep_results[32768]["FP16_full"]["ppl"]

cls_fp16 = _classify(fp16_32k)
cls_bf16 = _classify(bf16_32k)
cls_clip = _classify(clip_32k)
cls_full = _classify(full_32k)

print(f"\n  Classification at ctx = 32K (threshold: PPL ≤ "
      f"{DIVERG_THRESHOLD_PPL:.0f}):")
print(f"    FP16_orig       = {fp16_32k:>9.3f}  → {cls_fp16}")
print(f"    BF16_orig       = {bf16_32k:>9.3f}  → {cls_bf16}")
print(f"    FP16_clip_only  = {clip_32k:>9.3f}  → {cls_clip}")
print(f"    FP16_full       = {full_32k:>9.3f}  → {cls_full}")

# Decision matrix
if cls_full == "diverged":
    verdict_class = "QUANT_FAILS"
    verdict = (
        "FP16_full diverged at 32K — the C-7 stabilisation finding "
        "did NOT reproduce. Investigate: (a) was the calibrated "
        "range accidentally regressed? (b) did n_windows=4 sample "
        "different PG-19 entries with intrinsically harder long-"
        "context behaviour? (c) is there a numerical bug introduced "
        "by the new clip_only branch? Re-check the patched function "
        "before drawing any paper-level conclusion."
    )
elif cls_bf16 == "stable":
    # BF16 fixes it without quantisation → FP16 numeric bug
    verdict_class = "FP16_BUG"
    verdict = (
        "BF16_orig is stable while FP16_orig diverges. The C-7 long-"
        "context divergence is primarily a known FP16 numerical "
        "limitation of Mamba-2.8B-hf at extrapolated context — BF16's "
        "wider dynamic range alone resolves it without any "
        "quantisation. Paper C v1 should NOT advertise long-context "
        "stabilisation as a quantisation-specific finding. Reframe "
        "C-7 as: (i) the bit allocator's `full vs uniform` advantage "
        "is preserved at long context (the robust internal-comparison "
        "claim); (ii) state-quantisation incidentally provides "
        "FP16-numerical-stability protection for deployments that "
        "cannot or do not switch to BF16 (a deployment caveat, not "
        "a hero claim)."
    )
elif cls_clip == "stable":
    # BF16 diverges, clip_only stable → calibrated clipping suffices
    verdict_class = "CLIP_SUFFICES"
    verdict = (
        "BF16_orig diverges (so the FP16 issue is not pure precision) "
        "AND clip_only stabilises (so no quantisation grid is needed). "
        "The stabilisation is driven by calibrated clipping at "
        "r_g = k · RMS(h_g), not by the quantisation rounding. "
        "Paper C v1 should reframe the C-7 finding as: 'the bit "
        "allocator's calibrated range estimate r_g implicitly "
        "regularises Mamba-2.8B at long context, even when no "
        "quantisation is applied.' This still credits the bit "
        "allocator's calibration mechanism, but explicitly attributes "
        "the long-context benefit to clipping, not to quantisation. "
        "It is a more conservative — and more defensible — claim."
    )
elif cls_full == "stable":
    # BF16 diverges, clip_only diverges, full stable → quantisation
    # grid genuinely contributes to stabilisation
    verdict_class = "QUANT_GRID_MATTERS"
    verdict = (
        "BF16_orig and FP16_clip_only both diverge at 32K, while "
        "FP16_full remains stable. The stabilisation is genuinely "
        "quantisation-specific — neither BF16 precision nor the "
        "calibrated clipping alone reproduces the effect. Paper C v1 "
        "can advertise the long-context stabilisation finding as a "
        "quantisation-grid-level claim ('quantisation grid resets "
        "fp32-fp16 cast accumulation at every chunk boundary') with "
        "full backing. This is the strongest possible outcome: the "
        "C-7 hero finding is empirically supported as a "
        "quantisation-specific claim."
    )
else:
    verdict_class = "AMBIGUOUS"
    verdict = (
        "Some borderline classification — re-examine the 8K and "
        "16K results and consider the threshold (PPL ≤ "
        f"{DIVERG_THRESHOLD_PPL:.0f}) before drawing conclusions."
    )

print(f"\n  Verdict class: {verdict_class}")
print(f"  Verdict: {verdict}")


# =============================================================
# === CELL V11 === Save
# =============================================================

output = {
    "date":            datetime.utcnow().isoformat(timespec="seconds"),
    "phase":           "C-9",
    "purpose":         ("FP16 / BF16 / clip-only ablation for the "
                        "C-7 long-context stabilisation finding. "
                        "Disambiguates precision / clipping / "
                        "quantisation contributions on Mamba-2.8B-hf "
                        "at PG-19 ctx ∈ {8K, 16K, 32K}."),
    "transformers_version": _hf_transformers.__version__,
    "model":           MODEL_ID,
    "label":           MODEL_LABEL,
    "protocol": {
        "ctx_lens":      CTX_LENS,
        "n_windows":     N_WINDOWS,
        "b_0":           B_0,
        "b_min":         B_MIN,
        "range_k":       RANGE_K,
        "diverg_threshold_ppl": DIVERG_THRESHOLD_PPL,
        "calib_split":   "WikiText-2 train (1024 tokens)",
        "test_corpus":   test_corpus_used,
        "test_corpus_sources_tried": sources_tried,
        "conditions":    [
            "FP16_orig",
            "BF16_orig",
            "FP16_clip_only",
            "FP16_full",
        ],
        "quant_scheme":  ("per-(c, s) INT-b symmetric mid-tread, "
                          "range = k * RMS(h)"),
    },
    "reference_c7_n2": {
        "fp16_orig_8k":  20.4585,
        "fp16_orig_16k": 118.3430,
        "fp16_orig_32k": 172.6985,
        "full_int6_8k":  9.7881,
        "full_int6_16k": 9.4567,
        "full_int6_32k": 9.2562,
    },
    "sweep_results":   {str(k): v for k, v in sweep_results.items()},
    "classifications_at_32k": {
        "FP16_orig":      cls_fp16,
        "BF16_orig":      cls_bf16,
        "FP16_clip_only": cls_clip,
        "FP16_full":      cls_full,
    },
    "verdict_class":   verdict_class,
    "verdict":         verdict,
    "rho_gamma_rms_summary": calib_summary,
}

out_path = "/content/phase_c9_fp16_bf16_clip_ablation.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False, default=str)
print(f"\nSaved -> {out_path}")


# =============================================================
# === READING ===
# =============================================================
#
# QUANT_GRID_MATTERS  (★ strongest paper claim)
#   BF16_orig diverges, FP16_clip_only diverges, FP16_full stable.
#   The quantisation grid (round-to-INT-b) is causally responsible
#   for the long-context stabilisation. C-7 hero finding survives
#   ablation. Paper §7 can claim quantisation-specific
#   stabilisation with confidence.
#
# CLIP_SUFFICES  (intermediate; reframe required)
#   BF16_orig diverges, FP16_clip_only stable, FP16_full stable.
#   The calibrated range r_g is the active ingredient. The bit
#   allocator's CALIBRATION pipeline still earns credit for
#   discovering r_g, but the quantisation rounding itself is
#   not the cause. Paper §7 reframes claim from "quantisation
#   stabilises" to "calibrated clipping stabilises (and
#   quantisation does not destabilise)".
#
# FP16_BUG  (claim collapses at the C-7 hero level)
#   BF16_orig stable. The whole long-context divergence is a
#   FP16 numerical limitation that BF16 alone fixes. The C-7
#   finding is real but specific to FP16 deployments. Paper §7
#   demotes the stabilisation finding to a deployment caveat.
#   The robust paper claim is the `full vs uniform` advantage
#   growing with ctx (an internal comparison unaffected by
#   precision choice).
#
# QUANT_FAILS  (debug)
#   FP16_full does NOT reproduce C-7's stable behaviour. The
#   patched function or the clip_only branch may have introduced
#   a regression. Investigate before any paper-level reading.
#
# AMBIGUOUS
#   Borderline numbers around the divergence threshold. Inspect
#   8K / 16K trajectories before classifying.
# ============================================================
