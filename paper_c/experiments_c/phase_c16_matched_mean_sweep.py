# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Phase C-16: Matched-mean full sweep with the candidate recipe
# ============================================================
# Closes the open empirical question identified by both rounds
# of pseudo-review on v0.5 / v0.6: the section 6.7 head-to-head
# showed that half_slope_rg is structurally tighter than the
# deployed full_slope, but the actual mean bit width was not
# matched (half_slope_rg overshoots by 0.24 to 0.79 bits because
# the one-pass shift normalisation does not account for the
# b_min and ceiling clamps after the log r_g term shifts the
# distribution).
#
# Phase C-16 introduces two methodological additions to close
# the section 6.7 / round-4 critique:
#
#   (1) Binary-search shift normalisation (two-pass) such that
#       the post-clamp mean bit width matches the nominal b_0
#       within 0.01 bits across all four profiles. This makes
#       the head-to-head comparison strictly apples-to-apples.
#
#   (2) Full state-spaces Mamba-1 family sweep (130M, 370M,
#       790M, 1.4B, 2.8B) at b_0 in {4, 6}. This produces the
#       updated deployment table that the v1 paper headline
#       depends on.
#
# Profiles (4):
#   uniform:        extras = 0
#   full_slope:     extras = ceil(log_2 R_g)+
#   half_slope:     extras = ceil((1/2) log_2 R_g)+
#   half_slope_rg:  extras = ceil((1/2) log_2 R_g + log_2 r_g - mean(log_2 r_g))+
#
# Evaluation: WikiText-2 test, ctx = 1024, n_windows = 5 (matching
# C-4 / C-8a / C-12).
#
# Pre-registered hypotheses:
#
#   H1 (matched-mean structural test): At matched actual mean
#     bit width, half_slope_rg should still beat full_slope at
#     b_0 = 4 and b_0 = 6 on every model. If the gap survives
#     matching, the candidate recipe is structurally better; if
#     the gap collapses, the v0.5 'recommended' framing was
#     misleading and the paper's main character should revert.
#
#   H2 (rank-order transfer): The strict rank order
#     uniform >> full_slope >= half_slope > half_slope_rg
#     should hold on every model at every b_0.
#
#   H3 (deployment table update): At (b_0 = 6, half_slope_rg)
#     across the 5 state-spaces sizes, the headline ΔPPL is
#     expected to drop from the v0.6 +4 to +8 percent range
#     (full_slope) to roughly +3 to +5 percent (half_slope_rg),
#     based on the section 6.7 single-model extrapolation.
#
# Runtime estimate (A100 95 GB):
#   130M:  4 to 5 min  (8 quant conditions + 1 FP16 baseline)
#   370M:  10 to 12 min
#   790M:  14 to 16 min
#   1.4B:  24 to 26 min
#   2.8B:  30 to 35 min
#   Total: ~ 85 to 95 minutes
#
# Per-model checkpointing: a partial JSON is written to
#   /content/phase_c16_partial_<size>.json
# after each model completes, so the run can be resumed if the
# Colab kernel disconnects. The final merged JSON is written to
#   /content/phase_c16_matched_mean_sweep.json
# after the last model.
# ============================================================


# =============================================================
# === CELL V1 === Setup
# =============================================================

import gc
import math
import json
import os
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
# === CELL V2 === Constants and quantiser
# =============================================================

EPS = 1e-6
B_MIN, B_MAX, RANGE_K = 2, 16, 7.0
B0_VALUES = [4, 6]

CTX = 1024
N_WINDOWS = 5

# Models to sweep. Edit this list to skip/select models.
# In order of increasing size for monotonic VRAM growth.
MODEL_LIST = [
    ("state-spaces/mamba-130m-hf",   "Mamba-130M"),
    ("state-spaces/mamba-370m-hf",   "Mamba-370M"),
    ("state-spaces/mamba-790m-hf",   "Mamba-790M"),
    ("state-spaces/mamba-1.4b-hf",   "Mamba-1.4B"),
    ("state-spaces/mamba-2.8b-hf",   "Mamba-2.8B"),
]

TOKENIZER_FALLBACKS = [
    "state-spaces/mamba-2.8b-hf",
    "EleutherAI/gpt-neox-20b",
]

# Where partial JSONs go
PARTIAL_DIR = "/content"

# Which profiles to evaluate (in this order)
PROFILE_NAMES = ["uniform", "full_slope", "half_slope", "half_slope_rg"]


# =============================================================
# === CELL V3 === Patched slow_forward (per-token state quant)
# =============================================================
# Same patched function as C-4 / C-8a / C-12. Mode dispatch:
#   "off"        unpatched semantics (no quant, no calibration)
#   "calibrate"  records rho_emp / Gamma / RMS(h) per (c, s)
#   "quantise"   per-(c, s) INT-b mid-tread quantisation

_C16 = {
    "mode":           "off",
    "b_per_layer":    None,
    "r_per_layer":    None,
    "calib_accum":    None,
}


def _quantize_h_per_unit(h, b_per_unit, r_per_unit):
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


def _make_c16_forward(mixers, layer_name_by_id):
    """Build a forward function bound to the given mixer set."""

    def _c16_mamba_forward(mix, input_states):
        layer_name = layer_name_by_id.get(id(mix))
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
        if _C16["mode"] == "quantise":
            b = _C16["b_per_layer"].get(layer_name)
            r = _C16["r_per_layer"].get(layer_name)
            if b is not None and r is not None:
                b_layer = b.to(x_pre.device)
                r_layer = r.to(x_pre.device, dtype=torch.float32)

        if _C16["mode"] == "calibrate":
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
            if _C16["mode"] == "calibrate":
                rms_sumsq += (h ** 2).sum(dim=0)
            y_t = (h * C_f[:, t].unsqueeze(1)).sum(dim=-1)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)

        y = y + mix.D.float().unsqueeze(0).unsqueeze(0) * x_pre.float()
        y = y * F.silu(gate.float())
        out = mix.out_proj(y.to(mix.out_proj.weight.dtype))

        if _C16["mode"] == "calibrate":
            prev = _C16["calib_accum"].get(layer_name)
            if prev is None:
                _C16["calib_accum"][layer_name] = {
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

    def _patch(self, input_states, *args, **kwargs):
        return _c16_mamba_forward(self, input_states)

    return _patch


def _force_slow_path(model):
    for _, mod in model.named_modules():
        for attr in ("use_fast_path", "_use_fast_path",
                      "use_mambapy", "_uses_cuda_kernels"):
            if hasattr(mod, attr):
                try:
                    setattr(mod, attr, False)
                except Exception:
                    pass


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


# =============================================================
# === CELL V4 === Profile builders
# =============================================================
# Each builder returns a dict {layer_name: tensor of integer
# extras} BEFORE the matched-mean shift normalisation. The
# matched-mean shift is applied uniformly via binary search in
# CELL V5, so each builder produces the "raw" extras only.

def _extras_uniform(captured):
    return {n: torch.zeros_like(d["rho_emp"]) for n, d in captured.items()}


def _extras_full_slope(captured):
    """slope 1 on log_2 R_g (the deployed simplification)."""
    out = {}
    for n, d in captured.items():
        rho   = d["rho_emp"]
        gamma = d["gamma"].clamp(min=EPS)
        log_R = torch.log2(
            gamma.unsqueeze(0) / (1.0 - rho).clamp(min=EPS)
        )
        out[n] = torch.ceil(log_R).clamp(min=0)
    return out


def _extras_half_slope(captured):
    """slope 1/2 on log_2 R_g (strict optimum without r_g)."""
    out = {}
    for n, d in captured.items():
        rho   = d["rho_emp"]
        gamma = d["gamma"].clamp(min=EPS)
        log_R = torch.log2(
            gamma.unsqueeze(0) / (1.0 - rho).clamp(min=EPS)
        )
        out[n] = torch.ceil(0.5 * log_R).clamp(min=0)
    return out


def _extras_half_slope_rg(captured):
    """slope 1/2 + log_2 r_g - mean log_2 r_g (strict optimum
    with r_g, the candidate deployment recipe)."""
    rms_all = []
    for n, d in captured.items():
        rms_all.append(d["rms_h"].flatten())
    rms_cat = torch.cat(rms_all)
    log_rg_global_mean = float(
        torch.log2((RANGE_K * rms_cat).clamp(min=EPS)).mean()
    )

    out = {}
    for n, d in captured.items():
        rho   = d["rho_emp"]
        gamma = d["gamma"].clamp(min=EPS)
        rms   = d["rms_h"]
        log_R = torch.log2(
            gamma.unsqueeze(0) / (1.0 - rho).clamp(min=EPS)
        )
        log_rg = torch.log2((RANGE_K * rms).clamp(min=EPS))
        log_rg_centered = log_rg - log_rg_global_mean
        out[n] = torch.ceil(0.5 * log_R + log_rg_centered).clamp(min=0)
    return out


PROFILE_BUILDERS = {
    "uniform":        _extras_uniform,
    "full_slope":     _extras_full_slope,
    "half_slope":     _extras_half_slope,
    "half_slope_rg":  _extras_half_slope_rg,
}


# =============================================================
# === CELL V5 === Matched-mean binary-search shift
# =============================================================
# The key methodological addition vs C-12: a binary search over
# an additive shift such that the post-clamp mean bit width
# equals the target (within tolerance). This removes the
# one-pass-shift overshoot that confounded section 6.7's
# headline numbers in v0.6.

def _build_b_per_unit_matched_mean(extras_dict, target_mean,
                                    b_min=B_MIN, b_max=B_MAX,
                                    tol=0.005, max_iter=80):
    """Binary search for an additive shift such that the post-
    clamp mean bit width equals target_mean within tol.

    The integer bit width per group is

        b_g = clip(round(target_mean + extras_g + shift),
                   b_min, b_max).

    The function chooses shift so that mean_g(b_g) ≈ target_mean.
    Returns (b_per_layer, info_dict).
    """
    def mean_b_at_shift(shift):
        ss, nn = 0.0, 0
        for n, e in extras_dict.items():
            raw = float(target_mean) + e.float() + shift
            b_int = torch.round(raw).clamp(min=b_min, max=b_max)
            ss += float(b_int.sum())
            nn += b_int.numel()
        return ss / max(nn, 1)

    s_lo, s_hi = -10.0, 10.0
    iters = 0
    for it in range(max_iter):
        iters = it + 1
        s_mid = (s_lo + s_hi) / 2
        m = mean_b_at_shift(s_mid)
        if abs(m - target_mean) < tol:
            break
        if m < target_mean:
            s_lo = s_mid
        else:
            s_hi = s_mid

    final_shift = (s_lo + s_hi) / 2

    out = {}
    for n, e in extras_dict.items():
        raw = float(target_mean) + e.float() + final_shift
        out[n] = torch.round(raw).clamp(min=b_min, max=b_max).long()

    flat = torch.cat([b.flatten().float() for b in out.values()])
    info = {
        "shift":         float(final_shift),
        "iterations":    iters,
        "achieved_mean": float(flat.mean()),
        "target_mean":   float(target_mean),
        "diff":          abs(float(flat.mean()) - float(target_mean)),
        "b_min":         b_min,
        "b_max":         b_max,
    }
    return out, info


# =============================================================
# === CELL V6 === PPL eval helper
# =============================================================

def ppl_with_quant(model, ids, b_per_layer=None, r_per_layer=None,
                    ctx=CTX, n_windows=N_WINDOWS):
    """Sliding-window PPL. If b_per_layer/r_per_layer are None,
    runs in 'off' mode (no quant). Otherwise quantise with the
    provided per-layer specs."""
    stride = ctx // 2
    if b_per_layer is None or r_per_layer is None:
        _C16["mode"]        = "off"
        _C16["b_per_layer"] = None
        _C16["r_per_layer"] = None
    else:
        _C16["mode"]        = "quantise"
        _C16["b_per_layer"] = b_per_layer
        _C16["r_per_layer"] = r_per_layer

    nll_total = 0.0
    tok_total = 0
    windows_done = 0
    try:
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
    finally:
        _C16["mode"]        = "off"
        _C16["b_per_layer"] = None
        _C16["r_per_layer"] = None
    return math.exp(nll_total / tok_total), windows_done


# =============================================================
# === CELL V7 === Per-model run
# =============================================================

def run_one_model(model_id, model_label, tok, test_ids):
    """Load the model, calibrate, and run all (b_0, profile)
    conditions. Returns the per-model result dict and writes a
    partial JSON checkpoint to PARTIAL_DIR."""
    print(f"\n{'=' * 80}")
    print(f"Running {model_label} ({model_id})")
    print(f"{'=' * 80}")

    mdl = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16,
        device_map="auto", trust_remote_code=True,
    )
    mdl.eval()
    print(f"  VRAM after load: "
          f"{torch.cuda.memory_allocated() / 1e9:.2f} GB")

    mixers = find_mixers(mdl)
    print(f"  Found {len(mixers)} Mamba-1 mixers")
    layer_name_by_id = {id(mix): name for name, mix in mixers}

    _MixerClass = type(mixers[0][1])
    _orig_slow_forward = _MixerClass.slow_forward
    _MixerClass.slow_forward = _make_c16_forward(mixers, layer_name_by_id)
    _force_slow_path(mdl)

    # Calibrate
    print(f"  Calibrating on WikiText-2 train (1024 tokens)...")
    calib_text = "\n\n".join(
        [x for x in load_dataset("wikitext", "wikitext-2-raw-v1",
                                  split="train")["text"]
         if x.strip()])[:8000]
    calib_ids = tok.encode(calib_text, return_tensors="pt").to("cuda")[:, :1024]

    _C16["mode"]        = "calibrate"
    _C16["calib_accum"] = {}
    with torch.inference_mode():
        _ = mdl(calib_ids, use_cache=False)
    _C16["mode"] = "off"

    captured = {}
    for name, accum in _C16["calib_accum"].items():
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

    r_per_layer = {n: RANGE_K * d["rms_h"] for n, d in captured.items()}

    # FP16 baseline
    print(f"\n  FP16 baseline (no quant)...")
    fp16_ppl, n_win = ppl_with_quant(mdl, test_ids,
                                       ctx=CTX, n_windows=N_WINDOWS)
    print(f"    FP16 PPL = {fp16_ppl:.4f} (n_windows={n_win})")

    # Per-(b_0, profile) sweep
    sweep = {"FP16_baseline": {"ppl": fp16_ppl, "n_windows": n_win,
                                "achieved_mean": None,
                                "shift": None,
                                "delta_pct": 0.0}}

    for b_0 in B0_VALUES:
        print(f"\n  b_0 = {b_0}")
        for prof_name in PROFILE_NAMES:
            extras_builder = PROFILE_BUILDERS[prof_name]
            extras = extras_builder(captured)
            b_per_layer, shift_info = _build_b_per_unit_matched_mean(
                extras, target_mean=b_0,
            )
            ppl, n_win = ppl_with_quant(
                mdl, test_ids, b_per_layer, r_per_layer,
                ctx=CTX, n_windows=N_WINDOWS,
            )
            delta_pct = 100.0 * (ppl - fp16_ppl) / fp16_ppl
            key = f"b{b_0}_{prof_name}"
            sweep[key] = {
                "b_0":           b_0,
                "profile":       prof_name,
                "ppl":           ppl,
                "delta_pct":     delta_pct,
                "n_windows":     n_win,
                "achieved_mean": shift_info["achieved_mean"],
                "shift":         shift_info["shift"],
                "iterations":    shift_info["iterations"],
                "diff_target":   shift_info["diff"],
            }
            print(f"    {prof_name:<16s} mean_b={shift_info['achieved_mean']:.3f} "
                  f"(shift={shift_info['shift']:+.3f}, "
                  f"iter={shift_info['iterations']}, "
                  f"diff={shift_info['diff']:.4f}) "
                  f"PPL={ppl:>8.4f} ΔPPL={delta_pct:+.3f}%")

    # Per-model calibration summary
    calib_summary = []
    for name, d in captured.items():
        calib_summary.append({
            "layer_name":   name,
            "rho_p50":      float(d["rho_emp"].flatten().quantile(0.50)),
            "gamma_p50":    float(d["gamma"].quantile(0.50)),
            "rms_h_p50":    float(d["rms_h"].quantile(0.50)),
        })

    model_result = {
        "model_id":              model_id,
        "model_label":           model_label,
        "n_mixers":              len(mixers),
        "fp16_baseline_ppl":     fp16_ppl,
        "sweep":                 sweep,
        "calib_summary":         calib_summary,
        "date":                  datetime.utcnow().isoformat(timespec="seconds"),
    }

    # Save partial JSON checkpoint
    safe_label = model_label.replace("/", "_").replace(" ", "_")
    partial_path = os.path.join(
        PARTIAL_DIR, f"phase_c16_partial_{safe_label}.json"
    )
    with open(partial_path, "w") as f:
        json.dump(model_result, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Saved partial -> {partial_path}")

    # Restore original forward and unload
    _MixerClass.slow_forward = _orig_slow_forward
    del mdl, mixers
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"  VRAM after unload: "
          f"{torch.cuda.memory_allocated() / 1e9:.2f} GB")

    return model_result


# =============================================================
# === CELL V8 === Main loop
# =============================================================

# Load tokenizer (shared across all models in MODEL_LIST since
# they all use the same tokenizer family)
tok = None
for tid in TOKENIZER_FALLBACKS:
    try:
        tok = AutoTokenizer.from_pretrained(tid, trust_remote_code=True)
        print(f"  Tokenizer:  {tid}")
        break
    except Exception:
        pass
assert tok is not None

# WikiText-2 test text (shared)
test_text = "\n\n".join(
    [x for x in load_dataset("wikitext", "wikitext-2-raw-v1",
                              split="test")["text"]
     if x.strip()])[:200_000]
test_ids = tok.encode(test_text, return_tensors="pt").to("cuda")
print(f"  Test corpus: WikiText-2 test, {test_ids.shape[1]:,} tokens")


# Sweep all models
all_results = {}
for model_id, model_label in MODEL_LIST:
    try:
        all_results[model_label] = run_one_model(
            model_id, model_label, tok, test_ids,
        )
    except Exception as e:
        print(f"\n  [ERROR] {model_label} failed: "
              f"{type(e).__name__}: {str(e)[:200]}")
        all_results[model_label] = {
            "model_id":   model_id,
            "model_label": model_label,
            "error":      f"{type(e).__name__}: {str(e)[:500]}",
        }


# =============================================================
# === CELL V9 === Verdict and final save
# =============================================================

print(f"\n{'=' * 80}")
print(f"Phase C-16 verdict: matched-mean head-to-head across "
      f"{len(MODEL_LIST)} models")
print(f"{'=' * 80}")

# Headline table: ΔPPL by (model, profile, b_0) at matched mean
print(f"\nHeadline ΔPPL (% of FP16 baseline), matched mean bit width:\n")

for b_0 in B0_VALUES:
    print(f"\n  b_0 = {b_0}")
    print(f"  {'model':<14s}  {'FP16 PPL':>9s}  "
          f"{'uniform':>10s}  {'full_slope':>11s}  "
          f"{'half_slope':>11s}  {'half_rg':>10s}")
    print(f"  {'-' * 14}  {'-' * 9}  {'-' * 10}  "
          f"{'-' * 11}  {'-' * 11}  {'-' * 10}")
    for _, model_label in MODEL_LIST:
        r = all_results.get(model_label, {})
        if "error" in r:
            print(f"  {model_label:<14s}  [ERROR]")
            continue
        fp16 = r.get("fp16_baseline_ppl", float("nan"))
        cells = []
        for prof in PROFILE_NAMES:
            key = f"b{b_0}_{prof}"
            d = r.get("sweep", {}).get(key, {})
            cells.append(d.get("delta_pct", float("nan")))
        print(f"  {model_label:<14s}  {fp16:>9.4f}  "
              + "  ".join(f"{c:>+9.3f}%" for c in cells))

# H1 check: half_slope_rg vs full_slope at matched mean
print(f"\n\n  H1 (half_slope_rg vs full_slope at matched mean):")
all_pass_h1 = True
for b_0 in B0_VALUES:
    print(f"\n    b_0 = {b_0}")
    for _, model_label in MODEL_LIST:
        r = all_results.get(model_label, {})
        if "error" in r:
            continue
        full_d = r["sweep"].get(f"b{b_0}_full_slope", {}).get("delta_pct")
        rg_d = r["sweep"].get(f"b{b_0}_half_slope_rg", {}).get("delta_pct")
        full_mb = r["sweep"].get(f"b{b_0}_full_slope", {}).get("achieved_mean")
        rg_mb = r["sweep"].get(f"b{b_0}_half_slope_rg", {}).get("achieved_mean")
        if full_d is None or rg_d is None:
            continue
        diff = full_d - rg_d
        winner = ("half_rg" if rg_d < full_d - 0.5
                  else "full_slope" if full_d < rg_d - 0.5 else "tie")
        matched = "✓" if (full_mb is not None and rg_mb is not None
                          and abs(full_mb - rg_mb) < 0.05) else "≠"
        if winner != "half_rg":
            all_pass_h1 = False
        print(f"      {model_label:<14s} full={full_d:+7.3f}% (mb={full_mb:.3f})  "
              f"half_rg={rg_d:+7.3f}% (mb={rg_mb:.3f})  "
              f"diff={diff:+.3f}pp matched={matched} winner={winner}")

if all_pass_h1:
    print(f"\n  H1 PASS ✓ — half_slope_rg wins at matched mean on every "
          f"(model, b_0) cell.")
else:
    print(f"\n  H1 PARTIAL — half_slope_rg does not win at matched mean "
          f"on every cell; see per-cell winners above.")

# H3 check: deployment table update at b_0 = 6, half_slope_rg
print(f"\n\n  H3 (state-spaces 5-size summary, b_0 = 6):")
print(f"\n    {'model':<14s}  {'full_slope ΔPPL':>16s}  "
      f"{'half_slope_rg ΔPPL':>20s}  {'improvement':>12s}")
for _, model_label in MODEL_LIST:
    r = all_results.get(model_label, {})
    if "error" in r:
        continue
    full_d = r["sweep"].get("b6_full_slope", {}).get("delta_pct")
    rg_d = r["sweep"].get("b6_half_slope_rg", {}).get("delta_pct")
    if full_d is None or rg_d is None:
        continue
    impr = full_d - rg_d
    print(f"    {model_label:<14s}  {full_d:>+15.3f}%  "
          f"{rg_d:>+19.3f}%  {impr:>+11.3f}pp")


# Final merged JSON
output = {
    "date":            datetime.utcnow().isoformat(timespec="seconds"),
    "phase":           "C-16",
    "purpose":         ("Matched-mean full sweep with the candidate "
                        "deployment recipe (half_slope_rg). Closes "
                        "the section 6.7 / round-4 critique by "
                        "introducing a binary-search shift "
                        "normalisation that exactly matches mean "
                        "bit width across the four profiles, and "
                        "by running the full state-spaces Mamba-1 "
                        "family (5 sizes) at b_0 in {4, 6}."),
    "transformers_version": _hf_transformers.__version__,
    "model_list":      [{"id": m[0], "label": m[1]} for m in MODEL_LIST],
    "protocol": {
        "ctx":              CTX,
        "n_windows":        N_WINDOWS,
        "b_0_values":       B0_VALUES,
        "b_min":            B_MIN,
        "b_max":            B_MAX,
        "range_k":          RANGE_K,
        "calib_split":      "WikiText-2 train (1024 tokens)",
        "test_split":       "WikiText-2 test",
        "profiles":         PROFILE_NAMES,
        "shift_method":     "binary search, tol=0.005, max_iter=80",
        "quant_scheme":     "per-(c, s) INT-b symmetric mid-tread, range = k * RMS(h)",
    },
    "results":         all_results,
    "h1_all_pass":     all_pass_h1,
}

out_path = "/content/phase_c16_matched_mean_sweep.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False, default=str)
print(f"\nFinal merged JSON saved -> {out_path}")


# =============================================================
# === READING ===
# =============================================================
#
# Outcome interpretation (pre-registered):
#
#   H1 PASS (half_slope_rg wins at matched mean on every cell):
#     The candidate deployment recipe is structurally tighter
#     than the deployed full_slope simplification, even after
#     mean-bit matching. The v0.6 manuscript's recommendation
#     of half_slope_rg is empirically supported, and the v1
#     submission can adopt half_slope_rg as the main allocator
#     with the present data table.
#
#   H1 PARTIAL (half_slope_rg wins on some cells, ties on
#               others):
#     The structural advantage is real but model-dependent. The
#     v1 submission should still recommend half_slope_rg but
#     report per-model winners explicitly, treating the recipe
#     change as model-conditional.
#
#   H1 FAIL (full_slope wins at matched mean on most cells):
#     The section 6.7 advantage was an extra-bits artefact. The
#     v1 submission should revert main character to full_slope
#     and present half_slope_rg as a 'theoretically motivated
#     candidate' that does not improve on the deployed
#     simplification at matched mean. This would be a clean
#     scientific result (the heuristic is empirically near-
#     optimal at integer-bit granularity) but would walk back
#     the v0.5 / v0.6 main-character switch.
#
# Whatever the verdict, the matched-mean experiment closes the
# round-4 critique that the head-to-head was confounded by mean
# bit overshoot.
# ============================================================
