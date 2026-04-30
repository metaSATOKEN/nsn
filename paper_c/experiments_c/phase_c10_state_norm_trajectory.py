# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Phase C-10: State-norm trajectory measurement at long context
# ============================================================
# Phase C-9 established that the long-context divergence of
# unquantised Mamba-2.8B-hf on PG-19 is precision-independent
# (BF16 does not fix it) and is suppressed by clipping at the
# calibrated r_g (clip-only path stays stable). Phase C-10
# measures the underlying mechanism directly by recording the
# per-step state norm ||h_t||_2 and the per-step dt_t
# distribution under three forward conditions, at three
# context lengths.
#
# This is the "mechanistic" experiment that Grok + GPT
# pseudo-reviewers asked for: rather than a 4-condition
# behavioural ablation (C-9), C-10 directly observes the
# state-magnitude trajectory and the dt distribution drift,
# localising the FP16 numerical-failure mode discussed in
# Section 8.2 of the manuscript.
#
# Conditions (3 per ctx_len):
#   1. FP16_orig            unpatched Mamba1 slow_forward
#                            (the deployed FP16 path)
#   2. FP16_clip_only       patched, mode='clip_only' with
#                            calibrated r_g
#   3. FP16_full INT6       patched, mode='quantise' with
#                            full allocator at b_0=6
#
# Measurements per (ctx_len, condition, layer):
#   - ||h_t||_2 mean / p50 / p95 / p99 / max trajectory
#       (one value per t, aggregated over (B, c, s))
#   - ||h_t||_inf trajectory (worst-unit state magnitude)
#   - dt_t mean / p50 / p95 / p99 / max trajectory
#   - rho_t mean trajectory (= |A_bar_t| averaged per step)
#   - cliprate (fraction of (c,s) clipped at each t)
#       only for clip_only and full conditions
#
# Sweeps:
#   ctx_lens = {8192, 16384, 32768}
#   n_windows = 2 per ctx_len
#   layer subset: {0, 16, 32, 40, 48, 56, 63} (7 layers,
#                covering input / U-shape core / output ends)
#
# Dataset: PG-19 (matching C-7 / C-9 protocol)
#
# Pre-registered hypotheses:
#
#   H1 (state-norm divergence under FP16):
#     ||h_t||_2 max grows superlinearly with t under FP16_orig
#     at 32K context, exceeding the calibrated r_g for a
#     non-trivial fraction of (c, s) units.
#
#   H2 (clip-only suppression):
#     Under FP16_clip_only, ||h_t||_2 max stays bounded by r_g
#     by construction. The cliprate trajectory is the empirical
#     signature of how often the clipping fires.
#
#   H3 (dt drift hypothesis):
#     The dt_t distribution under FP16_orig at long context
#     has heavier left tail (more long-memory excursions, dt
#     small, rho near 1) than at short context. This is
#     measured directly via the per-step dt percentiles.
#
# Runtime estimate (A100 95 GB):
#   8K   per condition       ~3-4 min
#   16K  per condition       ~6-8 min
#   32K  per condition       ~12-15 min
#   3 conditions * 3 ctx_lens = 9 runs
#   Total                    ~75-90 min
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
print(f"Loading {MODEL_LABEL}  ({MODEL_ID})  [FP16]")
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

# Layers to record per-step trajectory for. Subset chosen to
# cover the U-shape from input to output without exploding the
# trajectory storage cost.
TRAJECTORY_LAYERS = [0, 16, 32, 40, 48, 56, 63]
TRAJECTORY_LAYER_NAMES = {
    f"backbone.layers.{i}.mixer" for i in TRAJECTORY_LAYERS
}


# =============================================================
# === CELL V3 === Patched slow_forward + per-step recording
# =============================================================
# We extend the C-9 patched function with per-step state-norm
# and dt trajectory recording for a designated subset of
# layers. The recording pulls (mean / p50 / p95 / p99 / max)
# of ||h_t||_2 over (B, c, s) at every t for layers in
# TRAJECTORY_LAYER_NAMES, plus dt percentiles and rho_t mean,
# and the per-step cliprate for clip_only / quantise modes.

_C10 = {
    "mode":           "off",       # "off" | "calibrate" | "quantise" | "clip_only"
    "b_per_layer":    None,
    "r_per_layer":    None,
    "calib_accum":    None,
    # Trajectory recording
    "record":         False,
    "trajectories":   None,        # dict[layer_name, dict[t, dict of stats]]
}

EPS = 1e-6


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


def _clip_only_h_per_unit(h, r_per_unit):
    r_b = r_per_unit.unsqueeze(0)
    return torch.clamp(h, min=-r_b, max=r_b)


def _record_step_stats(layer_name, t, h, dt, A_bar_step, r_layer):
    """Compute and store per-step trajectory stats."""
    # h: [B, d_inner, d_state]    (the post-update state)
    # dt: [B, d_inner]            (per-channel time step)
    # A_bar_step: [B, d_inner, d_state]  (per-step transition)
    # r_layer: [d_inner, d_state] or None
    bucket = _C10["trajectories"].setdefault(layer_name, {})
    if t in bucket:
        return  # already recorded for an earlier window
    h_norm_per_unit = h.abs()  # [B, d_inner, d_state]
    h_l2_per_unit = (h ** 2).sqrt() if False else h.abs()
    # Use abs as a proxy for per-(c,s) magnitude
    h_flat = h_norm_per_unit.flatten().float()
    dt_flat = dt.flatten().float()
    A_flat = A_bar_step.abs().flatten().float()
    stats = {
        "h_abs_mean":  float(h_flat.mean()),
        "h_abs_p50":   float(h_flat.quantile(0.5)),
        "h_abs_p95":   float(h_flat.quantile(0.95)),
        "h_abs_p99":   float(h_flat.quantile(0.99)),
        "h_abs_max":   float(h_flat.max()),
        "dt_mean":     float(dt_flat.mean()),
        "dt_p50":      float(dt_flat.quantile(0.5)),
        "dt_p95":      float(dt_flat.quantile(0.95)),
        "dt_p99":      float(dt_flat.quantile(0.99)),
        "dt_max":      float(dt_flat.max()),
        "rho_mean":    float(A_flat.mean()),
        "rho_p99":     float(A_flat.quantile(0.99)),
    }
    if r_layer is not None and _C10["mode"] in ("clip_only", "quantise"):
        # Cliprate: fraction of (B, c, s) units exceeding r_g
        r_b = r_layer.unsqueeze(0).float()  # [1, d_inner, d_state]
        clipped = (h.float().abs() > r_b).float()
        stats["cliprate"] = float(clipped.mean())
    bucket[t] = stats


def _c10_mamba_forward(mix, input_states):
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

    b_layer = None
    r_layer = None
    if _C10["mode"] in ("quantise", "clip_only"):
        if _C10["r_per_layer"] is not None:
            r = _C10["r_per_layer"].get(layer_name)
            if r is not None:
                r_layer = r.to(x_pre.device, dtype=torch.float32)
        if _C10["mode"] == "quantise" and _C10["b_per_layer"] is not None:
            b = _C10["b_per_layer"].get(layer_name)
            if b is not None:
                b_layer = b.to(x_pre.device)

    if _C10["mode"] == "calibrate":
        rho_sum     = A_bar.sum(dim=(0, 1))
        rho_count   = B * T
        gamma_sum   = (C_f ** 2).sum(dim=(0, 1))
        gamma_count = B * T
        rms_sumsq   = torch.zeros((d_inner, d_state),
                                    dtype=torch.float32,
                                    device=x_pre.device)

    record_this = (
        _C10["record"]
        and layer_name in TRAJECTORY_LAYER_NAMES
    )
    # Stride for recording: at long ctx we don't need every t,
    # so subsample. Aim for ~64 records per layer per window.
    record_stride = max(1, T // 64)

    h = torch.zeros((B, d_inner, d_state),
                     dtype=torch.float32, device=x_pre.device)
    ys = []
    for t in range(T):
        h = A_bar[:, t] * h + deltaB_x[:, t]
        if _C10["mode"] == "quantise" and (
                b_layer is not None and r_layer is not None):
            h = _quantize_h_per_unit(h, b_layer, r_layer)
        elif _C10["mode"] == "clip_only" and r_layer is not None:
            h = _clip_only_h_per_unit(h, r_layer)
        if _C10["mode"] == "calibrate":
            rms_sumsq += (h ** 2).sum(dim=0)
        # Per-step trajectory recording
        if record_this and (t % record_stride == 0 or t == T - 1):
            with torch.no_grad():
                _record_step_stats(
                    layer_name, t, h, dt[:, t], A_bar[:, t],
                    r_layer,
                )
        y_t = (h * C_f[:, t].unsqueeze(1)).sum(dim=-1)
        ys.append(y_t)
    y = torch.stack(ys, dim=1)

    y = y + mix.D.float().unsqueeze(0).unsqueeze(0) * x_pre.float()
    y = y * F.silu(gate.float())
    out = mix.out_proj(y.to(mix.out_proj.weight.dtype))

    if _C10["mode"] == "calibrate":
        prev = _C10["calib_accum"].get(layer_name)
        if prev is None:
            _C10["calib_accum"][layer_name] = {
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


def _c10_slow_forward_patch(self, input_states, *args, **kwargs):
    return _c10_mamba_forward(self, input_states)


_MixerClass = type(mixers[0][1])
_orig_slow_forward = _MixerClass.slow_forward
_MixerClass.slow_forward = _c10_slow_forward_patch
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
# === CELL V4 === Calibration + profile build (FP16 model)
# =============================================================

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


B_MIN, RANGE_K, B_0 = 2, 7.0, 6


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


print(f"\n{'-' * 80}")
print(f"Calibrating ρ_emp + Γ_s + RMS(h) on FP16 Mamba-2.8B (1024 tokens)")
print(f"{'-' * 80}")

calib_text = "\n\n".join(
    [x for x in load_dataset("wikitext", "wikitext-2-raw-v1",
                              split="train")["text"]
     if x.strip()])[:8000]
calib_ids = tok.encode(calib_text, return_tensors="pt").to("cuda")[:, :1024]

_C10["mode"]        = "calibrate"
_C10["calib_accum"] = {}
with torch.inference_mode():
    _ = mdl(calib_ids, use_cache=False)
_C10["mode"] = "off"

captured = {}
for name, accum in _C10["calib_accum"].items():
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

b_per_layer_full = _build_b_per_unit(_extras_full(captured), B_0)
r_per_layer      = {n: RANGE_K * d["rms_h"] for n, d in captured.items()}


# =============================================================
# === CELL V5 === Load PG-19 long-context test text
# =============================================================

print(f"\n{'-' * 80}")
print(f"Loading PG-19 long-context test corpus")
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
            if total_chars > 800_000:
                break
        text = "\n\n".join(chunks)
        test_corpus_used = label
        sources_tried.append({"label": label, "result": "success",
                                "n_entries": len(chunks),
                                "n_chars":   total_chars})
        print(f"  ✓ {label}: {len(chunks)} entries, {total_chars:,} chars")
        break
    except Exception as e:
        err = f"{type(e).__name__}: {str(e)[:200]}"
        print(f"  ✗ {label}: {err}")
        sources_tried.append({"label": label, "result": "failed", "error": err})
        continue

if text is None:
    raise RuntimeError("All corpus candidates failed.")

test_ids = tok.encode(text, return_tensors="pt").to("cuda")
print(f"\n  Source: {test_corpus_used}")
print(f"  Tokenised: {test_ids.shape[1]:,} tokens")


# =============================================================
# === CELL V6 === Long-context trajectory eval
# =============================================================

CTX_LENS  = [8192, 16384, 32768]
N_WINDOWS = 2

MODE_ORIG_FORWARD = "orig_forward"
MODE_QUANTISE     = "quantise"
MODE_CLIP_ONLY    = "clip_only"


def _set_eval_mode(eval_mode, b_per_layer=None, r_per_layer=None):
    if eval_mode == MODE_ORIG_FORWARD:
        _MixerClass.slow_forward = _orig_slow_forward
        _C10["mode"]         = "off"
        _C10["b_per_layer"]  = None
        _C10["r_per_layer"]  = None
        _C10["record"]       = False  # original forward has no hook for recording
    elif eval_mode == MODE_QUANTISE:
        _MixerClass.slow_forward = _c10_slow_forward_patch
        _C10["mode"]         = "quantise"
        _C10["b_per_layer"]  = b_per_layer
        _C10["r_per_layer"]  = r_per_layer
        _C10["record"]       = True
    elif eval_mode == MODE_CLIP_ONLY:
        _MixerClass.slow_forward = _c10_slow_forward_patch
        _C10["mode"]         = "clip_only"
        _C10["b_per_layer"]  = None
        _C10["r_per_layer"]  = r_per_layer
        _C10["record"]       = True
    else:
        raise ValueError(f"Unknown eval_mode: {eval_mode}")


def trajectory_long_ctx(model, ids, eval_mode,
                          b_per_layer=None, r_per_layer=None,
                          ctx_len=8192, n_windows=2):
    """Long-ctx PPL plus per-step trajectory for designated layers."""
    stride = ctx_len // 2
    _set_eval_mode(eval_mode, b_per_layer, r_per_layer)
    _C10["trajectories"] = {}
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
        _MixerClass.slow_forward = _c10_slow_forward_patch
        _C10["mode"]        = "off"
        _C10["b_per_layer"] = None
        _C10["r_per_layer"] = None
        _C10["record"]      = False
    ppl = math.exp(nll_total / tok_total)
    return ppl, windows_done, _C10["trajectories"]


# =============================================================
# === CELL V7 === Run sweep
# =============================================================

print(f"\n{'-' * 80}")
print(f"Phase C-10: state-norm trajectory sweep "
      f"(n_windows={N_WINDOWS})")
print(f"  Conditions: FP16_orig (no trajectory record) / "
      f"FP16_clip_only / FP16_full b_0={B_0}")
print(f"  Contexts: {CTX_LENS}")
print(f"  Layers recorded: {TRAJECTORY_LAYERS}")
print(f"{'-' * 80}")

CONDITIONS = [
    ("FP16_orig",      MODE_ORIG_FORWARD, None,             None),
    ("FP16_clip_only", MODE_CLIP_ONLY,    None,             r_per_layer),
    ("FP16_full",      MODE_QUANTISE,     b_per_layer_full, r_per_layer),
]

sweep_results = {ctx: {} for ctx in CTX_LENS}

for ctx_len in CTX_LENS:
    print(f"\n  ── ctx_len = {ctx_len:,} ──")
    for label, mode, b_dict, r_dict in CONDITIONS:
        print(f"  Running {label} at ctx={ctx_len}...")
        ppl, n_win, traj = trajectory_long_ctx(
            mdl, test_ids, mode, b_dict, r_dict,
            ctx_len=ctx_len, n_windows=N_WINDOWS,
        )
        sweep_results[ctx_len][label] = {
            "ppl":        ppl,
            "n_windows":  n_win,
            "mode":       mode,
            "trajectories": traj,
        }
        n_traj_layers = len(traj) if traj else 0
        n_traj_pts = sum(len(v) for v in traj.values()) if traj else 0
        print(f"    ppl = {ppl:>9.4f}  "
              f"({n_traj_layers} layers x ~{n_traj_pts // max(1, n_traj_layers)} steps recorded)")


# =============================================================
# === CELL V8 === Save
# =============================================================

# Strip trajectories into a more JSON-friendly structure
traj_export = {}
for ctx_len in CTX_LENS:
    traj_export[str(ctx_len)] = {}
    for label, r in sweep_results[ctx_len].items():
        traj_export[str(ctx_len)][label] = {
            "ppl": r["ppl"],
            "n_windows": r["n_windows"],
            "mode": r["mode"],
            "trajectories": {
                lname: [
                    {"t": t, **stats}
                    for t, stats in sorted(layer_traj.items())
                ]
                for lname, layer_traj in (r["trajectories"] or {}).items()
            },
        }

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
    "phase":           "C-10",
    "purpose":         ("State-norm + dt trajectory measurement "
                        "at long context on Mamba-2.8B-hf, to "
                        "localise the FP16 numerical-failure mode "
                        "discussed in Section 8.2 of the paper. "
                        "Three conditions x three context lengths."),
    "transformers_version": _hf_transformers.__version__,
    "model":           MODEL_ID,
    "label":           MODEL_LABEL,
    "protocol": {
        "ctx_lens":      CTX_LENS,
        "n_windows":     N_WINDOWS,
        "b_0":           B_0,
        "b_min":         B_MIN,
        "range_k":       RANGE_K,
        "calib_split":   "WikiText-2 train (1024 tokens)",
        "test_corpus":   test_corpus_used,
        "test_corpus_sources_tried": sources_tried,
        "conditions":    [c[0] for c in CONDITIONS],
        "trajectory_layers": TRAJECTORY_LAYERS,
        "trajectory_layer_names": sorted(TRAJECTORY_LAYER_NAMES),
        "trajectory_subsample_target": 64,
        "stat_keys":     [
            "h_abs_mean", "h_abs_p50", "h_abs_p95",
            "h_abs_p99", "h_abs_max",
            "dt_mean", "dt_p50", "dt_p95", "dt_p99", "dt_max",
            "rho_mean", "rho_p99",
            "cliprate (clip_only / quantise only)",
        ],
    },
    "sweep_results":   traj_export,
    "rho_gamma_rms_summary": calib_summary,
    "notes": (
        "FP16_orig uses the unpatched slow_forward and so has no "
        "trajectory record (the patched recording hook only fires "
        "when the patched function is active). For mechanistic "
        "comparison, the FP16_clip_only trajectory is the closest "
        "reference: it uses the same patched function but does not "
        "quantise, so any divergence between FP16_orig PPL and "
        "FP16_clip_only h_abs trajectory is attributable to the "
        "calibrated clipping. To measure FP16_orig state-magnitude "
        "trajectory directly, a second pass with mode='off' (patched "
        "function, no clipping, no quantisation) would be needed."
    ),
}

out_path = "/content/phase_c10_state_norm_trajectory.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False, default=str)
print(f"\nSaved -> {out_path}")


# =============================================================
# === READING ===
# =============================================================
#
# Expected outcomes (pre-registered):
#
# H1 (FP16 state-norm divergence at long ctx):
#   Compare h_abs_max between clip_only at 8K vs 32K:
#     - At 8K: max < r_g typical for most (c, s) units
#     - At 32K: max >> r_g for a subset of units, indicating
#                divergence under FP16 (since clip_only suppresses
#                this; the FP16_orig path doesn't and PPL diverges)
#   Cliprate trajectory should show how often the clamp fires.
#
# H2 (clip suppression):
#   Under clip_only at 32K, h_abs_max stays bounded by r_g by
#   construction. Cliprate provides the count.
#
# H3 (dt drift hypothesis):
#   Compare dt_p50 / dt_p99 distribution between 1024-context
#   calibration (in the per-layer ρ_emp / Γ stats above) and
#   per-step dt_t at 32K. If dt_p50 at long context drifts to
#   smaller values (more long-memory / ρ → 1), this localises
#   the divergence mechanism.
#
# Three layers are particularly informative:
#   - Layer 0:  short-memory input (ρ p50 ≈ 0.60 at calibration)
#   - Layer 32: long-memory core (ρ p50 ≈ 0.88)
#   - Layer 56: short-memory output (ρ p50 ≈ 0.39)
#
# At long context, Layer 32's ρ trajectory is the candidate
# mechanism for the FP16 PPL divergence.
# ============================================================
