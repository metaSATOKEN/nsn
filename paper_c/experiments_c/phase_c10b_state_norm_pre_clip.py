# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Phase C-10b: Pre-clip state-norm trajectory (C-10 fix)
# ============================================================
# C-10's cliprate measurement was computed AFTER the clip /
# quantise step had been applied, so it was always zero by
# construction (post-clip h is in [-r_b, r_b] strictly). This
# was a script flaw, not a finding — it told us nothing about
# how often clipping actually fired.
#
# C-10b fixes this with two changes:
#
#   (1) The trajectory recording is moved to BEFORE the clip /
#       quantise step in the patched forward, so that cliprate
#       measures the actual fraction of units whose pre-clip
#       state magnitude exceeds the calibrated r_g.
#
#   (2) A new condition "FP32_patched_off" is added (mode =
#       "off_record"): the patched forward function with
#       neither clipping nor quantisation, but with trajectory
#       recording active. This produces the trajectory under
#       the unmodified recurrence (which by the C-9 H0 sanity
#       check is bit-for-bit identical to FP16_orig).
#
# With these fixes, the trajectory data answers two distinct
# questions:
#
#   - "How often would clipping fire if we did not clip?"
#       -> cliprate from FP32_patched_off mode (no intervention,
#          natural state magnitudes recorded)
#
#   - "How often does clipping actually fire?"
#       -> cliprate from FP16_clip_only mode (now correctly
#          measured pre-clip)
#
# These two should agree on average (the clip operation is
# applied to the same step's pre-clip state in both), but the
# downstream trajectories diverge after the first firing event.
#
# Conditions (4 total, n_windows = 2):
#   1. FP16_orig            unpatched, no trajectory record
#                            (PPL reference only)
#   2. FP32_patched_off    patched, mode = "off_record":
#                            no clip, no quantise, trajectory
#                            recording active. Same forward
#                            mathematics as FP16_orig (per C-9
#                            H0 sanity).
#   3. FP16_clip_only      patched, mode = "clip_only":
#                            clip applied, trajectory recording
#                            now BEFORE the clip step.
#   4. FP16_full INT6      patched, mode = "quantise":
#                            quant applied, trajectory recording
#                            now BEFORE the quant step.
#
# Sweep: ctx_lens = {8192, 16384, 32768}, n_windows = 2,
# layers recorded = {0, 16, 32, 40, 48, 56, 63}.
#
# Pre-registered hypotheses:
#
#   H1 (clipping firing rate increases with ctx):
#     Under FP16_clip_only, cliprate at the END of the
#     32K window > cliprate at the END of the 8K window,
#     particularly on long-memory layers (32, 40).
#
#   H2 (FP32_patched_off natural magnitude):
#     Under FP32_patched_off, h_abs_max trajectory at long
#     ctx grows monotonically and exceeds r_b on a non-
#     trivial fraction of (c, s) units. This is the direct
#     measurement of the magnitude-growth mechanism.
#
#   H3 (clip and off agree on cliprate at t = 0):
#     At the first recording step, cliprate from
#     FP32_patched_off and FP16_clip_only should agree
#     (since no firing event has happened yet). As t grows,
#     the trajectories diverge and the two cliprates can
#     differ substantially.
#
# Runtime estimate (A100 95 GB):
#   ~ same as C-10, ~75-90 min total.
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

TRAJECTORY_LAYERS = [0, 16, 32, 40, 48, 56, 63]
TRAJECTORY_LAYER_NAMES = {
    f"backbone.layers.{i}.mixer" for i in TRAJECTORY_LAYERS
}


# =============================================================
# === CELL V3 === Patched slow_forward — recording moved BEFORE clip
# =============================================================
# The critical fix vs C-10 is the position of the recording
# call: it now happens immediately after the state update and
# BEFORE the clip / quantise step. cliprate therefore measures
# the natural pre-clip rate.

_C10b = {
    "mode":           "off",
    # Modes:
    #   "off"           unpatched semantics, no record (used for
    #                   calibration only since unpatched forward
    #                   handles eval directly)
    #   "off_record"    patched, no clip/quant, recording active.
    #                   The new mode added in C-10b.
    #   "calibrate"     standard calibration mode (unchanged)
    #   "quantise"      patched + INT-b quant, recording active
    #   "clip_only"     patched + clip only, recording active
    "b_per_layer":    None,
    "r_per_layer":    None,
    "calib_accum":    None,
    "record":         False,
    "trajectories":   None,
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
    """Record per-step trajectory stats. h here is PRE-clip."""
    bucket = _C10b["trajectories"].setdefault(layer_name, {})
    if t in bucket:
        return
    h_flat = h.float().abs().flatten()
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
    if r_layer is not None:
        # Pre-clip cliprate: fraction of (B, c, s) units whose
        # natural state magnitude exceeds r_g. THIS is the
        # mechanism signal we want.
        r_b = r_layer.unsqueeze(0).float()
        exceeds = (h.float().abs() > r_b).float()
        stats["cliprate_pre"] = float(exceeds.mean())
        # Also record per-(c, s) max-over-units cliprate at this
        # step (was any single (c, s) unit exceeding r_b in any
        # batch element)
        # Reduce over batch dim to get (c, s) shape, then over
        # (c, s) to scalar
        unit_exceeded = exceeds.amax(dim=0)  # [d_inner, d_state]
        stats["cliprate_unit_pre"] = float(unit_exceeded.mean())
    bucket[t] = stats


def _c10b_mamba_forward(mix, input_states):
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
    if _C10b["mode"] in ("quantise", "clip_only", "off_record"):
        if _C10b["r_per_layer"] is not None:
            r = _C10b["r_per_layer"].get(layer_name)
            if r is not None:
                r_layer = r.to(x_pre.device, dtype=torch.float32)
        if _C10b["mode"] == "quantise" and _C10b["b_per_layer"] is not None:
            b = _C10b["b_per_layer"].get(layer_name)
            if b is not None:
                b_layer = b.to(x_pre.device)

    if _C10b["mode"] == "calibrate":
        rho_sum     = A_bar.sum(dim=(0, 1))
        rho_count   = B * T
        gamma_sum   = (C_f ** 2).sum(dim=(0, 1))
        gamma_count = B * T
        rms_sumsq   = torch.zeros((d_inner, d_state),
                                    dtype=torch.float32,
                                    device=x_pre.device)

    record_this = (
        _C10b["record"]
        and layer_name in TRAJECTORY_LAYER_NAMES
    )
    record_stride = max(1, T // 64)

    h = torch.zeros((B, d_inner, d_state),
                     dtype=torch.float32, device=x_pre.device)
    ys = []
    for t in range(T):
        h = A_bar[:, t] * h + deltaB_x[:, t]
        # ---------- C-10b fix: record BEFORE clip / quant ----------
        if record_this and (t % record_stride == 0 or t == T - 1):
            with torch.no_grad():
                _record_step_stats(
                    layer_name, t, h, dt[:, t], A_bar[:, t], r_layer,
                )
        # ----------------------------------------------------------
        if _C10b["mode"] == "quantise" and (
                b_layer is not None and r_layer is not None):
            h = _quantize_h_per_unit(h, b_layer, r_layer)
        elif _C10b["mode"] == "clip_only" and r_layer is not None:
            h = _clip_only_h_per_unit(h, r_layer)
        # mode == "off_record" leaves h unchanged (no clip / quant)
        if _C10b["mode"] == "calibrate":
            rms_sumsq += (h ** 2).sum(dim=0)
        y_t = (h * C_f[:, t].unsqueeze(1)).sum(dim=-1)
        ys.append(y_t)
    y = torch.stack(ys, dim=1)

    y = y + mix.D.float().unsqueeze(0).unsqueeze(0) * x_pre.float()
    y = y * F.silu(gate.float())
    out = mix.out_proj(y.to(mix.out_proj.weight.dtype))

    if _C10b["mode"] == "calibrate":
        prev = _C10b["calib_accum"].get(layer_name)
        if prev is None:
            _C10b["calib_accum"][layer_name] = {
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


def _c10b_slow_forward_patch(self, input_states, *args, **kwargs):
    return _c10b_mamba_forward(self, input_states)


_MixerClass = type(mixers[0][1])
_orig_slow_forward = _MixerClass.slow_forward
_MixerClass.slow_forward = _c10b_slow_forward_patch
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
# === CELL V4 === Calibration + profile build
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

_C10b["mode"]        = "calibrate"
_C10b["calib_accum"] = {}
with torch.inference_mode():
    _ = mdl(calib_ids, use_cache=False)
_C10b["mode"] = "off"

captured = {}
for name, accum in _C10b["calib_accum"].items():
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

# Modes used in the eval loop
MODE_ORIG_FORWARD     = "orig_forward"      # unpatched, no record
MODE_OFF_RECORD       = "off_record"        # patched, no clip/quant, record
MODE_QUANTISE         = "quantise"          # patched, INT-b quant, record (pre-quant)
MODE_CLIP_ONLY        = "clip_only"         # patched, clip only, record (pre-clip)


def _set_eval_mode(eval_mode, b_per_layer=None, r_per_layer=None):
    if eval_mode == MODE_ORIG_FORWARD:
        _MixerClass.slow_forward = _orig_slow_forward
        _C10b["mode"]         = "off"
        _C10b["b_per_layer"]  = None
        _C10b["r_per_layer"]  = None
        _C10b["record"]       = False
    elif eval_mode == MODE_OFF_RECORD:
        _MixerClass.slow_forward = _c10b_slow_forward_patch
        _C10b["mode"]         = "off_record"
        _C10b["b_per_layer"]  = None
        # r_per_layer needed so cliprate can be computed; the
        # value is NOT used to clip anything in off_record mode
        _C10b["r_per_layer"]  = r_per_layer
        _C10b["record"]       = True
    elif eval_mode == MODE_QUANTISE:
        _MixerClass.slow_forward = _c10b_slow_forward_patch
        _C10b["mode"]         = "quantise"
        _C10b["b_per_layer"]  = b_per_layer
        _C10b["r_per_layer"]  = r_per_layer
        _C10b["record"]       = True
    elif eval_mode == MODE_CLIP_ONLY:
        _MixerClass.slow_forward = _c10b_slow_forward_patch
        _C10b["mode"]         = "clip_only"
        _C10b["b_per_layer"]  = None
        _C10b["r_per_layer"]  = r_per_layer
        _C10b["record"]       = True
    else:
        raise ValueError(f"Unknown eval_mode: {eval_mode}")


def trajectory_long_ctx(model, ids, eval_mode,
                          b_per_layer=None, r_per_layer=None,
                          ctx_len=8192, n_windows=2):
    stride = ctx_len // 2
    _set_eval_mode(eval_mode, b_per_layer, r_per_layer)
    _C10b["trajectories"] = {}
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
        _MixerClass.slow_forward = _c10b_slow_forward_patch
        _C10b["mode"]        = "off"
        _C10b["b_per_layer"] = None
        _C10b["r_per_layer"] = None
        _C10b["record"]      = False
    ppl = math.exp(nll_total / tok_total)
    return ppl, windows_done, _C10b["trajectories"]


# =============================================================
# === CELL V7 === Run sweep
# =============================================================

print(f"\n{'-' * 80}")
print(f"Phase C-10b: pre-clip trajectory sweep "
      f"(n_windows={N_WINDOWS})")
print(f"  Conditions: FP16_orig (no traj) / FP32_patched_off / "
      f"FP16_clip_only / FP16_full b_0={B_0}")
print(f"  Contexts: {CTX_LENS}")
print(f"  Layers recorded: {TRAJECTORY_LAYERS}")
print(f"{'-' * 80}")

CONDITIONS = [
    ("FP16_orig",         MODE_ORIG_FORWARD, None,             None),
    ("FP32_patched_off",  MODE_OFF_RECORD,   None,             r_per_layer),
    ("FP16_clip_only",    MODE_CLIP_ONLY,    None,             r_per_layer),
    ("FP16_full",         MODE_QUANTISE,     b_per_layer_full, r_per_layer),
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
    "phase":           "C-10b",
    "purpose":         ("Phase C-10 fix: trajectory recording moved "
                        "to BEFORE the clip/quant step, so cliprate "
                        "measures the natural rate at which state "
                        "magnitudes exceed the calibrated r_g. Adds "
                        "FP32_patched_off condition (patched forward "
                        "with no intervention) for direct unmodified-"
                        "recurrence trajectory measurement."),
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
            "cliprate_pre (fraction of (B, c, s) units exceeding r_b)",
            "cliprate_unit_pre (fraction of (c, s) units exceeded "
            "by at least one B element)",
        ],
        "fix_vs_C-10":   ("Recording moved to BEFORE the clip / "
                          "quant step in the patched forward. "
                          "cliprate is now pre-clip and measures "
                          "the natural firing rate. New condition "
                          "FP32_patched_off (mode='off_record') "
                          "added: patched forward with no clip/quant "
                          "and recording active, equivalent to "
                          "FP16_orig per the C-9 H0 sanity check."),
    },
    "sweep_results":   traj_export,
    "rho_gamma_rms_summary": calib_summary,
    "notes": (
        "FP32_patched_off PPL is expected to closely match "
        "FP16_orig PPL per the C-9 H0 sanity check (the patched "
        "function in mode 'off' is bit-for-bit identical to the "
        "upstream slow_forward). It serves here as a "
        "trajectory-recording reference for the unmodified "
        "recurrence."
    ),
}

out_path = "/content/phase_c10b_state_norm_pre_clip.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False, default=str)
print(f"\nSaved -> {out_path}")


# =============================================================
# === READING ===
# =============================================================
#
# The pre-registered hypotheses translate to specific signals
# in the trajectory data:
#
# H1 (clip firing rate increases with ctx):
#   Compare cliprate_pre at end of 8K vs end of 32K under
#   FP16_clip_only. Long-memory layers (32, 40) should show
#   the largest growth.
#
# H2 (FP32_patched_off natural state magnitude growth):
#   Under FP32_patched_off, h_abs_max trajectory should grow
#   monotonically over t at long ctx; cliprate_pre should
#   become non-zero on long-memory layers.
#
# H3 (clip and off agree at t = 0):
#   Compare cliprate_pre at t = 0 between FP32_patched_off
#   and FP16_clip_only. Should be approximately equal.
#
# Combined with C-10's results:
#   - h_abs_p99 grows 2-3x at long ctx (already established)
#   - rho_p99 climbs to 0.99+ mid-window (already established)
#   - dt_p50 drifts smaller on long-memory layers (already
#     established)
#
# C-10b adds the missing piece: directly measured cliprate
# at the natural pre-clip state.
# ============================================================
