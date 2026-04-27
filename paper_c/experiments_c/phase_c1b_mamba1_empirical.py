# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Phase C-1b: Mamba-1 empirical spectral track
# ============================================================
# Paper C follow-up. NOT for the NSN paper.
#
# Phase C-1 obtained the static spectral baseline on
# state-spaces/mamba-2.8b-hf (5.24M state units, 90 % short-
# memory, 12 units in the > 0.99 long-memory tail). The empirical
# track skipped because Mamba-1 mixers do not expose the Mamba-2
# `in_proj` / `n_heads` layout that the C-1 script was written
# against.
#
# Phase C-1b adds Mamba-1 specific empirical capture. The Mamba-1
# selective-SSM forward path is:
#
#   z, x_pre = in_proj(hidden_states).chunk(2, dim=-1)        # split
#   x_pre   = silu(conv1d(x_pre))                              # [B, T, D]
#   xbc     = x_proj(x_pre)                                    # [B, T, dt_rank + 2*d_state]
#   dt_raw, B, C = xbc.split([dt_rank, d_state, d_state], -1)
#   dt      = softplus(dt_proj(dt_raw))                        # [B, T, d_inner]
#   A       = -exp(A_log)                                      # [d_inner, d_state]
#   A_bar_t,c,s = exp( dt_t,c * A_c,s )                        # per (token, channel, state)
#   ...
#
# To compute the empirical rho_g distribution we only need
# A_bar_t,c,s averaged over t. Doing this directly would store a
# [B, T, 5120, 16] tensor per layer (~330 MB for T=1024, B=1, fp32),
# which is unnecessarily large. We instead accumulate the per-
# (channel, state) running mean of A_bar over the time axis as
# we compute it, materialising only [5120, 16] = 81 920 floats per
# layer. With 64 layers and float32 that is ~21 MB, fine.
#
# Empirical statistics produced per layer:
#
#   rho_emp[c, s]    = mean over (B, T) of A_bar_{t,c,s}
#   dt_p01, dt_mean, dt_p99 (per-token dt distribution diagnostic)
#
# These complement the static rho_static and let us observe
# how much input-dependent dt selectivity actually matters in
# practice. If empirical rho is much closer to 1 than static rho,
# the selectivity mechanism is biasing toward longer memory under
# real input; if it is much smaller, selectivity is biasing
# toward forgetting. Either way it informs how the Paper C bit
# allocator should weight static-vs-empirical estimates.
#
# Runtime ~5-10 min on A100 80 GB.
# ============================================================


# =============================================================
# === CELL V1 === Setup
# =============================================================

import gc
import math
import json
import torch
from datetime import datetime

try:
    del mdl, tok
except NameError:
    pass
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

!pip install -q -U transformers accelerate hf_transfer datasets

# We do NOT install mamba-ssm / causal-conv1d. The reference
# Python path runs the same math, just slower; mathematically the
# spectral statistics are identical.

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
# === CELL V2 === Load Mamba-1 + introspect
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
    except Exception as e:
        print(f"  tokenizer load failed for {tid!r}: {type(e).__name__}")
if tok is None:
    raise RuntimeError("Could not load any tokenizer; aborting.")

mdl = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
mdl.eval()
print(f"  VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")


def find_mamba1_mixers(model):
    """Walk named_modules; match Mamba-1 mixers by the joint
    presence of A_log + dt_proj + x_proj. This excludes Mamba-2
    mixers (which do not have x_proj as a separate Linear)."""
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


mixers = find_mamba1_mixers(mdl)
print(f"\n  Found {len(mixers)} Mamba-1 mixers")
if not mixers:
    raise RuntimeError(
        "No Mamba-1 mixers detected (need A_log + dt_proj + x_proj). "
        "If the model is Mamba-2 or hybrid, use a different empirical "
        "script."
    )

name0, mix0 = mixers[0]
print(f"\n  First mixer: {name0}")
print(f"    class:           {type(mix0).__name__}")
print(f"    A_log:           shape={tuple(mix0.A_log.shape)}")
print(f"    dt_proj.weight:  shape={tuple(mix0.dt_proj.weight.shape)}")
print(f"    x_proj.weight:   shape={tuple(mix0.x_proj.weight.shape)}")
print(f"    in_proj.weight:  shape={tuple(mix0.in_proj.weight.shape)}")
print(f"    conv1d.weight:   shape={tuple(mix0.conv1d.weight.shape)}")
print(f"    time_step_rank:  {getattr(mix0, 'time_step_rank', None)}")
print(f"    ssm_state_size:  {getattr(mix0, 'ssm_state_size', None)}")
print(f"    intermediate_size: {getattr(mix0, 'intermediate_size', None)}")


# =============================================================
# === CELL V3 === Forward hook to capture in_proj input
# =============================================================
#
# Mamba-1 forward (HF reference):
#   y = in_proj(hidden_states)        # [B, T, 2 * d_inner]
#   z, x_pre = y.chunk(2, dim=-1)     # each [B, T, d_inner]
#   x_pre = silu(conv1d(x_pre.transpose(1,2)).transpose(1,2)[:T])
#   xbc   = x_proj(x_pre)             # [B, T, dt_rank + 2*d_state]
#   dt_raw, B, C = xbc.split([dt_rank, d_state, d_state], dim=-1)
#   dt    = softplus(dt_proj(dt_raw)) # [B, T, d_inner]
#   A     = -exp(A_log)               # [d_inner, d_state]
#   A_bar = exp(dt[..., None] * A[None, None])  # [B, T, d_inner, d_state]
#
# We capture hidden_states via a forward pre-hook on the mixer
# itself, then re-run only the in_proj -> conv1d -> x_proj ->
# dt_proj subset to recover dt and A_bar. We avoid constructing
# the full [B, T, d_inner, d_state] tensor by accumulating the
# per-(channel, state) running mean of A_bar.
# =============================================================

captured_inputs = {}

def make_pre_hook(layer_name):
    def pre_hook(module, args, kwargs):
        if args:
            x = args[0]
        elif "hidden_states" in kwargs:
            x = kwargs["hidden_states"]
        else:
            x = None
        if x is not None and torch.is_tensor(x):
            captured_inputs[layer_name] = x.detach()
    return pre_hook

handles = []
for name, mix in mixers:
    h = mix.register_forward_pre_hook(make_pre_hook(name),
                                       with_kwargs=True)
    handles.append(h)


# =============================================================
# === CELL V4 === Run forward on a 1024-token window
# =============================================================

ds_test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
text   = "\n\n".join([x for x in ds_test["text"] if x.strip()])[:8000]
ids    = tok.encode(text, return_tensors="pt").to("cuda")
ids    = ids[:, :1024]
print(f"\n  Capture window: {ids.shape[1]} tokens")

print("  running forward ...", flush=True)
with torch.inference_mode():
    _ = mdl(ids, use_cache=False)
print(f"  captured inputs for {len(captured_inputs)} mixers")

for h in handles:
    h.remove()


# =============================================================
# === CELL V5 === Per-mixer empirical rho computation
# =============================================================

DT_TYPICAL_FOR_STATIC = 0.1
empirical_per_layer = []

print(f"\n{'-' * 80}")
print("Per-mixer empirical computation")
print(f"{'-' * 80}")

for layer_idx, (name, mix) in enumerate(mixers):
    if name not in captured_inputs:
        empirical_per_layer.append({
            "layer_name": name,
            "status":     "skipped (no captured input)",
        })
        continue

    x = captured_inputs[name]  # [B, T, hidden_size]

    try:
        # Read mixer's static parameters
        d_inner    = mix.intermediate_size
        d_state    = mix.ssm_state_size
        dt_rank    = mix.time_step_rank

        in_proj    = mix.in_proj
        x_proj     = mix.x_proj
        dt_proj    = mix.dt_proj
        conv1d     = mix.conv1d
        A_log      = mix.A_log.detach().float()  # [d_inner, d_state]
        A_neg      = -torch.exp(A_log)            # [d_inner, d_state]

        # Re-run the dt path. Use float32 throughout for numerical
        # stability of the spectral averages; we will cast inputs
        # to in_proj.weight.dtype where required.
        with torch.inference_mode():
            xT, T  = x.shape[1], x.shape[1]
            B      = x.shape[0]

            # in_proj
            y = in_proj(x.to(in_proj.weight.dtype))   # [B, T, 2*d_inner]
            z, x_pre = y.chunk(2, dim=-1)             # each [B, T, d_inner]

            # conv1d (causal): conv1d expects [B, d_inner, T]
            x_pre_conv = x_pre.transpose(1, 2).contiguous()
            # The HF Mamba1 implementation pads on the left with
            # (kernel-1) zeros, then trims to T at the end.
            ksize = conv1d.weight.shape[-1]
            x_pre_conv = torch.nn.functional.pad(
                x_pre_conv, (ksize - 1, 0)
            )
            x_pre_conv = conv1d(x_pre_conv)[:, :, :T]
            x_pre = torch.nn.functional.silu(x_pre_conv).transpose(1, 2)
            # x_pre now [B, T, d_inner]

            # x_proj -> dt_raw / B / C
            xbc = x_proj(x_pre.to(x_proj.weight.dtype))
            # xbc shape [B, T, dt_rank + 2 * d_state]
            dt_raw, _B, _C = xbc.split(
                [dt_rank, d_state, d_state], dim=-1
            )

            # dt_proj
            dt = dt_proj(dt_raw.to(dt_proj.weight.dtype))
            # softplus(dt). HF Mamba1 uses softplus with a small
            # offset for numerical stability; we use the plain
            # softplus, which differs by an O(epsilon) constant.
            dt = torch.nn.functional.softplus(dt.float())
            # dt shape [B, T, d_inner]

            # Empirical dt distribution
            dt_flat = dt.flatten()
            dt_p01    = float(dt_flat.quantile(0.01))
            dt_p50    = float(dt_flat.quantile(0.50))
            dt_p99    = float(dt_flat.quantile(0.99))
            dt_mean   = float(dt_flat.mean())

            # Streaming accumulation of mean A_bar to avoid
            # materialising [B, T, d_inner, d_state]:
            #   for each chunk of tokens, compute A_bar slice and
            #   add it to a running [d_inner, d_state] sum, then
            #   divide by total count.
            # In practice [B*T, d_inner, d_state] for B=1 T=1024
            # and d_inner=5120 d_state=16 is 80 MB in fp32, which
            # easily fits on A100, so we just compute it directly.
            # We do it in two halves to be safe on smaller GPUs.

            # Compute A_bar in halves over the time axis
            half = T // 2
            sum_A_bar = torch.zeros((d_inner, d_state),
                                     dtype=torch.float32,
                                     device=dt.device)
            count = 0
            for t_lo in (0, half):
                t_hi = min(t_lo + half, T)
                if t_hi <= t_lo:
                    continue
                dt_chunk = dt[:, t_lo:t_hi, :]
                # A_bar_{t,c,s} = exp(dt_t,c * A_neg[c, s])
                A_bar_chunk = torch.exp(
                    dt_chunk[..., None] *
                    A_neg.unsqueeze(0).unsqueeze(0)
                )  # [B, t_hi-t_lo, d_inner, d_state]
                sum_A_bar = sum_A_bar + A_bar_chunk.sum(dim=(0, 1))
                count    = count + B * (t_hi - t_lo)

            rho_emp = (sum_A_bar / count).cpu()  # [d_inner, d_state]

        rho_emp_flat = rho_emp.flatten()

        empirical_per_layer.append({
            "layer_name":         name,
            "status":             "ok",
            "A_log_shape":        list(A_log.shape),
            "n_units":            int(rho_emp_flat.numel()),
            "dt_p01":             dt_p01,
            "dt_p50":             dt_p50,
            "dt_p99":             dt_p99,
            "dt_mean":            dt_mean,
            "rho_emp_min":        float(rho_emp.min()),
            "rho_emp_p01":        float(rho_emp_flat.quantile(0.01)),
            "rho_emp_p50":        float(rho_emp_flat.quantile(0.50)),
            "rho_emp_p99":        float(rho_emp_flat.quantile(0.99)),
            "rho_emp_max":        float(rho_emp.max()),
            "rho_emp_mean":       float(rho_emp.mean()),
            # Bucket counts so we can get a per-layer histogram
            # without storing 80K floats per layer.
            "rho_emp_bucket_counts": [
                int(((rho_emp_flat >= 0.0) & (rho_emp_flat < 0.5)).sum()),
                int(((rho_emp_flat >= 0.5) & (rho_emp_flat < 0.8)).sum()),
                int(((rho_emp_flat >= 0.8) & (rho_emp_flat < 0.95)).sum()),
                int(((rho_emp_flat >= 0.95) & (rho_emp_flat < 0.99)).sum()),
                int(((rho_emp_flat >= 0.99) & (rho_emp_flat < 0.999)).sum()),
                int((rho_emp_flat >= 0.999).sum()),
            ],
        })

        if layer_idx % 8 == 0:
            print(f"  layer {layer_idx:>2d} ({name}): "
                  f"rho_emp p50={empirical_per_layer[-1]['rho_emp_p50']:.4f} "
                  f"max={empirical_per_layer[-1]['rho_emp_max']:.4f} "
                  f"dt_mean={empirical_per_layer[-1]['dt_mean']:.4f}")

    except Exception as e:
        empirical_per_layer.append({
            "layer_name": name,
            "status":     f"error: {type(e).__name__}: {e}",
        })


# =============================================================
# === CELL V6 === Aggregate readout
# =============================================================

ok_layers = [d for d in empirical_per_layer if d.get("status") == "ok"]
print(f"\n{'=' * 80}")
print(f"Mamba-1 empirical spectral readout on {MODEL_LABEL}")
print(f"{'=' * 80}")
print(f"  Empirical track succeeded on {len(ok_layers)} / "
      f"{len(empirical_per_layer)} mixers")

if ok_layers:
    # Aggregate bucket counts across layers
    bucket_labels = [
        "[0.000, 0.500)",
        "[0.500, 0.800)",
        "[0.800, 0.950)",
        "[0.950, 0.990)",
        "[0.990, 0.999)",
        "[0.999, 1.000]",
    ]
    agg = [0] * 6
    for d in ok_layers:
        for i, c in enumerate(d["rho_emp_bucket_counts"]):
            agg[i] += c
    total = sum(agg)

    print(f"\n  Aggregated empirical rho histogram across all "
          f"(layer, channel, state) units:")
    print(f"  Total units: {total:,}")
    for label, n in zip(bucket_labels, agg):
        bar = "#" * int(40 * n / max(1, total))
        print(f"    {label}  n={n:>9,}  {bar}")

    # Per-layer dt summary
    print(f"\n  dt distribution (selectivity diagnostic):")
    print(f"    {'layer':<55s}  {'dt_p01':>8s}  {'dt_p50':>8s}  "
          f"{'dt_p99':>8s}")
    for d in ok_layers[::8]:  # every 8th layer
        print(f"    {d['layer_name']:<55s}  "
              f"{d['dt_p01']:>8.4f}  "
              f"{d['dt_p50']:>8.4f}  "
              f"{d['dt_p99']:>8.4f}")


# =============================================================
# === CELL V7 === Save (summary only; raw per-unit lists were
# === already too large for git in C-1, and we do not regenerate
# === them here -- the layer summary + bucket counts are enough
# === for downstream analysis.)
# =============================================================

output = {
    "date":                datetime.utcnow().isoformat(timespec="seconds"),
    "phase":               "C-1b",
    "purpose":             ("Mamba-1 empirical spectral track. "
                            "Captures hidden_states per mixer, "
                            "replays in_proj -> conv1d -> x_proj -> "
                            "dt_proj on float32, computes "
                            "A_bar = exp(dt * (-exp(A_log))) and "
                            "averages over time to get rho_emp."),
    "model":               MODEL_ID,
    "label":               MODEL_LABEL,
    "n_mamba1_mixers":     len(mixers),
    "static_dt_used":      DT_TYPICAL_FOR_STATIC,
    "n_empirical_ok":      len(ok_layers),
    "empirical_per_layer": empirical_per_layer,
}

out_path = "/content/phase_c1b_mamba1_empirical.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\nSaved -> {out_path}")


# =============================================================
# === READING ===
# =============================================================
#
# Compare to Phase C-1's static rho distribution:
#
#   static, dt=0.1:
#     [0.000, 0.500): 58.0 %
#     [0.500, 0.800): 31.3 %
#     [0.800, 0.950): 10.4 %
#     [0.950, 0.990):  0.42 %
#     [0.990, 0.999):  2e-6 (12 units)
#     [0.999, 1.000]:  0
#
# Three readout patterns:
#
#   A. Empirical histogram is essentially identical to static.
#      The selectivity mechanism's input-dependent dt does not
#      systematically shift the forgetting distribution; static
#      rho is a sufficient proxy for the bit allocator. Cheap.
#
#   B. Empirical histogram is shifted toward LONGER memory
#      (heavier tail above 0.95). The selective dt typically
#      picks small dt values, so empirical A_bar = exp(-dt*|A|)
#      is closer to 1. Implication: Paper C should weight
#      empirical rho more heavily, and the long-memory minority
#      that needs high bit precision is LARGER than static
#      analysis suggests.
#
#   C. Empirical histogram is shifted toward SHORTER memory.
#      The selective dt typically picks larger dt values,
#      forgetting more aggressively in practice. Long-memory
#      tail is even smaller than static suggests. Bit allocator
#      can be more aggressive on the bulk of units.
#
# The dt distribution per layer (p01 / p50 / p99) tells us the
# range of selectivity in actual use. If dt_p99 / dt_p01 is large,
# the model is exercising selectivity strongly per layer; if
# small, selectivity is essentially constant and Mamba-1 is
# behaving like a non-selective SSM.
#
# Future Phase C-1c: extract C from x_proj's last d_state slice
# and compute Gamma_g properly, completing the R_g formula.
# ============================================================
