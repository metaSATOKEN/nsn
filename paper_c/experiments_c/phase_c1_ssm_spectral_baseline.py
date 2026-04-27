# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Phase C-1: SSM spectral baseline diagnostic on Zamba2-2.7B
# ============================================================
# Paper C kickoff probe. NOT for the NSN paper (Paper B).
#
# Goal. Compute, per (layer, head/group) of an SSM-bearing model,
# the three quantities that the Paper C bit allocator needs:
#
#   rho_g    = E_t || A_bar_t,g ||_2          (forgetting / memory)
#   Gamma_g  = E_t [ C_t,g^T C_t,g ]          (output coupling)
#   R_g      = || Gamma_g || / (1 - rho_g + eps)
#   b_g      = b_0 + ceil( log_2 R_g )+       (suggested bit budget)
#
# The intuition: long-memory groups (rho_g -> 1) and high-output-
# coupling groups (large || Gamma_g ||) get more bits; aggressive-
# forgetting low-coupling groups can be aggressively compressed.
# This is the per-state-group analog of Paper B's per-channel
# asymmetric quantisation, generalised to recurrent state.
#
# Model. We default to Zyphra/Zamba2-2.7B, an HF-native hybrid
# Mamba2 + softmax-attention model. Initial attempt with
# state-spaces/mamba2-2.7b failed because state-spaces ships
# checkpoints in the original Mamba authors' format (no
# `model_type` key in config.json), and the corresponding
# mamba-ssm / causal-conv1d kernels do not build on Colab.
# Zamba2 ships HF-native configs and uses HF transformers'
# Mamba2Mixer reference implementation by default, which works
# without those CUDA kernels.
#
# Bonus: Zamba2-2.7B is itself a hybrid model, which gives us a
# *second* hybrid datapoint beyond the Phase 29 Qwen3.6-27B
# experiments — directly relevant to the Paper-C-4 plan
# (synchronisation-pulse hypothesis on a second hybrid).
#
# Mixer detection is by A_log presence (the canonical Mamba A
# parameter), so this same script also runs on:
#   - state-spaces/mamba-2.8b-hf (Mamba-1 HF-native, fallback)
#   - any HF-native Mamba / Mamba-2 / hybrid checkpoint
#
# Two-track diagnostic so the script always produces SOMETHING:
#
#   Static track: read A_log directly from each SSM mixer's
#     parameters; report per-head static "forgetting tendency"
#     rho_static_g(dt) = exp( - dt * exp(A_log_g) ) at a typical
#     dt = 0.1. Always works.
#
#   Empirical track: register a forward pre-hook on each mixer,
#     capture the input hidden_states of one 1024-token WikiText-2
#     window, replay the dt computation manually using the mixer's
#     own parameters and standard Mamba-2 formulae, and average
#     over time. Best effort; if the module structure is not what
#     we expect, the empirical track is skipped and only the
#     static track is reported.
#
# Runtime ~10-15 min on A100 80 GB (download dominates; the
# spectral pass itself is one forward).
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

# Note: NOT installing mamba-ssm / causal-conv1d here. Those
# CUDA kernel packages reliably fail to build on Colab and are
# only needed when loading state-spaces/mamba* checkpoints in
# the original (non-HF) format. Zamba2-2.7B uses HF
# transformers' built-in Mamba2Mixer reference path and runs
# without them.

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
# === CELL V2 === Load Mamba-2 + introspect module structure
# =============================================================

# ─────────────────────────────────────────────────────────────
# >>>>>  Adjust if you want a smaller / faster probe  <<<<<
MODEL_ID    = "Zyphra/Zamba2-2.7B"
MODEL_LABEL = "Zamba2-2.7B"
# Alternatives that should also work with the same script:
#   - "Zyphra/Zamba2-1.2B"           (smaller, faster iteration)
#   - "state-spaces/mamba-2.8b-hf"   (Mamba-1 HF-native, static-only)
# ─────────────────────────────────────────────────────────────

# Zamba2 ships its own tokenizer; Mamba HF mirrors typically use
# GPT-NeoX-20B's tokenizer as a fallback. Try the model id first,
# fall back to known compatible alternatives.
TOKENIZER_FALLBACKS = [
    MODEL_ID,
    "Zyphra/Zamba2-2.7B",
    "EleutherAI/gpt-neox-20b",
]

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


def find_ssm_mixers(model):
    """
    Walk the model's named_modules and return a list of
    (qualified_name, module) pairs for everything that looks like
    a Mamba-2 SSM mixer.

    The HF / state-spaces implementations name these slightly
    differently across versions; we match by attribute presence
    rather than by class name to be version-tolerant.

    A module is considered an SSM mixer iff it has an A_log
    parameter (the canonical Mamba A_log).
    """
    found = []
    for name, mod in model.named_modules():
        if hasattr(mod, "A_log") and isinstance(
            getattr(mod, "A_log", None), torch.nn.Parameter
        ):
            found.append((name, mod))
    return found


ssm_mixers = find_ssm_mixers(mdl)
print(f"\n  Found {len(ssm_mixers)} SSM mixers (matched on A_log "
      f"parameter)")
if ssm_mixers:
    name0, mix0 = ssm_mixers[0]
    print(f"\n  First mixer: {name0}")
    print(f"    class:    {type(mix0).__name__}")
    print(f"    A_log:    shape={tuple(mix0.A_log.shape)}  "
          f"dtype={mix0.A_log.dtype}")

    # Probe a wide set of canonical Mamba-2 attribute names; this
    # is also our diagnostic for whether the empirical track will
    # work, and for spotting Zamba2's renamings if any.
    for attr in [
        "dt_bias", "D", "n_heads", "num_heads", "n_groups",
        "num_groups", "head_dim", "ssm_state_size", "d_state",
        "expand", "conv_kernel", "chunk_size",
        # projection layers; HF / mamba-ssm name them differently
        "in_proj", "xb_proj", "x_proj", "Wxbcdt",
        "out_proj", "norm", "act", "conv1d",
    ]:
        if hasattr(mix0, attr):
            v = getattr(mix0, attr)
            if isinstance(v, (torch.nn.Parameter, torch.Tensor)):
                print(f"    {attr}: tensor shape={tuple(v.shape)}  "
                      f"dtype={v.dtype}")
            elif hasattr(v, "weight"):
                w = v.weight
                print(f"    {attr}: {type(v).__name__} "
                      f"weight shape={tuple(w.shape)}")
            else:
                print(f"    {attr}: {v}")

    # Also dump non-private attribute names that we did not list,
    # so we can spot Zamba2-specific naming we did not anticipate.
    expected = {
        "A_log", "dt_bias", "D", "n_heads", "num_heads", "n_groups",
        "num_groups", "head_dim", "ssm_state_size", "d_state",
        "expand", "conv_kernel", "chunk_size", "in_proj", "xb_proj",
        "x_proj", "Wxbcdt", "out_proj", "norm", "act", "conv1d",
    }
    other = sorted([
        n for n in dir(mix0)
        if not n.startswith("_")
        and n not in expected
        and not callable(getattr(mix0, n, None))
    ])
    if other:
        print(f"    [other non-private attrs on the mixer]: "
              f"{', '.join(other[:25])}"
              f"{' ...' if len(other) > 25 else ''}")


# =============================================================
# === CELL V3 === Static spectral analysis (always works)
# =============================================================
#
# A_log has shape [n_heads]; A = -exp(A_log) is the per-head
# forgetting scalar. With a typical dt around 0.1, the per-step
# multiplicative attenuation is rho_static = exp(-dt * |A|).
# This is a static lower-bound estimate of "memory length": close
# to 1 means slow forgetting (long memory, needs more bits in any
# state-quantising scheme), close to 0 means fast forgetting.
# =============================================================

print(f"\n{'-' * 80}")
print("Static spectral track: A_log distribution per layer / head")
print(f"{'-' * 80}")

DT_TYPICAL = 0.1  # canonical dt for Mamba-2 inference; refines in V4

static_per_layer = []
all_rho_static = []

for name, mix in ssm_mixers:
    A_log = mix.A_log.detach().float().cpu()
    A_mag = torch.exp(A_log)               # |A| per head, > 0
    rho_static = torch.exp(-DT_TYPICAL * A_mag)
    info = {
        "layer_name":          name,
        "n_heads":             int(A_log.numel()),
        "A_log_min":           float(A_log.min()),
        "A_log_max":           float(A_log.max()),
        "A_log_mean":          float(A_log.mean()),
        "A_mag_mean":          float(A_mag.mean()),
        "rho_static_min":      float(rho_static.min()),
        "rho_static_max":      float(rho_static.max()),
        "rho_static_mean":     float(rho_static.mean()),
        "A_log_per_head":      A_log.tolist(),
        "rho_static_per_head": rho_static.tolist(),
    }
    static_per_layer.append(info)
    all_rho_static.extend(rho_static.tolist())

# Aggregate readout
all_rho_static_t = torch.tensor(all_rho_static)
print(f"  Total heads across model: {len(all_rho_static)}")
print(f"  rho_static  (dt={DT_TYPICAL}):")
print(f"    min={all_rho_static_t.min().item():.4f}  "
      f"median={all_rho_static_t.median().item():.4f}  "
      f"max={all_rho_static_t.max().item():.4f}")

# Coarse histogram of rho_static across heads
buckets_rho = [0.0, 0.50, 0.80, 0.95, 0.99, 0.999, 1.0]
print(f"  Histogram of rho_static across heads:")
for lo, hi in zip(buckets_rho[:-1], buckets_rho[1:]):
    n = ((all_rho_static_t >= lo) & (all_rho_static_t < hi)).sum().item()
    if hi == 1.0:
        n += (all_rho_static_t == 1.0).sum().item()
    bar = "#" * int(40 * n / max(1, len(all_rho_static)))
    print(f"    [{lo:.3f}, {hi:.3f})  n={n:>5d}  {bar}")


# =============================================================
# === CELL V4 === Empirical track (best-effort forward hook)
# =============================================================

print(f"\n{'-' * 80}")
print("Empirical spectral track: capture dt, B, C on a "
      "1024-token forward")
print(f"{'-' * 80}")

# WikiText-2 first 8K chars as the calibration window
ds_test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
text   = "\n\n".join([x for x in ds_test["text"] if x.strip()])[:8000]
ids    = tok.encode(text, return_tensors="pt").to("cuda")
ids    = ids[:, :1024]
print(f"  Capture window: {ids.shape[1]} tokens")

# We capture each mixer's INPUT (hidden_states) via a pre-hook.
# Then post-hoc we re-derive dt by re-running just the in_proj
# slice that produces dt, the A_bar = exp(dt * A) reduction, and
# the C produced by the same in_proj slice. This avoids depending
# on the mixer's internal scratch tensors that vary by transformer
# version.
#
# We deliberately do NOT try to compute the full SSD scan -- we
# only need (per-token, per-head) dt and (per-token, per-group) C
# and B for the spectral statistics.

captured_inputs = {}

def make_pre_hook(layer_name):
    def pre_hook(module, args, kwargs):
        # The first positional argument is hidden_states for the
        # standard HF Mamba2 mixer signature. Falling back to
        # kwargs in case of API drift.
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
for name, mix in ssm_mixers:
    h = mix.register_forward_pre_hook(make_pre_hook(name),
                                       with_kwargs=True)
    handles.append(h)

# Run a single forward
print("  running forward ...", flush=True)
with torch.inference_mode():
    _ = mdl(ids, use_cache=False)
print(f"  captured inputs for {len(captured_inputs)} mixers")

for h in handles:
    h.remove()

# Now compute, per layer, per head:
#   dt_t   = softplus( in_proj_dt(x_t) + dt_bias )
#   A_bar_{t,h} = exp( dt_t,h * (-exp(A_log_h)) )
#   rho_h     = E_t [ A_bar_{t,h} ]   (scalar, in (0, 1])
#   ||C||_h   = E_t [ ||C_t,h||_2^2 ]  (scalar)
#
# Mamba-2's in_proj packs [z, x, B, C, dt] in one big linear
# projection. We try several common attribute names; if any
# fails, the empirical readout for that mixer is skipped.

empirical_per_layer = []

for name, mix in ssm_mixers:
    if name not in captured_inputs:
        continue
    x = captured_inputs[name]  # [B, T, d_model]

    # Try the canonical Mamba-2 attribute names. Different HF
    # transformers versions, the original mamba_ssm library, and
    # downstream models like Zamba2 use slightly different layouts;
    # if the layout we recognise is not present we just skip the
    # empirical calc for this mixer (the static track already
    # covers it).
    def _first_attr(module, names):
        for n in names:
            v = getattr(module, n, None)
            if v is not None:
                return v, n
        return None, None

    in_proj, in_proj_name = _first_attr(
        mix, ["in_proj", "Wxbcdt", "xb_proj", "x_proj"]
    )
    dt_bias    = getattr(mix, "dt_bias", None)
    n_heads, _ = _first_attr(mix, ["n_heads", "num_heads"])
    n_groups, _ = _first_attr(mix, ["n_groups", "num_groups"])
    head_dim   = getattr(mix, "head_dim", None)
    state_size, _ = _first_attr(
        mix, ["ssm_state_size", "d_state", "state_size"]
    )

    A_log = mix.A_log.detach().float()
    A_mag = torch.exp(A_log)  # [n_heads]

    if in_proj is None or dt_bias is None or n_heads is None:
        empirical_per_layer.append({
            "layer_name": name,
            "status":     "skipped (mixer attributes not as expected)",
        })
        continue

    try:
        with torch.inference_mode():
            # Run the in_proj on the captured input. Output layout
            # in the canonical HF Mamba2 implementation is:
            #   [z, x_conv, dt] concat, with [B, C] derived from
            # x_conv's dimensions (n_groups * state_size each).
            # We don't need [z, x_conv, B], only dt and C.
            # The HF Mamba2 mixer typically exposes split sizes via
            # mix.intermediate_size etc., but to stay robust we just
            # try to slice the last n_heads of the projection as dt.
            zxbcdt = in_proj(x.to(in_proj.weight.dtype))
            # zxbcdt shape: [B, T, intermediate]
            # dt is typically the last n_heads channels of the
            # projection.
            dt = zxbcdt[..., -n_heads:].float() + dt_bias.float()
            dt = torch.nn.functional.softplus(dt)  # [B, T, n_heads]

            # A_bar_{t,h} = exp( dt_{t,h} * (-A_mag_h) )
            A_bar = torch.exp(-dt * A_mag.unsqueeze(0).unsqueeze(0))
            # mean over batch and time
            rho_h = A_bar.mean(dim=(0, 1))  # [n_heads]

            # Empirical dt distribution for diagnostics
            dt_mean   = float(dt.mean())
            dt_p99    = float(dt.flatten().float().quantile(0.99))
            dt_p01    = float(dt.flatten().float().quantile(0.01))

        # We deliberately do NOT extract C/B here in this first
        # pass -- their slice positions vary by HF version and
        # would require model-specific code. R_g still gets a
        # rho-only proxy, with C-coupling left for Phase C-1b.
        eps = 1e-4
        R_h_proxy = 1.0 / (1.0 - rho_h.clamp(max=1.0 - eps) + eps)
        b_proxy   = (math.log2(2.0) +
                      torch.clamp(torch.ceil(torch.log2(R_h_proxy)),
                                   min=0.0))

        empirical_per_layer.append({
            "layer_name":     name,
            "status":         "ok",
            "n_heads":        int(n_heads),
            "n_groups":       (int(n_groups) if n_groups is not None
                                else None),
            "head_dim":       (int(head_dim) if head_dim is not None
                                else None),
            "state_size":     (int(state_size) if state_size is not None
                                else None),
            "dt_p01":         dt_p01,
            "dt_mean":        dt_mean,
            "dt_p99":         dt_p99,
            "rho_per_head":   rho_h.detach().cpu().tolist(),
            "rho_min":        float(rho_h.min()),
            "rho_median":     float(rho_h.median()),
            "rho_max":        float(rho_h.max()),
            "R_proxy_min":    float(R_h_proxy.min()),
            "R_proxy_max":    float(R_h_proxy.max()),
            "b_proxy_min":    float(b_proxy.min()),
            "b_proxy_max":    float(b_proxy.max()),
            "b_proxy_mean":   float(b_proxy.float().mean()),
        })
    except Exception as e:
        empirical_per_layer.append({
            "layer_name": name,
            "status":     f"error: {type(e).__name__}: {e}",
        })


# =============================================================
# === CELL V5 === Empirical readout summary
# =============================================================

ok_layers = [d for d in empirical_per_layer if d.get("status") == "ok"]
print(f"\n  Empirical track succeeded on {len(ok_layers)} / "
      f"{len(empirical_per_layer)} layers")

if ok_layers:
    all_rho = []
    all_b   = []
    for d in ok_layers:
        all_rho.extend(d["rho_per_head"])
        # b_proxy distribution is harder to flatten; we just
        # report the layer-level min/max/mean
    all_rho_t = torch.tensor(all_rho)
    print(f"\n  Empirical rho across all (layer, head):")
    print(f"    n     = {all_rho_t.numel()}")
    print(f"    min   = {all_rho_t.min().item():.4f}")
    print(f"    median= {all_rho_t.median().item():.4f}")
    print(f"    max   = {all_rho_t.max().item():.4f}")
    print(f"\n  Histogram of empirical rho across heads:")
    for lo, hi in zip(buckets_rho[:-1], buckets_rho[1:]):
        n = ((all_rho_t >= lo) & (all_rho_t < hi)).sum().item()
        if hi == 1.0:
            n += (all_rho_t == 1.0).sum().item()
        bar = "#" * int(40 * n / max(1, all_rho_t.numel()))
        print(f"    [{lo:.3f}, {hi:.3f})  n={n:>5d}  {bar}")

    print(f"\n  Per-layer b_proxy summary "
          f"(R_proxy = 1 / (1 - rho); b = log2 R, +baseline):")
    print(f"    {'layer':<55s}  {'rho':>6s}  {'b_min':>6s}  "
          f"{'b_max':>6s}  {'b_mean':>6s}")
    for d in ok_layers[:8]:
        print(f"    {d['layer_name']:<55s}  "
              f"{d['rho_median']:>6.3f}  "
              f"{d['b_proxy_min']:>6.2f}  "
              f"{d['b_proxy_max']:>6.2f}  "
              f"{d['b_proxy_mean']:>6.2f}")
    if len(ok_layers) > 8:
        print(f"    ... ({len(ok_layers) - 8} more layers)")


# =============================================================
# === CELL V6 === Save
# =============================================================

output = {
    "date":             datetime.utcnow().isoformat(timespec="seconds"),
    "phase":            "C-1",
    "purpose":          ("SSM spectral baseline: per-(layer, head) "
                         "rho_g (forgetting), and a rho-based R_g "
                         "proxy for the Paper C bit allocator. "
                         "C-coupling Gamma_g is deferred to "
                         "Phase C-1b (model-specific extraction)."),
    "model":            MODEL_ID,
    "label":            MODEL_LABEL,
    "n_ssm_mixers":     len(ssm_mixers),
    "static_dt_used":   DT_TYPICAL,
    "static_track":     static_per_layer,
    "empirical_track":  empirical_per_layer,
    "n_empirical_ok":   len(ok_layers),
}

out_path = "/content/phase_c1_ssm_spectral_baseline.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\nSaved -> {out_path}")


# =============================================================
# === READING ===
# =============================================================
#
# What this run tells us:
#
#   1. How many SSM mixers there are, and how many heads per
#      mixer (architectural baseline for any future per-state-
#      group recipe).
#
#   2. Static spectral readout: how concentrated is rho_static
#      across heads? If most heads are at rho > 0.99, the model
#      is dominated by long-memory channels and aggressive
#      uniform quantisation will hurt. If rho is broadly spread
#      across [0.5, 0.999], a non-uniform bit allocator has
#      meaningful headroom.
#
#   3. Empirical rho readout: same statistic but driven by
#      *input-dependent* dt. For Mamba-2's selective SSM, dt is
#      input-conditional, so empirical rho can differ
#      substantially from the static estimate. The gap between
#      static and empirical is itself diagnostic about how
#      "active" the selectivity mechanism is per layer.
#
#   4. b_proxy distribution: this is the rho-only proxy
#      (Gamma_g coupling deferred). The shape of the histogram
#      tells us how much non-uniform allocation is justified --
#      flat distributions favour uniform quantisation, fat-tailed
#      distributions favour mixed-precision allocators.
#
# Phase C-1b will add C and B extraction (model-specific) so we
# can compute the full Gamma_g and the proper R_g.
#
# Phase C-2 will use the rho_g distribution as the basis for the
# update-decomposition design (which heads to compress, which to
# keep at higher precision).
# ============================================================
