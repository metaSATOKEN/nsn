# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Phase C-1: SSM spectral baseline diagnostic on Mamba-1 (HF)
# ============================================================
# Paper C kickoff probe. NOT for the NSN paper (Paper B).
#
# Goal. Compute, per (layer, state-unit) of an SSM-bearing model,
# the three quantities that the Paper C bit allocator needs:
#
#   rho_g    = E_t || A_bar_t,g ||_2          (forgetting / memory)
#   Gamma_g  = E_t [ C_t,g^T C_t,g ]          (output coupling)
#   R_g      = || Gamma_g || / (1 - rho_g + eps)
#   b_g      = b_0 + ceil( log_2 R_g )+       (suggested bit budget)
#
# The intuition: long-memory units (rho_g -> 1) and high-output-
# coupling units (large || Gamma_g ||) get more bits; aggressive-
# forgetting low-coupling units can be aggressively compressed.
# This is the per-state-group analog of Paper B's per-channel
# asymmetric quantisation, generalised to recurrent state.
#
# Model. We default to state-spaces/mamba-2.8b-hf, the HF-native
# port of the original Mamba (Mamba-1) at 2.8B parameters. We
# arrived here after two infrastructure pivots:
#
#   1. state-spaces/mamba2-2.7b (Mamba-2): config.json has no
#      `model_type` key, and the corresponding mamba-ssm /
#      causal-conv1d CUDA kernels reliably fail to build on Colab.
#
#   2. Zyphra/Zamba2-2.7B (hybrid Mamba2 + attention): HF-native
#      config, but the current transformers `get_expanded_tied_
#      weights_keys` machinery is not compatible with Zamba2's
#      shared-transformer weight tying (^layers.12 ↔ ^layers.47
#      adapter-list mismatch). Triggers a ValueError at model
#      construction time even before forward.
#
# Mamba-1 HF is rock solid: HF-native config, the reference
# Python forward path runs without any CUDA kernel package, and
# the spectral concepts (rho_g, R_g, b_g) transfer identically
# from Mamba-2 to Mamba-1 — only the A_log shape changes from
# 1D [n_heads] to 2D [d_inner, d_state]. The Paper C bit
# allocator math is shape-agnostic, so nothing is lost.
#
# Mixer detection is by A_log presence (the canonical Mamba A
# parameter), so this same script also runs on:
#   - state-spaces/mamba-130m-hf, -370m-hf, -790m-hf (smaller)
#   - state-spaces/mamba-1.4b-hf, -2.8b-hf (default)
#   - Any future HF-native Mamba / Mamba-2 / hybrid checkpoint
#     (the empirical track may need attribute aliases tweaked).
#
# Two-track diagnostic so the script always produces SOMETHING:
#
#   Static track: read A_log directly from each SSM mixer's
#     parameters; report per-unit static "forgetting tendency"
#     rho_static_g(dt) = exp( - dt * exp(A_log_g) ) at a typical
#     dt = 0.1. Always works. For Mamba-1 the A_log tensor is
#     2D [d_inner, d_state] and we flatten over both axes for
#     the distribution.
#
#   Empirical track: register a forward pre-hook on each mixer,
#     capture the input hidden_states of one 1024-token WikiText-2
#     window, replay the dt computation manually using the mixer's
#     own parameters, and average over time. Implemented for the
#     Mamba-2 attribute layout; on Mamba-1 it currently skips
#     (Mamba-1 mixers expose dt via dt_proj rather than the
#     Mamba-2 in_proj-last-n_heads slice; a Phase C-1b will add
#     Mamba-1 empirical support).
#
# Runtime ~5-10 min on A100 80 GB (download dominates; the
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
MODEL_ID    = "state-spaces/mamba-2.8b-hf"
MODEL_LABEL = "Mamba-2.8B-hf"
# Alternatives that should also work with the same script:
#   - "state-spaces/mamba-130m-hf"    (250 MB, fast smoke test)
#   - "state-spaces/mamba-370m-hf"
#   - "state-spaces/mamba-790m-hf"
#   - "state-spaces/mamba-1.4b-hf"
# ─────────────────────────────────────────────────────────────

# state-spaces' HF Mamba-1 ports reuse the GPT-NeoX-20B tokenizer.
# Try the model id first, fall back to known compatible alternatives.
TOKENIZER_FALLBACKS = [
    MODEL_ID,
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

    # Probe a wide set of canonical attribute names (covers
    # Mamba-1, Mamba-2, and common downstream renamings). This is
    # both our diagnostic for whether the empirical track will
    # work and our guide for what to alias in Phase C-1b.
    for attr in [
        # scalars / small params (Mamba-2 + Mamba-1 + variants)
        "dt_bias", "D", "n_heads", "num_heads", "n_groups",
        "num_groups", "head_dim", "ssm_state_size", "d_state",
        "expand", "conv_kernel", "chunk_size",
        # Mamba-1 specific
        "intermediate_size", "time_step_rank", "dt_rank",
        "use_bias", "use_conv_bias",
        # projection layers; Mamba-1, Mamba-2, mamba-ssm,
        # Zamba2 all name them differently
        "in_proj", "xb_proj", "x_proj", "Wxbcdt", "dt_proj",
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
    # so we can spot architecture-specific naming we did not
    # anticipate (e.g., Zamba2-specific renamings).
    expected = {
        "A_log", "dt_bias", "D", "n_heads", "num_heads", "n_groups",
        "num_groups", "head_dim", "ssm_state_size", "d_state",
        "expand", "conv_kernel", "chunk_size", "in_proj", "xb_proj",
        "x_proj", "Wxbcdt", "out_proj", "norm", "act", "conv1d",
        "intermediate_size", "time_step_rank", "dt_rank",
        "use_bias", "use_conv_bias", "dt_proj",
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
all_rho_static = []   # flat list across (layer, every A_log entry)

for name, mix in ssm_mixers:
    # A_log shape:
    #   Mamba-1: [d_inner, d_state]   (e.g. [5120, 16])
    #   Mamba-2: [n_heads]            (e.g. [80])
    #   Both:   any other layout where every entry is one
    #           independent forgetting parameter
    A_log = mix.A_log.detach().float().cpu()
    A_mag = torch.exp(A_log)
    rho_static = torch.exp(-DT_TYPICAL * A_mag)

    flat_rho = rho_static.flatten()
    n_units  = int(flat_rho.numel())

    info = {
        "layer_name":           name,
        "A_log_shape":          list(A_log.shape),
        "n_units":              n_units,
        "A_log_min":            float(A_log.min()),
        "A_log_max":            float(A_log.max()),
        "A_log_mean":           float(A_log.mean()),
        "A_mag_mean":           float(A_mag.mean()),
        "rho_static_min":       float(rho_static.min()),
        "rho_static_max":       float(rho_static.max()),
        "rho_static_mean":      float(rho_static.mean()),
        # Store flat (per-unit) lists so the JSON is shape-agnostic;
        # consumers can reshape using A_log_shape if needed.
        "A_log_flat":           A_log.flatten().tolist(),
        "rho_static_flat":      flat_rho.tolist(),
    }
    static_per_layer.append(info)
    all_rho_static.extend(flat_rho.tolist())

# Aggregate readout
all_rho_static_t = torch.tensor(all_rho_static)
print(f"  Total state units across model: {len(all_rho_static)}")
print(f"  rho_static  (dt={DT_TYPICAL}):")
print(f"    min={all_rho_static_t.min().item():.4f}  "
      f"median={all_rho_static_t.median().item():.4f}  "
      f"max={all_rho_static_t.max().item():.4f}")

# Coarse histogram of rho_static across all state units
buckets_rho = [0.0, 0.50, 0.80, 0.95, 0.99, 0.999, 1.0]
print(f"  Histogram of rho_static across state units:")
for lo, hi in zip(buckets_rho[:-1], buckets_rho[1:]):
    n = ((all_rho_static_t >= lo) & (all_rho_static_t < hi)).sum().item()
    if hi == 1.0:
        n += (all_rho_static_t == 1.0).sum().item()
    bar = "#" * int(40 * n / max(1, len(all_rho_static)))
    print(f"    [{lo:.3f}, {hi:.3f})  n={n:>6d}  {bar}")


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
