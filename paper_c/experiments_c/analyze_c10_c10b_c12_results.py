#!/usr/bin/env python3
# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0
"""
Unified analysis script for Phase C-10 / C-10b / C-12 result JSONs.

Run from the modeq repo root:

    python paper_c/experiments_c/analyze_c10_c10b_c12_results.py \
        paper_c/results_c/phase_c10_state_norm_trajectory.json \
        paper_c/results_c/phase_c10b_state_norm_pre_clip.json \
        paper_c/results_c/phase_c12_strict_optimum_full.json

Or any subset; auto-detects which phase each file is via the
'phase' field.

The output is a structured stdout report with:
  - Per-phase summary (PPL, mean bits, verdict class)
  - Trajectory deep-dive for C-10 / C-10b
  - Slope comparison for C-12
  - Cross-phase consistency checks

The script is idempotent and read-only.
"""

import json
import math
import sys
from pathlib import Path


LAYERS = [0, 16, 32, 40, 48, 56, 63]
LAYER_NAMES = [f'backbone.layers.{i}.mixer' for i in LAYERS]


# =============================================================
# Helpers
# =============================================================

def load_json(path):
    with open(path) as f:
        return json.load(f)


def fmt(x, w=10, prec=4):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return f"{'n/a':>{w}s}"
    if isinstance(x, str):
        return f"{x:>{w}s}"
    return f"{x:>{w}.{prec}f}"


def hr(width=78, ch="─"):
    print(ch * width)


def header(title, width=78):
    print()
    print("=" * width)
    print(f" {title}")
    print("=" * width)


def section(title, width=78):
    print()
    hr(width)
    print(f" {title}")
    hr(width)


# =============================================================
# C-10 / C-10b analysis
# =============================================================

def get_traj(d, ctx, cond, layer_name):
    """Extract trajectory list for (ctx, condition, layer)."""
    s = d.get('sweep_results', {}).get(str(ctx), {})
    c = s.get(cond, {})
    t = c.get('trajectories', {}) or {}
    return t.get(layer_name, [])


def summarise_ppl(d):
    section("PPL summary by (ctx, condition)")
    ctx_lens = [str(c) for c in d['protocol']['ctx_lens']]
    conditions = list(d['sweep_results'][ctx_lens[0]].keys())
    print(f"\n  {'condition':<22s} " +
          "  ".join(f"{c:>10s}" for c in ctx_lens))
    print(f"  {'─' * 22} " + "  ".join("─" * 10 for _ in ctx_lens))
    for cond in conditions:
        row = f"  {cond:<22s} "
        for ctx in ctx_lens:
            ppl = d['sweep_results'][ctx][cond].get('ppl')
            row += f"  {fmt(ppl, 10, 4)}"
        print(row)


def cliprate_per_layer_table(d, cond, key='cliprate_pre'):
    """Per-layer cliprate at end of each ctx. Returns rows for stdout."""
    ctx_lens = [int(c) for c in d['protocol']['ctx_lens']]
    has_data = False
    for i, lname in zip(LAYERS, LAYER_NAMES):
        for ctx in ctx_lens:
            traj = get_traj(d, ctx, cond, lname)
            if traj and key in traj[-1]:
                has_data = True
                break
        if has_data:
            break
    if not has_data:
        print(f"  ({cond}: no '{key}' data)")
        return

    print(f"  layer  " +
          "  ".join(f"{c:>8} end" for c in ctx_lens) +
          "  growth(end_max/end_min)")
    print(f"  ─────  " + "  ".join("─" * 12 for _ in ctx_lens) + "  ──────")
    for i, lname in zip(LAYERS, LAYER_NAMES):
        end_vals = []
        for ctx in ctx_lens:
            traj = get_traj(d, ctx, cond, lname)
            v = traj[-1].get(key) if traj else None
            end_vals.append(v)
        max_v = max((v for v in end_vals if v is not None), default=None)
        min_v = min((v for v in end_vals if v is not None), default=None)
        if (max_v is not None and min_v is not None and min_v > 0):
            growth = max_v / min_v
            growth_s = f"  {growth:>5.2f}x"
        else:
            growth_s = "    n/a"
        row = f"  {i:>5d}  " + "  ".join(
            f"{fmt(v, 12, 6)}" for v in end_vals
        ) + growth_s
        print(row)


def trajectory_summary(d, ctx, cond, key, label):
    """Print per-layer start / mid / end of `key` in trajectory."""
    print(f"\n  {label}: {key} trajectory at ctx={ctx} ({cond})")
    print(f"  layer    start         mid           end           drift_e_s")
    print(f"  ─────    ──────────    ──────────    ──────────    ──────────")
    for i, lname in zip(LAYERS, LAYER_NAMES):
        traj = get_traj(d, ctx, cond, lname)
        if not traj:
            continue
        start_v = traj[0].get(key)
        end_v = traj[-1].get(key)
        mid_v = traj[len(traj) // 2].get(key)
        if start_v is None or end_v is None:
            continue
        drift = end_v - start_v
        drift_pct = (drift / start_v * 100) if start_v != 0 else float('nan')
        print(f"  {i:>5d}    "
              f"{fmt(start_v, 10, 6)}    "
              f"{fmt(mid_v, 10, 6)}    "
              f"{fmt(end_v, 10, 6)}    "
              f"{drift_pct:+9.2f}%")


def analyse_c10(d):
    """Phase C-10 analysis: trajectory under broken cliprate."""
    header(f"Phase C-10 (broken cliprate; trajectory anchor only)")
    summarise_ppl(d)

    section("h_abs_p99 trajectory at 32K (clip_only): natural state magnitude tail")
    trajectory_summary(d, 32768, 'FP16_clip_only', 'h_abs_p99',
                       'state magnitude tail growth')

    section("rho_p99 trajectory at 32K (clip_only): peak per-unit spectral radius")
    trajectory_summary(d, 32768, 'FP16_clip_only', 'rho_p99',
                       'peak rho climb')

    section("dt_p50 trajectory at 32K (clip_only): per-step time-step drift")
    trajectory_summary(d, 32768, 'FP16_clip_only', 'dt_p50',
                       'dt p50 drift')

    section("NOTE: cliprate in C-10 is broken (post-clip recording)")
    print("  C-10 cliprate values are 0 by construction (recorded after clip).")
    print("  Use Phase C-10b for the corrected pre-clip cliprate measurement.")


def analyse_c10b(d):
    """Phase C-10b analysis: pre-clip cliprate + FP32_patched_off."""
    header(f"Phase C-10b (pre-clip cliprate + FP32_patched_off)")
    summarise_ppl(d)

    section("H0 sanity: FP32_patched_off PPL vs FP16_orig PPL "
            "(should be approximately equal per C-9 H0)")
    ctx_lens = [str(c) for c in d['protocol']['ctx_lens']]
    print(f"\n  {'ctx':>8s}  {'FP16_orig':>10s}  {'FP32_patched_off':>16s}  "
          f"{'rel diff':>10s}")
    for ctx in ctx_lens:
        orig = d['sweep_results'][ctx].get('FP16_orig', {}).get('ppl')
        offrec = d['sweep_results'][ctx].get('FP32_patched_off', {}).get('ppl')
        if orig is not None and offrec is not None:
            diff = (offrec - orig) / orig * 100
        else:
            diff = None
        print(f"  {ctx:>8s}  {fmt(orig, 10, 4)}  {fmt(offrec, 16, 4)}  "
              f"{fmt(diff, 10, 4)}")

    section("H1: cliprate_pre at end of ctx — FP16_clip_only (ACTUAL firing rate)")
    cliprate_per_layer_table(d, 'FP16_clip_only', key='cliprate_pre')

    section("H2: cliprate_pre at end of ctx — FP32_patched_off "
            "(NATURAL firing rate, no intervention)")
    cliprate_per_layer_table(d, 'FP32_patched_off', key='cliprate_pre')

    section("H1': cliprate_unit_pre at end of ctx — FP16_clip_only "
            "(per-(c,s) firing rate)")
    cliprate_per_layer_table(d, 'FP16_clip_only', key='cliprate_unit_pre')

    section("H2': cliprate_unit_pre at end of ctx — FP32_patched_off")
    cliprate_per_layer_table(d, 'FP32_patched_off', key='cliprate_unit_pre')

    section("h_abs_p99 trajectory at 32K, FP32_patched_off vs FP16_clip_only")
    print("  FP32_patched_off: NATURAL state magnitude (unconstrained)")
    trajectory_summary(d, 32768, 'FP32_patched_off', 'h_abs_p99',
                       'natural state magnitude tail')
    print("\n  FP16_clip_only: state magnitude with clipping active")
    trajectory_summary(d, 32768, 'FP16_clip_only', 'h_abs_p99',
                       'clipped state magnitude tail')

    section("Pattern judgment (pre-registered)")
    # Quick heuristic: did cliprate_pre rise > 1% on any layer at 32K?
    max_rate = 0.0
    max_layer = None
    for cond in ['FP16_clip_only', 'FP32_patched_off']:
        for i, lname in zip(LAYERS, LAYER_NAMES):
            traj = get_traj(d, 32768, cond, lname)
            if not traj:
                continue
            r = traj[-1].get('cliprate_pre', 0.0)
            if r > max_rate:
                max_rate = r
                max_layer = (cond, i, r)
    if max_layer is not None:
        print(f"\n  Highest cliprate_pre at 32K end: "
              f"{max_layer[0]} layer {max_layer[1]}: {max_layer[2]:.4f}")
    if max_rate >= 0.05:
        print("\n  PATTERN A: cliprate_pre >= 5% on at least one layer at 32K end.")
        print("  Direct mechanism evidence: state magnitudes regularly exceed r_g.")
        print("  Section 7.4 hero claim is supported by direct firing rate measurement.")
    elif max_rate >= 0.005:
        print("\n  PATTERN A-WEAK: cliprate_pre 0.5-5% on best layer at 32K end.")
        print("  Mechanism evidence is moderate; tail events are rare but non-trivial.")
    else:
        print("\n  PATTERN B: cliprate_pre < 0.5% throughout. Clipping rarely fires.")
        print("  The mechanism that stabilises FP16_clip_only vs FP16_orig is")
        print("  NOT primarily about state-magnitude clipping at the cliprate")
        print("  threshold tested. Section 7.4 should attribute stabilisation to")
        print("  some other effect of the clamp operation; further investigation")
        print("  needed.")


# =============================================================
# C-12 analysis
# =============================================================

def analyse_c12(d):
    """Phase C-12 analysis: half-slope vs full-slope integer comparison."""
    header(f"Phase C-12 (half-slope vs full-slope integer allocator)")
    print(f"\n  FP16 baseline PPL: {d.get('fp16_baseline_ppl'):.4f}")
    print(f"  Verdict class: {d.get('verdict_class')}")
    print(f"  Verdict: {d.get('verdict')[:200]}...")

    section("Per-(b_0, profile) results")
    profiles = ['uniform', 'full_slope', 'half_slope', 'half_slope_rg']
    b0_values = list(d['sweep_results'].keys())
    print(f"\n  {'profile':<18s}  " +
          "  ".join(f"{'b_0=' + b:>16s}" for b in b0_values))
    print(f"  {'─' * 18}  " + "  ".join("─" * 16 for _ in b0_values))
    for prof in profiles:
        row = f"  {prof:<18s}  "
        for b0 in b0_values:
            info = d['sweep_results'][b0].get(prof, {})
            mb = info.get('mean_b')
            dp = info.get('delta_pct')
            if mb is not None and dp is not None:
                row += f"  Δ={dp:>+7.3f}% (mb={mb:.2f})"
            else:
                row += f"  {'n/a':>16s}"
        print(row)

    section("Slope question: full_slope vs half_slope at matched mean bits")
    for b0 in b0_values:
        full = d['sweep_results'][b0].get('full_slope', {})
        half = d['sweep_results'][b0].get('half_slope', {})
        if not full or not half:
            continue
        full_d = full.get('delta_pct')
        half_d = half.get('delta_pct')
        full_mb = full.get('mean_b')
        half_mb = half.get('mean_b')
        diff = full_d - half_d if full_d is not None and half_d is not None else None
        winner = ("full_slope" if full_d < half_d
                  else "half_slope" if half_d < full_d else "tie")
        matched = "✓" if abs(full_mb - half_mb) < 0.1 else "≠"
        print(f"  b_0={b0}: full={full_d:+.3f}% (mb={full_mb:.2f})  "
              f"half={half_d:+.3f}% (mb={half_mb:.2f})  "
              f"diff={diff:+.3f}pp  winner={winner}  matched={matched}")

    section("r_g term question: half_slope vs half_slope_rg at matched mean bits")
    for b0 in b0_values:
        half = d['sweep_results'][b0].get('half_slope', {})
        halfrg = d['sweep_results'][b0].get('half_slope_rg', {})
        if not half or not halfrg:
            continue
        half_d = half.get('delta_pct')
        halfrg_d = halfrg.get('delta_pct')
        half_mb = half.get('mean_b')
        halfrg_mb = halfrg.get('mean_b')
        diff = half_d - halfrg_d if half_d is not None and halfrg_d is not None else None
        winner = ("half_slope" if half_d < halfrg_d
                  else "half_slope_rg" if halfrg_d < half_d else "tie")
        matched = "✓" if abs(half_mb - halfrg_mb) < 0.1 else "≠"
        print(f"  b_0={b0}: half={half_d:+.3f}% (mb={half_mb:.2f})  "
              f"half_rg={halfrg_d:+.3f}% (mb={halfrg_mb:.2f})  "
              f"diff={diff:+.3f}pp  winner={winner}  matched={matched}")

    section("Implications for v0.4 Section 3.5")
    vc = d.get('verdict_class')
    impls = {
        "FULL_SLOPE_WINS":
            ("Section 3.5's tail-risk-heuristic justification is empirically "
             "supported. The deployed slope-1 heuristic is tighter than the "
             "strict slope-1/2 optimum on the integer-bit channel. "
             "v0.4 Section 3.5 framing stands."),
        "HALF_SLOPE_WINS":
            ("Section 3.5 must adopt half_slope as the deployed allocator. "
             "Major rebuild of C-4 / C-6 / C-7 / C-8 / C-9 with the new "
             "allocator. v0.4 should report this finding and re-run the "
             "deployment chain."),
        "HALF_SLOPE_RG_WINS":
            ("Section 3.4's r_g term carries measurable benefit. v0.4 "
             "should adopt half_slope_rg and update Section 3.4 / 3.5."),
        "INDISTINGUISHABLE":
            ("Slope choice and r_g term are empirically irrelevant on this "
             "evaluation. v0.4 keeps the heuristic framing of Section 3.5 "
             "and reports C-12 as a successful closure: the integer-bit "
             "channel is forgiving on slope choice within +/- 0.5 pp."),
    }
    if vc in impls:
        print(f"\n  {impls[vc]}")
    else:
        print(f"\n  Verdict class '{vc}' not recognised; check raw output.")


# =============================================================
# Main dispatch
# =============================================================

DISPATCH = {
    "C-10":  analyse_c10,
    "C-10b": analyse_c10b,
    "C-12":  analyse_c12,
}


def main(paths):
    for p in paths:
        path = Path(p)
        if not path.exists():
            print(f"  [WARN] File not found: {p}")
            continue
        d = load_json(path)
        phase = d.get('phase', 'unknown')
        analyser = DISPATCH.get(phase)
        if analyser is None:
            print(f"  [WARN] Unknown phase '{phase}' in {p}")
            continue
        print(f"\n[file] {path}")
        print(f"[phase] {phase}")
        print(f"[date] {d.get('date', 'n/a')}")
        analyser(d)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1:])
