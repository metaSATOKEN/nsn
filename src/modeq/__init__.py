"""
modeq: Mode-aware KV cache quantization for Pre-LN Transformers.

Two empirically separable failure modes of naive INT KV quantization:

    Mode 1 (outlier-driven, rare):   catastrophic INT4 valley of death.
    Mode 2 (norm-variation-driven):  monotonic, handles any bits.

The v0.3 default at the INT3 tier is ``qa_nsep_pchan_asym`` with
real-packed storage via :class:`Int3AsymNsepPchanCache`: asymmetric
per-channel quantization on the norm-separated direction, fitting
each channel's ``[min, max]`` range instead of the symmetric window
around zero. Phase 14 WikiText-2 PPL showed this cuts ΔPPL by 74.3%
on Qwen2-7B (Mode 1) and 58.3% on Llama-3.1-8B (Mode 2) vs the
symmetric baseline at identical 3.0 bits/value. Phase 15 measured a
5.09x real-packed compression ratio vs FP16 (vs the 5.33x theoretical
limit; the 4.7% overhead is FP16 metadata for d_min / d_step / norm).

The v0.2 mechanisms remain available: ``qa_nsep_pchan`` (symmetric
baseline) and ``qa_nsep_pchan_resid_rms`` (per-channel RMS residual
correction). The mode-aware dispatcher ``select_quantizer`` routes
INT3 requests to asym (v0.3) and keeps INT3+RMS as the Mode-1 4-bit
default, since that composition still surpasses native INT4 on the
INT4-valley-of-death model (Qwen2-7B).
"""

from .quantize import (
    qa_perrow,
    qa_perchan,
    qa_nsep_pchan,
    qa_nsep_pchan_resid_rms,
    qa_nsep_pchan_resid_fixed,
    qa_nsep_pchan_rotated,
    qa_nsep_pchan_2tier,
    qa_nsep_pchan_asym,
    apply_method,
)
from .pack import (
    pack_int3_signed,
    unpack_int3_signed,
    pack_int3_unsigned,
    unpack_int3_unsigned,
)
from .cache import Int3NsepPchanCache, Int3AsymNsepPchanCache
from .detect import measure_kv_outlier_ratio, classify_mode
from .mode_aware import select_quantizer, describe_selection
from .rotate import hadamard_matrix, rotate, unrotate, should_rotate

__version__ = "0.3.8"

__all__ = [
    # quantization primitives
    "qa_perrow",
    "qa_perchan",
    "qa_nsep_pchan",
    "qa_nsep_pchan_resid_rms",
    "qa_nsep_pchan_resid_fixed",
    "qa_nsep_pchan_rotated",
    "qa_nsep_pchan_2tier",
    "qa_nsep_pchan_asym",
    "apply_method",
    # INT3 packing
    "pack_int3_signed",
    "unpack_int3_signed",
    "pack_int3_unsigned",
    "unpack_int3_unsigned",
    # high-level cache wrapper
    "Int3NsepPchanCache",
    "Int3AsymNsepPchanCache",
    # mode detection
    "measure_kv_outlier_ratio",
    "classify_mode",
    # v0.2: mode-aware dispatch
    "select_quantizer",
    "describe_selection",
    # v0.3: Walsh-Hadamard rotation
    "hadamard_matrix",
    "rotate",
    "unrotate",
    "should_rotate",
]
