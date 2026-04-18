"""
modeq: Mode-aware KV cache quantization for Pre-LN Transformers.

Two empirically separable failure modes of naive INT KV quantization:

    Mode 1 (outlier-driven, rare):   catastrophic INT4 valley of death.
    Mode 2 (norm-variation-driven):  monotonic, handles any bits.

We provide nsep+pchan preprocessing that works on both, real 3-bit
packing for 5.33x memory compression vs FP16, optional per-channel
RMS residual correction (v0.2) that surpasses native INT4 on Mode 1
at slightly above 4 bits/value, and a mode-aware dispatcher that
picks the right quantizer from a measured outlier profile.
"""

from .quantize import (
    qa_perrow,
    qa_perchan,
    qa_nsep_pchan,
    qa_nsep_pchan_resid_rms,
    qa_nsep_pchan_resid_fixed,
    apply_method,
)
from .pack import pack_int3_signed, unpack_int3_signed
from .cache import Int3NsepPchanCache
from .detect import measure_kv_outlier_ratio, classify_mode
from .mode_aware import select_quantizer, describe_selection

__version__ = "0.2.2"

__all__ = [
    # quantization primitives
    "qa_perrow",
    "qa_perchan",
    "qa_nsep_pchan",
    "qa_nsep_pchan_resid_rms",
    "qa_nsep_pchan_resid_fixed",
    "apply_method",
    # INT3 packing
    "pack_int3_signed",
    "unpack_int3_signed",
    # high-level cache wrapper
    "Int3NsepPchanCache",
    # mode detection
    "measure_kv_outlier_ratio",
    "classify_mode",
    # v0.2: mode-aware dispatch
    "select_quantizer",
    "describe_selection",
]
