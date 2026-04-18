"""
modeq: Mode-aware KV cache quantization for Pre-LN Transformers.

Two empirically separable failure modes of naive INT KV quantization:

    Mode 1 (outlier-driven, rare):   catastrophic INT4 valley of death.
    Mode 2 (norm-variation-driven):  monotonic, handles any bits.

We provide nsep+pchan preprocessing that works on both, plus real
3-bit packing for 5.33x memory compression vs FP16.
"""

from .quantize import qa_perrow, qa_perchan, qa_nsep_pchan, apply_method
from .pack import pack_int3_signed, unpack_int3_signed
from .cache import Int3NsepPchanCache
from .detect import measure_kv_outlier_ratio, classify_mode

__version__ = "0.1.0"

__all__ = [
    # quantization primitives
    "qa_perrow",
    "qa_perchan",
    "qa_nsep_pchan",
    "apply_method",
    # INT3 packing
    "pack_int3_signed",
    "unpack_int3_signed",
    # high-level cache wrapper
    "Int3NsepPchanCache",
    # mode detection
    "measure_kv_outlier_ratio",
    "classify_mode",
]
