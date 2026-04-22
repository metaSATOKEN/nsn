# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

"""
High-level KV cache wrappers with real bit-packing.

v0.3 exports two cache classes:

* :class:`Int3AsymNsepPchanCache` -- the v0.3 default. Asymmetric
  per-channel INT3 on the norm-separated direction. Phase 15
  measured 5.09x real compression vs FP16, and Phase 14 measured
  ΔPPL +0.10 on Qwen2-7B and +0.11 on Llama-3.1-8B at WikiText-2
  sliding-window PPL (ctx=1024). Metadata overhead vs the symmetric
  v0.2 cache is +0.008 bits/value (one extra FP16 per channel for
  d_min).

* :class:`Int3NsepPchanCache` -- the v0.2 symmetric cache, retained
  for backwards compatibility and for benchmarks that need the
  original per-channel absmax quantizer.

Typical usage (illustrative; see `modeq.integrations.*` for
drop-in HF Transformers hooks when they land):

    from modeq import Int3AsymNsepPchanCache

    # Compress a Key or Value tensor [B, H, T, D] from an FP16 cache
    cached = Int3AsymNsepPchanCache.from_fp16(k)

    # Reconstruct for attention (returns FP16 [B, H, T, D])
    k_recon = cached.to_fp16()

    # Check actual storage size
    bytes_used = cached.size_bytes()

Phase 15 real-packed memory measurements (2026-04-18, asymmetric):
    Qwen2-7B     (2048 ctx, 28 layers, 4 KV heads): 112 MB -> 22 MB (5.09x)
    Llama-3.1-8B (2048 ctx, 32 layers, 8 KV heads): 256 MB -> 50 MB (5.09x)

Quality at WikiText-2 (Phase 14, asym vs symmetric at 3.0 bits/value):
    Qwen2-7B  (Mode 1): symmetric +0.4021 -> asymmetric +0.1034 (-74.3%)
    Llama-3.1-8B (Mode 2): symmetric +0.2541 -> asymmetric +0.1059 (-58.3%)
"""

import torch

from .pack import (
    pack_int3_signed,
    unpack_int3_signed,
    pack_int3_unsigned,
    unpack_int3_unsigned,
)


class Int3NsepPchanCache:
    """
    Real-packed INT3 KV cache storage.

    Storage fields (all on same device as input):
        packed_d : uint8 tensor, shape [..., 3*D/8]
            3-bit direction values, 8 per 3 bytes.
        norm : float16 tensor, shape [..., T, 1]
            Per-token L2 norm of the original FP16 tensor.
        ch_scale : float16 tensor, shape [..., 1, D]
            Per-channel absmax-based scale of the direction vectors.

    Invariants:
        - head_dim D must be divisible by 8.
        - Input tensor shape is [..., T, D]; last dim D is the
          head dimension to be quantized.
    """

    def __init__(self):
        self.packed_d = None
        self.norm = None
        self.ch_scale = None

    @classmethod
    def from_fp16(cls, x, qm=3):
        """
        Encode an FP16 tensor into packed INT3 nsep+pchan form.

        Parameters
        ----------
        x : torch.Tensor, shape [..., T, D], dtype fp16 or fp32
            Input key/value tensor from the model's KV cache.
        qm : int, default 3
            Quantization maximum; INT3 symmetric uses qm=3 (range
            [-3, 3], 7 levels stored in 3 bits).

        Returns
        -------
        Int3NsepPchanCache
        """
        assert x.shape[-1] % 8 == 0, (
            f"Last dim (head_dim) must be divisible by 8, got {x.shape[-1]}"
        )
        obj = cls()
        xf = x.float()
        n = xf.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        d = xf / n
        ch_scale = d.abs().amax(dim=-2, keepdim=True).clamp(min=1e-12) / qm
        d_int3 = (d / ch_scale).round().clamp(-qm, qm).to(torch.int8)
        obj.packed_d = pack_int3_signed(d_int3)
        obj.norm = n.half()
        obj.ch_scale = ch_scale.half()
        return obj

    def to_fp16(self):
        """
        Reconstruct an approximate FP16 tensor from stored INT3 packing.

        Returns
        -------
        torch.Tensor, dtype fp16, shape [..., T, D]
        """
        *prefix, P = self.packed_d.shape
        D = P * 8 // 3
        d_int3 = unpack_int3_signed(self.packed_d, N=D)
        d_q = d_int3.float() * self.ch_scale.float()
        d_renorm = d_q / d_q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return (self.norm.float() * d_renorm).half()

    def size_bytes(self):
        """
        Total bytes in real storage (packed direction + norm + scale).
        """
        return (
            self.packed_d.numel() * 1
            + self.norm.numel() * 2
            + self.ch_scale.numel() * 2
        )


class Int3AsymNsepPchanCache:
    """
    Real-packed INT3 ASYMMETRIC nsep+pchan cache (v0.3 default).

    Compared to :class:`Int3NsepPchanCache` (symmetric, v0.2):
      * 8 levels instead of 7 (uses the full 3-bit unsigned range).
      * Stores per-channel ``d_min`` in addition to ``d_step``, so the
        metadata is 2 FP16 per channel per (head, K/V) direction
        instead of 1. Amortized over T cached tokens, this extra
        per-channel scalar costs +16/T bits/value -- roughly +0.008
        bits/value at T=2048 -- matching the Phase 15 packed
        measurement (3.141 vs 3.133 effective bits/value). Negligible
        next to the 58-74% ΔPPL reduction this method buys on
        WikiText-2 (Phase 14).

    Storage fields (all on same device as input):
        packed_d : uint8 tensor, shape [..., 3*D/8]
            3-bit unsigned direction indices in [0, 7], 8 per 3 bytes.
        norm : float16 tensor, shape [..., T, 1]
            Per-token L2 norm of the original FP16 tensor.
        d_min : float16 tensor, shape [..., 1, D]
            Per-channel minimum of the unit direction.
        d_step : float16 tensor, shape [..., 1, D]
            Per-channel quantization step = (d_max - d_min) / 7.
    """

    def __init__(self):
        self.packed_d = None
        self.norm = None
        self.d_min = None
        self.d_step = None

    @classmethod
    def from_fp16(cls, x, bits=3):
        """
        Encode an FP16 tensor into packed INT3 asymmetric nsep+pchan form.

        Parameters
        ----------
        x : torch.Tensor, shape [..., T, D], dtype fp16 or fp32
            Input key/value tensor.
        bits : int, default 3
            Bit width; only bits=3 is currently supported for real
            packing via :func:`pack_int3_unsigned`.
        """
        assert bits == 3, f"Int3AsymNsepPchanCache packs 3-bit only; got bits={bits}"
        assert x.shape[-1] % 8 == 0, (
            f"Last dim (head_dim) must be divisible by 8, got {x.shape[-1]}"
        )
        obj = cls()
        xf = x.float()
        n = xf.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        d = xf / n

        levels = 2 ** bits - 1  # 7 for INT3 (indices 0..7)
        d_min = d.amin(dim=-2, keepdim=True)
        d_max = d.amax(dim=-2, keepdim=True)
        d_range = (d_max - d_min).clamp(min=1e-12)
        d_step = d_range / levels

        d_idx = ((d - d_min) / d_step).round().clamp(0, levels).to(torch.int8)
        obj.packed_d = pack_int3_unsigned(d_idx)
        obj.norm = n.half()
        obj.d_min = d_min.half()
        obj.d_step = d_step.half()
        return obj

    def to_fp16(self):
        """Reconstruct an approximate FP16 tensor."""
        *prefix, P = self.packed_d.shape
        D = P * 8 // 3
        d_idx = unpack_int3_unsigned(self.packed_d, N=D)
        d_q = d_idx.float() * self.d_step.float() + self.d_min.float()
        d_renorm = d_q / d_q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return (self.norm.float() * d_renorm).half()

    def to_fp16_compiled(self):
        """torch.compile-accelerated variant of :meth:`to_fp16`.

        Numerically identical to :meth:`to_fp16` within FP16 rounding
        (same mathematical pipeline), but wraps the unpack +
        reconstruction + renormalization body in a ``torch.compile
        (dynamic=False)`` kernel so the 10+ small CUDA kernels that
        :meth:`to_fp16` launches get fused into 1-2 compiled kernels.

        On an NVIDIA A100 (Colab, 1024-token prefill, 50 decode steps
        median of 3 repeats) this reduces per-token decode overhead
        from roughly +33 % (eager :meth:`to_fp16`, per-layer batched)
        to +2.4 % on Qwen2-7B and +4.2 % on Llama-3.1-8B vs the FP16
        baseline -- effectively at measurement noise. See
        ``results/phase27_decode_latency.json`` and
        ``experiments/phase27_decode_latency.py`` for the full
        five-regime benchmark.

        Caveats:

        * First call to any compiled form triggers a JIT compilation
          that takes a few seconds. Subsequent calls on the same
          tensor shapes are effectively free. For reproducible
          warm-state benchmarking, do 5-10 warm-up calls before the
          timed window.
        * ``dynamic=False`` specializes on the first observed
          ``(..., T, D)`` shape. If you call across heterogeneous
          shapes the first invocation at each new shape triggers a
          recompile. For homogeneous models (all layers same shape)
          this is a one-time cost.
        * The compile cache is class-level (``_compiled_unpack``),
          so multiple cache instances share the same JITed kernel.
        """
        return self._get_compiled_unpack()(
            self.packed_d, self.norm, self.d_min, self.d_step
        )

    # Class-level cache for the compiled unpack. Initialized on first call.
    _compiled_unpack = None

    @classmethod
    def _get_compiled_unpack(cls):
        """Lazily compile :func:`_to_fp16_functional` once and cache
        it at the class level so all instances share it."""
        if cls._compiled_unpack is None:
            cls._compiled_unpack = torch.compile(
                _to_fp16_functional, dynamic=False
            )
        return cls._compiled_unpack

    def size_bytes(self):
        """Total bytes in real storage (packed + norm + d_min + d_step)."""
        return (
            self.packed_d.numel() * 1
            + self.norm.numel() * 2
            + self.d_min.numel() * 2
            + self.d_step.numel() * 2
        )


# ---------------------------------------------------------------------------
# Functional helpers (module-level) for torch.compile
# ---------------------------------------------------------------------------

def _to_fp16_functional(packed_d, norm, d_min, d_step):
    """Pure-functional equivalent of :meth:`Int3AsymNsepPchanCache.to_fp16`.

    Inlines :func:`unpack_int3_unsigned` so the whole pipeline
    (bit-unpack + reconstruct + L2 renormalize + fp16 cast) is a
    single contiguous graph that :func:`torch.compile` can fuse into
    1-2 GPU kernels.

    Parameters
    ----------
    packed_d : torch.Tensor (uint8, [..., 3*D/8])
    norm     : torch.Tensor (fp16,  [..., T, 1])
    d_min    : torch.Tensor (fp16,  [..., 1, D])
    d_step   : torch.Tensor (fp16,  [..., 1, D])

    Returns
    -------
    torch.Tensor (fp16, [..., T, D])
    """
    *prefix, P = packed_d.shape
    N = P * 8 // 3

    # Inlined unpack_int3_unsigned
    grouped = packed_d.view(*prefix, N // 8, 3)
    b0 = grouped[..., 0]
    b1 = grouped[..., 1]
    b2 = grouped[..., 2]
    packed64 = (
        b0.to(torch.int64)
        | (b1.to(torch.int64) << 8)
        | (b2.to(torch.int64) << 16)
    )
    shifts = torch.arange(8, device=packed_d.device, dtype=torch.int64) * 3
    u = (packed64.unsqueeze(-1) >> shifts) & 7
    d_idx = u.view(*prefix, N).to(torch.int8)

    # Reconstruct
    d_q = d_idx.float() * d_step.float() + d_min.float()
    d_renorm = d_q / d_q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return (norm.float() * d_renorm).half()
