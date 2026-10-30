"""Ten-line arithmetic core for valid NSN blocks; not a packed-cache implementation.

Input: [..., 64, 128] finite float32/float16/bfloat16 blocks that pass the
experiment's metadata-range checks. The function body has ten lines.
FP16 metadata rounding is included. Packing, cache state, validation and
attention kernels remain in the frozen experiment implementation.
"""
import torch


def nsn_int3_qdq(x):
    a = x.float()
    n = torch.linalg.vector_norm(a, dim=-1, keepdim=True)
    d = a / n.clamp_min(1e-12)
    m = d.amin(dim=-2, keepdim=True)
    s = (d.amax(dim=-2, keepdim=True) - m) / 7
    q = ((d - m) / torch.where(s == 0, torch.ones_like(s), s)).round().clamp(0, 7).to(torch.uint8)
    n, m, s = (t.half().float().contiguous() for t in (n, m, s))
    d = (m + s * q.float().contiguous()).contiguous()
    y = n * (d / torch.linalg.vector_norm(d, dim=-1, keepdim=True).clamp_min(1e-12))
    return torch.where(n == 0, torch.zeros_like(y), y).to(x.dtype)
