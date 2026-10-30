"""NB05 NSN arithmetic copied verbatim from the AC08-frozen pilot; no imports of torch on import."""
import loop_kv_nb05_contract as c
BLOCK=64
DIM=128
HEADS=8
LAYERS=22
EPS=1e-12
require=c.require
class NumericalFailure(RuntimeError):pass
class IntegrityError(RuntimeError):pass
def initialize(t,n):
    global torch,np
    torch=t;np=n

def pack_codes(codes, bits):
    a = np.asarray(codes, dtype=np.uint8).reshape(-1)
    require(bits in (3, 4, 5) and np.all(a < (1 << bits)), 'Invalid packed code')
    pos = np.arange(a.size, dtype=np.int64) * bits
    out = np.zeros((a.size * bits + 7) // 8, dtype=np.uint8)
    val = a.astype(np.uint16) << (pos % 8)
    np.bitwise_or.at(out, pos // 8, (val & 255).astype(np.uint8))
    crossing = pos % 8 + bits > 8
    np.bitwise_or.at(out, pos[crossing] // 8 + 1, (val[crossing] >> 8).astype(np.uint8))
    return out.tobytes()

def unpack_codes_reference(payload, count, bits):
    # Deliberately scalar and separate from vectorized pack.
    out = np.empty(count, dtype=np.uint8)
    for i in range(count):
        pos = i * bits
        val = int(payload[pos // 8]) >> (pos % 8)
        if pos % 8 + bits > 8:
            val |= int(payload[pos // 8 + 1]) << (8 - pos % 8)
        out[i] = val & ((1 << bits) - 1)
    return out

def nsn_encode(x, bits):
    require(x.ndim >= 2 and x.shape[-2:] == (BLOCK, DIM), 'NSN block shape')
    a = x.float()
    if not bool(torch.isfinite(a).all()):
        raise NumericalFailure('Nonfinite input to NSN')
    n = torch.linalg.vector_norm(a, dim=-1, keepdim=True)
    d = a / n.clamp_min(EPS)
    m = d.amin(dim=-2, keepdim=True)
    mx = d.amax(dim=-2, keepdim=True)
    s = (mx - m) / ((1 << bits) - 1)
    denom = torch.where(s == 0, torch.ones_like(s), s)
    q = torch.round((d - m) / denom).clamp(0, (1 << bits) - 1).to(torch.uint8)
    q = torch.where(s == 0, torch.zeros_like(q), q)
    metas = tuple(t.to(torch.float16) for t in (n, m, s))
    if not all(bool(torch.isfinite(t).all()) for t in metas):
        raise NumericalFailure('FP16 metadata overflow/nonfinite')
    if bool(((n != 0) & (metas[0] == 0)).any()) or bool(((s != 0) & (metas[2] == 0)).any()):
        raise NumericalFailure('Nonzero FP16 norm/step underflow')
    return q, metas

def nsn_decode(q, metas, dtype):
    # Canonical contiguous layout makes the arithmetic independent of storage
    # strides after packing/unpacking, including non-contiguous input fixtures.
    n, m, s = (t.float().contiguous() for t in metas)
    d = (m + s * q.float().contiguous()).contiguous()
    y = n * (d / torch.linalg.vector_norm(d, dim=-1, keepdim=True).clamp_min(EPS))
    y = torch.where(n == 0, torch.zeros_like(y), y).to(dtype)
    if not bool(torch.isfinite(y).all()):
        raise NumericalFailure('Nonfinite NSN reconstruction')
    return y

def nsn_qdq(x, bits):
    q, meta = nsn_encode(x, bits)
    return nsn_decode(q, meta, x.dtype)

def planned_bytes(shape, compressed_tokens, bits):
    b, h, length, d = shape
    require(compressed_tokens % BLOCK == 0 and 0 <= compressed_tokens <= length, 'Block accounting')
    if bits is None:
        return {'codes': 0, 'metadata': 0, 'native': b*h*length*d*2, 'block_headers': 0}
    blocks = b*h*(compressed_tokens // BLOCK)
    # Header per head-block: token offset uint64, count uint16, bits uint8,
    # reserved uint8. Full blocks need no bitstream byte padding.
    return {'codes': blocks*((BLOCK*d*bits + 7)//8),
            'metadata': blocks*(BLOCK*2 + d*4),
            'native': b*h*(length-compressed_tokens)*d*2, 'block_headers': blocks*12}
