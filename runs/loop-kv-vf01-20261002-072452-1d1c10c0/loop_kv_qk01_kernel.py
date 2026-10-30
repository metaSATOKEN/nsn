"""QK01: actual packed K -> BF16-rounded K -> FP32 scores. No GPU work on import."""
import math
import loop_kv_rf01_kernel as ref
w=ref.w
TOKEN_TILE=16
WARPS=4
_kernel=None

def initialize():
    global _kernel,tl
    if _kernel is not None:return
    import triton
    import triton.language as tl
    @triton.jit
    def _qk_kernel(Codes,N,M,S,Tail,Q,Score,Flags,DebugK,
                   C:tl.constexpr,T:tl.constexpr,H:tl.constexpr,TRACE:tl.constexpr,BT:tl.constexpr):
        h=tl.program_id(0).to(tl.int64)
        block=tl.program_id(1)
        t=(block*BT+tl.arange(0,BT)).to(tl.int64)
        d=tl.arange(0,128).to(tl.int64)
        valid=t<T
        q=tl.load(Q+h*128+d).to(tl.float32)
        if C>0:
            prefix=valid & (t<C)
            i=(h*C+t[:,None])*128+d[None,:]
            bit=i*3;byte=bit//8;shift=(bit%8).to(tl.uint32)
            lo=tl.load(Codes+byte,mask=prefix[:,None],other=0).to(tl.uint32)
            hi=tl.load(Codes+byte+1,mask=prefix[:,None] & (byte+1<H*C*128*3//8),other=0).to(tl.uint32)
            code=(((lo|(hi<<8))>>shift)&7).to(tl.float32)
            mi=(h*(C//64)+t[:,None]//64)*128+d[None,:]
            mm=tl.load(M+mi,mask=prefix[:,None],other=0).to(tl.float32)
            ss=tl.load(S+mi,mask=prefix[:,None],other=0).to(tl.float32)
            nn=tl.load(N+h*C+t,mask=prefix,other=0).to(tl.float32)
            product=ss*code
            direction=mm+product
            square=direction*direction
            denominator=tl.maximum(tl.sqrt(tl.sum(square,axis=1)),1.e-12)
            normalized=tl.div_rn(direction,denominator[:,None])
            scaled=nn[:,None]*normalized
            restored=tl.where(nn[:,None]==0,0.,scaled).to(tl.bfloat16,fp_downcast_rounding='rtne').to(tl.float32)
            nonfinite_prefix=prefix[:,None] & ((tl.abs(direction)==float('inf')) | (direction!=direction) | (tl.abs(restored)==float('inf')) | (restored!=restored))
            bad_prefix=tl.sum(tl.sum(nonfinite_prefix.to(tl.int32),axis=1),axis=0)>0
            bad_norm=tl.sum((prefix & ((tl.abs(denominator)==float('inf')) | (denominator!=denominator))).to(tl.int32),axis=0)>0
            tail=tl.load(Tail+(h*(T-C)+t[:,None]-C)*128+d[None,:],mask=valid[:,None] & (t[:,None]>=C),other=0).to(tl.float32)
            kval=tl.where((t<C)[:,None],restored,tail)
        else:
            kval=tl.load(Tail+(h*T+t[:,None])*128+d[None,:],mask=valid[:,None],other=0).to(tl.float32)
            bad_prefix=False
            bad_norm=False
        value=tl.sum(kval*q[None,:],axis=1)*0.08838834764831845
        bad_score=tl.sum((valid & ((tl.abs(value)==float('inf')) | (value!=value))).to(tl.int32),axis=0)>0
        bad_q=tl.sum(((tl.abs(q)==float('inf')) | (q!=q)).to(tl.int32),axis=0)>0
        bad_k=tl.sum(tl.sum((valid[:,None] & ((tl.abs(kval)==float('inf')) | (kval!=kval))).to(tl.int32),axis=1),axis=0)>0
        tl.store(Score+h*T+t,value,mask=valid)
        tl.store(Flags+h*tl.cdiv(T,BT)+block,(bad_prefix|bad_norm|bad_score|bad_q|bad_k).to(tl.int32))
        if TRACE:
            tl.store(DebugK+(h*T+t[:,None])*128+d[None,:],kval.to(tl.bfloat16),mask=valid[:,None])
    _kernel=_qk_kernel

def environment():
    base=ref.environment();w.require(base['baseline']['triton']=='3.6.0','QK01 needs Triton 3.6.0')
    return dict(reference=base,token_tile=TOKEN_TILE,num_warps=WARPS,autotune=False,enable_fp_fusion=False,
        normalization='FP32 sum-of-squares/sqrt in token row',division='div_rn',kv_rounding='BF16 rtne before QK',
        accumulation='FP32 reduction',scale=1/math.sqrt(128),mask='all stored positions; single decode query')

def validate(g,tail,q):
    t=w.torch
    w.require(tail.ndim==4 and tail.shape[0]==1 and tail.shape[-1]==128 and tail.dtype==t.bfloat16 and tail.is_contiguous(),'Tail shape/type')
    h=tail.shape[1];w.require(h>0 and tuple(q.shape)==(1,h,1,128) and q.dtype==t.bfloat16 and q.is_contiguous(),'Query shape/type')
    w.require(q.device==tail.device and tail.device.type=='cuda','CUDA only; no fallback')
    compressed=0
    if g is not None:
        b,gh,blocks=ref.shape_check(g,t.bfloat16);w.require(b==1 and gh==h and g['codes'].device==tail.device,'Packed head/device mismatch');compressed=blocks*64
    w.require(tail.shape[-2]+compressed>0,'Empty sequence')
    # Both arms receive the same invariant checks inside every measured call.
    values=[q,tail]+([] if g is None else list(g['metas']))
    if not bool(t.stack([t.isfinite(x).all() for x in values]).all()):raise w.NumericalFailure('Nonfinite QK input')
    return h,compressed,compressed+tail.shape[-2]

def reference_k(g,tail):
    if g is None:return tail
    return w.torch.cat((ref.restore_group(g,tail.dtype),tail),dim=-2)

def launch(g,tail,q,trace=False):
    t=w.torch;h=tail.shape[1];c=0 if g is None else g['codes'].shape[2]*64;total=c+tail.shape[-2]
    initialize();score=t.empty((1,h,total),device=tail.device,dtype=t.float32)
    flags=t.empty((h,(total+TOKEN_TILE-1)//TOKEN_TILE),device=tail.device,dtype=t.int32)
    debug=t.empty((1,h,total,128),device=tail.device,dtype=t.bfloat16) if trace else score
    # Dummy pointers in the native-only branch are never dereferenced as metadata.
    codes,n,m,s=(tail,tail,tail,tail) if g is None else (g['codes'],*g['metas'])
    _kernel[(h,(total+TOKEN_TILE-1)//TOKEN_TILE)](codes,n,m,s,tail,q,score,flags,debug,
        C=c,T=total,H=h,TRACE=trace,BT=TOKEN_TILE,num_warps=WARPS,enable_fp_fusion=False)
    if bool((flags!=0).any()):raise w.NumericalFailure('Nonfinite QK intermediate or score')
    return (score,debug) if trace else score

def run(arm,g,tail,q):
    validate(g,tail,q)
    if arm=='FUSED_QK':return launch(g,tail,q)
    full=reference_k(g,tail)
    if arm=='RF01_TORCH':
        score=(w.torch.matmul(q.float(),full.float().transpose(-2,-1)).squeeze(-2)*(1/math.sqrt(128))).contiguous()
        if not bool(w.torch.isfinite(score).all()):raise w.NumericalFailure('Nonfinite reference score')
        return score
    if arm=='RF01_TRITON':return launch(None,full,q)
    raise w.IntegrityError('Unknown QK arm')

def diagnostic(g,tail,q):
    validate(g,tail,q);return launch(g,tail,q,trace=True)
