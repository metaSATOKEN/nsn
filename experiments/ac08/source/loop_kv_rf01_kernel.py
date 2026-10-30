"""RF01 pointwise restore fusion, retaining torch vector_norm/clamp and immediate checks.
Importing never imports Triton or touches CUDA. No CPU fallback.
"""
import math
import loop_kv_pu01_kernel as baseline
c=baseline.c;w=c.w
BLOCK=1024;WARPS=4
_direction=None;_finish=None

def environment():
    return dict(baseline=baseline.environment(),block=BLOCK,num_warps=WARPS,autotune=False,
                enable_fp_fusion=False,division='div_rn',output_rounding='rtne',norm='torch_vector_norm_clamp')

def verify_environment(env):
    baseline.verify_environment(env['baseline'])
    w.require({k:v for k,v in env.items() if k!='baseline'}==dict(block=BLOCK,num_warps=WARPS,autotune=False,
        enable_fp_fusion=False,division='div_rn',output_rounding='rtne',norm='torch_vector_norm_clamp'),'Restore kernel configuration')

def initialize():
    global _direction,_finish,tl
    if _direction is not None:return
    import triton
    import triton.language as tl
    @triton.jit
    def _direction_kernel(Packed,Min,Scale,Output,N,P,BLOCK:tl.constexpr):
        i=tl.program_id(0).to(tl.int64)*BLOCK+tl.arange(0,BLOCK).to(tl.int64)
        bit=i*3;byte=bit//8;shift=(bit%8).to(tl.uint32)
        lo=tl.load(Packed+byte,mask=(i<N)&(byte<P),other=0).to(tl.uint32)
        hi=tl.load(Packed+byte+1,mask=(i<N)&(byte+1<P),other=0).to(tl.uint32)
        q=(((lo|(hi<<8))>>shift)&7).to(tl.float32)
        mi=(i//8192)*128+i%128
        m=tl.load(Min+mi,mask=i<N,other=0).to(tl.float32)
        s=tl.load(Scale+mi,mask=i<N,other=0).to(tl.float32)
        product=s*q
        value=m+product
        tl.store(Output+i,value,mask=i<N)
    @triton.jit
    def _finish_kernel(Direction,Denominator,Norm,Output,N,BLOCK:tl.constexpr):
        i=tl.program_id(0).to(tl.int64)*BLOCK+tl.arange(0,BLOCK).to(tl.int64)
        d=tl.load(Direction+i,mask=i<N,other=0)
        den=tl.load(Denominator+i//128,mask=i<N,other=1)
        n=tl.load(Norm+i//128,mask=i<N,other=0).to(tl.float32)
        normalized=tl.div_rn(d,den)
        scaled=n*normalized
        value=tl.where(n==0,0.,scaled).to(tl.bfloat16,fp_downcast_rounding='rtne')
        tl.store(Output+i,value,mask=i<N)
    _direction=_direction_kernel;_finish=_finish_kernel

def shape_check(g,dtype):
    t=w.torch;codes=g['codes']
    w.require(dtype==t.bfloat16,'RF01 output must be BF16')
    w.require(codes.ndim==4 and codes.dtype==t.uint8 and codes.is_contiguous() and codes.shape[-1]==3072,'RF01 packed layout')
    b,h,n,_=codes.shape
    w.require(b>0 and h>0 and n>0,'RF01 nonempty shape')
    shapes=[(b,h,n,64,1),(b,h,n,1,128),(b,h,n,1,128)]
    w.require(len(g['metas'])==3 and all(tuple(a.shape)==sh and a.dtype==t.float16 and a.is_contiguous() and a.device==codes.device for a,sh in zip(g['metas'],shapes)),'RF01 metadata layout')
    w.require(codes.device.type=='cuda','RF01 requires CUDA; no fallback')
    return b,h,n

def trace(g,dtype):
    t=w.torch;b,h,n=shape_check(g,dtype);initialize();size=b*h*n*8192
    direction=t.empty((b,h,n,64,128),device=g['codes'].device,dtype=t.float32)
    grid=((size+BLOCK-1)//BLOCK,)
    _direction[grid](g['codes'],g['metas'][1],g['metas'][2],direction,size,g['codes'].numel(),
        BLOCK=BLOCK,num_warps=WARPS,enable_fp_fusion=False)
    denominator=t.linalg.vector_norm(direction,dim=-1,keepdim=True).clamp_min(w.EPS)
    out=t.empty_like(direction,dtype=dtype)
    _finish[grid](direction,denominator,g['metas'][0],out,size,BLOCK=BLOCK,num_warps=WARPS,enable_fp_fusion=False)
    if not bool(t.isfinite(out).all()):raise w.NumericalFailure('Nonfinite NSN reconstruction')
    return direction,denominator,out

def restore_group(g,dtype):
    b,h,n,_=g['codes'].shape
    return trace(g,dtype)[2].reshape(b,h,n*64,128)

def make_cache(base):
    parent=c.make_packed_cache(base)
    class RestoreCache(parent):
        def materialize(self,slot,kv):
            tail=(self.key_cache if kv=='K' else self.value_cache)[slot]
            group=self.packed.get((slot,kv))
            if group is None:return tail
            return w.torch.cat((restore_group(group,tail.dtype),tail),-2)
    return RestoreCache
