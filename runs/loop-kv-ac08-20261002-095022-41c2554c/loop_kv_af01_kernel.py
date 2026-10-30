"""AF01 split-tile fused attention. No GPU work on import; no autotune/fallback."""
import loop_kv_qk01_kernel as qk
w=qk.w
TOKEN_TILE=16;WARPS=4
_partial=None;_merge=None

def initialize():
    global _partial,_merge,tl
    if _partial is not None:return
    import triton
    import triton.language as tl
    @triton.jit
    def partial(Codes,N,M,S,Tail,Q,V,Scores,Maxima,Denom,Weighted,Flags,TraceScore,
                C:tl.constexpr,T:tl.constexpr,H:tl.constexpr,FULL:tl.constexpr,TRACE:tl.constexpr,BT:tl.constexpr):
        h=tl.program_id(0).to(tl.int64);block=tl.program_id(1)
        t=(block*BT+tl.arange(0,BT)).to(tl.int64);d=tl.arange(0,128).to(tl.int64)
        valid=t<T
        if FULL:
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
            bad=bad_prefix|bad_norm|bad_score|bad_q|bad_k
        else:
            value=tl.load(Scores+h*T+t,mask=valid,other=0).to(tl.float32)
            bad=tl.sum((valid & ((tl.abs(value)==float('inf')) | (value!=value))).to(tl.int32),axis=0)>0
        masked=tl.where(valid,value,-float('inf'))
        maximum=tl.max(masked,axis=0)
        weight=tl.exp(masked-maximum)
        denominator=tl.sum(weight,axis=0)
        vv=tl.load(V+(h*T+t[:,None])*128+d[None,:],mask=valid[:,None],other=0).to(tl.float32)
        weighted=tl.sum(weight[:,None]*vv,axis=0)
        badv=tl.sum(tl.sum((valid[:,None] & ((tl.abs(vv)==float('inf')) | (vv!=vv))).to(tl.int32),axis=1),axis=0)>0
        badw=tl.sum(((tl.abs(weighted)==float('inf')) | (weighted!=weighted)).to(tl.int32),axis=0)>0
        blocks=tl.cdiv(T,BT)
        tl.store(Maxima+h*blocks+block,maximum)
        tl.store(Denom+h*blocks+block,denominator)
        tl.store(Weighted+(h*blocks+block)*128+d,weighted)
        tl.store(Flags+h*(blocks+1)+block,(bad|badv|badw|(denominator<=0)|(denominator!=denominator)).to(tl.int32))
        if TRACE:tl.store(TraceScore+h*T+t,value,mask=valid)
    @triton.jit
    def merge(Maxima,Denom,Weighted,Flags,Context,Output,H:tl.constexpr,NB:tl.constexpr,BP:tl.constexpr):
        h=tl.program_id(0).to(tl.int64)
        b=tl.arange(0,BP).to(tl.int64);d=tl.arange(0,128).to(tl.int64)
        m=tl.load(Maxima+h*NB+b,mask=b<NB,other=-float('inf'))
        z=tl.load(Denom+h*NB+b,mask=b<NB,other=0)
        u=tl.load(Weighted+(h*NB+b[:,None])*128+d[None,:],mask=(b<NB)[:,None],other=0)
        largest=tl.max(m,axis=0);alpha=tl.exp(m-largest)
        denominator=tl.sum(z*alpha,axis=0)
        context=tl.div_rn(tl.sum(u*alpha[:,None],axis=0),denominator)
        out=context.to(tl.bfloat16,fp_downcast_rounding='rtne')
        bad=tl.sum(((tl.abs(context)==float('inf')) | (context!=context) | (tl.abs(out.to(tl.float32))==float('inf'))).to(tl.int32),axis=0)>0
        tl.store(Context+h*128+d,context);tl.store(Output+h*128+d,out)
        tl.store(Flags+h*(NB+1)+NB,(bad|(denominator<=0)|(denominator!=denominator)).to(tl.int32))
    _partial=partial;_merge=merge

def config():
    return dict(token_tile=TOKEN_TILE,num_warps=WARPS,autotune=False,enable_fp_fusion=False,stages=2,partial_dtype='float32',output_dtype='bfloat16',division='div_rn',exp='tl.exp',v_input='restored BF16')

def validate_v(v,q,total):
    t=w.torch
    w.require(v.dtype==t.bfloat16 and v.is_contiguous() and tuple(v.shape)==(1,q.shape[1],total,128),'V shape/type')
    w.require(v.device==q.device and v.device.type=='cuda','V device')
    if not bool(t.isfinite(v).all()):raise w.NumericalFailure('Nonfinite V')

def validate_score(score,v):
    t=w.torch
    w.require(score.dtype==t.float32 and score.is_contiguous() and tuple(score.shape)==(1,v.shape[1],v.shape[2]),'Score shape/type')
    w.require(score.device==v.device and v.device.type=='cuda','Score device')
    if not bool(t.isfinite(score).all()):raise w.NumericalFailure('Nonfinite score')

def launch(g,tail,q,v,score=None,full=False,trace=False):
    """Caller supplies validated inputs; finite tile/output flags are always read before return."""
    t=w.torch;initialize();h=v.shape[1];total=v.shape[2];c=0 if g is None else g['codes'].shape[2]*64;nb=(total+15)//16
    m=t.empty((h,nb),device=v.device,dtype=t.float32);z=t.empty_like(m)
    u=t.empty((h,nb,128),device=v.device,dtype=t.float32);flags=t.empty((h,nb+1),device=v.device,dtype=t.int32)
    context=t.empty((1,h,1,128),device=v.device,dtype=t.float32);output=t.empty((1,1,h,128),device=v.device,dtype=t.bfloat16)
    trace_score=t.empty((1,h,total),device=v.device,dtype=t.float32) if trace else m
    codes,n,mean,scale=(tail,tail,tail,tail) if g is None else (g['codes'],*g['metas'])
    _partial[(h,nb)](codes,n,mean,scale,tail,q,v,q if score is None else score,m,z,u,flags,trace_score,C=c,T=total,H=h,FULL=full,TRACE=trace,BT=16,num_warps=4,enable_fp_fusion=False)
    _merge[(h,)](m,z,u,flags,context,output,H=h,NB=nb,BP=1<<(nb-1).bit_length(),num_warps=4,enable_fp_fusion=False)
    if bool((flags!=0).any()):raise w.NumericalFailure('Nonfinite fused attention intermediate/output')
    return output,context,trace_score if trace else None
