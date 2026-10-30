"""VF01 V restoration from copied AF01/QK01 arithmetic. Lazy Triton; no fallback."""
import loop_kv_af01_kernel as af
import loop_kv_iv01_kernel as iv
qk=af.qk;ref=qk.ref;w=qk.w
TOKEN_TILE=16;WARPS=4
_partial=None;_materialize=None
LAST_STAGE='idle'

def initialize():
    global _partial,_materialize,_read_v,tl
    if _partial is not None:return
    import triton
    import triton.language as tl
    @triton.jit
    def read_v(Codes,N,M,S,Tail,h,t,d,C:tl.constexpr,T:tl.constexpr,H:tl.constexpr):
        valid=t<T
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
            vv=tl.where((t<C)[:,None],restored,tail)
            bad=bad_prefix|bad_norm
        else:
            vv=tl.load(Tail+(h*T+t[:,None])*128+d[None,:],mask=valid[:,None],other=0).to(tl.float32)
            bad=False
        return vv,bad
    _read_v=read_v
    @triton.jit
    def materialize(Codes,N,M,S,Tail,Output,Flags,C:tl.constexpr,T:tl.constexpr,H:tl.constexpr,BT:tl.constexpr):
        h=tl.program_id(0).to(tl.int64);block=tl.program_id(1)
        t=(block*BT+tl.arange(0,BT)).to(tl.int64);d=tl.arange(0,128).to(tl.int64)
        vv,bad=_read_v(Codes,N,M,S,Tail,h,t,d,C,T,H)
        invalid=(t<T)[:,None] & ((vv!=vv)|(tl.abs(vv)==float('inf')))
        bad=bad | (tl.sum(tl.sum(invalid.to(tl.int32),axis=1),axis=0)>0)
        tl.store(Output+(h*T+t[:,None])*128+d[None,:],vv.to(tl.bfloat16),mask=(t<T)[:,None])
        tl.store(Flags+h*tl.cdiv(T,BT)+block,bad.to(tl.int32))
    @triton.jit
    def partial(Codes,N,M,S,Tail,Q,V,Scores,Maxima,Denom,Weighted,Flags,TraceScore,VCodes,VN,VM,VS,VTail,TraceV,
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
        vv,vrestore_bad=_read_v(VCodes,VN,VM,VS,VTail,h,t,d,C,T,H)
        bad=bad|vrestore_bad
        if TRACE:tl.store(TraceV+(h*T+t[:,None])*128+d[None,:],vv.to(tl.bfloat16),mask=valid[:,None])
        weighted=tl.sum(weight[:,None]*vv,axis=0)
        badv=tl.sum(tl.sum((valid[:,None] & ((tl.abs(vv)==float('inf')) | (vv!=vv))).to(tl.int32),axis=1),axis=0)>0
        badw=tl.sum(((tl.abs(weighted)==float('inf')) | (weighted!=weighted)).to(tl.int32),axis=0)>0
        blocks=tl.cdiv(T,BT)
        tl.store(Maxima+h*blocks+block,maximum)
        tl.store(Denom+h*blocks+block,denominator)
        tl.store(Weighted+(h*blocks+block)*128+d,weighted)
        tl.store(Flags+h*(blocks+1)+block,(bad|badv|badw|(denominator<=0)|(denominator!=denominator)).to(tl.int32))
        if TRACE:tl.store(TraceScore+h*T+t,value,mask=valid)
    _partial=partial;_materialize=materialize
    af.initialize()

def pointers(g,tail):
    return (tail,tail,tail,tail) if g is None else (g['codes'],*g['metas'])

def restore_v(g,tail):
    t=w.torch;initialize();h=tail.shape[1];c=0 if g is None else g['codes'].shape[2]*64;total=c+tail.shape[2];nb=(total+15)//16
    out=t.empty((1,h,total,128),device=tail.device,dtype=t.bfloat16)
    flags=t.empty((h,nb),device=tail.device,dtype=t.int32)
    _materialize[(h,nb)](*pointers(g,tail),tail,out,flags,C=c,T=total,H=h,BT=16,num_warps=4,enable_fp_fusion=False)
    if bool((flags!=0).any()):raise w.NumericalFailure('VF01 nonfinite V reconstruction')
    return out

def launch(g,kt,q,vg,vt,trace=False):
    t=w.torch;initialize();h=q.shape[1];c=0 if g is None else g['codes'].shape[2]*64;total=c+kt.shape[2];nb=(total+15)//16
    m=t.empty((h,nb),device=q.device,dtype=t.float32);z=t.empty_like(m)
    u=t.empty((h,nb,128),device=q.device,dtype=t.float32);flags=t.empty((h,nb+1),device=q.device,dtype=t.int32)
    context=t.empty((1,h,1,128),device=q.device,dtype=t.float32);out=t.empty((1,1,h,128),device=q.device,dtype=t.bfloat16)
    score=t.empty((1,h,total),device=q.device,dtype=t.float32) if trace else m
    debug_v=t.empty((1,h,total,128),device=q.device,dtype=t.bfloat16) if trace else m
    _partial[(h,nb)](*pointers(g,kt),kt,q,vt,q,m,z,u,flags,score,*pointers(vg,vt),vt,debug_v,C=c,T=total,H=h,FULL=True,TRACE=trace,BT=16,num_warps=4,enable_fp_fusion=False)
    # Byte-frozen AF01 merge, identical for all three arms.
    af._merge[(h,)](m,z,u,flags,context,out,H=h,NB=nb,BP=1<<(nb-1).bit_length(),num_warps=4,enable_fp_fusion=False)
    if bool((flags!=0).any()):raise w.NumericalFailure('VF01 nonfinite fused intermediate/output')
    return out,context,score if trace else None,debug_v if trace else None

def pipeline(arm,data,trace=False):
    global LAST_STAGE
    import loop_kv_vf01_checks as checks
    g,kt,q,vg,vt=data
    LAST_STAGE='structural';checks.structural(g,kt,q,vg,vt)
    LAST_STAGE='input_finite';checks.finite(g,kt,q,vg,vt)
    if arm=='V_FUSED':
        LAST_STAGE='attention';out,ctx,score,v=launch(g,kt,q,vg,vt,trace=trace)
    elif arm in ('RF01_MATERIALIZE','TILE_MATERIALIZE'):
        LAST_STAGE='restore'
        v=qk.reference_k(vg,vt) if arm=='RF01_MATERIALIZE' else restore_v(vg,vt)
        af.validate_v(v,q,v.shape[2])
        LAST_STAGE='attention';out,ctx,score=af.launch(g,kt,q,v,full=True,trace=trace)
    else:raise w.IntegrityError('Unknown VF01 arm')
    LAST_STAGE='output'
    if not bool(w.torch.isfinite(out).all()):raise w.NumericalFailure('Nonfinite VF01 output')
    LAST_STAGE='returned'
    return (out,ctx,score,v) if trace else out
