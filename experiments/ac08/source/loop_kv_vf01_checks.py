"""IV01 grouped finite checker extended to both raw packed inputs. Lazy GPU."""
import loop_kv_iv01_kernel as iv
w=iv.w;BLOCK=1024;WARPS=4;_kernel=None;launch_count=0

def structural(g,kt,q,vg,vt):
    iv.prepare_shapes();kh,kc,ktotal=iv._qshape(g,kt,q);vh,vc,vtotal=iv._qshape(vg,vt,q)
    w.require((kh,kc,ktotal)==(vh,vc,vtotal),'K/V lengths differ')
    for group in (g,vg):
        if group is not None:
            h=group.get('headers');b,nh,n,_=group['codes'].shape
            w.require(h is not None and h.dtype==w.torch.uint8 and h.is_contiguous() and tuple(h.shape)==(b,nh,n,12) and h.device==q.device,'Header layout')
    return ktotal

def initialize():
    global _kernel,tl
    if _kernel is not None:return
    import triton
    import triton.language as tl
    @triton.jit
    def finite_kernel(Q,KT,KN,KM,KS,VT,VN,VM,VS,Flags,NQ:tl.constexpr,NKT:tl.constexpr,NKN:tl.constexpr,NKM:tl.constexpr,NKS:tl.constexpr,NVT:tl.constexpr,NVN:tl.constexpr,NVM:tl.constexpr,NVS:tl.constexpr,BT:tl.constexpr):
        pid=tl.program_id(0);r=tl.arange(0,BT)
        BQ:tl.constexpr=(NQ+BT-1)//BT
        BKT:tl.constexpr=(NKT+BT-1)//BT
        BKN:tl.constexpr=(NKN+BT-1)//BT
        BKM:tl.constexpr=(NKM+BT-1)//BT
        BKS:tl.constexpr=(NKS+BT-1)//BT
        BVT:tl.constexpr=(NVT+BT-1)//BT
        BVN:tl.constexpr=(NVN+BT-1)//BT
        BVM:tl.constexpr=(NVM+BT-1)//BT
        BVS:tl.constexpr=(NVS+BT-1)//BT
        if pid<BQ:
            i=(pid-(0))*BT+r;x=tl.load(Q+i,mask=i<NQ,other=0).to(tl.float32)
        elif pid<BQ+BKT:
            i=(pid-(BQ))*BT+r;x=tl.load(KT+i,mask=i<NKT,other=0).to(tl.float32)
        elif pid<BQ+BKT+BKN:
            i=(pid-(BQ+BKT))*BT+r;x=tl.load(KN+i,mask=i<NKN,other=0).to(tl.float32)
        elif pid<BQ+BKT+BKN+BKM:
            i=(pid-(BQ+BKT+BKN))*BT+r;x=tl.load(KM+i,mask=i<NKM,other=0).to(tl.float32)
        elif pid<BQ+BKT+BKN+BKM+BKS:
            i=(pid-(BQ+BKT+BKN+BKM))*BT+r;x=tl.load(KS+i,mask=i<NKS,other=0).to(tl.float32)
        elif pid<BQ+BKT+BKN+BKM+BKS+BVT:
            i=(pid-(BQ+BKT+BKN+BKM+BKS))*BT+r;x=tl.load(VT+i,mask=i<NVT,other=0).to(tl.float32)
        elif pid<BQ+BKT+BKN+BKM+BKS+BVT+BVN:
            i=(pid-(BQ+BKT+BKN+BKM+BKS+BVT))*BT+r;x=tl.load(VN+i,mask=i<NVN,other=0).to(tl.float32)
        elif pid<BQ+BKT+BKN+BKM+BKS+BVT+BVN+BVM:
            i=(pid-(BQ+BKT+BKN+BKM+BKS+BVT+BVN))*BT+r;x=tl.load(VM+i,mask=i<NVM,other=0).to(tl.float32)
        elif pid<BQ+BKT+BKN+BKM+BKS+BVT+BVN+BVM+BVS:
            i=(pid-(BQ+BKT+BKN+BKM+BKS+BVT+BVN+BVM))*BT+r;x=tl.load(VS+i,mask=i<NVS,other=0).to(tl.float32)
        else:
            x=tl.full((BT,),0,tl.float32)
        bad=tl.sum(((x!=x)|(tl.abs(x)==float('inf'))).to(tl.int32),axis=0)>0
        tl.store(Flags+pid,bad.to(tl.int32))
    _kernel=finite_kernel

def finite(g,kt,q,vg,vt):
    global launch_count
    t=w.torch;initialize()
    km=[q,q,q] if g is None else list(g['metas']);vm=[q,q,q] if vg is None else list(vg['metas'])
    arrays=[q,kt,*km,vt,*vm]
    sizes=[q.numel(),kt.numel(),*([0,0,0] if g is None else [x.numel() for x in km]),vt.numel(),*([0,0,0] if vg is None else [x.numel() for x in vm])]
    blocks=sum((n+BLOCK-1)//BLOCK for n in sizes);flags=t.empty((blocks,),device=q.device,dtype=t.int32);launch_count+=1
    _kernel[(blocks,)](*arrays,flags,*sizes,BT=BLOCK,num_warps=WARPS,enable_fp_fusion=False)
    if bool((flags!=0).any()):raise w.NumericalFailure('VF01 nonfinite raw input')
