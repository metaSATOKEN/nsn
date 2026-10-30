"""NB06 structural GQA checks plus unchanged VF01 grouped finite scan."""
import loop_kv_nb06_primitives as w
BLOCK=1024;WARPS=4;_kernel=None;launch_count=0
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

def structural(g,kt,q,vg,vt):
    t=w.torch
    w.require(q.dtype==t.bfloat16 and tuple(q.shape)==(1,48,1,128) and q.is_contiguous() and q.device.type=='cuda','Q layout/device')
    dims=[]
    for group,tail in ((g,kt),(vg,vt)):
        w.require(tail.dtype==t.bfloat16 and tail.ndim==4 and tuple(tail.shape[:2])==(1,8) and tail.shape[-1]==128 and tail.device==q.device and tail.is_contiguous(),'Tail layout/device')
        cc=0
        if group is not None:
            codes=group['codes'];w.require(codes.dtype==t.uint8 and codes.ndim==4 and tuple(codes.shape[:2])==(1,8) and codes.shape[-1]==3072 and codes.shape[2]>0,'Codes layout')
            n=codes.shape[2];cc=n*64
            specs=[(codes,(1,8,n,3072),t.uint8),(group['headers'],(1,8,n,12),t.uint8)]
            specs.extend(zip(group['metas'],[(1,8,n,64,1),(1,8,n,1,128),(1,8,n,1,128)],[t.float16]*3))
            w.require(len(group['metas'])==3,'Metadata count')
            for x,shape,dtype in specs:w.require(tuple(x.shape)==shape and x.dtype==dtype and x.is_contiguous() and x.device==q.device,'Packed tensor layout/device')
        dims.append((cc,cc+tail.shape[2]))
    w.require(dims[0]==dims[1] and dims[0][1]>0,'K/V lengths differ')
    return dims[0][1]
