"""Only integer 3-bit unpack is fused. Triton is loaded on demand in Colab.
No model, CUDA call, Triton import, or installation happens on import.
"""
import math,re,importlib.metadata
import loop_kv_pk01_cache as c
w=c.w
BLOCK=1024
WARPS=4
_kernel=None

def environment():
    import triton
    requirements=importlib.metadata.requires('torch') or []
    pinned=[(x,re.match(r'^triton\s*==\s*([^;\s]+)',x)) for x in requirements]
    pinned=[(x,m.group(1)) for x,m in pinned if m]
    w.require(len(pinned)==1,'Pinned torch must declare one exact Triton dependency')
    installed=importlib.metadata.version('triton')
    w.require(installed==pinned[0][1]==triton.__version__,'Triton differs from pinned torch dependency')
    return dict(triton=installed,torch_requirement=pinned[0][0],block=BLOCK,num_warps=WARPS,autotune=False)

def verify_environment(env):
    required=re.match(r'^triton\s*==\s*([^;\s]+)',env['torch_requirement'])
    w.require(required and env['triton']==required.group(1),'Recorded Triton dependency mismatch')
    w.require(env['block']==BLOCK and env['num_warps']==WARPS and env['autotune'] is False,'Kernel launch configuration changed')

def initialize():
    global _kernel,tl
    if _kernel is not None:return
    import triton
    import triton.language as tl
    @triton.jit
    def _unpack_kernel(Packed,Output,N,P,BLOCK:tl.constexpr):
        i=tl.program_id(0).to(tl.int64)*BLOCK+tl.arange(0,BLOCK).to(tl.int64)
        bit=i*3
        byte=bit//8
        shift=(bit%8).to(tl.uint32)
        lo=tl.load(Packed+byte,mask=(i<N)&(byte<P),other=0).to(tl.uint32)
        hi=tl.load(Packed+byte+1,mask=(i<N)&(byte+1<P),other=0).to(tl.uint32)
        code=((lo|(hi<<8))>>shift)&7
        tl.store(Output+i,code,mask=i<N)
    _kernel=_unpack_kernel

def unpack3(payload,shape):
    t=w.torch
    w.require(isinstance(shape,(tuple,list)) and all(isinstance(x,int) and x>0 for x in shape),'Fused positive integer shape')
    n=math.prod(shape)
    w.require(payload.dtype==t.uint8 and payload.is_contiguous() and payload.numel()==(n*3+7)//8,'Fused payload dtype/shape/layout')
    w.require(payload.device.type=='cuda','Fused unpack requires CUDA; no fallback')
    initialize()
    out=t.empty(tuple(shape),device=payload.device,dtype=t.uint8)
    _kernel[((n+BLOCK-1)//BLOCK,)](payload,out,n,payload.numel(),BLOCK=BLOCK,num_warps=WARPS)
    return out

def restore_group(g,dtype):
    b,h,n,_=g['codes'].shape
    return w.nsn_decode(unpack3(g['codes'],(b,h,n,64,128)),g['metas'],dtype).reshape(b,h,n*64,128)

def make_cache(base):
    parent=c.make_packed_cache(base)
    class FusedUnpackCache(parent):
        def materialize(self,slot,kv):
            tail=(self.key_cache if kv=='K' else self.value_cache)[slot]
            g=self.packed.get((slot,kv))
            if g is None:return tail
            return w.torch.cat((restore_group(g,tail.dtype),tail),-2)
    return FusedUnpackCache

LENGTHS=(1,2,3,7,8,9,63,64,65,1023,1024,1025,8192)
def fixture_specs():
    return ([dict(id=f'length-{n}',n=n,shape=[n]) for n in LENGTHS]+
            [dict(id='byte-pairs',n=65536*8,shape=[65536*8])]+
            [dict(id=f'real-blocks-{b}',n=16*b*64*128,shape=[1,16,b,64,128]) for b in (11,12)])

def fixtures():
    t=w.torch;np=w.np;records=[]
    for spec in fixture_specs():
        n=spec['n']
        if spec['id']=='byte-pairs':
            a=np.repeat(np.arange(256,dtype=np.uint8),256);b=np.tile(np.arange(256,dtype=np.uint8),256)
            packed=np.stack((a,b,a^b),-1).reshape(-1)
            gold=w.unpack_codes_reference(packed.tobytes(),n,3)
        else:
            gold=np.random.default_rng(2026100101+n).integers(0,8,n,dtype=np.uint8)
            packed=np.frombuffer(w.pack_codes(gold,3),np.uint8).copy()
        payload=t.from_numpy(packed.copy()).to('cuda');before=payload.clone()
        eager=c.unpack3(payload,tuple(spec['shape']));fused=unpack3(payload,tuple(spec['shape']))
        w.require(t.equal(eager,fused) and np.array_equal(fused.cpu().numpy().reshape(-1),gold),'GPU unpack fixture '+spec['id'])
        w.require(t.equal(payload,before),'Unpack mutated payload')
        records.append(dict(spec,status='PASS',device='cuda',reference='independent_bitstream',code_sha256=w.sha(gold.tobytes())))
    g=c.encode_group(t.zeros(1,1,1,64,128,device='cuda',dtype=t.bfloat16),0)
    eager=c.restore_group(g,t.bfloat16);fused=restore_group(g,t.bfloat16)
    w.require(t.equal(eager.view(t.uint8),fused.view(t.uint8)),'Fused zero reconstruction')
    g['metas'][0].fill_(float('nan'))
    for name,fn in [('EAGER',c.restore_group),('FUSED',restore_group)]:
        caught=False
        try:fn(g,t.bfloat16)
        except w.NumericalFailure:caught=True
        w.require(caught,'NaN was not rejected '+name)
    records.append(dict(id='restore-zero-nan',status='PASS',device='cuda',zero_byte_equal=True,nan_rejected=['EAGER','FUSED']))
    return records

def verify_fixtures(records):
    w.require(len(records)==len(fixture_specs())+1,'Kernel fixture count')
    for row,spec in zip(records,fixture_specs()):
        w.require(all(row.get(k)==v for k,v in spec.items()) and row['status']=='PASS' and row['device']=='cuda' and row['reference']=='independent_bitstream' and len(row['code_sha256'])==64,'Kernel fixture missing/failed')
    w.require(records[-1]==dict(id='restore-zero-nan',status='PASS',device='cuda',zero_byte_equal=True,nan_rejected=['EAGER','FUSED']),'Kernel nonfinite fixture')
