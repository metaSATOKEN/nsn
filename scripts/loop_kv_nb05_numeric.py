"""NB03 numerical budgets copied unchanged; NumPy only."""
import numpy as np
import math,struct
from loop_kv_nb05_contract import require
SCALE=1/math.sqrt(128)
ARMS=("TORCH_SDPA","TILE_SDPA","GQA_FUSED")
qc=__import__(__name__)

def bf16_bits(a):
    a=np.asarray(a,dtype=np.float32);u=a.view(np.uint32)
    # Round-to-nearest-even, preserving NaNs as NaNs rather than infinity.
    rounded=((u+np.uint32(0x7fff)+((u>>16)&1))>>16).astype(np.uint16)
    return np.where(np.isnan(a),(u>>16).astype(np.uint16)|np.uint16(0x40),rounded).astype(np.uint16)

def from_bits(a):return (np.asarray(a,dtype=np.uint16).astype(np.uint32)<<16).view(np.float32)

def gold(kbits,qbits):
    k=from_bits(kbits).astype(np.float64);q=from_bits(qbits).astype(np.float64)
    prod=k*q
    return np.sum(prod,axis=-1)*SCALE,np.sum(np.abs(prod),axis=-1)*SCALE

def score_metrics(score,reference,abs_products):
    require(score.shape==reference.shape==abs_products.shape,'Score shape')
    require(np.isfinite(score).all() and np.isfinite(reference).all(),'Nonfinite saved score')
    err=np.asarray(score,np.float64)-reference;limit=1e-6+2**-7*abs_products
    rmse=np.sqrt(np.mean(err**2,axis=-1));scale=np.sqrt(np.mean(reference**2,axis=-1))
    bad=int(np.count_nonzero(np.abs(err)>limit));heads=int(np.count_nonzero(rmse>1e-6+2**-8*scale))
    return dict(pass_budget=bad==heads==0,max_abs=float(np.max(np.abs(err))),rmse=float(np.sqrt(np.mean(err**2))),
                element_exceed=bad,head_exceed=heads,elements=int(score.size))

def k_metrics(bits,reference):
    require(bits.dtype==reference.dtype==np.uint16 and bits.shape==reference.shape,'K shape/type')
    a=from_bits(bits).astype(np.float64);b=from_bits(reference).astype(np.float64)
    require(np.isfinite(a).all() and np.isfinite(b).all(),'Nonfinite saved K')
    err=a-b;bad=int(np.count_nonzero(np.max(np.abs(err),axis=-1)>1e-6+2**-7*np.max(np.abs(b),axis=-1)))
    heads=int(np.count_nonzero(np.sqrt(np.mean(err**2,axis=(-2,-1)))>1e-6+2**-8*np.sqrt(np.mean(b**2,axis=(-2,-1)))))
    return dict(pass_budget=bad==heads==0,bit_differences=int(np.count_nonzero(bits!=reference)),elements=int(bits.size),
                max_abs=float(np.max(np.abs(err))),rmse=float(np.sqrt(np.mean(err**2))),token_exceed=bad,head_exceed=heads)

def pack3(codes):
    flat=np.asarray(codes,np.uint8).reshape(-1);require(len(flat)%8==0,'Pack length')
    bits=((flat[:,None]>>np.arange(3))&1).reshape(-1)
    return np.packbits(bits,bitorder='little').reshape(*codes.shape[:3],3072)

def unpack3(a):
    bits=np.unpackbits(a['codes'].reshape(-1),bitorder='little').reshape(-1,3)
    codes=(bits@np.array([1,2,4],np.uint8)).reshape(*a['codes'].shape[:3],64,128)
    return codes

def cpu_restore(a):
    if 'codes' not in a:return a['tail'].copy()
    codes=unpack3(a).astype(np.float32)
    direction=a['meta1'].astype(np.float32)+a['meta2'].astype(np.float32)*codes
    den=np.maximum(np.sqrt(np.sum(direction*direction,axis=-1,keepdims=True)),np.float32(1e-12))
    n=a['meta0'].astype(np.float32)
    out=np.where(n==0,np.float32(0),n*(direction/den))
    prefix=bf16_bits(out).reshape(1,a['codes'].shape[1],-1,128)
    return np.concatenate([prefix,a['tail']],axis=-2)

def oracle(x):
    x=np.asarray(x,np.float64);p=np.exp(x-x.max(-1,keepdims=True));return p/p.sum(-1,keepdims=True)

def difference(x,y):
    require(x.shape==y.shape and np.isfinite(x).all() and np.isfinite(y).all(),'Difference input')
    d=x.astype(np.float64)-y.astype(np.float64)
    return dict(bit_equal=x.dtype==y.dtype and x.tobytes()==y.tobytes(),max_abs=float(np.abs(d).max()),rmse=float(np.sqrt(np.mean(d*d))))

def context_metrics(score,vbits,ctx):
    require(score.dtype==np.float32 and ctx.dtype==np.float32 and vbits.dtype==np.uint16,'Context audit dtype')
    v=qc.from_bits(vbits).astype(np.float64);p=oracle(score);prod=p[...,None]*v
    gold=prod.sum(-2,keepdims=True);scale=np.abs(prod).sum(-2,keepdims=True)
    require(ctx.shape==gold.shape and np.isfinite(ctx).all(),'Context audit shape/finite')
    e=ctx.astype(np.float64)-gold;bad=int((np.abs(e)>2e-6+2e-6*scale).sum())
    return dict(pass_budget=bad==0,elements=int(ctx.size),exceed=bad,max_abs=float(np.abs(e).max()),rmse=float(np.sqrt(np.mean(e*e))))

def structure(a,check_finite=False):
    require('q' in a and a['q'].shape==(1,48,1,128) and a['q'].dtype==np.uint16,'Q shape/type')
    totals=[]
    for side in ('k','v'):
        tail=a[side+'_tail'];require(tail.ndim==4 and tail.shape[:2]==(1,8) and tail.shape[-1]==128 and tail.dtype==np.uint16,'Tail shape/type')
        cc=0
        if side+'_codes' in a:
            codes=a[side+'_codes'];require(codes.ndim==4 and codes.shape[:2]==(1,8) and codes.shape[-1]==3072 and codes.shape[2]>0 and codes.dtype==np.uint8,'Codes shape/type')
            n=codes.shape[2];cc=n*64
            specs={'headers':((1,8,n,12),np.uint8),'meta0':((1,8,n,64,1),np.float16),'meta1':((1,8,n,1,128),np.float16),'meta2':((1,8,n,1,128),np.float16)}
            for key,(shape,dtype) in specs.items():
                x=a[side+'_'+key];require(x.shape==shape and x.dtype==dtype,'Packed layout '+side+key)
            for i in range(n):
                header=np.frombuffer(struct.pack('<QHBB',i*64,64,3,0),np.uint8)
                require(np.all(a[side+'_headers'][0,:,i]==header),'Header offset/count/bits')
        else:require(not any(side+'_'+n in a for n in ('headers','meta0','meta1','meta2')),'Orphan packed metadata')
        totals.append((cc,cc+tail.shape[2]))
    require(totals[0]==totals[1] and totals[0][1]>0,'K/V length')
    if check_finite:
        for n,x in a.items():
            v=from_bits(x) if n=='q' or n.endswith('_tail') else x
            require(np.isfinite(v).all(),'Nonfinite input '+n)
    return totals[0]

def sdpa_metrics(out,kbits,vbits,qbits):
    score,_=gold(np.repeat(kbits,6,axis=1),qbits)
    prod=oracle(score)[...,None]*from_bits(np.repeat(vbits,6,axis=1)).astype(np.float64)
    ref=prod.sum(-2,keepdims=True);scale=np.abs(prod).sum(-2,keepdims=True)
    actual=from_bits(out).transpose(0,2,1,3).astype(np.float64)
    require(actual.shape==ref.shape and np.isfinite(actual).all(),'SDPA output shape/finite')
    err=actual-ref;bad=int(np.count_nonzero(np.abs(err)>1e-6+2**-7*scale))
    heads=int(np.count_nonzero(np.sqrt(np.mean(err*err,axis=(-2,-1)))>1e-6+2**-8*np.sqrt(np.mean(scale*scale,axis=(-2,-1)))))
    return dict(pass_budget=bad==heads==0,max_abs=float(np.abs(err).max()),element_exceed=bad,head_exceed=heads)

