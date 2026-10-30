"""AF01 immutable input/schedule contract and CPU-only arithmetic."""
import hashlib,json,math,struct,zipfile
from pathlib import Path
import numpy as np
import loop_kv_qk01_contract as qc
ARMS=('SDPA','SEPARATE','FINISH_FUSED','FULL_FUSED');ROUNDS=5;WARMUP=5;REPEATS=20
class ContractError(ValueError):pass
def require(ok,msg):
    if not ok:raise ContractError(msg)
def sha(b):return hashlib.sha256(b).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines()] if Path(p).exists() else []
def append(p,obj):
    with Path(p).open('a') as f:f.write(json.dumps(obj,allow_nan=False)+'\n');f.flush()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {n:z[n] for n in z.files}
def index():return read(Path(__file__).with_name('loop_kv_af01_inputs.json'))
def array_hash(a):return sha(str((a.shape,str(a.dtype))).encode()+a.tobytes())
def order(i,r=0):
    n=(i//3+i+r)%4;return ARMS[n:]+ARMS[:n]
def cells(idx):return [dict(sample=s['id'],arm=a,length=s['length'],compressed=s['compressed']) for i,s in enumerate(idx['samples']) for a in order(i)]
def signature(cell):return f"{cell['arm']}-C{cell['compressed']}-T{cell['length']}-B1H16D128-BF16"
def schedule(idx):return [dict(sample=s['id'],arm=a,round=r) for r in range(ROUNDS) for i,s in enumerate(idx['samples']) for a in order(i,r)]
def validate_score(x,length):
    require(x.dtype==np.float32,'Softmax input dtype');require(x.shape==(1,16,length),'Softmax input shape');require(np.isfinite(x).all(),'Nonfinite softmax input')
def oracle(x):
    x=np.asarray(x,np.float64);p=np.exp(x-x.max(-1,keepdims=True));return p/p.sum(-1,keepdims=True)
def difference(x,y):
    require(x.shape==y.shape and np.isfinite(x).all() and np.isfinite(y).all(),'Difference input')
    d=x.astype(np.float64)-y.astype(np.float64)
    return dict(bit_equal=x.dtype==y.dtype and x.tobytes()==y.tobytes(),max_abs=float(np.abs(d).max()),rmse=float(np.sqrt(np.mean(d*d))))
def sample(root,sp):
    p=Path(root)/'inputs'/(sp['id']+'.npz');require(sha(p.read_bytes())==sp['sha256'],'Parent NPZ hash')
    a=load(p);t=sp['length'];comp=sp['compressed']
    require(a['q'].shape==(1,16,1,128) and a['q'].dtype==np.uint16,'Actual query shape')
    require(a['v'].shape==a['baseline_k'].shape==(1,16,t,128),'Actual KV shape')
    require(a['v'].dtype==a['baseline_k'].dtype==np.uint16 and np.isfinite(qc.from_bits(a['v'])).all(),'Actual V finite/dtype')
    require(np.isfinite(qc.from_bits(a['q'])).all(),'Actual query finite')
    spec=dict(length=t,compressed=comp,cache_digest=sha(str(((1,16,t,128),'torch.bfloat16')).encode()+a['baseline_k'].tobytes()))
    qc.check_sample({**{n:a[n] for n in ('codes','headers','meta0','meta1','meta2','tail')},'reference':a['baseline_k'][:,:,:comp].copy()},spec)
    return a
def extract(parent,root,idx):
    require(sha(Path(parent).read_bytes())==idx['parent_sha256'],'Wrong AC02 ZIP hash')
    with zipfile.ZipFile(parent) as z:
        require(len(z.namelist())==len(set(z.namelist())),'Duplicate parent members')
        dest=Path(root)/'inputs';dest.mkdir(exist_ok=True)
        for sp in idx['samples']:
            info=z.getinfo(sp['file']);require(info.file_size==sp['bytes'] and info.file_size<20*1024**2,'Parent member size')
            b=z.read(sp['file']);require(sha(b)==sp['sha256'],'Parent member hash');(dest/(sp['id']+'.npz')).write_bytes(b);sample(root,sp)
def equal(a,b,label):
    if isinstance(b,dict):
        require(isinstance(a,dict) and set(a)==set(b),'Keys '+label)
        for k in b:equal(a[k],b[k],label+'.'+k)
    elif isinstance(b,list):
        require(isinstance(a,list) and len(a)==len(b),'Length '+label)
        for x,y in zip(a,b):equal(x,y,label)
    elif isinstance(b,float):require(isinstance(a,(int,float)) and math.isfinite(a) and math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12),'Float '+label)
    else:require(type(a)==type(b) and a==b,'Value '+label)


def bf16_bits(x):
    u=np.asarray(x,np.float32).view(np.uint32);return ((u+0x7fff+((u>>16)&1))>>16).astype(np.uint16)

def context_metrics(score,vbits,ctx):
    require(score.dtype==np.float32 and ctx.dtype==np.float32 and vbits.dtype==np.uint16,'Context audit dtype')
    v=qc.from_bits(vbits).astype(np.float64);p=oracle(score);prod=p[...,None]*v
    gold=prod.sum(-2,keepdims=True);scale=np.abs(prod).sum(-2,keepdims=True)
    require(ctx.shape==gold.shape and np.isfinite(ctx).all(),'Context audit shape/finite')
    e=ctx.astype(np.float64)-gold;bad=int((np.abs(e)>2e-6+2e-6*scale).sum())
    return dict(pass_budget=bad==0,elements=int(ctx.size),exceed=bad,max_abs=float(np.abs(e).max()),rmse=float(np.sqrt(np.mean(e*e))))

def fixtures():
    return [dict(id=f'finish-T{n}-{p}',kind='finish',length=n,pattern=p) for n in (17,769,833) for p in ('uniform','dominant','cancel','zero')]+[dict(id=f'full-T{n}',kind='full',length=n,pattern='native') for n in (17,65,769,833)]

def fixture_data(sp):
    length=sp['length'];h=2
    q=(np.sin(np.arange(h*128).reshape(1,h,1,128)*.03)*.2).astype(np.float32)
    k=(np.cos(np.arange(h*length*128).reshape(1,h,length,128)*.007)*.5).astype(np.float32)
    v=(np.sin(np.arange(h*length*128).reshape(1,h,length,128)*.011)*.25).astype(np.float32)
    scores=np.zeros((1,h,length),np.float32)
    if sp['pattern']=='dominant':scores[:,:,length//2]=np.float32(math.log((length-1)*.9999/.0001))
    elif sp['pattern']=='cancel':
        v=np.broadcast_to(np.where(np.arange(length)[None,None,:,None]%2==0,1.,-1.),v.shape).copy().astype(np.float32)
    elif sp['pattern']=='zero':v[:]=0
    return dict(q=bf16_bits(q),tail=bf16_bits(k),v=bf16_bits(v),score=scores)

NEGATIVE=('q_nan','v_nan','score_nan','score_dtype','score_shape','empty_tail')
