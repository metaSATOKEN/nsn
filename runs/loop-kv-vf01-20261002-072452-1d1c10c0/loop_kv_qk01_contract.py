"""Fixed QK01 data contract and NumPy audit helpers; no GPU/model imports."""
import hashlib,json,math,struct,zipfile
from pathlib import Path
import numpy as np
ARMS=('RF01_TORCH','RF01_TRITON','FUSED_QK')
QUERIES=('gaussian','one_hot','alternating','zero')
NORMAL=('zero','constant','mixed','packed_pattern','tail_only')
ABNORMAL=('n_nan','m_nan','s_nan','n_inf','q_nan','tail_inf','score_overflow')
SEED=2026100104
ROUNDS=3;WARMUP=5;REPEATS=20
SCALE=1/math.sqrt(128)
class ContractError(Exception):pass
def require(ok,message):
    if not ok:raise ContractError(message)
def sha(data):return hashlib.sha256(data).hexdigest()
def file_sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def read_json(path):return json.loads(Path(path).read_text())
def write_json(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def bf16_bits(a):
    a=np.asarray(a,dtype=np.float32);u=a.view(np.uint32)
    # Round-to-nearest-even, preserving NaNs as NaNs rather than infinity.
    rounded=((u+np.uint32(0x7fff)+((u>>16)&1))>>16).astype(np.uint16)
    return np.where(np.isnan(a),(u>>16).astype(np.uint16)|np.uint16(0x40),rounded).astype(np.uint16)
def from_bits(a):return (np.asarray(a,dtype=np.uint16).astype(np.uint32)<<16).view(np.float32)
def query(index,kind,h=16):
    shape=(1,h,1,128)
    if kind=='gaussian':a=np.random.default_rng(SEED+index).standard_normal(shape).astype(np.float32)
    elif kind=='one_hot':
        a=np.zeros(shape,np.float32)
        for head in range(h):a[0,head,0,head%128]=1
    elif kind=='alternating':a=np.broadcast_to(np.where(np.arange(128)%2==0,1.,-1.),shape).copy()
    elif kind=='zero':a=np.zeros(shape,np.float32)
    else:raise ContractError('Unknown query')
    return bf16_bits(a)
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

def load_npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def check_sample(a,spec):
    c=spec['compressed'];t=spec['length'];b=c//64
    expected={'codes':((1,16,b,3072),'uint8'),'headers':((1,16,b,12),'uint8'),
        'meta0':((1,16,b,64,1),'float16'),'meta1':((1,16,b,1,128),'float16'),
        'meta2':((1,16,b,1,128),'float16'),'tail':((1,16,t-c,128),'uint16'),
        'reference':((1,16,c,128),'uint16')}
    require(set(a)==set(expected),'NPZ field set')
    for name,(shape,dtype) in expected.items():
        require(a[name].shape==shape and str(a[name].dtype)==dtype,'NPZ shape/type '+name)
    for i in range(b):
        expected_header=np.frombuffer(struct.pack('<QHBB',i*64,64,3,0),np.uint8)
        require(np.all(a['headers'][0,:,i]==expected_header),'Block header')
    full=np.concatenate([a['reference'],a['tail']],axis=-2)
    digest=sha(str((tuple(full.shape),'torch.bfloat16')).encode()+full.tobytes())
    require(digest==spec['cache_digest'],'RF01 cache digest')
    for key in ('meta0','meta1','meta2'):require(np.isfinite(a[key]).all(),'Nonfinite metadata')
    require(np.isfinite(from_bits(full)).all(),'Nonfinite K/tail')
    return a

def extract_inputs(parent,dest,index):
    require(file_sha(parent)==index['parent_zip_sha256'],'Parent ZIP hash')
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(parent) as z:
        names=z.namelist();require(len(names)==len(set(names)),'Duplicate ZIP names')
        for s in index['samples']:
            matches=[n for n in names if n==s['file'] or n.endswith('/'+s['file'])]
            require(len(matches)==1,'Missing/ambiguous selected input')
            info=z.getinfo(matches[0]);require(info.file_size<8*1024**2,'Input too large')
            data=z.read(matches[0]);require(sha(data)==s['sha256'],'Input NPZ hash')
            p=dest/(s['id']+'.npz');p.write_bytes(data);check_sample(load_npz(p),s)

def schedule(index):
    rows=[]
    for si,s in enumerate(index['samples']):
        for qi,q in enumerate(QUERIES):
            ci=si*len(QUERIES)+qi
            for rnd in range(ROUNDS):
                offset=(ci+rnd)%3
                for arm in ARMS[offset:]+ARMS[:offset]:rows.append(dict(sample=s['id'],query=q,round=rnd,arm=arm))
    return rows

def aggregate(timings,index):
    # Do not turn a slow result into incomplete or use repeated queries as independent N.
    summaries={}
    for field in ('cuda_ms','wall_ms'):
        case={}
        for s in index['samples']:
            for q in QUERIES:
                for arm in ARMS:
                    rr=[x[field] for x in timings if x['sample']==s['id'] and x['query']==q and x['arm']==arm]
                    require(len(rr)==ROUNDS and all(math.isfinite(v) and v>0 for v in rr),'Missing/invalid timing')
                    case[(s['id'],q,arm)]=float(np.median(rr))
        rows=[]
        for s in index['samples']:
            qr={q:{a:case[s['id'],q,'FUSED_QK']/case[s['id'],q,a] for a in ARMS[:2]} for q in QUERIES}
            rows.append(dict(sample=s['id'],phase=s['phase'],query_ratios=qr,
                nonzero_query_median={a:float(np.median([qr[q][a] for q in QUERIES[:-1]])) for a in ARMS[:2]}))
        main={}
        for arm in ARMS[:2]:
            values=[x['nonzero_query_median'][arm] for x in rows if x['phase']=='main']
            main[arm]=dict(n_samples=len(values),median=float(np.median(values)),min=min(values),max=max(values),faster_count=sum(v<1 for v in values))
        summaries[field]=dict(main=main,samples=rows)
    return summaries

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

def fixture(name):
    # Small fixed arrays, independent of model code. One packed block and a 65-token tail.
    h=2;n=np.ones((1,h,1,64,1),np.float16);m=np.zeros((1,h,1,1,128),np.float16);s=np.ones_like(m)
    codes=(np.arange(h*64*128)%8).astype(np.uint8).reshape(1,h,1,64,128)
    tail=bf16_bits(np.linspace(-.5,.5,h*65*128,dtype=np.float32).reshape(1,h,65,128))
    q=query(0,'alternating',h)
    if name=='zero':n.fill(0);tail.fill(0)
    elif name=='constant':n.fill(4);m.fill(1);s.fill(0);tail=bf16_bits(np.full(tail.shape,.5,np.float32));q=query(0,'one_hot',h)
    elif name=='mixed':m[:]=np.linspace(-.03,.03,128,dtype=np.float16);s.fill(.007);n[:]=np.linspace(0,.05,h*64).reshape(n.shape)
    elif name=='packed_pattern':m.fill(-3);s.fill(1)
    elif name=='tail_only':return {'tail':tail},q
    elif name=='score_overflow':return {'tail':bf16_bits(np.full((1,h,17,128),2**120,np.float32))},bf16_bits(np.full((1,h,1,128),2**120,np.float32))
    elif name not in ABNORMAL:raise ContractError('Unknown fixture')
    a=dict(codes=pack3(codes),meta0=n,meta1=m,meta2=s,tail=tail)
    for label,key in [('n_nan','meta0'),('m_nan','meta1'),('s_nan','meta2')]:
        if name==label:a[key].flat[0]=np.nan
    if name=='n_inf':a['meta0'].flat[0]=np.inf
    if name=='q_nan':q.flat[0]=0x7fc0
    if name=='tail_inf':tail.flat[0]=0x7f80
    return a,q

def fingerprints(a):return {k:sha(v.tobytes()) for k,v in a.items()}
def check_sources(folder,index):
    for name,digest in index['frozen_sources'].items():require(file_sha(Path(folder)/name)==digest,'Frozen source changed: '+name)
