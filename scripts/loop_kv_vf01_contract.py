"""VF01 frozen plan, data contract and independent NumPy arithmetic. No torch."""
import hashlib,json,math,struct,zipfile
from pathlib import Path
from collections import Counter
import numpy as np
import loop_kv_qk01_contract as qc
import loop_kv_af01_contract as fc
require=fc.require;read=fc.read;write=fc.write;rows=fc.rows;append=fc.append;load=fc.load;sha=fc.sha
PLAN=read(Path(__file__).with_name('loop_kv_vf01_inputs.json'))
CAPS=PLAN['caps_seconds'];ARMS=tuple(PLAN['arms']);QUERIES=tuple(PLAN['query']['kinds'])
SAMPLES=PLAN['samples'];ROUNDS=5;WARMUP=5;REPEATS=20

def file_sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def arrays_hash(arrays):return {n:fc.array_hash(a) for n,a in arrays.items()}
def first_cells():
    return [dict(sample=s['id'],query=q,arm=a) for i,s in enumerate(SAMPLES) for j,q in enumerate(QUERIES) for a in ARMS[(i*4+j)%3:]+ARMS[:(i*4+j)%3]]
def schedule():return PLAN['timing']['host_schedule']
def cases():return [dict(sample=s['id'],query=q) for s in SAMPLES for q in QUERIES]
def name(row):return row['sample']+'-'+row['query']
def signature(row):
    sp=next(x for x in SAMPLES if x['id']==row['sample'])
    return f"{row['arm']}-C{sp['compressed']}-T{sp['length']}-B1H16D128"
def manifest():return read(Path(__file__).with_name('vf01_material_manifest.json'))

def extract(parent,root):
    m=manifest();require(file_sha(parent)==m['sha256'],'Wrong material ZIP hash')
    target=Path(root)/'inputs';target.mkdir(exist_ok=True)
    with zipfile.ZipFile(parent) as z:
        require(len(z.namelist())==len(set(z.namelist())) and set(z.namelist())==set(m['entries']),'Material member set')
        for n,sp in m['entries'].items():
            info=z.getinfo(n);require(info.file_size==sp['bytes'] and info.file_size<8*1024**2,'Material member size')
            b=z.read(n);require(sha(b)==sp['sha256'],'Material member hash');(target/n).write_bytes(b)
    for sp in SAMPLES:sample(root,sp)
    write(Path(root)/'input_manifest.json',m)

def sample(root,sp):
    result={}
    for side in ('K','V'):
        p=Path(root)/'inputs'/(sp['id']+'-'+side+'.npz')
        require(file_sha(p)==sp[side]['sha256'],'Raw input hash')
        a=load(p);qc.check_sample(a,{**sp,'cache_digest':sp[side]['cache_digest']})
        for n,x in a.items():result[side.lower()+'_'+n]=x
    return result

def with_query(a,i,kind):
    out={n:x for n,x in a.items() if not n.endswith('_reference')};out['q']=qc.query(i,kind)
    return out

def cpu_structure(a):
    """Header contents are checked before transfer; finite fields checked on GPU too."""
    q=a['q'];require(q.dtype==np.uint16 and q.ndim==4 and q.shape[0]==1 and q.shape[2:]==(1,128),'Query layout')
    h=q.shape[1];require(h>0,'Empty head count');coords=[]
    want={'q'}
    for side in ('k','v'):
        tail=a[side+'_tail'];want.add(side+'_tail')
        require(tail.dtype==np.uint16 and tail.ndim==4 and tail.shape[:2]==(1,h) and tail.shape[-1]==128,'Tail layout')
        c=0
        if side+'_codes' in a:
            codes=a[side+'_codes'];require(codes.dtype==np.uint8 and codes.ndim==4 and codes.shape[:2]==(1,h) and codes.shape[2]>0 and codes.shape[3]==3072,'Codes layout')
            n=codes.shape[2];c=n*64
            for key,shape,dtype in [('headers',(1,h,n,12),np.uint8),('meta0',(1,h,n,64,1),np.float16),('meta1',(1,h,n,1,128),np.float16),('meta2',(1,h,n,1,128),np.float16)]:
                x=a[side+'_'+key];require(x.shape==shape and x.dtype==dtype,'Metadata/header layout');want.add(side+'_'+key)
            want.add(side+'_codes')
            for i in range(n):
                header=np.frombuffer(struct.pack('<QHBB',i*64,64,3,0),np.uint8)
                require(np.all(a[side+'_headers'][0,:,i]==header),'Header positions/format')
        total=c+tail.shape[2];require(total>0,'Empty sequence');coords.append((c,total))
    require(coords[0]==coords[1],'K/V length mismatch');require(set(a)==want,'Unexpected input fields')
    return h,*coords[0]

def finite_input(a):
    for n,x in a.items():
        if n=='q' or n.endswith('_tail') or '_meta' in n:
            require(np.isfinite(qc.from_bits(x) if x.dtype==np.uint16 else x).all(),'Nonfinite '+n)
    return True

def headers(h,c):
    return np.frombuffer(b''.join(struct.pack('<QHBB',i*64,64,3,0) for _ in range(h) for i in range(c//64)),np.uint8).copy().reshape(1,h,c//64,12)

def fixture(sp):
    # Extends QK01's deterministic packed-code fixture, without LLM outputs or encoding.
    h=2;c=sp['compressed'];total=sp['length'];pattern=sp['pattern'];result={'q':qc.query(0,'one_hot' if pattern=='constant' else 'alternating',h)}
    for side in ('k','v'):
        tail=qc.bf16_bits(np.linspace(-.5,.5,h*(total-c)*128,dtype=np.float32).reshape(1,h,total-c,128))
        if pattern=='zero':tail.fill(0)
        if pattern=='constant':tail=qc.bf16_bits(np.full(tail.shape,.5,np.float32))
        result[side+'_tail']=tail
        if c:
            codes=(np.arange(h*c*128)%8).astype(np.uint8).reshape(1,h,c//64,64,128)
            n=np.ones((1,h,c//64,64,1),np.float16);m=np.full((1,h,c//64,1,128),-3,np.float16);s=np.ones_like(m)
            if pattern=='zero':n.fill(0)
            if pattern=='constant':n.fill(4);m.fill(1);s.fill(0)
            if pattern=='partial_zero_norm':n.reshape(-1)[::2]=0
            result.update({side+'_codes':qc.pack3(codes),side+'_headers':headers(h,c),side+'_meta0':n,side+'_meta1':m,side+'_meta2':s})
    return result

def negative(kind):
    a=fixture(dict(compressed=64,length=65,pattern='mixed_codes'));mutation=None
    nf={'q_nan':('q',np.nan),'q_inf':('q',np.inf),'k_tail_nan':('k_tail',np.nan),'v_tail_inf':('v_tail',np.inf),
        'k_norm_nan':('k_meta0',np.nan),'k_min_nan':('k_meta1',np.nan),'k_scale_inf':('k_meta2',np.inf),
        'v_norm_nan':('v_meta0',np.nan),'v_min_nan':('v_meta1',np.nan),'v_scale_nan':('v_meta2',np.nan),'v_norm_inf':('v_meta0',np.inf)}
    if kind in nf:
        key,val=nf[kind];pos=-1 if kind in ('v_norm_nan','v_tail_inf') else 0
        a[key].flat[pos]=qc.bf16_bits(np.array(val,np.float32)).item() if a[key].dtype==np.uint16 else val
    elif kind=='v_zero_norm_nan_direction':a['v_meta0'].fill(0);a['v_meta1'].flat[0]=np.nan
    elif kind=='q_dtype':a['q']=qc.from_bits(a['q'])
    elif kind=='q_shape':a['q']=a['q'][...,:127].copy()
    elif kind in ('k_codes_dtype','v_codes_dtype'):key=kind[:1]+'_codes';a[key]=a[key].astype(np.int16)
    elif kind=='v_codes_short':a['v_codes']=a['v_codes'][...,:-1].copy()
    elif kind=='v_meta_shape':a['v_meta1']=a['v_meta1'][...,:127].copy()
    elif kind=='v_tail_noncontiguous':mutation='v_tail_noncontiguous'
    elif kind=='kv_length_mismatch':a['v_tail']=np.concatenate([a['v_tail'],a['v_tail']],2)
    elif kind in ('k_header_position','v_header_position'):a[kind[0]+'_headers'].flat[0]=1
    elif kind=='empty_sequence':a={'q':a['q'],'k_tail':a['k_tail'][:,:,:0].copy(),'v_tail':a['v_tail'][:,:,:0].copy()}
    elif kind=='score_overflow':
        a=fixture(dict(compressed=0,length=17,pattern='tail_only'))
        for n in ('q','k_tail'):a[n]=qc.bf16_bits(np.full(a[n].shape,2**120,np.float32))
    else:raise ValueError('Unknown negative '+kind)
    return a,mutation

def check_query(bits,sp,kind):
    i=SAMPLES.index(sp);want=qc.query(i,kind)
    require(np.array_equal(bits,want) and sha(bits.tobytes())==sp['query_hashes'][kind],'Query changed')

def aggregate(timings):
    result={};pairs=[('V_FUSED','TILE_MATERIALIZE'),('V_FUSED','RF01_MATERIALIZE'),('TILE_MATERIALIZE','RF01_MATERIALIZE')]
    for field in ('cuda_ms','wall_ms'):
        med={}
        for cell in first_cells():
            rr=[r[field] for r in timings if all(r[k]==cell[k] for k in ('sample','query','arm'))]
            require(len(rr)==5 and all(math.isfinite(v) and v>0 for v in rr),'Timing inventory')
            med[(cell['sample'],cell['query'],cell['arm'])]=float(np.median(rr))
        outputs={}
        for x,y in pairs:
            samples=[]
            for sp in SAMPLES:
                ratios={q:med[(sp['id'],q,x)]/med[(sp['id'],q,y)] for q in QUERIES}
                samples.append(dict(sample=sp['id'],book_id=sp['book_id'],phase=sp['phase'],slot=sp['slot'],query_ratios=ratios,nonzero_query_median=float(np.median([ratios[q] for q in QUERIES[:-1]]))))
            vals=[s['nonzero_query_median'] for s in samples if s['phase']=='main']
            outputs[x+'/'+y]=dict(samples=samples,main=dict(n_pairs=6,independent_books=2,median=float(np.median(vals)),min=min(vals),max=max(vals),faster_count=sum(v<1 for v in vals)))
        result[field]=outputs
    return result

def context_budget(score,vbits,refbits,ctx):
    p=fc.oracle(score);v=qc.from_bits(vbits).astype(np.float64);ref=qc.from_bits(refbits).astype(np.float64)
    gold=(p[...,None]*ref).sum(-2,keepdims=True);A=(p[...,None]*np.abs(v)).sum(-2,keepdims=True)
    B=(p[...,None]*np.abs(v-ref)).sum(-2,keepdims=True);error=np.abs(ctx.astype(np.float64)-gold)
    return dict(pass_budget=bool(np.all(error<=2e-6+2e-6*A+B)),exceed=int(np.count_nonzero(error>2e-6+2e-6*A+B)),max_abs=float(error.max()),v_error_bound_max=float(B.max()))
