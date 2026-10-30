"""DC01 fixed schedule and CPU-only validation helpers."""
import hashlib,json,math
from pathlib import Path
BOOKS=('3752','31131');LENGTHS=(769,831,833);SLOTS=(0,95,191)
STAGES=('V_MATERIALIZE','V_DEFERRED','INPUT_CHECKS','FUSED_CHECKED','FUSED_DEFERRED','PACKED_PIPELINE','READY_V_PIPELINE','RESTORED_SDPA')
ROUNDS=3;WARMUP=5;REPEATS=20
CAPS=dict(prepare=1800,capture=600,components=600,capture_case=600,component_case=120,worker_verify=120,recovery_verify=120,zip_recovery=120)
def require(ok,msg):
    if not ok:raise ValueError(msg)
def sha(b):return hashlib.sha256(b).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,o):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines()] if Path(p).exists() else []
def append(p,o):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a') as f:f.write(json.dumps(o,allow_nan=False)+'\n');f.flush()
def trajectories():return [dict(id='DC01-capture-b'+b,phase='main',book_id=b,arm='PACKED_FUSED',round=0,length=833,forwards=66,scored=65) for b in BOOKS]
def samples():return [dict(id=f'DC01-capture-b{b}-L{n}-S{s}',book_id=b,length=n,slot=s,compressed=704 if n<832 else 768) for b in BOOKS for n in LENGTHS for s in SLOTS]
def order(i,r):
    offset=(i+r)%len(STAGES);return STAGES[offset:]+STAGES[:offset]
def schedule():return [dict(sample=sp['id'],stage=stage,round=r) for r in range(ROUNDS) for i,sp in enumerate(samples()) for stage in order(i,r)]
def cells():return [dict(sample=sp['id'],stage=stage) for i,sp in enumerate(samples()) for stage in order(i,0)]
def load(p):
    import numpy as np
    with np.load(p,allow_pickle=False) as z:return {n:z[n] for n in z.files}
def fingerprint(a):return sha(str((a.shape,str(a.dtype))).encode()+a.tobytes())
def bf32(a):
    import numpy as np
    return (a.astype(np.uint32)<<16).view(np.float32)
def check_input(a,sp):
    import numpy as np,struct
    t=sp['length'];n=sp['compressed']//64;expected={}
    for kv in ('k','v'):
        expected.update({kv+'_codes':((1,16,n,3072),'uint8'),kv+'_headers':((1,16,n,12),'uint8'),
            kv+'_meta0':((1,16,n,64,1),'float16'),kv+'_meta1':((1,16,n,1,128),'float16'),kv+'_meta2':((1,16,n,1,128),'float16'),
            kv+'_tail':((1,16,t-64*n,128),'uint16'),kv+'_full':((1,16,t,128),'uint16')})
    expected.update(q=((1,16,1,128),'uint16'),actual_output=((1,1,16,128),'uint16'))
    require(set(a)==set(expected),'Paired input names')
    for k,(shape,dtype) in expected.items():
        require(a[k].shape==shape and str(a[k].dtype)==dtype,'Input shape/dtype '+k)
        if dtype=='float16':require(np.isfinite(a[k]).all(),'Nonfinite metadata')
        elif dtype=='uint16':require(np.isfinite(bf32(a[k])).all(),'Nonfinite BF16 '+k)
    for kv in ('k','v'):
        require(np.array_equal(a[kv+'_full'][:,:,64*n:],a[kv+'_tail']),'Tail mismatch')
        for i in range(n):require(np.all(a[kv+'_headers'][0,:,i]==np.frombuffer(struct.pack('<QHBB',64*i,64,3,0),np.uint8)),'Header mismatch')
    return a
def equal(a,b,label=''):
    if isinstance(b,dict):
        require(isinstance(a,dict) and set(a)==set(b),'Keys '+label)
        for k in b:equal(a[k],b[k],label+'.'+k)
    elif isinstance(b,list):
        require(isinstance(a,list) and len(a)==len(b),'Length '+label)
        for x,y in zip(a,b):equal(x,y,label)
    elif isinstance(b,float):require(isinstance(a,(float,int)) and math.isfinite(a) and math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-11),'Number '+label)
    else:require(type(a)==type(b) and a==b,'Value '+label)
