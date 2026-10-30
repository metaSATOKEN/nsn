"""AC08 fixed inventory and CPU numerical utilities; never loads a model."""
import hashlib,json,math
from pathlib import Path
np=None
def require(ok,msg):
    if not ok:raise ValueError(msg)
def sha(data):return hashlib.sha256(data).hexdigest()
def write(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines()] if Path(p).exists() else []
def bf32(bits):
    global np
    import numpy as np
    return (np.asarray(bits,dtype=np.uint16).astype(np.uint32)<<16).view(np.float32)
def bfloat_bits(x):
    import numpy as np
    u=np.asarray(x,np.float32).view(np.uint32);return ((u+0x7fff+((u>>16)&1))>>16).astype(np.uint16)
def nll(bits,target):
    x=bf32(bits).astype(np.float64);require(np.isfinite(x).all(),'Nonfinite raw logits')
    m=float(x.max());return m+math.log(float(np.exp(x-m).sum()))-float(x[target])
def drift(left,right):
    a=bf32(left).astype(np.float64);b=bf32(right).astype(np.float64);delta=a-b
    return dict(bit_differences=int(np.count_nonzero(left!=right)),elements=int(left.size),max_abs=float(np.abs(delta).max()),rmse=float(np.sqrt(np.mean(delta**2))))

PLAN=read(Path(__file__).with_name('loop_kv_ac08_inputs.json'))
ARMS=tuple(PLAN['arms']);PREFIX=768;LENGTH=1023;LOOPS=8;SLOTS=192
CAPTURE_LENGTHS=tuple(PLAN['capture_lengths']);CAPTURE_SLOTS=tuple(PLAN['capture_slots'])
TIMED_LENGTHS=tuple(PLAN['timed_lengths']);PPL_LIMIT=math.log(1.05);WARMUP_ROUNDS=2
_BOOKS=None
def book_ids():
    require(_BOOKS is not None,'Main materials have not been frozen')
    return list(_BOOKS)
def resolved_schedule(books):
    candidates=read(Path(__file__).with_name('loop_kv_ac08_candidates.json'))['candidates']
    allowed={x['book_id']:x['candidate_index'] for x in candidates}
    require(isinstance(books,list) and len(books)==len(set(books))==4 and set(books)<=set(allowed),'Four distinct eligible books required')
    require(books==sorted(books,key=allowed.__getitem__),'Selected book order changed')
    result=[]
    for template in PLAN['main_schedule_template']:
        sp={k:v for k,v in template.items() if k not in ('id_template','selected_book_rank')}
        sp['book_id']=books[template['selected_book_rank']]
        sp['id']=template['id_template'].format(book_id=sp['book_id']);result.append(sp)
    return result
def bind_books(books):
    global _BOOKS
    resolved_schedule(books);_BOOKS=tuple(books)
def schedule():return resolved_schedule(book_ids())
def gate_schedule():return PLAN['gate']
def warmup_schedule():return PLAN['warmup']
def packed(sp):return sp['arm'] not in ('OFFICIAL','OFFICIAL_REPEAT','NATIVE')
def connected(sp):return sp['arm'] not in ('OFFICIAL','OFFICIAL_REPEAT')
def has_diagnostic(sp,length):return length in CAPTURE_LENGTHS and (sp['phase']=='main' or (sp['phase']=='warmup' and sp['round']==0))
def should_capture(sp,length):return has_diagnostic(sp,length) and sp['arm']=='PACKED_KV_FUSED'
def diagnostic_ids():return [f"{sp['id']}-L{n}-S{s}" for sp in schedule() if sp['arm']=='PACKED_KV_FUSED' for n in CAPTURE_LENGTHS for s in CAPTURE_SLOTS]
def warmup_diagnostic_ids():return [f"{sp['id']}-L{n}-S{s}" for sp in warmup_schedule() if sp['round']==0 and sp['arm']=='PACKED_KV_FUSED' for n in CAPTURE_LENGTHS for s in CAPTURE_SLOTS]
def forward_compressed(length,packed=True):return post_compressed(length-1,packed) if length>PREFIX else 0
def post_compressed(length,packed=True):return max(0,64*((length-64)//64)) if packed else 0
def decode_signature(arm,length):return f'{arm}-C{forward_compressed(length,arm!="NATIVE")}-T{length}-B1H16D128-BF16'
def expected_signatures():return sorted(decode_signature(a,n) for a in ARMS for n in range(PREFIX+1,LENGTH+1))
def nominal_components(length,is_packed):
    if not is_packed:return dict(float_kv=192*2*16*length*128*2)
    c=post_compressed(length);blocks=192*2*16*(c//64)
    return dict(codes=blocks*3072,headers=blocks*12,metadata=blocks*640,native_tail=192*2*16*(length-c)*128*2)

STATE_LENGTHS=tuple(PLAN['state_lengths'])
FLUSH_LENGTHS=tuple(PLAN['flush_post_lengths'])
CAPS=PLAN['caps_seconds']
WINDOWS=[dict(id='ALL',first_target=769,last_target=LENGTH,scored=LENGTH-PREFIX,timed_lengths=list(TIMED_LENGTHS))]+PLAN['windows']
PAIRS=(('PACKED_KV_FUSED','PACKED_GROUPED'),('PACKED_KV_FUSED','NATIVE'),('PACKED_GROUPED','NATIVE'))
