"""NB05 fixed contract and bounded evidence IO. No torch/model imports."""
import hashlib
import json
import math
import os
from pathlib import Path

def require(ok, message):
    if not ok:
        raise ValueError(message)

def read(path):
    return json.loads(Path(path).read_text())

def sha(data):
    return hashlib.sha256(data).hexdigest()

PLAN = read(Path(__file__).with_name('loop_kv_nb05_inputs.json'))
CAPS = PLAN['phase_cap_seconds']
MODEL = PLAN['model']
SERIES = tuple(PLAN['series'])

def file_digest(path, algorithm='sha256', git_blob=False):
    path = Path(path)
    h = hashlib.new(algorithm)
    if git_blob:
        h.update(('blob ' + str(path.stat().st_size) + '\0').encode())
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def run_root(root):
    root=Path(root)
    if root.parent.name=='books' and root.name in ('b0','b1','b2','b3'):
        return root.parent.parent
    return root

def evidence_bytes(root):
    return sum(p.stat().st_size for p in Path(root).rglob('*') if p.is_file())

def reserve(root, additional, previous=0, closing=False):
    limit = PLAN['evidence_limit_bytes']
    if not closing:
        limit -= PLAN['closure_reserve_bytes']
    require(evidence_bytes(run_root(root)) + additional <= limit, 'Evidence cap before write')
    # Count the old destination too: atomic replacement temporarily needs both.

def write_json(root, name, obj, closing=False):
    root = Path(root);path = root / name
    data = (json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
    reserve(root, len(data), closing=closing)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_bytes(data);os.replace(tmp, path)
    # Mirror the current book to the launcher's root watchdog; book evidence remains local.
    if run_root(root)!=root and name in ('progress.json','active_case.json'):
        rank=int(root.name[1:]);binding=read(root/'book.json')
        mirror=dict(obj,selected_book_rank=rank,book_id=binding['book_id'])
        if name=='progress.json':
            mirror['attempted']=rank*PLAN['forward_calls_per_book']+obj['attempted']
            mirror['finished']=rank*PLAN['forward_calls_per_book']+obj['finished']
            mirror['series']=f'b{rank}-'+obj.get('series','')
        else:mirror['id']=f'b{rank}-'+obj['id']
        write_json(run_root(root),name,mirror,closing=closing)

def save_arrays(root, name, arrays):
    import numpy as np
    root = Path(root);path = root / name
    require(all(isinstance(v, np.ndarray) and not v.dtype.hasobject for v in arrays.values()), 'Numeric arrays only')
    reserve(root, sum(v.nbytes for v in arrays.values()) + 1024 * len(arrays) + 4096)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('wb') as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)

def tensor_hash(bits):
    import numpy as np
    a = np.ascontiguousarray(bits, dtype='<u2')
    header = json.dumps({'dtype':'bfloat16','shape':list(a.shape)}, sort_keys=True, separators=(',',':')).encode()
    return sha(header + b'\n' + a.tobytes())

def bf32(bits):
    import numpy as np
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)

def nll(logits, target):
    import numpy as np
    x = np.asarray(logits, dtype=np.float64)
    require(x.ndim == 1 and np.isfinite(x).all() and 0 <= target < len(x), 'Invalid logit/target')
    m = float(x.max())
    return m + math.log(float(np.exp(x - m).sum())) - float(x[target])

def drift(a, b, bf16=False):
    import numpy as np
    require(a.shape == b.shape and a.dtype == b.dtype, 'Comparison shape/dtype')
    av = bf32(a) if bf16 else a
    bv = bf32(b) if bf16 else b
    bits_a = a if bf16 else a.view(np.uint32)
    bits_b = b if bf16 else b.view(np.uint32)
    delta = np.asarray(av, np.float64) - np.asarray(bv, np.float64)
    require(np.isfinite(delta).all(), 'Nonfinite comparison')
    return {'bit_differences':int(np.count_nonzero(bits_a != bits_b)),
            'elements':int(a.size),'max_abs':float(np.abs(delta).max())}

def expected_keys():
    return [f'{kv}_{slot:02d}' for slot in range(MODEL['slots']) for kv in ('K','V')]

def expected_shape(length):
    return [1, MODEL['kv_heads'], length, MODEL['head_dim']]

def safe_source_name(name):
    p = Path(name)
    require(isinstance(name, str) and not p.is_absolute() and '..' not in p.parts and name != '', 'Unsafe source path')
    return p

def check_embedded(root):
    root = Path(root);manifest = read(root / 'embedded_sha256.json')
    for name, digest in manifest.items():
        p = root / safe_source_name(name)
        require(p.is_file() and file_digest(p) == digest, 'Embedded source changed: ' + name)
    require(read(root/'loop_kv_nb05_inputs.json') == PLAN, 'Plan differs from trusted verifier')
    require(file_digest(root/'nb05_approved_contract.json')==PLAN['design_contract_sha256'],'Approved contract hash')
    design=read(root/'nb05_approved_contract.json')
    for key,value in design.items():
        if key not in ('stage','version'):require(PLAN[key]==value,'Approved specification changed: '+key)
    require(PLAN['codec']==design['compression_contract'] and PLAN['captures']==design['captures_per_book'],'Runtime aliases')
    return len(manifest)

def rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines()] if Path(path).exists() else []

def append(root,name,obj):
    data=(json.dumps(obj,allow_nan=False)+'\n').encode();reserve(root,len(data))
    with (Path(root)/name).open('ab') as f:f.write(data);f.flush()

def arrays_hash(a):
    return {n:sha(str((x.shape,str(x.dtype))).encode()+x.tobytes()) for n,x in sorted(a.items())}

def equal(a,b,label='summary',diffs=None):
    if diffs is None:diffs=[]
    if isinstance(b,dict):
        require(isinstance(a,dict) and set(a)==set(b),'Keys '+label)
        for key in b:equal(a[key],b[key],label+'.'+key,diffs)
    elif isinstance(b,list):
        require(isinstance(a,list) and len(a)==len(b),'Length '+label)
        for i,(x,y) in enumerate(zip(a,b)):equal(x,y,label+'.'+str(i),diffs)
    elif isinstance(b,float):
        require(type(a) in (int,float) and math.isfinite(a) and math.isclose(a,b,rel_tol=PLAN['summary_float_rtol'],abs_tol=PLAN['summary_float_atol']),'Float '+label)
        if a!=b:diffs.append(dict(path=label,stored=a,recomputed=b,abs_delta=abs(a-b)))
    else:require(type(a)==type(b) and a==b,'Value '+label)
    return diffs

def load(path):
    import numpy as np
    with np.load(path,allow_pickle=False) as z:return {n:z[n] for n in z.files}

def array_hash(x):
    return sha(str((x.shape,str(x.dtype))).encode()+x.tobytes())

def schedule():
    return [{k:(v[3:] if k=='id' else v) for k,v in s.items() if k!='selected_book_rank'}
            for s in PLAN['schedule'] if s['selected_book_rank']==0]

def resolved_schedule(books):
    require(len(books)==len(set(books))==4,'Four unique book IDs')
    return [dict(s,book_id=books[s['selected_book_rank']]) for s in PLAN['schedule']]
file_sha=file_digest
