"""NB05 downloader reused; NB06 extends exactly the four previously frozen texts."""
import json,signal,time,urllib.request
from contextlib import contextmanager
from datetime import datetime,timezone
from pathlib import Path
import loop_kv_nb06_contract as c
from loop_kv_nb06_runtime import clean
HERE=Path(__file__).resolve().parent
TOTAL=385
MAX_TEXT_BYTES=16*1024**2
REQUEST_SECONDS=90

def token_sha(ids):return c.sha(json.dumps(ids,separators=(',',':')).encode())
def write(path,obj):c.write_json(path.parent,path.name,obj)
def append(path,obj):c.append(path.parent,path.name,obj)
def schedule(books):return c.resolved_schedule(books)
def previous():
    path=HERE/'nb06_previous_materials.json'
    c.require(c.file_digest(path)==c.PLAN['material_selection']['previous_sha256'],'Previous materials changed')
    data=c.read(path)
    c.require([d['book_id'] for d in data['diagnostic']]==c.PLAN['material_selection']['fixed_books'],'Fixed book IDs')
    c.require(data['tokenizer_revision']==c.MODEL['revision'],'Old tokenizer revision')
    return data

@contextmanager
def request_deadline(seconds):
    def expired(signum,frame):raise TimeoutError('Candidate download wall-time cap')
    old=signal.signal(signal.SIGALRM,expired)
    before=time.monotonic();previous=signal.setitimer(signal.ITIMER_REAL,0)
    signal.setitimer(signal.ITIMER_REAL,min(seconds,previous[0]) if previous[0]>0 else seconds)
    try:yield
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,old)
        if previous[0]>0:signal.setitimer(signal.ITIMER_REAL,max(1e-6,previous[0]-(time.monotonic()-before)),previous[1])

def fetch_text(url):
    # Reading one excess byte detects oversize, but no full oversize body is retained.
    with request_deadline(REQUEST_SECONDS):
        with urllib.request.urlopen(url,timeout=REQUEST_SECONDS) as response:
            raw=response.read(MAX_TEXT_BYTES+1)
    c.require(len(raw)<=MAX_TEXT_BYTES,'Candidate response exceeded 16 MiB')
    return raw

def select_materials(root,tokenizer,deadline,fetch=None):
    root=Path(root);old=previous();fetch=fetch or fetch_text
    for name in ('selection.jsonl','materials.json','schedule.json','selection_lock.json'):
        c.require(not (root/name).exists(),'Refuse to replace frozen material: '+name)
    selected=[];received=0
    for rank,prior in enumerate(old['diagnostic']):
        row=dict(rank=rank,book_id=prior['book_id'],source_url=prior['source_url'],status='STARTED')
        try:
            c.require(time.monotonic()<deadline,'Material deadline')
            start=time.monotonic();raw=fetch(prior['source_url'])
            c.require(isinstance(raw,bytes) and len(raw)<=MAX_TEXT_BYTES,'Text byte cap')
            digest=c.sha(raw);received+=len(raw)
            row.update(received_bytes=len(raw),text_sha256=digest,download_seconds=time.monotonic()-start)
            c.require(row['download_seconds']<=REQUEST_SECONDS and received<=4*MAX_TEXT_BYTES,'Download cap')
            c.require(digest==prior['text_sha256'],'NB05 text changed; no replacement')
            ids=tokenizer(raw.decode('utf-8'),add_special_tokens=False,truncation=True,max_length=TOTAL)['input_ids']
            c.require(len(ids)==TOTAL and all(type(x) is int and 0<=x<c.MODEL['vocab_size'] for x in ids),'385 valid token IDs required')
            c.require(ids[:137]==prior['tokens'] and token_sha(ids[:137])==prior['token_sha256'],'NB05 token prefix changed')
            c.require(time.monotonic()<deadline,'Material deadline')
            doc=dict(prior,tokens=ids,token_sha256=token_sha(ids),retrieved_utc=datetime.now(timezone.utc).isoformat(),prior_token_sha256=prior['token_sha256'])
            selected.append(doc);row.update(status='EXTENDED_FIXED_TEXT',token_count=TOTAL,token_sha256=doc['token_sha256'],prior_token_sha256=prior['token_sha256'])
            append(root/'selection.jsonl',row)
            write(root/'selection_progress.json',dict(status='EXTENDING_FIXED_TEXTS',selected_books=len(selected),received_bytes=received,quality_observed=False))
            del raw
        except BaseException as exc:
            row.update(status='PREPARATION_ERROR',error_type=type(exc).__name__,error=clean(exc));append(root/'selection.jsonl',row);raise
    data=dict(experiment='LOOP-KV-NB06',source='NB05 fixed four texts extended to 385 tokens',
        data_revision=old['data_revision'],tokenizer_revision=c.MODEL['revision'],tokenizer_settings=old['tokenizer_settings'],
        previous_materials_sha256=c.PLAN['material_selection']['previous_sha256'],diagnostic=selected)
    write(root/'materials.json',data);write(root/'schedule.json',schedule([d['book_id'] for d in selected]))
    lock=dict(status='FROZEN_BEFORE_MODEL_LOAD',frozen_monotonic=time.monotonic(),selected_ids=[d['book_id'] for d in selected],
        files={name:c.file_digest(root/name) for name in ('materials.json','schedule.json','selection.jsonl')})
    write(root/'selection_lock.json',lock)
    return read_frozen(root)

def read_frozen(root):
    root=Path(root);old=previous();lock=c.read(root/'selection_lock.json')
    c.require(lock['status']=='FROZEN_BEFORE_MODEL_LOAD' and set(lock['files'])=={'materials.json','schedule.json','selection.jsonl'},'Material lock')
    for name,digest in lock['files'].items():c.require(c.file_digest(root/name)==digest,'Frozen material changed: '+name)
    data=c.read(root/'materials.json');docs=data['diagnostic'];logs=c.rows(root/'selection.jsonl')
    c.require(len(docs)==len(logs)==4 and data['previous_materials_sha256']==c.PLAN['material_selection']['previous_sha256'],'Material count/ancestry')
    c.require(data['tokenizer_revision']==c.MODEL['revision'] and data['tokenizer_settings']==old['tokenizer_settings'] and data['data_revision']==old['data_revision'],'Tokenizer/data binding')
    books=[d['book_id'] for d in docs]
    c.require(books==lock['selected_ids']==c.PLAN['material_selection']['fixed_books'] and c.read(root/'schedule.json')==schedule(books),'Fixed order/schedule')
    for rank,(doc,prior,row) in enumerate(zip(docs,old['diagnostic'],logs)):
        for key in prior:
            if key not in ('tokens','token_sha256','retrieved_utc'):c.require(doc[key]==prior[key],'Material metadata '+key)
        ids=doc['tokens'];c.require(len(ids)==TOTAL and all(type(x) is int and 0<=x<c.MODEL['vocab_size'] for x in ids),'Material tokens')
        c.require(ids[:137]==prior['tokens'] and doc['prior_token_sha256']==prior['token_sha256'] and doc['token_sha256']==token_sha(ids),'Prefix/token hash')
        c.require(row['rank']==rank and row['book_id']==prior['book_id'] and row['source_url']==prior['source_url'] and row['text_sha256']==prior['text_sha256'],'Selection identity')
        c.require(row['status']=='EXTENDED_FIXED_TEXT' and row['token_count']==TOTAL and row['token_sha256']==doc['token_sha256'] and row['prior_token_sha256']==prior['token_sha256'],'Selection record')
        c.require(0<=row['download_seconds']<=REQUEST_SECONDS and 0<row['received_bytes']<=MAX_TEXT_BYTES,'Selection budget')
    return data
