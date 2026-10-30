"""NB05 score-blind selection, copied from the completed LC02 selector. No model imports."""
import csv,hashlib,io,json,re,signal,time,urllib.request
from contextlib import contextmanager
from datetime import datetime,timezone
from pathlib import Path
import loop_kv_nb05_contract as c
from loop_kv_nb05_runtime import clean
HERE=Path(__file__).resolve().parent
CANDIDATE_FILE=HERE/'loop_kv_nb05_candidates.json'
TRAIN_FILE=HERE/'nb05_sources/pg19-train-files.txt'
META_FILE=HERE/'nb05_sources/pg19-metadata.csv'
CANDIDATE_SHA=c.PLAN['material_selection']['sha256']
CANDIDATES=c.read(CANDIDATE_FILE)
TOTAL=137
MAX_TEXT_BYTES=16*1024**2
REQUEST_SECONDS=90

def token_sha(ids):return c.sha(json.dumps(ids,separators=(',',':')).encode())
def write(path,obj):c.write_json(path.parent,path.name,obj)
def append(path,obj):c.append(path.parent,path.name,obj)
def title_key(text):return re.sub(r'\s+', ' ', text.strip().lower())
def schedule(books):return c.resolved_schedule(books)
def verify_static():
    c.require(c.file_digest(CANDIDATE_FILE)==CANDIDATE_SHA and c.read(CANDIDATE_FILE)==CANDIDATES,'Candidate source changed')
    for p,entry in zip((TRAIN_FILE,META_FILE),CANDIDATES['source_manifests']):
        c.require(c.file_digest(p)==entry['sha256'],'PG19 source manifests changed')
    candidate=CANDIDATES
    c.require(candidate['selection_seed']==2026093001 and candidate['data_revision']=='4d28bd77e66947ad3835cf78ed7aaeb4dd87ad8b'
              and candidate['tokenizer_revision']==c.MODEL['revision'] and candidate['tokenizer_model']==c.MODEL['id'],'Selection revisions/seeds')
    paths=TRAIN_FILE.read_text().splitlines()
    c.require(len(paths)==len(set(paths))==28602 and all(re.fullmatch(r'train/[0-9]+\.txt',p) for p in paths),'Train inventory')
    ordered=sorted(paths,key=lambda p:c.sha(f'{CANDIDATES['selection_seed']}:{Path(p).stem}'.encode()))
    used={d['book_id'] for d in candidate['prior_materials']}
    c.require(len(used)==28 and {Path(x).stem for x in ordered[:28]}==used,'Original exclusion inventory')
    metadata={r[0]:r for r in csv.reader(io.StringIO(META_FILE.read_text())) if len(r)>=4};expected=[]
    for rank,p in enumerate(ordered):
        bid=Path(p).stem
        if bid in used:continue
        row=metadata[bid]
        expected.append(dict(candidate_index=len(expected),original_rank_zero_based=rank,book_id=bid,file=p,title=row[1],normalized_title=title_key(row[1]),publication_date=row[2],source_url='https://storage.googleapis.com/deepmind-gutenberg/'+p))
        if len(expected)==24:break
    c.require(candidate['candidates']==expected,'Candidate order or metadata changed')
    for key,value in dict(required_books=4,max_candidates=24,tokens=137,offset=0,network_retries=0,per_text_bytes_cap=MAX_TEXT_BYTES,per_request_seconds=90,candidate_bytes_cap=24*MAX_TEXT_BYTES,freeze_before_model_load=True,replace_after_selection=False,score_blind=True).items():
        c.require(candidate['selection_contract'][key]==value,'Selection contract: '+key)
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

def select_materials(root, tokenizer, deadline, fetch=None):
    root=Path(root);verify_static();fetch=fetch or fetch_text
    for name in ('selection.jsonl','materials.json','schedule.json','selection_lock.json'):
        c.require(not (root/name).exists(), 'Selection cannot restart or replace frozen files: '+name)
    selected=[];seen_title={x['normalized_title'] for x in CANDIDATES['prior_materials']}
    seen_hash={x['text_sha256'] for x in CANDIDATES['prior_materials']};received=0
    for candidate in CANDIDATES['candidates']:
        row=dict(candidate,status='STARTED',time_utc=datetime.now(timezone.utc).isoformat())
        try:
            if time.monotonic()>deadline:raise TimeoutError('Material preparation cap')
            key=candidate['normalized_title']
            if key in seen_title:
                row['status']='DUPLICATE_NORMALIZED_TITLE'
            else:
                begun=time.monotonic();raw=fetch(candidate['source_url'])
                c.require(isinstance(raw,bytes) and len(raw)<=MAX_TEXT_BYTES,'Candidate byte type/size cap')
                if time.monotonic()-begun>REQUEST_SECONDS:raise TimeoutError('Candidate download wall-time cap')
                if time.monotonic()>deadline:raise TimeoutError('Material preparation cap')
                received+=len(raw);c.require(received<=24*MAX_TEXT_BYTES,'Candidate cumulative cap')
                digest=c.sha(raw);row.update(text_sha256=digest,received_bytes=len(raw),download_seconds=time.monotonic()-begun)
                if digest in seen_hash:
                    row['status']='DUPLICATE_TEXT_SHA256'
                else:
                    ids=tokenizer(raw.decode('utf-8'),add_special_tokens=False,truncation=True,max_length=TOTAL)['input_ids']
                    c.require(isinstance(ids,list) and all(type(x) is int and 0<=x<c.MODEL['vocab_size'] for x in ids) and len(ids)<=TOTAL,'Tokenizer output')
                    if time.monotonic()>deadline:raise TimeoutError('Material preparation cap')
                    row['token_count']=len(ids)
                    if len(ids)<TOTAL:row['status']='TOO_SHORT'
                    else:
                        doc=dict(candidate,split='train',offset=0,tokens=ids,token_sha256=token_sha(ids),
                                 text_sha256=digest,retrieved_utc=row['time_utc'])
                        selected.append(doc);seen_title.add(key);seen_hash.add(digest)
                        row.update(status='SELECTED',token_sha256=doc['token_sha256'])
                del raw
            append(root/'selection.jsonl',row)
            write(root/'selection_progress.json',dict(status='SELECTING',selected_books=len(selected),
                selected_ids=[x['book_id'] for x in selected],candidate_attempts=candidate['candidate_index']+1,
                received_bytes=received,quality_observed=False))
            print(f"Materials: {len(selected)}/4, candidate {candidate['candidate_index']+1}: {row['status']}")
            if len(selected)==4:break
        except BaseException as exc:
            row.update(status='PREPARATION_ERROR',error_type=type(exc).__name__,error=clean(str(exc)))
            append(root/'selection.jsonl',row)
            write(root/'selection_progress.json',dict(status='PREPARATION_ERROR',selected_books=len(selected),
                selected_ids=[x['book_id'] for x in selected],received_bytes=received,quality_observed=False))
            raise
    if len(selected)!=4:
        write(root/'selection_progress.json',dict(status='INSUFFICIENT_ELIGIBLE_MATERIALS',
            selected_books=len(selected),selected_ids=[x['book_id'] for x in selected],received_bytes=received,quality_observed=False))
        raise ValueError('Fewer than four eligible books in fixed 24 candidates')
    data=dict(experiment='LOOP-KV-NB05',source='Fixed PG19 candidate order; selected before model loading',
        data_revision=CANDIDATES['data_revision'],tokenizer_revision=c.MODEL['revision'],selection_seed=CANDIDATES['selection_seed'],
        candidate_sha256=CANDIDATE_SHA,tokenizer_settings=CANDIDATES['tokenizer_settings'],diagnostic=selected)
    write(root/'materials.json',data)
    write(root/'schedule.json',schedule([d['book_id'] for d in selected]))
    lock=dict(status='FROZEN_BEFORE_MODEL_LOAD',frozen_monotonic=time.monotonic(),selected_ids=[d['book_id'] for d in selected],
              files={name:c.sha((root/name).read_bytes()) for name in ('materials.json','schedule.json','selection.jsonl')})
    write(root/'selection_lock.json',lock)
    write(root/'selection_progress.json',dict(status='FROZEN_BEFORE_MODEL_LOAD',selected_books=4,
        selected_ids=lock['selected_ids'],received_bytes=received,quality_observed=False))
    return read_frozen(root)

def read_frozen(root):
    root=Path(root);verify_static()
    lock=json.loads((root/'selection_lock.json').read_text())
    c.require(lock['status']=='FROZEN_BEFORE_MODEL_LOAD' and
              set(lock['files'])=={'materials.json','schedule.json','selection.jsonl'}, 'Selection lock')
    for name,digest in lock['files'].items():
        c.require(c.sha((root/name).read_bytes())==digest,'Frozen selection changed: '+name)
    data=json.loads((root/'materials.json').read_text());docs=data['diagnostic']
    c.require(data['candidate_sha256']==CANDIDATE_SHA and data['tokenizer_settings']==CANDIDATES['tokenizer_settings'] and data['data_revision']==CANDIDATES['data_revision']
              and data['tokenizer_revision']==c.MODEL['revision'] and data['selection_seed']==CANDIDATES['selection_seed'],'Material ancestry')
    books=[d['book_id'] for d in docs];expected_schedule=schedule(books)
    c.require(books==lock['selected_ids'] and json.loads((root/'schedule.json').read_text())==expected_schedule,
              'Locked schedule mismatch')
    candidates={x['book_id']:x for x in CANDIDATES['candidates']}
    for doc in docs:
        c.require(all(doc.get(k)==v for k,v in candidates[doc['book_id']].items()),'Selected metadata changed')
        c.require(doc['split']=='train' and doc['offset']==0 and len(doc['tokens'])==TOTAL and
                  all(type(t) is int and 0<=t<c.MODEL['vocab_size'] for t in doc['tokens']) and token_sha(doc['tokens'])==doc['token_sha256'],
                  'Selected token inventory')
        c.require(re.fullmatch(r'[0-9a-f]{64}',doc['text_sha256']) is not None,'Selected text hash')
    logs=c.rows(root/'selection.jsonl');n=0
    titles={x['normalized_title'] for x in CANDIDATES['prior_materials']}
    hashes={x['text_sha256'] for x in CANDIDATES['prior_materials']}
    c.require(4<=len(logs)<=24,'Selection log count')
    for i,row in enumerate(logs):
        candidate=CANDIDATES['candidates'][i]
        c.require(all(row.get(k)==v for k,v in candidate.items()) and n<4,'Selection skipped rank or continued after four')
        if candidate['normalized_title'] in titles:
            c.require(row['status']=='DUPLICATE_NORMALIZED_TITLE','Duplicate title not skipped');continue
        c.require(type(row.get('received_bytes')) is int and 0<=row['received_bytes']<=MAX_TEXT_BYTES,'Selection byte count')
        if row.get('text_sha256') in hashes:
            c.require(row['status']=='DUPLICATE_TEXT_SHA256','Duplicate text not skipped');continue
        if row['status']=='TOO_SHORT':
            c.require(type(row.get('token_count')) is int and 0<=row['token_count']<TOTAL,'Short material metadata');continue
        c.require(row['status']=='SELECTED' and row['token_count']==TOTAL and row['book_id']==books[n] and
                  row['token_sha256']==docs[n]['token_sha256'] and row['text_sha256']==docs[n]['text_sha256'],
                  'Selected row correspondence')
        titles.add(candidate['normalized_title']);hashes.add(row['text_sha256']);n+=1
    c.require(n==4,'Incomplete selection proof')
    c.require(sum(r.get('received_bytes',0) for r in logs)<=24*MAX_TEXT_BYTES,'Saved selection byte cap')
    c.require(all(0<=r.get('download_seconds',0)<=REQUEST_SECONDS for r in logs),'Saved request duration')
    return data
