"""AC08 score-blind selection, copied from the completed LC02 selector. No model imports."""
import csv,hashlib,io,json,re,signal,time,urllib.request
from contextlib import contextmanager
from datetime import datetime,timezone
from pathlib import Path
import loop_kv_pilot as w
import loop_kv_ac08_contract as ac
HERE=Path(__file__).resolve().parent
CANDIDATE_FILE=HERE/'loop_kv_ac08_candidates.json'
SMOKE_FILE=HERE/'loop_kv_pk01_materials.json'
TRAIN_FILE=HERE/'lc02_sources/pg19-train-files.txt'
META_FILE=HERE/'lc02_sources/pg19-metadata.csv'
CANDIDATE_SHA='974342850682387b283af77da32e139ecd01cfe12b7ad3524598872e76a37d8c'
SMOKE_SHA='395ee37a04a68723620165d16796cfc5df4d611dbbd592c61151659bc6a70c0f'
CANDIDATES=json.loads(CANDIDATE_FILE.read_text())
SMOKE=json.loads(SMOKE_FILE.read_text())
TOTAL=1024
MAX_TEXT_BYTES=16*1024**2
REQUEST_SECONDS=90

def title_key(text):return re.sub(r'\s+', ' ', text.strip().lower())
def schedule(books):return ac.resolved_schedule(books)
def verify_static():
    for path,expected,obj in [(CANDIDATE_FILE,CANDIDATE_SHA,CANDIDATES),(SMOKE_FILE,SMOKE_SHA,SMOKE)]:
        w.require(w.sha(path.read_bytes())==expected,'Static material source hash: '+path.name)
        w.require(json.loads(path.read_text())==obj,'In-memory material changed: '+path.name)
    w.require(w.sha(TRAIN_FILE.read_bytes())==w.TRAIN_SHA and w.sha(META_FILE.read_bytes())==w.META_SHA,'PG19 source manifests changed')
    c=CANDIDATES
    w.require(c['selection_seed']==w.SEED and c['data_revision']==w.DATA_REV and c['tokenizer_revision']==w.MODEL_REV,'Selection revisions/seeds')
    paths=TRAIN_FILE.read_text().splitlines()
    w.require(len(paths)==len(set(paths))==28602 and all(re.fullmatch(r'train/[0-9]+\.txt',p) for p in paths),'Train inventory')
    ordered=sorted(paths,key=lambda p:w.sha(f'{w.SEED}:{Path(p).stem}'.encode()))
    used={d['book_id'] for d in c['prior_materials']}
    w.require(len(used)==24 and {Path(x).stem for x in ordered[:24]}==used,'Original exclusion inventory')
    metadata={r[0]:r for r in csv.reader(io.StringIO(META_FILE.read_text())) if len(r)>=4};expected=[]
    for rank,p in enumerate(ordered):
        bid=Path(p).stem
        if bid in used:continue
        row=metadata[bid]
        expected.append(dict(candidate_index=len(expected),original_rank_zero_based=rank,book_id=bid,file=p,title=row[1],normalized_title=title_key(row[1]),publication_date=row[2],source_url='https://storage.googleapis.com/deepmind-gutenberg/'+p))
        if len(expected)==24:break
    w.require(c['candidates']==expected,'Candidate order or metadata changed')
    for key,value in dict(required_books=4,max_candidates=24,tokens=1024,offset=0,network_retries=0,per_text_bytes_cap=MAX_TEXT_BYTES,per_request_seconds=90,candidate_bytes_cap=24*MAX_TEXT_BYTES,freeze_before_model_load=True,replace_after_selection=False,score_blind=True).items():
        w.require(c['selection_contract'][key]==value,'Selection contract: '+key)
    w.require([d['book_id'] for d in SMOKE['smoke']]==['37013'],'Smoke inventory')
    d=SMOKE['smoke'][0]
    w.require(d['offset']==0 and len(d['tokens'])==1024 and w.token_sha(d['tokens'])==d['token_sha256'],'Frozen smoke content')

@contextmanager
def request_deadline(seconds):
    def expired(signum,frame):raise TimeoutError('Candidate download wall-time cap')
    old=signal.signal(signal.SIGALRM,expired)
    previous=signal.setitimer(signal.ITIMER_REAL,seconds)
    try:yield
    finally:
        signal.setitimer(signal.ITIMER_REAL,*previous)
        signal.signal(signal.SIGALRM,old)

def fetch_text(url):
    # Reading one excess byte detects oversize, but no full oversize body is retained.
    with request_deadline(REQUEST_SECONDS):
        with urllib.request.urlopen(url,timeout=REQUEST_SECONDS) as response:
            raw=response.read(MAX_TEXT_BYTES+1)
    w.require(len(raw)<=MAX_TEXT_BYTES,'Candidate response exceeded 16 MiB')
    return raw

def select_materials(root, tokenizer, deadline, fetch=None):
    fetch=fetch or fetch_text
    for name in ('selection.jsonl','materials.json','schedule.json','selection_lock.json'):
        w.require(not (root/name).exists(), 'Selection cannot restart or replace frozen files: '+name)
    selected=[];seen_title={x['normalized_title'] for x in CANDIDATES['prior_materials']}
    seen_hash={x['text_sha256'] for x in CANDIDATES['prior_materials']};received=0
    for candidate in CANDIDATES['candidates']:
        row=dict(candidate,status='STARTED',time_utc=datetime.now(timezone.utc).isoformat())
        try:
            if time.monotonic()>deadline:raise w.BudgetStop('Material preparation cap')
            key=candidate['normalized_title']
            if key in seen_title:
                row['status']='DUPLICATE_NORMALIZED_TITLE'
            else:
                begun=time.monotonic();raw=fetch(candidate['source_url'])
                w.require(isinstance(raw,bytes) and len(raw)<=MAX_TEXT_BYTES,'Candidate byte type/size cap')
                if time.monotonic()-begun>REQUEST_SECONDS:raise TimeoutError('Candidate download wall-time cap')
                if time.monotonic()>deadline:raise w.BudgetStop('Material preparation cap')
                received+=len(raw);w.require(received<=24*MAX_TEXT_BYTES,'Candidate cumulative cap')
                digest=w.sha(raw);row.update(text_sha256=digest,received_bytes=len(raw),download_seconds=time.monotonic()-begun)
                if digest in seen_hash:
                    row['status']='DUPLICATE_TEXT_SHA256'
                else:
                    ids=tokenizer(raw.decode('utf-8'),add_special_tokens=False,truncation=True,max_length=TOTAL)['input_ids']
                    w.require(isinstance(ids,list) and all(type(x) is int and x>=0 for x in ids) and len(ids)<=TOTAL,'Tokenizer output')
                    if time.monotonic()>deadline:raise w.BudgetStop('Material preparation cap')
                    row['token_count']=len(ids)
                    if len(ids)<TOTAL:row['status']='TOO_SHORT'
                    else:
                        doc=dict(candidate,split='train',offset=0,tokens=ids,token_sha256=w.token_sha(ids),
                                 text_sha256=digest,retrieved_utc=row['time_utc'])
                        selected.append(doc);seen_title.add(key);seen_hash.add(digest)
                        row.update(status='SELECTED',token_sha256=doc['token_sha256'])
                del raw
            w.append_json(root/'selection.jsonl',row)
            w.write_json(root/'selection_progress.json',dict(status='SELECTING',selected_books=len(selected),
                selected_ids=[x['book_id'] for x in selected],candidate_attempts=candidate['candidate_index']+1,
                received_bytes=received,quality_observed=False))
            w.log(f"Materials: {len(selected)}/4, candidate {candidate['candidate_index']+1}: {row['status']}")
            if len(selected)==4:break
        except BaseException as exc:
            row.update(status='PREPARATION_ERROR',error_type=type(exc).__name__,error=w.redact(str(exc)))
            w.append_json(root/'selection.jsonl',row)
            w.write_json(root/'selection_progress.json',dict(status='PREPARATION_ERROR',selected_books=len(selected),
                selected_ids=[x['book_id'] for x in selected],received_bytes=received,quality_observed=False))
            raise
    if len(selected)!=4:
        w.write_json(root/'selection_progress.json',dict(status='INSUFFICIENT_ELIGIBLE_MATERIALS',
            selected_books=len(selected),selected_ids=[x['book_id'] for x in selected],received_bytes=received,quality_observed=False))
        raise w.IntegrityError('Fewer than four eligible books in fixed 24 candidates')
    data=dict(experiment='LOOP-KV-AC08',source='Fixed PG19 candidate order; selected before model loading',
        data_revision=w.DATA_REV,tokenizer_revision=w.MODEL_REV,selection_seed=w.SEED,
        candidate_sha256=CANDIDATE_SHA,smoke_source_sha256=SMOKE_SHA,
        smoke=SMOKE['smoke'],diagnostic=selected)
    w.write_json(root/'materials.json',data)
    w.write_json(root/'schedule.json',schedule([d['book_id'] for d in selected]))
    lock=dict(status='FROZEN_BEFORE_MODEL_LOAD',frozen_monotonic=time.monotonic(),selected_ids=[d['book_id'] for d in selected],
              files={name:w.sha((root/name).read_bytes()) for name in ('materials.json','schedule.json','selection.jsonl')})
    w.write_json(root/'selection_lock.json',lock)
    w.write_json(root/'selection_progress.json',dict(status='FROZEN_BEFORE_MODEL_LOAD',selected_books=4,
        selected_ids=lock['selected_ids'],received_bytes=received,quality_observed=False))
    return read_frozen(root)

def read_frozen(root):
    lock=json.loads((root/'selection_lock.json').read_text())
    w.require(lock['status']=='FROZEN_BEFORE_MODEL_LOAD' and
              set(lock['files'])=={'materials.json','schedule.json','selection.jsonl'}, 'Selection lock')
    for name,digest in lock['files'].items():
        w.require(w.sha((root/name).read_bytes())==digest,'Frozen selection changed: '+name)
    data=json.loads((root/'materials.json').read_text());docs=data['diagnostic']
    w.require(data['candidate_sha256']==CANDIDATE_SHA and data['smoke_source_sha256']==SMOKE_SHA
              and data['smoke']==SMOKE['smoke'] and data['data_revision']==w.DATA_REV
              and data['tokenizer_revision']==w.MODEL_REV and data['selection_seed']==w.SEED,'Material ancestry')
    books=[d['book_id'] for d in docs];expected_schedule=schedule(books)
    w.require(books==lock['selected_ids'] and json.loads((root/'schedule.json').read_text())==expected_schedule,
              'Locked schedule mismatch')
    candidates={x['book_id']:x for x in CANDIDATES['candidates']}
    for doc in docs:
        w.require(all(doc.get(k)==v for k,v in candidates[doc['book_id']].items()),'Selected metadata changed')
        w.require(doc['split']=='train' and doc['offset']==0 and len(doc['tokens'])==TOTAL and
                  all(type(t) is int and t>=0 for t in doc['tokens']) and w.token_sha(doc['tokens'])==doc['token_sha256'],
                  'Selected token inventory')
        w.require(re.fullmatch(r'[0-9a-f]{64}',doc['text_sha256']) is not None,'Selected text hash')
    logs=w.read_rows(root/'selection.jsonl');n=0
    titles={x['normalized_title'] for x in CANDIDATES['prior_materials']}
    hashes={x['text_sha256'] for x in CANDIDATES['prior_materials']}
    w.require(4<=len(logs)<=24,'Selection log count')
    for i,row in enumerate(logs):
        c=CANDIDATES['candidates'][i]
        w.require(all(row.get(k)==v for k,v in c.items()) and n<4,'Selection skipped rank or continued after four')
        if c['normalized_title'] in titles:
            w.require(row['status']=='DUPLICATE_NORMALIZED_TITLE','Duplicate title not skipped');continue
        w.require(type(row.get('received_bytes')) is int and 0<=row['received_bytes']<=MAX_TEXT_BYTES,'Selection byte count')
        if row.get('text_sha256') in hashes:
            w.require(row['status']=='DUPLICATE_TEXT_SHA256','Duplicate text not skipped');continue
        if row['status']=='TOO_SHORT':
            w.require(type(row.get('token_count')) is int and 0<=row['token_count']<TOTAL,'Short material metadata');continue
        w.require(row['status']=='SELECTED' and row['token_count']==TOTAL and row['book_id']==books[n] and
                  row['token_sha256']==docs[n]['token_sha256'] and row['text_sha256']==docs[n]['text_sha256'],
                  'Selected row correspondence')
        titles.add(c['normalized_title']);hashes.add(row['text_sha256']);n+=1
    w.require(n==4,'Incomplete selection proof')
    w.require(sum(r.get('received_bytes',0) for r in logs)<=24*MAX_TEXT_BYTES,'Saved selection byte cap')
    w.require(all(0<=r.get('download_seconds',0)<=REQUEST_SECONDS for r in logs),'Saved request duration')
    return data
