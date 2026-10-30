"""Independent saved-artifact audit: NumPy only, no torch/worker/returned-code execution."""
import json,math,sys,hashlib,statistics
from pathlib import Path
import numpy as np
import loop_kv_ac08_contract as ac
import loop_kv_ac08_materials as m
import loop_kv_qk01_contract as qc
import loop_kv_ac08_audit as audit
import loop_kv_ac04_validation as validation
req=ac.require
def equal(a,b,label):
    if isinstance(b,dict):
        req(isinstance(a,dict) and set(a)==set(b),'Keys '+label)
        for k in b:equal(a[k],b[k],label+'.'+k)
    elif isinstance(b,list):
        req(isinstance(a,list) and len(a)==len(b),'Length '+label)
        for x,y in zip(a,b):equal(x,y,label)
    elif isinstance(b,float):req(isinstance(a,(float,int)) and math.isfinite(a) and math.isclose(a,b,rel_tol=2e-11,abs_tol=2e-12),'Value '+label)
    else:req(type(a)==type(b) and a==b,'Value '+label)
def path(root,sp):return Path(root)/sp['phase']/sp['id']
def load_npz(file):
    with np.load(file,allow_pickle=False) as z:return {n:z[n] for n in z.files}
def gate_checks(root):
    specs={s['arm']:s for s in ac.gate_schedule()};answer=[]
    for left,right in [('OFFICIAL','OFFICIAL_REPEAT'),('OFFICIAL','NATIVE')]:
        a,b=(path(root,specs[x]) for x in (left,right))
        x,y=(np.load(p/'raw_logits.npy',allow_pickle=False) for p in (a,b))
        state_equal=ac.read(a/'cache_state.json')==ac.read(b/'cache_state.json')
        bit_equal=x.dtype==y.dtype==np.uint16 and x.shape==y.shape and bool(np.array_equal(x,y))
        nll_equal=ac.rows(a/'per_token.jsonl')==ac.rows(b/'per_token.jsonl')
        answer.append(dict(left=left,right=right,logits_bitwise=bit_equal,cache_bitwise=state_equal,nll_equal=nll_equal,pass_=bool(bit_equal and state_equal and nll_equal)))
    return [{('pass' if k=='pass_' else k):v for k,v in row.items()} for row in answer]
def tensor_token_sha(tokens):
    return ac.sha(json.dumps(tokens,separators=(',',':')).encode())

def decode_coverage(root,sp):
    connected=sp['arm'] not in ('OFFICIAL','OFFICIAL_REPEAT')
    records=ac.rows(path(root,sp)/'decode_coverage.jsonl')
    req([r['length'] for r in records]==list(range(769,sp['length']+1)) if connected else records==[],'Decode coverage inventory')
    if connected:
        for r in records:
            req(r['grouped_checker_calls']==((r['length']-768)*192 if sp['arm']=='PACKED_GROUPED' else 0),'Per-length checker coverage')
            req(r['kv_checker_calls']==((r['length']-768)*192 if sp['arm']=='PACKED_KV_FUSED' else 0),'Per-length KV checker coverage')
            n=r['length'];equal(r['slot_cumulative_calls'],{str(s):n-768 for s in range(192)},'Per-length slot counts')
            req(r['compressed']==ac.forward_compressed(n,sp['arm']!='NATIVE'),'Forward compression coordinate')
    return records

def warmup_coverage(root):
    root=Path(root);signatures=[];last_finish=0.
    for rnd in range(2):
        sig=set()
        for sp in [s for s in ac.warmup_schedule() if s['round']==rnd]:
            r=ac.read(path(root,sp)/'result.json');req(r['status']=='COMPLETE_FINITE' and all(r[k]==v for k,v in sp.items()),'Incomplete warmup trajectory')
            clock=ac.read(path(root,sp)/'trajectory_clock.json')
            req(math.isfinite(clock['started_monotonic']) and math.isfinite(clock['finished_monotonic']) and clock['started_monotonic']>=last_finish and clock['finished_monotonic']>=clock['started_monotonic'],'Warmup chronology')
            last_finish=clock['finished_monotonic']
            for row in decode_coverage(root,sp):sig.add(ac.decode_signature(sp['arm'],row['length']))
        req(sorted(sig)==ac.expected_signatures(),'Warmup shape coverage');signatures.append(sorted(sig))
    rr=ac.rows(root/'warmup_query/index.jsonl');req([r['id'] for r in rr]==ac.warmup_diagnostic_ids(),'Warmup TRACE coverage')
    return dict(rounds=2,trajectories=6,signatures_per_round=765,signatures=signatures,last_finished_monotonic=last_finish,trace_samples=24)

def timing_preparation_check(root):
    coverage=warmup_coverage(root);ready=ac.read(root/'timing_ready.json');equal(ready['coverage'],coverage,'Preparation coverage')
    req(ready['completed_monotonic']>=coverage['last_finished_monotonic'],'Ready before warmup finished')
    rows=ac.rows(root/'main_cache.jsonl');req([r['id'] for r in rows]==[sp['id'] for sp in ac.schedule()],'Main cache inventory')
    r0=ac.read(root/'warmup_round0_cache.json');r1=ac.read(root/'warmup_round1_cache.json')
    stable=r0==r1==ready['cache'];previous=ready['completed_monotonic'];states=[]
    for row,sp in zip(rows,ac.schedule()):
        clock=ac.read(path(root,sp)/'trajectory_clock.json')
        req(previous<=row['started_monotonic']<=clock['started_monotonic']<=clock['finished_monotonic']<=row['finished_monotonic'],'Main starts before warmup / bad chronology')
        previous=row['finished_monotonic'];unchanged=row['before']==row['after']==ready['cache'];stable=stable and unchanged
        states.append(dict(id=sp['id'],unchanged_from_ready=unchanged))
    final=ac.read(root/'main_end_cache.json');stable=stable and final==ready['cache']
    return dict(status='TIMING_CACHE_UNCHANGED' if stable else 'TIMING_CACHE_CHANGED',prepared_timing_supported=stable,
        warmup_round1_cache_unchanged=r0==r1,ready_cache_files=len(ready['cache']),main=states)

def validate_state(rows,length,is_packed):
    req([r['slot'] for r in rows]==list(range(192)),'Cache state slot inventory')
    keys={'tail','codes','headers','meta0','meta1','meta2'} if is_packed else {'tail'}
    for row in rows:
        req(set(row)=={'slot','length','K','V'} and row['length']==length,'State coordinate')
        for kv in ('K','V'):
            req(set(row[kv])==keys,'Cache fingerprint representation')
            for h in row[kv].values():req(isinstance(h,str) and len(h)==64 and all(x in '0123456789abcdef' for x in h),'Cache digest format')

def timing_summary(record,lengths):
    selected=set(lengths);ts=[x for x in record['timing'] if x['length'] in selected]
    req([(x['length'],x['operation']) for x in ts]==[(n,o) for n in lengths for o in ('FORWARD','POST')],'Window timing inventory')
    initial=[x for x in record['timing'] if x['length']==768]
    def phase_summary(rows):
        return dict(intervals=len(rows),cuda_ms=sum(x['cuda_ms'] for x in rows),wall_seconds=sum(x['wall_seconds'] for x in rows),
            allocated_peak=max((x['allocated_peak'] for x in rows),default=None),reserved_peak=max((x['reserved_peak'] for x in rows),default=None))
    return dict(cuda_ms_per_token=math.fsum(x['cuda_ms'] for x in ts)/len(lengths),wall_seconds_per_token=math.fsum(x['wall_seconds'] for x in ts)/len(lengths),
        allocated_peak=max(x['allocated_peak'] for x in ts),reserved_peak=max(x['reserved_peak'] for x in ts),terminal_storage=record['inventory'][-1]['unique_storage_bytes'],
        kv_inventory=record['inventory'],resident=record['resident'],prefill_forward=initial[0],initial_compression=initial[1],
        decode_forward=phase_summary([x for x in ts if x['operation']=='FORWARD']),
        decode_post=phase_summary([x for x in ts if x['operation']=='POST']),
        flush_post=[x for x in ts if x['operation']=='POST' and x['length'] in ac.FLUSH_LENGTHS],
        regular_post=phase_summary([x for x in ts if x['operation']=='POST' and x['length'] not in ac.FLUSH_LENGTHS]),
        recorded_prefill_inclusive_peak={f:max(x[f] for x in record['timing']) for f in ('allocated_peak','reserved_peak')})

def pair_comparison(block,means,times,lo,hi):
    answer=[]
    for candidate,base in ac.PAIRS:
        delta=means[candidate]-means[base]
        req(all(times[base][f]>0 for f in ('cuda_ms_per_token','wall_seconds_per_token','allocated_peak','reserved_peak')),'Zero reference timing/memory')
        answer.append(dict(candidate=candidate,reference=base,delta_nll=delta,ppl_change_pct=100*math.expm1(delta) if delta<700 else None,
            within_five_percent=delta<=ac.PPL_LIMIT,time_ratio=times[candidate]['cuda_ms_per_token']/times[base]['cuda_ms_per_token'],
            wall_ratio=times[candidate]['wall_seconds_per_token']/times[base]['wall_seconds_per_token'],
            allocated_peak_ratio=times[candidate]['allocated_peak']/times[base]['allocated_peak'],reserved_peak_ratio=times[candidate]['reserved_peak']/times[base]['reserved_peak'],
            logit_drift=ac.drift(block[candidate]['raw'][lo-768:hi-767],block[base]['raw'][lo-768:hi-767])))
    return answer

def primary_quality_rows(comparisons):
    return [dict(book_id=r['book_id'],round=r['round'],window=wid,**p) for r in comparisons
        for wid in ac.PLAN['primary_quality_windows'] for p in r['windows'][wid]['pairs']
        if (p['candidate'],p['reference'])==('PACKED_KV_FUSED','NATIVE')]

def descriptive_summaries(comparisons):
    books=ac.book_ids();answer={}
    for window in ('ALL','W1','W2','W3','W4'):
        by_book=[]
        for book in books:
            rows=[r['windows'][window] for r in comparisons if r['book_id']==book]
            req(len(rows)==2,'Two technical repeats per book')
            delta=statistics.mean(next(p['delta_nll'] for p in r['pairs'] if (p['candidate'],p['reference'])==('PACKED_KV_FUSED','NATIVE')) for r in rows)
            times={c+'_over_'+b:{field:statistics.median(next(p[field] for p in r['pairs'] if (p['candidate'],p['reference'])==(c,b)) for r in rows) for field in ('time_ratio','wall_ratio')} for c,b in ac.PAIRS}
            by_book.append(dict(book_id=book,mean_delta_nll=delta,ppl_change_pct=100*math.expm1(delta) if delta<700 else None,within_five_percent=delta<=ac.PPL_LIMIT,repeat_median_ratios=times))
        delta=statistics.mean(x['mean_delta_nll'] for x in by_book)
        answer[window]=dict(independent_books=4,technical_repeats=2,books=by_book,mean_delta_nll=delta,ppl_change_pct=100*math.expm1(delta) if delta<700 else None,books_exceeding_five_percent=sum(not x['within_five_percent'] for x in by_book))
    return answer

def verify(root,check_summary=True):
    root=Path(root)
    m.verify_static();mats=m.read_frozen(root);ac.bind_books([x['book_id'] for x in mats['diagnostic']])
    lock=ac.read(root/'selection_lock.json');loaded=ac.read(root/'model_load_started.json')
    req(math.isfinite(lock['frozen_monotonic']) and 0<=lock['frozen_monotonic']<=loaded['started_monotonic'] and math.isfinite(loaded['started_monotonic']),'Material freeze must precede model load')
    req(loaded['selection_lock_sha256']==ac.sha((root/'selection_lock.json').read_bytes()),'Model input lock changed')
    req(sum(p.stat().st_size for p in root.rglob('*') if p.is_file())<=ac.PLAN['output_limit_bytes'],'Saved content exceeds 6 GiB cap')
    trusted=ac.read(Path(__file__).with_name('ac08_capsule.json'))
    equal(ac.read(root/'embedded_sha256.json'),trusted,'capsule')
    for n,h in trusted.items():req(ac.sha((root/n).read_bytes())==h,'Source '+n)
    manifest=ac.read(root/'manifest.json');env=ac.read(root/'environment.json')
    req(manifest['experiment']=='LOOP-KV-AC08' and manifest['model_revision']=='fcfcfbc429c7b10ee73dfc718033351d8936d2a8' and manifest['execution_dtype']=='BF16','Model identity')
    req(env['packages']==manifest['packages']=={k:v for k,v in ac.PLAN['packages'].items() if k!='triton'},'Environment versions')
    req(manifest['model']==ac.PLAN['model'] and manifest['tokenizer_sha256']==ac.PLAN['tokenizer_sha256'] and manifest['source_hashes']==ac.PLAN['source_hashes'],'Model source/tokenizer identity')
    for name,h in ac.PLAN['source_hashes'].items():req(ac.sha((root/'sources'/name).read_bytes())==h,'Pinned model source '+name)
    equal(ac.read(root/'fused_kernel_config.json'),dict(token_tile=16,num_warps=4,autotune=False,enable_fp_fusion=False,stages=2,partial_dtype='float32',output_dtype='bfloat16',division='div_rn',exp='tl.exp',v_input='restored BF16'),'AF01 config')
    equal(ac.read(root/'checker_config.json'),validation.CHECKER_CONFIG,'Checker config')
    validation_results=validation.audit(root/'validation');equal(ac.read(root/'validation_checks.json'),validation_results,'Validation preflight')
    from loop_kv_ac08_preflight_audit import verify as kv_audit
    kv_results=kv_audit(root/'kv_validation');equal(ac.read(root/'kv_validation_checks.json'),kv_results,'KV preflight');req(kv_results['status']=='PASS','KV preflight failed')
    req(ac.read(root/'triton_cache_initial.json')=={},'Initial Triton cache not empty')
    qenv=ac.read(root/'qk_environment.json')
    req(qenv['reference']['baseline']['triton']=='3.6.0' and qenv['token_tile']==16 and qenv['num_warps']==4 and qenv['autotune'] is False and qenv['enable_fp_fusion'] is False and qenv['division']=='div_rn','Frozen QK config')
    settings=ac.read(root/'execution_settings.json')
    req(settings['matmul_tf32'] is False and settings['cudnn_tf32'] is False and settings['float32_matmul_precision']=='highest' and settings['math_sdp_reduced'] is False and settings['gemm_bf16_reduced'] is True,'Precision settings')
    req(all(settings[x] for x in ('flash','efficient','math','cudnn')),'SDPA backend selector')
    equal(ac.read(root/'warmup_schedule.json'),ac.warmup_schedule(),'warmup schedule')
    equal(ac.read(root/'schedule.json'),ac.schedule(),'main schedule');equal(ac.read(root/'gate_schedule.json'),ac.gate_schedule(),'gate schedule')
    req(manifest['material_sha256']==ac.sha((root/'materials.json').read_bytes()),'Manifest material hash')
    docs={x['book_id']:x for x in mats['smoke']+mats['diagnostic']};observed={};max_nll_error=0.
    for sp in ac.gate_schedule()+ac.warmup_schedule()+ac.schedule():
        d=path(root,sp);result=ac.read(d/'result.json')
        req(all(result[k]==val for k,val in sp.items()) and result['status']=='COMPLETE_FINITE' and result['rng_unchanged'] is True,'Trajectory metadata')
        raw=np.load(d/'raw_logits.npy',allow_pickle=False)
        req(raw.dtype==np.uint16 and raw.shape==(sp['forwards'],manifest['vocab_size']) and ac.sha((d/'raw_logits.npy').read_bytes())==result['raw_logits_sha256'],'Raw logit shape/hash')
        req(np.isfinite(ac.bf32(raw)).all(),'Nonfinite saved logits')
        rr=ac.rows(d/'per_token.jsonl');req([r['length'] for r in rr]==list(range(769,sp['length']+1)),'Missing token predictions')
        ns=[]
        for i,r in enumerate(rr,1):
            pos=768+i;target=docs[sp['book_id']]['tokens'][pos]
            req(r['target_position']==pos and r['target_id']==target,'Target alignment')
            x=ac.bf32(raw[i]).astype(np.float64);mx=float(x.max());value=mx+math.log(float(np.exp(x-mx).sum()))-float(x[target])
            delta=abs(value-r['nll']);req(math.isfinite(r['nll']) and delta<=2e-5,'Independent NLL mismatch');max_nll_error=max(max_nll_error,delta);ns.append(r['nll'])
        equal(result['nll'],math.fsum(ns)/len(ns),'Mean NLL')
        tt=ac.rows(d/'timing.jsonl');req([(r['length'],r['operation']) for r in tt]==[(n,o) for n in range(768,sp['length']+1) for o in ('FORWARD','POST')],'Timing inventory')
        packed=sp['arm'] not in ('OFFICIAL','OFFICIAL_REPEAT','NATIVE')
        for r in tt:
            n=r['length'];req(r['step']==n-768 and r['diagnostic_step']==ac.has_diagnostic(sp,n) and r['timed_primary']==(sp['phase']=='main' and n in ac.TIMED_LENGTHS),'Timing coordinates')
            req(r['flush']==(packed and r['operation']=='POST' and n in (768,)+ac.FLUSH_LENGTHS),'Flush position')
            comp=ac.post_compressed(n,packed);req(r['compressed']==comp,'Compression count')
            for f in ('cuda_ms','wall_seconds'):req(math.isfinite(r[f]) and r[f]>=0,'Invalid time')
            for field in ('allocated','reserved'):req(type(r[field+'_before'])==type(r[field+'_peak'])==int and 0<=r[field+'_before']<=r[field+'_peak'],'Memory counters')
        states=ac.read(d/'cache_state.json');req(set(states)=={str(n) for n in ac.STATE_LENGTHS},'Cache checkpoint set')
        for n,ss in states.items():req([x['slot'] for x in ss]==list(range(192)) and all(x['length']==int(n) for x in ss),'Cache slot coordinates')
        inv=ac.read(d/'inventory.json');req([x['length'] for x in inv]==list(ac.STATE_LENGTHS),'Storage checkpoints')
        for x in inv:
            length=x['length'];comp=0 if not packed else ((length-64)//64)*64
            expected=192*2*(16*length*128*2 if not packed else (16*(comp//64)*(3072+64*2+128*4+12)+16*(length-comp)*128*2))
            req(x['compressed_tokens']==comp and x['tensor_bytes']==sum(x['component_bytes'].values())==expected,'Storage byte accounting')
            equal(x['component_bytes'],ac.nominal_components(length,packed),'Storage components')
            req(isinstance(x['unique_storage_bytes'],int) and x['unique_storage_bytes']>0 and x['devices']==['cuda:0'],'Storage devices/bytes')
            req(0<x['storage_count']<=x['tensor_count'],'Storage counts')
            if packed:req(x['unique_storage_bytes']==expected,'Unexpected packed full shadow')
        pre=ac.read(d/'prefill_precompression.json')
        req(pre['length']==768 and pre['logits_sha256']==ac.sha(raw[0].tobytes()),'Prefill logit hash')
        validate_state(pre['state'],768,False)
        for n,ss in states.items():validate_state(ss,int(n),packed)
        resident=ac.read(d/'resident.json');req(set(resident)=={'model_only','prefill_post','terminal_post'},'Resident checkpoints')
        for label,row in resident.items():
            req(set(row)=={'allocated','reserved'} and all(type(x)==int for x in row.values()) and 0<=row['allocated']<=row['reserved'],'Resident counters')
        connected=sp['arm'] not in ('OFFICIAL','OFFICIAL_REPEAT')
        req(result['grouped_checker_calls']==(sp['scored']*192 if sp['arm']=='PACKED_GROUPED' else 0),'Trajectory checker coverage')
        equal(result['connection_calls'],{str(s):sp['scored'] for s in range(192)} if connected else {},'Connected slot coverage')
        equal(result['skipped_k'],{str(s):sp['scored'] for s in range(192)} if connected and packed else {},'Lazy K coverage')
        equal(result['skipped_v'],{str(s):sp['scored'] for s in range(192)} if sp['arm']=='PACKED_KV_FUSED' else {},'Lazy V coverage')
        req(result['kv_checker_calls']==(sp['scored']*192 if sp['arm']=='PACKED_KV_FUSED' else 0),'Trajectory KV checker coverage')
        aux=ac.read(d/'auxiliary_warmup.json');equal(aux,dict(forwards=1,tokens=32,token_sha256=tensor_token_sha(docs[sp['book_id']]['tokens'][:32]),finite=True),'auxiliary warmup')
        decode_coverage(root,sp)
        observed[sp['id']]=dict(spec=sp,nlls=ns,raw=raw,result=result,timing=tt,state=states,inventory=inv,pre=pre,resident=resident)
    gates=gate_checks(root);equal(ac.read(root/'gate_checks.json'),gates,'gate results');req(all(x['pass'] for x in gates),'Gate route mismatch')
    planned={s['id'] for s in ac.gate_schedule()+ac.warmup_schedule()+ac.schedule()}
    req({p.parent.name for phase in ('gate','warmup','main') for p in (root/phase).glob('*/result.json')}==planned,'Extra completed trajectory')
    diagnostic=audit.captured_queries(root);local=diagnostic['real_query'];warm_local=diagnostic['warmup_query']
    fixture=audit.fixtures(root)
    timing_preparation=timing_preparation_check(root)
    comparisons=[];validation_pairs=[]
    for rnd in (0,1):
      for book in ac.book_ids():
        block={arm:observed[f'AC08-main{rnd}-b{book}-{arm}'] for arm in ac.ARMS}
        initial=block['NATIVE']
        req(all(np.array_equal(block[arm]['raw'][0],initial['raw'][0]) and block[arm]['pre']==initial['pre'] for arm in ac.ARMS),'Initial pre-compression state differs across methods')
        req(block['PACKED_GROUPED']['state']['768']==block['PACKED_KV_FUSED']['state']['768'],'Packed initial state differs')
        old,new=(block[a] for a in ('PACKED_GROUPED','PACKED_KV_FUSED'))
        vp=dict(book_id=book,round=rnd,logits_bitwise=bool(np.array_equal(old['raw'],new['raw'])),cache_bitwise=old['state']==new['state'],nll_equal=ac.rows(path(root,old['spec'])/'per_token.jsonl')==ac.rows(path(root,new['spec'])/'per_token.jsonl'))
        vp['match']=all(vp[k] for k in ('logits_bitwise','cache_bitwise','nll_equal'));validation_pairs.append(vp)
        windows={}
        for window in ac.WINDOWS:
            lo,hi=window['first_target'],window['last_target']
            times={arm:timing_summary(r,window['timed_lengths']) for arm,r in block.items()}
            means={arm:math.fsum(r['nlls'][lo-769:hi-768])/(hi-lo+1) for arm,r in block.items()}
            pairs=pair_comparison(block,means,times,lo,hi)
            windows[window['id']]=dict(n_scored=hi-lo+1,n_timed=len(window['timed_lengths']),nll=means,
                ppl={arm:math.exp(n) if n<700 else None for arm,n in means.items()},timing=times,pairs=pairs)
        comparisons.append(dict(book_id=book,round=rnd,**windows['ALL'],windows=windows))
    repeats=[]
    for book in ac.book_ids():
      for arm in ac.ARMS:
        x,y=(observed[f'AC08-main{i}-b{book}-{arm}'] for i in (0,1))
        repeats.append(dict(book_id=book,arm=arm,logits=ac.drift(y['raw'],x['raw']),nll_delta=y['result']['nll']-x['result']['nll'],terminal_cache_equal=x['state'][str(ac.LENGTH)]==y['state'][str(ac.LENGTH)]))
    window_aggregates={}
    for window in ac.WINDOWS:
        agg={}
        for cand,base in ac.PAIRS:
            values=[p for r in comparisons for p in r['windows'][window['id']]['pairs'] if (p['candidate'],p['reference'])==(cand,base)]
            agg[cand+'_over_'+base]={field:dict(median=statistics.median(p[field] for p in values),minimum=min(p[field] for p in values),maximum=max(p[field] for p in values),comparisons=len(values))
                for field in ('time_ratio','wall_ratio','allocated_peak_ratio','reserved_peak_ratio')}
        window_aggregates[window['id']]=agg
    aggregate=window_aggregates['ALL']
    req(not ac.rows(root/'failures.jsonl'),'Failure coexists with complete result')
    primary_quality=primary_quality_rows(comparisons)
    quality=all(p['within_five_percent'] for p in primary_quality)
    matched=all(r['match'] for r in validation_pairs)
    speed={wid:{field:dict(median_ratio=window_aggregates[wid]['PACKED_KV_FUSED_over_PACKED_GROUPED'][field]['median'],
            below_one=window_aggregates[wid]['PACKED_KV_FUSED_over_PACKED_GROUPED'][field]['median']<1)
            for field in ('time_ratio','wall_ratio')} for wid in ('ALL','W4')}
    local_ok=all(x['pass_budget'] and x['route_pass'] for x in local+warm_local)
    fixtures_ok=fixture['context_budget_pass']==fixture['calls'] and fixture['score_budget_pass']==sum(x['kind']=='full' for x in __import__('loop_kv_af01_contract').fixtures())
    hypotheses=dict(H1_connection_and_numerics=local_ok and fixtures_ok and kv_results['status']=='PASS',H2_speed=dict(windows=speed,prepared_timing_supported=timing_preparation['prepared_timing_supported']),
        H3_quality=dict(within_five_percent=quality,cases=primary_quality),H4_same_packed_storage=True)
    result=dict(status='COMPLETE_COMPARISON',experiment='LOOP-KV-AC08',n_trajectories=35,n_scored=8925,n_forwards=8960,n_auxiliary_forwards=35,n_model_calls_including_auxiliary=8995,n_real_query_samples=192,n_warmup_query_samples=24,
        packed_global_bit_match=matched,packed_global_comparisons=validation_pairs,input_validation=validation_results,kv_input_validation=kv_results,
        route_gate='PASS',local_numerics='WITHIN_BUDGET' if local_ok and fixtures_ok else 'OUTSIDE_BUDGET',quality='WITHIN_FIVE_PERCENT' if quality else 'OUTSIDE_FIVE_PERCENT',
        numerical_audit=dict(max_nll_error=max_nll_error),local_diagnostics=local,warmup_diagnostics=warm_local,fixtures=fixture,timing_preparation=timing_preparation,comparisons=comparisons,repeats=repeats,time_ratio_summary=aggregate,
        window_time_ratio_summary=window_aggregates,primary_quality=primary_quality,hypotheses=hypotheses,
        material_selection=dict(selected_ids=ac.book_ids(),known_materials=False,selection_basis='Fixed candidate order after excluding 24 prior books',frozen_before_model_load=True,candidate_file_sha256=m.CANDIDATE_SHA),
        descriptive=descriptive_summaries(comparisons),
        note='Four newly selected books, two technical repeats per method. No CI or p-value; no pooling of historical outcomes. ALL and W4 quality use KV_FUSED versus NATIVE. Whole implementation contrast includes validation-input and rounding differences. Packed global bit identity is descriptive, not a pass requirement. Finite failures retained. Prefill-inclusive peak covers logged FORWARD/POST intervals, not the entire process.')
    if check_summary:
        equal(ac.read(root/'summary.json'),result,'summary');req(ac.read(root/'run_status.json')['status']=='COMPLETE_COMPARISON','Run status')
    return result
if __name__=='__main__':
    import traceback
    root=Path(sys.argv[1])
    try:
        result=verify(root);ac.write(root/'completion_verification.json',dict(status='VERIFIED',n_scored=result['n_scored'],n_forwards=result['n_forwards'],local_numerics=result['local_numerics'],quality=result['quality']));print('AC08 independent verification: VERIFIED')
    except Exception as e:
        ac.write(root/'completion_verification.json',dict(status='FAILED',error=str(e)));print(traceback.format_exc());sys.exit(1)
