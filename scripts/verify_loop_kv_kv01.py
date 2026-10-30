"""Offline audit of a complete KV01 archive; no returned code or model is run.

Usage: python3 verify_loop_kv_kv01.py RESULT.zip STUDY_ROOT OUTPUT.json
Requires NumPy only. Numeric tolerances reconcile stored float32 computations;
they do not replace the preregistered research thresholds.
"""
import ast
from collections import Counter, defaultdict
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import sys
import zipfile
import numpy as np


def sha(x):
    return hashlib.sha256(x).hexdigest()


def mean(x):
    return math.fsum(x) / len(x)


def close(a, b, tol=1e-12):
    assert math.isfinite(a) and math.isfinite(b) and abs(a-b) <= tol, (a, b, tol)


def verify(archive, study):
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        assert len(names)<=1000 and sum(i.file_size for i in z.infolist())<=10*1024**3
        assert all(not n.endswith('/') and '\\' not in n for n in names)
        assert all(((i.external_attr>>16)&0o170000)!=0o120000 for i in z.infolist())
        assert len(names) == len(set(names))
        assert all(not PurePosixPath(n).is_absolute() and '..' not in PurePosixPath(n).parts for n in names)
        data = {n:z.read(n) for n in names}
    get = lambda n: json.loads(data[n])
    rows = lambda n: [json.loads(s) for s in data[n].decode().splitlines()]
    hashes = get('SHA256SUMS.json')
    assert set(data) == set(hashes) | {'SHA256SUMS.json'}
    assert all(sha(data[n]) == h for n,h in hashes.items())
    manifest, materials = get('manifest.json'), get('materials.json')
    for n,key in [('loop_kv_pilot.py','core_sha256'), ('loop_kv_p0_diagnostic.py','diagnostic_helper_sha256'), ('loop_kv_kv01.py','worker_sha256')]:
        assert sha(data[n]) == manifest[key] == sha((study/'scripts'/n).read_bytes())
    assert sha(data['materials.json']) == manifest['material_sha256'] == sha(data['loop_kv_p0r_materials.json']) == sha((study/'scripts/loop_kv_p0r_materials.json').read_bytes())
    identity=get('loop_kv_p1b_identity.json')
    assert sha(data['loop_kv_p1b_identity.json'])==manifest['historical_identity_sha256']==sha((study/'scripts/loop_kv_p1b_identity.json').read_bytes())
    assert sha((study/'runs'/identity['run_id']/'original.zip').read_bytes())==identity['archive_sha256']
    assert sha((study/'runs'/identity['run_id']/'artifacts/loop_kv_p0rb_p1b.py').read_bytes())==identity['worker_sha256']
    assert get('summary.json')['historical_parent']==identity and get('summary.json')['historical_outcomes_pooled'] is False
    parent = study/'runs'/manifest['material_source']['run_id']
    assert sha((parent/'original.zip').read_bytes()) == manifest['material_source']['original_zip_sha256']
    assert materials == json.loads((parent/'artifacts/materials.json').read_text())
    original_manifest = json.loads((parent/'artifacts/manifest.json').read_text())
    for key in ['packages','gpu','tokenizer_sha256','model_revision']:
        assert manifest[key] == original_manifest[key]
    for n,h in manifest['source_hashes'].items(): assert sha(data['sources/'+n]) == h
    assert manifest['execution_dtype'] == 'BF16'
    assert manifest['parameter_dtype_counts'] == {'torch.bfloat16':269}
    assert manifest['floating_buffer_dtype_counts'] == {'torch.float32':1}
    assert get('execution_settings.json') == dict(flash=True,efficient=True,math=True,cudnn=True,gemm_bf16_reduced=True,math_sdp_reduced=False,matmul_tf32=False,cudnn_tf32=False)
    assert manifest['readout'] == 'exit_at_step=r-1; labels=None; use_weighted_exit=False'
    # Compare schedules and embedded preregistration to the delivered cell without executing it.
    literals = {}
    for node in ast.parse((study/'colab/loop_kv_kv01_onecell.py').read_text()).body:
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name):
            name = node.targets[0].id
            if name in ('EXPECTED_SCHEDULE','EXPECTED_P0_SCHEDULE','EXPECTED_COMPATIBILITY_SCHEDULE'):
                literals[name] = json.loads(ast.literal_eval(node.value.args[0]))
            elif name in ('DESIGN_SOURCE','PREREG_SOURCE'):
                literals[name] = ast.literal_eval(node.value)
    assert data['design.md'].decode() == literals['DESIGN_SOURCE']
    assert data['prereg.md'].decode() == literals['PREREG_SOURCE']
    schedule, p0schedule = get('schedule.json'),get('p0r/schedule.json')
    assert schedule == literals['EXPECTED_SCHEDULE'] and p0schedule == literals['EXPECTED_P0_SCHEDULE']
    assert get('p0r/compatibility_schedule.json')==literals['EXPECTED_COMPATIBILITY_SCHEDULE']
    books = [x['book_id'] for x in materials['diagnostic']]
    assert books == ['3752','31131','31258','34475','4124','49966','49484','4537']
    tokens = {d['book_id']:d['tokens'] for part in ['smoke','diagnostic'] for d in materials[part]}
    for part in ['smoke','diagnostic']:
        for doc in materials[part]:
            assert len(doc['tokens']) == 1024 and doc['offset'] == 0
            assert sha(json.dumps(doc['tokens'],separators=(',',':')).encode()) == doc['token_sha256']
    targets=['NATIVE','K_ONLY','V_ONLY','KV']
    selected={'NATIVE':(), 'K_ONLY':('K',), 'V_ONLY':('V',), 'KV':('K','V')}
    assert len(schedule)==32 and {(s['book_id'],s['r'],s['mode'],s['target']) for s in schedule} == {(b,8,'PREFIX_ONLY',q) for b in books for q in targets}
    assert all(s['bits']==(None if s['target']=='NATIVE' else 3) for s in schedule)

    # P0R route, coordinate and original-tolerance diagnostics.
    cases, checks, trace, p0tokens = [rows(n) for n in ['p0r/cases.jsonl','p0r/checks.jsonl','p0r/cache_trace.jsonl','p0r/per_token.jsonl']]
    ids = [s['id'] for s in p0schedule]
    assert [c['id'] for c in cases] == ids and [t['case_id'] for t in trace] == ids
    assert Counter((c['case_id'],c['route']) for c in checks) == Counter((i,r) for i in ids for r in ['official_repeat','adapter','adapter_instrumented'])
    pt = defaultdict(list)
    for t in p0tokens: pt[t['case_id']].append(t)
    p0_recomputed=[]
    for spec,case,tr in zip(p0schedule,cases,trace):
        assert all(case[k]==v for k,v in spec.items()) and case['status']=='PASS'
        assert case['forwards']==spec['expected_forwards']
        for c in [c for c in checks if c['case_id']==case['id']]:
            assert c['status']=='PASS' and all(c[k] for k in ['logits_bitwise','kv_bitwise','rng_unchanged'])
            assert c['reference_logit_sha256']==c['compared_logit_sha256']
            assert c['trace_updates']==(24*spec['r']*spec['steps'] if c['route']=='adapter_instrumented' else 0)
        slots=24*spec['r']; assert len(tr['records'])==slots*spec['steps']
        for step in range(spec['steps']):
            old,added=(0,spec['prefill']) if step==0 else (spec['prefill']+step-1,1)
            for slot,row in enumerate(tr['records'][step*slots:(step+1)*slots]):
                assert row==dict(slot=slot,loop=slot//24,layer=slot%24,old=old,added=added,new=old+added)
        if not spec['full']:
            assert not pt[spec['id']]
            continue
        ts=pt[spec['id']]; assert len(ts)==spec['length']
        tok=tokens[spec['book_id']][:spec['length']] if spec['book_id'] else [(7+13*i)%49152 for i in range(spec['length'])]
        for i,t in enumerate(ts):
            assert t['input_position']==i and t['input_token']==tok[i]
            assert t['target_position']==(i+1 if i<len(ts)-1 else None)
            assert t['target_id']==(tok[i+1] if i<len(ts)-1 else None)
            assert math.isfinite(t['kl']) and math.isfinite(t['max_abs_logit'])
            if i<len(ts)-1: close(t['delta_nll'],t['incremental_nll']-t['full_nll'])
        delta=mean([t['delta_nll'] for t in ts[:-1]]) if len(ts)>1 else None
        rec=dict(id=spec['id'],kl=mean([t['kl'] for t in ts]),delta_nll=delta,max_abs_logit=max(t['max_abs_logit'] for t in ts))
        old=case['full_incremental']
        # GPU float32 mean vs exact summation of stored float32 terms.
        # Bound sequential summation plus one division using gamma_(n+1).
        eps=float(np.finfo(np.float32).eps)
        gamma=(len(ts)+1)*eps/(1-(len(ts)+1)*eps)
        bound=gamma*mean([abs(t['kl']) for t in ts])
        close(rec['kl'],old['kl'],bound)
        rec['kl_reduction_residual']=abs(rec['kl']-old['kl'])
        rec['kl_float32_rounding_bound']=bound
        close(rec['max_abs_logit'],old['max_abs_logit'])
        if delta is not None: close(delta,old['delta_nll'],3e-6)
        else: assert old['delta_nll'] is None and old['nll_test']=='NOT_APPLICABLE'
        rec['within_original_tolerance']=rec['kl']<=1e-4 and rec['max_abs_logit']<=.25 and (delta is None or abs(delta)<=.005)
        assert rec['within_original_tolerance']==old['within_original_tolerance']
        p0_recomputed.append(rec)
    p0summary=get('p0r/summary.json')
    assert p0summary['p0r_route_status']=='PASS_ROUTE_KV01' and p0summary['passed_cases']==15 and p0summary['route_comparisons']==45
    assert sum(c['forwards'] for c in cases)==p0summary['obtained_forwards']==5963
    assert p0summary['fixtures']==get('p0r/fixtures.json') and all(x['status']=='PASS' for x in p0summary['fixtures'])
    assert p0summary['gate_error'] is None and not p0summary['not_run_ids']

    compatibility=rows('p0r/compatibility.jsonl')
    assert [c['id'] for c in compatibility]==[c['id'] for c in literals['EXPECTED_COMPATIBILITY_SCHEDULE']]
    for spec,row in zip(literals['EXPECTED_COMPATIBILITY_SCHEDULE'],compatibility):
        assert all(row[k]==v for k,v in spec.items()) and row['status']=='PASS'
        assert all(row[k] is True for k in ('logits_bitwise','kv_bitwise','rng_unchanged'))
        assert row['old_logit_sha256']==row['new_logit_sha256'] and row['forwards']==512
    assert p0summary['compatibility_planned']==p0summary['compatibility_passed']==4
    assert not p0summary['compatibility_not_run_ids'] and p0summary['compatibility_forwards']==2048

    docs=rows('per_document.jsonl'); ts=rows('per_token.jsonl'); counters=rows('cache_counters.jsonl')
    assert [x['id'] for x in docs]==[x['id'] for x in schedule]==[x['id'] for x in counters]
    grouped=defaultdict(list)
    for t in ts:grouped[t['condition_id']].append(t)
    assert set(grouped)=={x['id'] for x in schedule} and len(ts)==8160
    lookup={(x['book_id'],x['target']):x for x in docs}
    rec_nll={}; npz_residuals=[]; per_book=[]
    for spec,doc,ctr in zip(schedule,docs,counters):
        assert all(doc[k]==v for k,v in spec.items())
        assert doc['status']=='COMPLETE_FINITE' and doc['rng_unchanged'] and doc['scored']==255 and doc['forwards']==256
        assert doc['slots']==24*doc['r'] and doc['update_count']==24*doc['r']*256
        assert all(doc[k]==v for k,v in ctr.items())
        tt=grouped[doc['id']]
        native=lookup[(doc['book_id'],'NATIVE')]; nt=grouped[native['id']]
        assert len(tt)==255
        for j,(t,n) in enumerate(zip(tt,nt)):
            assert all(t[k]==doc[k] for k in ['book_id','r','mode','bits','target'])
            assert t['input_position']==768+j and t['target_position']==769+j and t['target_id']==tokens[doc['book_id']][769+j]
            assert t['status']=='FINITE' and math.isfinite(t['target_log_probability'])
            if doc['bits'] is None:
                assert all(t[k] is None for k in ['native_target_log_probability','kl_native_to_condition','max_abs_logit_vs_native'])
            else:
                assert t['native_target_log_probability']==n['target_log_probability']
                assert all(math.isfinite(t[k]) for k in ['kl_native_to_condition','max_abs_logit_vs_native'])
        nll=-mean([t['target_log_probability'] for t in tt]);rec_nll[doc['id']]=nll
        close(nll,doc['nll']);close(math.exp(nll),doc['ppl'])
        if doc['bits'] is not None:close(mean([t['kl_native_to_condition'] for t in tt]),doc['mean_kl'])
        compressed=0 if doc['bits'] is None else (704 if doc['mode']=='PREFIX_ONLY' else 896)
        assert doc['compressed_tokens']==compressed and doc['implementation']=='FAKE_QUANT_FLOAT_CACHE'
        sideparts={}
        for kv in ('K','V'):
            span=704 if kv in selected[doc['target']] else 0
            blocks=192*16*(span//64)
            sideparts[kv]=dict(codes=blocks*64*128*3//8,metadata=blocks*(64*2+128*4),
                native=192*16*(1023-span)*128*2,block_headers=blocks*12)
        parts={k:sum(v[k] for v in sideparts.values()) for k in ('codes','metadata','native','block_headers')}
        assert doc['planned_components']==parts and doc['planned_components_by_kv']==sideparts
        assert doc['planned_bytes']==sum(parts.values()) and doc['floating_cache_bytes']==192*2*16*1023*128*2
        assert doc['compressed_tokens_by_kv']=={kv:704 if kv in selected[doc['target']] else 0 for kv in ('K','V')}
        with np.load(io.BytesIO(data[f'diagnostics/{doc["id"]}-p896.npz']),allow_pickle=False) as z:
            assert int(z['target_position'])==896 and int(z['target_id'])==tokens[doc['book_id']][896]
            for k in ['log_probs','native_log_probs','logits','native_logits']:assert z[k].shape==(49152,) and np.isfinite(z[k]).all()
            row=tt[127];assert float(z['log_probs'][int(z['target_id'])])==row['target_log_probability']
            targetnative=nt[127]['target_log_probability'];assert float(z['native_log_probs'][int(z['target_id'])])==targetnative
            a=z['logits'].astype(np.float64); normalizer=float(a.max()+np.log(np.exp(a-a.max()).sum()))
            residual=float(np.abs((a-normalizer)-z['log_probs']).max());assert residual<1e-5
            npz_residuals.append(residual)
            with np.load(io.BytesIO(data[f'diagnostics/{native["id"]}-p896.npz']),allow_pickle=False) as nz:
                assert np.array_equal(z['native_log_probs'],nz['log_probs']) and np.array_equal(z['native_logits'],nz['logits'])
            if doc['bits'] is not None:
                ref=z['native_log_probs'].astype(np.float64);lp=z['log_probs'].astype(np.float64)
                close(float(np.sum(np.exp(ref)*(ref-lp))),row['kl_native_to_condition'],1e-7)
                close(float(np.abs(z['logits']-z['native_logits']).max()),row['max_abs_logit_vs_native'])
            else:assert np.array_equal(z['log_probs'],z['native_log_probs'])
        per_book.append(dict(book_id=doc['book_id'],target=doc['target'],r=doc['r'],mode=doc['mode'],bits=doc['bits'],nll=nll,ppl=math.exp(nll),delta_nll=nll-rec_nll[native['id']],mean_kl=doc['mean_kl']))
    # Norm coordinates / summaries and NSN norm preservation diagnostics.
    norm_summaries=[]
    for prefix,identkey,definitions in [('', 'condition_id',docs),('p0r/','case_id',[c for c in cases if c['group']=='G4'])]:
        norms=rows(prefix+'norm_snapshots.jsonl');transfers=rows(prefix+'norm_transfers.jsonl')
        assert set(x[identkey] for x in norms)==set(c['id'] for c in definitions)==set(x[identkey] for x in transfers)
        for c in definitions:
            mode=c.get('mode','PREFIX_ONLY');slots=[24*l+k for l in range(c['r']) for k in [0,12,23]]
            n=[x for x in norms if x[identkey]==c['id']];tr=[x for x in transfers if x[identkey]==c['id']]
            assert Counter((x['slot'],x['kv'],x['cache_length']) for x in n)==Counter((s,k,L) for s in slots for k in ['K','V'] for L in [768,832,896,960,1023])
            lengths=[768,832,896,960] if mode=='DECODE_ONLINE' else [768]
            assert Counter((x['slot'],x['kv'],x['cache_length']) for x in tr)==Counter((s,k,L) for s in slots for k in ['K','V'] for L in lengths)
            for x in n:
                assert x['count']==16*x['cache_length'] and x['nonfinite']==0
                assert x['loop']==x['slot']//24 and x['layer']==x['slot']%24
                assert 0<=x['p50']<=x['p99']<=x['maximum'] and math.isfinite(x['maximum'])
            for x in tr:
                span=(0,704) if x['cache_length']==768 else (x['cache_length']-128,x['cache_length']-64)
                assert (x['start'],x['end_exclusive'])==span and x['quantization_applied']==(x['kv'] in selected[c.get('target','NATIVE')])
                for part in ['before','after']:
                    p=x[part];assert p['count']==16*(span[1]-span[0]) and p['nonfinite']==0 and 0<=p['p50']<=p['p99']<=p['maximum'] and math.isfinite(p['maximum'])
                assert math.isfinite(x['max_relative_norm_change']) and math.isfinite(x['max_absolute_norm_change'])
                if not x['quantization_applied']:assert x['before']==x['after'] and x['max_relative_norm_change']==x['max_absolute_norm_change']==0
        if not prefix:
            for target in targets:
                idset={c['id'] for c in docs if c['target']==target}
                n=[x for x in norms if x[identkey] in idset];t=[x for x in transfers if x[identkey] in idset]
                norm_summaries.append(dict(target=target,snapshots=len(n),transfers=len(t),nonfinite=0,
                    max_relative_transfer_change=max(x['max_relative_norm_change'] for x in t),
                    max_absolute_transfer_change=max(x['max_absolute_norm_change'] for x in t),
                    maximum_K_norm=max(x['maximum'] for x in n if x['kv']=='K'),maximum_V_norm=max(x['maximum'] for x in n if x['kv']=='V')))

    # All-slot intervention records: independently map expected K/V selection.
    intervention_summaries=[]
    for filename,idkey,definitions in [('interventions.jsonl','condition_id',docs),('p0r/compat_interventions.jsonl','case_id',compatibility)]:
        ii=rows(filename)
        assert set(x[idkey] for x in ii)=={c['id'] for c in definitions}
        for c in definitions:
            target=c.get('target','KV');items=[x for x in ii if x[idkey]==c['id']]
            assert Counter((x['slot'],x['kv']) for x in items)==Counter((slot,kv) for slot in range(192) for kv in ('K','V'))
            for item in items:
                active=item['kv'] in selected[target]
                assert item['selected']==active and item['cache_length']==768
                assert item['start']==0 and item['end_exclusive']==704 and item['protected_tail_bitwise'] is True
                assert math.isfinite(item['max_abs_change']) and math.isfinite(item['relative_l2_change'])
                assert item['max_abs_change']>=0 and item['relative_l2_change']>=0
                if not active:
                    assert item['full_tensor_bitwise'] is True
                    assert item['max_abs_change']==item['relative_l2_change']==0
        intervention_summaries.append(dict(file=filename,count=len(ii),selected_slices=sum(x['selected'] for x in ii),
            unselected_slices=sum(not x['selected'] for x in ii),protected_tail_violations=0,unselected_violations=0))

    summary=get('summary.json');comparisons=[];cond=[];margin=math.log(1.05)
    def value(book,target):return rec_nll[lookup[(book,target)]['id']]
    def bootstrap(a):
        a=np.asarray(a,dtype=np.float64)
        indices=np.random.default_rng(2026093002).integers(0,len(a),size=(10000,len(a)))
        return np.quantile(a[indices].mean(axis=1),[.025,.975]).tolist()
    for target in targets:
        nlls=[value(b,target) for b in books]
        deltas=[value(b,target)-value(b,'NATIVE') for b in books]
        nll=mean(nlls);delta=mean(deltas)
        cond.append(dict(target=target,n=8,mean_nll=nll,ppl=math.exp(nll),delta_nll=delta,
            ppl_change_percent=100*math.expm1(delta),
            mean_kl=None if target=='NATIVE' else mean([lookup[(b,target)]['mean_kl'] for b in books])))
        c=next(x for x in summary['condition_nll'] if x['target']==target)
        assert c['n']==8 and c['books']==books;close(c['mean_nll'],nll);close(c['ppl'],math.exp(nll))
        if target=='NATIVE':continue
        ci=bootstrap(deltas)
        decision='WITHIN_5PCT' if ci[1]<=margin else 'EXCEEDS_5PCT' if ci[0]>margin else 'UNCERTAIN'
        q=next(x for x in summary['quality_screens'] if x['target']==target)
        assert q['n']==8 and q['paired_books']==books and not q['missing_book_ids']
        assert q['status']==decision and q['books_above_5pct']==sum(d>margin for d in deltas)
        for a,b in zip(q['per_book_delta_nll'],deltas):close(a,b)
        for a,b in zip(q['per_book_relative_ppl'],deltas):close(a,math.expm1(b))
        for a,b in zip(q['bootstrap_95'],ci):close(a,b)
        for a,b in zip(q['relative_ppl_95'],ci):close(a,math.expm1(b))
        close(q['mean_delta_nll'],delta);close(q['relative_ppl'],math.expm1(delta));close(q['max_relative_ppl'],math.expm1(max(deltas)))
        comparisons.append(dict(target=target,books=books,per_book_delta_nll=deltas,mean=delta,bootstrap_95=ci,
            ppl_change_percent=100*math.expm1(delta),ppl_change_percent_95=[100*math.expm1(v) for v in ci],
            worsened_books=sum(v>0 for v in deltas),improved_books=sum(v<0 for v in deltas),
            max_book_ppl_change_percent=100*math.expm1(max(deltas)),books_above_5pct=sum(v>margin for v in deltas),status=decision))
    contrasts={}
    for key,needed,coefficients,positive,negative in [('D',['K_ONLY','V_ONLY'],[1,-1],'K_LARGER','V_LARGER'),
            ('I',['KV','K_ONLY','V_ONLY','NATIVE'],[1,-1,-1,1],'SUPERADDITIVE','SUBADDITIVE')]:
        vals=[math.fsum(c*value(b,t) for c,t in zip(coefficients,needed)) for b in books]
        m=mean(vals);ci=bootstrap(vals)
        decision=positive if m>=.005 else negative if m<=-.005 else 'WITHIN_BAND'
        got=summary[key]
        assert got['n']==8 and got['paired_books']==books and not got['missing_book_ids'] and got['status']==decision
        for a,b in zip(vals,got['values']):close(a,b)
        for a,b in zip(ci,got['bootstrap_95']):close(a,b)
        close(got['mean'],m)
        contrasts[key]=dict(status=decision,mean=m,bootstrap_95=ci,values=vals,books=books,
            positive_books=sum(v>0 for v in vals),negative_books=sum(v<0 for v in vals),threshold=.005)
    status=get('run_status.json')
    assert status['status']==summary['run_status']==summary['overall_status']=='COMPLETE'
    assert summary['planned']==summary['obtained']==32 and summary['status_counts']=={'COMPLETE_FINITE':32}
    assert not summary['not_run_ids'] and not summary['unresolved_ids']
    assert summary['observed_token_count']==summary['planned_token_count']==8160
    assert summary['independent_books']==8 and summary['norm_snapshot_count']==7680
    assert summary['norm_transfer_count']==1536 and summary['intervention_slice_count']==12288
    assert len(rows('norm_snapshots.jsonl'))==7680 and len(rows('norm_transfers.jsonl'))==1536
    assert len(rows('p0r/norm_snapshots.jsonl'))==960 and len(rows('p0r/norm_transfers.jsonl'))==192
    assert len(rows('interventions.jsonl'))==12288 and len(rows('p0r/compat_interventions.jsonl'))==1536
    assert get('launcher_status.json')['worker_exit_code']==0
    assert status['p0r_route_status']==summary['p0r_route_status']=='PASS_ROUTE_KV01' and status['p1_started']
    for obj in [status,summary,p0summary]:
        assert obj['legacy_p0_status']=='BLOCKED_IMPLEMENTATION_RETAINED' and obj['legacy_full_p0_pass'] is False
    assert not any('failure' in n or 'error' in n or 'timeout' in n for n in data)
    # Historical anchor comparison is descriptive; do not pool results.
    oldroot=study/'runs'/identity['run_id']/'artifacts'
    olddocs=[json.loads(x) for x in (oldroot/'per_document.jsonl').read_text().splitlines()]
    oldtokens=[json.loads(x) for x in (oldroot/'per_token.jsonl').read_text().splitlines()]
    oldgroup=defaultdict(list)
    for t in oldtokens:oldgroup[t['condition_id']].append(t)
    anchors=[]
    for book in books:
        for target,bits in [('NATIVE',None),('KV',3)]:
            previous=next(x for x in olddocs if x['book_id']==book and x['r']==8 and x['mode']=='PREFIX_ONLY' and x['bits']==bits)
            current=lookup[(book,target)]
            a,b=oldgroup[previous['id']],grouped[current['id']]
            exact=all(x['target_log_probability']==y['target_log_probability'] and x['target_position']==y['target_position'] and x['target_id']==y['target_id'] for x,y in zip(a,b)) and len(a)==len(b)==255
            anchors.append(dict(book_id=book,target=target,token_scores_identical=exact,
                nll_difference=current['nll']-previous['nll'],
                distribution_hash_identical=current['distribution_sha256']==previous['distribution_sha256'],
                final_cache_hash_identical=current['final_cache_sha256']==previous['final_cache_sha256']))
    # The previous bootstrap used sorted book IDs; this card uses material order.
    oldsummary=json.loads((oldroot/'summary.json').read_text())
    oldq=next(x for x in oldsummary['quality_screens'] if x['r']==8 and x['bits']==3)
    delta_by_book={b:value(b,'KV')-value(b,'NATIVE') for b in books}
    historical_order_ci=bootstrap([delta_by_book[b] for b in oldq['paired_books']])
    old_ci_match=all(abs(a-b)<1e-12 for a,b in zip(historical_order_ci,oldq['bootstrap_95']))
    memory=[dict(target=t,planned_bytes=lookup[(books[0],t)]['planned_bytes'],floating_cache_bytes=lookup[(books[0],t)]['floating_cache_bytes']) for t in targets]
    return dict(verification_status='PASS',run_id=manifest['run_id'],archive_sha256=sha(archive.read_bytes()),
        archive_files=len(data),hashes_verified=len(hashes),expanded_bytes=sum(map(len,data.values())),
        checks_verified=49,native_cases=15,compatibility_cases=4,p0_trace_updates=sum(len(t['records']) for t in trace),
        p0_forward_calls=8011,p0_original_threshold_inside=sum(x['within_original_tolerance'] for x in p0_recomputed),
        p0_original_threshold_outside=sum(not x['within_original_tolerance'] for x in p0_recomputed),p0_diagnostics=p0_recomputed,
        trajectories=32,predictions=8160,forward_calls=8192,independent_books=8,elapsed_seconds=status['elapsed_seconds'],
        gate_case_seconds=sum(c['seconds'] for c in cases+compatibility),trajectory_seconds=sum(c['seconds'] for c in docs),
        conditions=cond,comparisons=comparisons,**contrasts,norm_summaries=norm_summaries,intervention_summaries=intervention_summaries,
        diagnostic_distributions_checked=32,max_diagnostic_logsoftmax_residual=max(npz_residuals),per_book=per_book,memory=memory,
        historical_anchors=anchors,historical_interval_note=dict(old_order=oldq['paired_books'],current_order=books,
            old_order_bootstrap_matches_previous=old_ci_match,old_order_ci=historical_order_ci),
        scope='Original token scores independently aggregated; archive/code/material hashes, schedules, native references, fixed-point distributions, all-slot intervention evidence, norm coordinates and side-specific byte arithmetic checked. Raw full KV tensors and all-token vocabulary distributions are not archived; runtime bitwise checks and hashes are evidence, not a new independent GPU execution. No returned code is imported.',
        audit_tolerances=dict(document_nll=1e-12,p0_mean_kl='gamma_(n+1) * mean(abs(float32 terms)); eps=2**-23',p0_delta_nll=3e-6,saved_distribution=1e-5,saved_kl=1e-7),
        audit_script_sha256=sha(Path(__file__).read_bytes()))


if __name__=='__main__':
    archive,study,output=map(Path,sys.argv[1:])
    result=verify(archive,study)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['verification_status','run_id','archive_files','checks_verified','trajectories','predictions','D','I','conditions','historical_interval_note']},ensure_ascii=False,indent=2))
