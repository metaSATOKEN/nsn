"""NB05 independent NumPy audit: no torch, model, Triton or Hub imports."""
import json,math,sys
from pathlib import Path
import numpy as np
import loop_kv_nb05_contract as c
import loop_kv_nb05_numeric as nc
from loop_kv_nb05_diagnostic import diagnostic
import loop_kv_nb05_materials as materials

def artifacts(root,name,digest):
    p=root/c.safe_source_name(name);c.require(p.is_file() and c.file_digest(p)==digest,'Artifact hash '+name)
    return c.load(p)

def storage_check(inv,arm,length):
    packed=arm in ('PACKED_SDPA','GQA_FUSED')
    want=dict(c.PLAN['resident_bytes']['packed_components'],native_tail=2*44*8*(length-64)*128*2) if packed else {'float_kv':2*44*8*length*128*2}
    count=528 if packed else 88
    return bool(inv['component_bytes']==want and inv['tensor_bytes']==sum(want.values()) and inv['unique_storage_bytes']==sum(want.values())
        and inv['tensor_count']==inv['storage_count']==inv['all_attribute_tensor_count']==count
        and inv['all_attribute_unique_storage_bytes']==inv['unique_storage_bytes'] and inv['unexpected_persistent_tensor'] is False
        and inv['slot_lengths']==[length]*44 and inv['compressed_by_slot']==[64 if packed else 0]*44
        and inv['encoded_blocks_by_slot']==[1 if packed else 0]*44)

def route_check(row,spec):
    step=row['step'];arm=spec['arm'];adapter=arm!='OFFICIAL';packed=arm in ('PACKED_SDPA','GQA_FUSED')
    expected={'calls':[step+1]*44,'connections':[step if adapter else 0]*44,
              'skipped_k':[step if packed else 0]*44,'skipped_v':[step if packed else 0]*44}
    extra=12 if spec['phase']=='diagnostic' and step in (1,8) else 0
    return row['routes']==expected and row['attention_calls']==row['input_scan_calls']==(44 if step and adapter else 0)+extra

def timing_check(tm):
    c.require(all(type(tm[x]) in (int,float) and math.isfinite(tm[x]) and tm[x]>=0 for x in ['cuda_ms','wall_ms','started_monotonic','finished_monotonic']),'Timing finite/nonnegative')
    c.require(tm['cuda_ms']>0 and tm['wall_ms']>0 and tm['finished_monotonic']>=tm['started_monotonic'],'Timing positive/order')
    for kind in ['allocated','reserved']:c.require(type(tm[kind+'_before']) is int and type(tm[kind+'_peak']) is int and tm[kind+'_peak']>=tm[kind+'_before']>=0,'Memory before/peak')

def quality(rows):
    out={}
    for label,start in [('post_compression_8',1),('all_9',0)]:
        means={a:[float(np.mean([x['nll'] for x in rows[f'main-r{r}-'+a.lower()][start:]])) for r in range(3)] for a in c.SERIES}
        med={a:float(np.median(v)) for a,v in means.items()}
        contrasts={}
        for a,b in [('PACKED_SDPA','NATIVE'),('GQA_FUSED','NATIVE'),('GQA_FUSED','PACKED_SDPA')]:
            dn=med[a]-med[b];pct=100*math.expm1(dn) if dn<700 else None
            contrasts[a+'_vs_'+b]=dict(delta_nll=dn,relative_ppl_percent=pct,relative_ppl_overflow=pct is None,
                reference_5pct='WITHIN_5PCT_ON_THIS_TEXT' if pct is not None and pct<=5 else 'EXCEEDS_5PCT_ON_THIS_TEXT')
        out[label]=dict(positions=list(range(128+start,137)),mean_nll_by_round=means,median_mean_nll=med,
            ppl={a:math.exp(x) if x<700 else None for a,x in med.items()},contrasts=contrasts)
    return out

def performance(rows):
    out={}
    for field in ['cuda_ms','wall_ms']:
        cells={}
        for a in c.SERIES:
            cells[a]=[]
            for r in range(3):
                rr=rows[f'main-r{r}-'+a.lower()];prefill=rr[0]['model_timing'][field]
                compress=rr[0]['compression_timing'][field] if rr[0]['compression_timing'] else 0.
                decode=sum(x['model_timing'][field] for x in rr[1:])
                cells[a].append(dict(round=r,prefill_ms=prefill,compression_ms=compress,decode_ms=decode,phase_sum_ms=prefill+compress+decode))
        ratios={}
        for base in ('NATIVE','PACKED_SDPA'):
            vals=[cells['GQA_FUSED'][r]['decode_ms']/cells[base][r]['decode_ms'] for r in range(3)]
            ratios[base]=dict(ratios_by_round=vals,median_ratio=float(np.median(vals)),shorter_rounds=sum(x<1 for x in vals),rounds=3)
        out[field]=dict(phase_sum_fused_vs={base:dict(ratios_by_round=[cells['GQA_FUSED'][r]['phase_sum_ms']/cells[base][r]['phase_sum_ms'] for r in range(3)],median_ratio=float(np.median([cells['GQA_FUSED'][r]['phase_sum_ms']/cells[base][r]['phase_sum_ms'] for r in range(3)]))) for base in ('NATIVE','PACKED_SDPA')},by_arm=cells,median_decode_ms={a:float(np.median([x['decode_ms'] for x in rr])) for a,rr in cells.items()},fused_vs=ratios)
    return out

def memory(rows):
    return {a:{kind:{part:[max(x['model_timing'][kind+'_peak'] for x in rows[f'main-r{r}-'+a.lower()][(1 if part=='decode' else 0):(9 if part=='decode' else 1)]) for r in range(3)] for part in ['prefill','decode']} for kind in ['allocated','reserved']} for a in c.SERIES}

def verify_environment(root):
    import loop_kv_nb05_runtime as rt
    runtime=c.read(root/'runtime_after.json');plan=c.read(root/'runtime_plan.json');gpu=c.read(root/'runtime_cuda_probe.json')
    c.require(runtime['packages']==rt.PINS==plan['pins'] and runtime['installer_pip']==rt.INSTALLER_PIP and runtime['user_site'] is False and runtime['prefix']!=runtime['base_prefix'],'Isolated pinned runtime')
    c.require(gpu['status']=='AVAILABLE' and gpu['bf16_supported'] is True,'CUDA capability')
    conf=c.read(root/'loaded_config.json')
    for key,want in dict(num_hidden_layers=22,num_loops=2,num_attention_heads=48,num_key_value_heads=8,head_dim=128,vocab_size=166144).items():c.require(conf[key]==want,'Loaded configuration '+key)
    for key in ['enable_double_loop_split','loop_share_kv','enable_depth_attention','enable_hyper_connection']:c.require(not conf.get(key,False),'Unsupported loaded feature')
    files=c.read(root/'hub_files.json');metadata=c.read(root/'nb05_model_files.json')
    c.require(len(files)==len(metadata['files']) and {x['file'] for x in files}==set(metadata['files']),'Hub inventory')
    for x in files:
        spec=metadata['files'][x['file']];c.require(x['bytes']==spec['size'],'Hub byte count')
        digest=spec.get('lfs',{}).get('sha256',c.PLAN['public_source_sha256'].get(x['file']))
        if digest:c.require(x['sha256']==digest,'Hub digest')
    return runtime['packages']

def verify_book(root,tokens):
    root=Path(root)
    progress=c.read(root/'progress.json');c.require(progress['attempted']==progress['finished']==c.PLAN['forward_calls_per_book'],'Model forward count')
    records=c.rows(root/'trajectories.jsonl');c.require(len(records)==15 and [{key:x[key] for key in spec} for x,spec in zip(records,c.schedule())]==c.schedule(),'Fixed trajectory set/order')
    c.require({p.name for p in (root/'series').iterdir()}=={x['id'] for x in c.schedule()},'Series inventory')
    rows={};raw={};storage_errors=[];route_errors=[];prefix_errors=[];raw_bytes=0
    for record,spec in zip(records,c.schedule()):
        ident=spec['id'];base=root/'series'/ident
        c.require(c.read(base/'complete.json')==record and record['forward_calls']==9,'Trajectory completion')
        want={f'step-{i:03d}.json' for i in range(9)}|{'complete.json'}|({'logits.npz'} if spec['raw_logits'] else set())
        c.require({p.name for p in base.iterdir()}==want,'Trajectory file inventory')
        if spec['raw_logits']:
            a=artifacts(root,record['logits_file'],record['logits_sha256']);c.require(set(a)=={'logits'},'Logits keys')
            x=a['logits'];c.require(x.dtype==np.float32 and x.shape==(9,166144) and np.isfinite(x).all(),'Logit shape/finite');raw[ident]=x;raw_bytes+=x.nbytes
        rr=[]
        for step in range(9):
            row=c.read(base/f'step-{step:03d}.json');rr.append(row)
            c.require(row['id']==ident and row['arm']==spec['arm'] and row['phase']==spec['phase'] and row['step']==step and row['length']==128+step and row['target_token']==tokens[128+step],'Step/target identity')
            c.require(row['capture_enabled']==(spec['phase']=='diagnostic'),'Capture leaked into timing')
            c.require(row['logit_sha256']==record['logit_hashes'][step] and row['nll']==record['nll'][step],'Logit summary binding')
            if spec['raw_logits']:
                c.require(c.array_hash(raw[ident][step])==row['logit_sha256'],'Logit raw hash')
                c.equal(row['nll'],c.nll(raw[ident][step],tokens[128+step]),ident+'.nll')
            c.require(type(row['nll']) in (int,float) and math.isfinite(row['nll']) and row['nll']>=0,'Finite NLL')
            timing_check(row['model_timing'])
            c.require(record['started_monotonic']<=row['model_timing']['started_monotonic']<=row['model_timing']['finished_monotonic']<=record['finished_monotonic'],'Trajectory timing bounds')
            compressed=step==0 and spec['arm'] in ('PACKED_SDPA','GQA_FUSED')
            if compressed:timing_check(row['compression_timing'])
            else:c.require(row['compression_timing'] is None,'Unexpected compression')
            if not storage_check(row['inventory'],spec['arm'],128+step):storage_errors.append([ident,step])
            if not route_check(row,spec):route_errors.append([ident,step])
        rows[ident]=rr
        if record['prefix_before']!=record['prefix_after']:prefix_errors.append(ident)
        c.require(set(record['prefill_source'])==set(record['prefix_before']),'Prefill source inventory')
        c.require(set(record['prefix_before'])=={f'{i:02d}-{side}' for i in range(44) for side in ['K','V']},'Prefix inventory')
        c.require(record['routes']==rr[-1]['routes'],'Terminal route binding')
        c.require(record['captured']==(c.PLAN['captures'] if spec['phase']=='diagnostic' else []),'Captured-point inventory')
    gate=np.array_equal(raw['gate-official'].view(np.uint32),raw['gate-native'].view(np.uint32)) and records[0]['terminal_hashes']==records[1]['terminal_hashes']
    c.require(gate and c.read(root/'official_gate.json')==dict(bit_equal=True,logits_bit_equal=True,terminal_kv_equal=True),'Official/native gate')
    # Common prefill was never compressed. Check every recorded main prefill against the gate.
    prefill_equal=all(np.array_equal(x[0].view(np.uint32),raw['gate-official'][0].view(np.uint32)) for x in raw.values()) and all(x['prefill_source']==records[0]['prefill_source'] for x in records)
    repeat_stable={a:all(np.array_equal(raw[f'main-r0-'+a.lower()].view(np.uint32),raw[f'main-r{r}-'+a.lower()].view(np.uint32)) for r in [1,2]) for a in c.SERIES}
    warm_ready=c.read(root/'timing_ready.json');after=c.read(root/'timing_after.json')
    c.require(warm_ready['warmup_trajectories']==3 and all(x['finished_monotonic']<=warm_ready['completed_monotonic'] for x in records if x['phase'] in ('gate','diagnostic','warmup')),'Warmup boundary')
    c.require(all(warm_ready['completed_monotonic']<=x['started_monotonic']<=x['finished_monotonic']<=after['completed_monotonic'] for x in records if x['phase']=='main'),'Main timing boundary')
    warmed=warm_ready['cache']==after['cache']
    actual=c.rows(root/'actual_q.jsonl');c.require([{k:x[k] for k in ['slot','step','length']} for x in actual]==c.PLAN['captures'],'Actual-Q set/order')
    diagnostics=[]
    for row in actual:
        z=artifacts(root,row['file'],row['sha256']);raw_bytes+=sum(a.nbytes for a in z.values())
        a={n:z.pop(n) for n in list(z) if n=='q' or n.startswith(('k_','v_'))};model_out=z.pop('model_output')
        cc,tt=nc.structure(a,True);c.require(cc==64 and tt==row['length'],'Actual-Q length')
        c.require(c.arrays_hash(a)==row['input_before']==row['input_after'],'Actual-Q input mutation')
        c.require(model_out.dtype==np.uint16 and model_out.shape==(1,1,48,128),'Model attention output format')
        result=diagnostic(z,a);match=bool(np.array_equal(model_out,z['GQA_FUSED_normal']))
        diagnostics.append(dict(slot=row['slot'],step=row['step'],model_replay_equal=match,**result))
    fixture=c.read(root/'codec_fixtures.json');expected={f'pack_{n}' for n in [1,2,3,7,8,9,63,64,65,8192]}|{'group_'+n for n in ['zero','constant','strided','K_00','V_43']}
    c.require(len(fixture['checks'])==15 and {v['id'] for v in fixture['checks']}==expected and fixture['model_forward_calls']==0 and fixture['device']=='cuda','Codec fixture inventory')
    c.require(all(type(x['pass']) is bool for x in fixture['checks']),'Fixture bool')
    codec_ok=all(x['pass'] for x in fixture['checks']);c.require(fixture['pass']==codec_ok,'Codec fixture summary')
    numerical=all(x['pass_all'] and x['model_replay_equal'] for x in diagnostics)
    packed_prefixes=[x['prefix_before'] for x in records if x['arm'] in ('PACKED_SDPA','GQA_FUSED')]
    packed_initial_equal=all(x==packed_prefixes[0] for x in packed_prefixes)
    passed=codec_ok and prefill_equal and packed_initial_equal and numerical and not(storage_errors or route_errors or prefix_errors)
    stable=all(repeat_stable.values());status='INCONCLUSIVE' if not stable else 'PASS_MODEL_CONNECTION' if passed else 'FAIL_MODEL_CONNECTION'
    c.require(c.evidence_bytes(root)<=c.PLAN['evidence_limit_bytes'],'Evidence cap')
    result=dict(experiment='LOOP-KV-NB05',version=c.PLAN['version'],status=status,
        completed_forward_calls=135,material_count=1,main_repetitions_per_arm=3,official_native_equal=True,
        prefill_equal=prefill_equal,packed_initial_equal=packed_initial_equal,repeat_stable=repeat_stable,numerical_pass=numerical,actual_q=diagnostics,codec_pass=codec_ok,
        route_errors=route_errors,storage_errors=storage_errors,prefix_errors=prefix_errors,raw_array_bytes=raw_bytes,
        timing_prepared=warmed,timing_interpretation='prepared checked-model decode timing' if warmed else 'cache changed: do not claim warmed speed',
        quality=quality(rows),timing=performance(rows),memory_peak_bytes=memory(rows),
        terminal_kv_bytes={a:rows['main-r0-'+a.lower()][-1]['inventory']['tensor_bytes'] for a in c.SERIES},
        scope='One of four selected new passages; 8 post-compression predictions, 4 actual-Q points, 3 technical repetitions. Checked native baseline; setup and evidence IO excluded from timing.')
    return result

def aggregate(books):
    c.require(len(books)==4,'Aggregate requires all four books')
    qualities={}
    for label in ('post_compression_8','all_9'):
        means={arm:float(np.mean([b['quality'][label]['median_mean_nll'][arm] for b in books])) for arm in c.SERIES}
        contrasts={}
        for arm,base in [('PACKED_SDPA','NATIVE'),('GQA_FUSED','NATIVE'),('GQA_FUSED','PACKED_SDPA')]:
            delta=means[arm]-means[base]
            contrasts[arm+'_vs_'+base]=dict(delta_nll=delta,relative_ppl_percent=100*math.expm1(delta) if delta<700 else None)
        qualities[label]=dict(mean_of_book_mean_nll=means,ppl={a:math.exp(v) if v<700 else None for a,v in means.items()},contrasts=contrasts,
            unique_target_positions_per_arm=32 if label=='post_compression_8' else 36)
    timing={}
    for clock in ('cuda_ms','wall_ms'):
        timing[clock]={}
        for metric,key in [('decode','fused_vs'),('phase_sum','phase_sum_fused_vs')]:
            timing[clock][metric]={}
            for base in ('NATIVE','PACKED_SDPA'):
                medians=[b['timing'][clock][key][base]['median_ratio'] for b in books]
                blocks=[dict(selected_book_rank=i,round=r,ratio=value) for i,b in enumerate(books)
                        for r,value in enumerate(b['timing'][clock][key][base]['ratios_by_round'])]
                timing[clock][metric]['GQA_FUSED_vs_'+base]=dict(book_medians=medians,
                    median_of_book_medians=float(np.median(medians)),paired_blocks=blocks)
    quality_ok=all(b['quality']['post_compression_8']['contrasts']['GQA_FUSED_vs_NATIVE']['reference_5pct']=='WITHIN_5PCT_ON_THIS_TEXT' for b in books)
    prepared=all(b['timing_prepared'] for b in books)
    return dict(quality=qualities,timing=timing,timing_prepared=prepared,
        hypotheses=dict(H1_all_book_connections=all(b['status']=='PASS_MODEL_CONNECTION' for b in books),
            H2_all_book_fused_ppl_within_5pct=quality_ok,
            H3_fused_faster_than_full_restore=(timing['cuda_ms']['decode']['GQA_FUSED_vs_PACKED_SDPA']['median_of_book_medians']<1) if prepared else None),
        pooling='Four passages once each. Technical repeats are not additional texts; no old-run pooling, CI or p-value.')

def verify(root,check_summary=True):
    root=Path(root);source_count=c.check_embedded(root);packages=verify_environment(root)
    selected=materials.read_frozen(root);lock=c.read(root/'selection_lock.json');ready=c.read(root/'model_ready.json')
    c.require(c.file_digest(root/'nb05_codec_fixture.json')==c.PLAN['codec_fixture_sha256'],'Frozen codec fixture')
    c.require(ready['revision']==c.MODEL['revision'] and ready['input_lock_sha256']==c.file_digest(root/'selection_lock.json'),'Loaded/input identity')
    c.require(ready['input_frozen_monotonic']==lock['frozen_monotonic']<=ready['weights_started_monotonic']<=ready['loaded_monotonic']
        and ready['loaded_contract_ok'] is True and ready['model_calls_before_measurement']==0 and ready['fusion'] is True
        and ready['weight_quantization'] is False and ready['deterministic_algorithms'] is True and ready['tf32'] is False,'Loaded contract/chronology')
    started=c.read(root/'model_load_started.json')
    c.require(started['input_lock_sha256']==ready['input_lock_sha256'] and ready['weights_started_monotonic']<=started['monotonic']<=ready['loaded_monotonic'],'Load start binding')
    c.require({p.name for p in (root/'books').iterdir()}=={'b0','b1','b2','b3'},'Book directory inventory')
    results=[];schedule=[];previous=ready['loaded_monotonic']
    for rank,doc in enumerate(selected['diagnostic']):
        path=root/'books'/f'b{rank}';binding=c.read(path/'book.json');done=c.read(path/'book_complete.json')
        c.require(binding['selected_book_rank']==done['selected_book_rank']==rank and binding['book_id']==done['book_id']==doc['book_id']
            and binding['token_sha256']==doc['token_sha256'] and binding['selection_lock_sha256']==c.file_digest(root/'selection_lock.json'),'Book identity binding')
        c.require(previous<=binding['started_monotonic']<=done['finished_monotonic']<=binding['deadline_monotonic'] and done['attempted']==done['finished']==135,'Book chronology/count/budget')
        records=c.rows(path/'trajectories.jsonl');prior=binding['started_monotonic']
        for record,spec in zip(records,c.schedule()):
            c.require(prior<=record['started_monotonic']<=record['finished_monotonic']<=done['finished_monotonic'],'Sequential trajectories')
            c.require(record['finished_monotonic']-record['started_monotonic']<=c.CAPS['series'],'Series duration')
            prior=record['finished_monotonic']
            schedule.append(dict(spec,id=f'b{rank}-'+record['id'],selected_book_rank=rank,book_id=doc['book_id']))
        result=verify_book(path,doc['tokens'])
        results.append(dict(selected_book_rank=rank,book_id=doc['book_id'],**result));previous=done['finished_monotonic']
    c.require(schedule==c.read(root/'schedule.json'),'Global schedule reconciliation')
    progress=c.read(root/'progress.json')
    c.require(progress['attempted']==progress['finished']==c.PLAN['total_forward_calls'] and progress['selected_book_rank']==3 and progress['book_id']==selected['diagnostic'][3]['book_id'],'Global progress')
    status='INCONCLUSIVE' if any(b['status']=='INCONCLUSIVE' for b in results) else 'FAIL_MODEL_CONNECTION' if any(b['status']=='FAIL_MODEL_CONNECTION' for b in results) else 'PASS_MODEL_CONNECTION'
    c.require(c.evidence_bytes(root)<=c.PLAN['evidence_limit_bytes'],'Global evidence cap')
    result=dict(experiment='LOOP-KV-NB05',version=c.PLAN['version'],status=status,sources_checked=source_count,packages=packages,
        completed_forward_calls=540,material_count=4,trajectories=60,raw_logit_trajectories=48,actual_q_captures=16,
        raw_logit_float32_bytes=c.PLAN['counts']['raw_logit_float32_bytes'],raw_array_bytes=sum(b['raw_array_bytes'] for b in results),
        books=results,**aggregate(results))
    if check_summary:
        diffs=c.equal(c.read(root/'summary.json'),result);terminal=c.read(root/'run_status.json')
        c.require(terminal['status']==status and terminal['measurement_complete'] is True and terminal['attempted']==terminal['finished']==540,'Worker closure')
        print(json.dumps(dict(verified=True,status=status,derived_float_differences=diffs),ensure_ascii=False,allow_nan=False))
    return result
if __name__=='__main__':verify(sys.argv[1])
