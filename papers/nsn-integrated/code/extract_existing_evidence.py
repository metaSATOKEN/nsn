"""CPU-only extraction of frozen NSN/SADP/looped evidence. Never imports experiment modules.
Usage: python extract_existing_evidence.py --research-root PATH --output-dir PATH
Uses saved PPL / per-window changes / timing records. No weights, network or GPU.
Existing bootstrap_ci is extracted as an isolated AST function (no module execution).
"""
import argparse, ast, collections, hashlib, json, math, platform, statistics as st
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
from source_layout import source_file, original_result_ids, omitted_document
import torch

def main(root, out):
    root=root.resolve(); out=out.resolve()
    protected=[root/p for p in ['src','data','experiments','runs']]
    if out==root or any(out==p or p in out.parents for p in protected):
        raise ValueError('Output must be separate from original experiment trees')
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    sources={}; checks=[]; findings=[]; table={}; children=[]
    def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
    def read(p):
        sources[p]=sha(source_file(root,p));return json.loads(source_file(root,p).read_text())
    def code(p):
        sources[p]=sha(source_file(root,p));return source_file(root,p).read_text()
    def check(label,a,b,tol=1e-9):
        if isinstance(a,(int,float)) and isinstance(b,(int,float)):
            ok=(not math.isfinite(a) and not math.isfinite(b) and str(a)==str(b)) or (math.isfinite(a) and math.isfinite(b) and abs(a-b)<=tol)
        else:ok=a==b
        checks.append(dict(check=label,passed=bool(ok),computed=a,stored=b,tolerance=tol));return ok
    def at(src,path):return dict(source=src,pointer=('/'+ '/'.join(str(x).replace('~','~0').replace('/','~1') for x in path)) if path else '',sha256=sources[src])
    def save(fn,x):
        def clean(v):
            if isinstance(v,float) and not math.isfinite(v):return str(v)
            if isinstance(v,dict):return {k:clean(a) for k,a in v.items()}
            if isinstance(v,list):return [clean(a) for a in v]
            return v
        (out/fn).write_text(json.dumps(clean(x),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    m='modeq/'; s='Substrate-Aware-Diagnostic-and-Prescription/'
    # E1: Table 1 uses earliest per-run C/D numbers, not later rounded reprints.
    p14=m+'results/phase14_asym_wikitext2_ppl.json';d14=read(p14)
    p16=m+'results/phase16_asym_seven_model_sweep.json';d16=read(p16)
    p17=m+'results/phase17_2026_flagships.json';d17=read(p17)
    primary=[]
    for name,x in d14.items():
        if not isinstance(x,dict) or 'baseline_ppl' not in x:continue
        a=x['results'];primary.append(dict(model=x['label'].split(' (')[0],baseline=x['baseline_ppl'],C=a['nsep+pchan INT3 symmetric']['delta'],D=a['nsep+pchan INT3 asymmetric']['delta'],sym4=a['nsep+pchan INT4 symmetric']['delta'],origin=at(p14,[name]),precision='full_stored_float'))
    blocked=[]
    for src,d,key in [(p16,d16,'phase16_runs'),(p17,d17,'phase17_runs')]:
        for name,x in d[key].items():
            if 'baseline_ppl' not in x:blocked.append(dict(model=name,record=x,origin=at(src,[key,name])));continue
            primary.append(dict(model=name,baseline=x['baseline_ppl'],C=x['sym_int3_delta'],D=x['asym_int3_delta'],sym4=x['sym_int4_delta'],origin=at(src,[key,name]),precision='rounded_4dp'))
    for x in primary:
        x.update(C_percent=x['C']/x['baseline']*100,D_percent=x['D']/x['baseline']*100,reduction_percent=(1-x['D']/x['C'])*100,D_under5=x['D']/x['baseline']<.05,windows_cap=10,context=1024,stride=512,requested_dtype='float16',actual_dtype='unknown',evidence_level='saved_aggregate_PPL_not_raw_logits')
    # E1 four-way comparison: Phase18 new rows + Phase19 new rows only.
    p18=m+'results/phase18_ablation_vs_industry_naive.json';d18=read(p18)
    p19=m+'results/phase19_ablation_generalization.json';d19=read(p19)
    ablation=[]
    for src,d,key in [(p18,d18,'runs'),(p19,d19,'phase19_runs')]:
        for name,x in d[key].items():
            for method,v in x['results'].items():
                if not method.startswith(('(','A_','B_','C_','D_')):continue
                letter=method[1] if method.startswith('(') else method[0]
                calc=v['ppl']-x['baseline_ppl'];check(f'{src}/{name}/{letter}/delta',calc,v['delta'],tol=.000151 if src==p19 else 1e-10)
                ablation.append(dict(model=x['label'].split(' (')[0],model_id=x['model'],method=letter,baseline_ppl=x['baseline_ppl'],ppl=v['ppl'],delta_from_ppl=calc,delta_stored=v['delta'],percent=100*calc/x['baseline_ppl'],under5=calc/x['baseline_ppl']<.05,tokens=v.get('tokens','unknown'),context=1024,windows_cap=10,stride=512,evidence_level='saved_aggregate',origin=at(src,[key,name,'results',method])))
    check('E1 ablation cells',len(ablation),32)
    check('E1 primary models',len(primary),8)
    primary_summary=dict(models=8,D_under5=sum(x['D_under5'] for x in primary),C_under5=sum(x['C_percent']<5 for x in primary),mean_C=st.mean(x['C'] for x in primary),mean_D=st.mean(x['D'] for x in primary),mean_reduction_percent=st.mean(x['reduction_percent'] for x in primary),mean_D_percent=st.mean(x['D_percent'] for x in primary),four_way_under5={a:sum(x['under5'] for x in ablation if x['method']==a) for a in 'ABCD'})
    # E2: dtype probe, longer Gemma segments, small-model and bit-width boundaries.
    boundaries=[]
    for fn in ['phase20_bf16_disambiguation','phase21_gemma_b_anomaly_probe','phase28b_recipe_on_small_qwen','phase28d_qwen25_7b_asym_sweep']:
        src=m+'results/'+fn+'.json';d=read(src)
        if fn.startswith('phase20'):
            records=[(name,x,x['results'],['runs',name,'results']) for name,x in d['runs'].items()]
        elif fn.startswith('phase21'):
            records=[(seg,dict(x,model=d['model'],dtype=d['dtype']),x['methods'],['results',seg,'methods']) for seg,x in d['results'].items()]
        elif fn.startswith('phase28b'):
            records=[(name,x,x['per_method'],['results',name,'per_method']) for name,x in d['results'].items()]
        else:records=[(d['label'],d,d['per_method'],['per_method'])]
        for label,x,methods,ptr in records:
            for method,v in methods.items():
                if 'ppl' not in v:continue
                pct=100*(v['ppl']/x['baseline_ppl']-1)
                boundaries.append(dict(run=fn,label=label,model=x.get('model','unknown'),method=method,baseline=x['baseline_ppl'],ppl=v['ppl'],percent=pct,under5=pct<5,tokens=v.get('tokens','unknown'),windows_cap=d.get('windows_per_method',d.get('max_windows',d.get('protocol',{}).get('max_windows','unknown'))),actual_dtype=x.get('actual_param_dtype',x.get('dtype','unknown')),origin=at(src,ptr+[method])))
        if fn.startswith('phase20'):
            for x in primary:
                if x['model']=='Qwen3.5-9B':x['actual_dtype']='bfloat16_reported_by_phase20_not_original_runtime_manifest'
        if fn.startswith('phase21'):
            findings.append(dict(id='Gemma_window_cap',description='30 is max_windows; stored tokens 10240/11264 imply 20/22 scored windows of 512, not 30 completed.',origin=at(src,['results'])))
    # Preserve source interpretations separately (including parity claim), never use as metrics.
    table['legacy_primary']=primary;table['legacy_ablation']=ablation;table['legacy_boundaries']=boundaries;table['legacy_blocked']=blocked
    # Correlation is an appendix candidate: recompute from per-segment CV summaries; do not reconstruct removed raw norms.
    pcv=m+'results/phase23_norm_dispersion.json';dcv=read(pcv);cvrows=[]
    for name,x in dcv['per_model'].items():
        row=dict(model=name,origin=at(pcv,['per_model',name]))
        for side in ['k','v']:
            vals=[a['layer_stats']['L0'][side+'_pooled']['cv'] for a in x['per_segment']];value=st.mean(vals)
            check('CV '+name+'/'+side,value,x['summary']['L0_'+side+'_cv']['mean'])
            row[side+'_cv_mean']=value
        row['B_percent']=next(a['percent'] for a in ablation if a['model'].lower()==name.lower() and a['method']=='B');cvrows.append(row)
    def ranks(v):
        ids=sorted(range(len(v)),key=lambda i:v[i]);out=[0]*len(v)
        for n,i in enumerate(ids):out[i]=n+1
        assert len(set(v))==len(v);return out
    rho={a:st.correlation(ranks([v[a+'_cv_mean'] for v in cvrows]),ranks([v['B_percent'] for v in cvrows])) for a in ['k','v']}
    table['norm_dispersion']=cvrows
    # E4 packing and model PPL.
    p15=m+'results/phase15_real_compression.json';d15=read(p15);packing=[]
    for name,x in d15['results'].items():
        c=x['config'];n=2*c['n_layers']*c['n_kv_heads']*c['ctx']*c['head_dim'];check('FP16 size '+name,n*2,x['fp16_bytes'])
        for i,a in enumerate(x['rows']):
            bits=16 if 'FP16' in a['method'] else (4 if 'INT4' in a['method'] else 3)
            meta=0 if bits==16 else c['ctx']*2+c['head_dim']*(4 if 'asym' in a['method'] else 2)
            size=math.ceil(c['ctx']*c['head_dim']*bits/8)+meta
            total=size*2*c['n_layers']*c['n_kv_heads'];check('storage '+name+'/'+a['method'],total,a['total_bytes'])
            check('effective bits '+name+'/'+a['method'],total*8/n,a['bits_per_value'])
            packing.append(dict(model=name,method=a['method'],total_bytes=total,bits_per_value=total*8/n,baseline_bytes=n*2,compression_factor=n*2/total,storage_reduction_percent=100*(1-total/(n*2)),measurement='synthetic_slice_real_packed_bytes_scaled_to_geometry' if bits==3 else 'formula_only',origin=at(p15,['results',name,'rows',i])))
    p26=m+'results/phase26_real_cache_ppl.json';d26=read(p26);packed_ppl=[]
    for name,x in d26['results'].items():
        a=x['per_method'];sp=a['asym INT3 (simulator)']['ppl'];rp=a['asym INT3 (real cache)']['ppl'];delta=rp-sp
        check('P26 signed difference '+name,delta,a['sim_vs_real_diff']['abs'])
        packed_ppl.append(dict(model=name,baseline=x['baseline_ppl'],simulator_ppl=sp,packed_ppl=rp,packed_minus_simulator=delta,absolute_difference=abs(delta),packed_change_percent=100*(rp/x['baseline_ppl']-1),evidence_level='saved_aggregate',origin=at(p26,['results',name])))
    p27=m+'results/phase27_decode_latency.json';d27=read(p27);latency=[]
    for name,x in d27['results'].items():
        base=st.median(x['raw_runs']['baseline'])
        for method,vals in x['raw_runs'].items():
            med=st.median(vals);check('P27 median '+name+'/'+method,med,x['median_ms_per_step'][method])
            latency.append(dict(model=name,method=method,n_technical_replicates=len(vals),saved_ms_per_token=vals,median_ms_per_token=med,overhead_percent=100*(med/base-1),initial_pack_included=method=='roundtrip',baseline_ms_per_token=base,origin=at(p27,['results',name,'raw_runs',method])))
    table['storage']=packing;table['packed_ppl']=packed_ppl;table['legacy_latency']=latency
    # E3: reproduce exact existing bootstrap specification. Safe AST: only bootstrap_ci.
    def bootstrap_factory(src):
        tree=ast.parse(code(src));node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='bootstrap_ci')
        namespace={'torch':torch,'N_BOOTSTRAP':5000};exec(compile(ast.Module(body=[node],type_ignores=[]),src,'exec'),namespace)
        return namespace['bootstrap_ci']
    sadp_specs=[('e5_holdout_oracle','phase_e5_holdout_oracle.py'),('holdout_causal_faithful','phase_holdout_causal_faithful.py'),('w1d_causal_faithful_longctx','phase_w1d_causal_faithful_longctx.py'),('w1e_class_dependence','phase_w1e_class_dependence.py')]
    quality=[];gaps=[];verdicts=[];overlap=[];omissions=[]
    for run,script in sadp_specs:
        src=s+'results/'+run+'.json';d=read(src);boot=bootstrap_factory(s+'experiments/'+script);seed=d['knobs']['seed'];cells=d.get('models',d.get('cells'))
        for name,x in cells.items():
            ctx=x.get('ctx',d['knobs'].get('ctx',1024));starts=x['window_starts'];cpath=['models' if 'models' in d else 'cells',name]
            overlap.append(dict(run=run,cell=name,windows=len(starts),unique_starts=len(set(starts)),context=ctx,overlapping_pairs=sum(abs(a-b)<ctx for i,a in enumerate(starts) for b in starts[i+1:]),possible_pairs=len(starts)*(len(starts)-1)//2,origin=at(src,cpath+['window_starts'])))
            if run=='e5_holdout_oracle':
                recipes=[(key,'A0_full_window',q,cpath+['cells',key]) for key,q in x['cells'].items()]
            elif run=='w1e_class_dependence':recipes=[(key,'A0_full_window',q,cpath+['recipes',key]) for key,q in x['recipes'].items()]
            else:recipes=[(key,conf,q,cpath+['recipes',key,conf]) for key,configs in x['recipes'].items() for conf,q in configs.items()]
            values={}
            for recipe,conf,q,path in recipes:
                if 'per_window_dppl' not in q:
                    omissions.append(dict(run=run,cell=name,recipe=recipe,config=conf,record=q,origin=at(src,path)));continue
                vals=q['per_window_dppl'];values[(recipe,conf)]=vals;finite=sum(math.isfinite(v) for v in vals)
                check('SADP window count '+run+name+recipe+conf,len(vals),len(starts))
                stats=boot(vals,seed+1)
                if finite==len(vals):
                    check('independent mean '+run+name+recipe+conf,st.mean(vals),stats['mean'])
                    check('independent std '+run+name+recipe+conf,st.stdev(vals),stats['std'])
                for k in ['mean','std','ci_lo','ci_hi','ci_half_width']:check('SADP '+run+'/'+name+'/'+recipe+'/'+conf+'/'+k,stats[k],q[k])
                for i,v in enumerate(vals):children.append(dict(run=run,cell=name,recipe=recipe,config=conf,window_index=i,token_start=starts[i],context=ctx,relative_ppl_percent=v,origin=at(src,path+['per_window_dppl',i])))
                row=dict(run=run,cell=name,recipe=recipe,config=conf,context=ctx,windows=len(vals),finite_windows=finite,invalid_windows=len(vals)-finite,status='FINITE' if finite==len(vals) else 'NONFINITE_PRESERVED',stats=stats,baseline_mean_ppl=x['fp16_ref_ppl_mean'],k_bits=q.get('k_bits',d['knobs'].get('k_bits')),v_bits=q.get('k_bits',d['knobs'].get('v_bits',d['knobs'].get('k_bits'))),origin=at(src,path),metric='mean_of_window_relative_PPL_percent_not_pooled_token_PPL',evidence_level='saved_per_window_percent_change')
                if math.isfinite(stats['mean']):
                    row.update(point_within_plusminus_0_5=abs(stats['mean'])<.5,ci_within_plusminus_0_5=stats['ci_lo']>-.5 and stats['ci_hi']<.5,ci_upper_below_plus_0_5=stats['ci_hi']<.5)
                for flag in ['in_band','out_of_band']:
                    if flag in q:
                        row['stored_'+flag]=q[flag];flagcalc=stats['ci_hi']<.5 if flag=='in_band' else stats['ci_lo']>.5;check('stored one-sided '+run+name+recipe+conf+flag,flagcalc,q[flag])
                quality.append(row)
            pairs=[]
            if run=='e5_holdout_oracle':
                for key,g in x['paired'].items():
                    a,b=key.split('_vs_');pairs.append((key,values[(a,'A0_full_window')],values[(b,'A0_full_window')],g,cpath+['paired',key],seed+2))
            else:
                gs=x.get('cross_recipe_gaps',{'A0_full_window':x.get('cross_recipe_gap')})
                for conf,g in gs.items():
                    pairs.append((conf,values[('kivi',conf)],values[('nsn',conf)],g,cpath+(['cross_recipe_gaps',conf] if 'cross_recipe_gaps' in x else ['cross_recipe_gap']),seed+7))
            for key,a,b,g,path,bseed in pairs:
                vals=[u-v for u,v in zip(a,b)];stats=boot(vals,bseed)
                for k in ['mean','std','ci_lo','ci_hi','ci_half_width']:check('gap '+run+name+key+k,stats[k],g[k])
                gaps.append(dict(run=run,cell=name,comparison=key,direction='three_tier_mask_minus_comparator' if run=='e5_holdout_oracle' else 'plain_family_minus_NSN_family',n_pairs=len(vals),finite_pairs=sum(math.isfinite(v) for v in vals),stats=stats,origin=at(src,path)))
            if 'verdict' in x:verdicts.append(dict(run=run,cell=name,stored=x['verdict'],origin=at(src,cpath+['verdict'])))
    table.update(sadp_quality=quality,sadp_paired=gaps,sadp_verdicts=verdicts,sadp_skipped_configs=omissions,window_overlap=overlap)
    # E5: do not mistake nominal bytes for actual uint8 code storage.
    kernel=[];kernel_summary=[]
    for run in ['k1_production_kernel','k1b_compile_corroboration','k1c_triton_fused']:
        src=s+'results/'+run+'.json';d=read(src)
        for i,c in enumerate(d['cells']):kernel.append(dict(run=run,record=c,status='FAILED' if 'error' in c else 'RECORDED_MEDIAN',origin=at(src,['cells',i])))
        for model in sorted({c['model'] for c in d['cells']}):
            cs=[c for c in d['cells'] if c['model']==model];lookup={(c['batch'],c['ctx'],c['recipe'],c['bits']):c for c in cs};ratios=[]
            for c in cs:
                if c['recipe']!='nsn' or c['bits']!=5 or 'error' in c:continue
                b=lookup[(c['batch'],c['ctx'],'fp16',16)]
                if 'error' not in b:ratios.append(b['latency_ms_median']/c['latency_ms_median'])
            kernel_summary.append(dict(run=run,model_geometry=model,planned_cells=len(cs),measured_cells=sum('latency_ms_median' in c for c in cs),error_cells=sum('error' in c for c in cs),NSN5_vs_FP16_pairs=len(ratios),FP16_over_NSN5_median=st.median(ratios),NSN5_shorter=sum(v>1 for v in ratios),ratios=ratios,endpoint='QK_scores_only',actual_codes_dtype='uint8' if run=='k1c_triton_fused' else 'float16',input='synthetic_K'))
    table['kernel_cells']=kernel;table['kernel_summary']=kernel_summary
    # E6: use successful current extraction contracts; recompute from lower-level saved measurements.
    pp='papers/loop-kv/paper-data-v0.2-2026-10-03.json';paper=read(pp)
    for path,h in paper['source_sha256'].items():check('loop paper source '+path,sha(source_file(root,path)),h)
    acsrc='reports/ac08-descriptive-statistics.json';ac=read(acsrc)
    nbsrc='throughline/audits/loop-kv-nb05-results-v0.1-2026-10-03/supplementary-analysis.json';nb=read(nbsrc)
    vfsrc='throughline/audits/loop-kv-vf01-results-v0.1-2026-10-02/analysis.json';vf=read(vfsrc)
    loop_quality=[];loop_timing=[]
    for model,ref,fused in [('Ouro','NATIVE','PACKED_KV_FUSED'),('Nanbeige','NATIVE','GQA_FUSED')]:
        for q in [a for a in paper['quality'] if a['model']==model]:
            pct=100*(q['ppl'][fused]/q['ppl'][ref]-1);check('loop book quality '+q['book_id'],pct,q['fused_change_percent'])
            loop_quality.append(dict(model=model,book_id=q['book_id'],ppl=q['ppl'],fused_change_percent=pct,n_predictions=255 if model=='Ouro' else 8,technical_replicates=2 if model=='Ouro' else 3,evidence_level='saved_PPL_inherited_raw_logit_validation',origin=at(pp,['quality',paper['quality'].index(q)])))
    acmeans=[]
    for book in ac['quality']:
        rows=[v for v in ac['comparisons'] if v['book_id']==book['book_id']];assert len(rows)==2
        for arm in ['NATIVE','PACKED_GROUPED','PACKED_KV_FUSED']:check('AC08 repeat NLL '+book['book_id']+arm,rows[0]['nll'][arm],rows[1]['nll'][arm])
        acmeans.append(rows[0]['nll']['PACKED_KV_FUSED']-rows[0]['nll']['NATIVE'])
    acpct=100*math.expm1(st.mean(acmeans));check('AC08 aggregate',acpct,paper['ouro']['aggregate_ppl_change_percent']['ALL'])
    for candidate,ref in [('PACKED_KV_FUSED','NATIVE'),('PACKED_KV_FUSED','PACKED_GROUPED')]:
        ratios=[v['timing'][candidate]['cuda_ms_per_token']/v['timing'][ref]['cuda_ms_per_token'] for v in ac['comparisons']];value=st.median(ratios);check('AC08 timing '+ref,value,paper['ouro']['ratios'][candidate+'_over_'+ref]['time_ratio']['median'])
        loop_timing.append(dict(model='Ouro',scope='decode_240_positions',candidate=candidate,reference=ref,ratio=value,change_percent=100*(value-1),paired_blocks=8,texts=4,origin=at(acsrc,['comparisons'])))
    # NB05 raw step records: 4 books x 3 rounds x 3 arms x 9 forward records.
    nbroot='runs/loop-kv-nb05-20261003-090604-18baabf5';nbraw=[];nbnll=[]
    for rank,book in enumerate(nb['books']):
        rounds=[]
        for rnd in range(3):
            arms={}
            for arm in ['NATIVE','PACKED_SDPA','GQA_FUSED']:
                rel=f'{nbroot}/books/b{rank}/series/main-r{rnd}-{arm.lower()}'
                rows=[read(f'{rel}/step-{i:03d}.json') for i in range(9)]
                record=dict(prefill=rows[0]['model_timing']['cuda_ms'],compression=(rows[0]['compression_timing'] or {}).get('cuda_ms',0),decode=sum(x['model_timing']['cuda_ms'] for x in rows[1:]),mean_nll=st.mean(x['nll'] for x in rows[1:]),terminal_kv=rows[-1]['inventory']['unique_storage_bytes'])
                record['total']=record['prefill']+record['compression']+record['decode'];arms[arm]=record
                nbraw.append(dict(book_id=book['book_id'],round=rnd,arm=arm,**record,source_directory=rel))
                check('NB05 NLL '+rel,record['mean_nll'],book['quality']['post_compression_8']['mean_nll_by_round'][arm][rnd])
            rounds.append(arms)
        for ref in ['NATIVE','PACKED_SDPA']:
            for field,key in [('decode','decode'),('total','phase_sum')]:
                ratio=st.median(z['GQA_FUSED'][field]/z[ref][field] for z in rounds)
                stored=nb['timing']['cuda_ms'][key]['GQA_FUSED_vs_'+ref]['book_medians'][rank]
                check('NB05 book timing '+book['book_id']+ref+field,ratio,stored)
        nbnll.append(rounds[0]['GQA_FUSED']['mean_nll']-rounds[0]['NATIVE']['mean_nll'])
    nbpct=100*math.expm1(st.mean(nbnll));check('NB05 aggregate quality',nbpct,paper['nanbeige']['quality']['post_compression_8']['contrasts']['GQA_FUSED_vs_NATIVE']['relative_ppl_percent'])
    for ref in ['NATIVE','PACKED_SDPA']:
        for field,key in [('decode','decode'),('total','phase_sum')]:
            bookratios=[]
            for b in nb['books']:
                vals=[v for v in nbraw if v['book_id']==b['book_id']];rat=[]
                for rnd in range(3):
                    a={v['arm']:v for v in vals if v['round']==rnd};rat.append(a['GQA_FUSED'][field]/a[ref][field])
                bookratios.append(st.median(rat))
            ratio=st.median(bookratios);check('NB05 pooled '+ref+field,ratio,paper['nanbeige']['timing']['cuda_ms'][key]['GQA_FUSED_vs_'+ref]['ratio'])
            loop_timing.append(dict(model='Nanbeige',scope=key,candidate='GQA_FUSED',reference=ref,ratio=ratio,change_percent=100*(ratio-1),paired_blocks=12,texts=4,origin=at(nbsrc,['timing','cuda_ms',key])))
    bridge=[]
    vftime='runs/loop-kv-vf01-20261002-072452-1d1c10c0/timings.jsonl'
    rawvf=[json.loads(line) for line in code(vftime).splitlines() if line.strip()]
    check('VF01 timing rows',len(rawvf),540)
    for ref in ['RF01_MATERIALIZE','TILE_MATERIALIZE']:
        vals=[]
        for a in vf['per_sample']:
            if a['phase']!='main':continue
            qr=[]
            for q in ['gaussian','one_hot','alternating']:
                meds={}
                for arm in ['V_FUSED',ref]:
                    runs=[x['cuda_ms'] for x in rawvf if x['sample']==a['sample'] and x['query']==q and x['arm']==arm]
                    check('VF01 five replicates '+a['sample']+q+arm,len(runs),5);meds[arm]=st.median(runs)
                qr.append(meds['V_FUSED']/meds[ref])
            vals.append(st.median(qr))
        ratio=st.median(vals)
        check('VF01 ratio '+ref,ratio,paper['bridge']['ratios']['V_FUSED/'+ref]['median'])
        bridge.append(dict(candidate='V_FUSED',reference=ref,ratio=ratio,change_percent=100*(ratio-1),pairs=len(vals),texts=2,origin=at(vftime,[]),aggregation='median_5_runs_then_paired_ratio_by_query_then_median_3_nonzero_queries_then_median_6_samples'))
    block=64*128*3//8+(64+128+128)*2+12;check('block bytes',block,paper['block_storage_bytes'])
    expected={'Ouro':192*16*2*((896//64)*block+127*128*2),'Nanbeige':44*8*2*(block+72*128*2)}
    memory=[]
    for model,b,comp in [('Ouro',paper['ouro']['memory']['NATIVE']['terminal_storage'],paper['ouro']['memory']['PACKED_KV_FUSED']['terminal_storage']),('Nanbeige',nbraw[0]['terminal_kv'],next(a['terminal_kv'] for a in nbraw if a['arm']=='GQA_FUSED'))]:
        check('loop terminal storage '+model,expected[model],comp);memory.append(dict(model=model,baseline_bytes=b,packed_bytes=comp,reduction_percent=100*(1-comp/b),measurement='recorded_actual_terminal_KV_storage',origin=at(pp,['ouro' if model=='Ouro' else 'nanbeige'])))
    table.update(loop_quality=loop_quality,loop_timing=loop_timing,loop_memory=memory,loop_nb05_raw_summaries=nbraw,bridge=bridge)
    # Preserve previous numerical audit and verify all declared source hashes, without repeating GPU work.
    pa='papers/loop-kv/numerical-audit-v0.4-2026-10-03.json';audit=read(pa)
    for key in ['source_sha256','contract_source_sha256','raw_logit_sha256']:
        for path,h in audit[key].items():check('numerical audit source '+path,sha(source_file(root,path)),h)
    # Documentation/code references (read and hash; never run historical notebook top-level).
    source_codes=['modeq/src/nsn/quantize.py','modeq/src/nsn/cache.py','modeq/src/nsn/pack.py','modeq/experiments/phase18_ablation_vs_industry_naive.py','modeq/experiments/phase19_ablation_generalization.py','modeq/experiments/phase21_gemma_b_anomaly_probe.py','modeq/experiments/phase15_real_compression_bench.py','modeq/experiments/phase26_real_cache_ppl.py','modeq/experiments/phase27_decode_latency.py','modeq/paper/nsn.md',s+'Diagnostic-and-Transfer-Study.md',s+'experiments/phase_k1_production_kernel_benchmark.py',s+'experiments/phase_k1b_compile_corroboration.py',s+'experiments/phase_k1c_triton_fused_kernel.py',s+'experiments/holdout_causal_faithful_preregistration.md',s+'experiments/W1d_causal_faithful_preregistration.md',s+'experiments/W1e_class_dependence_preregistration.md','papers/loop-kv/code/build_paper_assets_v0_2.py']
    for p in source_codes:
        if omitted_document(root,p):continue  # Manuscript hashes are not numerical checks.
        if source_file(root,p).exists():code(p)
        else:findings.append(dict(id='optional_code_not_found',path=p))
    protocols={
        'schema_version':1,
        'missing_value_policy':{'unknown':'Not established from selected records; never inferred from a model label.','not_applicable':'Metric or condition not used by this protocol.'},
        'series':{
            'legacy_quality':{'tables':['legacy_primary','legacy_ablation','legacy_boundaries'],'dataset':'WikiText-2 test','metric':'relative change of saved aggregate PPL','requested_dtype':'float16 except explicit BF16 diagnostics','actual_dtype':'unknown except recorded diagnostics','model_revision':'unknown in selected primary aggregates','tokenizer_revision':'unknown','gpu':'unknown per primary run; manuscript environment is separate evidence','raw_logits':'not used in this extraction','context':1024,'prefix_compressed_tokens':512,'scored_tokens_per_window':512,'stride':512,'initial_first_target_logit':'computed before compression','packing':'QDQ simulator except Phase26 real-cache comparison','runtime_versions':'unknown per legacy primary run'},
            'sadp_quality':{'tables':['sadp_quality','sadp_paired','sadp_verdicts','sadp_skipped_configs','window_overlap'],'dataset':'WikiText-2 test','metric':'mean of per-window relative PPL changes (%)','requested_dtype':'float16','actual_dtype':'unknown per tensor; code-level precision documented separately','model_revision':'unknown in selected result metadata','tokenizer_revision':'unknown','gpu':'unknown in selected result metadata','raw_logits':'not used; saved per-window changes','windows':30,'bootstrap_replicates':5000,'window_independence':'overlapping windows present; original bootstrap retained without new dependence correction','other_components':'Uniform NSN/plain comparison leaves E/M/S unquantized; three-tier-mask rows are separate auxiliary comparisons','plain_comparator':'local plain-asymmetric K/V recipe, not complete official KIVI implementation'},
            'legacy_storage_latency':{'tables':['storage','packed_ppl','legacy_latency'],'gpu_latency':'A100 recorded in Phase27 protocol','latency_replicates':3,'decode_steps_timed':50,'warmup_steps':{'ordinary':5,'compile':10},'initial_pack':'outside timed loop for unpack-only paths; roundtrip repacks in timed loop','storage_measurement':'synthetic 2048x128 slice expanded by model geometry; INT4 formula only','model_revision':'unknown','runtime_versions':'unknown'},
            'sadp_kernel':{'tables':['kernel_cells','kernel_summary'],'input':'synthetic tensors; model names label geometry','endpoint':'QK scores','warmup_steps':30,'timing_iterations':100,'raw_timing_iterations':'not available in selected saved JSON','code_storage':{'K1':'float16','K1b':'float16','K1c':'uint8, one byte per code'},'actual_INT4_or_INT5_bitpacking':False,'gpu':'NVIDIA RTX PRO 6000 Blackwell Server Edition; recorded in all three source headers','ppl':'not_applicable'},
            'looped':{'tables':['loop_quality','loop_timing','loop_memory','loop_nb05_raw_summaries','bridge'],'dataset':'PG19 training split','model_and_tokenizer_revisions':'frozen in papers/loop-kv/evidence-index-v0.4-2026-10-03.md and referenced run inputs','gpu_and_versions':'recorded per run; see existing S1','dtype':'BF16 model, FP16 stored auxiliaries; intermediate arithmetic per frozen contract','sdpa_actual_backend':'unknown; automatic selection, not individually identified','independent_texts_claim':'4 distinct texts per model; no technical replicate inflation','metric':'equal-text mean NLL difference exponentiated','raw_logit_check':'previous within-project verification inherited; source hashes rechecked','new_lower_level_reaggregation':'NB05 324 main step JSON, VF01 540 timing JSONL rows; AC08 existing comparison summaries'}
        },
        'saved_run_metadata':{}
    }
    for run,_ in sadp_specs:
        src=s+'results/'+run+'.json';d=read(src)
        protocols['saved_run_metadata'][run]=dict(metadata={k:v for k,v in d.items() if k in ['phase','knobs','transformers_version','generated_at','generator','frozen_compiler_sha','preregistration_lock2','preregistration','derived_fields_note','started','finished']},origin=at(src,[]))
    for run in ['k1_production_kernel','k1b_compile_corroboration','k1c_triton_fused']:
        src=s+'results/'+run+'.json';d=read(src)
        protocols['saved_run_metadata'][run]=dict(metadata=d['meta'],origin=at(src,['meta']))
    for p,h in sources.items():check('source unchanged '+p,sha(source_file(root,p)),h)
    catalog=[]
    for folder in [m+'results',s+'results']:
        for rel in original_result_ids(root,folder):
            p=source_file(root,rel);catalog.append(dict(path=rel,sha256=sha(p),bytes=p.stat().st_size,coverage='selected_extracted' if rel in sources else 'inventory_only_not_recomputed'))
    summary=dict(scope='Existing evidence only; no new model/GPU/network calls. New descriptive synthesis, not preregistered evidence.',primary=primary_summary,norm_spearman=rho,packed_roundtrip_sanity=d26['sanity_simulator_vs_real_on_synthetic'],loop_aggregate_percent={'Ouro':acpct,'Nanbeige':nbpct},selected_result_files=sum(x['coverage']=='selected_extracted' for x in catalog),catalogued_result_files=len(catalog),tables={k:len(v) for k,v in table.items()},saved_window_values=len(children),checks=len(checks),failed_checks=sum(not c['passed'] for c in checks),findings=findings,environment={'python':platform.python_version(),'torch':torch.__version__,'cpu_only':True},known_scope_limits=['old NSN: saved aggregate PPL only','SADP: per-window relative changes, not raw token logits','looped: earlier raw-logit validation inherited; NB05 step NLL/timings recalculated; AC08 comparison summaries recalculated','bootstrap reproduces original window-unit intervals; overlapping windows are not independent texts; no new cluster correction','correlation exploratory; no causal inference','E7 downstream tasks remain appendix candidates, not audited here'])
    save('evidence-tables.json',table);save('window-values.json',children);save('summary.json',summary);save('checks.json',checks);save('source-manifest.json',sources);save('result-catalog.json',catalog);save('protocol-specifications.json',protocols)
    print(json.dumps({k:summary[k] for k in ['primary','norm_spearman','selected_result_files','catalogued_result_files','tables','checks','failed_checks','findings']},ensure_ascii=False,indent=2))
    if summary['failed_checks']:print('CHECK FAILURES:',json.dumps([c for c in checks if not c['passed']],ensure_ascii=False)[:5000])
    if summary['failed_checks']:raise AssertionError('Reaggregation mismatches; inspect checks.json')
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--research-root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();main(a.research_root,a.output_dir)
