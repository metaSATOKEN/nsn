from pathlib import Path
import json, hashlib, math
import numpy as np
import argparse
parser=argparse.ArgumentParser(description="Read saved diagnostics and CPU-recalculate saved NB05 NLL; no model or GPU calls.")
parser.add_argument('--research-root',type=Path,required=True)
parser.add_argument('--output-dir',type=Path,required=True)
args=parser.parse_args()
R=args.research_root.resolve();P=args.output_dir.resolve();A=P
if P.is_relative_to(R/'runs'):
    parser.error('Output must not be inside the evidence runs directory')
for name in ('numerical-audit-v0.4-2026-10-03.json','nb05-nll-checks.json'):
    if (P/name).exists():parser.error('Output exists; choose a fresh directory: '+str(P/name))
P.mkdir(parents=True,exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
paths={n:R/'runs'/r for n,r in [('ac08','loop-kv-ac08-20261002-095022-41c2554c'),('nb05','loop-kv-nb05-20261003-090604-18baabf5'),('vf01','loop-kv-vf01-20261002-072452-1d1c10c0')]}
docs={n:json.loads((p/'summary.json').read_text()) for n,p in paths.items()}
sources={str(p/'summary.json'):sha(p/'summary.json') for p in paths.values()}
ac=docs['ac08'];nb=docs['nb05'];vf=docs['vf01']
def walk(x,keys):
    for k in keys:x=x[k]
    return x
def maximum(rows,*keys):return max(walk(x,keys) for x in rows)
am=[r['numerics']['metrics']['V_FUSED'] for r in ac['local_diagnostics']]
nm=[r for b in nb['books'] for r in b['actual_q']]
acm={k:{'max_abs':maximum(am,k,'max_abs'),'pass_count':sum(x[k]['pass_budget'] for x in am)} for k in ['v_restore','score','context','combined_context']}
acm['bridge_equal_count']=sum(all(x['numerics']['route'][k] for k in ['bridge_v_equal','bridge_context_equal','bridge_output_equal']) for x in ac['local_diagnostics'])
acm['rounding_equal_count']=sum(x['rounding'] for x in am)
acm['final_vs_reference']={'max_abs':max(x['numerics']['metrics']['final_vs_reference']['V_FUSED']['max_abs'] for x in ac['local_diagnostics']),'role':'descriptive, not an additional gate'}
nbm={k:{'max_abs':maximum(nm,k,'max_abs'),'pass_count':sum(x[k]['pass_budget'] for x in nm)} for k in ['score','context']}
nbm['restoration']={k:{'max_abs':maximum(nm,'restoration',k,'max_abs'),'pass_count':sum(x['restoration'][k]['pass_budget'] for x in nm)} for k in ['k','v']}
nbm['bridge_equal_count']=sum(all(x['tile_fused_restore_equal'].values()) for x in nm)
nbm['rounding_equal_count']=sum(x['output_rounding_equal'] for x in nm)
nbm['sdpa']={k:{'max_abs':maximum(nm,'sdpa',k,'max_abs'),'pass_count':sum(x['sdpa'][k]['pass_budget'] for x in nm)} for k in ['TORCH_SDPA','TILE_SDPA']}
nbm['fused_vs_sdpa']={k:maximum(nm,'fused_vs_sdpa',k,'max_abs') for k in ['TORCH_SDPA','TILE_SDPA']}
# Recalculate only saved NB05 token NLLs on CPU; no run code is imported or executed.
nll_rows=[];raw_hashes={}
for book in sorted((paths['nb05']/'books').iterdir()):
    if not book.is_dir():continue
    records=[json.loads(s) for s in (book/'trajectories.jsonl').read_text().splitlines()]
    for record in records:
        if 'logits_file' not in record:continue
        path=book/record['logits_file'];h=sha(path);assert h==record['logits_sha256'];raw_hashes[str(path.relative_to(R))]=h
        with np.load(path,allow_pickle=False) as z:x=z['logits']
        assert x.shape==(9,166144) and x.dtype==np.float32
        for step,row in enumerate(x):
            meta=json.loads((book/'series'/record['id']/f'step-{step:03d}.json').read_text())
            y=row.astype(np.float64);m=float(y.max());v=m+math.log(float(np.exp(y-m).sum()))-float(y[meta['target_token']]);error=abs(v-meta['nll'])
            limit=max(1e-12,1e-12*max(abs(v),abs(meta['nll'])))
            assert error<=limit
            nll_rows.append(dict(book=book.name,series=record['id'],step=step,error=error,limit=limit))
plan=json.loads((R/'scripts/loop_kv_ac08_inputs.json').read_text())
early=set(range(769,777));capture=set(plan['capture_lengths']);timed=set(plan['timed_lengths']);all_lengths=set(range(769,1024))
assert timed==all_lengths-(early|capture) and len(timed)==240
result={'scope':'Extracted saved diagnostic summaries and frozen contracts; NB05 NLL recalculated from saved logits on CPU. No GPU rerun.',
 'ac08':dict(points=len(am),metrics=acm,nll={'count':ac['n_scored'],'max_abs':ac['numerical_audit']['max_nll_error'],'absolute_limit':2e-5,'source':'prior independent verification summary'}),
 'nb05':dict(points=len(nm),metrics=nbm,nll={'raw_trajectories':len(raw_hashes),'count':len(nll_rows),'max_abs':max(r['error'] for r in nll_rows),'relative_tolerance':1e-12,'absolute_tolerance':1e-12,'source':'CPU recalculation in this revision'}),
 'vf01':{'normal_fixtures':len(vf['fixtures']),'saved_input_query_cases':len(vf['cases']),'bridge_exact_cases':sum(all(r['route'][k] for k in ['bridge_v_equal','bridge_context_equal','bridge_output_equal']) for r in vf['fixtures']+vf['cases'])},
 'timing_ac08':dict(all_count=255,initial_exclusions=sorted(early),capture_exclusions=sorted(capture),overlap=sorted(early&capture),timed_count=len(timed),flush=plan['flush_post_lengths']),
 'source_sha256':{str(Path(p).relative_to(R)):h for p,h in sources.items()},'raw_logit_sha256':raw_hashes,'new_gpu_calls':0}
contract_sources=['scripts/loop_kv_ac08_audit.py', 'scripts/loop_kv_ac08_attention.py', 'scripts/loop_kv_ac08_contract.py', 'scripts/loop_kv_ac08_inputs.json', 'scripts/loop_kv_ac08.py', 'scripts/loop_kv_ac08_materials.py', 'scripts/loop_kv_nb05_attention.py', 'scripts/loop_kv_nb05_kernel.py', 'scripts/loop_kv_nb05_numeric.py', 'scripts/loop_kv_nb05_diagnostic.py', 'scripts/loop_kv_nb05_contract.py', 'scripts/loop_kv_nb05_inputs.json', 'scripts/loop_kv_nb05_measure.py', 'scripts/loop_kv_nb05_materials.py', 'scripts/loop_kv_nb05_packed.py', 'scripts/loop_kv_vf01_contract.py', 'scripts/loop_kv_af01_contract.py', 'scripts/loop_kv_qk01_contract.py', 'scripts/loop_kv_vf01_kernel.py', 'scripts/loop_kv_rf01_kernel.py', 'scripts/verify_loop_kv_ac08.py', 'scripts/verify_loop_kv_nb05.py', 'scripts/verify_loop_kv_vf01.py']
result["contract_source_sha256"]={rel:sha(R/rel) for rel in contract_sources}
(P/'numerical-audit-v0.4-2026-10-03.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
(A/'nb05-nll-checks.json').write_text(json.dumps(nll_rows,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':'PASS','saved_query_counts':{'ac08':len(am),'nb05':len(nm)},'nb05_nll_predictions':len(nll_rows),'new_gpu_calls':0,'output_dir':str(P)},ensure_ascii=False,indent=2))
