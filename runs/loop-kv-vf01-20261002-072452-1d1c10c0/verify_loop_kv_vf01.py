"""Independent saved-evidence NumPy audit, adapted from AF01. No torch imports."""
import math,sys
from pathlib import Path
import loop_kv_vf01_contract as c
np=c.np;qc=c.qc;fc=c.fc

def file(root,row):
    p=Path(row['file']);c.require(not p.is_absolute() and '..' not in p.parts,'Unsafe evidence path')
    p=root/p;c.require(c.file_sha(p)==row['sha256'],'Evidence hash: '+str(p));return p

def typed(a,shape,dtype,bits=False):
    c.require(a.shape==shape and str(a.dtype)==dtype,'Evidence shape/dtype')
    c.require(np.isfinite(qc.from_bits(a) if bits else a).all(),'Nonfinite evidence');return a

def plans(got,want,label):
    c.require(len(got)==len(want),'Missing/extra '+label)
    for row,plan in zip(got,want):c.require(all(k in row and type(row[k])==type(v) and row[k]==v for k,v in plan.items()),'Order/ID '+label)

def time_row(row,n):
    c.require(row['calls']==n,'Timing calls')
    for key in ('cuda_ms','wall_ms','started_monotonic','finished_monotonic'):
        c.require(isinstance(row[key],(int,float)) and math.isfinite(row[key]) and row[key]>0,'Invalid timing')
    c.require(row['finished_monotonic']>=row['started_monotonic'],'Timing order')
    for key in ('allocated_before','allocated_peak','reserved_before','reserved_peak','output_storage_bytes'):c.require(type(row[key])==int and row[key]>=0,'Invalid memory count')
    c.require(row['allocated_peak']>=row['allocated_before'] and row['reserved_peak']>=row['reserved_before'],'Peak below baseline')
    c.require(row['output_storage_bytes']==4096,'Unexpected output storage')

def eq(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def numerics(z,q,old_k=None,old_v=None):
    h=q.shape[1];refk=z['reference_k'];refv=z['reference_v'];total=refv.shape[2]
    typed(refk,(1,h,total,128),'uint16',True);typed(refv,refk.shape,'uint16',True)
    gold,A=qc.gold(refk,q);result={};complete=True
    for arm in c.ARMS:
        keys=[arm+'_'+x for x in ['score','context','output','normal','v']]
        if not all(n in z for n in keys):complete=False;result[arm]=dict(complete=False);continue
        score=typed(z[arm+'_score'],(1,h,total),'float32')
        ctx=typed(z[arm+'_context'],(1,h,1,128),'float32')
        out=typed(z[arm+'_output'],(1,1,h,128),'uint16',True)
        normal=typed(z[arm+'_normal'],out.shape,'uint16',True)
        v=typed(z[arm+'_v'],refv.shape,'uint16',True)
        rounding=eq(out,qc.bf16_bits(ctx).transpose(0,2,1,3).copy())
        pv=fc.context_metrics(score,v,ctx);vr=qc.k_metrics(v,refv);sm=qc.score_metrics(score,gold,A)
        combined=c.context_budget(score,v,refv,ctx)
        attention_gold=(fc.oracle(gold)[...,None]*qc.from_bits(refv).astype(np.float64)).sum(-2,keepdims=True).transpose(0,2,1,3)
        result[arm]=dict(complete=True,rounding=rounding,trace_normal_equal=eq(out,normal),v_restore=vr,score=sm,context=pv,combined_context=combined,fp64_full_attention=fc.difference(qc.from_bits(out).astype(np.float64),attention_gold))
    route=dict(complete=complete)
    if complete:
        route.update(scores_equal=all(eq(z[a+'_score'],z[c.ARMS[0]+'_score']) for a in c.ARMS),bridge_v_equal=eq(z['TILE_MATERIALIZE_v'],z['V_FUSED_v']),bridge_context_equal=eq(z['TILE_MATERIALIZE_context'],z['V_FUSED_context']),bridge_output_equal=eq(z['TILE_MATERIALIZE_output'],z['V_FUSED_output']),trace_normal_equal=all(result[a]['trace_normal_equal'] for a in c.ARMS),rounding=all(result[a]['rounding'] for a in c.ARMS))
        route['direct_baseline_equal']=all(n in z for n in ['direct_v','direct_context','direct_output','direct_score']) and all(eq(z['direct_'+s],z['RF01_MATERIALIZE_'+s]) for s in ['v','context','output','score']) and eq(z['direct_v'],refv)
        result['final_vs_reference']={a:qc.k_metrics(z[a+'_output'].transpose(0,2,1,3),z['RF01_MATERIALIZE_output'].transpose(0,2,1,3)) for a in c.ARMS}
    old={}
    if old_k is not None:old['k']=qc.k_metrics(refk,old_k)
    if old_v is not None:old['v']=qc.k_metrics(refv,old_v)
    if old_k is not None and old_v is not None:
        oldscore,_=qc.gold(old_k,q)
        oldgold=(fc.oracle(oldscore)[...,None]*qc.from_bits(old_v).astype(np.float64)).sum(-2,keepdims=True).transpose(0,2,1,3)
        old['full_attention']={a:fc.difference(qc.from_bits(z[a+'_output']).astype(np.float64),oldgold) for a in c.ARMS if a+'_output' in z}
    return dict(complete=complete,route=route,metrics=result,old_reference_difference=old)

def audit_fixtures(root):
    normal=c.rows(root/'fixtures.jsonl');plans(normal,c.PLAN['normal_fixtures'],'normal fixtures');rr=[];unchanged=True
    for row in normal:
        z=c.load(file(root,row));want=c.fixture(row)
        c.require(all(n in z and eq(z[n],x) for n,x in want.items()),'Fixture input changed')
        c.require(row['input_before']==row['input_after']==c.arrays_hash(want),'Fixture input mutation')
        report=numerics(z,want['q']);analytic=True
        if row['pattern']=='zero':analytic=all(np.count_nonzero(z[a+'_output'])==0 for a in c.ARMS if a+'_output' in z)
        # Reference values for simple fixtures can be recomputed without GPU reduction.
        if row['pattern'] in ('zero','constant'):
            ref={side:qc.cpu_restore({n[len(side)+1:]:x for n,x in want.items() if n.startswith(side+'_')}) for side in ('k','v')}
            analytic=analytic and eq(ref['v'],z['reference_v']) and eq(ref['k'],z['reference_k'])
        report.update(id=row['id'],analytic=analytic);rr.append(report)
    negative=c.rows(root/'rejections.jsonl');plans(negative,[dict(kind=x) for x in c.PLAN['negative_fixtures']],'negative fixtures')
    checks=[]
    for row in negative:
        z=c.load(file(root,row));wanted,mutation=c.negative(row['kind']);c.require(set(z)==set(wanted) and all(eq(z[n],x) for n,x in wanted.items()),'Negative fixture changed');c.require(row['mutation']==mutation,'Mutation descriptor')
        plans(row['results'],[dict(arm=a) for a in c.ARMS],'Rejection arms')
        try:c.cpu_structure(wanted);structural=True
        except c.fc.ContractError:structural=False
        if not structural:stage='cpu_structure'
        elif mutation:stage='structural'
        elif row['kind']=='score_overflow':stage='attention'
        else:
            try:c.finite_input(wanted);finite=True
            except c.fc.ContractError:finite=False
            c.require(not finite,'Negative fixture must contain nonfinite input');stage='input_finite'
        good=all(r['rejected'] is True and r['stage']==stage and r['input_unchanged'] is True for r in row['results'])
        checks.append(dict(kind=row['kind'],expected_stage=stage,pass_rejection=good))
    return rr,checks

def verify(root,check_summary=True):
    root=Path(root);plan=c.read(root/'loop_kv_vf01_inputs.json');c.require(plan==c.PLAN,'Frozen input contract changed')
    manifest=c.read(root/'input_manifest.json');c.require(manifest==c.manifest(),'Material manifest')
    for n,h in c.read(root/'capsule.json').items():c.require(c.file_sha(root/n)==h,'Frozen source digest '+n)
    env=c.read(root/'environment.json');c.require(env['model_loaded'] is False and env['torch']=='2.11.0+cu128' and env['numpy']=='2.1.3','Environment mismatch')
    c.require(env['kernel']['reference']['baseline']['triton']=='3.6.0','Triton version')
    settings=c.read(root/'execution_settings.json')
    wanted=dict(flash=True,efficient=True,math=True,cudnn=True,gemm_bf16_reduced=True,math_sdp_reduced=False,matmul_tf32=False,cudnn_tf32=False,float32_matmul_precision='highest')
    c.require(settings==wanted,'Precision settings changed')
    fixtures,rejections=audit_fixtures(root)
    originals={sp['id']:c.sample(root,sp) for sp in c.SAMPLES}
    expected={}
    for i,sp in enumerate(c.SAMPLES):
        for q in c.QUERIES:
            a=c.with_query(originals[sp['id']],i,q);c.check_query(a['q'],sp,q);expected[sp['id']+'-'+q]=c.arrays_hash(a)
    before=c.read(root/'inputs_before.json');after=c.read(root/'inputs_after.json');c.require(before==after==expected,'GPU input fingerprints do not match frozen inputs')
    resident=c.read(root/'resident_inputs.json');c.require(resident==c.read(root/'resident_after.json'),'Resident input storage changed')
    # Exact unique input storage (9 K/V pairs plus 36 query arrays), no BF16 shadow cache.
    bytes_expected=sum(x.nbytes for a in originals.values() for n,x in a.items() if not n.endswith('_reference'))+36*16*128*2
    tensors_expected=sum(sum(not n.endswith('_reference') for n in a) for a in originals.values())+36
    c.require(resident==dict(unique_storage_bytes=bytes_expected,unique_storages=tensors_expected),'Unexpected persistent input buffers')
    first=c.rows(root/'first_calls.jsonl');plans(first,c.first_cells(),'first calls')
    warm=c.rows(root/'warmup.jsonl');plans(warm,c.first_cells(),'warmup')
    timing=c.rows(root/'timings.jsonl');plans(timing,c.schedule(),'timing')
    dr=c.rows(root/'diagnostics.jsonl');plans(dr,c.cases(),'diagnostics');reports=[];saved={}
    for row in dr:
        sp=next(s for s in c.SAMPLES if s['id']==row['sample']);a=originals[sp['id']];q=qc.query(c.SAMPLES.index(sp),row['query']);z=c.load(file(root,row))
        origk=np.concatenate([a['k_reference'],a['k_tail']],2);origv=np.concatenate([a['v_reference'],a['v_tail']],2)
        rep=numerics(z,q,origk,origv);rep.update(sample=sp['id'],query=row['query']);reports.append(rep)
        for arm in c.ARMS:
            if arm+'_normal' in z:saved[(row['sample'],row['query'],arm)]=z[arm+'_normal'].copy()
    ready=c.read(root/'timing_ready.json');c.require(ready['warmup_cells']==108 and ready['signatures']==sorted(set(c.signature(x) for x in first)) and ready['resident']==resident,'Timing preparation inventory')
    ready_time=ready['completed_monotonic'];c.require(all(x['finished_monotonic']<=ready_time for x in warm),'Warmup after ready')
    stable=c.read(root/'timing_cache_after.json')==ready['cache']
    full=True;timed_match=True;memory=[]
    for row in first+timing:
        if row['status']!='FINITE':full=False;continue
        time_row(row,1 if 'round' not in row else 20)
        arr=np.load(file(root,row),allow_pickle=False);key=(row['sample'],row['query'],row['arm'])
        typed(arr,(1,1,16,128),'uint16',True)
        timed_match=timed_match and key in saved and eq(arr,saved[key])
        if 'round' in row:
            c.require(row['started_monotonic']>=ready_time,'Timing before all warmup')
            memory.append(dict(id=row['id'],arm=row['arm'],allocated_before=row['allocated_before'],allocated_peak=row['allocated_peak'],allocated_extra=row['allocated_peak']-row['allocated_before'],reserved_before=row['reserved_before'],reserved_peak=row['reserved_peak'],reserved_extra=row['reserved_peak']-row['reserved_before']))
    for row in warm:
        if row['status']!='FINITE':full=False
        else:c.require(row['calls']==5,'Warmup calls')
    allreports=fixtures+reports
    complete=full and all(r['complete'] for r in allreports)
    route_ok=all(all(r['route'].values()) for r in allreports) and timed_match and all(r['analytic'] for r in fixtures)
    numeric_ok=complete and all(all(r['metrics'][a][key]['pass_budget'] for a in c.ARMS for key in ['v_restore','score','context','combined_context']) for r in allreports)
    reject_ok=all(x['pass_rejection'] for x in rejections)
    # Stable live allocations are a timing validity check, not hidden correction.
    baselines=[x['allocated_before'] for x in memory];memory_stable=len(set(baselines))==1
    speeds=c.aggregate(timing) if full else None
    h3=None if not full or not stable or not memory_stable else all(speeds[f]['V_FUSED/TILE_MATERIALIZE']['main']['median']<1 for f in ('cuda_ms','wall_ms'))
    result=dict(status='COMPLETE_COMPARISON' if complete else 'INCONCLUSIVE',experiment='LOOP-KV-VF01',planned=dict(pairs=9,cases=36,first_calls=108,warmup_calls=540,intervals=540,measured_calls=10800,normal_outputs=48,rejection_decisions=72),complete=complete,hypotheses=dict(H1=complete and route_ok and reject_ok,H2=numeric_ok,H3=h3),route_pass=route_ok,numerics_pass=numeric_ok,rejections_pass=reject_ok,timing_cache_unchanged=stable,timing_status='READY' if stable else 'TIMING_CACHE_CHANGED',allocated_baseline_unchanged=memory_stable,timing_valid=full and stable and memory_stable,fixtures=fixtures,rejections=rejections,cases=reports,timing=speeds,memory=memory,resident_inputs=resident)
    if check_summary:fc.equal(c.read(root/'summary.json'),result,'summary')
    return result
if __name__=='__main__':
    result=verify(sys.argv[1]);print('VF01_OFFLINE_VERIFIED',result['status'])
