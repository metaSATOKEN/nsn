"""NumPy-only AC08 fixture and saved-query audit. No worker or CUDA imports."""
import math
from pathlib import Path
import numpy as np
import loop_kv_af01_contract as fc
import loop_kv_ac08_contract as ac
import loop_kv_vf01_contract as vc
import verify_loop_kv_vf01 as va
qc=fc.qc
def file(root,row):
    p=Path(row['file']);fc.require(not p.is_absolute() and '..' not in p.parts,'Unsafe evidence path')
    p=root/p;fc.require(fc.sha(p.read_bytes())==row['sha256'],'Evidence hash: '+str(p));return p

def typed(a,shape,dtype,bits=False):
    fc.require(a.shape==shape and str(a.dtype)==dtype,'Evidence shape/dtype')
    fc.require(np.isfinite(qc.from_bits(a) if bits else a).all(),'Nonfinite evidence');return a

def plans(got,want,label):
    fc.require(len(got)==len(want),'Missing/extra '+label)
    for row,plan in zip(got,want):
        fc.require(all(k in row and type(row[k])==type(v) and row[k]==v for k,v in plan.items()),'Order/ID '+label)

def time_row(row,n):
    fc.require(row['calls']==n,'Timing calls')
    for key in ('cuda_ms','wall_ms','started_monotonic','finished_monotonic'):
        fc.require(isinstance(row[key],(int,float)) and math.isfinite(row[key]) and row[key]>0,'Invalid timing')
    fc.require(row['finished_monotonic']>=row['started_monotonic'],'Time order')
    fc.require(isinstance(row['allocated_before'],int) and isinstance(row['allocated_peak'],int) and 0<=row['allocated_before']<=row['allocated_peak'],'Memory metrics')

def fixtures(root):
    rr=fc.rows(root/'fixtures.jsonl');plans(rr,fc.fixtures(),'fixtures')
    fc.equal(fc.rows(root/'rejections.jsonl'),[dict(kind=k,rejected_before_triton=True) for k in fc.NEGATIVE],'input rejection')
    result=[]
    for sp,r in zip(fc.fixtures(),rr):
        z=fc.load(file(root,r));original=fc.fixture_data(sp)
        fc.require(set(z)==set(original)|{'trace_score','context','output'},'Fixture fields')
        for key,a in original.items():fc.require(z[key].dtype==a.dtype and np.array_equal(z[key],a),'Fixture input changed')
        score=typed(z['trace_score'],(1,2,sp['length']),'float32');ctx=typed(z['context'],(1,2,1,128),'float32')
        out=typed(z['output'],(1,1,2,128),'uint16',True)
        fc.require(np.array_equal(fc.bf16_bits(ctx).transpose(0,2,1,3),out),'Fixture output rounding')
        metrics=dict(id=sp['id'],context_budget=fc.context_metrics(score,z['v'],ctx))
        if sp['kind']=='finish':fc.require(np.array_equal(score,z['score']),'Finish score input changed')
        else:
            gold,scale=qc.gold(z['tail'],z['q']);metrics['score_budget']=qc.score_metrics(score,gold,scale)
        result.append(metrics)
    return dict(calls=len(rr),input_rejections=len(fc.NEGATIVE),cases=result,
        context_budget_pass=sum(r['context_budget']['pass_budget'] for r in result),
        score_budget_pass=sum(r['score_budget']['pass_budget'] for r in result if 'score_budget' in r))


def diagnostic(root,row,folder,captured):
    name=row['id'];length=row['length'];comp=ac.forward_compressed(length)
    ac.require(row['compressed']==comp and row['slot'] in ac.CAPTURE_SLOTS,'Diagnostic coordinates')
    ac.require(row['file']==f'{folder}/{name}.npz' and name==f"{row['trajectory']}-L{length}-S{row['slot']}",'Diagnostic identity')
    original=captured[name]
    ac.require(row['capture_file']==original['file'] and row['capture_sha256']==original['sha256'],'Capture link')
    ac.require(all(row[k]==original[k] for k in ('id','trajectory','slot','length','compressed')),'Capture coordinates')
    inputs=fc.load(file(root,original));z=fc.load(file(root,row))
    a={n:x for n,x in inputs.items() if n!='actual_output'}
    vc.cpu_structure(a)
    ac.require(set(a)=={'q','k_tail','v_tail'}|{side+'_'+n for side in ('k','v') for n in ('codes','headers','meta0','meta1','meta2')},'Capture fields')
    expected=set(inputs)|{'reference_k','reference_v'}|{arm+'_'+n for arm in vc.ARMS for n in ('score','context','output','normal','v')}|{'direct_'+n for n in ('v','output','context','score')}
    ac.require(set(z)==expected,'Replay fields')
    for n,x in inputs.items():ac.require(z[n].dtype==x.dtype and np.array_equal(z[n],x),'Captured bytes changed: '+n)
    typed(a['q'],(1,16,1,128),'uint16',True)
    fp={n:fc.array_hash(x) for n,x in a.items()}
    for x in (original['input_fingerprints'],row['input_before'],row['input_after']):fc.equal(x,fp,'Capture fingerprint')
    for side in ('k','v'):
        ref=z['reference_'+side];typed(ref,(1,16,length,128),'uint16',True)
        sp=dict(compressed=comp,length=length,cache_digest=ac.sha(str(((1,16,length,128),'torch.bfloat16')).encode()+ref.tobytes()))
        qc.check_sample({**{n:a[side+'_'+n] for n in ('codes','headers','meta0','meta1','meta2')},'tail':a[side+'_tail'],'reference':ref[:,:,:comp].copy()},sp)
    report=va.numerics(z,a['q']);typed(z['actual_output'],(1,1,16,128),'uint16',True)
    ac.require(np.array_equal(z['actual_output'],z['V_FUSED_normal']),'Actual forward / normal mismatch')
    ac.require(set(row['statuses'])==set(vc.ARMS) and all(x=='FINITE' for x in row['statuses'].values()),'Replay status')
    budget=report['complete'] and all(report['metrics'][arm][key]['pass_budget'] for arm in vc.ARMS for key in ('v_restore','score','context','combined_context'))
    return dict(id=name,actual_normal_bit_equal=True,route_pass=all(report['route'].values()),pass_budget=budget,numerics=report)

def captured_queries(root):
    rows=fc.rows(root/'query_capture.jsonl');wanted=ac.warmup_diagnostic_ids()+ac.diagnostic_ids()
    ac.require([r['id'] for r in rows]==wanted,'Capture inventory/order')
    byid={r['id']:r for r in rows}
    for r in rows:
        ac.require(r['file']==f"query_inputs/{r['id']}.npz",'Capture file identity')
        ac.require(r['phase']==('warmup' if r['id'] in ac.warmup_diagnostic_ids() else 'main'),'Capture phase')
    outputs={}
    for folder,ids in [('warmup_query',ac.warmup_diagnostic_ids()),('real_query',ac.diagnostic_ids())]:
        rr=fc.rows(root/folder/'index.jsonl');ac.require([r['id'] for r in rr]==ids,'Replay inventory/order')
        outputs[folder]=[diagnostic(root,r,folder,byid) for r in rr]
    return outputs
