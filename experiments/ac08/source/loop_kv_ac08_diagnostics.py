"""Capture exact model inputs, then replay after the trajectory's measurements."""
from pathlib import Path
import time
import loop_kv_ac08_contract as ac
import loop_kv_af01_contract as fc
import loop_kv_qk01_kernel as qk
import loop_kv_af01_kernel as fused
import loop_kv_ac01_attention as legacy
import loop_kv_vf01 as vf
w=qk.w

def array(x,bits=False):
    x=x.detach().contiguous()
    if bits:x=x.view(w.torch.uint16)
    return x.cpu().numpy().copy()
def tensor(a,bf=False):
    x=w.torch.from_numpy(a.copy()).to('cuda')
    return x.view(w.torch.bfloat16) if bf else x
def tensors(data):
    return (dict(codes=tensor(data['codes']),headers=tensor(data['headers']),metas=[tensor(data['meta'+str(i)]) for i in range(3)]),tensor(data['tail'],True),tensor(data['q'],True),tensor(data['v'],True))
def input_arrays(g,tail,q,v):
    return dict(codes=array(g['codes']),headers=array(g['headers']),**{'meta'+str(i):array(x) for i,x in enumerate(g['metas'])},tail=array(tail,True),q=array(q,True),v=array(v,True))
def fingerprints(arrays):return {n:fc.array_hash(a) for n,a in arrays.items()}

def capture(root,sp,slot,length,g,tail,q,vg,vt,out):
    if slot not in ac.CAPTURE_SLOTS or not ac.should_capture(sp,length):return
    name=f"{sp['id']}-L{length}-S{slot}";folder=root/'query_inputs';folder.mkdir(exist_ok=True)
    arrays=vf.input_arrays((g,tail,q,vg,vt));fp=fingerprints(arrays);arrays['actual_output']=array(out,True)
    file=folder/(name+'.npz');w.np.savez_compressed(file,**arrays)
    fc.append(root/'query_capture.jsonl',dict(id=name,trajectory=sp['id'],phase=sp['phase'],slot=slot,length=length,compressed=g['codes'].shape[2]*64,file=str(file.relative_to(root)),sha256=fc.sha(file.read_bytes()),input_fingerprints=fp))

def replay(root,sp,check_time):
    # All diagnostic work follows this trajectory's measured intervals.
    vf.ROOT=root
    for row in [r for r in fc.rows(root/'query_capture.jsonl') if r['trajectory']==sp['id']]:
        check_time();a=fc.load(root/row['file']);inputs={n:x for n,x in a.items() if n!='actual_output'}
        data=vf.inputs(inputs);before=fingerprints(vf.input_arrays(data))
        z,statuses=vf.diagnostic(row['id'],data);z.update(a)
        after=fingerprints(vf.input_arrays(data));dest=root/('real_query' if sp['phase']=='main' else 'warmup_query');dest.mkdir(exist_ok=True)
        file=dest/(row['id']+'.npz');w.np.savez_compressed(file,**z)
        fc.append(dest/'index.jsonl',dict(id=row['id'],trajectory=sp['id'],slot=row['slot'],length=row['length'],compressed=row['compressed'],file=str(file.relative_to(root)),sha256=fc.sha(file.read_bytes()),input_before=before,input_after=after,capture_file=row['file'],capture_sha256=row['sha256'],statuses=statuses))
        w.require(before==after==row['input_fingerprints'],'Captured input changed during replay')
        w.require(all(v=='FINITE' for v in statuses.values()),'Nonfinite local replay')
        w.require(w.np.array_equal(z['actual_output'],z['V_FUSED_normal']),'Actual forward / normal output mismatch')
        del data,z,a,inputs;check_time()

def kv_preflight(root,check_time):
    # Use the exact NC01 GPU helpers with AC08's deadline and evidence destination.
    import loop_kv_nc01 as nc
    from loop_kv_ac08_preflight_audit import verify
    dest=root/'kv_validation';dest.mkdir(exist_ok=True)
    previous=(nc.ROOT,vf.ROOT,nc.start_case,nc.end_case)
    deadline=time.monotonic()+ac.CAPS['input_fixtures']
    def start(name):
        check_time();w.require(time.monotonic()<deadline,'KV preflight cap')
        ac.write(root/'active_case.json',dict(id='kv_preflight:'+name,status='STARTED',deadline_monotonic=min(deadline,time.monotonic()+120)))
    def end():
        check_time();w.require(time.monotonic()<deadline,'KV preflight cap');ac.write(root/'active_case.json',dict(status='FINISHED'))
    nc.ROOT=dest;vf.ROOT=dest;nc.start_case=start;nc.end_case=end
    try:
        for spec in nc.c.SPECS:
            a=nc.c.base.fixture(spec);data=vf.inputs(a)
            nc.controls(spec,data,a);nc.witness(spec,data)
            for attempt in nc.c.ATTEMPTS:
                if attempt['fixture']==spec['id']:nc.negative(spec,attempt,data)
            del a,data
        result=verify(dest);ac.write(root/'kv_validation_checks.json',result)
        w.require(result['status']=='PASS','KV validation preflight failed')
    finally:nc.ROOT,vf.ROOT,nc.start_case,nc.end_case=previous

def fixture_block(root,check_time):
    t=w.torch
    for sp in fc.fixtures():
        check_time();a=fc.fixture_data(sp)
        q,tail,v=(tensor(a[n],True) for n in ('q','tail','v'));score=tensor(a['score'])
        qk.validate(None,tail,q);fused.validate_v(v,q,sp['length'])
        if sp['kind']=='finish':fused.validate_score(score,v)
        out,ctx,trace=fused.launch(None,tail,q,v,score=score,full=sp['kind']=='full',trace=True)
        saved={**a,'trace_score':array(trace),'context':array(ctx),'output':array(out,True)}
        w.require(w.np.isfinite(saved['context']).all(),'Nonfinite fixture')
        file=root/'fixture_arrays'/(sp['id']+'.npz');file.parent.mkdir(exist_ok=True);w.np.savez_compressed(file,**saved)
        fc.append(root/'fixtures.jsonl',dict(sp,file=str(file.relative_to(root)),sha256=fc.sha(file.read_bytes())))
    a=fc.fixture_data(fc.fixtures()[0])
    for kind in fc.NEGATIVE:
        check_time();q,tail,v=(tensor(a[n],True) for n in ('q','tail','v'));score=tensor(a['score'])
        if kind=='q_nan':q.flatten()[0]=float('nan')
        elif kind=='v_nan':v.flatten()[0]=float('nan')
        elif kind=='score_nan':score.flatten()[0]=float('nan')
        elif kind=='score_dtype':score=score.double()
        elif kind=='score_shape':score=score[:,:,:-1].contiguous()
        elif kind=='empty_tail':tail=tail[:,:,:0].contiguous()
        rejected=False
        try:qk.validate(None,tail,q);fused.validate_v(v,q,17);fused.validate_score(score,v)
        except (w.IntegrityError,w.NumericalFailure):rejected=True
        fc.append(root/'rejections.jsonl',dict(kind=kind,rejected_before_triton=rejected))
        w.require(rejected,'Invalid input accepted: '+kind)
    check_time()
