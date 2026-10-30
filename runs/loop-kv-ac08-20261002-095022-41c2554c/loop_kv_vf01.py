"""VF01 saved-input worker, adapted from AF01. No model load or text retrieval."""
import gc,os,sys,time,signal,traceback
from pathlib import Path
import loop_kv_vf01_contract as c
import loop_kv_vf01_kernel as k
import loop_kv_vf01_checks as checks
import loop_kv_p0_diagnostic as d
w=k.w;ROOT=None;DEADLINE=None;BAD=set()

def guard():
    c.require(time.monotonic()<DEADLINE,'Experiment 20-minute cap')
    c.require(sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts)<=c.PLAN['output_limit_bytes'],'Output 1 GiB cap')
def alarm(*_):raise TimeoutError('VF01 condition 120-second cap')
def start_case(label):
    guard();c.write(ROOT/'active_case.json',dict(id=label,status='STARTED',deadline_monotonic=min(DEADLINE,time.monotonic()+120)));signal.alarm(120)
def end_case():signal.alarm(0);c.write(ROOT/'active_case.json',dict(status='FINISHED'))
def array(x,bits=False):
    x=x.detach().contiguous()
    if bits:x=x.view(w.torch.uint16)
    return x.cpu().numpy().copy()
def tensor(a,bf=False):
    x=w.torch.from_numpy(a.copy()).to('cuda');return x.view(w.torch.bfloat16) if bf else x

def inputs(a):
    c.cpu_structure(a)
    def group(side):
        if side+'_codes' not in a:return None
        return dict(codes=tensor(a[side+'_codes']),headers=tensor(a[side+'_headers']),metas=[tensor(a[side+'_meta'+str(i)]) for i in range(3)])
    return group('k'),tensor(a['k_tail'],True),tensor(a['q'],True),group('v'),tensor(a['v_tail'],True)
def input_arrays(data):
    g,kt,q,vg,vt=data;out={'q':array(q,True),'k_tail':array(kt,True),'v_tail':array(vt,True)}
    for side,group in [('k',g),('v',vg)]:
        if group is not None:
            out[side+'_codes']=array(group['codes']);out[side+'_headers']=array(group['headers'])
            out.update({side+'_meta'+str(i):array(x) for i,x in enumerate(group['metas'])})
    return out

def save_npz(folder,ident,a):
    p=ROOT/folder/(ident+'.npz');p.parent.mkdir(exist_ok=True);c.np.savez_compressed(p,**a)
    return dict(file=str(p.relative_to(ROOT)),sha256=c.file_sha(p))
def save_output(folder,ident,x):
    p=ROOT/folder/(ident+'.npy');p.parent.mkdir(exist_ok=True);c.np.save(p,array(x,True),allow_pickle=False)
    return dict(file=str(p.relative_to(ROOT)),sha256=c.file_sha(p))
def failure(label,exc):
    c.append(ROOT/'failures.jsonl',dict(id=label,stage=k.LAST_STAGE,error_type=type(exc).__name__,error=str(exc)))

def diagnostic(ident,data):
    g,kt,q,vg,vt=data
    saved=dict(reference_k=array(k.qk.reference_k(g,kt),True),reference_v=array(k.qk.reference_k(vg,vt),True))
    statuses={}
    for arm in c.ARMS:
        try:
            out,ctx,score,v=k.pipeline(arm,data,trace=True)
            saved.update({arm+'_output':array(out,True),arm+'_context':array(ctx),arm+'_score':array(score),arm+'_v':array(v,True)})
            del out,ctx,score,v
            plain=k.pipeline(arm,data);saved[arm+'_normal']=array(plain,True);del plain
            statuses[arm]='FINITE'
        except w.NumericalFailure as exc:
            failure(ident+'-'+arm,exc);statuses[arm]='REJECTED_NONFINITE';BAD.add((ident,arm))
    # Independent direct calls to the frozen baseline, not the new wrapper.
    try:
        v=k.qk.reference_k(vg,vt);saved['direct_v']=array(v,True)
        k.qk.validate(g,kt,q);k.af.validate_v(v,q,v.shape[2])
        out,ctx,score=k.af.launch(g,kt,q,v,full=True,trace=True)
        saved.update(direct_output=array(out,True),direct_context=array(ctx),direct_score=array(score));del v,out,ctx,score
    except w.NumericalFailure as exc:failure(ident+'-direct',exc);statuses['direct']='REJECTED_NONFINITE'
    return saved,statuses

def fixture_block():
    for sp in c.PLAN['normal_fixtures']:
        start_case('fixture:'+sp['id']);a=c.fixture(sp);data=inputs(a);before=c.arrays_hash(input_arrays(data))
        saved,statuses=diagnostic(sp['id'],data);saved.update(a)
        meta=save_npz('fixture_arrays',sp['id'],saved)
        c.append(ROOT/'fixtures.jsonl',dict(sp,**meta,statuses=statuses,input_before=before,input_after=c.arrays_hash(input_arrays(data))))
        del data,saved,a;end_case()
    for kind in c.PLAN['negative_fixtures']:
        start_case('negative:'+kind);a,mutation=c.negative(kind);meta=save_npz('negative_arrays',kind,a);results=[]
        for arm in c.ARMS:
            data=None;before=None;k.LAST_STAGE='cpu_structure'
            try:
                data=inputs(a)
                if mutation:
                    g,kt,q,vg,vt=data;vt=vt.transpose(-1,-2).contiguous().transpose(-1,-2);data=(g,kt,q,vg,vt)
                    c.require(not vt.is_contiguous(),'Noncontiguous fixture failed')
                before=c.arrays_hash(input_arrays(data))
                out=k.pipeline(arm,data);del out
                rr=dict(arm=arm,rejected=False,stage=k.LAST_STAGE,error_type=None)
            except (c.fc.ContractError,w.IntegrityError,w.NumericalFailure) as exc:
                rr=dict(arm=arm,rejected=True,stage=k.LAST_STAGE,error_type=type(exc).__name__)
            if data is not None:
                rr['input_unchanged']=before is not None and before==c.arrays_hash(input_arrays(data));del data
            else:rr['input_unchanged']=True
            results.append(rr)
        c.append(ROOT/'rejections.jsonl',dict(kind=kind,mutation=mutation,results=results,**meta));end_case()
    print('VF01 fixtures recorded: 16 normal cases and 24 rejection cases.',flush=True)

def resident_inventory(loaded):
    seen={}
    for data in loaded.values():
        g,kt,q,vg,vt=data
        for x in [kt,q,vt]+([] if g is None else [g['codes'],g['headers'],*g['metas']])+([] if vg is None else [vg['codes'],vg['headers'],*vg['metas']]):
            st=x.untyped_storage();seen[(str(x.device),st.data_ptr())]=st.nbytes()
    return dict(unique_storage_bytes=sum(seen.values()),unique_storages=len(seen))

def timed(arm,data,n):
    t=w.torch;t.cuda.synchronize();a=t.cuda.Event(enable_timing=True);b=t.cuda.Event(enable_timing=True)
    began=time.monotonic();before=t.cuda.memory_allocated();reserved=t.cuda.memory_reserved();t.cuda.reset_peak_memory_stats();start=time.perf_counter();a.record()
    out=None
    for _ in range(n):
        if out is not None:del out
        out=k.pipeline(arm,data)
    b.record();b.synchronize();elapsed=time.perf_counter()-start
    return out,dict(started_monotonic=began,finished_monotonic=time.monotonic(),calls=n,cuda_ms=a.elapsed_time(b)/n,wall_ms=elapsed*1000/n,allocated_before=before,allocated_peak=t.cuda.max_memory_allocated(),reserved_before=reserved,reserved_peak=t.cuda.max_memory_reserved(),output_storage_bytes=out.untyped_storage().nbytes())
def cache_snapshot():
    p=Path(os.environ['TRITON_CACHE_DIR']);return {str(x.relative_to(p)):c.file_sha(x) for x in sorted(p.rglob('*')) if x.is_file() and not x.name.endswith('.lock') and '.tmp' not in x.name}

def timing_block():
    t=w.torch;loaded={}
    for i,sp in enumerate(c.SAMPLES):
        a=c.with_query(c.sample(ROOT,sp),i,'gaussian');g,kt,_,vg,vt=inputs(a)
        for kind in c.QUERIES:loaded[sp['id']+'-'+kind]=(g,kt,tensor(c.qc.query(i,kind),True),vg,vt)
        del a
    del g,kt,vg,vt,_
    original={sid:c.arrays_hash(input_arrays(data)) for sid,data in loaded.items()}
    c.write(ROOT/'inputs_before.json',original);storage=resident_inventory(loaded);c.write(ROOT/'resident_inputs.json',storage)
    for cell in c.first_cells():
        ident=c.name(cell);label=ident+'-'+cell['arm'];start_case('first:'+label);before=cache_snapshot()
        try:
            out,tm=timed(cell['arm'],loaded[ident],1);meta=save_output('first_outputs',label,out);del out
            row=dict(cell,status='FINITE',signature=c.signature(cell),cache_before=before,cache_after=cache_snapshot(),**meta,**tm)
        except w.NumericalFailure as exc:
            failure(label,exc);BAD.add((ident,cell['arm']));row=dict(cell,status='REJECTED_NONFINITE')
        c.append(ROOT/'first_calls.jsonl',row);end_case()
    # All diagnostic kernel compilations precede warmup and the fixed timing cache.
    for cell in c.cases():
        ident=c.name(cell);start_case('diagnostic:'+ident)
        saved,statuses=diagnostic(ident,loaded[ident]);meta=save_npz('diagnostics',ident,saved)
        c.append(ROOT/'diagnostics.jsonl',dict(cell,statuses=statuses,**meta));del saved;end_case()
    for cell in c.first_cells():
        ident=c.name(cell);start_case('warm:'+ident+'-'+cell['arm']);status='SKIPPED_NONFINITE'
        if (ident,cell['arm']) not in BAD:
            try:
                for _ in range(c.WARMUP):out=k.pipeline(cell['arm'],loaded[ident]);del out
                t.cuda.synchronize();status='FINITE'
            except w.NumericalFailure as exc:failure(ident+'-'+cell['arm'],exc);BAD.add((ident,cell['arm']))
        c.append(ROOT/'warmup.jsonl',dict(cell,signature=c.signature(cell),status=status,calls=c.WARMUP if status=='FINITE' else None,finished_monotonic=time.monotonic()));end_case()
    gc.collect();t.cuda.synchronize()
    # No restored V or TRACE buffer remains live here. Only common packed inputs/queries.
    ready=cache_snapshot();c.write(ROOT/'timing_ready.json',dict(warmup_cells=108,signatures=sorted(set(c.signature(x) for x in c.first_cells())),resident=resident_inventory(loaded),cache=ready,completed_monotonic=time.monotonic()))
    for plan in c.schedule():
        ident=c.name(plan);start_case('timing:'+plan['id']);status='SKIPPED_NONFINITE'
        row=dict(plan,status=status)
        if (ident,plan['arm']) not in BAD:
            try:
                out,tm=timed(plan['arm'],loaded[ident],c.REPEATS);meta=save_output('timing_outputs',plan['id'],out);del out
                row=dict(plan,status='FINITE',**meta,**tm)
            except w.NumericalFailure as exc:failure(plan['id'],exc);BAD.add((ident,plan['arm']));row['status']='REJECTED_NONFINITE'
        c.append(ROOT/'timings.jsonl',row);end_case()
        if plan==c.schedule()[(plan['round']+1)*108-1]:print('VF01 timing round',plan['round']+1,'/ 5 complete.',flush=True)
    c.write(ROOT/'timing_cache_after.json',cache_snapshot())
    c.write(ROOT/'inputs_after.json',{sid:c.arrays_hash(input_arrays(data)) for sid,data in loaded.items()})
    c.write(ROOT/'resident_after.json',resident_inventory(loaded));del loaded;gc.collect();t.cuda.synchronize()

def main(root):
    global ROOT,DEADLINE
    ROOT=Path(root);started=time.monotonic();status='INCONCLUSIVE'
    signal.signal(signal.SIGALRM,alarm);signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('Launcher termination')))
    try:
        for n,h in c.read(ROOT/'capsule.json').items():c.require(c.file_sha(ROOT/n)==h,'Source changed: '+n)
        d.numeric_imports();t=w.torch;t.set_grad_enabled(False)
        c.require(str(t.__version__)=='2.11.0+cu128' and w.np.__version__=='2.1.3','Fixed torch/NumPy')
        c.require(t.cuda.is_available() and t.cuda.is_bf16_supported(),'CUDA BF16 required')
        t.use_deterministic_algorithms(True);t.backends.cuda.matmul.allow_tf32=False;t.backends.cudnn.allow_tf32=False;t.backends.cudnn.benchmark=False;t.backends.cudnn.deterministic=True;t.set_float32_matmul_precision('highest')
        c.write(ROOT/'environment.json',dict(torch=str(t.__version__),numpy=w.np.__version__,cuda=t.version.cuda,gpu=t.cuda.get_device_name(),kernel=k.qk.environment(),model_loaded=False))
        c.extract(c.read(ROOT/'input_path.json')['path'],ROOT)
        c.require(time.monotonic()<float(os.environ['LOOP_KV_VF01_PREPARE_STARTED'])+c.CAPS['prepare'],'Shared preparation cap')
        c.require(not cache_snapshot(),'Dedicated cache must begin empty')
        DEADLINE=time.monotonic()+c.CAPS['experiment'];c.write(ROOT/'phase.json',dict(stage='EXPERIMENT',deadline_monotonic=DEADLINE))
        with d.precision_context(dict(kernel='AUTO',gemm_reduced=True)) as settings:
            c.write(ROOT/'execution_settings.json',dict(settings,float32_matmul_precision=t.get_float32_matmul_precision()))
            fixture_block();timing_block()
        from verify_loop_kv_vf01 import verify
        c.write(ROOT/'phase.json',dict(stage='VERIFY',deadline_monotonic=time.monotonic()+120));signal.alarm(120)
        result=verify(ROOT,check_summary=False);signal.alarm(0);c.write(ROOT/'summary.json',result);status=result['status']
    except BaseException as exc:
        signal.alarm(0);c.write(ROOT/'failure.json',dict(error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc()));print(traceback.format_exc(),flush=True)
    finally:
        signal.alarm(0)
        if not (ROOT/'summary.json').exists():c.write(ROOT/'summary.json',dict(status='INCONCLUSIVE',experiment='LOOP-KV-VF01',reason='See partial evidence/failure'))
        c.write(ROOT/'run_status.json',dict(status=status,experiment='LOOP-KV-VF01',model_loaded=False,seconds=time.monotonic()-started));print('Closed VF01:',status,flush=True)
if __name__=='__main__':main(sys.argv[1])
