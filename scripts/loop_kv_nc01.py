"""NC01 bounded fixture check. Reuses the actual VF01 worker helpers and kernels."""
import os,sys,signal,time,traceback
from pathlib import Path
import loop_kv_nc01_contract as c
import loop_kv_nc01_layout as layout
import loop_kv_vf01 as vf
k=vf.k;w=vf.w;checks=vf.checks;d=vf.d
ROOT=None;DEADLINE=None

def alarm(*_):raise TimeoutError('NC01 condition/verification cap')
def start_case(name):
    c.require(time.monotonic()<DEADLINE,'NC01 experiment five-minute cap')
    c.require(sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts)<=c.PLAN['output_limit_bytes'],'NC01 output cap')
    signal.alarm(c.CAPS['condition']);c.write(ROOT/'active_case.json',dict(id=name,status='STARTED',deadline_monotonic=min(DEADLINE,time.monotonic()+c.CAPS['condition'])))
def end_case():signal.alarm(0);c.write(ROOT/'active_case.json',dict(status='FINISHED'))

def controls(sp,data,a):
    start_case('control:'+sp['id']);before=c.arrays_hash(vf.input_arrays(data));z,statuses=vf.diagnostic(sp['id'],data);z.update(a)
    meta=vf.save_npz('controls',sp['id'],z)
    c.append(ROOT/'controls.jsonl',dict(sp,statuses=statuses,input_before=before,input_after=c.arrays_hash(vf.input_arrays(data)),v_layout=layout.layout(data[-1]),**meta));end_case()

def witness(sp,data):
    start_case('old:'+sp['id']);vt=data[-1];old=layout.old_view(vt)
    z=dict(original=vf.array(vt,True),old_view=vf.array(old,True));meta=vf.save_npz('old_layouts',sp['id'],z)
    c.append(ROOT/'old_layouts.jsonl',dict(fixture=sp['id'],original_layout=layout.layout(vt),old_layout=layout.layout(old),values_equal=c.eq(z['original'],z['old_view']),**meta));del old;end_case()

def negative(spec,attempt,data):
    start_case(attempt['id']);vt=data[-1];record=dict(attempt,construction='FAILED',target_called=False,result='NOT_RUN',stage=None,error_type=None,error=None,input_unchanged=None,finite_launch_delta=None)
    # No assertion is used to mislabel a failed construction as a guard rejection.
    view,backing=layout.noncontiguous_view(w.torch,vt)
    newdata=(*data[:-1],view)
    z=dict(original=vf.array(vt,True),view_before=vf.array(view,True),backing_before=vf.array(backing,True))
    original_layout=layout.layout(vt);view_layout=layout.layout(view);backing_layout=layout.layout(backing)
    shared=view.untyped_storage().data_ptr()==backing.untyped_storage().data_ptr()
    ready=c.construction_checks(z['original'],z['view_before'],z['backing_before'],original_layout,view_layout,backing_layout,shared)
    record.update(construction_checks=ready,original_layout=original_layout,view_layout=view_layout,backing_layout=backing_layout,shared_storage=shared)
    if all(ready.values()):
        record['construction']='READY';record['target_called']=True
        before=c.arrays_hash(vf.input_arrays(newdata));record['input_before']=before
        count=checks.launch_count;k.LAST_STAGE='not_started';fatal=None
        try:
            out=k.pipeline(attempt['arm'],newdata);z['unexpected_output']=vf.array(out,True);del out
            record['result']='ACCEPTED'
        except (w.IntegrityError,w.NumericalFailure) as exc:
            record.update(result='REJECTED',error_type=type(exc).__name__,error=str(exc))
        except BaseException as exc:
            record.update(result='ERROR',error_type=type(exc).__name__,error=str(exc));fatal=exc
        record.update(stage=k.LAST_STAGE,finite_launch_delta=checks.launch_count-count)
        # On a device error, retain the attempt before trying further GPU reads.
        if fatal is not None:
            record.update(**vf.save_npz('attempt_arrays',attempt['id'],z));c.append(ROOT/'attempts.jsonl',record);raise fatal
        after=c.arrays_hash(vf.input_arrays(newdata));record['input_after']=after
        z.update(view_after=vf.array(view,True),backing_after=vf.array(backing,True))
        record['input_unchanged']=before==after and c.eq(z['view_before'],z['view_after']) and c.eq(z['backing_before'],z['backing_after'])
    record.update(**vf.save_npz('attempt_arrays',attempt['id'],z));c.append(ROOT/'attempts.jsonl',record);end_case()

def main(root):
    global ROOT,DEADLINE
    ROOT=Path(root);vf.ROOT=ROOT;started=time.monotonic();status='INCONCLUSIVE'
    signal.signal(signal.SIGALRM,alarm);signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('Launcher termination')))
    try:
        for n,h in c.read(ROOT/'capsule.json').items():c.require(c.file_sha(ROOT/n)==h,'Source digest '+n)
        d.numeric_imports();t=w.torch;t.set_grad_enabled(False)
        c.require(str(t.__version__)=='2.11.0+cu128' and w.np.__version__=='2.1.3','Fixed torch/NumPy')
        c.require(t.cuda.is_available() and t.cuda.is_bf16_supported(),'CUDA BF16 required')
        t.use_deterministic_algorithms(True);t.backends.cuda.matmul.allow_tf32=False;t.backends.cudnn.allow_tf32=False;t.backends.cudnn.benchmark=False;t.backends.cudnn.deterministic=True;t.set_float32_matmul_precision('highest')
        c.write(ROOT/'environment.json',dict(torch=str(t.__version__),numpy=w.np.__version__,cuda=t.version.cuda,gpu=t.cuda.get_device_name(),kernel=k.qk.environment(),model_loaded=False))
        c.require(time.monotonic()<float(os.environ['LOOP_KV_NC01_PREPARE_STARTED'])+c.CAPS['prepare'],'Shared preparation cap')
        DEADLINE=time.monotonic()+c.CAPS['experiment'];c.write(ROOT/'phase.json',dict(stage='EXPERIMENT',deadline_monotonic=DEADLINE))
        with d.precision_context(dict(kernel='AUTO',gemm_reduced=True)) as settings:
            c.write(ROOT/'execution_settings.json',dict(settings,float32_matmul_precision=t.get_float32_matmul_precision()))
            for spec in c.SPECS:
                start_case('inputs:'+spec['id']);a=c.base.fixture(spec);data=vf.inputs(a);end_case()
                controls(spec,data,a);witness(spec,data)
                for attempt in c.ATTEMPTS:
                    if attempt['fixture']==spec['id']:negative(spec,attempt,data)
                del data,a
                print('NC01 completed fixture',spec['id'],flush=True)
        from verify_loop_kv_nc01 import verify
        c.write(ROOT/'phase.json',dict(stage='VERIFY',deadline_monotonic=time.monotonic()+c.CAPS['worker_verify']));signal.alarm(c.CAPS['worker_verify'])
        result=verify(ROOT,check_summary=False);signal.alarm(0);c.write(ROOT/'summary.json',result);status=result['status']
    except BaseException as exc:
        signal.alarm(0);c.write(ROOT/'failure.json',dict(error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc()));print(traceback.format_exc(),flush=True)
    finally:
        signal.alarm(0)
        if not (ROOT/'summary.json').exists():c.write(ROOT/'summary.json',dict(status='INCONCLUSIVE',experiment='LOOP-KV-NC01',reason='See partial evidence/failure'))
        c.write(ROOT/'run_status.json',dict(status=status,experiment='LOOP-KV-NC01',model_loaded=False,seconds=time.monotonic()-started));print('Closed NC01:',status,flush=True)
if __name__=='__main__':main(sys.argv[1])
