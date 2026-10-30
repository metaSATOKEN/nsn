"""IV01 saved-query GPU comparison. No model or HF access."""
import os,sys,time,signal,traceback,zipfile,gc
from pathlib import Path
import loop_kv_iv01_contract as c
import loop_kv_iv01_kernel as k
import loop_kv_rf01_kernel as rf
import loop_kv_af01_kernel as af
w=k.w;ROOT=None;DEADLINE=None

def start_case(name):
 c.require(time.monotonic()<DEADLINE,'Phase cap');c.require(sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())<512*1024**2,'512 MiB cap')
 c.write(ROOT/'active_case.json',dict(id=name,status='STARTED',deadline_monotonic=time.monotonic()+120));signal.alarm(120)
def end_case():signal.alarm(0);c.write(ROOT/'active_case.json',dict(status='FINISHED'))
def phase(name,sec):
 global DEADLINE
 DEADLINE=time.monotonic()+sec;c.write(ROOT/'phase.json',dict(stage=name,deadline_monotonic=DEADLINE))
def tensor(a,bf=False):
 x=w.torch.from_numpy(a.copy()).to('cuda');return x.view(w.torch.bfloat16) if bf else x
def array(x):
 a=x.detach().contiguous();return a.view(w.torch.uint16).cpu().numpy().copy() if x.dtype==w.torch.bfloat16 else a.cpu().numpy().copy()
def source_data(a):
 return dict(g=dict(codes=tensor(a['k_codes']),metas=[tensor(a['k_meta'+str(i)]) for i in range(3)]),tail=tensor(a['k_tail'],True),q=tensor(a['q'],True),v=tensor(a['v_full'],True),vg=dict(codes=tensor(a['v_codes']),metas=[tensor(a['v_meta'+str(i)]) for i in range(3)]),vt=tensor(a['v_tail'],True))
def named(data):
 a={n:data[n] for n in ('q','tail','v')};g=data['g']
 if g is not None:a.update(codes=g['codes'],**{'meta'+str(i):x for i,x in enumerate(g['metas'])})
 if 'vg' in data:a.update(v_codes=data['vg']['codes'],v_tail=data['vt'],**{'v_meta'+str(i):x for i,x in enumerate(data['vg']['metas'])})
 return a
def snapshot(data):
 xs=named(data);arrays={n:array(x) for n,x in xs.items()}
 desc=dict(packed=data['g'] is not None,meta_count=0 if data['g'] is None else len(data['g']['metas']),tensors={n:dict(shape=list(x.shape),dtype=str(x.dtype),contiguous=x.is_contiguous(),device=x.device.type,stride=list(x.stride())) for n,x in xs.items()})
 return desc,arrays
def fingerprints(data):
 desc,aa=snapshot(data);return dict(descriptor=desc,arrays={n:c.fingerprint(x) for n,x in aa.items()})
def make_fixture(sp):
 a=c.fixture_arrays(sp.get('case','zero') if sp['kind']=='normal' else 'zero')
 d={n:tensor(a[n],True) for n in ['q','tail','v']};d['g']=dict(codes=tensor(a['codes']),metas=[tensor(a['meta'+str(i)]) for i in range(3)]) if 'codes' in a else None
 if sp['kind']=='nonfinite':
  x=d[sp['field']] if sp['field'] in ('q','tail','v') else d['g']['metas'][int(sp['field'][-1])]
  x.flatten()[0 if sp['position']=='first' else -1]={'nan':float('nan'),'posinf':float('inf'),'neginf':-float('inf')}[sp['value']]
 elif sp['kind']=='structural':
  case=sp['case'];t=w.torch
  def noncontig(x):
   y=t.empty((*x.shape[:-1],x.shape[-1]*2),dtype=x.dtype,device=x.device);v=y[...,::2];v.copy_(x);return v
  if case=='empty_sequence':d['g']=None;d['tail']=d['tail'][:,:,:0];d['v']=d['v'][:,:,:0]
  elif case.startswith(('q_','tail_','v_')):
   key,op=case.split('_');x=d[key]
   d[key]=x.float() if op=='dtype' else noncontig(x) if op=='noncontig' else x.cpu() if op=='device' else x[...,:127].contiguous()
  elif case.startswith('codes_'):
   op=case.split('_')[1];x=d['g']['codes'];d['g']['codes']=x.to(t.int16) if op=='dtype' else x.cpu() if op=='device' else x[...,:3071].contiguous()
  else:
   op=case.split('_')[1];xs=d['g']['metas']
   if op=='count':d['g']['metas']=xs[:2]
   else:
    x=xs[1];xs[1]=x.float() if op=='dtype' else noncontig(x) if op=='noncontig' else x.cpu() if op=='device' else x[...,:127].contiguous()
 return d

def fixture_block():
 for sp in c.fixtures():
  start_case('fixture:'+sp['id']);data=make_fixture(sp);desc,a=snapshot(data);expected,kind=c.cpu_accept(desc,a)
  c.require(expected==sp['accept'] and kind==sp['kind'],'Fixture oracle differs '+sp['id'])
  path=ROOT/'fixture_arrays'/(sp['id']+'.npz');path.parent.mkdir(exist_ok=True);c.np.savez_compressed(path,**a)
  results=[];before=fingerprints(data)
  for arm in c.ARMS:
   entered=[0];count=k.launch_count;error=None
   def next_step():entered[0]+=1;return True
   try:k.checked_call(arm,data['g'],data['tail'],data['q'],data['v'],next_step);accepted=True
   except (w.IntegrityError,w.NumericalFailure) as exc:accepted=False;error=type(exc).__name__
   delta=k.launch_count-count
   results.append(dict(arm=arm,accepted=accepted,error_type=error,continued=entered[0],checker_launches=delta))
   c.require(accepted==expected and entered[0]==int(expected),'Acceptance/continuation mismatch '+sp['id'])
   c.require(delta==(1 if arm=='TRITON_GROUPED' and kind!='structural' else 0),'Unexpected checker launch')
  c.require(before==fingerprints(data),'Fixture mutation')
  c.append(ROOT/'fixtures.jsonl',dict(sp,descriptor=desc,file=str(path.relative_to(ROOT)),sha256=c.file_sha(path),results=results,input_unchanged=True));end_case()
 print('64 fixtures x 3 methods matched independent oracle',flush=True)

def run_cell(arm,scope,data):
 t=w.torch
 if scope=='CHECK_ONLY':return k.validate(arm,data['g'],data['tail'],data['q'],data['v'])
 v=t.cat((rf.restore_group(data['vg'],t.bfloat16),data['vt']),-2)
 def attention():
  out=af.launch(data['g'],data['tail'],data['q'],v,full=True)[0]
  if not bool(t.isfinite(out).all()):raise w.NumericalFailure('Nonfinite pipeline output')
  return out
 return k.checked_call(arm,data['g'],data['tail'],data['q'],v,attention)

def measured(arm,scope,data,n):
 t=w.torch;t.cuda.synchronize();a=t.cuda.Event(enable_timing=True);b=t.cuda.Event(enable_timing=True)
 before=t.cuda.memory_allocated();t.cuda.reset_peak_memory_stats();started=time.monotonic();wall=time.perf_counter();a.record();out=None
 for _ in range(n):
  if out is not None:del out
  out=run_cell(arm,scope,data)
 b.record();b.synchronize()
 return out,dict(calls=n,cuda_ms=a.elapsed_time(b)/n,wall_ms=(time.perf_counter()-wall)*1000/n,allocated_before=before,allocated_peak=t.cuda.max_memory_allocated(),started_monotonic=started,finished_monotonic=time.monotonic())
def output(x):return c.np.asarray(True,c.np.bool_) if x is True else array(x)
def cache_snapshot():
 root=Path(os.environ['TRITON_CACHE_DIR']);return {str(p.relative_to(root)):c.file_sha(p) for p in sorted(root.rglob('*')) if p.is_file() and not p.name.endswith('.lock') and '.tmp' not in p.name}
def timing_block():
 loaded={};gold={};first={}
 for sp in c.index()['samples']:
  a=c.d.check_input(c.load(ROOT/sp['file']),sp);loaded[sp['id']]=source_data(a);gold[sp['id']]=a['actual_output']
 before={n:fingerprints(d) for n,d in loaded.items()};c.write(ROOT/'inputs_before.json',before)
 for cell in c.cells():
  start_case('first:'+str(cell));x,tm=measured(cell['arm'],cell['scope'],loaded[cell['sample']],1);val=output(x)
  want=c.np.asarray(True,c.np.bool_) if cell['scope']=='CHECK_ONLY' else gold[cell['sample']]
  c.require(val.shape==want.shape and val.dtype==want.dtype and c.np.array_equal(val,want),'Original output mismatch')
  name=cell['sample']+'-'+cell['scope']+'-'+cell['arm'];p=ROOT/'first_outputs'/(name+'.npy');p.parent.mkdir(exist_ok=True);c.np.save(p,val,allow_pickle=False);h=c.fingerprint(val);first[(cell['sample'],cell['scope'],cell['arm'])]=h
  c.append(ROOT/'first_calls.jsonl',dict(cell,**tm,file=str(p.relative_to(ROOT)),sha256=c.file_sha(p),array_hash=h));del x;end_case()
 for cell in c.cells():
  start_case('warm:'+str(cell));x,tm=measured(cell['arm'],cell['scope'],loaded[cell['sample']],c.WARMUP);h=c.fingerprint(output(x));c.require(h==first[(cell['sample'],cell['scope'],cell['arm'])],'Warmup output changed')
  c.append(ROOT/'warmup.jsonl',dict(cell,**tm,array_hash=h));del x;end_case()
 c.write(ROOT/'timing_ready.json',dict(cells=108,cache=cache_snapshot(),completed_monotonic=time.monotonic()))
 print('All 108 combinations warmed; measuring 324 intervals',flush=True)
 for row in c.schedule():
  start_case('timing:'+str(row));x,tm=measured(row['arm'],row['scope'],loaded[row['sample']],c.REPEATS);h=c.fingerprint(output(x));c.require(h==first[(row['sample'],row['scope'],row['arm'])],'Repeated output changed')
  c.append(ROOT/'timings.jsonl',dict(row,**tm,array_hash=h));del x;end_case()
 c.write(ROOT/'timing_cache_after.json',cache_snapshot());after={n:fingerprints(d) for n,d in loaded.items()};c.write(ROOT/'inputs_after.json',after);c.require(before==after,'Real input mutation')

def load_parent(parent):
 idx=c.index();c.require(c.file_sha(parent)==idx['parent_sha256'],'Parent ZIP mismatch')
 with zipfile.ZipFile(parent) as z:
  names=z.namelist();c.require(len(names)==len(set(names)),'Duplicate ZIP entries')
  for sp in idx['samples']:
   n=sp['file'];c.require(n.startswith('paired_inputs/') and '..' not in Path(n).parts,'Input path');data=z.read(n);c.require(c.sha(data)==sp['sha256'],'Parent array hash')
   p=ROOT/n;p.parent.mkdir(exist_ok=True);p.write_bytes(data);c.d.check_input(c.load(p),sp)
 c.write(ROOT/'parent.json',dict(zip=idx['parent_zip'],sha256=idx['parent_sha256'],samples=18))

def main(root,parent):
 global ROOT
 ROOT=Path(root);started=time.monotonic();status='INCONCLUSIVE'
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('IV01 case cap')))
 signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('IV01 launcher stop')))
 try:
  phase('PREPARE',600)
  cap=c.read(ROOT/'capsule.json')
  for n,h in cap.items():c.require(c.file_sha(ROOT/'source'/n)==h,'Source mismatch '+n)
  for n,h in c.index()['dependencies'].items():c.require(c.file_sha(Path(__file__).with_name(n))==h,'Frozen dependency mismatch')
  load_parent(Path(parent))
  import torch,importlib.metadata
  w.torch=torch;w.np=c.np
  c.require(torch.__version__=='2.11.0+cu128' and c.np.__version__=='2.1.3','Pinned torch/numpy required')
  c.require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(),'BF16 GPU required')
  torch.set_grad_enabled(False);torch.use_deterministic_algorithms(True);torch.manual_seed(2026100106);torch.set_float32_matmul_precision('highest')
  torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
  torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=True
  env=dict(torch=torch.__version__,numpy=c.np.__version__,triton=importlib.metadata.version('triton'),gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda,restore=rf.environment(),matmul_tf32=False,cudnn_tf32=False,bf16_reduced=True)
  c.require(env['triton']=='3.6.0','Pinned Triton required');c.write(ROOT/'environment.json',env);c.write(ROOT/'checker_config.json',k.config());c.write(ROOT/'fused_config.json',af.config())
  for fn,kind in [(k.qk.validate,'qk'),(af.validate_v,'v')]:
   _,src=k.structural_function(fn,kind);(ROOT/('structural_'+kind+'.py.txt')).write_text(src)
  k.prepare_shapes();phase('EXPERIMENT',600);fixture_block();timing_block();phase('VERIFY',120);signal.alarm(120)
  from verify_loop_kv_iv01 import verify
  answer=verify(ROOT,check_summary=False);c.write(ROOT/'summary.json',answer);status='COMPLETE_COMPARISON';signal.alarm(0)
 except BaseException as exc:
  signal.alarm(0);c.append(ROOT/'failures.jsonl',dict(error_type=type(exc).__name__,error=w.redact(str(exc)),traceback=w.redact(traceback.format_exc())));print(w.redact(traceback.format_exc()),flush=True)
 finally:
  signal.alarm(0)
  if status!='COMPLETE_COMPARISON':c.write(ROOT/'summary.json',dict(status=status,experiment='LOOP-KV-IV01',note='Partial evidence retained'))
  c.write(ROOT/'status.json',dict(status=status,experiment='LOOP-KV-IV01',elapsed_seconds=time.monotonic()-started));print('Closed IV01: '+status,flush=True)
if __name__=='__main__':main(sys.argv[1],sys.argv[2])
