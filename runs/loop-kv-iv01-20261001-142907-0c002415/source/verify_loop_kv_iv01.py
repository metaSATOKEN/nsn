"""NumPy-only independent IV01 inventory, fixture, output and timing audit."""
from pathlib import Path
import sys,math,statistics
import numpy as np
import loop_kv_iv01_contract as c

def evidence(root,row):
 p=Path(row['file']);c.require(not p.is_absolute() and '..' not in p.parts,'Evidence path');p=root/p;c.require(c.file_sha(p)==row['sha256'],'Evidence hash');return p
def inventory(rows,plans,label):
 c.require(len(rows)==len(plans),'Missing/extra '+label)
 for row,sp in zip(rows,plans):
  c.require(all(type(row.get(k))==type(v) and row[k]==v for k,v in sp.items()),'Schedule mismatch '+label)
def timing(row,n):
 c.require(row['calls']==n,'Call count')
 for k in ('cuda_ms','wall_ms','started_monotonic','finished_monotonic'):
  c.require(isinstance(row[k],(int,float)) and math.isfinite(row[k]) and row[k]>=0,'Nonfinite/negative time')
 c.require(row['cuda_ms']>0 and row['wall_ms']>0 and row['finished_monotonic']>=row['started_monotonic'],'Time order')
 c.require(type(row['allocated_peak'])==type(row['allocated_before'])==int and row['allocated_peak']>=row['allocated_before']>=0,'Memory counters')
def descriptor_arrays(desc,arrays):
 c.require(set(desc['tensors'])==set(arrays),'Descriptor inventory')
 for n,m in desc['tensors'].items():
  a=arrays[n];dt={'torch.bfloat16':'uint16','torch.float16':'float16','torch.float32':'float32','torch.int16':'int16','torch.uint8':'uint8'}[m['dtype']]
  c.require(list(a.shape)==m['shape'] and str(a.dtype)==dt,'Descriptor dtype/shape')
def verify(root,check_summary=True):
 root=Path(root);idx=c.index();caps=c.read(Path(__file__).with_name('iv01_capsule.json'));c.equal(c.read(root/'capsule.json'),caps,'Capsule')
 for n,h in caps.items():c.require(c.file_sha(root/'source'/n)==h,'Source '+n)
 c.equal(c.read(root/'parent.json'),dict(zip=idx['parent_zip'],sha256=idx['parent_sha256'],samples=18),'Parent')
 env=c.read(root/'environment.json');c.require(env['torch']=='2.11.0+cu128' and env['numpy']=='2.1.3' and env['triton']=='3.6.0','Pinned versions')
 c.require(env['matmul_tf32'] is False and env['cudnn_tf32'] is False and env['bf16_reduced'] is True,'Precision')
 import loop_kv_iv01_kernel as k
 c.equal(c.read(root/'checker_config.json'),k.config(),'Checker config');c.equal(c.read(root/'fused_config.json'),k.fused.config(),'Frozen fusion')
 for fn,kind in [(k.qk.validate,'qk'),(k.fused.validate_v,'v')]:
  _,src=k.structural_function(fn,kind);c.require((root/('structural_'+kind+'.py.txt')).read_text()==src,'Structural extraction')
 gold={};raws={}
 for sp in idx['samples']:
  a=c.d.check_input(c.load(evidence(root,sp)),sp);gold[sp['id']]=a['actual_output'];raws[sp['id']]=a
 ff=c.rows(root/'fixtures.jsonl');inventory(ff,c.fixtures(),'Fixtures')
 for row in ff:
  a=c.load(evidence(root,row));desc=row['descriptor'];descriptor_arrays(desc,a);expected,kind=c.cpu_accept(desc,a)
  c.require(expected==row['accept'] and kind==row['kind'] and row['input_unchanged'] is True,'CPU fixture oracle')
  if kind=='normal':
   wanted=c.fixture_arrays(row['case']);c.require(set(wanted)==set(a) and all(np.array_equal(a[n],x) for n,x in wanted.items()),'Normal fixture values')
  if kind=='nonfinite':
   for field in c.FIELDS:
    x=c.bf32(a[field]) if a[field].dtype==np.uint16 else a[field];bad=np.flatnonzero(~np.isfinite(x.flatten()))
    if field==row['field']:
     pos=0 if row['position']=='first' else x.size-1;c.require(bad.tolist()==[pos],'Nonfinite position')
     value=x.flatten()[pos];c.require(np.isnan(value) if row['value']=='nan' else value==float('inf') if row['value']=='posinf' else value==-float('inf'),'Nonfinite kind')
    else:c.require(len(bad)==0,'Unexpected nonfinite field')
  inventory(row['results'],[dict(arm=x) for x in c.ARMS],'Fixture arms')
  for rr in row['results']:
   c.require(rr['accepted'] is expected and rr['continued']==int(expected),'Fixture acceptance/continuation')
   c.require(rr['checker_launches']==(1 if rr['arm']=='TRITON_GROUPED' and kind!='structural' else 0),'Checker launch gate')
   c.require(rr['error_type']==(None if expected else 'NumericalFailure' if kind=='nonfinite' else 'IntegrityError'),'Error category')
 before=c.read(root/'inputs_before.json');after=c.read(root/'inputs_after.json');c.equal(after,before,'Input mutation');c.require(set(before)==set(gold),'Input inventory')
 for sid,snap in before.items():
  a=raws[sid];mapped={'q':a['q'],'tail':a['k_tail'],'v':a['v_full'],'codes':a['k_codes'],'v_codes':a['v_codes'],'v_tail':a['v_tail'],**{'meta'+str(i):a['k_meta'+str(i)] for i in range(3)},**{'v_meta'+str(i):a['v_meta'+str(i)] for i in range(3)}}
  c.equal(snap['arrays'],{n:c.fingerprint(x) for n,x in mapped.items()},'GPU input transfer');descriptor_arrays(snap['descriptor'],mapped);c.require(c.cpu_accept(snap['descriptor'],mapped)==(True,'normal'),'Real input oracle')
 first=c.rows(root/'first_calls.jsonl');warm=c.rows(root/'warmup.jsonl');tt=c.rows(root/'timings.jsonl');inventory(first,c.cells(),'First');inventory(warm,c.cells(),'Warmup');inventory(tt,c.schedule(),'Timing')
 hashes={};last=0
 for rr in first:
  timing(rr,1);c.require(rr['started_monotonic']>=last,'First order');last=rr['finished_monotonic']
  got=np.load(evidence(root,rr),allow_pickle=False);wanted=np.asarray(True,np.bool_) if rr['scope']=='CHECK_ONLY' else gold[rr['sample']]
  c.require(got.dtype==wanted.dtype and got.shape==wanted.shape and np.array_equal(got,wanted),'Forward/output gate');h=c.fingerprint(got);c.require(rr['array_hash']==h,'Array hash');hashes[(rr['sample'],rr['scope'],rr['arm'])]=h
 for rr in warm:
  timing(rr,c.WARMUP);c.require(rr['started_monotonic']>=last and rr['array_hash']==hashes[(rr['sample'],rr['scope'],rr['arm'])],'Warmup order/output');last=rr['finished_monotonic']
 ready=c.read(root/'timing_ready.json');c.require(ready['cells']==108 and ready['completed_monotonic']>=last,'All warmup completed');last=ready['completed_monotonic']
 for rr in tt:
  timing(rr,c.REPEATS);c.require(rr['started_monotonic']>=last and rr['array_hash']==hashes[(rr['sample'],rr['scope'],rr['arm'])],'Timing order/output');last=rr['finished_monotonic']
 per=[];contrasts=[]
 for sp in idx['samples']:
  for scope in c.SCOPES:
   values={arm:{field:statistics.median(x[field] for x in tt if x['sample']==sp['id'] and x['scope']==scope and x['arm']==arm) for field in ('cuda_ms','wall_ms','allocated_peak')} for arm in c.ARMS}
   per.append(dict(sample=sp['id'],scope=scope,values=values))
 for scope in c.SCOPES:
  for cand,ref in [('TRITON_GROUPED','LEGACY'),('TORCH_GROUPED','LEGACY'),('TRITON_GROUPED','TORCH_GROUPED')]:
   for field in ('cuda_ms','wall_ms'):
    ratios=[x['values'][cand][field]/x['values'][ref][field] for x in per if x['scope']==scope]
    contrasts.append(dict(scope=scope,candidate=cand,reference=ref,field=field,ratio_median=statistics.median(ratios),ratio_min=min(ratios),ratio_max=max(ratios),shorter_samples=sum(x<1 for x in ratios),samples=18))
 stable=ready['cache']==c.read(root/'timing_cache_after.json');c.require(not c.rows(root/'failures.jsonl'),'Failures present')
 result=dict(status='COMPLETE_COMPARISON',experiment='LOOP-KV-IV01',fixtures=64,fixture_decisions=192,samples=18,first_cells=108,timing_intervals=324,timing_calls=6480,timing_status='TIMING_CACHE_UNCHANGED' if stable else 'TIMING_CACHE_CHANGED',per_sample=per,comparisons=contrasts,note='Same finite-input contract; checks complete before attention. Components only; no model/PPL run or CI/p-value.')
 if check_summary:c.equal(c.read(root/'summary.json'),result,'Summary');c.require(c.read(root/'status.json')['status']=='COMPLETE_COMPARISON','Run status')
 return result
if __name__=='__main__':
 import traceback
 root=Path(sys.argv[1])
 try:
  result=verify(root);c.write(root/'completion_verification.json',dict(status='VERIFIED',fixtures=64,timing_intervals=324));print('IV01 verified')
 except Exception as exc:c.write(root/'completion_verification.json',dict(status='FAILED',error=str(exc)));print(traceback.format_exc());sys.exit(1)
