"""AC04 validation preflight and independent saved-fixture audit.
Cold import is standard-library only. GPU fixture generation is explicit.
"""
from pathlib import Path
import time

CHECKER_CONFIG=dict(block=1024,num_warps=4,autotune=False,finite_fields=['q','tail','meta0','meta1','meta2','v'],host_decisions=1,kernels=1,flag_reduction='torch_any',enable_fp_fusion=False)

def numeric_imports():
 global c,np
 import numpy as np
 import loop_kv_iv01_contract as c

def evidence(root,row):
 p=Path(row['file']);c.require(not p.is_absolute() and '..' not in p.parts,'Evidence path');p=root/p;c.require(c.file_sha(p)==row['sha256'],'Evidence hash');return p
def inventory(rows,plans,label):
 c.require(len(rows)==len(plans),'Missing/extra '+label)
 for row,sp in zip(rows,plans):
  c.require(all(type(row.get(k))==type(v) and row[k]==v for k,v in sp.items()),'Schedule mismatch '+label)
def descriptor_arrays(desc,arrays):
 c.require(set(desc['tensors'])==set(arrays),'Descriptor inventory')
 for n,m in desc['tensors'].items():
  a=arrays[n];dt={'torch.bfloat16':'uint16','torch.float16':'float16','torch.float32':'float32','torch.int16':'int16','torch.uint8':'uint8'}[m['dtype']]
  c.require(list(a.shape)==m['shape'] and str(a.dtype)==dt,'Descriptor dtype/shape')
def audit(root):
 numeric_imports();root=Path(root)
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
 return dict(status='PASS',fixtures=len(ff),decisions=sum(len(r['results']) for r in ff),normal=sum(r['kind']=='normal' for r in ff),nonfinite=sum(r['kind']=='nonfinite' for r in ff),structural=sum(r['kind']=='structural' for r in ff))

def fixture_block(root,engine):
 import loop_kv_iv01 as iv
 root=Path(root);dest=root/'validation';dest.mkdir(exist_ok=False)
 old=(iv.ROOT,iv.start_case,iv.end_case,engine.condition_deadline)
 deadline=time.monotonic()+120
 def start(name):
  engine.condition_deadline=deadline;engine.check_time()
  iv.c.write(root/'active_case.json',dict(id='validation:'+name,status='STARTED',deadline_monotonic=deadline))
 def end():
  engine.check_time();iv.c.write(root/'active_case.json',dict(status='FINISHED'))
 try:
  iv.ROOT=dest;iv.start_case=start;iv.end_case=end
  iv.k.prepare_shapes();iv.fixture_block()
  result=audit(dest);iv.c.write(root/'validation_checks.json',result)
  return result
 finally:
  iv.ROOT,iv.start_case,iv.end_case,engine.condition_deadline=old
