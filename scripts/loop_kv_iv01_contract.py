"""Fixed IV01 inventory and independent CPU acceptance oracle."""
from pathlib import Path
import json,math,hashlib
import numpy as np
import loop_kv_dc01_contract as d
require=d.require;sha=d.sha;read=d.read;write=d.write;rows=d.rows;append=d.append;equal=d.equal;load=d.load;fingerprint=d.fingerprint;bf32=d.bf32
ARMS=('LEGACY','TORCH_GROUPED','TRITON_GROUPED');SCOPES=('CHECK_ONLY','PACKED_PIPELINE')
ROUNDS=3;WARMUP=5;REPEATS=20
FIELDS=('q','tail','meta0','meta1','meta2','v')
NORMAL=('zero','constant','signed','finite_extremes','subnormal','tail_only_short','tail_only_odd','packed_odd')
STRUCT=('q_dtype','q_shape','q_noncontig','q_device','tail_dtype','tail_shape','tail_noncontig','empty_sequence','codes_dtype','codes_width','codes_device','meta_count','meta_shape','meta_dtype','meta_noncontig','meta_device','v_dtype','v_shape','v_noncontig','v_device')
def index():return read(Path(__file__).with_name('loop_kv_iv01_inputs.json'))
def cells():
 return [dict(sample=s['id'],scope=scope,arm=arm) for i,s in enumerate(index()['samples']) for scope in SCOPES for arm in ARMS[i%3:]+ARMS[:i%3]]
def schedule():
 return [dict(sample=s['id'],scope=scope,arm=arm,round=r) for r in range(ROUNDS) for i,s in enumerate(index()['samples']) for scope in SCOPES for arm in ARMS[(i+r)%3:]+ARMS[:(i+r)%3]]
def fixtures():
 return ([dict(id='normal-'+x,kind='normal',case=x,accept=True) for x in NORMAL]+
 [dict(id=f'{f}-{v}-{p}',kind='nonfinite',field=f,value=v,position=p,accept=False) for f in FIELDS for v in ('nan','posinf','neginf') for p in ('first','last')]+
 [dict(id='struct-'+x,kind='structural',case=x,accept=False) for x in STRUCT])
def bits(x):
 x=np.asarray(x,np.float32);u=x.view(np.uint32);return ((u+0x7fff+((u>>16)&1))>>16).astype(np.uint16)
def fixture_arrays(case):
 h=1;blocks=1;tail=1
 if case=='signed':h=2;blocks=2;tail=65
 if case=='tail_only_short':blocks=0;tail=1
 if case=='tail_only_odd':blocks=0;h=2;tail=129
 if case=='packed_odd':blocks=2;tail=127
 total=blocks*64+tail
 a=dict(q=bits(np.zeros((1,h,1,128))),tail=bits(np.zeros((1,h,tail,128))),v=bits(np.zeros((1,h,total,128))))
 if blocks:a.update(codes=np.zeros((1,h,blocks,3072),np.uint8),meta0=np.zeros((1,h,blocks,64,1),np.float16),meta1=np.zeros((1,h,blocks,1,128),np.float16),meta2=np.zeros((1,h,blocks,1,128),np.float16))
 for key,x in a.items():
  if key=='codes':continue
  if case=='constant':x.fill(0x3f80 if x.dtype==np.uint16 else 1)
  elif case in ('signed','packed_odd'):
   x.flat[::2]=0x3f80 if x.dtype==np.uint16 else 1;x.flat[1::2]=0xbf80 if x.dtype==np.uint16 else -1
  elif case=='finite_extremes':
   if x.dtype==np.uint16:x.flat[::2]=0x7f7f;x.flat[1::2]=0xff7f
   else:x.flat[::2]=65504;x.flat[1::2]=-65504
  elif case=='subnormal':
   if x.dtype==np.uint16:x.flat[::2]=1;x.flat[1::2]=0x8001
   else:x.flat[::2]=2**-24;x.flat[1::2]=-2**-24
 return a

def cpu_accept(desc,arrays):
 """Independent structural checks over observed metadata, then finite values."""
 try:
  m=desc['tensors'];g=desc['packed'];tail=m['tail'];q=m['q'];v=m['v']
  def layout(x,shape,dtype):return x['shape']==list(shape) and x['dtype']==dtype and x['contiguous'] is True
  ts=tail['shape'];h=ts[1]
  valid=len(ts)==4 and ts[0]==1 and ts[-1]==128 and layout(tail,ts,'torch.bfloat16') and h>0 and layout(q,(1,h,1,128),'torch.bfloat16') and tail['device']=='cuda' and q['device']==tail['device']
  comp=0;names=['q','tail']
  if g:
   co=m['codes'];b,gh,blocks,width=co['shape']
   valid=valid and b==1 and gh==h and blocks>0 and width==3072 and layout(co,(b,gh,blocks,3072),'torch.uint8') and co['device']=='cuda' and desc['meta_count']==3
   for i,shape in enumerate([(b,gh,blocks,64,1),(b,gh,blocks,1,128),(b,gh,blocks,1,128)]):
    x=m['meta'+str(i)];valid=valid and layout(x,shape,'torch.float16') and x['device']==co['device']
   comp=blocks*64;names+=['meta0','meta1','meta2']
  total=ts[2]+comp
  valid=valid and total>0 and layout(v,(1,h,total,128),'torch.bfloat16') and v['device']==q['device']
  if not valid:return False,'structural'
  for name in names+['v']:
   a=arrays[name];val=bf32(a) if m[name]['dtype']=='torch.bfloat16' else a
   if not np.isfinite(val).all():return False,'nonfinite'
  return True,'normal'
 except (KeyError,ValueError,IndexError,TypeError):return False,'structural'

def file_sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
