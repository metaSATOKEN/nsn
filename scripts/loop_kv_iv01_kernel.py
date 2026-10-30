"""IV01 finite-input checker. No CUDA/Triton activity on import."""
import ast,inspect,textwrap
import loop_kv_qk01_kernel as qk
import loop_kv_af01_kernel as fused
w=qk.w;BLOCK=1024;WARPS=4;_kernel=None;_qshape=None;_vshape=None;launch_count=0

def structural_function(fn,kind):
 tree=ast.parse(textwrap.dedent(inspect.getsource(fn)));node=tree.body[0]
 at=-2 if kind=='qk' else -1;gate=node.body[at]
 if not (isinstance(gate,ast.If) and not gate.orelse and len(gate.body)==1 and isinstance(gate.body[0],ast.Raise) and 'isfinite' in ast.unparse(gate)):
  raise ValueError('Frozen finite gate changed')
 if kind=='qk' and not isinstance(node.body[-1],ast.Return):raise ValueError('Return contract changed')
 del node.body[at];node.name='_iv01_shape_'+kind;ast.fix_missing_locations(tree)
 exec(compile(tree,'<IV01 structural '+kind+'>','exec'),fn.__globals__)
 return fn.__globals__[node.name],ast.unparse(tree)+'\n'

def prepare_shapes():
 global _qshape,_vshape
 if _qshape is None:_qshape,_=structural_function(qk.validate,'qk');_vshape,_=structural_function(fused.validate_v,'v')
def structural(g,tail,q,v):
 prepare_shapes();h,c,total=_qshape(g,tail,q);_vshape(v,q,total);return total

def initialize():
 global _kernel,tl
 if _kernel is not None:return
 import triton
 import triton.language as tl
 @triton.jit
 def finite_kernel(Q,T,N,M,S,V,Flags,NQ:tl.constexpr,NT:tl.constexpr,NN:tl.constexpr,NM:tl.constexpr,NS:tl.constexpr,NV:tl.constexpr,BT:tl.constexpr):
  pid=tl.program_id(0);r=tl.arange(0,BT)
  # Pure constexpr integer arithmetic: tl.cdiv is a JIT tensor operation in Triton 3.6.
  BQ:tl.constexpr=(NQ+BT-1)//BT;BTL:tl.constexpr=(NT+BT-1)//BT;BN:tl.constexpr=(NN+BT-1)//BT;BM:tl.constexpr=(NM+BT-1)//BT;BS:tl.constexpr=(NS+BT-1)//BT
  if pid<BQ:
   i=pid*BT+r;x=tl.load(Q+i,mask=i<NQ,other=0).to(tl.float32)
  elif pid<BQ+BTL:
   i=(pid-BQ)*BT+r;x=tl.load(T+i,mask=i<NT,other=0).to(tl.float32)
  elif pid<BQ+BTL+BN:
   i=(pid-BQ-BTL)*BT+r;x=tl.load(N+i,mask=i<NN,other=0).to(tl.float32)
  elif pid<BQ+BTL+BN+BM:
   i=(pid-BQ-BTL-BN)*BT+r;x=tl.load(M+i,mask=i<NM,other=0).to(tl.float32)
  elif pid<BQ+BTL+BN+BM+BS:
   i=(pid-BQ-BTL-BN-BM)*BT+r;x=tl.load(S+i,mask=i<NS,other=0).to(tl.float32)
  else:
   i=(pid-BQ-BTL-BN-BM-BS)*BT+r;x=tl.load(V+i,mask=i<NV,other=0).to(tl.float32)
  bad=tl.sum(((x!=x)|(tl.abs(x)==float('inf'))).to(tl.int32),axis=0)>0
  tl.store(Flags+pid,bad.to(tl.int32))
 _kernel=finite_kernel

def triton_finite(g,tail,q,v):
 global launch_count
 t=w.torch;initialize();metas=list(g['metas']) if g is not None else [q,q,q]
 sizes=[q.numel(),tail.numel()]+([x.numel() for x in metas] if g is not None else [0,0,0])+[v.numel()]
 blocks=sum((n+BLOCK-1)//BLOCK for n in sizes);flags=t.empty((blocks,),dtype=t.int32,device=q.device)
 launch_count+=1
 _kernel[(blocks,)](q,tail,*metas,v,flags,*sizes,BT=BLOCK,num_warps=WARPS,enable_fp_fusion=False)
 if bool((flags!=0).any()):raise w.NumericalFailure('IV01 nonfinite input')

def validate(arm,g,tail,q,v):
 t=w.torch
 if arm=='LEGACY':qk.validate(g,tail,q);fused.validate_v(v,q,tail.shape[2]+(0 if g is None else g['codes'].shape[2]*64))
 else:
  structural(g,tail,q,v)
  if arm=='TORCH_GROUPED':
   xs=[q,tail]+([] if g is None else list(g['metas']))+[v]
   if not bool(t.stack([t.isfinite(x).all() for x in xs]).all()):raise w.NumericalFailure('IV01 nonfinite input')
  elif arm=='TRITON_GROUPED':triton_finite(g,tail,q,v)
  else:raise ValueError('Unknown validation arm')
 return True

def checked_call(arm,g,tail,q,v,next_fn):
 validate(arm,g,tail,q,v);return next_fn()

def config():return dict(block=BLOCK,num_warps=WARPS,autotune=False,finite_fields=['q','tail','meta0','meta1','meta2','v'],host_decisions=1,kernels=1,flag_reduction='torch_any',enable_fp_fusion=False)
