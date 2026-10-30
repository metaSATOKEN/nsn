import ast,hashlib,json,sys
from pathlib import Path
import torch
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--research-root',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--listing',type=Path)
args=parser.parse_args()
R=args.research_root.resolve();S=R
source=R/'scripts/loop_kv_nb05_primitives.py'
# Reuse frozen CPU-compatible arithmetic only; no model, kernel or network loading.
tree=ast.parse(source.read_text());names={'nsn_encode','nsn_decode','nsn_qdq'}
funcs=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
assert len(funcs)==3
class NumericalFailure(RuntimeError):pass
def require(ok,msg):
 if not ok:raise ValueError(msg)
ns={'torch':torch,'BLOCK':64,'DIM':128,'EPS':1e-12,'require':require,'NumericalFailure':NumericalFailure}
exec(compile(ast.Module(body=funcs,type_ignores=[]),str(source),'exec'),ns)
code=args.listing or S/'papers/loop-kv/code/nsn_int3_core.py';short={};exec(compile(code.read_text(),str(code),'exec'),short)
func=next(x for x in ast.parse(code.read_text()).body if isinstance(x,ast.FunctionDef))
assert len(func.body)==10 and func.end_lineno-func.lineno==10
G=torch.Generator(device='cpu').manual_seed(2026100303)
x=torch.randn(2,64,128,generator=G)
strided=torch.randn(2,64,256,generator=G)[...,::2];assert not strided.is_contiguous()
zero_rows=x.clone();zero_rows[:,::4,:]=0
fixtures={'normal':x,'zero':torch.zeros_like(x),'constant_across_tokens':torch.ones_like(x),'zero_rows':zero_rows,'norm_dispersion':x*torch.logspace(-3,3,64).reshape(1,64,1),'strided_input':strided,'shifted':x+5,'multi_head':torch.randn(2,3,64,128,generator=G)}
rows=[]
for name,x in fixtures.items():
 for dtype in [torch.float32,torch.float16,torch.bfloat16]:
  a=x.to(dtype);ref=ns['nsn_qdq'](a,3);got=short['nsn_int3_qdq'](a)
  equal=torch.equal(ref.contiguous().view(torch.uint8),got.contiguous().view(torch.uint8))
  assert equal,(name,dtype)
  rows.append({'fixture':name,'dtype':str(dtype),'shape':list(a.shape),'bitwise_equal':equal,'finite':bool(torch.isfinite(got).all())})
A=args.output.resolve().parent;A.mkdir(parents=True,exist_ok=True)
r={'status':'PASS','scope':'Synthetic CPU arithmetic checks only; not a new model experiment or fused-kernel equivalence claim','torch_version':torch.__version__,'body_lines':10,'frozen_source':str(source.relative_to(R)),'frozen_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'listing_sha256':hashlib.sha256(code.read_bytes()).hexdigest(),'cases':rows,'total':len(rows),'all_bitwise_equal':True,'new_model_calls':0,'gpu_calls':0}
args.output.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in r.items() if k!='cases'},ensure_ascii=False,indent=2))
