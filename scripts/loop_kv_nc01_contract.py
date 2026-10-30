"""NC01 plan and CPU-only evidence contract; frozen VF01 numerical criteria."""
from pathlib import Path
import loop_kv_vf01_contract as base
from verify_loop_kv_vf01 import numerics,eq,file,plans
np=base.np;require=base.require;read=base.read;write=base.write;rows=base.rows;append=base.append;load=base.load;file_sha=base.file_sha;arrays_hash=base.arrays_hash
PLAN=read(Path(__file__).with_name('loop_kv_nc01_inputs.json'));CAPS=PLAN['caps_seconds'];ARMS=tuple(PLAN['arms']);SPECS=PLAN['fixtures'];ATTEMPTS=PLAN['attempts']
SETTINGS=dict(flash=True,efficient=True,math=True,cudnn=True,gemm_bf16_reduced=True,math_sdp_reduced=False,matmul_tf32=False,cudnn_tf32=False,float32_matmul_precision='highest')

def contiguous(shape,strides):
    expected=1
    for size,stride in zip(shape[::-1],strides[::-1]):
        if size!=1 and stride!=expected:return False
        expected*=size
    return True

def check_layout(layout,shape,noncontiguous=False):
    require(layout['shape']==list(shape) and len(layout['stride'])==len(shape),'Layout shape')
    require(all(type(x)==int and x>0 for x in layout['stride']),'Stride values')
    require(layout['dtype']=='torch.bfloat16' and layout['device'].startswith('cuda'),'GPU BF16 layout')
    require(type(layout['contiguous'])==bool and layout['contiguous']==contiguous(shape,layout['stride']),'Inconsistent contiguity metadata')
    require(layout['storage_offset']==0,'Storage offset')
    if noncontiguous:require(not layout['contiguous'] and layout['stride'][-1]==2,'Expected noncontiguous stride-2 view')
    return True

def construction_checks(original,view,backing,original_layout,view_layout,backing_layout,shared):
    # Logical values are compared separately from the physical layout.
    return dict(shape=tuple(view.shape)==tuple(original.shape),dtype=view.dtype==original.dtype==np.uint16,
        device=view_layout['device']==original_layout['device']==backing_layout['device'],
        values=eq(original,view),noncontiguous=not view_layout['contiguous'],last_stride=view_layout['stride'][-1]==2,
        backing_shape=tuple(backing.shape)==(*original.shape[:-1],original.shape[-1]*2),
        backing_view=eq(backing[...,::2].copy(),view),odd_padding_zero=bool(np.count_nonzero(backing[...,1::2])==0),shared_storage=bool(shared))

def numeric_ok(report):
    return report['complete'] and all(report['metrics'][arm][kind]['pass_budget'] for arm in ARMS for kind in ('v_restore','score','context','combined_context'))
