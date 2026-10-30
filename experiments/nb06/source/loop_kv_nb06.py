"""NB06 worker. Heavy imports/load occur only in main, on the Colab GPU."""
import gc
import json
import math
import os
import platform
from pathlib import Path
import random
import shutil
import signal
import sys
import time
import traceback
import loop_kv_nb06_contract as c
from loop_kv_nb06_measure import measure
import loop_kv_nb06_materials as materials
from loop_kv_nb06_runtime import clean, preparation_started

def set_alarm(deadline):
    remaining=deadline-time.monotonic()
    if remaining<=0:raise TimeoutError('NB06 phase/series deadline')
    signal.alarm(max(1,math.ceil(remaining)))

def validate_hub_file(path, entry):
    path=Path(path)
    c.require(path.stat().st_size==entry['size'], 'Hub file size: '+entry['rfilename'])
    if 'lfs' in entry:
        digest=c.file_digest(path);c.require(digest==entry['lfs']['sha256'], 'LFS hash mismatch')
    else:
        c.require(c.file_digest(path,'sha1',git_blob=True)==entry['blobId'], 'Git blob mismatch')
        digest=c.file_digest(path)
    expected=c.PLAN['public_source_sha256'].get(entry['rfilename'])
    if expected:c.require(expected==digest,'Design/source hash mismatch')
    return {'file':entry['rfilename'],'bytes':path.stat().st_size,'sha256':digest}

def prepare_model(root, deadline):
    # Environment checks precede torch import and prohibit accidental local model loading.
    c.require(sys.platform=='linux' and platform.machine()=='x86_64' and sys.version_info[:2]==(3,13), 'Colab CPython 3.13/Linux x86_64 only')
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    c.require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), 'CUDA/BF16 unavailable')
    root=Path(root);c.check_embedded(root)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    metadata=c.read(root/'nb06_model_files.json')
    c.require(metadata['revision']==c.MODEL['revision'] and metadata['repo']==c.MODEL['id'], 'Download revision')
    paths={};files=[]
    common=dict(repo_id=c.MODEL['id'],revision=c.MODEL['revision'],token=os.environ.get('HF_TOKEN'),
                cache_dir=str(root.parent/'hf_cache'),max_workers=1)
    small=[name for name in metadata['files'] if name!='model.safetensors']
    folder=Path(snapshot_download(**common,allow_patterns=small))
    for name in small:
        paths[name]=folder/name;files.append(validate_hub_file(paths[name],metadata['files'][name]))
    config=c.read(paths['config.json'])
    for key,want in {'num_hidden_layers':22,'num_loops':2,'num_attention_heads':48,'num_key_value_heads':8,'head_dim':128,'vocab_size':166144}.items():
        c.require(config[key]==want,'Configuration changed: '+key)
    tokenizer=AutoTokenizer.from_pretrained(str(folder),use_fast=False,trust_remote_code=True,local_files_only=True,legacy=True)
    selected=materials.select_materials(root,tokenizer,deadline)
    frozen=c.read(root/'selection_lock.json')['frozen_monotonic']
    del tokenizer
    c.require(shutil.disk_usage(root).free>=c.PLAN['minimum_before_weights_free_bytes'],'Insufficient space before model weights')
    set_alarm(deadline);weights_started=time.monotonic()
    folder=Path(snapshot_download(**common,allow_patterns=['model.safetensors']))
    files.append(validate_hub_file(folder/'model.safetensors',metadata['files']['model.safetensors']))
    c.write_json(root,'hub_files.json',files)
    c.write_json(root,'model_load_started.json',{'monotonic':time.monotonic(),'input_lock_sha256':c.file_digest(root/'selection_lock.json')})
    model=AutoModelForCausalLM.from_pretrained(str(folder),trust_remote_code=True,local_files_only=True,
        torch_dtype=torch.bfloat16,device_map={'':0},low_cpu_mem_usage=True,use_safetensors=True,attn_implementation='sdpa')
    model.eval();model.requires_grad_(False)
    validate_loaded(model)
    # Check the imported copy too, after dynamic module loading.
    import inspect
    source_module=inspect.getmodule(model.__class__)
    config_module=inspect.getmodule(model.config.__class__)
    for module,name in [(source_module,'modeling_nanbeige.py'),(config_module,'configuration_nanbeige.py')]:
        c.require(c.file_digest(module.__file__)==c.PLAN['public_source_sha256'][name],'Imported model source differs')
    c.write_json(root,'loaded_config.json',json.loads(model.config.to_json_string(use_diff=False)))
    c.write_json(root,'model_ready.json',{'revision':c.MODEL['revision'],'loaded_contract_ok':True,'model_calls_before_measurement':0,
        'input_lock_sha256':c.file_digest(root/'selection_lock.json'),'input_frozen_monotonic':frozen,'weights_started_monotonic':weights_started,
        'loaded_monotonic':time.monotonic(),'gpu':torch.cuda.get_device_name(0),'parameter_count':sum(p.numel() for p in model.parameters()),
        'deterministic_algorithms':True,'tf32':False,'attention':'NanbeigeSdpaAttention','kv_compression_planned':True,'weight_quantization':False,'fusion':True})
    return model,torch,np,selected

def validate_loaded(model):
    conf=model.config
    for key,want in {'num_hidden_layers':22,'num_loops':2,'num_attention_heads':48,'num_key_value_heads':8,'head_dim':128,'vocab_size':166144}.items():
        c.require(getattr(conf,key)==want,'Loaded config: '+key)
    for key in ['enable_double_loop_split','loop_share_kv','enable_depth_attention','enable_hyper_connection']:
        c.require(not getattr(conf,key,False),'Unsupported enabled feature: '+key)
    c.require(model.model.ngram_embeddings is None and len(model.model.layers)==22 and model.model._get_num_loops()==2,'Model layer/loop/embedding contract')
    for layer in model.model.layers:
        attn=layer.self_attn
        c.require(type(attn).__name__=='NanbeigeSdpaAttention','Attention class/fallback changed')
        c.require((attn.num_heads,attn.num_key_value_heads,attn.head_dim)==(48,8,128),'Attention dimensions')
        c.require(tuple(attn.q_proj.weight.shape)==(6144,3072) and tuple(attn.k_proj.weight.shape)==tuple(attn.v_proj.weight.shape)==(1024,3072),'Projection shapes')
    c.require(not model.training and all(str(p.dtype)=='torch.bfloat16' and p.device.type=='cuda' and p.device.index==0 for p in model.parameters()),'Model precision/device')

def measure_books(root,model,torch,np,selected,deadline,progress,alarm,base=None):
    for rank,doc in enumerate(selected['diagnostic']):
        book=root/'books'/f'b{rank}';book.mkdir(parents=True,exist_ok=False)
        end=min(deadline,time.monotonic()+c.CAPS['book']);alarm(end)
        c.write_json(book,'book.json',dict(selected_book_rank=rank,book_id=doc['book_id'],token_sha256=doc['token_sha256'],
            selection_lock_sha256=c.file_digest(root/'selection_lock.json'),started_monotonic=time.monotonic(),deadline_monotonic=end))
        print(f"NB06 book {rank+1}/4: {doc['book_id']}; 3855 forwards.",flush=True)
        local={'attempted':0,'finished':0}
        c.write_json(book,'progress.json',local,closing=True)
        try:
            args={} if base is None else {'base':base}
            measure(book,model,torch,np,doc['tokens'],end,local,alarm,**args)
        finally:
            progress.update(attempted=rank*c.PLAN['forward_calls_per_book']+local['attempted'],finished=rank*c.PLAN['forward_calls_per_book']+local['finished'],
                selected_book_rank=rank,book_id=doc['book_id'])
        c.write_json(book,'book_complete.json',dict(selected_book_rank=rank,book_id=doc['book_id'],finished_monotonic=time.monotonic(),**local))

def main(root):
    root=Path(root);progress={'attempted':0,'finished':0};started=preparation_started()
    def timeout(*_):raise TimeoutError('NB06 time cap or termination')
    signal.signal(signal.SIGALRM,timeout);signal.signal(signal.SIGTERM,timeout)
    try:
        prepare_deadline=started+c.CAPS['prepare'];set_alarm(prepare_deadline)
        model,torch,np,selected=prepare_model(root,prepare_deadline)
        deadline=min(started+c.CAPS['outer']-c.CAPS['verify']-c.CAPS['zip'],time.monotonic()+c.CAPS['measure'])
        c.write_json(root,'phase.json',{'stage':'MEASURE','deadline_monotonic':deadline})
        measure_books(root,model,torch,np,selected,deadline,progress,set_alarm)
        del model;gc.collect();torch.cuda.empty_cache()
        verify_deadline=min(started+c.CAPS['outer']-c.CAPS['zip'],time.monotonic()+c.CAPS['verify'])
        set_alarm(verify_deadline);c.write_json(root,'phase.json',{'stage':'VERIFY','deadline_monotonic':verify_deadline})
        from verify_loop_kv_nb06 import verify
        result=verify(root,check_summary=False);c.write_json(root,'summary.json',result,closing=True)
        c.write_json(root,'run_status.json',{'status':result['status'],'measurement_complete':True,**progress},closing=True)
        print('NB06 connection:',result['status'],flush=True)
        return 0
    except BaseException as exc:
        signal.alarm(0)
        c.write_json(root,'worker_error.json',{'error_type':type(exc).__name__,'error':clean(exc),'traceback':clean(traceback.format_exc()),**progress},closing=True)
        result={'experiment':'LOOP-KV-NB06','status':'INCONCLUSIVE','measurement_complete':False,**progress}
        c.write_json(root,'run_status.json',result,closing=True);c.write_json(root,'summary.json',result,closing=True)
        print(clean(traceback.format_exc()),flush=True);return 1
    finally:signal.alarm(0)

if __name__=='__main__':sys.exit(main(sys.argv[1]))
