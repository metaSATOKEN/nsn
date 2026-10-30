"""AC08 model connection worker. No weight load or CUDA work on import."""
import os,sys,time,math,gc,json,signal,shutil,hashlib,traceback,importlib,importlib.metadata
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import loop_kv_p0_diagnostic as d
import loop_kv_pk01_cache as c
import loop_kv_rf01_kernel as restore
import loop_kv_qk01_kernel as qk
import loop_kv_ac08_contract as ac
import loop_kv_ac08_materials as m
import loop_kv_ac08_attention as link
import loop_kv_ac04_validation as validation
input_checks=link.input_checks
vchecks=link.vchecks
import loop_kv_af01_kernel as fused
from loop_kv_pd01_score import nll as score_nll
w=d.w;kernel=restore.baseline
EXPERIMENT='LOOP-KV-AC08';SOFTWARE_VERSION='0.1.0'
PREP_SECONDS=ac.CAPS["prepare"];GATE_SECONDS=ac.CAPS["gate"];WARMUP_SECONDS=ac.CAPS["warmup"];MAIN_SECONDS=ac.CAPS["main"];CASE_SECONDS=ac.CAPS["case"];MAX_BYTES=ac.PLAN["output_limit_bytes"]
def numeric_imports():
    global torch,qc,diag
    d.numeric_imports();torch=w.torch
    import loop_kv_qk01_contract as qc
    import loop_kv_ac08_diagnostics as diag
def materials(root):
    data=m.read_frozen(Path(root));ac.bind_books([x['book_id'] for x in data['diagnostic']]);return data
def bits(x):return x.detach().contiguous().view(torch.uint16).cpu().numpy().copy()
def array(x):return x.detach().contiguous().cpu().numpy().copy()
def digest(x):return w.sha(str((tuple(x.shape),str(x.dtype))).encode()+x.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
def state(cache):
    rows=[]
    for s in range(ac.SLOTS):
        row=dict(slot=s,length=cache.get_seq_length(s))
        for kv,values in [('K',cache.key_cache),('V',cache.value_cache)]:
            x=dict(tail=digest(values[s]))
            g=getattr(cache,'packed',{}).get((s,kv))
            if g:x.update(codes=digest(g['codes']),headers=digest(g['headers']),**{'meta'+str(i):digest(a) for i,a in enumerate(g['metas'])})
            row[kv]=x
        rows.append(row)
    return rows
def cleanup():gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache()
def measured(fn):
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    before=dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved())
    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    start=time.perf_counter();a.record();out=fn();b.record();b.synchronize()
    return out,dict(cuda_ms=a.elapsed_time(b),wall_seconds=time.perf_counter()-start,
        allocated_before=before['allocated'],reserved_before=before['reserved'],allocated_peak=torch.cuda.max_memory_allocated(),reserved_peak=torch.cuda.max_memory_reserved())
def phase(engine,root,name,seconds):
    engine.stage_deadline=time.monotonic()+seconds
    w.write_json(root/'phase.json',dict(stage=name,deadline_monotonic=engine.stage_deadline))
def check_output_cap(root):w.require(sum(p.stat().st_size for p in root.rglob('*') if p.is_file())<=MAX_BYTES,'Output 6 GiB cap')
def verify_sources(root):
    expected=json.loads((root/'embedded_sha256.json').read_text())
    for name,h in expected.items():w.require(w.sha((root/name).read_bytes())==h,'Embedded source changed: '+name)
    pinned=ac.PLAN['frozen_dependencies']
    for name,h in pinned.items():w.require(w.sha(Path(__file__).with_name(name).read_bytes())==h,'Frozen dependency changed: '+name)

def make_cache(engine,arm):
    if arm in ('OFFICIAL','OFFICIAL_REPEAT'):
        x=engine.base_class(max_cache_size=ac.SLOTS);x.r=8;return x
    w.require(arm in ac.ARMS,'Unknown AC08 cache arm')
    return link.make_cache(engine.base_class)(8,None if arm=='NATIVE' else 3,arm)

def resident():
    torch.cuda.synchronize()
    return dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved())

def trajectory(engine,root,doc,spec):
    ident=spec['id'];dest=root/spec['phase']/ident;dest.mkdir(parents=True,exist_ok=False)
    engine.condition_deadline=time.monotonic()+CASE_SECONDS
    w.write_json(root/'active_case.json',dict(id=ident,status='STARTED',deadline_monotonic=engine.condition_deadline))
    engine.set_r(8);warm=engine.forward(doc['tokens'][:32],None,8)
    w.require(bool(torch.isfinite(warm).all()),'Nonfinite auxiliary warmup')
    w.write_json(dest/'auxiliary_warmup.json',dict(forwards=1,tokens=32,token_sha256=w.token_sha(doc['tokens'][:32]),finite=True));del warm;cleanup()
    memory={'model_only':resident()};cache=make_cache(engine,spec['arm']);raw=[];records=[];times=[];states={};inventory=[]
    rng=(torch.random.get_rng_state().clone(),torch.cuda.get_rng_state().clone())
    started=time.monotonic();checker_start=input_checks.launch_count;vchecker_start=vchecks.launch_count
    w.write_json(dest/'trajectory_clock.json',dict(started_monotonic=started))
    try:
        for step,end in enumerate(range(768,spec['length']+1)):
            engine.check_time();first=step==0
            diagnostic_step=ac.has_diagnostic(spec,end)
            if getattr(cache,'ac01_connection',False):
                cache.capture=(lambda slot,length,g,tail,q,vg,vt,out:diag.capture(root,spec,slot,length,g,tail,q,vg,vt,out)) if ac.should_capture(spec,end) else None
            inp=doc['tokens'][:end] if first else [doc['tokens'][end-1]]
            out,tm=measured(lambda:engine.forward(inp,cache,8));cpu=out[0,-1].cpu();del out
            if not first and getattr(cache,'ac01_connection',False):
                counts=dict(cache.connection_calls);w.require(counts=={s:step for s in range(ac.SLOTS)},'Per-length connected slot coverage')
                packed=ac.packed(spec);compressed=cache.compressed.get(0,0) if hasattr(cache,'compressed') else 0
                w.require(compressed==ac.forward_compressed(end,packed),'Pre-post compression coordinate')
                checker_calls=input_checks.launch_count-checker_start
                w.require(checker_calls==(step*ac.SLOTS if spec['arm']=='PACKED_GROUPED' else 0),'Grouped checker coverage')
                vchecker_calls=vchecks.launch_count-vchecker_start
                w.require(vchecker_calls==(step*ac.SLOTS if spec['arm']=='PACKED_KV_FUSED' else 0),'Packed KV checker coverage')
                w.append_json(dest/'decode_coverage.jsonl',dict(length=end,compressed=compressed,kv_checker_calls=vchecker_calls,grouped_checker_calls=checker_calls,slot_cumulative_calls={str(s):n for s,n in counts.items()}))
            w.require(bool(torch.isfinite(cpu).all()),'Nonfinite model logits')
            raw.append(bits(cpu));w.np.save(dest/'raw_logits.npy',w.np.stack(raw),allow_pickle=False)
            if not first:
                row=dict(length=end,target_position=end,target_id=doc['tokens'][end],nll=score_nll(cpu,doc['tokens'][end]))
                records.append(row);w.append_json(dest/'per_token.jsonl',row)
            del cpu
            if first:
                w.write_json(dest/'prefill_precompression.json',dict(length=end,logits_sha256=w.sha(raw[0].tobytes()),state=state(cache)))
            before=cache.compressed.get(0,0) if hasattr(cache,'compressed') else 0
            def post():
                fn=getattr(cache,'post_prefill' if first else 'post_decode',None)
                if fn:fn()
            _,pt=measured(post);after=cache.compressed.get(0,0) if hasattr(cache,'compressed') else 0
            for operation,value in [('FORWARD',tm),('POST',pt)]:
                value.update(step=step,length=end,operation=operation,diagnostic_step=diagnostic_step,
                    timed_primary=spec['phase']=='main' and end in ac.TIMED_LENGTHS,flush=operation=='POST' and after>before,compressed=after)
                times.append(value);w.append_json(dest/'timing.jsonl',value)
            if first or end==spec['length']:
                memory['prefill_post' if first else 'terminal_post']=resident()
                w.write_json(dest/'resident.json',memory)
            if end in ac.STATE_LENGTHS:
                states[str(end)]=state(cache)
                inv=dict(length=end,**c.tensor_inventory(cache));inventory.append(inv)
                w.write_json(dest/'cache_state.json',states);w.write_json(dest/'inventory.json',inventory)
            if step%16==0:w.log(f'{ident}: {end}/{spec["length"]}')
            check_output_cap(root)
        if hasattr(cache,'calls'):c.validate(cache,spec['length'],spec['forwards'],3 if ac.packed(spec) else None)
        if getattr(cache,'ac01_connection',False):
            w.require(dict(cache.connection_calls)=={s:spec['scored'] for s in range(ac.SLOTS)},'Missing decode attention connection')
            expect={s:spec['scored'] for s in range(ac.SLOTS)} if ac.packed(spec) else {}
            w.require(dict(cache.skipped_k)==expect,'Unexpected eager K restoration')
            w.require(dict(cache.skipped_v)==({s:spec['scored'] for s in range(ac.SLOTS)} if spec['arm']=='PACKED_KV_FUSED' else {}),'Unexpected eager V restoration')
        for a,b in zip(rng,(torch.random.get_rng_state(),torch.cuda.get_rng_state())):w.require(torch.equal(a,b),'Inference changed RNG')
        result=dict(spec,status='COMPLETE_FINITE',seconds=time.monotonic()-started,nll=math.fsum(x['nll'] for x in records)/len(records),
            connection_calls={str(s):n for s,n in getattr(cache,'connection_calls',{}).items()},skipped_k={str(s):n for s,n in getattr(cache,'skipped_k',{}).items()},
            skipped_v={str(s):n for s,n in getattr(cache,'skipped_v',{}).items()},kv_checker_calls=vchecks.launch_count-vchecker_start,
            grouped_checker_calls=input_checks.launch_count-checker_start,raw_logits_sha256=w.sha((dest/'raw_logits.npy').read_bytes()),rng_unchanged=True)
        del cache;cache=None;cleanup()
        diag.replay(root,spec,engine.check_time)
        check_output_cap(root)
        result['seconds']=time.monotonic()-started
        w.write_json(dest/'result.json',result)
        w.write_json(dest/'trajectory_clock.json',dict(started_monotonic=started,finished_monotonic=time.monotonic()))
        w.write_json(root/'active_case.json',dict(id=ident,status='COMPLETE_FINITE'))
    finally:
        del cache;cleanup()

def cache_snapshot():
    p=Path(os.environ['TRITON_CACHE_DIR'])
    return {str(x.relative_to(p)):w.sha(x.read_bytes()) for x in sorted(p.rglob('*')) if x.is_file() and not x.name.endswith('.lock') and '.tmp' not in x.name}

def main(root):
    from loop_kv_ac08_runtime import preparation_started
    root=Path(root);started=preparation_started();status='INCONCLUSIVE';undo=None
    for p in ('gate','warmup','main','real_query','warmup_query','query_inputs'): (root/p).mkdir(exist_ok=True)
    w.write_json(root/'warmup_schedule.json',ac.warmup_schedule())
    w.write_json(root/'gate_schedule.json',ac.gate_schedule())
    w.write_json(root/'phase.json',dict(stage='PREPARE',deadline_monotonic=started+PREP_SECONDS))
    w.write_json(root/'triton_cache_initial.json',cache_snapshot())
    w.require(not cache_snapshot(),'AC08 needs a new empty Triton cache')
    def terminate(*_):raise w.BudgetStop('Launcher termination')
    signal.signal(signal.SIGTERM,terminate)
    try:
        verify_sources(root);model,engine=prepare_model(root,started)
        w.require(shutil.disk_usage(root).free>=2*MAX_BYTES,'Reserve 12 GiB evidence/ZIP space after model download')
        torch.set_float32_matmul_precision('highest');qenv=qk.environment();w.write_json(root/'qk_environment.json',qenv)
        w.write_json(root/'fused_kernel_config.json',fused.config())
        w.write_json(root/'checker_config.json',input_checks.config())
        model_module=importlib.import_module(type(model).__module__)
        w.require((model.config.num_attention_heads,model.config.num_key_value_heads,model.config.num_hidden_layers)==(16,16,24),'No GQA / fixed layer dimensions')
        undo=link.install(model,model_module);data=materials(root);docs={x['book_id']:x for x in data['smoke']+data['diagnostic']}
        from verify_loop_kv_ac08 import gate_checks,verify
        with d.precision_context(dict(kernel='AUTO',gemm_reduced=True)) as actual:
            w.write_json(root/'execution_settings.json',dict(actual,float32_matmul_precision=torch.get_float32_matmul_precision()))
            phase(engine,root,'GATE',GATE_SECONDS)
            engine.condition_deadline=math.inf
            validation.fixture_block(root,engine);cleanup()
            diag.kv_preflight(root,engine.check_time);cleanup()
            diag.fixture_block(root,engine.check_time);cleanup()
            for sp in ac.gate_schedule():trajectory(engine,root,docs[sp['book_id']],sp)
            gates=gate_checks(root);w.write_json(root/'gate_checks.json',gates);w.require(all(x['pass'] for x in gates),'Attention connection gate failed')
            phase(engine,root,'WARMUP',WARMUP_SECONDS)
            for rnd in range(ac.WARMUP_ROUNDS):
                for sp in [x for x in ac.warmup_schedule() if x['round']==rnd]:trajectory(engine,root,docs[sp['book_id']],sp)
                w.write_json(root/f'warmup_round{rnd}_cache.json',cache_snapshot())
            from verify_loop_kv_ac08 import warmup_coverage
            coverage=warmup_coverage(root);w.write_json(root/'timing_ready.json',dict(coverage=coverage,completed_monotonic=time.monotonic(),cache=cache_snapshot()))
            phase(engine,root,'MAIN',MAIN_SECONDS)
            for sp in ac.schedule():
                before=cache_snapshot();begin=time.monotonic();trajectory(engine,root,docs[sp['book_id']],sp)
                w.append_json(root/'main_cache.jsonl',dict(id=sp['id'],before=before,after=cache_snapshot(),started_monotonic=begin,finished_monotonic=time.monotonic()))
            w.write_json(root/'main_end_cache.json',cache_snapshot())
            phase(engine,root,'VERIFY',ac.CAPS['worker_verify'])
            # Save candidate summary from the separate NumPy-only rescorer, then launcher rechecks it.
            signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Worker verification cap')));signal.alarm(ac.CAPS["worker_verify"])
            result=verify(root,check_summary=False);signal.alarm(0);w.write_json(root/'summary.json',result);status='COMPLETE_COMPARISON'
    except BaseException as exc:
        signal.alarm(0)
        w.append_json(root/'failures.jsonl',dict(error_type=type(exc).__name__,error=w.redact(str(exc)),traceback=w.redact(traceback.format_exc())))
        w.log(w.redact(traceback.format_exc()))
    finally:
        if undo:undo()
        if status!='COMPLETE_COMPARISON':w.write_json(root/'summary.json',dict(status='INCONCLUSIVE',experiment=EXPERIMENT,reason='See failures.jsonl; partial evidence preserved'))
        w.write_json(root/'run_status.json',dict(status=status,experiment=EXPERIMENT,elapsed_seconds=time.monotonic()-started,legacy_full_p0_pass=False))
        w.log('Closed AC08: '+status)
def prepare_model(root, started):
    d.install_packages();numeric_imports();torch.set_grad_enabled(False)
    w.require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(),'Select a BF16-capable Colab GPU')
    kernel_env=kernel.environment()
    restore_env=restore.environment()
    w.write_json(root/'restore_kernel_environment.json',restore_env)
    w.write_json(root/'kernel_environment.json',kernel_env)
    packages={n:importlib.metadata.version(n) for n in w.PACKAGE_NAMES}
    w.require(packages==d.PARENT['packages'],'Dependency versions differ')
    w.require(torch.__version__==packages['torch'] and w.np.__version__==packages['numpy'],'Loaded versions differ')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True);torch.manual_seed(w.SEED)
    env=dict(packages=packages,gpu=torch.cuda.get_device_name(0),gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
             cuda=torch.version.cuda,python=sys.version)
    w.write_json(root/'environment.json',env)
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig,AutoTokenizer,AutoModelForCausalLM
    secret=os.environ.get('HF_TOKEN') or None
    source=root/'sources';source.mkdir(exist_ok=True)
    for name,digest in w.SOURCE_HASHES.items():
        path=Path(hf_hub_download(w.MODEL,name,revision=w.MODEL_REV,token=secret))
        w.require(w.sha(path.read_bytes())==digest,'Pinned source mismatch: '+name);shutil.copy2(path,source/name)
    config=AutoConfig.from_pretrained(w.MODEL,revision=w.MODEL_REV,token=secret,trust_remote_code=True)
    w.require((config.num_hidden_layers,config.num_key_value_heads,config.head_dim)==(24,16,128),'Model dimensions')
    w.require(all(x=='full_attention' for x in config.layer_types),'Attention types changed')
    tokenizer=AutoTokenizer.from_pretrained(w.MODEL,revision=w.MODEL_REV,token=secret,trust_remote_code=True)
    tok_hash=w.sha(tokenizer.backend_tokenizer.to_str().encode())
    w.require(tok_hash==d.PARENT['tokenizer_sha256'],'Tokenizer hash changed')
    w.require(config.max_position_embeddings>=1024,'Model context limit')
    m.verify_static()
    fixed=m.select_materials(root,tokenizer,started+PREP_SECONDS)
    ac.bind_books([x['book_id'] for x in fixed['diagnostic']])
    w.require(all(max(x['tokens'])<config.vocab_size for x in fixed['smoke']+fixed['diagnostic']),'Token vocabulary')
    config.total_ut_steps=8
    w.write_json(root/'model_load_started.json',dict(started_monotonic=time.monotonic(),selection_lock_sha256=w.sha((root/'selection_lock.json').read_bytes())))
    w.log('Loading pinned Ouro-1.4B BF16 for AC08. Four new books frozen before model load.')
    model=AutoModelForCausalLM.from_pretrained(w.MODEL,config=config,revision=w.MODEL_REV,token=secret,
        trust_remote_code=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa',low_cpu_mem_usage=True).to('cuda').eval()
    param_dtypes=dict(Counter(str(p.dtype) for p in model.parameters()))
    buffers=dict(Counter(str(b.dtype) for b in model.buffers() if b.is_floating_point()))
    w.require(model.dtype==torch.bfloat16 and set(param_dtypes)=={'torch.bfloat16'},'Wrong BF16 parameter dtype')
    w.require(set(buffers)<={'torch.bfloat16','torch.float32'},'Unexpected floating buffer dtype')
    if time.monotonic()-started>PREP_SECONDS:raise w.BudgetStop('Preparation cap')
    base=importlib.import_module(type(model).__module__).UniversalTransformerCache
    engine=w.Engine(model,root,w.make_adapter_class(base),base)
    w.write_json(root/'manifest.json',{**env,'experiment':EXPERIMENT,'software_version':SOFTWARE_VERSION,'model':w.MODEL,'model_revision':w.MODEL_REV,
        'tokenizer_sha256':tok_hash,'vocab_size':config.vocab_size,'execution_dtype':'BF16','parameter_dtype_counts':param_dtypes,
        'floating_buffer_dtype_counts':buffers,'source_hashes':w.SOURCE_HASHES,
        'worker_sha256':w.sha(Path(__file__).read_bytes()),'legacy_full_p0_pass':False,
        'readout':'exit_at_step=7; labels=None; use_weighted_exit=False','nll_dtype':'CPU float64','unpack_kernel':kernel_env,'restore_kernel':restore_env,
        'material_sha256':w.sha((root/'materials.json').read_bytes()),
        'caps_seconds':dict(prepare=PREP_SECONDS,gate=GATE_SECONDS,warmup=WARMUP_SECONDS,main=MAIN_SECONDS,case=CASE_SECONDS)})
    return model,engine

if __name__=='__main__':main(sys.argv[1])
