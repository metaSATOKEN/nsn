"""Bounded model trajectories. Diagnostics and evidence IO stay outside main timing."""
import gc,os,time,random,struct
from pathlib import Path
import loop_kv_nb06_contract as c
import loop_kv_nb06_attention as link
import loop_kv_nb06_evidence as ev
import loop_kv_nb06_kernel as k
import loop_kv_nb06_checks as checks
import loop_kv_nb06_capture as cap

def measured(t,fn):
    # AC08 measurement boundary: inputs already on device; evidence saved after return.
    t.cuda.synchronize();t.cuda.reset_peak_memory_stats()
    before=dict(allocated=t.cuda.memory_allocated(),reserved=t.cuda.memory_reserved())
    a=t.cuda.Event(enable_timing=True);b=t.cuda.Event(enable_timing=True)
    began=time.monotonic();start=time.perf_counter();a.record();out=fn();b.record();b.synchronize()
    return out,dict(started_monotonic=began,finished_monotonic=time.monotonic(),
        cuda_ms=a.elapsed_time(b),wall_ms=(time.perf_counter()-start)*1000,
        allocated_before=before['allocated'],reserved_before=before['reserved'],
        allocated_peak=t.cuda.max_memory_allocated(),reserved_peak=t.cuda.max_memory_reserved())

def cache_snapshot():
    p=Path(os.environ['TRITON_CACHE_DIR'])
    return {str(x.relative_to(p)):c.file_digest(x) for x in sorted(p.rglob('*')) if x.is_file() and not x.name.endswith('.lock') and '.tmp' not in x.name}

def packed_state(cache,first_only=False):
    out={}
    for slot in range(44):
        for side in ('K','V'):
            g=cache.packed.get((slot,side));label=f'{slot:02d}-{side}';out[label]=[]
            if g is None:continue
            count=1 if first_only else g['codes'].shape[2]
            for block in range(count):
                a={'codes':cap.array(g['codes'][:,:,block:block+1]),'headers':cap.array(g['headers'][:,:,block:block+1]),
                   **{'meta'+str(i):cap.array(x[:,:,block:block+1]) for i,x in enumerate(g['metas'])}}
                want=ev.w.np.frombuffer(struct.pack('<QHBB',block*64,64,3,0),ev.w.np.uint8)
                c.require(a['headers'].shape==(1,8,1,12) and bool((a['headers'][0,:,0]==want).all()),'Header contract')
                out[label].append(c.arrays_hash(a))
    return out

def prefix_state(cache):
    if cache.bits is not None and cache.packed:return packed_state(cache,first_only=True)
    return {f'{slot:02d}-{side}':{'native':c.array_hash(ev.bits(values[slot][:,:,:64,:]))}
        for slot in range(44) for side,values in [('K',cache.key_cache),('V',cache.value_cache)]}

def routes(cache):
    return {name:[getattr(cache,name)[i] for i in range(44)] for name in ['calls','connections','skipped_k','skipped_v']}

def trajectory(root,model,t,np,tokens,spec,cache_type,deadline,progress,set_alarm):
    ident=spec['id'];arm=spec['arm'];cache=cache_type(arm)
    random.seed(c.MODEL['seed']);np.random.seed(c.MODEL['seed']);t.manual_seed(c.MODEL['seed']);t.cuda.manual_seed_all(c.MODEL['seed'])
    end=min(deadline,time.monotonic()+c.CAPS['series']);set_alarm(end)
    c.write_json(root,'active_case.json',dict(id=ident,status='STARTED',deadline_monotonic=end))
    started=time.monotonic();saved=[];saved_steps=[];logit_hashes=[];nll=[];prefix_before=None;prefill_source=None;captured=[]
    current_step=[0]
    def capture(slot,data,actual):
        point=dict(slot=slot,step=current_step[0],length=128+current_step[0])
        if point not in c.PLAN['captures']:return
        label=f'step-{point["step"]}-slot-{slot:02d}';arrays=cap.input_arrays(data);before=c.arrays_hash(arrays)
        z=cap.diagnostic(data);after=c.arrays_hash(cap.input_arrays(data))
        name='actual_q/'+label+'.npz';c.save_arrays(root,name,{**arrays,**z,'model_output':cap.array(actual,True)})
        row=dict(point,file=name,sha256=c.file_digest(root/name),input_before=before,input_after=after)
        c.append(root,'actual_q.jsonl',row);captured.append(point)
    cache.capture=capture if spec['phase']=='diagnostic' else None
    for step in range(c.PLAN['forward_calls_per_series']):
        current_step[0]=step;length=128+step;first=0 if step==0 else length-1
        ids=t.tensor([tokens[first:length]],dtype=t.long,device='cuda:0')
        mask=t.ones((1,length),dtype=t.long,device='cuda:0')
        position=t.arange(first,length,dtype=t.long,device='cuda:0')
        progress['attempted']+=1;progress.update(series=ident,step=step);c.write_json(root,'progress.json',progress,closing=True)
        scan_before=checks.launch_count;attn_before=k.attention_launch_count
        def forward():
            with t.inference_mode():
                return model(input_ids=ids,attention_mask=mask,cache_position=position,past_key_values=cache,
                    use_cache=True,return_dict=True,output_attentions=False,output_hidden_states=False)
        output,tm=measured(t,forward)
        c.require(output.past_key_values is cache,'Cache object identity changed')
        c.require(tuple(output.logits.shape)==(1,length-first,c.MODEL['vocab_size']) and output.logits.dtype==t.float32,'Logit shape/type')
        x=output.logits[0,-1].detach().cpu().numpy().copy()
        c.require(bool(np.isfinite(x).all()),'Nonfinite logits');logit_hashes.append(c.array_hash(x))
        stats=c.nll_components(x,tokens[length]);nll.append(c.from_components(stats))
        save_raw=step in spec['raw_logit_steps']
        if save_raw:saved.append(x);saved_steps.append(step)
        # Free the huge prefill logits before compression; do not charge them as persistent KV.
        del output
        compression=None;post=None;update=None
        flushing=cache.bits is not None and step in c.PLAN['compression_contract']['post_decode_flush_steps']
        before=packed_state(cache) if flushing else None
        if step==0:
            prefill_source=prefix_state(cache)
            if cache.bits is not None:_,compression=measured(t,cache.post_prefill)
            prefix_before=prefix_state(cache)
        else:
            # Equal instrumentation boundaries: native post_decode is a no-op.
            _,post=measured(t,cache.post_decode)
            if flushing:update=dict(before=before,after=packed_state(cache))
        inventory=ev.persistent_inventory(cache)
        row=dict(id=ident,arm=arm,phase=spec['phase'],step=step,length=length,target_token=tokens[length],
            logit_sha256=logit_hashes[-1],nll=nll[-1],nll_components=stats,model_timing=tm,compression_timing=compression,post_timing=post,packed_update=update,
            routes=routes(cache),input_scan_calls=checks.launch_count-scan_before,
            attention_calls=k.attention_launch_count-attn_before,inventory=inventory,
            capture_enabled=cache.capture is not None)
        (root/'series'/ident).mkdir(parents=True,exist_ok=True)
        c.append(root,f'series/{ident}/steps.jsonl',row)
        if save_raw:c.save_arrays(root,f'series/{ident}/logits.npz',{'steps':np.asarray(saved_steps,np.int64),'logits':np.stack(saved)})
        progress['finished']+=1;c.write_json(root,'progress.json',progress,closing=True)
        del x,ids,mask,position
        if step in c.PLAN['checkpoints']:print(f'NB06 {root.name}/{ident}: decode {step}/256',flush=True)
    cache.capture=None
    prefix_after=prefix_state(cache)
    # Gate hashes need no full-cache arrays in the ZIP. No full restoration in timed paths.
    terminal,_=ev.snapshot(cache,c.PLAN['terminal_cache_length']) if spec['phase']=='gate' else (None,None)
    done=dict(spec,started_monotonic=started,finished_monotonic=time.monotonic(),forward_calls=c.PLAN['forward_calls_per_series'],
        logit_hashes=logit_hashes,nll=nll,prefill_source=prefill_source,prefix_before=prefix_before,prefix_after=prefix_after,
        terminal_hashes=terminal,captured=captured,routes=routes(cache))
    if spec['raw_logits']:
        done['logits_file']=f'series/{ident}/logits.npz';done['logits_sha256']=c.file_digest(root/done['logits_file'])
    c.write_json(root,f'series/{ident}/complete.json',done)
    c.append(root,'trajectories.jsonl',done)
    c.write_json(root,'active_case.json',dict(id=ident,status='COMPLETE',deadline_monotonic=end))
    del cache,saved;gc.collect();t.cuda.synchronize();t.cuda.empty_cache()
    print(f"NB06 {root.name}/{ident}: {c.PLAN['forward_calls_per_series']} forwards saved.",flush=True)
    return done

def measure(root,model,t,np,tokens,deadline,progress,set_alarm,base=None):
    if base is None:
        from transformers.cache_utils import DynamicCache
        base=DynamicCache
    ev.initialize(t,np);set_alarm(deadline)
    c.write_json(root,'codec_fixtures.json',ev.codec_fixtures(root))
    c.write_json(root,'environment.json',dict(torch=t.__version__,numpy=np.__version__,gpu=t.cuda.get_device_name(0),
        model_loaded=True,tf32=False,sdpa_policy='PyTorch default; copied official repeat_kv; checked native baseline'))
    cache_type=link.make_cache(base);restore=link.install(model);finished=[]
    try:
        for spec in c.schedule():
            if spec['id']=='main-r0-native':
                c.write_json(root,'timing_ready.json',dict(completed_monotonic=time.monotonic(),cache=cache_snapshot(),warmup_trajectories=3))
            finished.append(trajectory(root,model,t,np,tokens,spec,cache_type,deadline,progress,set_alarm))
            if spec['id']=='gate-native':
                left,right=finished[:2]
                match=left['logit_hashes']==right['logit_hashes'] and left['terminal_hashes']==right['terminal_hashes']
                c.write_json(root,'official_gate.json',dict(bit_equal=match,logits_bit_equal=left['logit_hashes']==right['logit_hashes'],terminal_kv_equal=left['terminal_hashes']==right['terminal_hashes']))
                c.require(match,'Official/native connection gate failed; no compressed model comparison')
        c.write_json(root,'timing_after.json',dict(completed_monotonic=time.monotonic(),cache=cache_snapshot()))
    finally:restore()
