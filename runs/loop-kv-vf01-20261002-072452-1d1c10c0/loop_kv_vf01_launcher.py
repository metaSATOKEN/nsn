# Embedded launcher. Always attempts ZIP recovery; no Drive mount.
def _loopkv_vf01_cell():
    import hashlib,shutil,base64,zlib
    run_id='loop-kv-vf01-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8]
    work=Path('/content')/run_id;work.mkdir(parents=True,exist_ok=False)
    root=work/'artifacts';root.mkdir()
    token='';proc=None;reader=None;capture=work/'console.log'
    runtime_python=work/'runtime'/'bin'/'python'
    started=time.monotonic();outer_deadline=started+EXPECTED_CAPS['outer']
    def clean(s):
        s=str(s)
        if isinstance(token,str) and token:s=s.replace(token,'[REDACTED]')
        return re.sub(r'hf_[A-Za-z0-9]{20,}','[REDACTED]',s)
    def save(name,obj):
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
        tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2));os.replace(tmp,p)
    def load(name):
        try:return json.loads((root/name).read_text())
        except (OSError,ValueError):return None
    def stop():
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:proc.wait(timeout=45)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
    try:
        payload=base64.b64decode(EMBEDDED_PAYLOAD,validate=True)
        if hashlib.sha256(payload).hexdigest()!=PAYLOAD_SHA256:raise RuntimeError('Payload digest mismatch')
        sources=json.loads(zlib.decompress(payload))
        if set(sources)!=set(EMBEDDED_SHA256):raise RuntimeError('Source inventory mismatch')
        for name,body in sources.items():
            if Path(name).is_absolute() or '..' in Path(name).parts:raise RuntimeError('Unsafe payload path')
            if hashlib.sha256(body.encode()).hexdigest()!=EMBEDDED_SHA256[name]:raise RuntimeError('Source digest mismatch: '+name)
            p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(body)
        save('embedded_sha256.json',EMBEDDED_SHA256);save('capsule.json',EMBEDDED_SHA256)
        save('vf01_capsule.json',EMBEDDED_SHA256)
        parent=Path('/content')/MATERIAL_NAME
        if not parent.is_file():
            print('Place '+MATERIAL_NAME+' in /content, or choose that material ZIP now. No Drive/HF token.',flush=True)
            from google.colab import files
            uploaded=files.upload()
            candidates=[n for n in uploaded if Path(n).name==MATERIAL_NAME]
            if len(candidates)!=1:raise RuntimeError('The VF01 material ZIP was not selected.')
            incoming=Path(candidates[0]);data=uploaded[candidates[0]]
            if not incoming.is_file():incoming.write_bytes(data)
            if incoming.resolve()!=parent.resolve():shutil.copy2(incoming,parent)
            del uploaded,data
        # Human file selection time is outside the computation cap.
        started=time.monotonic();outer_deadline=started+EXPECTED_CAPS['outer']
        with parent.open('rb') as f:
            if hashlib.file_digest(f,'sha256').hexdigest()!=MATERIAL_SHA256:raise RuntimeError('VF01 material ZIP hash mismatch')
        save('input_path.json',dict(path=str(parent),sha256=MATERIAL_SHA256))
        if shutil.disk_usage(work).free < 2*OUTPUT_LIMIT_BYTES:raise RuntimeError('At least 2 GiB free disk is required for evidence plus ZIP, in addition to runtime packages.')
        print('VF01: saved packed K/V, 36 synthetic-query cases x 3 arms. 16 normal and 24 invalid fixtures; 108 first calls, full warmup, 540 timing intervals. No model loading.',flush=True)
        triton_cache=work/'triton_cache';triton_cache.mkdir()
        env={k:v for k,v in os.environ.items() if not k.startswith(('TRITON_','HF_','HUGGING_FACE_')) and k not in ('HUGGINGFACE_TOKEN','HUGGINGFACEHUB_API_TOKEN')}
        env.update(TRITON_CACHE_DIR=str(triton_cache),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',TOKENIZERS_PARALLELISM='false',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        env.update(LOOP_KV_VF01_PREPARE_STARTED=str(started),PYTHONNOUSERSITE='1')
        env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
        save('phase.json',dict(stage='PREPARE',deadline_monotonic=started+EXPECTED_CAPS['prepare']))
        print('Preparing a separate cu128 environment; package download is included in the 30-minute preparation limit.',flush=True)
        proc=subprocess.Popen([sys.executable,'-u',str(root/'loop_kv_vf01_runtime.py'),str(root)],env=env,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        del env
        def drain():
            with capture.open('w') as f:
                for line in proc.stdout:
                    s=clean(line);f.write(s);f.flush();print(s,end='',flush=True)
        reader=threading.Thread(target=drain,daemon=True);reader.start()
        deadline=min(outer_deadline-EXPECTED_CAPS['independent_verify']-EXPECTED_CAPS['zip'],time.monotonic()+sum(EXPECTED_CAPS[k] for k in ('prepare','experiment','worker_verify')))
        while proc.poll() is None:
            time.sleep(1);phase=load('phase.json');active=load('active_case.json')
            expired=active and active.get('status')=='STARTED' and time.monotonic()>active['deadline_monotonic']+15
            if expired or (phase and time.monotonic()>phase['deadline_monotonic']+15) or time.monotonic()>deadline or sum(p.stat().st_size for p in root.rglob('*') if p.is_file())>OUTPUT_LIMIT_BYTES:
                save('launcher_timeout.json',dict(reason='Case/phase/overall cap',phase=phase,active=active));stop();break
        reader.join(timeout=10)
        save('launcher_status.json',dict(worker_exit_code=proc.returncode,run_id=run_id))
    except BaseException as exc:
        stop();save('launcher_error.json',dict(error_type=type(exc).__name__,error=clean(exc)));print(clean(exc),flush=True)
    finally:
        if reader is not None:reader.join(timeout=10)
        if capture.exists():shutil.copy2(capture,root/'console.log')
        interrupted=(root/'launcher_error.json').exists() or (root/'launcher_timeout.json').exists()
        verified=False
        terminal=load('run_status.json') or {}
        complete_claim='planned' in (load('summary.json') or {})
        if not complete_claim:
            save('completion_check_skipped.json',dict(reason='Worker did not claim complete measurements',worker_status=terminal.get('status')))
        if complete_claim and not interrupted and proc is not None and proc.returncode==0:
            try:
                check=subprocess.run([str(runtime_python),str(root/'verify_loop_kv_vf01.py'),str(root)],
                    env={**{k:v for k,v in os.environ.items() if not k.startswith(('HF_','HUGGING_FACE_')) and k not in ('PYTHONPATH','PYTHONHOME','HUGGINGFACE_TOKEN','HUGGINGFACEHUB_API_TOKEN')},'PYTHONNOUSERSITE':'1'},
                    stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=max(1,min(EXPECTED_CAPS['independent_verify'],outer_deadline-time.monotonic()-EXPECTED_CAPS['zip'])))
                (root/'recovery_verification.log').write_text(clean(check.stdout))
                verified=check.returncode==0
            except BaseException as exc:save('verification_error.json',dict(error=clean(exc)))
        if not verified:
            for name in ['run_status.json','summary.json']:
                if (root/name).exists():shutil.copy2(root/name,root/('worker_'+name))
            counts={}
            for name in ('fixtures','rejections','first_calls','diagnostics','warmup','timings'):
                p=root/(name+'.jsonl');counts[name]=len(p.read_text().splitlines()) if p.exists() else 0
            timing=[]
            if (root/'timings.jsonl').exists():
                for line in (root/'timings.jsonl').read_text().splitlines():
                    try:timing.append(json.loads(line))
                    except ValueError:pass
            observed={r.get('id') for r in timing}
            recovery=dict(status='INCONCLUSIVE',experiment='LOOP-KV-VF01',recorded_lines=counts,
                missing_timing_ids=[r['id'] for r in EXPECTED_TIMING if r['id'] not in observed],active_case=load('active_case.json'),
                note='Original worker results and partial evidence retained. No unverified complete claim.')
            save('launcher_recovery.json',recovery);save('run_status.json',recovery);save('summary.json',recovery)
        import signal
        def recovery_timeout(*_):raise TimeoutError('ZIP recovery capped at 120 seconds / remaining outer budget')
        previous_handler=signal.signal(signal.SIGALRM,recovery_timeout);signal.alarm(max(1,min(EXPECTED_CAPS['zip'],int(outer_deadline-time.monotonic()))))
        try:
            hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob('*'))
                    if p.is_file() and p.name!='SHA256SUMS.json' and not p.name.endswith('.tmp') and '__pycache__' not in p.parts}
            save('SHA256SUMS.json',hashes)
            result=Path('/content')/(run_id+'.zip')
            temporary=result.with_suffix('.zip.tmp')
            with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
                for p in sorted(root.rglob('*')):
                    if p.is_file() and not p.name.endswith('.tmp') and '__pycache__' not in p.parts:z.write(p,str(p.relative_to(root)))
            os.replace(temporary,result)
        except Exception as exc:
            print('ZIP recovery failed:',clean(exc),'Files remain at',root,flush=True)
            return
        finally:
            signal.alarm(0);signal.signal(signal.SIGALRM,previous_handler)
        print('Result ZIP:',result,flush=True)
        try:
            from google.colab import files
            files.download(str(result));print('Download requested. The ZIP is also in the Colab Files panel.',flush=True)
        except Exception as exc:print('Download request failed:',clean(exc),'ZIP remains at',result,flush=True)
