# Embedded stdlib-only launcher. No Drive, token request, model download, or returned-code execution.
def _loopkv_iv01_cell():
    import base64,zlib,hashlib,shutil,signal,importlib.metadata
    run_id='loop-kv-iv01-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8]
    work=Path('/content')/run_id;work.mkdir(parents=True,exist_ok=False)
    root=work/'artifacts';root.mkdir();src=root/'source';src.mkdir()
    proc=None;reader=None;env=None;capture=work/'console.log'
    def clean(s):return re.sub(r'hf_[A-Za-z0-9]{20,}','[REDACTED]',str(s))
    def save(name,obj):
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
        tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)
    def load(name):
        try:return json.loads((root/name).read_text())
        except (OSError,ValueError):return None
    def stop():
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:proc.wait(timeout=15)
            except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=10)
    try:
        payload=base64.b64decode(EMBEDDED_PAYLOAD,validate=True)
        if hashlib.sha256(payload).hexdigest()!=PAYLOAD_SHA256:raise RuntimeError('Payload digest mismatch')
        sources=json.loads(zlib.decompress(payload))
        if set(sources)!=set(EMBEDDED_SHA256):raise RuntimeError('Source inventory mismatch')
        for name,body in sources.items():
            if Path(name).is_absolute() or len(Path(name).parts)!=1:raise RuntimeError('Unsafe source path')
            if hashlib.sha256(body.encode()).hexdigest()!=EMBEDDED_SHA256[name]:raise RuntimeError('Source digest mismatch: '+name)
            (src/name).write_text(body)
        save('capsule.json',EMBEDDED_SHA256);save('source/iv01_capsule.json',EMBEDDED_SHA256)
        parent=Path('/content')/PARENT_ZIP_NAME
        if not parent.is_file():
            print('Upload the original DC01 result ZIP when the chooser appears. No Drive or HF_TOKEN is used.',flush=True)
            from google.colab import files
            uploaded=files.upload()
            candidates=[name for name in uploaded if Path(name).name==PARENT_ZIP_NAME]
            if len(candidates)!=1:raise RuntimeError('Original DC01 ZIP not selected; place it in /content and rerun.')
            incoming=Path(candidates[0]);data=uploaded[candidates[0]]
            if not incoming.is_file():incoming.write_bytes(data)
            if incoming.resolve()!=parent.resolve():shutil.copy2(incoming,parent)
            del uploaded,data
        # Verify before importing/running numerical code. Return ZIP code is never read or executed.
        h=hashlib.sha256()
        with parent.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
        if h.hexdigest()!=PARENT_ZIP_SHA256:raise RuntimeError('Wrong DC01 ZIP hash; original file required.')
        for name,version in [('torch','2.11.0+cu128'),('triton','3.6.0')]:
            if importlib.metadata.version(name)!=version:raise RuntimeError('Requires '+name+' '+version+'; no automatic torch/Triton replacement.')
        if importlib.metadata.version('numpy')!='2.1.3':
            print('Preparing NumPy 2.1.3 for a fresh child process.',flush=True)
            pp=subprocess.run([sys.executable,'-m','pip','install','--quiet','--disable-pip-version-check','numpy==2.1.3'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=180)
            (root/'numpy_install.log').write_text(clean(pp.stdout))
            if pp.returncode:raise RuntimeError('NumPy preparation failed')
        env=os.environ.copy()
        for key in list(env):
            if key.startswith(('HF_','HUGGING_FACE_','TRITON_')) or key in ('HUGGINGFACE_TOKEN','HUGGINGFACEHUB_API_TOKEN'):env.pop(key,None)
        env.update(TRITON_CACHE_DIR=str(work/'triton_cache'),PYTHONPATH=str(src),CUBLAS_WORKSPACE_CONFIG=':4096:8',
            HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1')
        print('IV01: saved inputs only. 64 fixtures x 3 methods; 108 first calls, full warmup, and 324 timing intervals.',flush=True)
        save('phase.json',dict(stage='PREPARE',deadline_monotonic=time.monotonic()+600))
        proc=subprocess.Popen([sys.executable,'-u',str(src/'loop_kv_iv01.py'),str(root),str(parent)],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        def drain():
            with capture.open('w') as f:
                for line in proc.stdout:
                    s=clean(line);f.write(s);f.flush();print(s,end='',flush=True)
        reader=threading.Thread(target=drain,daemon=True);reader.start();deadline=time.monotonic()+1320
        while proc.poll() is None:
            time.sleep(1);phase=load('phase.json');active=load('active_case.json')
            expired=active and active.get('status')=='STARTED' and time.monotonic()>active['deadline_monotonic']+5
            if expired or (phase and time.monotonic()>phase['deadline_monotonic']+5) or time.monotonic()>deadline or sum(p.stat().st_size for p in root.rglob('*') if p.is_file())>512*1024**2:
                save('launcher_timeout.json',dict(reason='Case/phase/overall/size cap',phase=phase,active=active));stop();break
        reader.join(timeout=10);save('launcher_status.json',dict(worker_exit_code=proc.returncode,run_id=run_id))
    except BaseException as exc:
        stop();save('launcher_error.json',dict(error_type=type(exc).__name__,error=clean(exc)));print(clean(exc),flush=True)
    finally:
        if reader is not None:reader.join(timeout=10)
        if capture.exists():shutil.copy2(capture,root/'console.log')
        verified=False;terminal=load('status.json') or {}
        if terminal.get('status')=='COMPLETE_COMPARISON' and not (root/'launcher_error.json').exists() and not (root/'launcher_timeout.json').exists() and proc is not None and proc.returncode==0:
            try:
                check=subprocess.run([sys.executable,str(src/'verify_loop_kv_iv01.py'),str(root)],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=120)
                (root/'recovery_verification.log').write_text(clean(check.stdout));verified=check.returncode==0
            except BaseException as exc:save('verification_error.json',dict(error=clean(exc)))
        if not verified:
            for name in ('status.json','summary.json'):
                if (root/name).exists():shutil.copy2(root/name,root/('worker_'+name))
            counts={}
            for name in ('fixtures','rejections','first_calls','warmup','timings','post'):
                p=root/(name+'.jsonl');counts[name]=len(p.read_text().splitlines()) if p.exists() else 0
            recovery=dict(status='INCONCLUSIVE',experiment='LOOP-KV-IV01',recorded_lines=counts,active_case=load('active_case.json'),note='Partial evidence and worker files retained; no complete claim.')
            save('launcher_recovery.json',recovery);save('status.json',recovery);save('summary.json',recovery)
        def recovery_timeout(*_):raise TimeoutError('ZIP recovery 120-second cap')
        previous=signal.signal(signal.SIGALRM,recovery_timeout);signal.alarm(120)
        try:
            keep=lambda p:p.is_file() and not p.name.endswith('.tmp') and '__pycache__' not in p.parts
            hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob('*')) if keep(p) and p.name!='SHA256SUMS.json'}
            save('SHA256SUMS.json',hashes);result=Path('/content')/(run_id+'.zip');partial=result.with_suffix('.zip.tmp')
            with zipfile.ZipFile(partial,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
                for p in sorted(root.rglob('*')):
                    if keep(p):z.write(p,str(p.relative_to(root)))
            partial.replace(result)
        except Exception as exc:
            print('ZIP recovery failed:',clean(exc),'Files remain at',root,flush=True);return
        finally:signal.alarm(0);signal.signal(signal.SIGALRM,previous)
        print('Result ZIP:',result,flush=True)
        try:
            from google.colab import files
            files.download(str(result));print('Download requested. ZIP also remains in the Colab Files panel.',flush=True)
        except Exception as exc:print('Download request failed:',clean(exc),'ZIP remains at',result,flush=True)
