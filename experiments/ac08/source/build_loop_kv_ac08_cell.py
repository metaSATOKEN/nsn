"""Offline build of a small source-only AC08 Colab cell."""
import base64,hashlib,json,zlib
from pathlib import Path
import loop_kv_ac08_contract as ac
ROOT=Path(__file__).resolve().parents[1]
NEW_FILES=['loop_kv_ac08_runtime.py', 'loop_kv_ac08.py', 'loop_kv_ac08_contract.py', 'loop_kv_ac08_diagnostics.py', 'loop_kv_ac08_audit.py', 'loop_kv_ac08_inputs.json', 'verify_loop_kv_ac08.py', 'loop_kv_ac08_launcher.py', 'build_loop_kv_ac08_cell.py', 'loop_kv_ac08_materials.py', 'loop_kv_ac08_candidates.json', 'lc02_sources/pg19-train-files.txt', 'lc02_sources/pg19-metadata.csv', 'loop_kv_ac08_attention.py', 'loop_kv_ac08_preflight_audit.py']
def build(root=ROOT):
    plan=json.loads((root/'scripts/loop_kv_ac08_inputs.json').read_text());pins=plan['frozen_dependencies']
    assert (root/'scripts/loop_kv_ac08_inputs.json').read_bytes()==(root/'plans/loop-kv-ac08-input-contract-v0.1-2026-10-02.json').read_bytes()
    for name,digest in pins.items():assert hashlib.sha256((root/'scripts'/name).read_bytes()).hexdigest()==digest,name
    sources={n:(root/'scripts'/n).read_text() for n in list(pins)+NEW_FILES}
    sources.update({'design.md':(root/'plans/loop-kv-ac08-design-v0.1-2026-10-02.md').read_text(),
        'prereg.md':(root/'throughline/prereg/loop-kv-ac08-v0.1-2026-10-02.md').read_text()})
    manifest={n:hashlib.sha256(text.encode()).hexdigest() for n,text in sources.items()}
    (root/'scripts/ac08_capsule.json').write_text(json.dumps(manifest,indent=2)+'\n')
    data=zlib.compress(json.dumps(sources,ensure_ascii=False,separators=(',',':')).encode(),9);encoded=base64.b64encode(data).decode()
    code='''# LOOP-KV-AC08 v0.1.0 / 2026-10-02
# Paste this whole file into one fresh Colab GPU cell. HF_TOKEN; no Drive or prior ZIP.
# Isolated cu128 runtime; Colab CPython 3.13/Linux x86_64. Base environment unchanged.
# Same pinned Ouro-1.4B model, 4-new-book x 3-method x 2-repeat comparison, automatic result ZIP download.
import os,sys,json,time,threading,subprocess,zipfile,uuid,re
from pathlib import Path
from datetime import datetime,timezone
HF_SECRET_NAME = "HF_TOKEN"
'''
    code+='PAYLOAD_SHA256 = '+repr(hashlib.sha256(data).hexdigest())+'\n'
    code+='EMBEDDED_SHA256 = '+repr(manifest)+'\nEXPECTED_MAIN_TEMPLATE = '+repr(ac.PLAN['main_schedule_template'])+'\nEXPECTED_BASE_SCHEDULE = '+repr(ac.gate_schedule()+ac.warmup_schedule())+'\nEXPECTED_CAPS = '+repr(ac.CAPS)+'\nOUTPUT_LIMIT_BYTES = '+repr(ac.PLAN['output_limit_bytes'])+'\n'
    code+='EMBEDDED_PAYLOAD = (\n'+''.join('    '+repr(encoded[i:i+160])+'\n' for i in range(0,len(encoded),160))+')\n'
    code+=sources['loop_kv_ac08_launcher.py']+'\n_loopkv_ac08_cell()\n'
    compile(code,'ac08_onecell','exec');dest=root/'colab/loop_kv_ac08_onecell.py';dest.parent.mkdir(exist_ok=True);dest.write_text(code)
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(accelerator='GPU',kernelspec=dict(name='python3',display_name='Python 3')),
        cells=[dict(cell_type='code',id='ac08-connection',metadata={},execution_count=None,outputs=[],source=code.splitlines(keepends=True))])
    dest.with_suffix('.ipynb').write_text(json.dumps(nb,ensure_ascii=False,separators=(',',':'))+'\n')
    report=dict(python_bytes=dest.stat().st_size,python_sha256=hashlib.sha256(code.encode()).hexdigest(),embedded_files=len(sources),
        notebook_bytes=dest.with_suffix('.ipynb').stat().st_size,notebook_sha256=hashlib.sha256(dest.with_suffix('.ipynb').read_bytes()).hexdigest(),frozen_dependencies=pins)
    (root/'colab/ac08-build.json').write_text(json.dumps(report,indent=2)+'\n');return report
if __name__=='__main__':print(json.dumps(build(),indent=2))
