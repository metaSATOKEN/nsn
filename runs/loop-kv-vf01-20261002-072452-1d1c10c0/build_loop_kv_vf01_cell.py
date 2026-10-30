"""Offline source-only cell builder, copied/adapted from AF01. No GPU/model import."""
import base64,hashlib,json,zlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def build(root=ROOT):
    inputs=json.loads((root/'scripts/loop_kv_vf01_inputs.json').read_text());material=json.loads((root/'scripts/vf01_material_manifest.json').read_text())
    frozen=json.loads((root/'scripts/vf01_frozen_dependencies.json').read_text())
    names=list(frozen)+['loop_kv_vf01_inputs.json','vf01_material_manifest.json','vf01_frozen_dependencies.json','loop_kv_vf01_contract.py','loop_kv_vf01_kernel.py','loop_kv_vf01_checks.py','loop_kv_vf01.py','verify_loop_kv_vf01.py','loop_kv_vf01_runtime.py','loop_kv_vf01_launcher.py','build_loop_kv_vf01_cell.py']
    for n,h in frozen.items():assert hashlib.sha256((root/'scripts'/n).read_bytes()).hexdigest()==h,n
    sources={n:(root/'scripts'/n).read_text() for n in names}
    sources.update({'design.md':(root/'plans/loop-kv-vf01-design-v0.1-2026-10-02.md').read_text(),'prereg.md':(root/'throughline/prereg/loop-kv-vf01-v0.1-2026-10-02.md').read_text()})
    manifest={n:hashlib.sha256(t.encode()).hexdigest() for n,t in sources.items()}
    (root/'scripts/vf01_capsule.json').write_text(json.dumps(manifest,indent=2)+'\n')
    data=zlib.compress(json.dumps(sources,ensure_ascii=False,separators=(',',':')).encode(),9);encoded=base64.b64encode(data).decode()
    code='''# LOOP-KV-VF01 v0.1.0 / 2026-10-02
# Put loop_kv_vf01_materials.zip in /content, then paste ALL this file into ONE Colab GPU cell.
# No Drive, HF_TOKEN or model loading. Result ZIP auto-downloads.
import os,sys,json,time,threading,subprocess,zipfile,uuid,re
from pathlib import Path
from datetime import datetime,timezone
'''
    constants=dict(MATERIAL_NAME=material['file'],MATERIAL_SHA256=material['sha256'],EXPECTED_CAPS=inputs['caps_seconds'],OUTPUT_LIMIT_BYTES=inputs['output_limit_bytes'],EXPECTED_TIMING=inputs['timing']['host_schedule'],PAYLOAD_SHA256=hashlib.sha256(data).hexdigest(),EMBEDDED_SHA256=manifest)
    for n,v in constants.items():code+=n+' = '+repr(v)+'\n'
    code+='EMBEDDED_PAYLOAD = (\n'+''.join('    '+repr(encoded[i:i+160])+'\n' for i in range(0,len(encoded),160))+')\n'
    code+=sources['loop_kv_vf01_launcher.py']+'\n_loopkv_vf01_cell()\n';compile(code,'vf01_onecell','exec')
    dest=root/'colab/loop_kv_vf01_onecell.py';dest.parent.mkdir(exist_ok=True);dest.write_text(code)
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(accelerator='GPU',kernelspec=dict(name='python3',display_name='Python 3')),cells=[dict(cell_type='code',id='vf01-component',metadata={},execution_count=None,outputs=[],source=code.splitlines(keepends=True))])
    dest.with_suffix('.ipynb').write_text(json.dumps(nb,ensure_ascii=False,separators=(',',':'))+'\n')
    report=dict(python_bytes=dest.stat().st_size,python_sha256=hashlib.sha256(code.encode()).hexdigest(),embedded_files=len(sources),notebook_bytes=dest.with_suffix('.ipynb').stat().st_size,notebook_sha256=hashlib.sha256(dest.with_suffix('.ipynb').read_bytes()).hexdigest(),frozen_dependencies=frozen,embedded_arrays=False,material_zip=material['file'],material_sha256=material['sha256'],material_bytes=material['bytes'])
    (root/'colab/vf01-build.json').write_text(json.dumps(report,indent=2)+'\n');return report
if __name__=='__main__':print(json.dumps(build(),indent=2))
