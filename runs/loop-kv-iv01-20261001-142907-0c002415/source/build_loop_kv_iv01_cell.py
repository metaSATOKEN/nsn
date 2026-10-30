"""Offline source-only build. No GPU/model imports; no saved arrays embedded."""
import base64,hashlib,json,zlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def build(root=ROOT):
    idx=json.loads((root/'scripts/loop_kv_iv01_inputs.json').read_text())
    names=list(idx['dependencies'])+['loop_kv_iv01_inputs.json','loop_kv_iv01_contract.py','loop_kv_iv01_kernel.py','loop_kv_iv01.py','verify_loop_kv_iv01.py','loop_kv_iv01_launcher.py','build_loop_kv_iv01_cell.py']
    for n,h in idx['dependencies'].items():assert hashlib.sha256((root/'scripts'/n).read_bytes()).hexdigest()==h,n
    sources={n:(root/'scripts'/n).read_text() for n in names}
    sources.update({'design.md':(root/'plans/loop-kv-iv01-design-v0.1-2026-10-01.md').read_text(),'prereg.md':(root/'throughline/prereg/loop-kv-iv01-v0.1-2026-10-01.md').read_text()})
    manifest={n:hashlib.sha256(t.encode()).hexdigest() for n,t in sources.items()}
    (root/'scripts/iv01_capsule.json').write_text(json.dumps(manifest,indent=2)+'\n')
    data=zlib.compress(json.dumps(sources,ensure_ascii=False,separators=(',',':')).encode(),9);encoded=base64.b64encode(data).decode()
    code='''# LOOP-KV-IV01 v0.1.1 / 2026-10-01
# Paste all into ONE fresh Colab GPU cell. Place the original DC01 ZIP in /content first.
# No Drive / HF_TOKEN / model loading. Missing ZIP opens an upload chooser. Result ZIP auto-downloads.
import os,sys,json,time,threading,subprocess,zipfile,uuid,re
from pathlib import Path
from datetime import datetime,timezone
'''
    code+='PARENT_ZIP_NAME = '+repr(idx['parent_zip'])+'\nPARENT_ZIP_SHA256 = '+repr(idx['parent_sha256'])+'\n'
    code+='PAYLOAD_SHA256 = '+repr(hashlib.sha256(data).hexdigest())+'\nEMBEDDED_SHA256 = '+repr(manifest)+'\n'
    code+='EMBEDDED_PAYLOAD = (\n'+''.join('    '+repr(encoded[i:i+160])+'\n' for i in range(0,len(encoded),160))+')\n'
    code+=sources['loop_kv_iv01_launcher.py']+'\n_loopkv_iv01_cell()\n';compile(code,'iv01_onecell','exec')
    dest=root/'colab/loop_kv_iv01_onecell.py';dest.parent.mkdir(exist_ok=True);dest.write_text(code)
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(accelerator='GPU',kernelspec=dict(name='python3',display_name='Python 3')),cells=[dict(cell_type='code',id='iv01-diagnostic',metadata={},execution_count=None,outputs=[],source=code.splitlines(keepends=True))])
    dest.with_suffix('.ipynb').write_text(json.dumps(nb,ensure_ascii=False,separators=(',',':'))+'\n')
    report=dict(python_bytes=dest.stat().st_size,python_sha256=hashlib.sha256(code.encode()).hexdigest(),embedded_files=len(sources),
        notebook_bytes=dest.with_suffix('.ipynb').stat().st_size,notebook_sha256=hashlib.sha256(dest.with_suffix('.ipynb').read_bytes()).hexdigest(),frozen_dependencies=idx['dependencies'],embedded_arrays=False)
    (root/'colab/iv01-build.json').write_text(json.dumps(report,indent=2)+'\n');return report
if __name__=='__main__':print(json.dumps(build(),indent=2))
