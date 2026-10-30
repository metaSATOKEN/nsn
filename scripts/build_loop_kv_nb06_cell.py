"""AC08 source-capsule builder adapted to the small NB06 connectivity test."""
import base64,hashlib,json,zlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
NEW_FILES=['loop_kv_nb06_runtime.py', 'loop_kv_nb06_contract.py', 'loop_kv_nb06_primitives.py', 'loop_kv_nb06_codec.py', 'loop_kv_nb06_packed.py', 'loop_kv_nb06_evidence.py', 'loop_kv_nb06_launcher.py', 'loop_kv_nb06_kernel.py', 'loop_kv_nb06_checks.py', 'loop_kv_nb06_attention.py', 'loop_kv_nb06_numeric.py', 'loop_kv_nb06_diagnostic.py', 'loop_kv_nb06_capture.py', 'loop_kv_nb06_measure.py', 'loop_kv_nb06.py', 'verify_loop_kv_nb06.py', 'build_loop_kv_nb06_cell.py', 'loop_kv_nb06_inputs.json', 'nb06_model_files.json', 'nb06_codec_fixture.json', 'loop_kv_nb06_materials.py', 'nb06_previous_materials.json', 'nb06_approved_contract.json']
def build(root=ROOT):
    plan=json.loads((root/'scripts/loop_kv_nb06_inputs.json').read_text())
    assert (root/'scripts/nb06_approved_contract.json').read_bytes()==(root/'plans/loop-kv-nb06-input-contract-v0.1-2026-10-08.json').read_bytes()
    assert hashlib.sha256((root/'scripts/nb06_approved_contract.json').read_bytes()).hexdigest()==plan['design_contract_sha256']
    sources={n:(root/'scripts'/n).read_text() for n in NEW_FILES}
    for name,digest in plan['public_source_sha256'].items():
        body=(root/'scripts/nb06_model_source'/name).read_text()
        assert hashlib.sha256(body.encode()).hexdigest()==digest,name
        sources['nb06_model_source/'+name]=body
    sources.update({'design.md':(root/'plans/loop-kv-nb06-design-v0.1-2026-10-08.md').read_text(),
        'prereg.md':(root/'throughline/prereg/loop-kv-nb06-v0.1-2026-10-08.md').read_text()})
    manifest={n:hashlib.sha256(text.encode()).hexdigest() for n,text in sources.items()}
    (root/'scripts/nb06_capsule.json').write_text(json.dumps(manifest,indent=2)+'\n')
    data=zlib.compress(json.dumps(sources,ensure_ascii=False,separators=(',',':')).encode(),9);encoded=base64.b64encode(data).decode()
    code='''# LOOP-KV-NB06 v0.1.0 / 2026-10-08
# Paste this entire file into one fresh Colab GPU cell. HF_TOKEN; no Drive or prior ZIP.
# Separate cu128 runtime; CPython 3.13/Linux x86_64. No model loading on the local Mac.
# Nanbeige-3B-Base: 15420 model forwards over 4 fixed extended passages; actual-Q GQA fusion, 3 main timing rounds, evidence <=768 MiB, ZIP <=512 MiB.
import os,sys,json,time,threading,subprocess,zipfile,uuid,re
from pathlib import Path
from datetime import datetime,timezone
HF_SECRET_NAME = "HF_TOKEN"
'''
    code+='PAYLOAD_SHA256 = '+repr(hashlib.sha256(data).hexdigest())+'\nEMBEDDED_SHA256 = '+repr(manifest)+'\n'
    for name,value in {'EXPECTED_CAPS':plan['phase_cap_seconds'],'OUTPUT_LIMIT_BYTES':plan['evidence_limit_bytes'],
            'ZIP_LIMIT_BYTES':plan['zip_limit_bytes'],'INITIAL_FREE_BYTES':plan['minimum_initial_free_bytes']}.items():
        code+=name+' = '+repr(value)+'\n'
    code+='EMBEDDED_PAYLOAD = (\n'+''.join('    '+repr(encoded[i:i+160])+'\n' for i in range(0,len(encoded),160))+')\n'
    code+=sources['loop_kv_nb06_launcher.py']+'\n_loopkv_nb06_cell()\n'
    compile(code,'nb06_onecell','exec');dest=root/'colab/loop_kv_nb06_onecell.py';dest.parent.mkdir(exist_ok=True);dest.write_text(code)
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(accelerator='GPU',kernelspec=dict(name='python3',display_name='Python 3')),
        cells=[dict(cell_type='code',id='nb06-connection',metadata={},execution_count=None,outputs=[],source=code.splitlines(keepends=True))])
    dest.with_suffix('.ipynb').write_text(json.dumps(nb,ensure_ascii=False,separators=(',',':'))+'\n')
    report=dict(python_bytes=dest.stat().st_size,python_sha256=hashlib.sha256(code.encode()).hexdigest(),embedded_files=len(sources),
        notebook_bytes=dest.with_suffix('.ipynb').stat().st_size,notebook_sha256=hashlib.sha256(dest.with_suffix('.ipynb').read_bytes()).hexdigest(),gpu_execution_verified=False)
    (root/'colab/nb06-build.json').write_text(json.dumps(report,indent=2)+'\n');return report
if __name__=='__main__':print(json.dumps(build(),indent=2))
