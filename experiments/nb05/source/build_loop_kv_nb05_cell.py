"""AC08 source-capsule builder adapted to the small NB05 connectivity test."""
import base64,hashlib,json,zlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
NEW_FILES=['loop_kv_nb05_runtime.py', 'loop_kv_nb05_contract.py', 'loop_kv_nb05_primitives.py', 'loop_kv_nb05_codec.py', 'loop_kv_nb05_packed.py', 'loop_kv_nb05_evidence.py', 'loop_kv_nb05_launcher.py', 'loop_kv_nb05_kernel.py', 'loop_kv_nb05_checks.py', 'loop_kv_nb05_attention.py', 'loop_kv_nb05_numeric.py', 'loop_kv_nb05_diagnostic.py', 'loop_kv_nb05_capture.py', 'loop_kv_nb05_measure.py', 'loop_kv_nb05.py', 'verify_loop_kv_nb05.py', 'build_loop_kv_nb05_cell.py', 'loop_kv_nb05_inputs.json', 'nb05_model_files.json', 'nb05_codec_fixture.json', 'loop_kv_nb05_materials.py', 'loop_kv_nb05_candidates.json', 'nb05_sources/pg19-train-files.txt', 'nb05_sources/pg19-metadata.csv', 'nb05_approved_contract.json']
def build(root=ROOT):
    plan=json.loads((root/'scripts/loop_kv_nb05_inputs.json').read_text())
    assert (root/'scripts/nb05_approved_contract.json').read_bytes()==(root/'plans/loop-kv-nb05-input-contract-v0.1-2026-10-03.json').read_bytes()
    assert hashlib.sha256((root/'scripts/nb05_approved_contract.json').read_bytes()).hexdigest()==plan['design_contract_sha256']
    sources={n:(root/'scripts'/n).read_text() for n in NEW_FILES}
    for name,digest in plan['public_source_sha256'].items():
        body=(root/'scripts/nb05_model_source'/name).read_text()
        assert hashlib.sha256(body.encode()).hexdigest()==digest,name
        sources['nb05_model_source/'+name]=body
    sources.update({'design.md':(root/'plans/loop-kv-nb05-design-v0.1-2026-10-03.md').read_text(),
        'prereg.md':(root/'throughline/prereg/loop-kv-nb05-v0.1-2026-10-03.md').read_text()})
    manifest={n:hashlib.sha256(text.encode()).hexdigest() for n,text in sources.items()}
    (root/'scripts/nb05_capsule.json').write_text(json.dumps(manifest,indent=2)+'\n')
    data=zlib.compress(json.dumps(sources,ensure_ascii=False,separators=(',',':')).encode(),9);encoded=base64.b64encode(data).decode()
    code='''# LOOP-KV-NB05 v0.1.0 / 2026-10-03
# Paste this entire file into one fresh Colab GPU cell. HF_TOKEN; no Drive or prior ZIP.
# Separate cu128 runtime; CPython 3.13/Linux x86_64. No model loading on the local Mac.
# Nanbeige-3B-Base: 540 model forwards over 4 new passages; actual-Q GQA fusion, 3 main timing rounds, evidence <=384 MiB, ZIP <=256 MiB.
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
    code+=sources['loop_kv_nb05_launcher.py']+'\n_loopkv_nb05_cell()\n'
    compile(code,'nb05_onecell','exec');dest=root/'colab/loop_kv_nb05_onecell.py';dest.parent.mkdir(exist_ok=True);dest.write_text(code)
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(accelerator='GPU',kernelspec=dict(name='python3',display_name='Python 3')),
        cells=[dict(cell_type='code',id='nb05-connection',metadata={},execution_count=None,outputs=[],source=code.splitlines(keepends=True))])
    dest.with_suffix('.ipynb').write_text(json.dumps(nb,ensure_ascii=False,separators=(',',':'))+'\n')
    report=dict(python_bytes=dest.stat().st_size,python_sha256=hashlib.sha256(code.encode()).hexdigest(),embedded_files=len(sources),
        notebook_bytes=dest.with_suffix('.ipynb').stat().st_size,notebook_sha256=hashlib.sha256(dest.with_suffix('.ipynb').read_bytes()).hexdigest(),gpu_execution_verified=False)
    (root/'colab/nb05-build.json').write_text(json.dumps(report,indent=2)+'\n');return report
if __name__=='__main__':print(json.dumps(build(),indent=2))
