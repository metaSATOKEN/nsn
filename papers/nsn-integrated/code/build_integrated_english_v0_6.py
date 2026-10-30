"""Render the English translation from a fixed Japanese source and English prose.

No model calls or numerical reanalysis. Existing figures and paper-data remain shared.
Run from the research root with --research-root . --output-dir /tmp/nsn-integrated-en-check
"""
from pathlib import Path
import argparse, hashlib, json, re
SOURCE = 'papers/nsn-integrated/manuscript-ja-v0.7-2026-10-05.md'
SOURCE_SHA256 = '0adc993b78821e7672caa9d524dc83b6208a8676c1e145155b28e52cf401b9b7'

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--research-root', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    args = ap.parse_args()
    source = args.research_root / SOURCE
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError('Japanese source differs from the approved translation source')
    ja = raw.decode('utf-8')
    base = Path(__file__).resolve().parent
    template = base / 'manuscript-en-template-v0.6.md'
    table_path = base / 'table-translations-en-v0.6.json'
    text = template.read_text()
    tables = json.loads(table_path.read_text())
    math = re.findall(r'\\\[\n.*?\n\\\]', ja, re.S)
    code = re.findall(r'```[^\n]*\n.*?\n```', ja, re.S)
    for kind, mapping in [('MATH', dict(enumerate(math))), ('CODE', {i: code[i] for i in [0, 2, 3]}), ('TABLE', tables)]:
        for key, value in mapping.items():
            tag = '{{' + kind + '_' + str(key) + '}}'
            if text.count(tag) != 1:
                raise ValueError(f'Expected exactly one {tag}')
            text = text.replace(tag, value.rstrip())
    if '{{' in text or re.search(r'[\u3001\u301c\u3040-\u30ff\u3400-\u9fff]', text):
        raise ValueError('Unresolved placeholder or untranslated Japanese text')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / 'manuscript-en-v0.6-2026-10-05.md'
    output.write_text(text)
    manifest = {
        'source': SOURCE, 'source_sha256': SOURCE_SHA256,
        'template_sha256': hashlib.sha256(template.read_bytes()).hexdigest(),
        'table_translations_sha256': hashlib.sha256(table_path.read_bytes()).hexdigest(),
        'output_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
        'language': 'en', 'version': '0.6', 'shared_figures': 'figures/v0.2',
        'shared_paper_data': 'paper-data-v0.4-2026-10-05.json',
        'scope': 'Editorial revision centering quality-preserving asymmetric NSN settings and the costs of realizing execution benefits; numerical evidence and comparison boundaries preserved.'
    }
    (args.output_dir / 'english-build-manifest-v0.6.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'output': str(output), 'bytes': output.stat().st_size, 'math_blocks': len(math), 'tables': len(tables)}))

if __name__ == '__main__':
    main()
