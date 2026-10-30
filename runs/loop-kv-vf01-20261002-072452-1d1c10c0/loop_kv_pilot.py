# LOOP-KV-PILOT-01 / version 0.1 / Kentaro Sato / 2026-09-30
# Colab worker. Importing this file does not install packages or load a model.
import os
import sys
import json
import math
import time
import csv
import io
import re
import hashlib
import random
import signal
import shutil
import zipfile
import struct
import traceback
import subprocess
import importlib
import importlib.metadata
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone

VERSION = '0.1'
MODEL = 'ByteDance/Ouro-1.4B'
MODEL_REV = 'fcfcfbc429c7b10ee73dfc718033351d8936d2a8'
DATA_REV = '4d28bd77e66947ad3835cf78ed7aaeb4dd87ad8b'
TRAIN_SHA = '145b5f0896cc2fff3231222874a0f324ee04559144ecf25a873d294eb161719c'
META_SHA = 'fbb2fdb48522927b2e16aa52950f2afeb83c6fa8fed45f0c3dd834e9bc9b43b9'
SOURCE_HASHES = {
    'modeling_ouro.py': 'c5c68fbb368ce2909c257ae2afc50719be8c91539333d3295e19312c4316f413',
    'configuration_ouro.py': '950443e32929047aa08d02abad2e1888bc1914b3db988d3d675f70787f65dafb',
    'config.json': '474ea0d5e14ad62aee3245771cedf132a6920c311ae2dc12a8b2271cd868d242',
}
SEED = 2026093001
BOOT_SEED = 2026093002
PREFIX = 768
SCORED = 255
BLOCK = 64
LAYERS = 24
HEADS = 16
DIM = 128
EPS = 1e-12
PACKAGE_SPECS = ['transformers==4.56.2', 'huggingface-hub>=0.34.0,<1.0',
                 'accelerate>=0.26.0,<2', 'safetensors>=0.4.3', 'sentencepiece']
PACKAGE_NAMES = ['torch', 'transformers', 'huggingface-hub', 'accelerate',
                 'tokenizers', 'safetensors', 'numpy', 'sentencepiece']


class IntegrityError(RuntimeError):
    pass


class NumericalFailure(RuntimeError):
    pass


class BudgetStop(RuntimeError):
    pass


def sha(data):
    return hashlib.sha256(data).hexdigest()


def token_sha(ids):
    return sha(json.dumps(ids, separators=(',', ':')).encode())


def redact(text):
    text = str(text)
    secret = os.environ.get('HF_TOKEN', '')
    if secret:
        text = text.replace(secret, '[REDACTED]')
    return re.sub(r'hf_[A-Za-z0-9]{20,}', '[REDACTED]', text)


def require(ok, message):
    if not ok:
        raise IntegrityError(message)


def finite_json(obj):
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {str(k): finite_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [finite_json(v) for v in obj]
    return obj


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(finite_json(obj), ensure_ascii=False, indent=2,
                              allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def append_json(path, obj):
    with Path(path).open('a', encoding='utf-8') as f:
        f.write(json.dumps(finite_json(obj), ensure_ascii=False, allow_nan=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


def read_rows(path):
    if not Path(path).exists():
        return []
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def checkpoint_zip(root):
    # Called synchronously between trajectories, with no concurrent artifact writer.
    files = [p for p in root.rglob('*') if p.is_file() and p.name != 'SHA256SUMS.json' and not p.name.endswith('.tmp')]
    write_json(root/'SHA256SUMS.json', {str(p.relative_to(root)):sha(p.read_bytes()) for p in sorted(files)})
    target = root.parent/'checkpoint.zip'
    tmp = target.with_suffix('.tmp')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in sorted(root.rglob('*')):
            if p.is_file() and not p.name.endswith('.tmp'):
                z.write(p,str(p.relative_to(root)))
    os.replace(tmp,target)


def log(text):
    print(redact(text), flush=True)


def numeric_imports():
    global torch, np
    import torch
    import numpy as np


def safe_ppl(nll):
    return math.exp(nll) if math.isfinite(nll) and nll < 709 else None


def sample_order(files):
    return sorted(files, key=lambda x: sha(f'{SEED}:{Path(x).stem}'.encode()))


def schedule(part, docs):
    rng = random.Random(SEED)
    docs = list(docs)
    rng.shuffle(docs)
    rows = []
    for doc in docs:
        rs = [4] if part == 'P0_P1A' else [1, 2, 8]
        rng.shuffle(rs)
        for r in rs:
            modes = ['PREFIX_ONLY', 'DECODE_ONLINE'] if part == 'P0_P1A' else ['PREFIX_ONLY']
            rng.shuffle(modes)
            for mode in modes:
                bits = [3, 4, 5]
                rng.shuffle(bits)
                for b in [None] + bits:
                    ident = f"{doc['book_id']}-r{r}-{mode}-{'native' if b is None else 'nsn'+str(b)}"
                    rows.append({'id': ident, 'book_id': doc['book_id'], 'r': r,
                                 'mode': mode, 'bits': b, 'expected_scored': SCORED})
    require(len(rows) == (64 if part == 'P0_P1A' else 96), 'Wrong schedule length')
    require(len({x['id'] for x in rows}) == len(rows), 'Duplicate condition IDs')
    return rows


# Independent little-endian packed layout. P1 uses decoded floating KV.
def pack_codes(codes, bits):
    a = np.asarray(codes, dtype=np.uint8).reshape(-1)
    require(bits in (3, 4, 5) and np.all(a < (1 << bits)), 'Invalid packed code')
    pos = np.arange(a.size, dtype=np.int64) * bits
    out = np.zeros((a.size * bits + 7) // 8, dtype=np.uint8)
    val = a.astype(np.uint16) << (pos % 8)
    np.bitwise_or.at(out, pos // 8, (val & 255).astype(np.uint8))
    crossing = pos % 8 + bits > 8
    np.bitwise_or.at(out, pos[crossing] // 8 + 1, (val[crossing] >> 8).astype(np.uint8))
    return out.tobytes()


def unpack_codes_reference(payload, count, bits):
    # Deliberately scalar and separate from vectorized pack.
    out = np.empty(count, dtype=np.uint8)
    for i in range(count):
        pos = i * bits
        val = int(payload[pos // 8]) >> (pos % 8)
        if pos % 8 + bits > 8:
            val |= int(payload[pos // 8 + 1]) << (8 - pos % 8)
        out[i] = val & ((1 << bits) - 1)
    return out


def nsn_encode(x, bits):
    require(x.ndim >= 2 and x.shape[-2:] == (BLOCK, DIM), 'NSN block shape')
    a = x.float()
    if not bool(torch.isfinite(a).all()):
        raise NumericalFailure('Nonfinite input to NSN')
    n = torch.linalg.vector_norm(a, dim=-1, keepdim=True)
    d = a / n.clamp_min(EPS)
    m = d.amin(dim=-2, keepdim=True)
    mx = d.amax(dim=-2, keepdim=True)
    s = (mx - m) / ((1 << bits) - 1)
    denom = torch.where(s == 0, torch.ones_like(s), s)
    q = torch.round((d - m) / denom).clamp(0, (1 << bits) - 1).to(torch.uint8)
    q = torch.where(s == 0, torch.zeros_like(q), q)
    metas = tuple(t.to(torch.float16) for t in (n, m, s))
    if not all(bool(torch.isfinite(t).all()) for t in metas):
        raise NumericalFailure('FP16 metadata overflow/nonfinite')
    if bool(((n != 0) & (metas[0] == 0)).any()) or bool(((s != 0) & (metas[2] == 0)).any()):
        raise NumericalFailure('Nonzero FP16 norm/step underflow')
    return q, metas


def nsn_decode(q, metas, dtype):
    # Canonical contiguous layout makes the arithmetic independent of storage
    # strides after packing/unpacking, including non-contiguous input fixtures.
    n, m, s = (t.float().contiguous() for t in metas)
    d = (m + s * q.float().contiguous()).contiguous()
    y = n * (d / torch.linalg.vector_norm(d, dim=-1, keepdim=True).clamp_min(EPS))
    y = torch.where(n == 0, torch.zeros_like(y), y).to(dtype)
    if not bool(torch.isfinite(y).all()):
        raise NumericalFailure('Nonfinite NSN reconstruction')
    return y


def nsn_qdq(x, bits):
    q, meta = nsn_encode(x, bits)
    return nsn_decode(q, meta, x.dtype)


def planned_bytes(shape, compressed_tokens, bits):
    b, h, length, d = shape
    require(compressed_tokens % BLOCK == 0 and 0 <= compressed_tokens <= length, 'Block accounting')
    if bits is None:
        return {'codes': 0, 'metadata': 0, 'native': b*h*length*d*2, 'block_headers': 0}
    blocks = b*h*(compressed_tokens // BLOCK)
    # Header per head-block: token offset uint64, count uint16, bits uint8,
    # reserved uint8. Full blocks need no bitstream byte padding.
    return {'codes': blocks*((BLOCK*d*bits + 7)//8),
            'metadata': blocks*(BLOCK*2 + d*4),
            'native': b*h*(length-compressed_tokens)*d*2, 'block_headers': blocks*12}


def causal_mask(positions, total, dtype, device):
    q = positions.reshape(-1, 1)
    k = torch.arange(total, device=device).reshape(1, -1)
    mask = torch.zeros((len(positions), total), dtype=dtype, device=device)
    mask.masked_fill_(k > q, torch.finfo(dtype).min)
    return {'full_attention': mask[None, None, :, :]}


def make_adapter_class(base):
    class StudyCache(base):
        def __init__(self, r, bits=None, mode='PREFIX_ONLY', trace=False):
            super().__init__(max_cache_size=LAYERS*r)
            self.r, self.bits, self.mode, self.trace = r, bits, mode, trace
            self.calls = Counter()
            self.records = []
            self.compressed = {}
            self.encoded_blocks = Counter()

        def update(self, key, value, layer_idx, cache_kwargs=None):
            require(0 <= layer_idx < LAYERS*self.r, 'Unexpected cache slot')
            require(key.shape == value.shape and tuple(key.shape[:2]) == (1, HEADS), 'KV dimensions')
            require(key.shape[-1] == DIM, 'KV head dimension')
            old = self.get_seq_length(layer_idx)
            pos = (cache_kwargs or {}).get('cache_position')
            if pos is not None:
                want = torch.arange(old, old + key.shape[-2], device=pos.device)
                require(torch.equal(pos, want), 'Position mismatch / duplicate append')
            result = super().update(key, value, layer_idx, cache_kwargs)
            require(self.get_seq_length(layer_idx) == old + key.shape[-2], 'Append length mismatch')
            self.calls[layer_idx] += 1
            if self.trace:
                self.records.append({'slot': layer_idx, 'loop': layer_idx//LAYERS,
                                     'layer': layer_idx%LAYERS, 'old': old, 'added': key.shape[-2],
                                     'new': self.get_seq_length(layer_idx)})
            return result

        def compress_to(self, target):
            if self.bits is None:
                return
            for slot in range(LAYERS*self.r):
                require(self.get_seq_length(slot) >= target + BLOCK, 'Insufficient protected tail')
                start = self.compressed.get(slot, 0)
                require(target >= start and (target-start) % BLOCK == 0, 'Requantization / block overlap')
                if target == start:
                    continue
                for data in (self.key_cache, self.value_cache):
                    original = data[slot]
                    nblocks = (target-start)//BLOCK
                    group = original[:, :, start:target, :].reshape(1, HEADS, nblocks, BLOCK, DIM)
                    restored = nsn_qdq(group, self.bits).reshape(1, HEADS, target-start, DIM)
                    data[slot] = torch.cat((original[:, :, :start, :], restored,
                                            original[:, :, target:, :]), dim=-2)
                self.encoded_blocks[slot] += (target-start)//BLOCK
                self.compressed[slot] = target

        def post_prefill(self):
            length = self.get_seq_length()
            target = max(0, ((length-BLOCK)//BLOCK)*BLOCK)
            self.compress_to(target)

        def post_decode(self):
            if self.bits is None or self.mode != 'DECODE_ONLINE':
                return
            length = self.get_seq_length()
            current = self.compressed.get(0, 0)
            if length-current >= 2*BLOCK:
                self.compress_to(current+BLOCK)

        def validate(self, expected_calls, length):
            require(set(self.calls) == set(range(LAYERS*self.r)), 'Missing/extra slot')
            require(all(v == expected_calls for v in self.calls.values()), 'Wrong update count')
            require(all(self.get_seq_length(s) == length for s in self.calls), 'Wrong cache length')
            if self.bits is not None:
                target = 704 if self.mode == 'PREFIX_ONLY' else 896
                require(all(self.compressed.get(s) == target for s in self.calls), 'Wrong final compressed length')
                require(all(self.encoded_blocks[s] == target//BLOCK for s in self.calls), 'Duplicate block encoding')

        def accounting(self):
            components = Counter()
            allocated = 0
            for slot in range(LAYERS*self.r):
                for data in (self.key_cache, self.value_cache):
                    t = data[slot]
                    components.update(planned_bytes(t.shape, self.compressed.get(slot, 0), self.bits))
                    allocated += t.numel()*t.element_size()
            return {'planned_components': dict(components), 'planned_bytes': sum(components.values()),
                    'floating_cache_bytes': allocated, 'implementation': 'FAKE_QUANT_FLOAT_CACHE',
                    'update_count': sum(self.calls.values()), 'slots': len(self.calls),
                    'compressed_tokens': self.compressed.get(0, 0)}
    return StudyCache


def tensor_snapshot(cache):
    return [(cache.key_cache[i].detach().cpu().clone(), cache.value_cache[i].detach().cpu().clone())
            for i in range(len(cache.key_cache))]


def snapshots_equal(a, b):
    return len(a) == len(b) and all(torch.equal(x[0], y[0]) and torch.equal(x[1], y[1])
                                  for x, y in zip(a, b))


def scalar_metrics(full, incremental, tokens):
    a = torch.log_softmax(full.float(), dim=-1)
    b = torch.log_softmax(incremental.float(), dim=-1)
    require(a.shape == b.shape and len(tokens) == a.shape[0], 'P0 logit shape')
    target = torch.as_tensor(tokens[1:], dtype=torch.long)
    idx = torch.arange(len(target))
    nlla = -a[:-1][idx, target].mean()
    nllb = -b[:-1][idx, target].mean()
    return {'kl': float((a.exp()*(a-b)).sum(-1).mean()),
            'delta_nll': float(nllb-nlla),
            'max_abs_logit': float((full-incremental).abs().max())}


def metrics_pass(m, fp32=False):
    limits = (1e-6, 1e-4, 1e-3) if fp32 else (1e-4, 0.005, 0.25)
    return all(math.isfinite(v) for v in m.values()) and m['kl'] <= limits[0] and abs(m['delta_nll']) <= limits[1] and m['max_abs_logit'] <= limits[2]


def cpu_fixtures():
    numeric_imports()
    g = torch.Generator(device='cpu').manual_seed(SEED)
    result = []
    for bits in (3, 4, 5):
        for count in (1, 7, 63, 64, 65, 127, 128, 129, 8192):
            codes = np.random.default_rng(SEED+count+bits).integers(0, 1 << bits, count, dtype=np.uint8)
            packed = pack_codes(codes, bits)
            require(len(packed) == (count*bits+7)//8, 'Packed byte count')
            require(np.array_equal(codes, unpack_codes_reference(packed, count, bits)), 'Packed roundtrip')
        tensors = [torch.zeros(BLOCK, DIM), torch.ones(BLOCK, DIM),
                   torch.randn(DIM, BLOCK, generator=g).T,
                   torch.randn(2, BLOCK, DIM, generator=g)]
        for t in tensors:
            q, meta = nsn_encode(t, bits)
            q2 = torch.from_numpy(unpack_codes_reference(pack_codes(q.numpy(), bits), q.numel(), bits).copy()).reshape(q.shape)
            meta2 = tuple(torch.from_numpy(np.frombuffer(m.numpy().astype('<f2').tobytes(), dtype='<f2').copy()).reshape(m.shape) for m in meta)
            require(all(torch.equal(a,b) for a,b in zip(meta,meta2)), 'FP16 metadata roundtrip')
            require(torch.equal(nsn_decode(q, meta, t.dtype), nsn_decode(q2, meta2, t.dtype)), 'Decoded roundtrip')
        header=struct.pack('<QHBB',0,BLOCK,bits,0)
        q,meta=nsn_encode(tensors[1],bits)
        blob=header+pack_codes(q.numpy(),bits)+b''.join(m.numpy().astype('<f2').tobytes() for m in meta)
        expected=planned_bytes((1,1,BLOCK,DIM),BLOCK,bits)
        require(len(blob)==sum(expected.values()) and struct.unpack('<QHBB',blob[:12])==(0,BLOCK,bits,0),'Actual packed block/header bytes')
        result.append({'test': 'codec', 'bits': bits, 'status': 'PASS'})
    try:
        nsn_encode(torch.full((BLOCK,DIM), 1e5), 3)
        raise IntegrityError('Overflow fixture was not detected')
    except NumericalFailure:
        result.append({'test': 'metadata_overflow_detection', 'status': 'PASS'})
    for length in (1,63,64,65,127,128,129):
        p = torch.arange(length)
        m = causal_mask(p,length,torch.float32,'cpu')['full_attention'][0,0]
        for i in range(length):
            require(torch.all(m[i,:i+1] == 0) and torch.all(m[i,i+1:] < 0), 'Causal mask wrong')
    logits = torch.tensor([[0., math.log(3.)], [math.log(3.), 0.]])
    lp = torch.log_softmax(logits, -1)
    actual = float(-lp[torch.arange(2),torch.tensor([1,0])].mean())
    require(abs(actual-math.log(4/3)) < 1e-6, 'Manual scorer wrong')
    require(planned_bytes((1,1,64,128),64,4) == {'codes':4096,'metadata':640,'native':0,'block_headers':12}, 'Layout bytes wrong')
    result.append({'test': 'causal_mask_and_independent_arithmetic', 'status': 'PASS'})
    return result


def gpu_codec_fixtures():
    g=torch.Generator(device='cpu').manual_seed(SEED)
    samples=[torch.zeros(BLOCK,DIM),torch.ones(BLOCK,DIM),torch.randn(DIM,BLOCK,generator=g).T]
    checks=[]
    for bits in (3,4,5):
        for source in samples:
            x=source.to('cuda')
            q,meta=nsn_encode(x,bits)
            payload=pack_codes(q.cpu().numpy(),bits)
            packed_q=torch.from_numpy(unpack_codes_reference(payload,q.numel(),bits).copy()).reshape(q.shape).to('cuda')
            packed_meta=tuple(torch.from_numpy(np.frombuffer(m.cpu().numpy().astype('<f2').tobytes(),dtype='<f2').copy()).reshape(m.shape).to('cuda') for m in meta)
            require(torch.equal(nsn_decode(q,meta,x.dtype),nsn_decode(packed_q,packed_meta,x.dtype)),'GPU packed/fake codec mismatch')
        checks.append({'test':'gpu_codec_roundtrip','bits':bits,'status':'PASS'})
    return checks


def download_public(url, cap=16*1024*1024, timeout=90):
    import urllib.request
    with urllib.request.urlopen(url, timeout=timeout) as r:
        data = r.read(cap+1)
    require(len(data) <= cap, 'Public download exceeded byte cap')
    return data


def install_packages(parent):
    specs = PACKAGE_SPECS
    if parent:
        specs = [f'{n}=={v}' for n,v in parent['packages'].items() if n != 'torch' and v]
    log('Installing isolated worker dependencies; keeping Colab PyTorch.')
    p = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *specs],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=3600)
    log(p.stdout)
    require(p.returncode == 0, 'Dependency installation failed')


def materials(root, tokenizer, parent_materials=None):
    if parent_materials is not None:
        require(len(parent_materials['smoke']) == 4 and len(parent_materials['diagnostic']) == 8, 'Parent material count')
        for d in parent_materials['smoke']+parent_materials['diagnostic']:
            require(len(d['tokens']) == 1024 and token_sha(d['tokens']) == d['token_sha256'], 'Parent token hash')
        write_json(root/'materials.json', parent_materials)
        return parent_materials
    source = root/'sources'
    source.mkdir(exist_ok=True)
    train = download_public(f'https://huggingface.co/datasets/deepmind/pg19/resolve/{DATA_REV}/data/train_files.txt')
    require(sha(train) == TRAIN_SHA, 'PG19 train manifest changed')
    metadata = download_public('https://storage.googleapis.com/deepmind-gutenberg/metadata.csv')
    require(sha(metadata) == META_SHA, 'PG19 metadata changed')
    (source/'pg19-train-files.txt').write_bytes(train)
    (source/'pg19-metadata.csv').write_bytes(metadata)
    table = {r[0]: r for r in csv.reader(io.StringIO(metadata.decode('utf-8'))) if len(r)>=4}
    paths = train.decode().splitlines()
    require(len(paths) == 28602 and all(re.fullmatch(r'train/[0-9]+\.txt', p) for p in paths), 'Train split manifest invalid')
    used_hash, used_title, selected = set(), set(), []
    for attempt, file in enumerate(sample_order(paths)):
        require(attempt < 256, 'Material selection exceeded 256 candidates')
        book = Path(file).stem
        text = download_public('https://storage.googleapis.com/deepmind-gutenberg/'+file)
        text_sha = sha(text)
        title = re.sub(r'\s+', ' ', table[book][1].strip().lower())
        if text_sha in used_hash or title in used_title:
            append_json(root/'selection.jsonl', {'book_id':book,'status':'DUPLICATE','text_sha256':text_sha})
            continue
        ids = tokenizer(text.decode('utf-8'), add_special_tokens=False, truncation=True, max_length=1024)['input_ids']
        if len(ids) < 1024:
            append_json(root/'selection.jsonl', {'book_id':book,'status':'TOO_SHORT','tokens':len(ids)})
            continue
        used_hash.add(text_sha)
        used_title.add(title)
        d = {'book_id':book,'split':'train','file':file,'title':table[book][1],
             'publication_date':table[book][2],'text_sha256':text_sha,'offset':0,
             'token_sha256':token_sha(ids),'tokens':ids,'rank_attempt':attempt}
        selected.append(d)
        append_json(root/'selection.jsonl', {'book_id':book,'status':'SELECTED','token_sha256':d['token_sha256']})
        log(f'Materials {len(selected)}/12: book {book}')
        if len(selected) == 12:
            break
    require(len(selected) == 12, 'Insufficient materials')
    result = {'source':'PG19 official asset URLs / official HF split manifest', 'data_revision':DATA_REV,
              'selection_seed':SEED,'tokenizer_revision':MODEL_REV,
              'smoke':selected[:4], 'diagnostic':selected[4:]}
    write_json(root/'materials.json', result)
    return result


class Engine:
    def __init__(self, model, root, adapter_class, base_class):
        self.model, self.root = model, root
        self.adapter_class, self.base_class = adapter_class, base_class
        self.stage_deadline = math.inf
        self.condition_deadline = math.inf

    def check_time(self):
        now = time.monotonic()
        if now > self.stage_deadline:
            raise BudgetStop('Stage time cap reached')
        if now > self.condition_deadline:
            raise TimeoutError('Condition exceeded 15 minutes')

    def set_r(self, r):
        self.model.config.total_ut_steps = r
        self.model.model.config.total_ut_steps = r
        self.model.model.total_ut_steps = r
        require(self.model.model.total_ut_steps == self.model.config.total_ut_steps, 'Loop setting mismatch')

    def forward(self, ids, cache, r, keep=1):
        self.check_time()
        old = cache.get_seq_length() if cache is not None else 0
        inp = torch.tensor([ids], dtype=torch.long, device='cuda')
        pos = torch.arange(old, old+len(ids), device='cuda')
        mask = causal_mask(pos, old+len(ids), self.model.dtype, 'cuda')
        with torch.inference_mode():
            output = self.model(input_ids=inp, past_key_values=cache, use_cache=cache is not None,
                                cache_position=pos, position_ids=pos[None,:], attention_mask=mask,
                                exit_at_step=r-1, labels=None, use_weighted_exit=False,
                                logits_to_keep=keep, return_dict=True)
        if cache is not None:
            require(output.past_key_values is cache, 'Cache adapter was replaced')
        return output.logits.detach()

    def smoke_run(self, tokens, r, kind, trace=False):
        self.set_r(r)
        self.condition_deadline = time.monotonic()+900
        cache = self.base_class(max_cache_size=LAYERS*r) if kind == 'official' else self.adapter_class(r, trace=trace)
        start = time.monotonic()
        rng0 = (torch.random.get_rng_state().clone(), torch.cuda.get_rng_state().clone())
        parts = [self.forward(tokens[:64], cache, r, keep=0)[0].cpu()]
        for t in tokens[64:]:
            parts.append(self.forward([t], cache, r)[0].cpu())
        logits = torch.cat(parts)
        require(torch.equal(rng0[0],torch.random.get_rng_state()) and torch.equal(rng0[1],torch.cuda.get_rng_state()), 'Inference changed RNG')
        if kind != 'official':
            require(set(cache.calls) == set(range(LAYERS*r)), 'Smoke missing slot')
            require(all(v == 66 for v in cache.calls.values()), 'Smoke call count')
            require(all(cache.get_seq_length(s) == 129 for s in cache.calls), 'Smoke length')
        snap = tensor_snapshot(cache)
        records = list(getattr(cache,'records',[]))
        del cache
        return logits, snap, time.monotonic()-start, records

    def p0(self, smoke, revalidation=False):
        checks = cpu_fixtures()+gpu_codec_fixtures()
        write_json(self.root/'p0_checks.json', {'status':'RUNNING','checks':checks})
        timings = []
        try:
            for doc in smoke:
                tokens = doc['tokens'][:129]
                for r in (1,2,4,8):
                    log(f"P0 native fidelity: book {doc['book_id']} / r={r}")
                    official, orig_cache, elapsed, _ = self.smoke_run(tokens,r,'official')
                    timings.append({'r':r,'seconds':elapsed})
                    for route, trace in [('official',False),('adapter',False),('adapter',True)]:
                        got, got_cache, _, records = self.smoke_run(tokens,r,route,trace)
                        require(torch.equal(official,got), f'P0 logit mismatch r={r} {route}/{trace}')
                        require(snapshots_equal(orig_cache,got_cache), f'P0 KV mismatch r={r} {route}/{trace}')
                        if records:
                            append_json(self.root/'p0_cache_trace.jsonl', {'book_id':doc['book_id'],'r':r,'records':records})
                        del got, got_cache
                    self.condition_deadline = time.monotonic()+900
                    full = self.forward(tokens,None,r,keep=0)[0].cpu()
                    m = scalar_metrics(full, official, tokens)
                    append_json(self.root/'p0_full_incremental.jsonl', {'book_id':doc['book_id'],'r':r,'dtype':'BF16',**m})
                    if not metrics_pass(m):
                        log('BF16 full/incremental tolerance failed; running FP32 diagnosis, then stopping.')
                        self.model.to(dtype=torch.float32)
                        try:
                            full32 = self.forward(tokens,None,r,keep=0)[0].cpu()
                            inc32, _, _, _ = self.smoke_run(tokens,r,'official')
                            m32 = scalar_metrics(full32,inc32,tokens)
                            append_json(self.root/'p0_full_incremental.jsonl', {'book_id':doc['book_id'],'r':r,'dtype':'FP32','passed':metrics_pass(m32,True),**m32})
                        finally:
                            self.model.to(dtype=torch.bfloat16)
                        raise IntegrityError('BF16 full/incremental fidelity failed; P1 not started')
                    checks.append({'test':'fidelity','book_id':doc['book_id'],'r':r,'status':'PASS',**m})
                    write_json(self.root/'p0_checks.json', {'status':'RUNNING','checks':checks})
                    del official, orig_cache, full
            # Model boundary checks for each specified synthetic length.
            self.set_r(4)
            for length in (1,63,64,65,127,128,129):
                self.condition_deadline = time.monotonic()+900
                ids = [(7+i*13) % self.model.config.vocab_size for i in range(length)]
                a = self.base_class(max_cache_size=LAYERS*4)
                b = self.adapter_class(4,trace=True)
                x = self.forward(ids,a,4,keep=0)
                y = self.forward(ids,b,4,keep=0)
                require(torch.equal(x,y) and snapshots_equal(tensor_snapshot(a),tensor_snapshot(b)), 'Boundary cache mismatch')
                require(all(v == 1 for v in b.calls.values()) and len(b.calls)==96, 'Boundary hook count')
                checks.append({'test':'model_boundary','length':length,'status':'PASS'})
                del a,b,x,y
            write_json(self.root/'p0_checks.json', {'status':'PASS','checks':checks,'native_timings':timings,'revalidation':revalidation})
            return timings
        except BaseException as e:
            write_json(self.root/'p0_checks.json', {'status':'BLOCKED_IMPLEMENTATION','checks':checks,
                                                  'exception':redact(traceback.format_exc()),'native_timings':timings})
            raise

    def trajectory(self, doc, spec, baseline=None):
        self.set_r(spec['r'])
        self.condition_deadline = time.monotonic()+900
        cache = self.adapter_class(spec['r'],spec['bits'],spec['mode'])
        tokens = doc['tokens']
        rows, native_lp = [], []
        start = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        try:
            self.forward(tokens[:PREFIX],cache,spec['r'])
            cache.post_prefill()
            for j in range(SCORED):
                logits = self.forward([tokens[PREFIX+j]],cache,spec['r'])[0,0].float()
                lp = torch.log_softmax(logits,dim=-1).cpu().numpy()
                if not bool(np.isfinite(lp).all()):
                    raise NumericalFailure('Nonfinite log probability')
                target = tokens[PREFIX+j+1]
                target_lp = float(lp[target])
                kl = None
                ref_lp = None
                if baseline is not None:
                    ref = baseline[j]
                    ref_lp = float(ref[target])
                    kl = float(np.sum(np.exp(ref)*(ref-lp),dtype=np.float32))
                    if not math.isfinite(kl):
                        raise NumericalFailure('Nonfinite KL')
                row = {'condition_id':spec['id'],'book_id':doc['book_id'],'r':spec['r'],
                       'mode':spec['mode'],'bits':spec['bits'],'input_position':PREFIX+j,
                       'target_position':PREFIX+j+1,'target_id':target,
                       'target_log_probability':target_lp,'native_target_log_probability':ref_lp,
                       'kl_native_to_condition':kl,'status':'FINITE'}
                rows.append(row)
                append_json(self.root/'per_token.jsonl',row)
                if spec['bits'] is None:
                    native_lp.append(lp)
                # Fixed complete distribution for independent scorer checks.
                if j == 127:
                    ref_saved = baseline[j] if baseline is not None else (lp if spec['bits'] is None else np.empty(0,dtype=np.float32))
                    np.savez_compressed(self.root/'diagnostics'/f"{spec['id']}-p896.npz", log_probs=lp,
                                        native_log_probs=ref_saved,
                                        target_id=np.array(target),target_position=np.array(PREFIX+j+1))
                cache.post_decode()
            cache.validate(256,1023)
            require(len(rows)==255 and [x['target_position'] for x in rows]==list(range(769,1024)), 'Scoring position mismatch')
            result = {**spec,'status':'COMPLETE_FINITE','scored':255,
                      'nll':sum(-x['target_log_probability'] for x in rows)/255,
                      'mean_kl':None if baseline is None else sum(x['kl_native_to_condition'] for x in rows)/255,
                      'seconds':time.monotonic()-start,'peak_gpu_allocated':torch.cuda.max_memory_allocated(),
                      'peak_gpu_reserved':torch.cuda.max_memory_reserved(),**cache.accounting()}
            result['ppl'] = safe_ppl(result['nll'])
            return result, rows, np.stack(native_lp) if native_lp else None
        except BaseException as e:
            if isinstance(e, IntegrityError): status='BLOCKED_IMPLEMENTATION'
            elif isinstance(e, NumericalFailure): status='NUMERICAL_FAILURE'
            elif isinstance(e, BudgetStop): status='PARTIAL'
            elif isinstance(e, torch.cuda.OutOfMemoryError): status='RESOURCE_FAILURE'
            elif isinstance(e, TimeoutError): status='PARTIAL'
            elif isinstance(e, (KeyboardInterrupt,SystemExit)): status='PARTIAL'
            else: status='BLOCKED_IMPLEMENTATION'
            result = {**spec,'status':status,'scored':len(rows),'seconds':time.monotonic()-start,
                      'error_type':type(e).__name__,'error':redact(str(e)),
                      'traceback':redact(traceback.format_exc())}
            if isinstance(e,(BudgetStop,KeyboardInterrupt,SystemExit)):
                result['stage_stop'] = True
            return result, rows, None
        finally:
            del cache
            import gc
            gc.collect()
            torch.cuda.empty_cache()


def aggregate(root, planned):
    raw = read_rows(root/'per_document.jsonl')
    token_rows = read_rows(root/'per_token.jsonl')
    tokens_by_id = defaultdict(list)
    for t in token_rows:
        tokens_by_id[t['condition_id']].append(t)
    seen = {}
    for r in raw:
        require(r['id'] not in seen,'Duplicate trajectory in output')
        if r['status']=='COMPLETE_FINITE':
            ts = tokens_by_id[r['id']]
            require(len(ts)==255 and len({t['target_position'] for t in ts})==255,'Output scoring count')
            # Independent recomputation from stored rows, using math.fsum.
            calculated = -math.fsum(t['target_log_probability'] for t in ts)/len(ts)
            require(abs(calculated-r['nll']) <= 1e-10,'Summary NLL disagreement')
        seen[r['id']]=r
    lookup={(r['book_id'],r['r'],r['mode'],r['bits']):r for r in raw}
    comparisons=[]
    configs=sorted({(p['r'],p['mode'],p['bits']) for p in planned if p['bits'] is not None})
    for depth,mode,b in configs:
        diffs,books=[],[]
        for doc in sorted({p['book_id'] for p in planned}):
            x=lookup.get((doc,depth,mode,b)); y=lookup.get((doc,depth,mode,None))
            if x and y and x['status']==y['status']=='COMPLETE_FINITE':
                diffs.append(x['nll']-y['nll']);books.append(doc)
        item={'r':depth,'mode':mode,'bits':b,'paired_books':books,'n':len(diffs),'per_book_delta_nll':diffs,
              'status':'COMPLETE' if len(diffs)==8 else 'INCONCLUSIVE'}
        if diffs:
            a=np.array(diffs,dtype=np.float64)
            draws=np.random.default_rng(BOOT_SEED).integers(0,len(a),(10000,len(a)))
            ci=np.quantile(a[draws].mean(axis=1),[0.025,0.975]).tolist()
            item.update(mean_delta_nll=float(a.mean()),median_delta_nll=float(np.median(a)),
                        range_delta_nll=[float(a.min()),float(a.max())],bootstrap_95=ci,
                        ppl_ratio=safe_ppl(float(a.mean())))
        comparisons.append(item)
    counts=Counter(r['status'] for r in raw)
    missing=[p['id'] for p in planned if p['id'] not in seen]
    report={'planned':len(planned),'obtained':len(raw),'independent_diagnostic_books':8,
            'status_counts':dict(counts),'not_run_ids':missing,'comparisons':comparisons,
            'overall_status':'COMPLETE' if not missing and counts.get('COMPLETE_FINITE')==len(planned) else 'INCONCLUSIVE',
            'ci_label':'exploratory paired book percentile bootstrap; not a confirmatory test',
            'memory_label':'planned packed bytes / actual floating cache / process peak are separate'}
    write_json(root/'summary.json',report)
    return report


def load_parent(path):
    parent=Path(path)
    require(parent.is_dir(),'Missing parent artifact directory')
    checks=parent/'SHA256SUMS.json'
    require(checks.exists(),'Parent ZIP lacks hash list')
    for name,h in json.loads(checks.read_text()).items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,'Unsafe parent path')
        p=parent/name
        require(p.is_file() and sha(p.read_bytes())==h,f'Parent hash mismatch: {name}')
    manifest=json.loads((parent/'manifest.json').read_text())
    p0=json.loads((parent/'p0_checks.json').read_text())
    require(p0['status']=='PASS','Parent P0 did not pass')
    require(manifest['part']=='P0_P1A' and manifest['data_revision']==DATA_REV and manifest['seed']==SEED,'Upload the matching P0/P1A ZIP')
    require(manifest['model_revision']==MODEL_REV and manifest['worker_version']==VERSION,'Parent configuration differs')
    require(manifest['worker_sha256']==sha(Path(__file__).read_bytes()),'Parent used a different worker implementation')
    return manifest,json.loads((parent/'materials.json').read_text())


def main():
    root=Path(sys.argv[1]); part=sys.argv[2]
    root.mkdir(parents=True,exist_ok=True)
    (root/'diagnostics').mkdir(exist_ok=True)
    started=time.monotonic()
    write_json(root/'phase.json',{'stage':'PREPARE','deadline_monotonic':started+3600})
    parent_manifest=parent_materials=None
    status='STARTING'
    def terminate(signum, frame):
        raise BudgetStop('Parent requested termination')
    signal.signal(signal.SIGTERM,terminate)
    try:
        require(part in ('P0_P1A','P1B'),'Invalid part')
        if part=='P1B':
            parent_manifest,parent_materials=load_parent(sys.argv[3])
        import torch as base_torch
        require(base_torch.cuda.is_available(),'Select a GPU runtime in Colab first')
        require(base_torch.cuda.is_bf16_supported(),'Native BF16 required; no automatic dtype substitution')
        install_packages(parent_manifest)
        require(time.monotonic()-started <= 3600,'Preparation exceeded 60-minute cap')
        numeric_imports()
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(SEED)
        from huggingface_hub import hf_hub_download
        from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer
        secret=os.environ.get('HF_TOKEN') or None
        source=root/'sources';source.mkdir(exist_ok=True)
        for name,h in SOURCE_HASHES.items():
            f=Path(hf_hub_download(MODEL,name,revision=MODEL_REV,token=secret))
            require(sha(f.read_bytes())==h,f'Pinned model source mismatch: {name}')
            shutil.copy2(f,source/name)
        packages={n:importlib.metadata.version(n) for n in PACKAGE_NAMES}
        if parent_manifest:
            require(packages==parent_manifest['packages'],'Parent dependency versions differ (PyTorch included)')
        config=AutoConfig.from_pretrained(MODEL,revision=MODEL_REV,token=secret,trust_remote_code=True)
        require((config.num_hidden_layers,config.num_key_value_heads,config.head_dim)==(LAYERS,HEADS,DIM),'Model dimensions differ')
        require(all(t=='full_attention' for t in config.layer_types),'Unexpected sliding attention')
        tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_REV,token=secret,trust_remote_code=True)
        tok_sha=sha(tokenizer.backend_tokenizer.to_str().encode())
        if parent_manifest:
            require(tok_sha==parent_manifest['tokenizer_sha256'],'Parent tokenizer differs')
        data=materials(root,tokenizer,parent_materials)
        planned=schedule(part,data['diagnostic'])
        write_json(root/'schedule.json',planned)
        config.total_ut_steps=4
        log('Loading Ouro-1.4B BF16 on Colab GPU (weights unchanged).')
        model=AutoModelForCausalLM.from_pretrained(MODEL,config=config,revision=MODEL_REV,
                    token=secret,trust_remote_code=True,torch_dtype=torch.bfloat16,
                    attn_implementation='sdpa',low_cpu_mem_usage=True).to('cuda').eval()
        require(time.monotonic()-started <= 3600,'Preparation exceeded 60-minute cap')
        official=importlib.import_module(type(model).__module__)
        base=official.UniversalTransformerCache
        cls=make_adapter_class(base)
        manifest={'worker_version':VERSION,'part':part,'model':MODEL,'model_revision':MODEL_REV,
                  'tokenizer_revision':MODEL_REV,'tokenizer_sha256':tok_sha,'data_revision':DATA_REV,
                  'source_hashes':SOURCE_HASHES,'packages':packages,'python':sys.version,
                  'gpu':torch.cuda.get_device_name(0),'gpu_total_bytes':torch.cuda.get_device_properties(0).total_memory,
                  'torch_cuda':torch.version.cuda,'backend':'sdpa','dtype':'BF16',
                  'deterministic_algorithms':True,'tf32':False,
                  'readout':'exit_at_step=r-1; labels=None','mask':'explicit full causal for official and adapter',
                  'seed':SEED,'worker_sha256':sha(Path(__file__).read_bytes()),
                  'parent_run_id':parent_manifest.get('run_id') if parent_manifest else None,
                  'run_id':root.parent.name,'preparation_seconds':time.monotonic()-started}
        write_json(root/'manifest.json',manifest)
        engine=Engine(model,root,cls,base)
        # Revalidation in P1B shares its two-hour budget; no extra budget.
        engine.stage_deadline=time.monotonic()+7200
        write_json(root/'phase.json',{'stage':'P0_REVALIDATION' if part=='P1B' else 'P0','deadline_monotonic':engine.stage_deadline})
        p0_start=time.monotonic()
        engine.p0(data['smoke'],revalidation=(part=='P1B'))
        manifest['p0_seconds']=time.monotonic()-p0_start
        manifest['p0_status']='PASS'
        write_json(root/'manifest.json',manifest)
        log('P0 PASS. Starting fixed P1 schedule.')
        if part=='P0_P1A':
            engine.stage_deadline=time.monotonic()+7200
        write_json(root/'phase.json',{'stage':'P1A' if part=='P0_P1A' else 'P1B','deadline_monotonic':engine.stage_deadline})
        timing_rows=json.loads((root/'p0_checks.json').read_text())['native_timings']
        by_r={r:[v['seconds'] for v in timing_rows if v['r']==r] for r in (1,2,4,8)}
        estimate=sum(sum(by_r[s['r']])/len(by_r[s['r']])*256/66 for s in planned)
        remaining=engine.stage_deadline-time.monotonic()
        write_json(root/'time_estimate.json',{'rough_seconds':estimate,'remaining_cap_seconds':remaining,
                     'method':'native smoke wall time scaled by forward count; quantization/prefill overhead not measured'})
        if estimate>remaining:
            raise BudgetStop('Native smoke extrapolation exceeds remaining job cap; no low-bit results opened')
        baselines={}
        bybook={x['book_id']:x for x in data['diagnostic']}
        for number,spec in enumerate(planned,1):
            engine.condition_deadline=math.inf
            engine.check_time()
            log(f"[{number}/{len(planned)}] {spec['id']}")
            key=(spec['book_id'],spec['r'],spec['mode'])
            # Only one reference matrix is held at once; schedule groups conditions.
            if spec['bits'] is None:
                baselines.clear()
            result,rows,base_lp=engine.trajectory(bybook[spec['book_id']],spec,baselines.get(key))
            append_json(root/'per_document.jsonl',result)
            append_json(root/'cache_counters.jsonl', {k:result[k] for k in ['id','status','update_count','slots','compressed_tokens','planned_components','planned_bytes','floating_cache_bytes'] if k in result})
            if result['status']!='COMPLETE_FINITE':
                append_json(root/'failures.jsonl',result)
            if base_lp is not None:
                baselines[key]=base_lp
            report=aggregate(root,planned)
            log(f"  {result['status']} / scored={result['scored']} / seconds={result['seconds']:.1f}")
            write_json(root/'checkpoint.json',{'completed':number,'total':len(planned),'last_id':spec['id']})
            if number % 4 == 0 or result['status']!='COMPLETE_FINITE':
                checkpoint_zip(root)
            if result.get('stage_stop') or result['status']=='BLOCKED_IMPLEMENTATION':
                break
            # Payload cap. It does not include the model download cache.
            if sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) > 10*1024**3:
                raise BudgetStop('Artifact payload cap reached')
        status=aggregate(root,planned)['overall_status']
    except BaseException as e:
        status='BLOCKED_IMPLEMENTATION' if isinstance(e,IntegrityError) else 'INCONCLUSIVE'
        append_json(root/'failures.jsonl', {'stage':'worker','status':status,'error_type':type(e).__name__,
                                          'error':redact(str(e)),'traceback':redact(traceback.format_exc())})
        log(traceback.format_exc())
        if (root/'schedule.json').exists():
            try: aggregate(root,json.loads((root/'schedule.json').read_text()))
            except BaseException as inner: log(f'Final aggregation failed: {redact(inner)}')
    finally:
        write_json(root/'run_status.json', {'status':status,'part':part,'elapsed_seconds':time.monotonic()-started,
                                           'time_utc':datetime.now(timezone.utc).isoformat()})
        log(f'Worker finished: {status}. Packaging is handled by the Colab cell.')


if __name__ == '__main__':
    main()
