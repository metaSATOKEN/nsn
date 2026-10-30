# LOOP-KV-KV01 / v0.1 / Kentaro Sato / 2026-09-30
# Importing this module does not load a model, use a GPU, or access the network.
import os
import sys
import json
import math
import random
import time
import signal
import shutil
import hashlib
import traceback
import importlib
import importlib.metadata
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime, timezone

import loop_kv_p0_diagnostic as d
w = d.w
EXPERIMENT = 'LOOP-KV-KV01'
VERSION = '0.1'
CORE_SHA = '4312155fc121d926f0a9bd794a62758f7acab634f8991d671e60fe22a7b0b1c9'
DIAGNOSTIC_SHA = '8d9fdff352cf0ebae5ba850ecd2413ba65af33143c331c7e0adf01c2fa62c5cd'
MATERIAL_PATH = Path(__file__).with_name('loop_kv_p0r_materials.json')
MATERIAL_SHA = '739a619d7486f34a8ab11c494702e804803ac170e45563fd7abde16bbcc0cc6f'
MATERIALS = json.loads(MATERIAL_PATH.read_text())
SMOKE_IDS = ['37013', '2976', '9442', '56102']
DIAGNOSTIC_IDS = ['3752', '31131', '31258', '34475', '4124', '49966', '49484', '4537']
PREP_SECONDS, P0_SECONDS, P1_SECONDS = 1800, 1800, 3600
CASE_SECONDS, TRAJECTORY_SECONDS = 180, 900
MAX_BYTES = 10 * 1024**3
ROUTES = ('official_repeat', 'adapter', 'adapter_instrumented')
WATCH_LENGTHS = (768, 832, 896, 960, 1023)
WATCH_LAYERS = (0, 12, 23)
LEGACY_STATUS = 'BLOCKED_IMPLEMENTATION_RETAINED'
QUALITY_MARGIN = math.log(1.05)
IDENTITY_PATH = Path(__file__).with_name('loop_kv_p1b_identity.json')
IDENTITY_SHA = 'f0cbed21c4a018d3fde1e1e07086c363f0a86c6a51a49c351f5e2dd3fe096b00'


def numeric_imports():
    global torch, np
    d.numeric_imports()
    torch, np = w.torch, w.np


def verify_inputs():
    w.require(w.sha(Path(w.__file__).read_bytes()) == CORE_SHA, 'Frozen core changed')
    w.require(w.sha(Path(d.__file__).read_bytes()) == DIAGNOSTIC_SHA, 'Frozen diagnostic helper changed')
    w.require(w.sha(MATERIAL_PATH.read_bytes()) == MATERIAL_SHA, 'Material file changed')
    w.require(w.sha(IDENTITY_PATH.read_bytes()) == IDENTITY_SHA, 'Historical identity changed')
    original = json.loads(MATERIAL_PATH.read_text())
    w.require(MATERIALS == original, 'In-memory materials changed')
    w.require(MATERIALS['data_revision'] == w.DATA_REV and MATERIALS['selection_seed'] == w.SEED,
              'Material revision/seed changed')
    w.require(MATERIALS['tokenizer_revision'] == w.MODEL_REV, 'Tokenizer revision changed')
    for split, ids in [('smoke', SMOKE_IDS), ('diagnostic', DIAGNOSTIC_IDS)]:
        docs = MATERIALS[split]
        w.require([x['book_id'] for x in docs] == ids, 'Material IDs/order changed')
        for doc in docs:
            w.require(len(doc['tokens']) == 1024 and doc['offset'] == 0, 'Material length/offset')
            w.require(w.token_sha(doc['tokens']) == doc['token_sha256'], 'Token digest mismatch')


def p0_schedule():
    rows = [dict(id=f'G2-b{b}-r{r}', group='G2', book_id=b, r=r, length=129,
                 prefill=64, steps=66, full=True, expected_forwards=265)
            for b in SMOKE_IDS for r in (8,)]
    for length in (1, 63, 64, 65, 127, 128, 129):
        prefill = min(64, length)
        steps = 1 + length - prefill
        rows.append(dict(id=f'G3-L{length}-r8', group='G3', book_id=None, r=8,
                         length=length, prefill=prefill, steps=steps, full=True,
                         expected_forwards=4*steps+1))
    rows += [dict(id=f'G4-b{b}-r{r}', group='G4', book_id=b, r=r, length=1023,
                  prefill=768, steps=256, full=False, expected_forwards=1024)
             for b in SMOKE_IDS for r in (8,)]
    return rows


TARGETS = ('NATIVE', 'K_ONLY', 'V_ONLY', 'KV')


def selected_parts(target, bits):
    w.require(target in TARGETS, 'Unknown compression target')
    w.require(bits in (None, 3), 'KV01 bits must be native or NSN3')
    w.require((bits is None and target in ('NATIVE','KV')) or
              (bits == 3 and target != 'NATIVE'), 'Inconsistent target/bits')
    if bits is None: return ()
    return {'K_ONLY':('K',), 'V_ONLY':('V',), 'KV':('K','V')}[target]


def p1_schedule():
    rng=random.Random(w.SEED)
    books=list(DIAGNOSTIC_IDS);rng.shuffle(books);result=[]
    for book in books:
        interventions=['K_ONLY','V_ONLY','KV'];rng.shuffle(interventions)
        for target in ['NATIVE']+interventions:
            result.append(dict(id=f'KV01-b{book}-r8-{target}',book_id=book,r=8,
                mode='PREFIX_ONLY',target=target,bits=None if target=='NATIVE' else 3,
                expected_scored=255))
    return result


def compatibility_schedule():
    return [dict(id=f'G5-b{b}-r8-KV',book_id=b,r=8,length=1023,prefill=768,
                 steps=256,expected_forwards=512) for b in SMOKE_IDS]


def finite(tensor, label):
    if not bool(torch.isfinite(tensor).all()):
        raise w.NumericalFailure('Nonfinite ' + label)


def digest_tensor(tensor):
    t = tensor.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str((tuple(t.shape), str(t.dtype))).encode())
    h.update(t.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def rng_state():
    return (torch.random.get_rng_state().clone(), torch.cuda.get_rng_state().clone())


def rng_unchanged(before):
    now = rng_state()
    w.require(all(torch.equal(a, b) for a, b in zip(before, now)), 'Inference changed RNG')


def snapshot(cache):
    result = w.tensor_snapshot(cache)
    for pair in result:
        for t in pair:
            finite(t, 'native KV')
    return result


def digest_cache(cache):
    h = hashlib.sha256()
    for slot, (k, v) in enumerate(zip(cache.key_cache, cache.value_cache)):
        for label, t in [('K', k), ('V', v)]:
            finite(t, 'final KV')
            h.update(f'{slot}:{label}:{digest_tensor(t)}'.encode())
    return h.hexdigest()


def norm_summary(n):
    finite(n, 'KV norm')
    a = n.detach().float().reshape(-1).cpu()
    return dict(count=a.numel(), p50=float(torch.quantile(a, .5)),
                p99=float(torch.quantile(a, .99)), maximum=float(a.max()), nonfinite=0)


def selected_norms(cache, start=0, end=None):
    values = {}
    for slot in range(w.LAYERS*cache.r):
        if slot % w.LAYERS not in WATCH_LAYERS:
            continue
        for label, data in [('K', cache.key_cache), ('V', cache.value_cache)]:
            x = data[slot][:, :, start:end, :].detach().float()
            values[(slot, label)] = torch.linalg.vector_norm(x, dim=-1)
    return values


def post_step(cache, prefill, root=None, identity=None):
    length = cache.get_seq_length()
    observe = root is not None and length in WATCH_LENGTHS
    span = None
    if observe and length == 768:
        span = (0, 704)
    elif observe and cache.mode == 'DECODE_ONLINE' and length in (832, 896, 960):
        span = (length-128, length-64)
    before = selected_norms(cache, *span) if span else None
    if prefill:
        cache.post_prefill()
    else:
        cache.post_decode()
    if not observe:
        return
    for (slot, label), norms in selected_norms(cache).items():
        w.append_json(root/'norm_snapshots.jsonl', {**identity, 'cache_length': length,
            'slot': slot, 'loop': slot//w.LAYERS, 'layer': slot % w.LAYERS,
            'kv': label, **norm_summary(norms)})
    if span:
        for (slot, label), after in selected_norms(cache, *span).items():
            old = before[(slot, label)]
            absolute = (after-old).abs()
            relative = absolute / old.clamp_min(w.EPS)
            finite(relative, 'relative norm error')
            w.append_json(root/'norm_transfers.jsonl', {**identity, 'cache_length': length,
                'slot': slot, 'kv': label, 'start': span[0], 'end_exclusive': span[1],
                'quantization_applied': cache.bits is not None and label in getattr(cache,'selected',('K','V')),
                'before': norm_summary(old), 'after': norm_summary(after),
                'max_absolute_norm_change': float(absolute.max()),
                'max_relative_norm_change': float(relative.max()),
                'zero_norm_count': int((old == 0).sum())})


def make_selective_adapter(base):
    Old = w.make_adapter_class(base)
    class SelectiveCache(Old):
        def __init__(self,r,bits=None,mode='PREFIX_ONLY',trace=False,target='KV',audit=False):
            w.require(mode=='PREFIX_ONLY', 'KV01 supports PREFIX_ONLY only')
            self.target=target
            self.selected=selected_parts(target,bits)
            self.audit=audit;self.interventions=[];self.prefilled=False
            self.side_blocks=Counter()
            super().__init__(r,bits,mode,trace)

        def compress_to(self,end):
            if self.bits is None:return
            for slot in range(w.LAYERS*self.r):
                w.require(self.get_seq_length(slot)>=end+w.BLOCK,'Insufficient protected tail')
                start=self.compressed.get(slot,0)
                w.require(end>=start and (end-start)%w.BLOCK==0,'Requantization / overlap')
                if end==start:continue
                for label,data in [('K',self.key_cache),('V',self.value_cache)]:
                    if label not in self.selected:continue
                    original=data[slot];nblocks=(end-start)//w.BLOCK
                    group=original[:,:,start:end,:].reshape(1,w.HEADS,nblocks,w.BLOCK,w.DIM)
                    restored=w.nsn_qdq(group,self.bits).reshape(1,w.HEADS,end-start,w.DIM)
                    data[slot]=torch.cat((original[:,:,:start,:],restored,original[:,:,end:,:]),dim=-2)
                    self.side_blocks[(slot,label)]+=nblocks
                self.encoded_blocks[slot]+=(end-start)//w.BLOCK
                self.compressed[slot]=end

        def post_prefill(self):
            w.require(not self.prefilled, 'Duplicate prefix intervention')
            before=[(k.detach().clone(),v.detach().clone()) for k,v in zip(self.key_cache,self.value_cache)] if self.audit else None
            super().post_prefill()
            self.prefilled=True
            if before is not None:self.audit_intervention(before)

        def audit_intervention(self,before):
            w.require(len(before)==w.LAYERS*self.r,'Intervention slot inventory')
            for slot,(oldk,oldv) in enumerate(before):
                for label,old,new in [('K',oldk,self.key_cache[slot]),('V',oldv,self.value_cache[slot])]:
                    selected=label in self.selected
                    finite(new,'post-intervention KV')
                    w.require(old.shape==new.shape and old.dtype==new.dtype,'Intervention shape/dtype')
                    tail_equal=torch.equal(old[:,:,704:,:],new[:,:,704:,:])
                    all_equal=torch.equal(old,new)
                    w.require(tail_equal,'Protected tail modified')
                    w.require(selected or all_equal,'Unselected KV modified')
                    diff=(new[:,:,:704,:].float()-old[:,:,:704,:].float())
                    rel=float(torch.linalg.vector_norm(diff)/torch.linalg.vector_norm(old[:,:,:704,:].float()).clamp_min(w.EPS))
                    maximum=float(diff.abs().max())
                    w.require(math.isfinite(rel) and math.isfinite(maximum),'Nonfinite intervention metric')
                    self.interventions.append(dict(slot=slot,kv=label,selected=selected,start=0,end_exclusive=704,
                        cache_length=self.get_seq_length(slot),protected_tail_bitwise=tail_equal,
                        full_tensor_bitwise=all_equal,max_abs_change=maximum,relative_l2_change=rel))

        def validate(self,expected_calls,length):
            super().validate(expected_calls,length)
            for slot in range(w.LAYERS*self.r):
                for label in ('K','V'):
                    w.require(self.side_blocks[(slot,label)]==(11 if label in self.selected else 0), 'Side block count')
            if self.audit:validate_interventions(self.interventions,self.r,self.target,self.bits)

        def accounting(self):
            combined=Counter();by_side={};allocated=0
            for label,data in [('K',self.key_cache),('V',self.value_cache)]:
                components=Counter();active=label in self.selected
                for slot in range(w.LAYERS*self.r):
                    tensor=data[slot]
                    components.update(w.planned_bytes(tensor.shape,self.compressed.get(slot,0) if active else 0,self.bits if active else None))
                    allocated+=tensor.numel()*tensor.element_size()
                by_side[label]=dict(components);combined.update(components)
            return dict(planned_components=dict(combined),planned_components_by_kv=by_side,
                planned_bytes=sum(combined.values()),floating_cache_bytes=allocated,
                implementation='FAKE_QUANT_FLOAT_CACHE',update_count=sum(self.calls.values()),slots=len(self.calls),
                compressed_tokens=self.compressed.get(0,0),
                compressed_tokens_by_kv={label:self.compressed.get(0,0) if label in self.selected else 0 for label in ('K','V')})
    return SelectiveCache


def validate_interventions(rows,r,target,bits):
    expected={(s,kv) for s in range(w.LAYERS*r) for kv in ('K','V')}
    keys=[(x['slot'],x['kv']) for x in rows];parts=selected_parts(target,bits)
    w.require(len(keys)==len(set(keys)) and set(keys)==expected,'Intervention coordinate inventory')
    for row in rows:
        active=row['kv'] in parts
        w.require(row['selected']==active and row['start']==0 and row['end_exclusive']==704
            and row['cache_length']==768,'Intervention target/span metadata')
        w.require(row['protected_tail_bitwise'] is True,'Intervention tail evidence')
        w.require(active or row['full_tensor_bitwise'] is True,'Unselected intervention evidence')
        for k in ('max_abs_change','relative_l2_change'):
            w.require(math.isfinite(row[k]) and row[k]>=0 and (active or row[k]==0),'Intervention error metric')


def expected_accounting(spec):
    parts=selected_parts(spec['target'],spec['bits']);components=Counter();sides={}
    for kv in ('K','V'):
        single=w.planned_bytes((1,w.HEADS,1023,w.DIM),704 if kv in parts else 0,3 if kv in parts else None)
        sides[kv]={k:v*w.LAYERS*spec['r'] for k,v in single.items()};components.update(sides[kv])
    return dict(planned_components=dict(components),planned_components_by_kv=sides,
        planned_bytes=sum(components.values()),floating_cache_bytes=2*w.LAYERS*spec['r']*w.HEADS*1023*w.DIM*2,
        compressed_tokens_by_kv={kv:704 if kv in parts else 0 for kv in ('K','V')},
        slots=w.LAYERS*spec['r'],update_count=256*w.LAYERS*spec['r'])


class RouteEngine(w.Engine):
    def __init__(self, *args):
        super().__init__(*args)
        self.forward_count = 0

    def forward(self, *args, **kwargs):
        result = super().forward(*args, **kwargs)
        self.forward_count += 1
        finite(result, 'model logits')
        return result

    def route(self, tokens, spec, route):
        r = spec['r']
        self.set_r(r)
        official = route.startswith('official')
        instrumented = route == 'adapter_instrumented'
        cache = (self.base_class(max_cache_size=w.LAYERS*r) if official else
                 self.adapter_class(r, None, 'PREFIX_ONLY', trace=instrumented))
        before = rng_state()
        start = time.monotonic()
        prefill = spec['prefill']
        parts = [self.forward(tokens[:prefill], cache, r, keep=0)[0].cpu()]
        observe_root = self.root/'p0r' if instrumented else None
        ident = {'phase': 'P0KV', 'case_id': spec['id']}
        if not official:
            post_step(cache, True, observe_root, ident)
        for token in tokens[prefill:]:
            parts.append(self.forward([token], cache, r)[0].cpu())
            if not official:
                post_step(cache, False, observe_root, ident)
        logits = torch.cat(parts)
        finite(logits, 'route logits')
        w.require(logits.shape[0] == spec['length'], 'Route output length')
        if not official:
            cache.validate(spec['steps'], spec['length'])
        rng_unchanged(before)
        records = list(getattr(cache, 'records', []))
        kv = snapshot(cache)
        return logits, kv, time.monotonic()-start, records


def validate_trace(records, spec):
    expected = []
    for step in range(spec['steps']):
        old, added = (0, spec['prefill']) if step == 0 else (spec['prefill']+step-1, 1)
        for slot in range(w.LAYERS*spec['r']):
            expected.append(dict(slot=slot, loop=slot//w.LAYERS, layer=slot % w.LAYERS,
                                 old=old, added=added, new=old+added))
    w.require(records == expected, 'Trace coordinate/order mismatch')
    return len(records)


def full_metrics(full, inc, tokens, case_id):
    w.require(full.shape == inc.shape and len(tokens) == full.shape[0], 'Full/inc shapes')
    a, b = torch.log_softmax(full.float(), -1), torch.log_softmax(inc.float(), -1)
    finite(a, 'full distribution'); finite(b, 'incremental distribution')
    kls = (a.exp()*(a-b)).sum(-1)
    maxima = (full-inc).abs().float().amax(-1)
    finite(kls, 'full/incremental KL'); finite(maxima, 'full/incremental logit difference')
    rows = []
    for i in range(len(tokens)):
        target = tokens[i+1] if i+1 < len(tokens) else None
        row = dict(case_id=case_id, input_position=i, input_token=tokens[i],
                   target_position=i+1 if target is not None else None,
                   target_id=target, kl=float(kls[i]), max_abs_logit=float(maxima[i]))
        if target is not None:
            row.update(full_nll=float(-a[i, target]), incremental_nll=float(-b[i, target]))
            row['delta_nll'] = row['incremental_nll'] - row['full_nll']
        rows.append(row)
    delta = None
    if len(tokens) > 1:
        idx, targets = torch.arange(len(tokens)-1), torch.tensor(tokens[1:])
        delta = float((-b[:-1][idx, targets]).mean() - (-a[:-1][idx, targets]).mean())
        if not math.isfinite(delta):raise w.NumericalFailure('Nonfinite full/incremental NLL difference')
    metrics = dict(kl=float(kls.mean()), delta_nll=delta, max_abs_logit=float(maxima.max()))
    within = metrics['kl'] <= 1e-4 and metrics['max_abs_logit'] <= .25 and (delta is None or abs(delta) <= .005)
    return {**metrics, 'within_original_tolerance': within,
            'nll_test': 'NOT_APPLICABLE' if delta is None else 'MEASURED'}, rows


def classify_error(exc, route=False):
    if isinstance(exc, (w.IntegrityError, w.NumericalFailure)):
        return 'BLOCKED_ROUTE' if route else ('NUMERICAL_FAILURE' if isinstance(exc, w.NumericalFailure) else 'BLOCKED_IMPLEMENTATION')
    oom = getattr(getattr(globals().get('torch'), 'cuda', None), 'OutOfMemoryError', ())
    if isinstance(exc, oom):
        return 'RESOURCE_FAILURE'
    if isinstance(exc, (TimeoutError, w.BudgetStop, KeyboardInterrupt, SystemExit)):
        return 'PARTIAL'
    return 'BLOCKED_ROUTE' if route else 'BLOCKED_IMPLEMENTATION'


def payload_check(root):
    if sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) > MAX_BYTES:
        raise w.BudgetStop('Artifact payload cap reached')


def validate_norm_inventory(norms, transfers, ident, id_key, r=4, mode='DECODE_ONLINE'):
    keys=[(x['slot'],x['kv'],x['cache_length']) for x in norms if x.get(id_key)==ident]
    slots=[loop*w.LAYERS+layer for loop in range(r) for layer in WATCH_LAYERS if layer<w.LAYERS]
    expected={(s,kv,length) for s in slots for kv in ('K','V') for length in WATCH_LENGTHS}
    w.require(len(keys)==len(set(keys)) and set(keys)==expected,'Missing/duplicate norm coordinates')
    for x in norms:
        if x.get(id_key)==ident:
            w.require(x['count']==w.HEADS*x['cache_length'] and x['nonfinite']==0,'Norm population/finite count')
    transfer_keys=[(x['slot'],x['kv'],x['cache_length']) for x in transfers if x.get(id_key)==ident]
    lengths=(768,832,896,960) if mode=='DECODE_ONLINE' else (768,)
    expected_transfer={(s,kv,length) for s in slots for kv in ('K','V') for length in lengths}
    w.require(len(transfer_keys)==len(set(transfer_keys)) and set(transfer_keys)==expected_transfer,'Missing/duplicate norm transfer coordinates')


def current_p0(root, fixtures=None):
    dest = root/'p0r'
    cases, checks = w.read_rows(dest/'cases.jsonl'), w.read_rows(dest/'checks.jsonl')
    expected = {x['id']: x for x in p0_schedule()}
    seen = {x['id']: x for x in cases}
    w.require(len(seen) == len(cases) and set(seen) <= set(expected), 'P0 duplicate/unplanned cases')
    keys = [(x['case_id'], x['route']) for x in checks]
    w.require(len(keys) == len(set(keys)) and set(keys) <= {(i,r) for i in expected for r in ROUTES}, 'P0 duplicate/unplanned checks')
    passed = [x for x in cases if x['status'] == 'PASS']
    for row in passed:
        spec = expected[row['id']]
        w.require(all(row.get(k)==v for k,v in spec.items()),'P0 case metadata changed')
        w.require(row['forwards'] == spec['expected_forwards'], 'P0 forward count')
        w.require(all(any(c['case_id']==row['id'] and c['route']==r and c['status']=='PASS'
            and c['logits_bitwise'] and c['kv_bitwise'] and c['rng_unchanged']
            and c['reference_logit_sha256']==c['compared_logit_sha256']
            and c['trace_updates']==(spec['steps']*w.LAYERS*spec['r'] if r=='adapter_instrumented' else 0)
            for c in checks) for r in ROUTES), 'Missing/invalid route comparison')
    if fixtures is None:
        fixtures = json.loads((dest/'fixtures.json').read_text()) if (dest/'fixtures.json').exists() else []
    fixture_ok = bool(fixtures) and all(x['status']=='PASS' for x in fixtures)
    norms=w.read_rows(dest/'norm_snapshots.jsonl');transfers=w.read_rows(dest/'norm_transfers.jsonl')
    for row in passed:
        if row['group']=='G4':validate_norm_inventory(norms,transfers,row['id'],'case_id',row['r'],'PREFIX_ONLY')
    compat=w.read_rows(dest/'compatibility.jsonl')
    comp_specs={x['id']:x for x in compatibility_schedule()};comp_seen={x['id']:x for x in compat}
    w.require(len(comp_seen)==len(compat) and set(comp_seen)<=set(comp_specs),'Compatibility inventory')
    comp_pass=[]
    for row in compat:
        w.require(all(row.get(k)==v for k,v in comp_specs[row['id']].items()),'Compatibility metadata')
        if row['status']=='PASS':
            w.require(row['forwards']==512 and row['logits_bitwise'] and row['kv_bitwise'] and row['rng_unchanged']
                and row['old_logit_sha256']==row['new_logit_sha256'],'Compatibility evidence')
            records=[x for x in w.read_rows(dest/'compat_interventions.jsonl') if x['case_id']==row['id']]
            validate_interventions(records,row['r'],'KV',3)
            comp_pass.append(row)
    complete = len(passed) == len(expected) and len(checks) == 3*len(expected) and fixture_ok and len(comp_pass)==len(comp_specs)
    gate_error=json.loads((dest/'gate_error.json').read_text()) if (dest/'gate_error.json').exists() else None
    blocked=any(x['status']=='BLOCKED_ROUTE' for x in cases+compat) or (gate_error and gate_error['status']=='BLOCKED_ROUTE')
    status = 'BLOCKED_ROUTE' if blocked else 'PASS_ROUTE_KV01' if complete and not gate_error else 'INCONCLUSIVE'
    report = dict(p0r_route_status=status, legacy_p0_status=LEGACY_STATUS,
                  legacy_full_p0_pass=False, fixtures=fixtures, planned_cases=len(expected),
                  passed_cases=len(passed), route_comparisons=len(checks),
                  compatibility_planned=len(comp_specs),compatibility_passed=len(comp_pass),
                  compatibility_not_run_ids=[i for i in comp_specs if i not in comp_seen],
                  compatibility_forwards=sum(x.get('forwards',0) for x in compat),
                  obtained_forwards=sum(x.get('forwards',0) for x in cases),
                  not_run_ids=[i for i in expected if i not in seen],
                  full_incremental_diagnostic=[{'id':x['id'],**x['full_incremental']} for x in passed if x.get('full_incremental')],
                  gate_error=gate_error,full_incremental_is_progression_gate=False)
    w.write_json(dest/'summary.json', report)
    return report


def execute_p0(engine, root, docs, fixtures):
    dest = root/'p0r'; dest.mkdir(exist_ok=True)
    w.write_json(dest/'fixtures.json', fixtures)
    bybook = {x['book_id']: x for x in docs}
    for spec in p0_schedule():
        start = time.monotonic(); n0 = engine.forward_count
        engine.condition_deadline = start+CASE_SECONDS
        w.write_json(root/'active_case.json',dict(id=spec['id'],phase='P0KV',status='STARTED',deadline_monotonic=engine.condition_deadline))
        w.log('P0KV '+spec['id'])
        row = dict(spec, status='STARTED')
        try:
            engine.check_time()
            w.require(engine.model.dtype == torch.bfloat16, 'P0KV must use BF16')
            tokens = (bybook[spec['book_id']]['tokens'][:spec['length']] if spec['book_id'] else
                      [(7+13*i) % engine.model.config.vocab_size for i in range(spec['length'])])
            original, kv, elapsed, _ = engine.route(tokens, spec, 'official')
            for route in ROUTES:
                got, got_kv, _, trace = engine.route(tokens, spec, route)
                logit_equal = torch.equal(original, got) and original.dtype == got.dtype
                kv_equal = w.snapshots_equal(kv, got_kv)
                trace_count = validate_trace(trace, spec) if route == 'adapter_instrumented' else 0
                check = dict(case_id=spec['id'], route=route, status='PASS' if logit_equal and kv_equal else 'FAIL',
                             logits_bitwise=logit_equal, kv_bitwise=kv_equal, rng_unchanged=True,
                             reference_logit_sha256=digest_tensor(original), compared_logit_sha256=digest_tensor(got), trace_updates=trace_count)
                w.append_json(dest/'checks.jsonl',check)
                if trace:
                    w.append_json(dest/'cache_trace.jsonl',dict(case_id=spec['id'],records=trace))
                w.require(logit_equal and kv_equal, 'Same-shape native route mismatch')
                del got, got_kv
            row['native_seconds'] = elapsed
            if spec['full']:
                full = engine.forward(tokens,None,spec['r'],keep=0)[0].cpu()
                diagnostic, rows = full_metrics(full,original,tokens,spec['id'])
                row['full_incremental'] = diagnostic
                for item in rows: w.append_json(dest/'per_token.jsonl',item)
                del full
            del original, kv
            row['status'] = 'PASS'
            payload_check(root)
        except BaseException as exc:
            row.update(status=classify_error(exc,True), error_type=type(exc).__name__,error=w.redact(str(exc)))
            w.append_json(root/'failures.jsonl',dict(phase='P0KV',**row,traceback=w.redact(traceback.format_exc())))
        row.update(forwards=engine.forward_count-n0,seconds=time.monotonic()-start)
        w.append_json(dest/'cases.jsonl',row)
        w.write_json(root/'active_case.json',dict(id=spec['id'],status=row['status']))
        report = current_p0(root,fixtures)
        w.checkpoint_zip(root)
        if row['status'] != 'PASS': return report['p0r_route_status']
    execute_compatibility(engine,root,docs)
    return current_p0(root,fixtures)['p0r_route_status']


def compatibility_route(engine,tokens,spec,new):
    engine.set_r(spec['r'])
    cache=(engine.adapter_class(spec['r'],3,'PREFIX_ONLY',target='KV',audit=True) if new else
           w.make_adapter_class(engine.base_class)(spec['r'],3,'PREFIX_ONLY'))
    before=rng_state()
    outputs=[engine.forward(tokens[:768],cache,spec['r'],keep=0)[0].cpu()]
    cache.post_prefill()
    for token in tokens[768:]:
        outputs.append(engine.forward([token],cache,spec['r'])[0].cpu());cache.post_decode()
    cache.validate(256,1023);rng_unchanged(before)
    logits=torch.cat(outputs);finite(logits,'compatibility logits')
    return logits,snapshot(cache),getattr(cache,'interventions',[])


def execute_compatibility(engine,root,docs):
    bybook={x['book_id']:x for x in docs}
    for spec in compatibility_schedule():
        start=time.monotonic();n0=engine.forward_count;engine.condition_deadline=start+CASE_SECONDS
        w.write_json(root/'active_case.json',dict(id=spec['id'],phase='P0KV',status='STARTED',deadline_monotonic=engine.condition_deadline))
        row=dict(spec,status='STARTED');w.log('Compatibility '+spec['id'])
        try:
            tokens=bybook[spec['book_id']]['tokens'][:1023]
            old,oldkv,_=compatibility_route(engine,tokens,spec,False)
            new,newkv,audit=compatibility_route(engine,tokens,spec,True)
            row.update(logits_bitwise=torch.equal(old,new) and old.dtype==new.dtype,
                kv_bitwise=w.snapshots_equal(oldkv,newkv),rng_unchanged=True,
                old_logit_sha256=digest_tensor(old),new_logit_sha256=digest_tensor(new))
            for item in audit:w.append_json(root/'p0r/compat_interventions.jsonl',dict(case_id=spec['id'],**item))
            w.require(row['logits_bitwise'] and row['kv_bitwise'],'Old/new KV compression mismatch')
            del old,new,oldkv,newkv
            row['status']='PASS';payload_check(root)
        except BaseException as exc:
            row.update(status=classify_error(exc,True),error_type=type(exc).__name__,error=w.redact(str(exc)))
            w.append_json(root/'failures.jsonl',dict(phase='COMPATIBILITY',**row,traceback=w.redact(traceback.format_exc())))
        row.update(forwards=engine.forward_count-n0,seconds=time.monotonic()-start)
        w.append_json(root/'p0r/compatibility.jsonl',row)
        w.write_json(root/'active_case.json',dict(id=spec['id'],status=row['status']))
        current_p0(root);w.checkpoint_zip(root)
        if row['status']!='PASS':return


class FixtureCache:
    def __init__(self,max_cache_size):
        self.key_cache=[];self.value_cache=[]
    def get_seq_length(self,layer_idx=0):
        return 0 if layer_idx>=len(self.key_cache) else self.key_cache[layer_idx].shape[-2]
    def update(self,key,value,layer_idx,cache_kwargs=None):
        while len(self.key_cache)<=layer_idx:
            self.key_cache.append(None);self.value_cache.append(None)
        for data,new in [(self.key_cache,key),(self.value_cache,value)]:
            data[layer_idx] = new if data[layer_idx] is None else torch.cat((data[layer_idx],new),-2)
        return self.key_cache[layer_idx],self.value_cache[layer_idx]


def cache_fixtures():
    # Exercise the frozen adapter on one head/one layer, without model weights.
    old = (w.LAYERS,w.HEADS)
    results=[]
    try:
        w.LAYERS,w.HEADS=1,1
        cls=w.make_adapter_class(FixtureCache)
        for bits in (3,4,5):
            for mode in ('PREFIX_ONLY','DECODE_ONLINE'):
                c=cls(1,bits,mode)
                x=torch.ones(1,1,768,128)
                c.update(x,x.clone(),0,{'cache_position':torch.arange(768)})
                protected=c.key_cache[0][:,:,704:,:].clone()
                c.post_prefill()
                w.require(c.compressed[0]==704 and torch.equal(protected,c.key_cache[0][:,:,704:,:]),'Prefix fixture')
                fixed=c.key_cache[0][:,:,:704,:].clone()
                flush=[]; previous=704
                for j in range(255):
                    x=torch.ones(1,1,1,128)
                    c.update(x,x.clone(),0,{'cache_position':torch.tensor([768+j])})
                    c.post_decode()
                    if c.compressed[0]!=previous:flush.append(j+1);previous=c.compressed[0]
                    w.require(torch.equal(c.key_cache[0][:,:,:704,:],fixed),'Reencoded old blocks')
                c.validate(256,1023)
                w.require(flush==([64,128,192] if mode=='DECODE_ONLINE' else []),'Flush positions')
                results.append(dict(test='cache_schedule_fixture',bits=bits,mode=mode,status='PASS',flush=flush,
                                    compressed=c.compressed[0],native_tail=1023-c.compressed[0]))
    finally:
        w.LAYERS,w.HEADS=old
    return results


def selective_fixtures():
    old=(w.LAYERS,w.HEADS);results=[]
    try:
        w.LAYERS,w.HEADS=1,1
        cls=make_selective_adapter(FixtureCache)
        positions=torch.arange(768).reshape(1,1,768,1)
        channels=torch.arange(1,129).reshape(1,1,1,128)
        x=(1.1+torch.sin(positions*channels*.031)).to(torch.bfloat16)
        for target in TARGETS:
            bits=None if target=='NATIVE' else 3
            c=cls(1,bits,target=target,audit=True)
            c.update(x.clone(),x.clone(),0,{'cache_position':torch.arange(768)})
            c.post_prefill();validate_interventions(c.interventions,1,target,bits)
            results.append(dict(test='selective_intervention',target=target,status='PASS'))
    finally:w.LAYERS,w.HEADS=old
    return results


def trajectory(engine, root, doc, spec, baseline=None):
    engine.set_r(spec['r']);engine.condition_deadline=time.monotonic()+TRAJECTORY_SECONDS
    cache=engine.adapter_class(spec['r'],spec['bits'],spec['mode'],target=spec['target'],audit=True)
    tokens=doc['tokens'];rows=[];native_lp=[];native_logits=[]
    distribution_digest=hashlib.sha256()
    before=rng_state();start=time.monotonic();n0=engine.forward_count
    d.device_enter()
    last_input_position=None
    try:
        engine.forward(tokens[:w.PREFIX],cache,spec['r'])
        ident=dict(phase='KV01',condition_id=spec['id'])
        post_step(cache,True,root,ident)
        for item in cache.interventions:w.append_json(root/'interventions.jsonl',dict(condition_id=spec['id'],**item))
        for j in range(w.SCORED):
            last_input_position=w.PREFIX+j
            logits=engine.forward([tokens[w.PREFIX+j]],cache,spec['r'])[0,0].float()
            lp=torch.log_softmax(logits,-1).cpu().numpy()
            logits_cpu=logits.cpu().numpy()
            distribution_digest.update(lp.tobytes())
            if not bool(np.isfinite(lp).all()):raise w.NumericalFailure('Nonfinite log probability')
            target=tokens[w.PREFIX+j+1]
            ref=baseline['log_probs'][j] if baseline is not None else None
            kl=None if ref is None else float(np.sum(np.exp(ref)*(ref-lp),dtype=np.float32))
            if kl is not None and not math.isfinite(kl):raise w.NumericalFailure('Nonfinite KL')
            row=dict(condition_id=spec['id'],book_id=doc['book_id'],r=spec['r'],mode=spec['mode'],bits=spec['bits'],target=spec['target'],
                     input_position=w.PREFIX+j,target_position=w.PREFIX+j+1,target_id=target,
                     target_log_probability=float(lp[target]),native_target_log_probability=None if ref is None else float(ref[target]),
                     kl_native_to_condition=kl,
                     max_abs_logit_vs_native=None if baseline is None else float(np.abs(baseline['logits'][j]-logits_cpu).max()),status='FINITE')
            if row['max_abs_logit_vs_native'] is not None and not math.isfinite(row['max_abs_logit_vs_native']):
                raise w.NumericalFailure('Nonfinite logit difference')
            rows.append(row);w.append_json(root/'per_token.jsonl',row)
            if spec['bits'] is None:
                native_lp.append(lp);native_logits.append(logits_cpu)
            if j==127:
                np.savez_compressed(root/'diagnostics'/f"{spec['id']}-p896.npz",log_probs=lp,
                    native_log_probs=ref if ref is not None else (lp if spec['bits'] is None else np.empty(0,dtype=np.float32)),
                    logits=logits_cpu,native_logits=baseline['logits'][j] if baseline is not None else (logits_cpu if spec['bits'] is None else np.empty(0,dtype=np.float32)),
                    target_id=np.array(target),target_position=np.array(w.PREFIX+j+1))
            post_step(cache,False,root,ident)
        cache.validate(256,1023);rng_unchanged(before)
        w.require([x['target_position'] for x in rows]==list(range(769,1024)), 'P1 scoring positions')
        result=dict(spec,status='COMPLETE_FINITE',scored=len(rows),nll=-math.fsum(x['target_log_probability'] for x in rows)/255,
                    mean_kl=None if baseline is None else math.fsum(x['kl_native_to_condition'] for x in rows)/255,
                    seconds=time.monotonic()-start,forwards=engine.forward_count-n0,
                    final_cache_sha256=digest_cache(cache),distribution_sha256=distribution_digest.hexdigest(),
                    rng_unchanged=True,**cache.accounting(),**d.device_peaks())
        result['seconds']=time.monotonic()-start
        result['ppl']=w.safe_ppl(result['nll'])
        w.require(result['forwards']==256,'P1 forward count')
        return result,dict(log_probs=np.stack(native_lp),logits=np.stack(native_logits)) if native_lp else None
    except BaseException as exc:
        result=dict(spec,status=classify_error(exc),scored=len(rows),forwards=engine.forward_count-n0,
                    last_input_position=last_input_position,seconds=time.monotonic()-start,
                    error_type=type(exc).__name__,error=w.redact(str(exc)),traceback=w.redact(traceback.format_exc()))
        return result,None
    finally:
        del cache
        d.device_cleanup()


def directional(values, positive, negative):
    if len(values)!=8:return dict(status='NOT_EVALUABLE',n=len(values),values=values)
    mean=math.fsum(values)/8
    return dict(status=positive if mean>=.005 else negative if mean<=-.005 else 'WITHIN_BAND',n=8,mean=mean,values=values)


def bootstrap_interval(values):
    w.require(bool(values) and all(math.isfinite(x) for x in values), 'Bootstrap requires finite values')
    a=np.asarray(values,dtype=np.float64)
    draws=np.random.default_rng(w.BOOT_SEED).integers(0,len(a),(10000,len(a)))
    return np.quantile(a[draws].mean(axis=1),[.025,.975]).tolist()


def quality_screen(values, books):
    w.require(len(values)==len(books) and len(set(books))==len(books), 'Quality pairing inventory')
    w.require(set(books)<=set(DIAGNOSTIC_IDS), 'Unplanned quality book')
    w.require(all(math.isfinite(x) for x in values), 'Nonfinite paired difference')
    result=dict(status='NOT_EVALUABLE',n=len(values),paired_books=books,per_book_delta_nll=values,
                missing_book_ids=[b for b in DIAGNOSTIC_IDS if b not in books],
                per_book_relative_ppl=[w.safe_ppl(x)-1 if w.safe_ppl(x) is not None else None for x in values],
                books_above_5pct=sum(x>QUALITY_MARGIN for x in values),
                max_relative_ppl=None,mean_delta_nll=None,relative_ppl=None,bootstrap_95=None,relative_ppl_95=None)
    if not values:return result
    ci=bootstrap_interval(values);mean=math.fsum(values)/len(values)
    ratio=w.safe_ppl(mean);max_ratio=w.safe_ppl(max(values))
    result.update(mean_delta_nll=mean,relative_ppl=ratio-1 if ratio is not None else None,
                  max_relative_ppl=max_ratio-1 if max_ratio is not None else None,bootstrap_95=ci,
                  relative_ppl_95=[w.safe_ppl(x)-1 if w.safe_ppl(x) is not None else None for x in ci])
    if len(values)==8:
        result['status']='WITHIN_5PCT' if ci[1]<=QUALITY_MARGIN else 'EXCEEDS_5PCT' if ci[0]>QUALITY_MARGIN else 'UNCERTAIN'
    return result


def summarize_p1(root, planned):
    w.require(planned==p1_schedule(),'KV01 schedule changed')
    rows=w.read_rows(root/'per_document.jsonl');tokens=w.read_rows(root/'per_token.jsonl')
    specs={s['id']:s for s in planned};seen={x['id']:x for x in rows}
    w.require(len(seen)==len(rows) and set(seen)<=set(specs),'Duplicate/unplanned KV01 results')
    by_id=defaultdict(list)
    for t in tokens:
        w.require(t['condition_id'] in specs,'Unexpected token condition')
        by_id[t['condition_id']].append(t)
    docs={x['book_id']:x for x in MATERIALS['diagnostic']}
    norms=w.read_rows(root/'norm_snapshots.jsonl');transfers=w.read_rows(root/'norm_transfers.jsonl')
    interventions=w.read_rows(root/'interventions.jsonl')
    w.require(all(x.get('condition_id') in specs for x in norms+transfers+interventions),'Unknown diagnostic condition')
    for ident,row in seen.items():
        spec=specs[ident]
        w.require(all(row.get(k)==v for k,v in spec.items()),'KV01 metadata mismatch')
        ts=by_id[ident]
        w.require(len(ts)==row['scored'] and [t['target_position'] for t in ts]==list(range(769,769+len(ts))),'Partial/complete token positions')
        for t in ts:
            w.require(t['target_id']==docs[row['book_id']]['tokens'][t['target_position']],'Target token mismatch')
            w.require(all(t.get(k)==spec[k] for k in ('book_id','r','mode','bits','target')),'Token condition metadata')
            w.require(math.isfinite(t['target_log_probability']),'Nonfinite recorded log probability')
        if row['status']=='COMPLETE_FINITE':
            w.require(row['scored']==255 and row['forwards']==256,'Complete KV01 count')
            calculated=-math.fsum(t['target_log_probability'] for t in ts)/255
            w.require(math.isfinite(row['nll']) and abs(row['nll']-calculated)<1e-10,'NLL/token mismatch')
            w.require(all(row.get(k)==v for k,v in expected_accounting(spec).items()),'Side accounting mismatch')
            validate_norm_inventory(norms,transfers,ident,'condition_id',spec['r'],spec['mode'])
            validate_interventions([x for x in interventions if x['condition_id']==ident],spec['r'],spec['target'],spec['bits'])
            for x in transfers:
                if x['condition_id']==ident:
                    w.require(x['quantization_applied']==(x['kv'] in selected_parts(spec['target'],spec['bits'])),'Transfer target metadata')
    lookup={(x['book_id'],x['target']):x for x in rows if x['status']=='COMPLETE_FINITE'}
    for row in rows:
        native=lookup.get((row['book_id'],'NATIVE'))
        if row['target']=='NATIVE':continue
        nts=by_id[native['id']] if native else []
        for j,t in enumerate(by_id[row['id']]):
            ref=t.get('native_target_log_probability')
            w.require((ref is None and native is None) or (native is not None and ref==nts[j]['target_log_probability']),'Native reference mismatch')
    def contrast(required,coefficients,positive,negative,formula):
        books=[book for book in DIAGNOSTIC_IDS if all((book,t) in lookup for t in required)]
        values=[math.fsum(c*lookup[(book,t)]['nll'] for c,t in zip(coefficients,required)) for book in books]
        answer=directional(values,positive,negative)
        answer.update(paired_books=books,missing_book_ids=[b for b in DIAGNOSTIC_IDS if b not in books],
            mean=math.fsum(values)/len(values) if values else None,
            bootstrap_95=bootstrap_interval(values) if values else None,formula=formula,
            estimate_scope='all_fixed_books' if len(books)==8 else 'available_pairs_reference_only')
        return answer
    D=contrast(('K_ONLY','V_ONLY'),(1,-1),'K_LARGER','V_LARGER','NLL(K_ONLY)-NLL(V_ONLY)')
    I=contrast(('KV','K_ONLY','V_ONLY','NATIVE'),(1,-1,-1,1),'SUPERADDITIVE','SUBADDITIVE','NLL(KV)-NLL(K_ONLY)-NLL(V_ONLY)+NLL(NATIVE)')
    quality=[];condition=[]
    for target in TARGETS:
        books=[b for b in DIAGNOSTIC_IDS if (b,target) in lookup]
        values=[lookup[(b,target)]['nll'] for b in books];mean=math.fsum(values)/len(values) if values else None
        condition.append(dict(target=target,n=len(values),books=books,mean_nll=mean,ppl=w.safe_ppl(mean) if mean is not None else None))
        if target!='NATIVE':
            paired=[b for b in books if (b,'NATIVE') in lookup]
            deltas=[lookup[(b,target)]['nll']-lookup[(b,'NATIVE')]['nll'] for b in paired]
            quality.append(dict(target=target,**quality_screen(deltas,paired)))
    complete=len(rows)==len(planned) and all(x['status']=='COMPLETE_FINITE' for x in rows)
    report=dict(experiment=EXPERIMENT,overall_status='COMPLETE' if complete else 'INCONCLUSIVE',
        planned=len(planned),obtained=len(rows),status_counts=dict(Counter(x['status'] for x in rows)),
        not_run_ids=[s['id'] for s in planned if s['id'] not in seen],
        unresolved_ids=[x['id'] for x in rows if x['status']!='COMPLETE_FINITE'],
        observed_token_count=len(tokens),planned_token_count=8160,independent_books=8,
        legacy_p0_status=LEGACY_STATUS,legacy_full_p0_pass=False,D=D,I=I,
        quality_screens=quality,condition_nll=condition,quality_margin_relative_ppl=.05,
        quality_label='Exploratory per-comparison interval; not a simultaneous guarantee',
        norm_snapshot_count=len(norms),norm_transfer_count=len(transfers),intervention_slice_count=len(interventions),
        historical_parent=json.loads(IDENTITY_PATH.read_text()),historical_outcomes_pooled=False)
    w.write_json(root/'summary.json',report)
    return report


def execute_p1(engine,root,planned,docs):
    w.require(planned==p1_schedule(),'P1 execution schedule changed')
    bybook={x['book_id']:x for x in docs};baseline=None;baseline_key=None
    for number,spec in enumerate(planned,1):
        engine.condition_deadline=math.inf;engine.check_time()
        key=(spec['book_id'],spec['r'],spec['mode'])
        if spec['bits'] is None:
            baseline=None;baseline_key=key
        else:
            w.require(baseline_key==key, 'Native baseline belongs to a different book/r/mode')
        w.log(f"KV01 [{number}/{len(planned)}] {spec['id']}")
        w.write_json(root/'active_case.json',dict(id=spec['id'],phase='KV01',status='STARTED',deadline_monotonic=time.monotonic()+TRAJECTORY_SECONDS))
        result,base=trajectory(engine,root,bybook[spec['book_id']],spec,baseline)
        w.append_json(root/'per_document.jsonl',result)
        w.append_json(root/'cache_counters.jsonl',{k:v for k,v in result.items() if k in ('id','status','slots','update_count','compressed_tokens','planned_components','planned_components_by_kv','compressed_tokens_by_kv','target','planned_bytes','floating_cache_bytes','final_cache_sha256')})
        w.write_json(root/'active_case.json',dict(id=spec['id'],status=result['status']))
        if base is not None:baseline=base
        if result['status']!='COMPLETE_FINITE':w.append_json(root/'failures.jsonl',dict(phase='KV01',**result))
        report=summarize_p1(root,planned)
        payload_check(root)
        w.write_json(root/'checkpoint.json',dict(recorded_conditions=number,total=len(planned),last_id=spec['id']))
        w.checkpoint_zip(root)
        w.log(f"  {result['status']} / {result['scored']} predictions / {result['seconds']:.1f}s")
        if result['status'] not in ('COMPLETE_FINITE','NUMERICAL_FAILURE'):
            return 'BLOCKED_IMPLEMENTATION' if result['status']=='BLOCKED_IMPLEMENTATION' else 'INCONCLUSIVE'
    return summarize_p1(root,planned)['overall_status']


def prepare_model(root, started):
    d.install_packages();numeric_imports()
    w.require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(),'Select a BF16-capable Colab GPU')
    packages={n:importlib.metadata.version(n) for n in w.PACKAGE_NAMES}
    w.require(packages==d.PARENT['packages'],'Dependency versions differ')
    w.require(torch.__version__==packages['torch'] and np.__version__==packages['numpy'],'Loaded versions differ')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True);torch.manual_seed(w.SEED)
    env=dict(packages=packages,gpu=torch.cuda.get_device_name(0),gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
             cuda=torch.version.cuda,python=sys.version)
    w.write_json(root/'environment.json',env)
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig,AutoTokenizer,AutoModelForCausalLM
    secret=os.environ.get('HF_TOKEN') or None
    source=root/'sources';source.mkdir(exist_ok=True)
    for name,digest in w.SOURCE_HASHES.items():
        path=Path(hf_hub_download(w.MODEL,name,revision=w.MODEL_REV,token=secret))
        w.require(w.sha(path.read_bytes())==digest,'Pinned source mismatch: '+name);shutil.copy2(path,source/name)
    config=AutoConfig.from_pretrained(w.MODEL,revision=w.MODEL_REV,token=secret,trust_remote_code=True)
    w.require((config.num_hidden_layers,config.num_key_value_heads,config.head_dim)==(24,16,128),'Model dimensions')
    w.require(all(x=='full_attention' for x in config.layer_types),'Attention types changed')
    tokenizer=AutoTokenizer.from_pretrained(w.MODEL,revision=w.MODEL_REV,token=secret,trust_remote_code=True)
    tok_hash=w.sha(tokenizer.backend_tokenizer.to_str().encode())
    w.require(tok_hash==d.PARENT['tokenizer_sha256'],'Tokenizer hash changed')
    config.total_ut_steps=4
    w.log('Loading pinned Ouro-1.4B BF16. P0KV must pass before the fixed 32 KV01 trajectories.')
    model=AutoModelForCausalLM.from_pretrained(w.MODEL,config=config,revision=w.MODEL_REV,token=secret,
        trust_remote_code=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa',low_cpu_mem_usage=True).to('cuda').eval()
    param_dtypes=dict(Counter(str(p.dtype) for p in model.parameters()))
    buffers=dict(Counter(str(b.dtype) for b in model.buffers() if b.is_floating_point()))
    w.require(model.dtype==torch.bfloat16 and set(param_dtypes)=={'torch.bfloat16'},'Wrong BF16 parameter dtype')
    w.require(set(buffers)<={'torch.bfloat16','torch.float32'},'Unexpected floating buffer dtype')
    if time.monotonic()-started>PREP_SECONDS:raise w.BudgetStop('Preparation cap')
    base=importlib.import_module(type(model).__module__).UniversalTransformerCache
    engine=RouteEngine(model,root,make_selective_adapter(base),base)
    w.write_json(root/'manifest.json',{**env,'experiment':EXPERIMENT,'version':VERSION,'run_id':root.parent.name,
        'model':w.MODEL,'model_revision':w.MODEL_REV,'tokenizer_sha256':tok_hash,'material_source':d.PARENT,
        'material_sha256':MATERIAL_SHA,'core_sha256':CORE_SHA,'diagnostic_helper_sha256':DIAGNOSTIC_SHA,
        'worker_sha256':w.sha(Path(__file__).read_bytes()),'source_hashes':w.SOURCE_HASHES,
        'historical_identity_sha256':IDENTITY_SHA,'quality_margin_relative_ppl':.05,
        'main_contrast':'NLL(K_ONLY)-NLL(V_ONLY)', 'interaction':'KV-K_ONLY-V_ONLY+NATIVE',
        'historical_outcomes_pooled':False,'targets':TARGETS,
        'execution_dtype':'BF16','parameter_dtype_counts':param_dtypes,'floating_buffer_dtype_counts':buffers,
        'legacy_p0_status':LEGACY_STATUS,'legacy_full_p0_pass':False,'p1_auto_start_requires':'PASS_ROUTE_KV01',
        'prepared_seconds':time.monotonic()-started,'independent_diagnostic_books':8,
        'readout':'exit_at_step=r-1; labels=None; use_weighted_exit=False',
        'caps_seconds':dict(prepare=PREP_SECONDS,p0kv=P0_SECONDS,kv01=P1_SECONDS,case=CASE_SECONDS,trajectory=TRAJECTORY_SECONDS),
        'artifact_bytes_cap':MAX_BYTES})
    return model,engine


def main():
    root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=True)
    (root/'p0r').mkdir(exist_ok=True);(root/'diagnostics').mkdir(exist_ok=True)
    started=time.monotonic();planned=p1_schedule();status='STARTING';p0='INCONCLUSIVE';p1_started=False
    w.write_json(root/'schedule.json',planned);w.write_json(root/'p0r/schedule.json',p0_schedule())
    w.write_json(root/'p0r/compatibility_schedule.json',compatibility_schedule())
    w.write_json(root/'materials.json',MATERIALS)
    w.write_json(root/'phase.json',dict(stage='PREPARE',deadline_monotonic=started+PREP_SECONDS))
    def terminate(signum,frame):raise w.BudgetStop('Launcher termination requested')
    signal.signal(signal.SIGTERM,terminate)
    try:
        verify_inputs()
        model,engine=prepare_model(root,started)
        with d.precision_context(dict(kernel='AUTO',gemm_reduced=True)) as actual:
            w.write_json(root/'execution_settings.json',actual)
            engine.stage_deadline=time.monotonic()+P0_SECONDS
            w.write_json(root/'phase.json',dict(stage='P0KV',deadline_monotonic=engine.stage_deadline))
            fixtures=w.cpu_fixtures()+w.gpu_codec_fixtures()+cache_fixtures()+selective_fixtures()
            p0=execute_p0(engine,root,MATERIALS['smoke'],fixtures)
            if p0!='PASS_ROUTE_KV01':
                status=p0
            else:
                engine.stage_deadline=time.monotonic()+P1_SECONDS
                w.write_json(root/'phase.json',dict(stage='KV01',deadline_monotonic=engine.stage_deadline))
                g4=[x['native_seconds'] for x in w.read_rows(root/'p0r/cases.jsonl') if x['group']=='G4']
                w.write_json(root/'time_estimate.json',dict(native_only_projection_seconds=len(planned)*math.fsum(g4)/len(g4),
                    remaining_cap_seconds=P1_SECONDS,note='Projection excludes NSN and instrumentation overhead; not a stop rule.'))
                p1_started=True
                status=execute_p1(engine,root,planned,MATERIALS['diagnostic'])
    except BaseException as exc:
        status=classify_error(exc,not p1_started)
        if status in ('PARTIAL','RESOURCE_FAILURE'):status='INCONCLUSIVE'
        w.append_json(root/'failures.jsonl',dict(phase='WORKER',status=status,error_type=type(exc).__name__,
            error=w.redact(str(exc)),traceback=w.redact(traceback.format_exc())))
        if not p1_started:w.write_json(root/'p0r/gate_error.json',dict(status=status,error_type=type(exc).__name__,error=w.redact(str(exc))))
        w.log(traceback.format_exc())
    finally:
        try:
            numeric_imports()
            p0=current_p0(root)['p0r_route_status']
            summary=summarize_p1(root,planned)
            summary.update(p0r_route_status=p0,run_status=status,p1_started=p1_started)
            w.write_json(root/'summary.json',summary)
        except BaseException as exc:
            status='BLOCKED_IMPLEMENTATION'
            w.append_json(root/'failures.jsonl',dict(phase='FINAL_SUMMARY',error=w.redact(str(exc))))
            if (root/'summary.json').exists():shutil.copy2(root/'summary.json',root/'unverified_summary.json')
            w.write_json(root/'summary.json',dict(experiment=EXPERIMENT,overall_status='INCONCLUSIVE',
                run_status=status,p0r_route_status=p0,p1_started=p1_started,legacy_full_p0_pass=False,
                summary_error=w.redact(str(exc)),note='Final verification failed; unverified summary retained separately.'))
        w.write_json(root/'run_status.json',dict(status=status,experiment=EXPERIMENT,p0r_route_status=p0,
            p1_started=p1_started,legacy_p0_status=LEGACY_STATUS,legacy_full_p0_pass=False,
            elapsed_seconds=time.monotonic()-started,time_utc=datetime.now(timezone.utc).isoformat()))
        w.log(f'Closed: {status} / P0KV={p0} / KV01 started={p1_started}')


if __name__=='__main__':main()
