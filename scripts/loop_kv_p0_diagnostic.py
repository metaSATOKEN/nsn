# LOOP-KV-P0-DIAG-01 / v0.1 / Kentaro Sato / 2026-09-30
# This module imports no numerical libraries or model at import time.
import os
import sys
import json
import time
import math
import signal
import shutil
import traceback
import subprocess
import importlib
import importlib.metadata
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime, timezone

import loop_kv_pilot as w

EXPERIMENT = 'LOOP-KV-P0-DIAG-01'
VERSION = '0.1'
PREP_SECONDS = 1800
DIAG_SECONDS = 1200
CASE_SECONDS = 180
CORE_SHA = '4312155fc121d926f0a9bd794a62758f7acab634f8991d671e60fe22a7b0b1c9'
MATERIAL = json.loads('{\n  "book_id": "37013",\n  "split": "train",\n  "file": "train/37013.txt",\n  "title": "The Pleasures of the Country by Harriet Myrtle",\n  "publication_date": "1851",\n  "text_sha256": "23067064f55adc225897bd8867111934c38e2858de8a59f7db5c5467c1d38a9a",\n  "offset": 0,\n  "rank_attempt": 0,\n  "tokens": [\n    2738,\n    16980,\n    198,\n    18082,\n    766,\n    411,\n    6374,\n    647,\n    462,\n    2959,\n    33717,\n    28,\n    14872,\n    48019,\n    284,\n    260,\n    8646,\n    201,\n    198,\n    16535,\n    4663,\n    32334,\n    16593,\n    4004,\n    12733,\n    418,\n    201,\n    198,\n    5031,\n    1742,\n    3065,\n    30,\n    10605,\n    24901,\n    4055,\n    6503,\n    30,\n    4205,\n    18682,\n    18682,\n    18682,\n    18682,\n    18682,\n    216,\n    6462,\n    18682,\n    216,\n    377,\n    4796,\n    2211,\n    42606,\n    6016,\n    6462,\n    3214,\n    5633,\n    10895,\n    73,\n    18682,\n    216,\n    14599,\n    18682,\n    216,\n    5022,\n    2460,\n    20813,\n    12111,\n    9882,\n    5633,\n    55,\n    20045,\n    7627,\n    4796,\n    28,\n    18682,\n    216,\n    18172,\n    18682,\n    216,\n    372,\n    12086,\n    30,\n    407,\n    2285,\n    5135,\n    3438,\n    34475,\n    16895,\n    4796,\n    30,\n    18682,\n    23799,\n    25438,\n    414,\n    20128,\n    339,\n    16421,\n    69,\n    17136,\n    28074,\n    18172,\n    530,\n    18692,\n    62,\n    452,\n    4192,\n    50,\n    16090,\n    30,\n    18682,\n    216,\n    5665,\n    13832,\n    30,\n    18682,\n    216,\n    933,\n    39392,\n    1409,\n    77,\n    18682,\n    216,\n    4528,\n    42,\n    23799,\n    5236,\n    6279,\n    73,\n    7485,\n    3214\n  ],\n  "parent_full_token_sha256": "acf76d08910c552b7047b4b17e022b3897a93240073d4654d00dcb0f2d28831f",\n  "token_sha256": "876dd571f1fd41614bc5d2452a9aa5289cfceb935c3348dc22d4ac0247658efa"\n}')
PARENT = json.loads('{\n  "run_id": "loop-kv-p0_p1a-20260930-073005-5928f5a5",\n  "original_zip_sha256": "f79154199da43865f8bc6995c667e82b5818ad67b2b9a68771df968c7500a0b6",\n  "worker_sha256": "4312155fc121d926f0a9bd794a62758f7acab634f8991d671e60fe22a7b0b1c9",\n  "packages": {\n    "torch": "2.11.0+cu128",\n    "transformers": "4.56.2",\n    "huggingface-hub": "0.36.2",\n    "accelerate": "1.14.0",\n    "tokenizers": "0.22.2",\n    "safetensors": "0.8.0",\n    "numpy": "2.1.3",\n    "sentencepiece": "0.2.2"\n  },\n  "gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition",\n  "tokenizer_sha256": "fbd16291c3d91d530a306acbef718fa37eb7d9f1beeff0a3293f664688ece2e6",\n  "model_revision": "fcfcfbc429c7b10ee73dfc718033351d8936d2a8"\n}')


def numeric_imports():
    global torch
    w.numeric_imports()
    torch = w.torch


def schedule():
    rows = []
    for r in (1, 4):
        for label, kernel, reduced in [('A', 'AUTO', True), ('B', 'AUTO', False),
                                        ('C', 'MATH', True), ('D', 'MATH', False)]:
            rows.append({'id': f'r{r}-BF16-{label}', 'r': r, 'dtype': 'BF16',
                         'label': label, 'kernel': kernel, 'gemm_reduced': reduced,
                         'bits': None, 'book_id': MATERIAL['book_id'],
                         'expected_forwards': 265})
    for r in (1, 4):
        rows.append({'id': f'r{r}-FP32-F', 'r': r, 'dtype': 'FP32', 'label': 'F',
                     'kernel': 'AUTO', 'gemm_reduced': True, 'bits': None,
                     'book_id': MATERIAL['book_id'], 'expected_forwards': 265})
    return rows


def thresholds(dtype):
    w.require(dtype in ('BF16', 'FP32'), 'Unknown diagnostic dtype')
    return dict(zip(('kl', 'abs_delta_nll', 'max_abs_logit'),
                    (1e-4, 0.005, 0.25) if dtype == 'BF16' else (1e-6, 1e-4, 1e-3)))


def assess(metrics, dtype):
    bounds = thresholds(dtype)
    values = {'kl': metrics['kl'], 'abs_delta_nll': abs(metrics['delta_nll']),
              'max_abs_logit': metrics['max_abs_logit']}
    unmet = [k for k, v in values.items() if not math.isfinite(v) or v > bounds[k]]
    return {'within_tolerance': not unmet, 'unmet': unmet, 'thresholds': bounds}


def settings():
    c = torch.backends.cuda
    return {'flash': c.flash_sdp_enabled(), 'efficient': c.mem_efficient_sdp_enabled(),
            'math': c.math_sdp_enabled(), 'cudnn': c.cudnn_sdp_enabled(),
            'gemm_bf16_reduced': c.matmul.allow_bf16_reduced_precision_reduction,
            'math_sdp_reduced': c.fp16_bf16_reduction_math_sdp_allowed(),
            'matmul_tf32': c.matmul.allow_tf32,
            'cudnn_tf32': torch.backends.cudnn.allow_tf32}


@contextmanager
def precision_context(spec):
    from torch.nn.attention import sdpa_kernel, SDPBackend
    w.require(spec['kernel'] in ('AUTO', 'MATH'), 'Unknown SDPA selector')
    old = settings()
    backends = [SDPBackend.MATH] if spec['kernel'] == 'MATH' else [
        SDPBackend.MATH, SDPBackend.FLASH_ATTENTION,
        SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION]
    try:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = spec['gemm_reduced']
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
        with sdpa_kernel(backends):
            current = settings()
            w.require(current['gemm_bf16_reduced'] == spec['gemm_reduced'] and
                      not current['math_sdp_reduced'], 'Precision flags did not take effect')
            w.require(not current['matmul_tf32'] and not current['cudnn_tf32'], 'TF32 must remain OFF')
            if spec['kernel'] == 'MATH':
                w.require(current['math'] and not any(current[k] for k in ('flash', 'efficient', 'cudnn')),
                          'MATH-only selection did not take effect')
            else:
                w.require(all(current[k] for k in ('math', 'flash', 'efficient', 'cudnn')),
                          'AUTO selector must enable the four standard backends')
            yield current
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = old['gemm_bf16_reduced']
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(old['math_sdp_reduced'])
        w.require(settings() == old, 'Precision settings were not restored after condition')


class DiagnosticEngine(w.Engine):
    def __init__(self, *args):
        super().__init__(*args)
        self.case_deadline = math.inf
        self.forward_count = 0

    def check_time(self):
        super().check_time()
        if time.monotonic() > self.case_deadline:
            raise TimeoutError('Diagnostic condition exceeded 3 minutes')

    def forward(self, *args, **kwargs):
        output = super().forward(*args, **kwargs)
        self.forward_count += 1
        return output


def require_finite(logits, snapshot):
    if not bool(torch.isfinite(logits).all()) or any(
            not bool(torch.isfinite(t).all()) for pair in snapshot for t in pair):
        raise w.NumericalFailure('Nonfinite native logits/KV in diagnostic')


def validate_trace(records, r):
    slots = set(range(w.LAYERS*r))
    actual = {s: [] for s in slots}
    for row in records:
        w.require(row['slot'] in slots, 'Extra trace slot')
        actual[row['slot']].append(row)
    for slot, rows in actual.items():
        w.require(len(rows) == 66, 'Trace update count mismatch')
        old = 0
        for index, row in enumerate(rows):
            added = 64 if index == 0 else 1
            w.require(row['loop'] == slot//w.LAYERS and row['layer'] == slot % w.LAYERS,
                      'Trace coordinate mismatch')
            w.require(row['old'] == old and row['added'] == added and row['new'] == old+added,
                      'Trace append mismatch')
            old += added
        w.require(old == 129, 'Trace final length mismatch')
    return {'slots': len(actual), 'updates_each': 66, 'final_length_each': 129}


def token_metrics(full, incremental, tokens, case_id):
    w.require(full.shape == incremental.shape and full.shape[0] == 129 and len(tokens) == 129,
              'Diagnostic scorer shape/position mismatch')
    a = torch.log_softmax(full.float(), dim=-1)
    b = torch.log_softmax(incremental.float(), dim=-1)
    kl = (a.exp()*(a-b)).sum(-1)
    # Keep the original scorer's dtype for logit subtraction, then serialize FP32.
    maximum = (full-incremental).abs().float().amax(-1)
    tops = full.argmax(-1) == incremental.argmax(-1)
    result = []
    for i in range(129):
        target = tokens[i+1] if i < 128 else None
        row = {'case_id': case_id, 'input_position': i, 'input_token': tokens[i],
               'target_position': i+1 if target is not None else None,
               'target_token': target, 'region': 'prefill' if i < 64 else 'decode',
               'kl': float(kl[i]), 'max_abs_logit': float(maximum[i]),
               'top1_equal': bool(tops[i])}
        if target is not None:
            row.update(full_nll=float(-a[i, target]), incremental_nll=float(-b[i, target]))
            row['delta_nll'] = row['incremental_nll']-row['full_nll']
        result.append(row)
    return result


def summarize_regions(rows):
    output = {}
    for region in ('prefill', 'decode'):
        subset = [x for x in rows if x['region'] == region]
        scored = [x for x in subset if x['target_token'] is not None]
        output[region] = {'distribution_positions': len(subset), 'nll_predictions': len(scored),
                          'mean_kl': math.fsum(x['kl'] for x in subset)/len(subset),
                          'max_abs_logit': max(x['max_abs_logit'] for x in subset),
                          'mean_delta_nll': math.fsum(x['delta_nll'] for x in scored)/len(scored),
                          'top1_equal': sum(x['top1_equal'] for x in subset)}
    return output


def measure_case(engine, root, spec, doc):
    tokens = doc['tokens']
    r = spec['r']
    engine.set_r(r)
    baseline, saved_kv, _, _ = engine.smoke_run(tokens, r, 'official')
    require_finite(baseline, saved_kv)
    for route, trace in [('official', False), ('adapter', False), ('adapter', True)]:
        label = 'official_repeat' if route == 'official' else 'adapter_trace_' + str(trace).lower()
        got, kv, _, records = engine.smoke_run(tokens, r, route, trace)
        require_finite(got, kv)
        check = {'case_id': spec['id'], 'route': label,
                 'logits_bitwise': torch.equal(baseline, got),
                 'kv_bitwise': w.snapshots_equal(saved_kv, kv),
                 'rng_unchanged': True}  # smoke_run asserts CPU and CUDA RNG equality.
        if trace:
            w.append_json(root/'cache_trace.jsonl', {'case_id': spec['id'], 'records': records})
            check.update(validate_trace(records, r))
        check['status'] = 'PASS' if check['logits_bitwise'] and check['kv_bitwise'] else 'MISMATCH'
        w.append_json(root/'checks.jsonl', check)
        w.require(check['status'] == 'PASS', f"Native route mismatch: {label}")
        del got, kv
    full = engine.forward(tokens, None, r, keep=0)[0].cpu()
    require_finite(full, [])
    metrics = w.scalar_metrics(full, baseline, tokens)
    if any(not math.isfinite(v) for v in metrics.values()):
        raise w.NumericalFailure('Nonfinite comparison metric despite finite native logits')
    rows = token_metrics(full, baseline, tokens, spec['id'])
    for row in rows:
        w.append_json(root/'per_token.jsonl', row)
    return metrics, summarize_regions(rows)


def build_summary(planned, cases):
    ids = [x['id'] for x in planned]
    w.require(len(ids) == 10 and len(set(ids)) == 10, 'Diagnostic schedule is not the fixed 10 cases')
    got = {x['id']: x for x in cases}
    w.require(len(got) == len(cases) and set(got) <= set(ids), 'Duplicate/unplanned result IDs')
    measured = {i: x for i, x in got.items() if x['status'] == 'MEASURED'}
    result = {'experiment': EXPERIMENT, 'planned_cases': 10, 'recorded_cases': len(got),
              'finite_measured_cases': len(measured), 'independent_documents': 1,
              'not_run_ids': [i for i in ids if i not in got],
              'unresolved_ids': [i for i in ids if i in got and i not in measured],
              'p1_automatically_started': False, 'full_p0_completed': False,
              'comparisons': [], 'candidate_configs': []}
    complete_bf16 = all(f'r{r}-BF16-{label}' in measured for r in (1, 4) for label in 'ABCD')
    for label in 'ABCD':
        rows = [measured.get(f'r{r}-BF16-{label}') for r in (1, 4)]
        if all(x and x['within_tolerance'] for x in rows):
            result['candidate_configs'].append(label)
    for r in (1, 4):
        for old, new, factor in [('A', 'B', 'GEMM'), ('A', 'C', 'KERNEL'),
                                  ('B', 'D', 'KERNEL'), ('C', 'D', 'GEMM')]:
            a, b = measured.get(f'r{r}-BF16-{old}'), measured.get(f'r{r}-BF16-{new}')
            if a is None or b is None:
                result['comparisons'].append({'r': r, 'contrast': old+'->'+new, 'status': 'UNRESOLVED'})
                continue
            result['comparisons'].append({'r': r, 'contrast': old+'->'+new, 'factor': factor,
                'status': 'DESCRIPTIVE', 'delta_kl': b['metrics']['kl']-a['metrics']['kl'],
                'delta_abs_nll_gap': abs(b['metrics']['delta_nll'])-abs(a['metrics']['delta_nll']),
                'delta_max_logit_gap': b['metrics']['max_abs_logit']-a['metrics']['max_abs_logit'],
                'from_within_tolerance': a['within_tolerance'], 'to_within_tolerance': b['within_tolerance']})
    a = measured.get('r1-BF16-A')
    fp32 = [measured.get(f'r{r}-FP32-F') for r in (1, 4)]
    result['D0'] = ('NOT_EVALUABLE' if a is None else
                    'NOT_REPRODUCED' if a['within_tolerance'] else 'THRESHOLD_EXCEEDANCE_REPRODUCED')
    result['D2'] = ('NOT_EVALUABLE' if any(x is None for x in fp32) else
                    'BOTH_WITHIN_TOLERANCE' if all(x['within_tolerance'] for x in fp32) else 'FP32_REFERENCE_EXCEEDED')
    if not complete_bf16:
        result['next_step'] = 'INCONCLUSIVE'
    elif 'A' in result['candidate_configs']:
        result['next_step'] = 'REPRODUCIBILITY_REVIEW'
    elif result['candidate_configs']:
        result['next_step'] = 'FULL_P0_REVALIDATION_CANDIDATE'
        result['preferred_candidate'] = next(x for x in 'BCD' if x in result['candidate_configs'])
    else:
        result['next_step'] = 'NO_BF16_CANDIDATE'
    result['overall_status'] = 'COMPLETED_DIAGNOSTIC' if len(measured) == 10 else 'INCONCLUSIVE'
    result['note'] = 'Candidate means only this 1-document r1/r4 diagnostic, not full P0 PASS or causal identification.'
    return result


def device_enter():
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def device_peaks():
    return {'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved()}


def device_cleanup():
    torch.cuda.empty_cache()


def execute_cases(engine, model, root, planned, doc):
    status = 'STARTING'
    for number, spec in enumerate(planned, 1):
        engine.case_deadline = math.inf
        engine.condition_deadline = math.inf
        engine.check_time()
        engine.case_deadline = time.monotonic()+CASE_SECONDS
        engine.forward_count = 0
        case_start = time.monotonic()
        row = {**spec, 'status': 'STARTED', 'deadline_monotonic': engine.case_deadline}
        w.write_json(root/'active_case.json', row)
        w.log(f"[{number}/10] {spec['id']} / {spec['kernel']} / GEMM reduced={spec['gemm_reduced']}")
        stage_stop = False
        try:
            dtype = torch.bfloat16 if spec['dtype'] == 'BF16' else torch.float32
            model.to(dtype=dtype)
            device_enter()
            with precision_context(spec) as actual:
                row['actual_settings'] = actual
                w.write_json(root/'active_case.json', row)
                metrics, regions = measure_case(engine, root, spec, doc)
                engine.check_time()
            row.update(status='MEASURED', metrics=metrics, regions=regions, **assess(metrics, spec['dtype']))
            row['floor_status'] = 'FLOOR_WITHIN_TOLERANCE' if row['within_tolerance'] else 'FLOOR_EXCEEDS_TOLERANCE'
            w.require(engine.forward_count == spec['expected_forwards'], 'Unexpected forward count')
        except w.NumericalFailure as e:
            row.update(status='NUMERICAL_FAILURE', error=w.redact(str(e)))
        except (w.BudgetStop, TimeoutError) as e:
            row.update(status='TIME_CAP', error=w.redact(str(e)))
            stage_stop = True
        except BaseException as e:
            row.update(status='BLOCKED_IMPLEMENTATION', error_type=type(e).__name__,
                       error=w.redact(str(e)), traceback=w.redact(traceback.format_exc()))
            stage_stop = True
        row.update(seconds=time.monotonic()-case_start, forwards=engine.forward_count)
        try:
            row.update(device_peaks())
        except BaseException as e:
            row.update(status='BLOCKED_IMPLEMENTATION', memory_stats_error=w.redact(str(e)))
            stage_stop = True
        w.append_json(root/'cases.jsonl', row)
        w.write_json(root/'active_case.json', row)
        if row['status'] != 'MEASURED':
            w.append_json(root/'failures.jsonl', row)
        w.write_json(root/'summary.json', build_summary(planned, w.read_rows(root/'cases.jsonl')))
        w.checkpoint_zip(root)
        w.log(f"  {row['status']} / {row.get('floor_status','unresolved')} / {row['seconds']:.1f}s")
        if stage_stop:
            status = 'BLOCKED_IMPLEMENTATION' if row['status'] == 'BLOCKED_IMPLEMENTATION' else 'INCONCLUSIVE'
            break
        device_cleanup()
    else:
        status = build_summary(planned, w.read_rows(root/'cases.jsonl'))['overall_status']
    return status


def verify_inputs():
    w.require(w.sha(Path(w.__file__).read_bytes()) == CORE_SHA, 'Frozen helper changed')
    w.require(MATERIAL['book_id'] == '37013' and len(MATERIAL['tokens']) == 129 and MATERIAL['offset'] == 0,
              'Wrong fixed smoke material')
    w.require(w.token_sha(MATERIAL['tokens']) == MATERIAL['token_sha256'], 'Embedded token hash mismatch')
    w.require(MATERIAL['token_sha256'] == '876dd571f1fd41614bc5d2452a9aa5289cfceb935c3348dc22d4ac0247658efa',
              'Fixed token column changed')
    w.require(PARENT['worker_sha256'] == CORE_SHA and PARENT['model_revision'] == w.MODEL_REV,
              'Parent identity mismatch')


def install_packages():
    # Install before importing torch/numpy so the child uses the resolved versions.
    w.require(importlib.metadata.version('torch') == PARENT['packages']['torch'],
              'Colab PyTorch differs from the original run; it will not be replaced automatically')
    specs = [f'{n}=={v}' for n, v in PARENT['packages'].items() if n != 'torch']
    w.log('Installing original-run dependency versions; keeping Colab PyTorch.')
    p = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *specs],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=PREP_SECONDS)
    w.log(p.stdout)
    w.require(p.returncode == 0, 'Pinned dependency installation failed')


def main():
    root = Path(sys.argv[1])
    root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    w.write_json(root/'phase.json', {'stage': 'PREPARE', 'deadline_monotonic': started+PREP_SECONDS})
    status = 'STARTING'
    planned = schedule()
    w.write_json(root/'schedule.json', planned)
    w.write_json(root/'material.json', MATERIAL)
    w.write_json(root/'parent_identity.json', PARENT)
    def terminate(signum, frame):
        raise w.BudgetStop('Launcher requested diagnostic termination')
    signal.signal(signal.SIGTERM, terminate)
    try:
        verify_inputs()
        install_packages()
        numeric_imports()
        w.require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), 'A native-BF16 Colab GPU is required')
        packages = {n: importlib.metadata.version(n) for n in w.PACKAGE_NAMES}
        w.require(packages == PARENT['packages'], 'Original dependency versions differ')
        w.require(torch.__version__ == packages['torch'] and w.np.__version__ == packages['numpy'],
                  'Loaded libraries and installed versions differ')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(w.SEED)
        initial_settings = settings()
        from huggingface_hub import hf_hub_download
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        secret = os.environ.get('HF_TOKEN') or None
        source = root/'sources'; source.mkdir(exist_ok=True)
        for name, digest in w.SOURCE_HASHES.items():
            path = Path(hf_hub_download(w.MODEL, name, revision=w.MODEL_REV, token=secret))
            w.require(w.sha(path.read_bytes()) == digest, f'Pinned model source mismatch: {name}')
            shutil.copy2(path, source/name)
        config = AutoConfig.from_pretrained(w.MODEL, revision=w.MODEL_REV, token=secret, trust_remote_code=True)
        w.require((config.num_hidden_layers, config.num_key_value_heads, config.head_dim) ==
                  (w.LAYERS, w.HEADS, w.DIM), 'Model dimensions differ')
        w.require(all(t == 'full_attention' for t in config.layer_types), 'Unexpected attention types')
        tokenizer = AutoTokenizer.from_pretrained(w.MODEL, revision=w.MODEL_REV, token=secret, trust_remote_code=True)
        tok_hash = w.sha(tokenizer.backend_tokenizer.to_str().encode())
        w.require(tok_hash == PARENT['tokenizer_sha256'], 'Original tokenizer hash differs')
        config.total_ut_steps = 4
        w.log('Loading Ouro-1.4B BF16 for native-only diagnostics.')
        model = AutoModelForCausalLM.from_pretrained(w.MODEL, config=config, revision=w.MODEL_REV,
                    token=secret, trust_remote_code=True, torch_dtype=torch.bfloat16,
                    attn_implementation='sdpa', low_cpu_mem_usage=True).to('cuda').eval()
        w.require(time.monotonic()-started <= PREP_SECONDS, 'Preparation time cap exceeded')
        official = importlib.import_module(type(model).__module__)
        base = official.UniversalTransformerCache
        engine = DiagnosticEngine(model, root, w.make_adapter_class(base), base)
        engine.stage_deadline = time.monotonic()+DIAG_SECONDS
        w.write_json(root/'phase.json', {'stage': 'DIAGNOSTIC', 'deadline_monotonic': engine.stage_deadline})
        manifest = {'experiment': EXPERIMENT, 'version': VERSION, 'run_id': root.parent.name,
                    'parent_run_id': PARENT['run_id'], 'model': w.MODEL, 'model_revision': w.MODEL_REV,
                    'tokenizer_sha256': tok_hash, 'helper_sha256': CORE_SHA,
                    'diagnostic_sha256': w.sha(Path(__file__).read_bytes()), 'packages': packages,
                    'gpu': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda,
                    'same_gpu_name_as_parent': torch.cuda.get_device_name(0) == PARENT['gpu'],
                    'python': sys.version, 'source_hashes': w.SOURCE_HASHES,
                    'initial_settings': initial_settings, 'weight_load_dtype': 'BF16',
                    'fp32_reference': 'BF16-loaded weights cast to FP32; no new checkpoint',
                    'quantization': 'NONE', 'readout': 'exit_at_step=r-1; labels=None',
                    'prepared_seconds': time.monotonic()-started,
                    'caps': {'prepare': PREP_SECONDS, 'diagnostic': DIAG_SECONDS, 'case': CASE_SECONDS},
                    'p1_auto_start': False, 'independent_documents': 1}
        w.write_json(root/'manifest.json', manifest)
        status = execute_cases(engine, model, root, planned, MATERIAL)
    except BaseException as e:
        status = 'BLOCKED_IMPLEMENTATION' if isinstance(e, w.IntegrityError) else 'INCONCLUSIVE'
        w.append_json(root/'failures.jsonl', {'stage': 'worker', 'status': status,
                    'error_type': type(e).__name__, 'error': w.redact(str(e)),
                    'traceback': w.redact(traceback.format_exc())})
        w.log(traceback.format_exc())
    finally:
        try:
            w.write_json(root/'summary.json', build_summary(planned, w.read_rows(root/'cases.jsonl')))
        except BaseException as e:
            w.log(f'Final summary failed: {w.redact(str(e))}')
        w.write_json(root/'run_status.json', {'status': status, 'experiment': EXPERIMENT,
            'elapsed_seconds': time.monotonic()-started, 'time_utc': datetime.now(timezone.utc).isoformat(),
            'full_p0_pass': False, 'p1_started': False})
        w.log(f'Diagnostic closed: {status}. No P1 started. ZIP is created by the Colab cell.')


if __name__ == '__main__':
    main()
