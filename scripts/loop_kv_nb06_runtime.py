"""NB06 isolated Colab runtime bootstrap. Stdlib only; never imports torch.

Install into a new per-run venv, then replace this process with the worker.
The original PREPARE deadline is shared by installation and model loading.
"""
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback

VERSION = '0.1.0'
INSTALLER_PIP = '25.1.1'
START_KEY = 'LOOP_KV_NB06_PREPARE_STARTED'
TORCH_WHEEL = ('https://download.pytorch.org/whl/cu128/'
    'torch-2.11.0%2Bcu128-cp313-cp313-manylinux_2_28_x86_64.whl'
    '#sha256=db964b33c55035a72ab3e2162287af8f1cc276039c65d015740cc88c26dcedf7')
PINS = {'torch': '2.11.0+cu128', 'triton': '3.6.0',
    'transformers': '4.45.1', 'huggingface-hub': '0.36.2',
    'accelerate': '1.14.0', 'tokenizers': '0.20.3',
    'safetensors': '0.8.0', 'numpy': '2.1.3', 'sentencepiece': '0.2.2'}


def clean(text):
    text = str(text)
    secret = os.environ.get('HF_TOKEN')
    if secret:
        text = text.replace(secret, '[REDACTED]')
    return re.sub(r'hf_[A-Za-z0-9]{20,}', '[REDACTED]', text)


def bounded_log(text):
    limit = 1024 * 1024
    data = text.encode()
    return text if len(data) <= limit else data[:limit].decode(errors="replace") + "\n[DIAGNOSTIC LOG TRUNCATED AT 1 MiB]\n"


def save(root, name, obj):
    path = root / name
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    os.replace(tmp, path)


def preparation_started():
    now = time.monotonic()
    start = float(os.environ.get(START_KEY, now))
    if not math.isfinite(start) or start < 0 or start > now:
        raise ValueError('Invalid shared preparation clock')
    return start


def remaining(deadline):
    left = deadline - time.monotonic()
    if left <= 0:
        raise TimeoutError('Preparation cap includes isolated runtime installation')
    return left


def inventory():
    packages = {}
    for name in PINS:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    try:
        bootstrap_pip = metadata.version('pip')
    except metadata.PackageNotFoundError:
        bootstrap_pip = None
    return dict(python=sys.version, executable=sys.executable, bootstrap_pip=bootstrap_pip,
        platform=sys.platform, machine=platform.machine(), packages=packages)


def require_bootstrap_pip():
    version = metadata.version('pip')
    match = re.match(r'^(\d+)\.(\d+)', version)
    if not match or tuple(map(int, match.groups())) < (22, 3):
        raise RuntimeError('Bootstrap pip >=22.3 is required for --python; base packages will not be upgraded')
    return version


def venv_command(runtime):
    # Colab's ensurepip failed in the returned run. Do not call it, even as a fallback.
    return [sys.executable, '-I', '-m', 'venv', '--without-pip', str(runtime)]


def install_command(python, requirements):
    # pip documents --python for environments that do not yet contain pip.
    return [sys.executable, '-I', '-m', 'pip', '--isolated', '--python', str(python),
        'install', '--disable-pip-version-check', '--no-cache-dir', '--only-binary=:all:',
        '--index-url', 'https://pypi.org/simple', *requirements]


def require_target():
    if (sys.platform != 'linux' or platform.machine() != 'x86_64'
            or platform.python_implementation() != 'CPython'
            or sys.version_info[:2] != (3, 13)):
        raise RuntimeError('This repair targets Colab CPython 3.13 / Linux x86_64; actual environment was saved.')


def package_environment():
    # No notebook package path, pip configuration or HF credential in pip children.
    # AC04's successful launcher preserved the host GPU loader environment.
    # In particular, LD_LIBRARY_PATH / LD_PRELOAD are NOT Python package paths.
    blocked = {'PYTHONPATH', 'PYTHONHOME', 'HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN',
        'VIRTUAL_ENV'}
    env = {k: v for k, v in os.environ.items()
        if k not in blocked and not k.startswith('PIP_')}
    env.update(PYTHONNOUSERSITE='1', PIP_CONFIG_FILE=os.devnull)
    return env


def host_gpu_report(root, deadline, env):
    # A missing nvidia-smi is diagnostic, not an independent acceptance gate.
    result = dict(loader_environment={k: clean(env[k]) for k in
        ('LD_LIBRARY_PATH', 'LD_PRELOAD', 'CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES') if k in env})
    try:
        cmd = ['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader,nounits']
        p = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=min(15, remaining(deadline)))
        result.update(returncode=p.returncode, output=clean(p.stdout))
    except Exception as exc:
        result.update(error_type=type(exc).__name__, error=clean(exc))
    save(root, 'runtime_gpu_host.json', result)


def cuda_probe(root, torch_module=None):
    """Record the exact CUDA failure in a short-lived process, before model work."""
    result = dict(status='UNAVAILABLE', model_calls=0, python=sys.executable,
        loader_environment={k: clean(os.environ[k]) for k in
            ('LD_LIBRARY_PATH', 'LD_PRELOAD', 'CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES') if k in os.environ})
    try:
        if torch_module is None:
            import torch as torch_module
        t = torch_module
        result.update(torch_version=t.__version__, cuda_build=t.version.cuda,
            cuda_available=bool(t.cuda.is_available()), device_count=int(t.cuda.device_count()))
        # init() gives an explicit diagnostic where is_available() merely returns False.
        t.cuda.init()
        result['bf16_supported'] = bool(t.cuda.is_bf16_supported())
        if result['cuda_available'] and result['bf16_supported']:
            result.update(status='AVAILABLE', gpu=t.cuda.get_device_name(0),
                capability=list(t.cuda.get_device_capability(0)))
    except Exception as exc:
        result.update(error_type=type(exc).__name__, error=clean(exc))
    save(Path(root), 'runtime_cuda_probe.json', result)
    return result


def run_step(root, name, command, deadline, env):
    save(root, 'runtime_step.json', dict(step=name, status='STARTED',
        command=command, deadline_monotonic=deadline))
    print('NB06 runtime:', name, flush=True)
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, timeout=remaining(deadline), env=env)
        (root / ('runtime_' + name + '.log')).write_text(bounded_log(clean(result.stdout)))
        if result.returncode:
            raise RuntimeError(f'Isolated runtime {name} failed; see runtime_{name}.log')
        remaining(deadline)
        return result.stdout
    except subprocess.TimeoutExpired as exc:
        output = exc.output or ''
        if isinstance(output, bytes):
            output = output.decode(errors='replace')
        (root / ('runtime_' + name + '.log')).write_text(bounded_log(clean(output)))
        raise


def verify_report(report, runtime):
    if report['packages'] != PINS:
        raise RuntimeError('Isolated package versions do not match the fixed NB06 versions')
    if Path(report['prefix']).absolute() != runtime.absolute() or report['prefix'] == report['base_prefix']:
        raise RuntimeError('Package validation did not run in the isolated venv')
    if report['user_site']:
        raise RuntimeError('User site-packages must be disabled')
    if report['installer_pip'] != INSTALLER_PIP:
        raise RuntimeError('Isolated pip version does not match the runtime bootstrap contract')


def main(root):
    from loop_kv_nb06_contract import CAPS, PLAN
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    started = preparation_started()
    deadline = started + CAPS['prepare']
    runtime = root.parent / 'runtime'
    python = runtime / 'bin' / 'python'
    save(root, 'phase.json', dict(stage='PREPARE', deadline_monotonic=deadline))
    save(root, 'runtime_before.json', inventory())
    save(root, 'runtime_plan.json', dict(version=VERSION, pins=PINS, installer_pip=INSTALLER_PIP, torch_wheel=TORCH_WHEEL,
        runtime_directory=str(runtime), started_monotonic=started,
        deadline_monotonic=deadline, base_environment_modified=False))
    def terminate(*_):
        raise TimeoutError('Launcher terminated runtime preparation')
    signal.signal(signal.SIGTERM, terminate)
    try:
        require_target()
        require_bootstrap_pip()
        if runtime.exists():
            raise RuntimeError('Refuse to reuse an existing runtime directory')
        env = package_environment()
        host_gpu_report(root, deadline, env)
        run_step(root, 'venv', venv_command(runtime), deadline, env)
        command = install_command(python, [f'pip=={INSTALLER_PIP}', TORCH_WHEEL] +
            [f'{name}=={version}' for name, version in PINS.items() if name != 'torch'])
        run_step(root, 'install', command, deadline, env)
        run_step(root, 'pip_check', [str(python), '-I', '-m', 'pip', '--isolated', 'check'], deadline, env)
        code = ('import importlib.metadata as m,json,sys,site;'
            f'names={list(PINS)!r};'
            'print(json.dumps(dict(packages={n:m.version(n) for n in names},installer_pip=m.version("pip"),'
            'all_packages=sorted([(d.metadata["Name"],d.version) for d in m.distributions()]),'
            'prefix=sys.prefix,base_prefix=sys.base_prefix,user_site=site.ENABLE_USER_SITE)))')
        report = json.loads(run_step(root, 'versions', [str(python), '-I', '-c', code], deadline, env))
        save(root, 'runtime_after.json', report)
        verify_report(report, runtime)
        run_step(root, 'cuda_probe', [str(python), '-I', str(Path(__file__).resolve()),
            '--cuda-probe', str(root)], min(deadline, time.monotonic() + 60), env)
        gpu = json.loads((root / 'runtime_cuda_probe.json').read_text())
        if gpu['status'] != 'AVAILABLE':
            raise RuntimeError('CUDA/BF16 is unavailable in the pinned environment; see runtime_cuda_probe.json and runtime_gpu_host.json')
        if shutil.disk_usage(root).free < PLAN['minimum_before_weights_free_bytes']:
            raise RuntimeError('Reserve at least 12 GiB after runtime installation for the pinned 7.77 GiB model and bounded evidence/ZIP')
        remaining(deadline)
        save(root, 'runtime_step.json', dict(step='READY', status='COMPLETE',
            elapsed_seconds=time.monotonic() - started, deadline_monotonic=deadline))
        # Use the identical interpreter for the worker and later independent verification.
        worker_env = env.copy()
        worker_env.update(VIRTUAL_ENV=str(runtime), **{START_KEY: str(started)})
        if os.environ.get('HF_TOKEN'):
            worker_env['HF_TOKEN'] = os.environ['HF_TOKEN']
        print('NB06 runtime ready: torch 2.11.0+cu128; starting the fixed experiment.', flush=True)
        os.execve(str(python), [str(python), '-u', str(root / 'loop_kv_nb06.py'), str(root)], worker_env)
    except BaseException as exc:
        save(root, 'runtime_error.json', dict(error_type=type(exc).__name__,
            error=clean(exc), traceback=clean(traceback.format_exc())))
        save(root, 'run_status.json', dict(status='INCONCLUSIVE', experiment='LOOP-KV-NB06',
            stage='RUNTIME_PREPARE', elapsed_seconds=time.monotonic() - started, model_calls=0))
        print('Runtime preparation stopped:', clean(exc), flush=True)
        return 1
    return 0


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--cuda-probe':
        cuda_probe(sys.argv[2])
    else:
        sys.exit(main(sys.argv[1]))
