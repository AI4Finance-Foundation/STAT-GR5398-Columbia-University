"""New course-named entry point, NOT historically executed.

The real training implementation is preserved and pinned, not replaced with a
fake DeepSpeed run. Default prints a plan. --execute explicitly starts a fresh
legacy-methodology reproduction on the original Linux data-volume environment.
It does not train on a corrected dataset and does not open test data.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys


def inside(root, relative):
    p = (root / relative).resolve()
    if not p.is_relative_to(root.resolve()):
        raise ValueError('Configured path escaped runtime root')
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, default=Path(__file__).with_name('config.json'))
    ap.add_argument('--model', choices=['qwen', 'deepseek', 'llama'], required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--execute', action='store_true')
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    entry = cfg['models'][args.model]
    if entry.get('ready') is False:
        raise RuntimeError(entry['reason'])
    root_spec = PurePosixPath(cfg['server_root'])
    if not root_spec.is_absolute():
        raise ValueError('Linux runtime root must be absolute')
    relative_paths = [entry['backend_script'], cfg['python_relative'], cfg['gpu_lock_relative']]
    if any(PurePosixPath(p).is_absolute() or '..' in PurePosixPath(p).parts for p in relative_paths):
        raise ValueError('Runtime component must be a relative path within root')
    command = [str(root_spec / cfg['python_relative']), '-u', str(root_spec / entry['backend_script']),
               '--model', args.model, '--out', args.out]
    plan = {'command': command, 'execute_requested': args.execute,
            'model_id': entry['model_id'], 'data_policy': cfg['methodology_status'],
            'script_sha256': entry['backend_script_sha256'],
            'test_read': False, 'new_wrapper_not_historic_execution': True}
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return 0
    if sys.platform != 'linux':
        raise RuntimeError('Execute via the connected Linux server, not local Windows')
    root = Path(root_spec)
    backend = inside(root, entry['backend_script'])
    if str(root) != '/root/shared-nvme/AI4FinRuntime':
        raise ValueError('Historical backends have fixed server paths; arbitrary rebasing unsupported')
    out = Path(args.out).resolve()
    if not out.is_relative_to(root.resolve()) or out == root.resolve():
        raise ValueError('Fresh output must be beneath the data-volume runtime')
    if out.exists():
        raise RuntimeError('Refusing any existing output directory; all old results preserved')
    sources = [(backend, entry['backend_script_sha256'])]
    sources.extend((inside(root, relative), expected)
                   for relative, expected in entry.get('dependency_sources', {}).items())
    for name, item in entry['prepared_sources'].items():
        path = inside(root, item['path'])
        sources.append((path, item['sha256']))
    for path, expected in sources:
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Pinned source mismatch: ' + str(path))
    import fcntl
    with inside(root, cfg['gpu_lock_relative']).open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        active = subprocess.check_output(
            ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
        if active.strip():
            raise RuntimeError('GPU compute process already active')
        env = os.environ.copy()
        env.update(HF_HOME=str(root/'huggingface'), HF_HUB_OFFLINE='1',
                   TRANSFORMERS_OFFLINE='1', TMPDIR=str(root/'tmp'),
                   PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        # A fresh data-volume location is required; no weights are downloaded.
        (root/'tmp').mkdir(exist_ok=True)
        return subprocess.run(command, env=env, check=False).returncode


if __name__ == '__main__':
    sys.exit(main())
