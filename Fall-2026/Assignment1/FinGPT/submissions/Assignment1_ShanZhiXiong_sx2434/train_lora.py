"""Reproduce preserved retrospective training in a fresh output directory.

Default: print a CPU-only plan. This packaging wrapper produced none of the
reported historical runs. --execute is for a deliberate future reproduction.
No test inputs or network downloads are used.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone

ROOT = PurePosixPath('/root/shared-nvme/AI4FinRuntime')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', choices=['qwen', 'deepseek', 'llama'], required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--execute', action='store_true')
    args = ap.parse_args()
    cfg = json.loads(Path(__file__).with_name('config.json').read_text(encoding='utf-8'))
    out = PurePosixPath(args.out)
    if not out.is_absolute() or '..' in out.parts or not out.is_relative_to(ROOT) or out == ROOT:
        raise ValueError('Fresh output must be an absolute path beneath the runtime data volume')
    entry = cfg['models'][args.model]
    if args.model != 'llama':
        command = [str(ROOT / cfg['python_relative']), '-u',
                   str(ROOT / entry['backend_script']), '--model', args.model, '--out', str(out)]
        bundle = None
    else:
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', out.name):
            raise ValueError('Use a short alphanumeric fresh output name')
        bundle = ROOT / 'setup' / ('course_reproduction_' + out.name)
        command = [str(ROOT / cfg['python_relative']), '-u',
                   str(bundle / 'llama_full_int8_train.py'), '--model', 'llama', '--out', str(out)]
    summary = {'model_id': entry['model_id'], 'command': command,
               'fresh_bundle': str(bundle) if bundle else None,
               'data_policy': cfg['methodology_status'], 'test_read': False,
               'execute_requested': args.execute, 'new_wrapper_not_historic_execution': True,
               'original_outputs_preserved': True}
    if not args.execute:
        print(json.dumps(summary, indent=2)); return 0
    if sys.platform != 'linux':
        raise RuntimeError('Execute only on the original Linux runtime')
    if Path(str(out)).exists():
        raise ValueError('Refusing an existing output directory')
    if args.model != 'llama':
        # The preserved wrapper checks sources/prepared data, holds the GPU lock,
        # rejects active compute, and uses offline local caches.
        wrapper = Path(__file__).with_name('legacy_training_dispatch.py')
        if sha(wrapper) != cfg['legacy_dispatch_sha256']:
            raise ValueError('Preserved dispatch changed')
        spec = importlib.util.spec_from_file_location('legacy_training_dispatch', wrapper)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        sys.argv = [str(wrapper), '--config', str(Path(__file__).with_name('config.json')),
                    '--model', args.model, '--out', str(out), '--execute']
        return module.main()
    source = Path(str(ROOT / entry['source_bundle']))
    plan_path = source / 'FROZEN_TRAINING_PLAN.json'
    if sha(plan_path) != entry['source_plan_sha256']:
        raise ValueError('Executed Llama training plan changed')
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    for name, digest in plan['dependency_sha256'].items():
        if Path(name).name != name or sha(source / name) != digest:
            raise ValueError('Executed Llama source changed')
    target = Path(str(bundle))
    if target.exists():
        raise ValueError('Refusing an existing reproduction bundle')
    target.mkdir()
    for name in plan['dependency_sha256']:
        shutil.copyfile(source / name, target / name)
    plan.update(run_id=target.name, remote_bundle=str(target), output_directory=str(out),
                frozen_at_utc=datetime.now(timezone.utc).isoformat(),
                reproduction_of_plan_sha256=entry['source_plan_sha256'])
    frozen = target / 'FROZEN_TRAINING_PLAN.json'
    frozen.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    env = os.environ.copy()
    env.update(FINGPT_TRAINING_PLAN_SHA=sha(frozen),
               HF_HOME=str(ROOT / 'huggingface'), HF_ENDPOINT='https://huggingface.co',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TMPDIR=str(ROOT / 'tmp'),
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    # Actual preflight owns the lock for the entire child run and rechecks
    # scope, weights, versions, all1230 targets, and persistently records access.
    return subprocess.run(command, env=env, check=False).returncode

if __name__ == '__main__':
    sys.exit(main())
