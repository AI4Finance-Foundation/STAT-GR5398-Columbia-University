"""Run Qwen then DeepSeek, each holding the shared GPU lock independently."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', type=Path, required=True)
    ap.add_argument('--plan-sha', required=True)
    ap.add_argument('--bundle-manifest-sha', required=True)
    args = ap.parse_args()
    bundle = args.bundle.resolve()
    manifest_path = bundle / 'EXECUTION_BUNDLE_MANIFEST.json'
    if sha(manifest_path) != args.bundle_manifest_sha:
        raise ValueError('Execution bundle SHA differs')
    manifest = json.loads(manifest_path.read_text())
    for name, expected in manifest['files'].items():
        if Path(name).name != name or sha(bundle / name) != expected:
            raise ValueError('Execution bundle file changed')
    completed = []
    for model in ['qwen', 'deepseek']:
        status_path = bundle / ('run_' + model) / 'status.json'
        if status_path.exists() and json.loads(status_path.read_text()).get('stage') == 'completed':
            completed.append(model)
            continue
        log_path = bundle / (model + '_worker.log')
        with log_path.open('a') as stream:
            process = subprocess.Popen([sys.executable, '-u', str(bundle / 'hitl_retrospective_gpu_worker.py'),
                                        '--bundle', str(bundle), '--plan-sha', args.plan_sha, '--model', model,
                                        '--bundle-manifest-sha', args.bundle_manifest_sha],
                                       stdin=subprocess.DEVNULL, stdout=stream, stderr=stream)
            last = 0
            while process.poll() is None:
                now = time.monotonic()
                if now - last >= 15:
                    save(bundle / 'controller_status.json', {'stage': 'running', 'controller_pid': os.getpid(),
                         'worker_pid': process.pid, 'current_model': model, 'completed_models': completed,
                         'updated_at_utc': datetime.now(timezone.utc).isoformat(), 'test_read': False, 'new_training': False})
                    last = now
                time.sleep(1)
        if process.returncode:
            save(bundle / 'controller_status.json', {'stage': 'failed', 'model': model, 'exit_code': process.returncode,
                 'completed_models': completed, 'outputs_preserved': True, 'updated_at_utc': datetime.now(timezone.utc).isoformat()})
            return process.returncode
        completed.append(model)
    save(bundle / 'controller_status.json', {'stage': 'completed', 'completed_models': completed,
         'new_validation_outputs': 48, 'new_training': False, 'test_read': False,
         'updated_at_utc': datetime.now(timezone.utc).isoformat(), 'qualitative_effect_review_pending': True})
    return 0

if __name__ == '__main__':
    sys.exit(main())
