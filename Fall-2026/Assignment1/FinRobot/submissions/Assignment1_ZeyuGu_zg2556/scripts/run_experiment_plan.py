"""Execute a saved ordered experiment plan; every paid process remains serial.

A plan contains argv lists and explicit completion-manifest paths. Completed
steps are logged as reuse, never called an independent fresh experiment.
Credentials are supplied at invocation and never written to this log.
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path
from audit_io import append_jsonl, utc_now


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--upstream', type=Path)
    args = parser.parse_args()
    failures=[]
    for step in json.loads(args.plan.read_text(encoding='utf-8')):
        marker = Path(step['completion_manifest'])
        if marker.exists():
            data = json.loads(marker.read_text(encoding='utf-8'))
            if data.get(step.get('state_field', 'state')) in step['completed_states']:
                append_jsonl(args.log, {'time':utc_now(), 'step':step['id'], 'event':'reuse_completed_step'})
                continue
        command=[str(args.upstream) if token=='{upstream}' and args.upstream else token for token in step['argv']]
        if '{upstream}' in command:
            raise ValueError('Plan requires --upstream')
        if marker.exists() and step.get('resume_if_incomplete'):
            command.append('--resume')
        argv = [sys.executable] + command
        if step.get('uses_credentials', True):
            argv += ['--config', args.config]
        append_jsonl(args.log, {'time':utc_now(), 'step':step['id'], 'event':'start', 'argv_without_credentials':step['argv']})
        result = subprocess.run(argv)
        append_jsonl(args.log, {'time':utc_now(), 'step':step['id'], 'event':'exit', 'exit_code':result.returncode})
        if result.returncode:
            failures.append(step['id'])
            if not step.get('continue_on_error',False):
                raise SystemExit(result.returncode)
    if failures:
        print('Incomplete plan steps:', ', '.join(failures))
        raise SystemExit(1)


if __name__ == '__main__':
    main()
