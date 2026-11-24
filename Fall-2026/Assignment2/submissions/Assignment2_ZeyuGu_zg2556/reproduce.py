"""Reproduce development and validation results in an isolated copy of the course framework."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = HERE.parents[1] / 'source_code'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--out', type=Path, default=HERE/'reproduced')
    parser.add_argument('--with-bonus', action='store_true')
    args = parser.parse_args()
    source = args.source.resolve()
    dest = args.out.resolve()
    if not (source/'data/fundamentals.db').is_file():
        parser.error('Place the course data bundle in source_code/data first.')
    if dest.exists() and any(dest.iterdir()):
        parser.error('Use a new output directory to preserve previous results.')
    dest.mkdir(exist_ok=True, parents=True)
    env = dict(os.environ, PYTHONHASHSEED='0')
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp)/'source_code'
        shutil.copytree(source, src, ignore=shutil.ignore_patterns('output', 'data', '__pycache__'))
        (src/'data').symlink_to(source/'data', target_is_directory=True)
        shutil.copy2(HERE/'student_valuation.py', src/'valuation/student_valuation.py')
        def run(*args, stdout=None):
            subprocess.run([sys.executable, *map(str,args)], env=env, cwd=src, stdout=stdout, check=True)
        for period in ('dev', 'val'):
            for stage in ('sensitivity', 'evaluate', 'portfolio'):
                run(HERE/'research.py', stage, period)
        with (dest/'check_submission.txt').open('w') as stream:
            run('check_submission.py', stdout=stream)
        run(HERE/'official_reference.py', '--destination', src/'output')
        shutil.copytree(src/'output', dest/'output')
        (dest/'output/test_runs.jsonl').touch(exist_ok=True)
        if args.with_bonus:
            bonus = dest/'bonus'
            run(HERE/'bonus/retrain.py', '--source', src, '--out', bonus)
            run(HERE/'bonus/path_efficiency/run.py', '--source', src,
                '--control-results', bonus/'control', '--out', bonus/'path_efficiency')


if __name__ == '__main__':
    main()
