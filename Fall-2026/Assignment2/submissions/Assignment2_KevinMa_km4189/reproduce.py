#!/usr/bin/env python3
"""Reproduce DEV/VAL in a disposable course-code copy. Never runs TEST."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent

def digest(path, text=False):
    b = path.read_bytes()
    return hashlib.sha256(b.replace(b'\r\n', b'\n') if text else b).hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--course-root', type=Path, default=HERE.parent.parent/'source_code')
    p.add_argument('--data-dir', type=Path, help='Extracted official data/ directory (default: course-root/data)')
    p.add_argument('--out', type=Path, default=Path.cwd()/'reproduced_pretest')
    p.add_argument('--verify-training', action='store_true', help='Retrain and verify the frozen banks; requires CPU PyTorch')
    p.add_argument('--sensitivity', action='store_true', help='Also retrain 22/25/28/30 percent DEV models, run 25/28 VAL and the early 22 percent internal variants; requires CPU PyTorch')
    p.add_argument('--check-only', action='store_true', help='Check fingerprints without running a backtest')
    a = p.parse_args()
    root = a.course_root.resolve(); data = (a.data_dir or root/'data').resolve()
    out = a.out.resolve()
    refs = json.loads((HERE/'course_fingerprints.json').read_text())
    errors=[]
    for base,key,is_text in [(root,'source_text_lf_sha256',True),(data,'data_sha256',False)]:
        for rel, expected in refs[key].items():
            f = base/rel
            if not f.is_file() or digest(f,is_text)!=expected:
                errors.append(str(f))
    if errors:
        raise SystemExit('Course files missing or differ from the original experiment snapshot:\n'+'\n'.join(errors))
    print('Course source and data fingerprints MATCH.', flush=True)
    if a.check_only: return
    if out.exists(): raise SystemExit(f'Choose a new --out path; will not overwrite {out}')
    out.mkdir(parents=True)
    env = dict(os.environ, PYTHONHASHSEED='5398')
    frozen = (HERE/'student_timing.py').read_text(encoding='utf-8')
    with tempfile.TemporaryDirectory(prefix='gr5398_') as tmp:
        work = Path(tmp)
        for rel in refs['source_text_lf_sha256']:
            dest=work/rel; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(root/rel,dest)
        shutil.copytree(data,work/'data')
        (work/'student_data').mkdir()
        for f in ('train_timing.py','ablation_timing.py'):
            shutil.copy2(HERE/f,work/f)
        target=work/'timing/student_timing.py'; target.write_text(frozen,encoding='utf-8')
        def run(*args):
            print('>', ' '.join(args),flush=True)
            subprocess.run([sys.executable,*args],cwd=work,env=env,check=True)
        def evaluate(period):
            run('evaluate_timing.py','--period',period)
            run('run_portfolio.py','--period',period,'--ablation','--timing','student')
        if a.verify_training: run('train_timing.py','--course-root','.','--verify')
        run('check_submission.py','--timing')
        evaluate('dev'); evaluate('val')
        shutil.move(str(work/'output'),str(out/'final_28'))
        if a.sensitivity:
            for vol in (.22,.25,.28,.30):
                target.write_text(re.sub(r'(?m)^TARGET_ANNUAL_VOL = .*$',f'TARGET_ANNUAL_VOL = {vol}',frozen),encoding='utf-8')
                run('train_timing.py','--course-root','.')
                evaluate('dev')
                if vol in (.25,.28): evaluate('val')
                if vol==.22: run('ablation_timing.py','--portfolio')
                dest=out/f'vol{round(vol*100)}'
                shutil.move(str(work/'output'),str(dest))
                shutil.copy2(target,dest/'student_timing.py')
    print('DEV/VAL reproduction output:',out)

if __name__=='__main__': main()
