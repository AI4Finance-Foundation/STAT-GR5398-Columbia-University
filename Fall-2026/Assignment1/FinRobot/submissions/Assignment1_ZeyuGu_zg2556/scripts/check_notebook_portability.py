"""Execute real independent-valuation notebook cells in an isolated relocated tree.

No upstream checkout, data API, model call or optional Python package is used.
Only the notebook, independent script and its three frozen input files are copied.
The original valuation CSVs are hashed before and after and never rewritten.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(actual, expected, location='results'):
    differences=[]
    if isinstance(expected,dict):
        if set(actual)!=set(expected): raise AssertionError('Different keys at '+location)
        for key in expected: differences+=compare(actual[key],expected[key],location+'.'+key)
    elif isinstance(expected,list):
        if len(actual)!=len(expected): raise AssertionError('Different length at '+location)
        for i,(a,b) in enumerate(zip(actual,expected)): differences+=compare(a,b,f'{location}[{i}]')
    elif isinstance(expected,(int,float)) and not isinstance(expected,bool):
        if not math.isclose(actual,expected,rel_tol=1e-12,abs_tol=1e-7):
            raise AssertionError('Numerical mismatch at '+location)
        if actual!=expected: differences.append({'field':location,'recomputed':actual,'saved':expected,'delta':actual-expected})
    elif actual != expected:
        raise AssertionError('Value mismatch at '+location)
    return differences


def check(root):
    rows=[]
    notebooks=sorted(root.glob('snapshots/*/valuation/independent_valuation.ipynb'))
    if not notebooks:
        raise ValueError('No actual valuation notebooks to validate')
    all_csvs={str(p.relative_to(root)):digest(p) for p in root.glob('snapshots/*/valuation/*.csv')}
    child='''import json, socket, sys
from pathlib import Path
def forbidden(*args, **kwargs):
    raise RuntimeError("Network forbidden in independent offline notebook verification")
socket.create_connection = forbidden
socket.socket.connect = forbidden
notebook = json.loads(Path("independent_valuation.ipynb").read_text(encoding="utf-8"))
namespace = {"__name__": "__notebook_portability_check__"}
for cell in notebook["cells"]:
    if cell["cell_type"] == "code":
        exec(compile("".join(cell["source"]), "relocated_notebook", "exec"), namespace)
if any(name == "pandas" or name.startswith("modules.valuation_engine") for name in sys.modules):
    raise AssertionError("Independent notebook imported an engine dependency")
print(json.dumps(namespace["results"], sort_keys=True, allow_nan=False))
'''
    for notebook in notebooks:
        snapshot=notebook.parent.parent
        original=json.loads((notebook.parent/'independent_valuation.json').read_text(encoding='utf-8'))['independent_results']
        code=''.join(''.join(c['source']) for c in json.loads(notebook.read_text(encoding='utf-8'))['cells'] if c['cell_type']=='code')
        if str(root) in code or '.cache' in code or 'C:\\Users' in code:
            raise ValueError('Notebook contains a machine-specific executable path: '+str(notebook))
        files=[root/'scripts/independent_valuation.py',notebook,snapshot/'valuation_inputs.json',
               snapshot/'analysis/financial_metrics_and_forecasts.csv',snapshot/'analysis/key_metrics_raw_data.csv']
        with tempfile.TemporaryDirectory(prefix='finrobot_relocated_notebook_') as directory:
            relocated=Path(directory)/'submission'
            for source in files:
                target=relocated/source.relative_to(root)
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(source,target)
            proc=subprocess.run([sys.executable,'-I','-c',child],cwd=relocated/notebook.parent.relative_to(root),
                                capture_output=True,text=True,encoding='utf-8',check=True)
            actual=json.loads(proc.stdout)
        differences=compare(actual,original)
        rows.append({'snapshot':snapshot.name,'notebook':notebook.relative_to(root).as_posix(),
                     'notebook_sha256':digest(notebook),'independent_result_numeric_match':True,
                     'bitwise_numeric_match':not differences,'floating_point_differences':differences,
                     'relocated_without_upstream':True,'network_disabled':True,
                     'source_inputs':{p.relative_to(root).as_posix():digest(p) for p in files}})
    after={str(p.relative_to(root)):digest(p) for p in root.glob('snapshots/*/valuation/*.csv')}
    if after != all_csvs:
        raise AssertionError('Frozen valuation CSVs changed during portability verification')
    return {'checked_at_utc':datetime.now(timezone.utc).isoformat(),'status':'PASS',
            'notebooks_checked':len(rows),'valuation_csvs_unchanged':True,'valuation_csv_sha256':all_csvs,
            'python_version':sys.version,'relative_tolerance':1e-12,'absolute_tolerance':1e-7,
            'method':'Execute original notebook code cells in fresh Python -I, relocated minimal submission, sockets forbidden; deep result comparison with floating-point tolerance. No kernel rendering was needed for numeric cells.',
            'checks':rows}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('evidence/notebook_portability.json'))
    args=parser.parse_args()
    result=check(Path(__file__).resolve().parents[1])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(f"PASS: {result['notebooks_checked']} relocated notebooks; all valuation CSV hashes unchanged.")


if __name__=='__main__':
    main()
