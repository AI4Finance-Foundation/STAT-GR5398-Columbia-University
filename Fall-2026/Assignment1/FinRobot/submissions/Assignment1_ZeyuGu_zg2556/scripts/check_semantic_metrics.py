"""Recompute saved scorecards and independently check primitive claim counts."""
import argparse,csv,json,subprocess,sys
from collections import Counter
from pathlib import Path
from audit_io import sha256,write_json,utc_now
ROOT=Path(__file__).resolve().parents[1]

def rows(path):return list(csv.DictReader(path.open(encoding='utf-8',newline='')))
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    checks=[];counts=[]
    wrapper="import runpy,socket,sys,os\ndef no(*a,**k): raise RuntimeError('Metric reproduction prohibits network')\nsocket.create_connection=no\nsocket.socket.connect=no\nsys.argv=sys.argv[1:]\nsys.path.insert(0,os.path.dirname(sys.argv[0]))\nrunpy.run_path(sys.argv[0],run_name='__main__')"
    for version in ['v8_2_corrected','v9','v9_1']:
        pair=ROOT/f'config/grounding_reasoning_{version}_pairs.json'
        for kind,script,extra in [('comparison','grounding_audit.py',['compare']),('annotation','analyze_verifier_annotations.py',[])]:
            dest=a.output/(version+'_'+kind)
            command=[sys.executable,'-c',wrapper,str(ROOT/'scripts'/script),*extra,'--pairs',str(pair),'--output',str(dest)]
            proc=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
            (a.output/(version+'_'+kind+'.log')).write_text(proc.stdout+proc.stderr,encoding='utf-8')
            proc.check_returncode()
            original=ROOT/f'audits/grounding/{kind}_reasoning_{version}'
            assert sorted(p.name for p in original.glob('*.csv'))==sorted(p.name for p in dest.glob('*.csv'))
            for file in original.glob('*.csv'):
                assert file.read_bytes()==(dest/file.name).read_bytes(),str(file)
                checks.append({'version':version,'file':file.relative_to(ROOT).as_posix(),'sha256':sha256(file.read_bytes()),'byte_identical':True})
        pair_rows=json.loads(pair.read_text(encoding='utf-8'));score=rows(ROOT/f'audits/grounding/comparison_reasoning_{version}/before_after_scorecard.csv')
        for side in ['before','after']:
            raw=[r for pr in pair_rows for r in rows(ROOT/pr[side]/'claims.csv')]
            total=next(r for r in score if r['ticker']=='ALL' and r['section']=='ALL')
            tally=Counter(r['classification'] for r in raw);numeric=Counter(r['numeric_status'] for r in raw)
            assert len(raw)==int(total[side+'_total_claims'])
            for label in ['Supported','Unsupported','Contradicted']:assert tally[label]==int(total[side+'_'+label.lower()])
            for label in ['correct','incorrect','unverifiable']:assert numeric[label]==int(total[side+'_numeric_'+label])
            assert abs(float(total[side+'_grounding_rate'])-tally['Supported']/len(raw))<1e-12
            counts.append({'version':version,'side':side,'claims':len(raw),'classifications':dict(tally),'numeric_statuses':dict(numeric)})
    write_json(ROOT/'evidence/reproducibility/semantic_release_metrics.json',{'checked_utc':utc_now(),'passed':True,'network_disabled':True,
        'new_model_calls':0,'csv_checks':checks,'independent_primitive_counts':counts,
        'scope':'Exact metric reproduction from saved labels plus direct CSV counts, not another semantic adjudication.',
        'script_sha256':sha256(Path(__file__).read_bytes())})
    print(f'{len(checks)} scorecard CSVs byte-identical; primitive label/numeric counts independently matched.')
if __name__=='__main__':main()
