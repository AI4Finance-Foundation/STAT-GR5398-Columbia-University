"""Replay all seven frozen data pipelines, independent arithmetic and saved text.

No credentials or model calls. Network socket creation is rejected in each child.
Output must be a new directory. Original snapshots and text are never overwritten.
"""
import argparse,json,os,shutil,subprocess,sys,time
from pathlib import Path
from audit_io import write_json,sha256,utc_now
ROOT=Path(__file__).resolve().parents[1]
SNAPSHOTS={'NVDA':'NVDA','AMD':'AMD','INTC':'INTC_news_v2','AAPL':'AAPL_news_v2','GOOGL':'GOOGL','AVGO':'AVGO_public_v1','QCOM':'QCOM_public_v1'}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--upstream',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--render',action='store_true');a=p.parse_args()
    a.output=a.output.resolve();a.upstream=a.upstream.resolve();a.output.mkdir(parents=True,exist_ok=False);records=[]
    wrapper="import runpy,socket,sys\ndef forbidden(*a,**k): raise RuntimeError('Network forbidden during offline reproduction')\nsocket.create_connection=forbidden\nsocket.socket.connect=forbidden\nsys.argv=sys.argv[1:]\nsys.path.insert(0,'scripts')\nrunpy.run_path(sys.argv[0],run_name='__main__')"
    def run(argv,label):
        start=time.perf_counter();r=subprocess.run([sys.executable,'-c',wrapper,*map(str,argv)],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',env=dict(os.environ,PYTHONIOENCODING='utf-8'))
        (a.output/(label+'.log')).write_text(r.stdout+r.stderr,encoding='utf-8')
        records.append({'label':label,'argv':list(map(str,argv)),'exit_code':r.returncode,'runtime_seconds':time.perf_counter()-start})
        write_json(a.output/'progress.json',records)
        if r.returncode:raise RuntimeError('Offline reproduction failed: '+label)
    for ticker,name in SNAPSHOTS.items():
        source=ROOT/'snapshots'/name;dest=a.output/'snapshots'/name
        if ticker in {'AVGO','QCOM'}:
            run(['scripts/public_semiconductor_pipeline.py','--upstream',a.upstream,'--ticker',ticker,'--output',dest],ticker+'_data')
        else:
            m=json.loads((source/'manifest.json').read_text());s=m['parameters']
            run(['scripts/snapshot_pipeline.py','--upstream',a.upstream,'--ticker',ticker,'--company-name',s['company_name'],'--peers',*s['peers'],'--snapshot',source,'--mode','replay','--replay-output',dest,'--as-of',s['as_of_utc'],'--years-limit',s['years_limit'],'--news-days-back',s['news_days_back'],'--news-limit',s['news_limit'],'--news-snapshot',ROOT/'evidence/issuer_news'/ticker/'articles.json'],ticker+'_data')
        matched=[]
        for f in (source/'analysis').iterdir():
            if f.is_file():
                g=dest/'analysis'/f.name
                if not g.exists() or f.read_bytes()!=g.read_bytes():raise ValueError('Deterministic analysis mismatch: '+str(f))
                matched.append({'name':f.name,'sha256':sha256(g.read_bytes())})
        write_json(a.output/(ticker+'_analysis_comparison.json'),matched)
        if a.render:
            for version in ['baseline_1','final_v6']:
                original=ROOT/'runs'/f'{ticker}_{version}';copy=a.output/'runs'/original.name
                shutil.copytree(original,copy)
                run(['scripts/experiment.py','--upstream',a.upstream,'render','--run',copy],original.name+'_render')
    write_json(a.output/'manifest.json',{'status':'PASS','completed_utc':utc_now(),'upstream':str(a.upstream),'network_disabled':True,'new_model_calls':0,'snapshots_checked':7,'rendered_saved_text_reports':14 if a.render else 0,'records':records})
    print('PASS: seven frozen pipelines reproduced with network disabled; no model calls.')
if __name__=='__main__':main()
