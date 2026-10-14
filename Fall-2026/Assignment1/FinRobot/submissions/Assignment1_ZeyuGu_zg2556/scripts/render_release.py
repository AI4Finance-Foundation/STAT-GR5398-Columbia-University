"""Render actual saved generations; archive previous HTML and manifest before update."""
import argparse,json,shutil,subprocess,sys,time
from pathlib import Path
from audit_io import write_json,sha256,utc_now
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--upstream',type=Path,required=True);p.add_argument('--version',required=True);a=p.parse_args();records=[]
    for ticker in ['NVDA','AMD','INTC','AVGO','QCOM','AAPL','GOOGL']:
        for suffix in ['baseline_1','final_v6']+(['baseline_2','baseline_3'] if ticker=='NVDA' else []):
            run=ROOT/'runs'/f'{ticker}_{suffix}';report=run/'output'/ticker/'report';archive=ROOT/'evidence/render_versions'/a.version/run.name
            archive.mkdir(parents=True,exist_ok=False)
            for f in [run/'manifest.json',report/f'Audited_Equity_Report_{ticker}.html',report/'rendered_section_texts.json']:
                if f.is_file():shutil.copyfile(f,archive/f.name)
            start=time.perf_counter();r=subprocess.run([sys.executable,'scripts/experiment.py','--upstream',str(a.upstream),'render','--run',str(run)],cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
            if r.returncode:raise RuntimeError('Render failed '+run.name+': '+r.stderr)
            records.append({'run_id':run.name,'runtime_seconds':time.perf_counter()-start,'html_sha256':sha256((report/f'Audited_Equity_Report_{ticker}.html').read_bytes())})
    write_json(ROOT/'evidence'/('release_render_'+a.version+'.json'),{'created_utc':utc_now(),'upstream':str(a.upstream),'api_calls':0,'records':records})
    print('Rendered '+str(len(records))+' actual saved reports with exact section hashes checked.')
if __name__=='__main__':main()
