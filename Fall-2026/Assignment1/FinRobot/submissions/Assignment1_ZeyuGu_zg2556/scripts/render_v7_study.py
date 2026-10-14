"""Render the completed development treatment without fetching or regenerating text."""
import subprocess,sys,time
from pathlib import Path
from audit_io import write_json
ROOT=Path(__file__).resolve().parents[1]
UP=Path.home()/'.cache/finrobot-zg2556/clean-verify-v7-1'
rows=[]
for ticker in ['NVDA','AMD','INTC','AVGO','QCOM','AAPL','GOOGL']:
    name=f'{ticker}_final_v7_1';start=time.perf_counter()
    subprocess.run([sys.executable,'scripts/experiment.py','--upstream',str(UP),
        'render','--run',f'runs/{name}'],cwd=ROOT,check=True)
    rows.append({'run_id':name,'runtime_seconds':time.perf_counter()-start})
write_json(ROOT/'evidence/render_v7_study.json',{'records':rows,
    'scope':'Development V7.1 candidate reports; canonical published/local release selection is separate. Exact source/HTML hashes are asserted by experiment.py.'})
print('Rendered seven development reports with exact-source assertions.')
