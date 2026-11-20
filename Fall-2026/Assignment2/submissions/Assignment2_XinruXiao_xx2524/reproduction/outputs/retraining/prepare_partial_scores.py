"""Atomic score snapshot for fully trained early evaluation periods only."""
from pathlib import Path
import pandas as pd,json,argparse,hashlib
p=argparse.ArgumentParser();p.add_argument('--task-root',default='.');p.add_argument('--through',required=True);a=p.parse_args();r=Path(a.task_root).resolve();o=r/'outputs/retraining';months=pd.period_range('2017-01',a.through,freq='M').astype(str).tolist();records=[]
for version in ['baseline','improved']:
 for month in months:assert (o/(version+'_parts')/(month+'.parquet')).exists(),(version,month)
for version in ['baseline','improved']:
 target=o/(version+'_scores.parquet')
 if json.loads((o/'training_status.json').read_text()).get('status')=='complete':raise SystemExit('Full training already complete; do not replace full score tables.')
 frames=[]
 for month in months:
  f=o/(version+'_parts')/(month+'.parquet');assert f.exists(),f
  frames.append(pd.read_parquet(f))
 df=pd.concat(frames,ignore_index=True);assert not df.duplicated(['month','symbol','bucket']).any()
 tmp=target.with_suffix('.staged.parquet');df.to_parquet(tmp,index=False);tmp.replace(target)
 records.append({'version':version,'through':a.through,'months':len(months),'rows':len(df),'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'purpose':'temporary snapshot for fully trained early periods; full training overwrites with all 119 months before final evaluation and validation'})
history=o/'staged_score_snapshots.json'
previous=json.loads(history.read_text()) if history.exists() else []
history.write_text(json.dumps(previous+records,indent=2));print(records)
