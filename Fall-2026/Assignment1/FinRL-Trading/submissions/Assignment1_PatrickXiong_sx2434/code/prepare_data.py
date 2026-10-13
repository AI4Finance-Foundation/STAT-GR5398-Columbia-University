"""Build the four locally licensed inputs from WRDS CSV exports.

Daily CSV field labels are normalized; membership and name history are kept in
WRDS export form. No credentials or remote data transfer occur in this script.
"""
from pathlib import Path
import argparse,shutil,json,hashlib
import pandas as pd
p=argparse.ArgumentParser();p.add_argument('--daily',nargs='+',required=True);p.add_argument('--membership',required=True);p.add_argument('--names',required=True);p.add_argument('--delisting',required=True);p.add_argument('--output-dir',required=True)
a=p.parse_args();out=Path(a.output_dir).resolve()
if out.exists() and any(out.iterdir()):raise ValueError('Use a new empty input directory.')
out.mkdir(parents=True,exist_ok=True)
frames=[]
for path in a.daily:
 d=pd.read_csv(path,low_memory=False);d.columns=d.columns.str.lower();d.dlycaldt=pd.to_datetime(d.dlycaldt);frames.append(d)
pd.concat(frames,ignore_index=True).to_parquet(out/'crsp_daily_raw.parquet',index=False)
for src,name in [(a.membership,'crsp_sp500_members.csv.gz'),(a.names,'crsp_names_all.csv'),(a.delisting,'crsp_delisting.csv')]:shutil.copy2(src,out/name)
manifest={q.name:{'bytes':q.stat().st_size,'sha256':hashlib.sha256(q.read_bytes()).hexdigest()} for q in out.iterdir() if q.is_file()}
(out/'frozen_manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2))
