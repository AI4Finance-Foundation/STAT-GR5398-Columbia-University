"""Fetch exact public source snapshots and regenerate prepared course inputs."""
import argparse,hashlib,json,shutil,subprocess,sys,urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument('--cache',required=True,type=Path);a=ap.parse_args();a.cache.mkdir(parents=True,exist_ok=True)
manifest=json.loads((ROOT/'data_sources.json').read_text())
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
for rec in manifest:
    dest=a.cache/rec['file']
    if dest.exists():
        assert digest(dest)==rec['sha256'],f'Existing source differs: {dest}'
        continue
    url=rec.get('url',rec.get('archive_url'))
    download=a.cache/('finrl_trading.7z' if 'archive_url' in rec else rec['file'])
    with urllib.request.urlopen(url,timeout=90) as response,download.open('wb') as handle:
        shutil.copyfileobj(response,handle)
    if 'archive_url' in rec:
        assert digest(download)==rec['archive_sha256']
        seven=shutil.which('7z');tar=shutil.which('bsdtar') or shutil.which('tar')
        if seven:subprocess.run([seven,'x',str(download),'-o'+str(a.cache),'-y'],check=True)
        elif tar:subprocess.run([tar,'-xf',str(download),'-C',str(a.cache)],check=True)
        else:raise RuntimeError('Install 7-Zip or bsdtar to extract the data archive.')
    assert digest(dest)==rec['sha256'],f'Download mismatch: {dest}'
subprocess.run([sys.executable,str(ROOT/'code/prepare_data.py'),'--db',str(a.cache/'finrl_trading.db'),
    '--fundamentals',str(a.cache/'fundamental_data_full.csv'),'--constituents',str(a.cache/'sp500_historical_constituents.csv'),
    '--output-dir',str(a.cache/'prepared'),'--start','2015-01-01','--end','2025-09-30','--trade-start','2020-09-30'],check=True)
print('Public inputs verified; prepared files regenerated.')
