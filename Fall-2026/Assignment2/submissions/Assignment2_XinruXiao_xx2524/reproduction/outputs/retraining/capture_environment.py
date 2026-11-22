"""Record reproducibility inputs without credentials or raw course data."""
from pathlib import Path
import hashlib,json,platform,sys,importlib.metadata as m
r=Path(__file__).resolve().parents[2];o=r/'outputs/retraining';data=r/'work/course/Fall-2026/Assignment2/source_code/data'
record={'python':sys.version.split()[0],'os':platform.system(),'architecture':platform.machine(),'dependencies':{},'course_data_hashes':{}}
for name in ['numpy','pandas','pyarrow','scikit-learn','scipy','pydantic','xgboost','lightgbm','threadpoolctl','joblib']:
 try:record['dependencies'][name]=m.version(name)
 except m.PackageNotFoundError:record['dependencies'][name]='unavailable'
for p in data.rglob('*'):
 if p.is_file() and p.suffix in ['.db','.parquet','.json']:record['course_data_hashes'][str(p.relative_to(data))]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
base=r/'work/retraining/baseline/selection/selector.py';official=r/'work/course/Fall-2026/Assignment2/source_code/selection/selector.py'
record['baseline_selector_byte_identical_to_course']=base.read_bytes()==official.read_bytes();assert record['baseline_selector_byte_identical_to_course']
(o/'environment_and_data_provenance.json').write_text(json.dumps(record,indent=2));print('Recorded dependencies and',len(record['course_data_hashes']),'course data hashes; original selector matches.')
