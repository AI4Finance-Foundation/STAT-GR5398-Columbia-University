"""Verify the financial sources against their actual frozen Git objects."""
from pathlib import Path
import json,hashlib,subprocess
r=Path(__file__).resolve().parents[2];o=r/'outputs/retraining';repo=r/'work/retraining';manifest=json.loads((o/'expanded_study_manifest.json').read_text());commit=manifest['frozen_code_commit']
assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==commit
assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip()
paths=['baseline/selection/selector.py','baseline/selection/configs/growth.json','baseline/valuation/student_valuation.py','improved/selection/selector.py','improved/selection/configs/growth.json','improved/selection/configs/improved.json','improved/valuation/student_valuation.py'];checks={}
for name in paths:
 actual=(repo/name).read_bytes();frozen=subprocess.check_output(['git','-C',str(repo),'show',commit+':'+name]);assert actual==frozen,name
 checks[name]=hashlib.sha256(actual).hexdigest()
original=r/'work/course/Fall-2026/Assignment2/source_code';unchanged=[]
for f in (original/'valuation/finrobot').rglob('*.py'):
 relative=f.relative_to(original)
 for v in ['baseline','improved']:assert f.read_bytes()==(repo/v/relative).read_bytes(),(v,str(relative))
 unchanged.append(str(relative))
for name in ['portfolio/backtester.py','selection/allocation.py','selection/strategy.py','shared/data_view.py','shared/metrics.py']:
 for v in ['baseline','improved']:assert (original/name).read_bytes()==(repo/v/name).read_bytes(),(v,name)
 unchanged.append(name)
result={'status':'PASS','frozen_commit':commit,'financial_sources_match_git_objects':checks,'unchanged_course_components':unchanged,'note':'Wrapper/runtime and report-source hashes are recorded separately; no financial source changed after freeze.'}
(o/'frozen_source_verification.json').write_text(json.dumps(result,indent=2));print('PASS: frozen selectors/configs/valuation match Git; course engine, allocator, backtester and DataView unchanged.')
