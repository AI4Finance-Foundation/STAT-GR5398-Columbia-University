"""Check planned text artifacts without ever printing a matched credential."""
import argparse,re,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--task-root',default='.');p.add_argument('--package');a=p.parse_args();r=Path(a.task_root).resolve();o=r/'outputs/retraining'
patterns={
 'openai_key':re.compile(r'(?<![\w-])sk-(?:proj-)?[A-Za-z0-9_-]{20,}'),
 'aws_access_key':re.compile(r'AKIA[0-9A-Z]{16}'),
 'google_api_key':re.compile(r'AIza[0-9A-Za-z_-]{30,}'),
 'github_token':re.compile(r'gh[pousr]_[A-Za-z0-9]{30,}'),
 'private_key':re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
 'quoted_credential':re.compile(r'''(?i)(?:api[_-]?key|access[_-]?token|secret[_-]?key)\s*[:=]\s*["']([A-Za-z0-9_/+-]{20,})["']''')}
roots=[Path(a.package).resolve()] if a.package else [r/'outputs/assignment2/submissions/Assignment2_XinruXiao_xx2524',r/'work/retraining/baseline',r/'work/retraining/improved',r/'work/retraining/evaluation',o,*list((r/'work').glob('evaluation_*.log'))]
seen=set();hits=[];scanned=0;size=0
for root in roots:
 for f in ([root] if root.is_file() else root.rglob('*')):
  if not f.is_file() or f.suffix.lower() not in {'.py','.json','.jsonl','.csv','.md','.txt','.patch','.log','.toml','.yaml','.yml'}:continue
  rel=f.relative_to(root)
  if any(x in {'data','.git','__pycache__','submissions'} for x in rel.parts):continue
  if root.name in {'baseline','improved'} and 'output' in rel.parts:continue
  resolved=f.resolve()
  if resolved in seen:continue
  seen.add(resolved);raw=f.read_bytes();s=raw.decode('utf-8',errors='replace');scanned+=1;size+=len(raw)
  for kind,pattern in patterns.items():
   for match in pattern.finditer(s):
    if kind=='quoted_credential' and match.group(1).upper().startswith(('YOUR_','EXAMPLE_','REDACTED','PLACEHOLDER')):continue
    hits.append({'file':str(f.relative_to(r)) if f.is_relative_to(r) else str(rel),'pattern':kind})
result={'status':'PASS' if not hits else 'FAIL','text_files_scanned':scanned,'bytes_scanned':size,'matches_without_values':hits,'scope':'planned submission text files; never reads environment variables or private credential files'}
(o/'credential_scan.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if hits:raise SystemExit('Credential pattern detected. Values withheld; do not assemble the package.')
