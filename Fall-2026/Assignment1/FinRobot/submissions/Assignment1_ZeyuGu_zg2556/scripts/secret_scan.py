"""Fail closed if actual local credential bytes occur anywhere in this submission.

Only filenames and counts are reported, never matched values. Config is read
outside the submission solely to obtain the actual secret values to scan for.
"""
import argparse
from pathlib import Path
from urllib.parse import quote
from audit_io import credentials, utc_now, write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    args=p.parse_args()
    root=args.root.resolve()
    cfg=credentials(args.config)
    values=[cfg[k] for k in ('openai_api_key','fmp_api_key') if cfg.get(k)]
    needles={x.encode() for value in values for x in (value,quote(value,safe=''))}
    hits=[];forbidden=[];scanned=0
    # Extended paths avoid silently skipping archived responses past MAX_PATH.
    import os
    scan_root=Path('\\\\?\\'+str(root)) if os.name=='nt' else root
    for file in scan_root.rglob('*'):
        if not file.is_file(): continue
        if any(part in {'__pycache__','.pytest_cache'} for part in file.parts): continue
        rel=file.relative_to(scan_root).as_posix()
        if file.name in {'config.ini','.env','pgpass.conf'} or '.git' in file.relative_to(scan_root).parts:
            forbidden.append(rel)
        blob=file.read_bytes();scanned+=1
        if any(needle in blob for needle in needles): hits.append(rel)
    result={'checked_at_utc':utc_now(),'scanned_files':scanned,'known_secret_count':len(values),
        'credential_match_files':hits,'forbidden_path_files':forbidden,
        'passed':bool(values) and not hits and not forbidden,
        'scope':'Exact actual FMP/OpenAI secrets plus URL encodings in all non-cache submission files. WRDS login stored outside submission; no claim that arbitrary unknown secrets can be detected.'}
    write_json(root/'evidence/secret_scan.json',result)
    print('Secret scan:', 'PASS' if result['passed'] else 'FAIL', 'files=',scanned,'matched_files=',len(hits))
    if not result['passed']: raise SystemExit(1)


if __name__=='__main__': main()
