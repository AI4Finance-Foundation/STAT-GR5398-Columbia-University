"""Record hashes only; never parse or expose licensed financial observations."""
from pathlib import Path
from audit_io import sha256,utc_now,write_json
ROOT=Path(__file__).resolve().parents[1]
PRIVATE=Path.home()/'.cache/finrobot-zg2556/private_wrds/20261011'
records=[]
for name,query,rows,scope in [('zbbwzhvadz3iudpv',11774671,31,'Compustat North America annual; 2020-01-01 to 2026-10-31; NVDA AMD INTC AVGO QCOM; USD STD INDL C; active/inactive'),('p2vb5gk8vrf9g82e',11774673,8510,'Compustat North America security daily; 2020-01-01 to 2026-10-09; NVDA AMD INTC AVGO QCOM')]:
    records.append({'query_id':query,'row_count':rows,'scope':scope,'downloaded_date_local':'2026-10-11','private_files':[{'filename':name+ext,'sha256':sha256((PRIVATE/(name+ext)).read_bytes()),'bytes':(PRIVATE/(name+ext)).stat().st_size} for ext in ('.csv','.sql')],'values_read_into_llm':False,'used_in_pipeline':False,'distributed_with_submission':False})
write_json(ROOT/'evidence/wrds/web_exports_manifest.json',{'recorded_at_utc':utc_now(),'queries':records,'decision':'User chose public materials; WRDS exports retained privately, not used in LLM experiments','policy_record':'docs/wrds_and_public_source_decision.md'})
print('Recorded two export manifests without exposing data values.')
