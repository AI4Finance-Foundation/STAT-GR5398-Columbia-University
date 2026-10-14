"""Prepare a full three-company Option A model follow-up; no call on preparation."""
import json
from pathlib import Path
from audit_io import write_json,sha256,utc_now
ROOT=Path(__file__).resolve().parents[1]
policy=json.loads((ROOT/'config/budget.reasoning_pilot.json').read_text())
policy.update(approved_cap_usd=14.0,authorization='User USD20 TOTAL authorization; this three-company follow-up enforces USD14 ABSOLUTE cumulative ledger cap. Same evidence/costs.jsonl, never an additional allowance.')
write_json(ROOT/'config/budget.reasoning_followup.json',policy)
plan=[]
for ticker in ['NVDA','AMD','INTC']:
    run=f'{ticker}_final_reasoning_v8_1'
    plan.append({'id':run,'completion_manifest':f'runs/{run}/manifest.json',
        'completed_states':['generated','rendered'],'resume_if_incomplete':True,'continue_on_error':False,
        'argv':['scripts/experiment.py','--upstream','{upstream}','verify','--run',f'runs/{ticker}_baseline_1',
            '--final-run',f'runs/{run}','--budget','config/budget.reasoning_followup.json','--ledger','evidence/costs.jsonl',
            '--max-tokens','20000','--reasoning-effort','low']})
write_json(ROOT/'config/verifier_reasoning_v8_1_plan.json',plan)
write_json(ROOT/'evidence/verifier_versions/reasoning_v8_1_design.json',{'prepared_utc':utc_now(),
    'model':policy['model'],'reasoning_effort':'low','max_completion_tokens':20000,'temperature':'omitted',
    'source_sha256':sha256((ROOT/'evidence/verifier_versions/verifier_agent_v8_1.py').read_bytes()),
    'cohort':['NVDA','AMD','INTC'],'scope':'All eight sections, same original baseline writer evidence and text.',
    'independent_new_verifier_draws':True,'pilot_reuse':False,
    'hypothesis':'Period-aware semantic reviews plus exact-wrapper transport may reduce residual narrative errors. A known false contradiction on a true CAGR inequality prevents any pilot-only efficacy claim.',
    'evaluation':'Complete source-unit coverage and atomic judgments under the prior independent protocol. Exact unchanged units can carry independently reviewed labels only with matching evidence. No verifier classifications are truth labels.',
    'development_bias':'Adaptive model/configuration choice on the same evaluated cohort; no held-out or causal attribution across multiple changed components.',
    'publication':'Deferred by user. Current V6/V7.1 releases remain unchanged pending complete evaluation.'})
print('Prepared three-company reasoning follow-up under cumulative USD14 guard.')
