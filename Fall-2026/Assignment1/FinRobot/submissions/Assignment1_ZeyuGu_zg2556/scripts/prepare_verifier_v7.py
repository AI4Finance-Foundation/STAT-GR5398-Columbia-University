"""Archive the developed verifier and plan real calls without changing frozen writers."""
import json
import shutil
from pathlib import Path
from audit_io import sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
UP = Path.home()/'.cache/finrobot-zg2556/upstream'
source = UP/'finrobot_equity/core/src/modules/equity_agents/verifier_agent.py'
archive = ROOT/'evidence/verifier_versions/verifier_agent_v7.py'
if archive.exists() and archive.read_bytes() != source.read_bytes():
    raise ValueError('V7 archive is immutable')
if not archive.exists():
    shutil.copy2(source, archive)
plan=[]
for ticker in ['NVDA','AMD','INTC','AVGO','QCOM','AAPL','GOOGL']:
    run=f'{ticker}_final_v7'
    plan.append({'id':run, 'completion_manifest':f'runs/{run}/manifest.json',
        'completed_states':['generated','rendered'],'resume_if_incomplete':True,
        'continue_on_error':True,
        'argv':['scripts/experiment.py','--upstream','{upstream}','verify',
            '--run',f'runs/{ticker}_baseline_1','--final-run',f'runs/{run}',
            '--budget','config/budget.evaluation.json','--ledger','evidence/costs.jsonl',
            '--max-tokens','26000']})
write_json(ROOT/'config/verifier_v7_plan.json',plan)
write_json(ROOT/'evidence/verifier_versions/v7_development.json',{
    'version':'7-semantic-period-peer-horizon-review','source_sha256':sha256(source.read_bytes()),
    'development_cohort':['NVDA','AMD','INTC'],
    'uses_audit_feedback_for_prompt_design':True,
    'held_out_efficacy_claim':False,
    'design':'Generic period, entity, metric, trend, CAGR horizon and flattened-news column instructions; deterministic exact release-string deduplication. Original writer evidence is unchanged.',
    'input_to_verifier':'Original writer prompt and original draft only. No independent audit labels or later sources.',
    'comparison_warning':'Adaptive development on the evaluated cohort. V6 results and failures remain immutable; V7 is not a held-out confirmation.',
    'cost_policy':'New independent verifier draws, same cumulative USD 20 ledger; no cached response reuse.'})
print('Prepared immutable V7 source and seven serial real verifier runs.')
