"""Finite-sample witnesses for dated-universe retention and causal information clocks.

These checks prove properties of the frozen pipeline. They do not certify every
vendor release vintage or the representativeness of unknown supervised outcomes.
"""
from pathlib import Path
import os,sys,json,hashlib
import numpy as np,pandas as pd,joblib
ROOT=Path(__file__).resolve().parents[1]
INP=Path(os.environ.get('FINRL_INPUT_DIR',str(ROOT/'inputs')))
DATA=Path(os.environ.get('FINRL_DATA_DIR',str(ROOT/'data-source')))
RUN=Path(os.environ.get('FINRL_RUN_DIR',str(ROOT/'run')))
OUT=RUN/'bias_proof';OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(Path(__file__).resolve().parent/'support'))
from build_features import FEATURES
from causal_policy import causal_partition,decision_eligibility,verify_known_horizon_outcomes
p=pd.read_csv(RUN/'model_panel.csv.gz',parse_dates=['decision_date','label_end','available'],float_precision='round_trip')
pred=pd.read_csv(RUN/'predictions.csv',parse_dates=['decision_date','available'],float_precision='round_trip')
m=pd.read_csv(INP/'crsp_sp500_members.csv.gz',usecols=['PERMNO','MbrStartDt','MbrEndDt']);m.columns=m.columns.str.lower()
for c in ['mbrstartdt','mbrenddt']:m[c]=pd.to_datetime(m[c])
spells=m.drop_duplicates();last_date=pd.Timestamp('2025-09-30')
last=set(spells[spells.mbrstartdt.le(last_date)&(spells.mbrenddt.ge(last_date)|spells.mbrenddt.isna())].permno)
e=pd.read_csv(DATA/'event_recognition.csv',parse_dates=['crsp_return_date','recognition_date','execution_date'])
finished=set(e[e.crsp_return_date.le(pd.Timestamp('2025-09-30'))].permno)
repairs=pd.read_csv(DATA/'missing_quote_resume_audit.csv',parse_dates=['gap_day','resume_day'])
# Derived observation availability clocks, used only to verify recorded fit rows.
p['label_source_available']=p.label_end
for row in e.itertuples(index=False):
 hit=p.permno.eq(row.permno)&p.decision_date.lt(row.crsp_return_date)&p.label_end.ge(row.crsp_return_date)
 p.loc[hit,'label_source_available']=p.loc[hit,'label_source_available'].map(lambda t:max(t,row.recognition_date))
for row in repairs.itertuples(index=False):
 hit=p.permno.eq(row.permno)&p.decision_date.lt(row.gap_day)&p.label_end.ge(row.gap_day)
 p.loc[hit,'label_source_available']=p.loc[hit,'label_source_available'].map(lambda t:max(t,row.resume_day))
p['label_available_date']=p.label_source_available
p['supervised_decision_eligible']=verify_known_horizon_outcomes(p,pd.Timestamp('2025-09-30'))
complete=p.label_end.le(pd.Timestamp('2025-09-30'))
assert p.loc[complete,'supervised_decision_eligible'].equals(p.loc[complete,'target'].notna())
unknown=p[p.label_status.eq('unknown_complete_delisting_proceeds')]
assert (~unknown.supervised_decision_eligible).all()
# Causal guard must stop rather than discard an eligible failed bank with a missing label.
missing_failure=p.copy();bad=(missing_failure.permno.eq(11786)&missing_failure.decision_date.eq(pd.Timestamp('2023-03-01')))
assert bad.sum()==1 and missing_failure.loc[bad,'supervised_decision_eligible'].all()
missing_failure.loc[bad,'target']=np.nan
try:verify_known_horizon_outcomes(missing_failure,pd.Timestamp('2025-09-30'))
except ValueError:fail_closed_missing_failure=True
else:raise AssertionError('Eligible failed bank silently discarded')
observed=p.target.notna()
late=p[observed & p.label_source_available.gt(p.label_end)]
assert late.empty,late[['ticker','decision_date','label_end','label_source_available']]
records=[];future_membership_changes=0
for day,g in p.groupby('decision_date'):
 official=spells[spells.mbrstartdt.le(day)&(spells.mbrenddt.ge(day)|spells.mbrenddt.isna())]
 expected=set(official.permno)
 assert set(g.permno)==expected
 # Erase every future start and future removal date BEFORE universe construction.
 prefix=spells[spells.mbrstartdt.le(day)].copy()
 prefix.loc[prefix.mbrenddt.ge(day),'mbrenddt']=pd.NaT
 from_prefix=set(prefix[prefix.mbrenddt.isna()|prefix.mbrenddt.ge(day)].permno)
 future_membership_changes+=len(expected.symmetric_difference(from_prefix))
 if day>=pd.Timestamp('2020-06-01'):
  assert set(pred[pred.decision_date.eq(day)].security_id)==set(g.security_id)
 records.append(dict(decision_date=day,official_members=len(expected),retained_panel_rows=len(g),
  members_later_absent_from_final_universe=len(expected-last),later_terminal_securities=len(expected&finished),
  universe_difference_after_erasing_future_membership_records=len(expected.symmetric_difference(from_prefix)),
  missing_expected_members=0,unexpected_members=0))
assert future_membership_changes==0
pd.DataFrame(records).to_csv(OUT/'historical_universe_witness.csv',index=False)
logs=[json.loads(line) for line in (RUN/'model_fit_execution.jsonl').read_text().splitlines() if line.strip()]
panel_digest=hashlib.sha256((RUN/'model_panel.csv.gz').read_bytes()).hexdigest()
value_bound_records=[r for r in logs if 'panel_sha256' in r.get('context',{})]
assert all(r['context']['panel_sha256']==panel_digest for r in value_bound_records)
seen=set();boundary=[];negative_control=[];prediction_deltas=[];feature_names=[]
all_rows=list(p.itertuples(index=False));role_cache={}
# Reconstruct expected role keys independently of the production split-mask code.
by_key={(row.security_id,str(row.decision_date.date())):row for row in p.itertuples(index=False)}
for entry in logs:
 role=entry['role'];key=(entry['window'],role);assert key not in seen;seen.add(key)
 day=pd.Timestamp(entry['window'].split('/')[0]);cutoff=pd.Timestamp(entry['cutoff'])
 lower=pd.Timestamp(entry['lower']) if entry['lower'] else None
 cache_key=(day,role)
 if cache_key not in role_cache:
  rows=[]
  for row in all_rows:
   if role=='outer_train':valid=row.decision_date<cutoff and row.supervised_decision_eligible and row.label_source_available<cutoff
   elif role=='outer_validation':valid=row.decision_date>=lower and row.decision_date<day and row.supervised_decision_eligible and row.label_source_available<day
   else:valid=row.decision_date==day
   if valid:rows.append(row)
  keys=sorted(f'{r.security_id}|{r.decision_date.date()}' for r in rows)
  digest=hashlib.sha256('\n'.join(keys).encode()).hexdigest()
  role_cache[cache_key]=(rows,digest)
 rows,digest=role_cache[cache_key]
 assert digest==entry['key_sha256'] and len(rows)==entry['rows'],key
 assert all(r.available<r.decision_date for r in rows),key
 if role!='inference':assert all(r.label_source_available<cutoff for r in rows),key
 if role=='outer_train':
  known,valid,forecast=causal_partition(p,cutoff,day)
  legacy=p[p.decision_date.lt(cutoff)&p.label_end.lt(cutoff)&p.target.notna()]
  assert known.index.equals(legacy.index)
  assert np.array_equal(known[FEATURES+['target']].to_numpy(),legacy[FEATURES+['target']].to_numpy(),equal_nan=True)
  altered=p.copy()
  future=altered.decision_date.ge(cutoff)|altered.label_source_available.ge(cutoff)
  # Poison future covariates and every outcome unavailable at this fit boundary.
  altered.loc[future,FEATURES]=1e12;altered.loc[future,'target']=-1e12
  causal,_,_=causal_partition(altered,cutoff,day)
  assert np.array_equal(known[FEATURES+['target']].to_numpy(),causal[FEATURES+['target']].to_numpy(),equal_nan=True)
  artifact=joblib.load(RUN/'models'/(entry['window'].replace('/','_')+'.joblib'))
  im=artifact['imputer'];estimator=artifact['model']
  cols=[c for f in FEATURES for c in [f,f+'_missing']]
  assert list(estimator.feature_names_in_)==cols
  assert all(im.medians_[f]==known[f].median() for f in FEATURES)
  inf=p[p.decision_date.eq(day)].copy()
  normal_mat=im.transform(inf[FEATURES]);normal=estimator.predict(normal_mat)
  inf['target']=-1e30;inf['stock_total_return_target']=1e30;inf['qqq_target']=1e30
  inf['identity_tier']='adversarial_future_status';inf['ticker']='future_failure';inf['permno']=-99
  changed_mat=im.transform(inf[FEATURES]);assert np.array_equal(normal_mat.to_numpy(),changed_mat.to_numpy())
  changed=estimator.predict(changed_mat)
  delta=float(np.max(np.abs(normal-changed)));assert delta<1e-12
  prediction_deltas.append(delta);feature_names.append(cols)
  negative_control.append(bool((altered[FEATURES].median()-known[FEATURES].median()).abs().max()>1e-9))
 if entry['window'].endswith('/rf_seed_42'):
  boundary.append(dict(decision_date=str(day.date()),role=role,cutoff=str(cutoff.date()),rows=len(rows),
    executed_key_hash_matches=True,max_feature_observation=max((r.available for r in rows),default=pd.NaT),
    max_used_label_available=max((r.label_source_available for r in rows),default=pd.NaT) if role!='inference' else pd.NaT,
    temporal_breaches=0))
assert len(logs)==858 and len(prediction_deltas)==286 and all(negative_control)
pd.DataFrame(boundary).to_csv(OUT/'executed_information_clock_witness.csv',index=False)
examples=[]
for perm,ticker in [(11786,'SIVB'),(90090,'SBNY'),(12448,'FRC'),(75107,'ABMD'),(19502,'WBA'),(14704,'INFO')]:
 g=p[p.permno.eq(perm)]
 failures=g[g.decision_date.eq(pd.Timestamp('2023-03-01'))] if ticker in ['SIVB','SBNY','FRC'] else g[g.target.isna()]
 examples.append(dict(ticker=ticker,PERMNO=perm,historical_member_cases=len(g),forecast_cases=int(pred.security_id.eq('CRSP_'+str(perm)).sum()),
  observed_training_label_cases=int(g.target.notna().sum()),unknown_label_cases=int(g.label_status.eq('unknown_complete_delisting_proceeds').sum()),
  finite_failure_quarter_label_cases=int(failures.target.notna().sum()) if ticker in ['SIVB','SBNY','FRC'] else None))
pd.DataFrame(examples).to_csv(OUT/'named_failure_retention_witness.csv',index=False)
summary=dict(scope='Finite proof of frozen-source historical-universe retention and decision/training causality; conditional on CRSP historical record semantics.',
 official_member_quarter_cases=len(p),unique_historical_stock_PERMNOs=p.permno.nunique(),final_universe_PERMNOs=len(last),final_universe_date=str(last_date.date()),
 former_members_absent_from_final_universe=len(set(p.permno)-last),
 retained_cases_that_today_only_filter_would_delete=int((~p.permno.isin(last)).sum()),
 later_terminal_historical_securities=len(set(p.permno)&finished),later_terminal_historical_cases=int(p.permno.isin(finished).sum()),
 missing_official_members=0,forecast_cases=len(pred),missing_forecast_cases=0,future_membership_erasure_differences=future_membership_changes,
 executed_fit_score_predict_records_verified=len(logs),
 fit_records_binding_full_archived_panel_hash=len(value_bound_records),archived_model_panel_sha256=panel_digest,estimator_artifacts_verified=len(prediction_deltas),
 finite_labels_using_a_source_observation_after_label_end=len(late),source_label_clock_breaches=0,
 future_covariate_and_unavailable_label_poisoning_training_changes=0,
 poisoned_future_metadata_and_inference_targets_maximum_prediction_difference=max(prediction_deltas),
 deliberate_global_median_leak_detected_in_all_286_cases=bool(all(negative_control)),
 unknown_complete_proceeds_cases=len(unknown),
 unknown_cases_already_ineligible_under_public_decision_rule=int((~unknown.supervised_decision_eligible).sum()),
 eligible_failed_bank_missing_label_causes_error=fail_closed_missing_failure,
 causal_training_policy_matches_all_recorded_training_values=True,
 supervised_eligible_complete_horizon_cases=int((complete & p.supervised_decision_eligible).sum()),
 silent_missing_eligible_outcome_exclusions=0,
 right_censored_cases=int(p.label_status.eq('right_censored').sum()),
 residual_claim_limits=['The dated public-rights eligibility rule excludes two unknown consideration cases before the forecast horizon. This is a developed, nonexhaustive liquidity rule; representativeness for an unrestricted all-stock target is not claimed.',
 'Historical original-vintage CRSP revisions and membership-announcement timestamps are not reconstructed.',
 'The QQQ80/top5 enhancement and developed rights list were researched on this history; no untouched holdout or absence of research-selection bias is proved.'],
 claim='Classical current-universe survivorship filtering and future-observation/label leakage are avoided in the audited pipeline. A blanket claim of all forms of bias eliminated is not supported.')
(OUT/'bias_proof_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
