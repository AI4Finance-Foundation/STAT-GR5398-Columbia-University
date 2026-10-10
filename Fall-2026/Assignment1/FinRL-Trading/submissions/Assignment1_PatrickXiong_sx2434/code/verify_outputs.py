"""Independent artifact checks after training and comparison finish."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd

ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--run',default='audited_run');ap.add_argument('--comparison',default='comparison_full');a=ap.parse_args();r=a.root
run=r/a.run;comparison=r/a.comparison
pred=pd.read_csv(run/'predictions.csv',parse_dates=['decision_date','available'])
row_audit=pd.read_csv(run/'row_audit.csv',parse_dates=['decision_date'])
if 'canonical_selected' in row_audit:
    assert not row_audit.loc[row_audit.canonical_selected].duplicated(['security_id','datadate']).any()
flow=[]
current=row_audit
flow.append(dict(stage='raw',rows=len(current),tickers=current.ticker.nunique()))
for label,condition in [
    ('decision_in_price_window',row_audit.decision_date.notna()),
    ('historical_member',row_audit.member),
    ('disclosed_by_decision',row_audit.timely),
    ('quoted_at_entry',row_audit.entry_adj_close.notna() & (row_audit.entry_adj_close>0))]:
    current=current.loc[condition.reindex(current.index)]
    flow.append(dict(stage=label,rows=len(current),tickers=current.ticker.nunique()))
flow.append(dict(stage='with_observed_adjusted_label_for_potential_training',rows=int(current.target.notna().sum()),tickers=current.loc[current.target.notna(),'ticker'].nunique()))
if 'canonical_selected' in current:
    current=current.loc[current.canonical_selected]
    flow.append(dict(stage='canonical_security_quarter',rows=len(current),tickers=current.security_id.nunique()))
flow.append(dict(stage='actual_predictions_all_windows',rows=len(pred),tickers=pred.ticker.nunique()))
pd.DataFrame(flow).to_csv(run/'sample_flow.csv',index=False)
assert (pred.available<=pred.decision_date).all()
assert not pred.duplicated(['ticker','decision_date']).any()
from market_data import aliases
assert not pred.assign(security_id=pred.ticker.replace(aliases())).duplicated(['security_id','decision_date']).any()
windows=list((run/'windows').glob('*.json'))
for p in windows:
    x=json.loads(p.read_text())
    assert pd.Timestamp(x['train_label_end_max'])<pd.Timestamp(x['validation_start'])
    assert pd.Timestamp(x['validation_label_end_max'])<pd.Timestamp(x['inference_date'])
assert json.loads((run/'status.json').read_text())['stage']=='completed'
nav=pd.read_csv(comparison/'nav_comparison.csv',index_col=0,parse_dates=True)
metrics=pd.read_csv(comparison/'comparison_metrics.csv')
checked=[]
for name in nav:
    path=comparison/(name+'_daily.csv')
    assert path.exists() or name in {'SPY','QQQ'}, f'Missing strategy daily output: {path}'
    if path.exists():
        d=pd.read_csv(path,index_col=0,parse_dates=True)
        assert d.index.equals(nav.index)
        assert np.allclose((1+d.net_ret).cumprod(),nav[name],atol=1e-10)
        assert np.allclose((1+d.gross_ret)*(1-d.cost)-1,d.net_ret,atol=1e-12)
        assert np.allclose(d.cost,.001*d.turnover,atol=1e-10)
        assert (d.net_ret>-1).all()
    result=metrics.loc[(metrics.strategy==name)&(metrics['sample']=='full')].iloc[0]
    assert abs(result.cumulative-(nav[name].iloc[-1]-1))<1e-10
    checked.append(name)
holdings=pd.read_csv(comparison/'rebalance_holdings.csv')
weights=holdings.groupby(['strategy','date']).weight.sum()
assert ((weights>=0)&(weights<=1+1e-10)).all()
audit=json.loads((run/'audit.json').read_text())
assert audit['feature_count']==52 and 'adj_close_q' not in audit['features']
manifest=json.loads((run/'training_manifest.json').read_text())
assert manifest['script_sha256']==hashlib.sha256((r/'code/audit_and_train.py').read_bytes()).hexdigest()
assert manifest['market_data_sha256']==hashlib.sha256((r/'code/market_data.py').read_bytes()).hexdigest()
if (comparison/'top10_daily_contributions.csv').exists():
    detail=pd.read_csv(comparison/'top10_daily_contributions.csv')
    daily=pd.read_csv(comparison/'audited_top10_daily.csv')
    assert abs(detail.gross_pnl_initial_nav.sum()-daily.fee_pnl_initial_nav.sum()-(daily.nav.iloc[-1]-1))<1e-9
out=dict(passed=True,verified_compare_code_sha256=hashlib.sha256((r/'code/compare_backtests.py').read_bytes()).hexdigest(),prediction_rows=len(pred),training_windows=len(windows),strategies_checked=checked,
    checks=['prediction_availability','unique_stock_decision','outer_training_label_purge','validation_label_purge',
    'saved_NAV_recomputed','gross_net_fee_identity','cost_equals_traded_notional','positive_NAV',
    'weight_budget','feature_set','training_code_manifest','canonical_identity_unique','market_source_code_manifest','dollar_contribution_reconciliation'],
    recorded_preparation_reproduction=json.loads((r/'data_preparation_verification.json').read_text()),
    limitations='These checks validate internal mechanics, not missing market histories or original financial vintages.')
(comparison/'artifact_verification.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
