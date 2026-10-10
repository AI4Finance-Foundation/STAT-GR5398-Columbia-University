"""Audit the supplied data and fit a conservative, reproducible sensitivity run.

This cannot recover missing delisted prices or historical financial vintages.
See README for the remaining limitations; outputs are research diagnostics.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import argparse
import hashlib
import json
import sqlite3
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from market_data import prices as load_prices, aliases
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
warnings.filterwarnings('ignore',message='X does not have valid feature names')


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False))


def status(out, stage, **kw):
    value = dict(stage=stage, time=pd.Timestamp.now(tz='America/New_York').isoformat(), **kw)
    save_json(out/'status.json', value)
    print(json.dumps(value, default=str), flush=True)


def membership(cache):
    m = pd.read_csv(cache/'sp500_historical_constituents.csv', parse_dates=['date']).sort_values('date')
    dates = m.date.to_numpy()
    mapping=aliases()
    sets = [{mapping.get(t,t) for t in s.split(',')} for s in m.tickers]
    def at(date):
        pos = dates.searchsorted(np.datetime64(date), side='right')-1
        return sets[pos] if pos >= 0 else set()
    return at


def prepare(cache, prior, out):
    f = pd.read_csv(cache/'fundamental_data_full.csv')
    for c in ['datadate', 'filing_date', 'accepted_date', 'actual_tradedate']:
        f[c] = pd.to_datetime(f[c], errors='coerce')
    f = f.sort_values(['ticker', 'datadate']).reset_index(drop=True)
    assert not f.duplicated(['ticker','datadate']).any()
    p = load_prices(cache,canonical=True)
    assert not p.duplicated(['tic','datadate']).any()
    p[['tic','source_ticker','price_source']].drop_duplicates().to_csv(out/'price_source_choices.csv',index=False)
    px = p.pivot(index='datadate', columns='tic', values='adj_close').sort_index()
    at = membership(cache)
    nominal = (f.datadate.dt.to_period('Q')+1).dt.start_time + pd.DateOffset(months=2)
    next_nominal = nominal + pd.DateOffset(months=3)
    def trade_date(d):
        i = px.index.searchsorted(d)
        return px.index[i] if i < len(px) else pd.NaT
    f['decision_date'] = nominal.map(trade_date)
    f['label_end'] = next_nominal.map(trade_date)
    f['available'] = f[['filing_date','accepted_date']].max(axis=1) + pd.offsets.BDay(1)
    # Read all payload dates; match by the actual filing/fiscal period below.
    with sqlite3.connect(cache/'finrl_trading.db') as db:
        payloads=pd.read_sql_query("SELECT ticker,date,payload,row_json FROM raw_payloads WHERE payload != 'profile'",db)
    raw_rows=[]
    for r in payloads.itertuples():
        x=json.loads(r.row_json);period=str(x.get('period','')).upper();year=x.get('calendarYear')
        if period=='FY': period='Q4'
        try:
            bucket=pd.Period(str(year)+period).end_time.normalize() if year and period in ['Q1','Q2','Q3','Q4'] else pd.to_datetime(x.get('date')).to_period('Q').end_time.normalize()
        except (ValueError,TypeError,AttributeError): continue
        filed=pd.to_datetime(x.get('fillingDate') or x.get('filingDate'),errors='coerce')
        accepted=pd.to_datetime(x.get('acceptedDate'),errors='coerce')
        fiscal_end=pd.to_datetime(x.get('date'),errors='coerce')
        latest=pd.Series([filed,accepted,fiscal_end]).max()
        raw_rows.append(dict(ticker=r.ticker,datadate=bucket,payload=r.payload,latest=latest,
                             fiscal_end=fiscal_end,filing_date=filed))
    raw=pd.DataFrame(raw_rows)
    # Match aggregate inputs to the actual income filing, then the same fiscal
    # period across payloads. Fiscal Q1 is not necessarily calendar Q1.
    income=raw.loc[raw.payload=='income'].copy()
    income['filing_key']=income.filing_date.dt.normalize()
    anchors=income.groupby(['ticker','filing_key']).fiscal_end.max().rename('anchor_fiscal_end').reset_index()
    f['filing_key']=f.filing_date.dt.normalize()
    f=f.merge(anchors,on=['ticker','filing_key'],how='left',validate='many_to_one')
    grouped=raw.groupby(['ticker','fiscal_end']).agg(raw_latest=('latest','max'),payload_count=('payload','nunique'),fiscal_min=('fiscal_end','min'),fiscal_max=('fiscal_end','max')).reset_index().rename(columns={'fiscal_end':'anchor_fiscal_end'})
    f=f.merge(grouped,on=['ticker','anchor_fiscal_end'],how='left',validate='many_to_one')
    f['aggregate_available']=f.available
    raw_available=f.raw_latest.dt.normalize()+pd.offsets.BDay(1)
    f['available']=pd.concat([f.available,raw_available],axis=1).max(axis=1)
    f[['ticker','datadate','aggregate_available','available','raw_latest','payload_count','fiscal_min','fiscal_max','decision_date']].to_csv(out/'raw_payload_timing_audit.csv',index=False)
    # Strict recorded-timestamp rule. No imputed disclosure dates in this run.
    f['timestamps_present'] = f.filing_date.notna() & f.accepted_date.notna()
    f['timely'] = f.timestamps_present & (f.available <= f.decision_date)
    f['security_id']=f.ticker.replace(aliases())
    f['member'] = [t in at(d) if pd.notna(d) else False for t,d in zip(f.security_id,f.decision_date)]
    lookup = p.set_index(['tic','datadate']).adj_close
    f['entry_adj_close'] = lookup.reindex(pd.MultiIndex.from_arrays([f.security_id,f.decision_date])).to_numpy()
    f['exit_adj_close'] = lookup.reindex(pd.MultiIndex.from_arrays([f.security_id,f.label_end])).to_numpy()
    f['target'] = np.log(f.exit_adj_close/f.entry_adj_close)
    f['label_error'] = f.y_return-f.target
    # Do not train on target rows with an unknown endpoint or zero/nonpositive price.
    f.loc[(f.entry_adj_close<=0)|(f.exit_adj_close<=0), 'target'] = np.nan
    features = [c for c in f.select_dtypes('number') if c not in {
        'id','y_return','trade_price','adj_close_q','entry_adj_close','exit_adj_close','target','label_error','payload_count'}]
    assert len(features)==52, features
    f[features] = f[features].replace([np.inf,-np.inf],np.nan)
    # Resolve repeated names of one security BEFORE all training/validation.
    # Prefer a timely row, then its canonical ticker, then the latest known
    # information; no outcome or future exit-price criterion enters this choice.
    order=f.assign(canonical_name=(f.ticker==f.security_id)).sort_values(
        ['security_id','datadate','timely','canonical_name','available','ticker'],
        ascending=[True,True,False,False,False,True],na_position='last',kind='stable')
    selected=order.drop_duplicates(['security_id','datadate']).index
    f['canonical_selected']=f.index.isin(selected)
    duplicates=f.duplicated(['security_id','datadate'],keep=False)
    f.loc[duplicates,['security_id','ticker','datadate','decision_date','available','timely','canonical_selected']+features].to_csv(out/'canonical_row_decisions.csv',index=False)
    f[['ticker','security_id','canonical_selected','datadate','filing_date','accepted_date','actual_tradedate','decision_date',
       'label_end','available','timely','member','anchor_fiscal_end','payload_count','trade_price','entry_adj_close','exit_adj_close',
       'y_return','target','label_error']].to_csv(out/'row_audit.csv',index=False)
    valid = f.label_error.dropna()
    mismatch = f.loc[f.actual_tradedate.notna() & f.decision_date.notna() & (f.actual_tradedate != f.decision_date),
                     ['ticker','datadate','actual_tradedate','decision_date']]
    mismatch.to_csv(out/'trade_date_mismatches.csv',index=False)
    a = f.merge(p,left_on=['security_id','actual_tradedate'],right_on=['tic','datadate'],how='left',suffixes=('','_price'))
    a['relative_price_difference'] = a.trade_price/a.prccd-1
    a.loc[a.relative_price_difference.abs()>.01, ['ticker','datadate','actual_tradedate','trade_price','prccd','relative_price_difference']].to_csv(out/'trade_price_mismatches.csv',index=False)

    # Audit the exact exclusions from the first overnight experiment.
    e = pd.read_csv(prior/'entry_exclusions.csv').drop_duplicates(['date','ticker'])
    e['date'] = pd.to_datetime(e.date)
    e['absent_from_daily_table'] = ~e.ticker.replace(aliases()).isin(px.columns)
    e['member_on_date'] = [aliases().get(t,t) in at(d) for t,d in zip(e.ticker,e.date)]
    e.to_csv(out/'prior_entry_exclusions_audited.csv',index=False)
    cov=[]
    for d in sorted(f.decision_date.dropna().unique()):
        d=pd.Timestamp(d)
        if d < pd.Timestamp('2020-06-01'): continue
        members=at(d)
        quoted={aliases().get(t,t) for t in px if pd.notna(px.at[d,t])}
        available=members & quoted
        cov.append(dict(date=d,members=len(members),quoted_members=len(available),missing_members=len(members-available),missing_tickers=','.join(sorted(members-available))))
    pd.DataFrame(cov).to_csv(out/'universe_coverage.csv',index=False)

    # Reconcile early filing dates to the ORIGINAL fiscal period in raw FMP records.
    payloads=payloads.loc[payloads.payload=='income']
    fiscal=[]
    for r in payloads.itertuples():
        x=json.loads(r.row_json)
        filing=pd.to_datetime(x.get('fillingDate') or x.get('filingDate'),errors='coerce')
        fiscal.append(dict(ticker=r.ticker,filing_date=filing,fiscal_period_end=pd.to_datetime(x.get('date'),errors='coerce'),calendarYear=x.get('calendarYear'),fiscal_period=x.get('period')))
    fiscal=pd.DataFrame(fiscal).drop_duplicates(['ticker','filing_date','fiscal_period_end'])
    early=f.loc[f.filing_date < f.datadate,['ticker','datadate','filing_date']]
    joined=early.merge(fiscal,on=['ticker','filing_date'],how='left')
    joined['fiscal_before_filing']=joined.fiscal_period_end <= joined.filing_date
    joined.to_csv(out/'fiscal_date_reconciliation.csv',index=False)
    reconciled=joined.loc[joined.fiscal_before_filing,['ticker','datadate']].drop_duplicates()
    recovery=Path(__file__).resolve().parents[1]/'evidence/price_recovery'
    recovery_manifest=json.loads((recovery/'recovery_manifest.json').read_text())
    audit=dict(raw_rows=len(f),canonical_model_rows=int(f.canonical_selected.sum()),
        canonical_duplicate_groups=int(f.loc[duplicates,['security_id','datadate']].drop_duplicates().shape[0]),
        canonical_rows_removed=int((~f.canonical_selected).sum()),
        canonical_selection_rule='timely first, canonical ticker, latest available, lexical source ticker; no outcome filtering',
        canonical_price_rule='one whole accepted recovery series per issuer when available; otherwise one pinned series; identical training and execution source',
        security_aliases=aliases(),
        recovery_price_sha256={t:hashlib.sha256((recovery/'accepted'/f'{t}.csv').read_bytes()).hexdigest() for t in recovery_manifest['accepted']},
        feature_count=len(features),features=features,
        raw_payload_date_matched_rows=int(f.raw_latest.notna().sum()),
        raw_timestamp_delays_rows=int((f.available>f.aggregate_available).sum()),
        newly_late_rows=int(((f.aggregate_available<=f.decision_date)&(f.available>f.decision_date)).sum()),
        early_filing_rows=len(early),early_rows_with_prior_fiscal_period_match=len(reconciled),
        missing_either_timestamp=int((~f.timestamps_present).sum()),
        raw_trade_date_mismatches=len(mismatch),raw_price_over_1pct_mismatch_rows=int((a.relative_price_difference.abs()>.01).sum()),
        raw_price_over_1pct_mismatch_tickers=sorted(a.loc[a.relative_price_difference.abs()>.01,'ticker'].unique().tolist()),
        label_comparable_rows=len(valid),label_log_difference_median=float(valid.median()),
        label_log_difference_abs_p95=float(valid.abs().quantile(.95)),label_log_difference_abs_over_1pct=int((valid.abs()>.01).sum()),
        prior_excluded_unique_tickers=e.ticker.nunique(),prior_excluded_rows=len(e),
        prior_excluded_rows_absent_table=int(e.absent_from_daily_table.sum()),
        prior_excluded_rows_not_members=int((~e.member_on_date).sum()),
        raw_zero_targets=int((f.y_return==0).sum()),adjusted_zero_targets=int((f.target==0).sum()),
        original_financial_vintages_verified=False,delisting_returns_complete=False,
        input_sha256=hashlib.sha256((cache/'fundamental_data_full.csv').read_bytes()).hexdigest(),
        price_sha256=hashlib.sha256((cache/'prepared/daily.csv').read_bytes()).hexdigest(),
        recovery_manifest=recovery_manifest)
    save_json(out/'audit.json',audit)
    modeling=f.loc[f.canonical_selected].copy()
    modeling['source_ticker']=modeling.ticker
    modeling['ticker']=modeling.security_id
    modeling=modeling.sort_values(['ticker','datadate'],kind='stable').reset_index(drop=True)
    assert not modeling.duplicated(['ticker','datadate']).any()
    return modeling,features,px


def specs():
    return {
        'rf':(RandomForestRegressor(random_state=42,n_jobs=4),{'model__n_estimators':[150,300],'model__min_samples_leaf':[.05,.1],'model__max_features':['sqrt']}),
        'xgb':(XGBRegressor(random_state=42,n_jobs=4,tree_method='hist',max_depth=3),{'model__learning_rate':[.03,.1],'model__n_estimators':[150,300]}),
        'gbm':(LGBMRegressor(random_state=42,n_jobs=4,verbosity=-1,num_leaves=15),{'model__learning_rate':[.03,.1],'model__n_estimators':[150,300]})}


def forward_splits(d):
    qs=sorted(d.datadate.unique()); n=len(qs); splits=[]
    for k in [int(n*.5),int(n*.67),int(n*.83)]:
        if k<4 or k>=n: continue
        q=qs[k]; cutoff=d.loc[d.datadate==q,'decision_date'].min()
        train=np.flatnonzero(((d.datadate<q)&(d.label_end<cutoff)&(d.available<cutoff)).to_numpy())
        test=np.flatnonzero(((d.datadate>=q)&(d.datadate<=qs[min(n-1,k+2)])).to_numpy())
        if len(train)>=20 and len(test)>=5:
            assert d.iloc[train].label_end.max()<d.iloc[test].decision_date.min()
            splits.append((train,test))
    return splits


def train(f,features,px,out):
    windows=out/'windows'; windows.mkdir(exist_ok=True)
    manifest=dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  audit_sha256=hashlib.sha256((out/'audit.json').read_bytes()).hexdigest(),features=features,
                  market_data_sha256=hashlib.sha256(Path(__file__).with_name('market_data.py').read_bytes()).hexdigest())
    manifest_file=out/'training_manifest.json'
    if manifest_file.exists():
        assert json.loads(manifest_file.read_text())==manifest,'Inputs or code changed; use a fresh output directory.'
    else: save_json(manifest_file,manifest)
    # Missing current prices limit the inferable universe. Never filter inference
    # on whether a future exit price is present; only training labels need endpoints.
    eligible=f.loc[f.timely & f.member & f.entry_adj_close.notna()].copy()
    assert not eligible.duplicated(['security_id','datadate']).any()
    for sector,frame in eligible.groupby('gsector',sort=True):
        all_q=sorted(f.loc[f.gsector==sector,'datadate'].unique())
        for q in sorted(frame.datadate.unique()):
            if pd.Timestamp(q)<pd.Timestamp('2020-03-31'): continue
            infer=frame.loc[frame.datadate==q]
            decision=infer.decision_date.min()
            if pd.isna(decision) or decision>=px.index[-1]: continue
            dest=windows/(sector.replace(' ','_')+'_'+pd.Timestamp(q).strftime('%Y-%m-%d'))
            if dest.with_suffix('.csv').exists(): continue
            splitq=all_q[max(0,all_q.index(q)-4)]
            validation=frame.loc[(frame.datadate>=splitq)&(frame.datadate<q)&(frame.label_end<decision)&frame.target.notna()].copy()
            if validation.empty: continue
            # Purge outer training labels at the FIRST validation decision, too.
            outer_cutoff=validation.decision_date.min()
            training=frame.loc[(frame.datadate<splitq)&(frame.label_end<outer_cutoff)&(frame.available<outer_cutoff)&frame.target.notna()].reset_index(drop=True)
            splits=forward_splits(training)
            if len(training)<40 or len(validation)<10 or len(splits)<2:
                status(out,'skip',sector=sector,quarter=q,train_rows=len(training),validation_rows=len(validation))
                continue
            assert training.label_end.max()<outer_cutoff
            assert validation.label_end.max()<decision
            assert (infer.available<=infer.decision_date).all()
            scores={}; predictions={}; params={}; imps=[]
            for name,(model,grid) in specs().items():
                status(out,'training',sector=sector,quarter=q,model=name,train_rows=len(training),validation_rows=len(validation))
                pipe=Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),('model',model)])
                search=GridSearchCV(pipe,grid,cv=splits,scoring='neg_mean_squared_error',n_jobs=1,error_score='raise')
                search.fit(training[features],training.target)
                fitted=search.best_estimator_
                scores[name]=float(mean_squared_error(validation.target,fitted.predict(validation[features])))
                predictions[name]=fitted.predict(infer[features])
                params[name]=search.best_params_
                importance=fitted.named_steps['model'].feature_importances_.astype(float)
                importance=importance/importance.sum() if importance.sum()>0 else importance
                imps.extend(dict(sector=sector,quarter=q,model=name,feature=c,importance=float(v)) for c,v in zip(features,importance))
            winner=min(scores,key=scores.get)
            pred=infer[['ticker','datadate','decision_date','available','gsector']].copy().reset_index(drop=True)
            for name,v in predictions.items(): pred[name]=v
            pred['best']=predictions[winner];pred['winner']=winner
            pd.DataFrame(imps).to_csv(str(dest)+'_importance.csv',index=False)
            save_json(dest.with_suffix('.json'),dict(scores=scores,params=params,train_rows=len(training),validation_rows=len(validation),
                inference_rows=len(infer),train_label_end_max=training.label_end.max(),validation_start=outer_cutoff,
                validation_label_end_max=validation.label_end.max(),inference_date=decision))
            pred.to_csv(dest.with_suffix('.csv'),index=False)
    files=[p for p in windows.glob('*.csv') if not p.stem.endswith('_importance')]
    pd.concat([pd.read_csv(p) for p in files],ignore_index=True).to_csv(out/'predictions.csv',index=False)
    imp=pd.concat([pd.read_csv(p) for p in windows.glob('*_importance.csv')],ignore_index=True)
    imp.groupby(['model','feature']).importance.mean().reset_index().to_csv(out/'feature_importance_normalized.csv',index=False)
    records=[]
    for p in windows.glob('*.json'):
        x=json.loads(p.read_text())
        for m,s in x['scores'].items(): records.append(dict(window=p.stem,model=m,mse=s,winner=m==min(x['scores'],key=x['scores'].get)))
    pd.DataFrame(records).to_csv(out/'model_scores.csv',index=False)
    flow=[];current=f.copy()
    raw_audit=pd.read_csv(out/'row_audit.csv')
    flow.append(dict(stage='raw',rows=len(raw_audit),tickers=raw_audit.ticker.nunique()))
    flow.append(dict(stage='canonical_security_quarter',rows=len(current),tickers=current.ticker.nunique()))
    for label,condition in [('decision_in_price_window',f.decision_date.notna()),('historical_member',f.member),('disclosed_by_decision',f.timely),('quoted_at_entry',f.entry_adj_close.notna() & (f.entry_adj_close>0))]:
        current=current.loc[condition.reindex(current.index)]
        flow.append(dict(stage=label,rows=len(current),tickers=current.ticker.nunique()))
    flow.append(dict(stage='with_observed_adjusted_label_for_potential_training',rows=int(current.target.notna().sum()),tickers=current.loc[current.target.notna(),'ticker'].nunique()))
    pred=pd.read_csv(out/'predictions.csv')
    flow.append(dict(stage='actual_predictions_all_windows',rows=len(pred),tickers=pred.ticker.nunique()))
    pd.DataFrame(flow).to_csv(out/'sample_flow.csv',index=False)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--cache',required=True,type=Path);ap.add_argument('--prior',required=True,type=Path)
    ap.add_argument('--output',required=True,type=Path);ap.add_argument('--audit-only',action='store_true');a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);started=time.time()
    try:
        status(a.output,'auditing');f,features,px=prepare(a.cache,a.prior,a.output)
        if not a.audit_only: train(f,features,px,a.output)
        status(a.output,'completed',elapsed_minutes=(time.time()-started)/60)
    except BaseException as exc:
        status(a.output,'failed',error=str(exc));raise
