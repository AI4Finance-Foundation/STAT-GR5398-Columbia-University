"""Offline audit and purged quarterly ML experiment. No trading API calls."""
import argparse,json,sqlite3,hashlib,sys,platform,time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

FEATURES=['EPS','BPS','DPS','cur_ratio','quick_ratio','cash_ratio','acc_rec_turnover','debt_ratio','debt_to_equity','roe','net_income_ratio','gross_margin','operating_margin','ebitda_margin','pretax_margin','asset_turnover','debt_to_assets','debt_to_capital','lt_debt_to_capital','fcf_per_share','ocf_per_share','cash_per_share','revenue_per_share']
def main(source,output,benchmark_dir=None):
 source=Path(source); out=Path(output); out.mkdir(parents=True,exist_ok=True)
 start=time.time(); data=source/'data'; base=source/'baseline_run_2020-06-30_20261005_223942'
 if not base.exists(): base=Path(__file__).resolve().parent/'cache'
 f=pd.read_csv(data/'fundamental_data_full.csv'); f=f.sort_values(['ticker','datadate']).reset_index(drop=True)
 for col in ['datadate','filing_date','accepted_date','actual_tradedate']: f[col]=pd.to_datetime(f[col],errors='coerce')
 f['label_end']=f.groupby('ticker')['actual_tradedate'].shift(-1)
 f['available']=f[['datadate','filing_date','accepted_date']].max(axis=1)+pd.Timedelta(days=1)
 f.loc[f['filing_date'].isna() & f['accepted_date'].isna(),'available']=pd.NaT
 m=pd.read_csv(data/'sp500_historical_constituents.csv',parse_dates=['date']).sort_values('date')
 md=m['date'].to_numpy(); ms=[set(str(x).split(',')) for x in m['tickers']]
 def members(d):
  i=np.searchsorted(md,np.datetime64(d),side='right')-1
  return ms[i] if i>=0 else set()
 candidate=f[f['y_return'].notna() & f['datadate'].between('2015-01-01','2025-09-30')].copy()
 keep=np.array([t in members(d) for t,d in zip(candidate['ticker'],candidate['datadate'])]); pit=candidate[keep].copy()
 prepared=pd.read_csv(source/'prepared/final_ratios.csv')
 assert len(pit)==len(prepared)==19007
 selected=pd.read_csv(base/'selection_output/stock_selected.csv',parse_dates=['trade_date'])
 con=sqlite3.connect(f'file:{data}/finrl_trading.db?mode=ro',uri=True)
 px=pd.read_sql_query('SELECT ticker,date,open,close,adj_close FROM price_data',con,parse_dates=['date']); con.close()
 assert not px.duplicated(['ticker','date']).any(); assert not selected.duplicated(['tic','trade_date']).any()
 pricepool=set(px['ticker']); missing=sorted(set(selected['tic'])-pricepool)
 counts=selected[selected.tic.isin(missing)].groupby('tic').size().rename('selected_records').reset_index(); counts.to_csv(out/'missing_price_tickers.csv',index=False)
 stockpx=px[~px.ticker.isin(['SPY','QQQ'])]; days=pd.DatetimeIndex(sorted(stockpx.loc[stockpx.date.between('2020-05-01','2025-09-30'),'date'].unique()))
 close=px.pivot(index='date',columns='ticker',values='adj_close').reindex(days)
 op=px.assign(open_adj=px.open*px.adj_close/px.close).pivot(index='date',columns='ticker',values='open_adj').reindex(days)
 sp=selected[selected.trade_date.between(days[0],days[-1])].copy(); sp['market_date']=sp.trade_date.isin(days)
 joined=sp.merge(px[['ticker','date','adj_close']],left_on=['tic','trade_date'],right_on=['ticker','date'],how='left',validate='one_to_one')
 coverage=joined.groupby('trade_date').agg(selected=('tic','size'),quoted=('adj_close','count'),market_date=('market_date','first')); coverage['quote_fraction']=coverage.quoted/coverage.selected; coverage.to_csv(out/'original_signal_coverage.csv')
 origweights=selected.pivot(index='trade_date',columns='tic',values='predicted_return').notna().astype(float); origweights=origweights.div(origweights.sum(axis=1),axis=0).reindex(days).fillna(0)
 origret=(close/op-1).reindex(columns=origweights.columns)
 # Reproduction diagnostic only: explicitly report unpriced weight, do not call it a valid portfolio.
 diagnostic=pd.DataFrame(index=days); diagnostic['unpriced_weight']=(origweights*origret.isna()).sum(axis=1)
 diagnostic['quoted_contribution']=(origweights*origret).sum(axis=1,min_count=1)
 diagnostic['turnover']=origweights.diff().abs().sum(axis=1); diagnostic.iloc[0,diagnostic.columns.get_loc('turnover')]=origweights.iloc[0].sum()
 diagnostic['cost']=diagnostic.turnover*.001; diagnostic['partial_nav']=(1+diagnostic.quoted_contribution-diagnostic.cost).cumprod(); diagnostic.to_csv(out/'biased_baseline_diagnostic.csv')
 wins={}
 for sector in range(10,65,5):
  w=pd.read_csv(base/f'results/sector{sector}/df_best_model_name.csv',index_col=0)
  for v in w.to_numpy().ravel():
   if pd.notna(v):wins[str(v)]=wins.get(str(v),0)+1
 print('original winners',wins,flush=True)
 # Fixed accounting features, fixed two-model validation selection; no return-based strategy tuning.
 X=f[FEATURES].replace([np.inf,-np.inf],np.nan)
 inference=f[f.datadate.between('2015-01-01','2025-09-30')].copy()
 inference=inference[[t in members(d) for t,d in zip(inference.ticker,inference.datadate)]]
 weights=pd.DataFrame(0.,index=days,columns=sorted(set(pit.ticker)))
 events=[]; predrows=[]; importance=[]; model_scores=[]; current=pd.Series(0.,index=weights.columns); last_cut=None
 cutoffs=pd.date_range('2020-06-30','2025-06-30',freq='QE')
 for cutoff in cutoffs:
  execution=days[days>cutoff]
  if not len(execution):continue
  execution=execution[0]
  known=pit[(pit.available<=cutoff)&(pit.label_end<=cutoff)&pit.y_return.notna()]
  quarters=sorted(known.datadate.unique()); qval=quarters[-4:]; qtrain=quarters[-20:-4]
  train=known[known.datadate.isin(qtrain)]; val=known[known.datadate.isin(qval)]
  assert len(train)>0 and len(val)>0
  assert train.label_end.max()<=cutoff and val.label_end.max()<=cutoff
  eligible=inference[(inference.available<=cutoff)&inference.ticker.isin(members(cutoff))].sort_values(['ticker','available','datadate']).groupby('ticker').tail(1)
  imp=SimpleImputer(strategy='median',keep_empty_features=True); xt=imp.fit_transform(X.loc[train.index]); xv=imp.transform(X.loc[val.index])
  models={'RF':RandomForestRegressor(n_estimators=80,max_depth=6,min_samples_leaf=15,max_features=.7,n_jobs=2,random_state=5398),'ET':ExtraTreesRegressor(n_estimators=80,max_depth=6,min_samples_leaf=15,max_features=.7,n_jobs=2,random_state=5398)}
  mse={}
  for name,model in models.items():
   model.fit(xt,train.y_return); mse[name]=mean_squared_error(val.y_return,model.predict(xv));model_scores.append(dict(cutoff=str(cutoff.date()),model=name,mse=mse[name]))
  winner=min(mse,key=mse.get); model=models[winner]
  # Winner refit using only realized labels known at cutoff, same fixed historical window.
  fit=known[known.datadate.isin(qtrain+qval)]; imp=SimpleImputer(strategy='median',keep_empty_features=True); model.fit(imp.fit_transform(X.loc[fit.index]),fit.y_return)
  eligible=eligible.copy();eligible['prediction']=model.predict(imp.transform(X.loc[eligible.index]));eligible=eligible.sort_values(['prediction','ticker'],ascending=[False,True])
  chosen=eligible.head(max(1,int(np.ceil(len(eligible)*.25)))); current[:]=0.;target=1/len(chosen)
  for row in chosen.itertuples():
   available_price= row.ticker in close and pd.notna(close.at[execution,row.ticker]) and close.at[execution,row.ticker]>0
   if available_price:current[row.ticker]=target
   predrows.append(dict(cutoff=str(cutoff.date()),execution=str(execution.date()),ticker=row.ticker,prediction=row.prediction,target_weight=target,executed_weight=target if available_price else 0,available=str(row.available.date()),price_available=available_price))
  nxt=cutoffs[cutoffs>cutoff]; until=nxt[0] if len(nxt) else pd.Timestamp('2025-09-30')
  block=days[(days>=execution)&(days<=until)];weights.loc[block]=current.to_numpy()
  events.append(dict(cutoff=str(cutoff.date()),execution=str(execution.date()),train_rows=len(train),validation_rows=len(val),train_label_end=str(train.label_end.max().date()),validation_label_end=str(val.label_end.max().date()),model=winner,universe=len(eligible),selected=len(chosen),unexecuted_weight=1-current.sum()))
  importance.extend([dict(cutoff=str(cutoff.date()),feature=k,importance=v) for k,v in zip(FEATURES,model.feature_importances_)])
  print(cutoff.date(),winner,len(train),len(val),'selected',len(chosen),'cash',round(1-current.sum(),3),flush=True)
 # Hold shares between quarterly close executions, cash reserved for unavailable entry targets.
 initial_day=days[days<pd.Timestamp(events[0]['execution'])][-1]
 simdays=days[days>=initial_day]; executions=set(pd.Timestamp(x['execution']) for x in events)
 def simulate(targets,name):
  shares=pd.Series(0.,index=targets.columns); cash=1.;previous=1.; rows=[]
  for day in simdays:
   p=close.loc[day].reindex(shares.index); held=shares>1e-15
   if p[held].isna().any():raise ValueError(f'Unvalued held assets {name} {day}: {p[held][p[held].isna()].index.tolist()}')
   stockvalue=shares*p.fillna(0);before=float(stockvalue.sum()+cash); turn=0.; cost=0.
   if day in executions:
    target=targets.loc[day]; assert abs(target.sum())<=1.0000001
    desired=before*target; turn=float((desired-stockvalue).abs().sum()/before)
    cost=.001*turn*before
    # Cost paid pro-rata from target stock and cash allocations. Gross targets define turnover.
    shares=(before-cost)*target/p; shares=shares.fillna(0)
    cash=(before-cost)*(1-target.sum())
   after=float((shares*p.fillna(0)).sum()+cash)
   rows.append(dict(date=day,gross_ret=before/previous-1,turnover=turn,cost=cost/previous,net_ret=after/previous-1,nav=after,cash_weight=cash/after));previous=after
  return pd.DataFrame(rows).set_index('date')
 strategy=simulate(weights,'Purged ML'); strategy.to_csv(out/'strategy_daily.csv'); weights.loc[simdays].to_csv(out/'target_weights.csv')
 benchmarks={}
 for tic in ['SPY','QQQ']:
  series=close[tic].copy()
  if benchmark_dir and (Path(benchmark_dir)/f'{tic}_download.csv').is_file():
   b=pd.read_csv(Path(benchmark_dir)/f'{tic}_download.csv',index_col=0,parse_dates=True); series=b['Adj Close'].reindex(days)
  benchmarks[tic]=series
 common=simdays
 for series in benchmarks.values():common=common.intersection(series[series.notna()].index)
 assert len(common)>200
 assert np.isfinite(strategy[['net_ret','nav']].to_numpy()).all()
 assert np.allclose(strategy.nav.to_numpy(),(1+strategy.net_ret).cumprod().to_numpy())
 # Compare exactly from common initial close; reset strategy NAV at common start.
 nav=pd.DataFrame({'Purged ML':strategy.loc[common,'nav']/strategy.loc[common[0],'nav']},index=common)
 for tic,series in benchmarks.items():nav[tic]=series.loc[common]/series.loc[common[0]]
 nav.to_csv(out/'comparison_nav.csv')
 rows=[]
 for name in nav:
  r=nav[name].pct_change(fill_method=None).iloc[1:];years=(common[-1]-common[0]).days/365.25
  rows.append(dict(strategy=name,start=str(common[0].date()),end=str(common[-1].date()),days=len(r),final_nav=nav[name].iloc[-1],cumulative_return=nav[name].iloc[-1]-1,cagr=nav[name].iloc[-1]**(1/years)-1,volatility=r.std(ddof=1)*np.sqrt(252),sharpe=r.mean()/r.std(ddof=1)*np.sqrt(252),max_drawdown=(nav[name]/nav[name].cummax()-1).min(),mean_turnover=strategy.loc[common,'turnover'].mean() if name=='Purged ML' else 0.,total_cost_sum=strategy.loc[common,'cost'].sum() if name=='Purged ML' else 0.))
 metrics=pd.DataFrame(rows);metrics.to_csv(out/'metrics.csv',index=False)
 fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True);nav.plot(ax=axes[0]);axes[0].set_ylabel('NAV at common initial close = 1');axes[0].set_title('Restricted price universe: purged ML vs SPY / QQQ')
 (nav/nav.cummax()-1).plot(ax=axes[1],legend=False);axes[1].set_ylabel('Drawdown');fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
 pd.DataFrame(events).to_csv(out/'timing_and_rebalance_audit.csv',index=False); pd.DataFrame(predrows).to_csv(out/'purged_selected.csv',index=False);pd.DataFrame(model_scores).to_csv(out/'model_validation_scores.csv',index=False)
 fi=pd.DataFrame(importance).groupby('feature').importance.mean().sort_values(ascending=False);fi.to_csv(out/'feature_importance.csv')
 fig,ax=plt.subplots(figsize=(8,5));fi.head(12).sort_values().plot.barh(ax=ax);ax.set_title('Mean impurity importance across purged refits');fig.tight_layout();fig.savefig(out/'feature_importance.png',dpi=160);plt.close(fig)
 summary=dict(original_rows=len(selected),original_tickers=selected.tic.nunique(),original_dates=[str(selected.trade_date.min().date()),str(selected.trade_date.max().date())],missing_tickers=missing,missing_records_all=int(counts.selected_records.sum()),price_rows=len(px),price_tickers=px.ticker.nunique(),prepared_rows=len(prepared),prepared_tickers=prepared.tic.nunique(),quarters=prepared.date.nunique(),pre_pit_rows=len(candidate),pre_pit_tickers=candidate.ticker.nunique(),pit_removed_rows=int((~keep).sum()),pit_removed_tickers=candidate.ticker.nunique()-pit.ticker.nunique(),original_winners=wins,non_market_records=int((~joined.market_date).sum()),missing_market_records=int((joined.market_date & joined.adj_close.isna()).sum()),market_records=int(joined.market_date.sum()),new_missing_targets=sum(not x['price_available'] for x in predrows),new_target_records=len(predrows),new_winners=pd.DataFrame(events).model.value_counts().to_dict(),mean_unexecuted_weight=float(pd.DataFrame(events).unexecuted_weight.mean()),run_seconds=time.time()-start,python=sys.version,executable=sys.executable,platform=platform.platform(),features=FEATURES,comparison_full_window=common[0]==simdays[0],validation='Assertions passed: unique price keys, unique selection keys, PIT prepared counts, all training/validation label ends <= cutoff, no missing held valuation, weights <= 1, finite common NAV')
 summary['input_sha256']={(str(p.relative_to(source)) if p.is_relative_to(source) else 'cache/'+str(p.relative_to(base))):hashlib.sha256(p.read_bytes()).hexdigest() for p in [data/'fundamental_data_full.csv',data/'sp500_historical_constituents.csv',source/'prepared/final_ratios.csv',base/'selection_output/stock_selected.csv']}
 (out/'audit_summary.json').write_text(json.dumps(summary,indent=2,default=str)); print(metrics.to_string(index=False)); print('PASS',json.dumps(summary,default=str),flush=True)
 return metrics
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',default='results');p.add_argument('--benchmark-dir');a=p.parse_args();main(a.source,a.output,a.benchmark_dir)
