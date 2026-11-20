"""Reconstruct realized positions and nominal costs from saved executions only."""
from pathlib import Path
import argparse,sys
import pandas as pd,numpy as np
p=argparse.ArgumentParser();p.add_argument('--task-root',default='.');a=p.parse_args();r=Path(a.task_root).resolve();sys.path.insert(0,str(r/'work/retraining/baseline'))
from shared.prices import PriceStore
px=PriceStore(r/'work/course/Fall-2026/Assignment2/source_code/data/prices/daily_adjusted.parquet').close
rows=[]
for variant in ['course','retrained_baseline','retrained_improved']:
 out=r/'work/retraining/evaluation'/variant/'output'
 for period in ['dev','val','test','forward']:
  files=list(out.glob(f'ablation_{period}_*.csv'))
  if not files:continue
  ab=pd.read_csv(files[0]);sets={};current=[]
  for group in ['A_selection','B_valuation']:
   name=ab[ab.run.str.endswith(group)].iloc[0].run;folder=out/name
   eq=pd.read_csv(folder/'equity.csv',index_col=0,parse_dates=True).equity
   trades=pd.read_csv(folder/'trades.csv');trades['date']=pd.to_datetime(trades.date)
   orders=trades.pivot_table(index='date',columns='symbol',values='qty',aggfunc='sum',fill_value=0)
   qty=orders.reindex(eq.index,fill_value=0).cumsum();held=qty.gt(0);sets[group]=held
   # Fill over the full history before selecting dates, so pre-period stale quotes are explicit.
   marks=px.reindex(columns=qty.columns).ffill().reindex(eq.index);weights=(qty*marks).div(eq,axis=0)
   stock_exposure=weights.sum(axis=1)
   current.append({'variant':variant,'period':period,'group':group,'mean_holdings':held.sum(axis=1).mean(),'max_holdings':int(held.sum(axis=1).max()),'minimum_reconstructed_shares':qty.min().min(),'max_realized_stock_weight_pct':weights.max().max()*100,'mean_realized_stock_exposure_pct':stock_exposure.mean()*100,'max_realized_stock_exposure_pct':stock_exposure.max()*100,'min_implied_cash_pct':(1-stock_exposure).min()*100,'nominal_cost_estimate_dollars':trades.notional.abs().sum()*.0006,'trades':len(trades)})
  cols=sets['A_selection'].columns.union(sets['B_valuation'].columns)
  diff=sets['A_selection'].reindex(columns=cols,fill_value=False).ne(sets['B_valuation'].reindex(columns=cols,fill_value=False)).any(axis=1)
  for row in current:row['held_names_differ_days_pct']=diff.mean()*100;rows.append(row)
dest=r/'outputs/retraining/analysis';dest.mkdir(exist_ok=True);pd.DataFrame(rows).to_csv(dest/'holdings_and_costs.csv',index=False);print('Saved holdings/cost records:',len(rows))
