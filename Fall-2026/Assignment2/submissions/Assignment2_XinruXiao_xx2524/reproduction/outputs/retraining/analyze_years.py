import argparse
from pathlib import Path
import pandas as pd,numpy as np
p=argparse.ArgumentParser();p.add_argument('--task-root',default='.');a=p.parse_args();r=Path(a.task_root).resolve();rows=[]
for variant in ['course','retrained_baseline','retrained_improved']:
 out=r/'work/retraining/evaluation'/variant/'output'
 for f in out.glob('*/equity.csv'):
  d=pd.read_csv(f,index_col=0,parse_dates=True).equity.sort_index();ret=d.pct_change().dropna()
  for year,g in ret.groupby(ret.index.year):
   curve=(1+g).cumprod();peak=curve.cummax().clip(lower=1)
   q05=g.quantile(.05);tail=g[g<=q05]
   rows.append({'variant':variant,'run':f.parent.name,'year':year,'sessions':len(g),'total_return_pct':(curve.iloc[-1]-1)*100,'max_drawdown_pct':(curve/peak-1).min()*100,'volatility_pct':g.std(ddof=1)*np.sqrt(252)*100,'sharpe_rf0':g.mean()/g.std(ddof=1)*np.sqrt(252) if g.std(ddof=1)>0 else None,'historical_daily_var95_loss_pct':max(0.,-q05*100),'historical_daily_cvar95_loss_pct':max(0.,-tail.mean()*100)})
dest=r/'outputs/retraining/analysis';dest.mkdir(exist_ok=True);pd.DataFrame(rows).to_csv(dest/'annual_results.csv',index=False);print('Annual records:',len(rows))
