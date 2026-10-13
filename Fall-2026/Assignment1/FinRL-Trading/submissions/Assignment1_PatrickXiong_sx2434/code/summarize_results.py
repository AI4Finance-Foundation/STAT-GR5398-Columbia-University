"""Regenerate compact metrics, model importance summaries and research charts."""
from pathlib import Path
import sys,json,os
import numpy as np,pandas as pd,joblib
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];RUN=Path(os.environ.get('FINRL_RUN_DIR',str(ROOT/'run')))
sys.path.insert(0,str(Path(__file__).resolve().parent/'support'))
from portfolio_math import metrics
from build_features import FEATURES
OUT=RUN/'backtest';nav=pd.read_csv(OUT/'nav.csv',index_col=0,parse_dates=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(9,3.5))
for key,label,color in [('SPY80_Best20_quarterly','80% SPY + 20% stocks','#075985'),('SPY80_Best20_quarterly_fee20bps_delay1','20bps + one-day delay','#0891b2'),('SPY','SPY','#64748b'),('QQQ','QQQ','#b45309')]:ax.plot(nav.index,nav[key],label=label,color=color,lw=1.6)
ax.set(ylabel='Wealth from $1',title='CRSP total returns | May 2020 - September 2025');ax.grid(alpha=.18);ax.legend(ncol=2,fontsize=8,frameon=False,loc='upper left');fig.tight_layout();fig.savefig(OUT/'nav.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(9,2.8))
for key,label,color in [('SPY80_Best20_quarterly','Primary','#075985'),('SPY','SPY','#64748b'),('QQQ','QQQ','#b45309')]:ax.plot(nav.index,(nav[key]/nav[key].cummax().clip(lower=1)-1)*100,label=label,color=color,lw=1.3)
ax.set(ylabel='Drawdown (%)');ax.grid(alpha=.18);ax.legend(ncol=3,fontsize=8,frameon=False);fig.tight_layout();fig.savefig(OUT/'drawdown.png',dpi=180);plt.close(fig)
importance=[]
for path in sorted((RUN/'models').glob('*.joblib')):
 a=joblib.load(path);m=a['model'];family='rf' if '_rf_seed_' in path.name else ('xgb' if '_xgb' in path.name else 'gbm')
 vals=m.booster_.feature_importance(importance_type='gain') if family=='gbm' else m.feature_importances_
 vals=np.array(vals,float);vals=vals/vals.sum() if vals.sum() else vals
 importance.append({'family':family,**{f:float(vals[2*i:2*i+2].sum()) for i,f in enumerate(FEATURES)}})
imp=pd.DataFrame(importance).groupby('family')[FEATURES].mean().T
imp.to_csv(OUT/'feature_importance.csv',index_label='feature')
slice_rows=[]
for name in nav:
 n=nav.loc['2020-06-30':,name];r=n.pct_change().fillna(0);slice_rows.append({'strategy':name,**metrics(r)})
pd.DataFrame(slice_rows).to_csv(OUT/'june30_descriptive_slice.csv',index=False)
pd.read_csv(OUT/'SPY80_Best20_quarterly_selected.csv').to_csv(OUT/'stock_selected.csv',index=False)
comparison=pd.read_csv(OUT/'comparison.csv').set_index('strategy');scores=pd.read_csv(RUN/'model_scores.csv');held=pd.read_csv(OUT/'held_exit_events.csv');counts=pd.read_csv(OUT/'selection_counts.csv');primary=pd.read_csv(OUT/'SPY80_Best20_quarterly_daily.csv')
facts={'metrics':comparison.reset_index().to_dict('records'),'model_wins':scores[scores.winner].model.value_counts().to_dict(),'feature_importance':imp.reset_index().rename(columns={'index':'feature'}).to_dict('records'),'primary_held_exits':held[held.strategy.eq('SPY80_Best20_quarterly')].to_dict('records'),'selected_count_range':[int(counts.selected.min()),int(counts.selected.max())],'primary_maximum_pending_exit_weight':float(primary.pending_exit_weight.max()),'primary_mean_cash_weight':float(primary.cash_weight.mean()),'full_window':['2020-05-01','2025-09-30'],'days':len(nav),'excess_cumulative_percentage_points':float(100*(comparison.loc['SPY80_Best20_quarterly','cumulative']-comparison.loc['SPY','cumulative']))}
(RUN/'research_facts.json').write_text(json.dumps(facts,indent=2));print(json.dumps({k:v for k,v in facts.items() if k not in ['metrics','feature_importance','primary_held_exits']},indent=2))
