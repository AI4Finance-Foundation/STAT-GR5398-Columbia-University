from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parents[1]
nav=pd.read_csv(P/'results/three_strategy_nav.csv',index_col=0,parse_dates=True)
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(2,1,figsize=(9,5.4),sharex=True,gridspec_kw={'height_ratios':[2,1]})
for label,col,color in [('QQQ enhancement','QQQ80_Best20_quarterly','#075985'),('Active validation','QQQtarget_Best_top5_quarterly','#7c3aed'),('SPY enhancement','WRDS_SPY80_RELBest20_quarterly','#0284c7'),('Active RF comparator','QQQtarget_RF_top5_quarterly','#dc2626'),('QQQ','QQQ','#b45309'),('SPY','SPY','#64748b')]:
 ax[0].plot(nav.index,nav[col],label=label,color=color,lw=1.3)
 ax[1].plot(nav.index,100*(nav[col]/nav[col].cummax().clip(lower=1)-1),color=color,lw=1.1)
ax[0].legend(ncol=3,fontsize=7.7,frameon=False,loc='upper left');ax[0].set_ylabel('Growth of $1');ax[1].set_ylabel('Drawdown (%)')
for a in ax:a.grid(alpha=.18)
fig.tight_layout();fig.savefig(P/'results/three_strategy_paths.png',dpi=180);plt.close(fig)
