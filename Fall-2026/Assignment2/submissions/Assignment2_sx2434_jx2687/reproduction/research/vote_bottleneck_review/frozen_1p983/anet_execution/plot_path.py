"""Weekly position carried after the open, first validation rebalance to June 6."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

here=Path(__file__).resolve().parent
out=here.parents[2]/"final_report_material"
trace=pd.read_csv(here/"trace/weekly_anet.csv",parse_dates=["day"])
end=pd.Timestamp("2022-06-06")
fig,axes=plt.subplots(2,1,figsize=(8.4,4.75),sharex=True,sharey=True)
fig.subplots_adjust(left=.092, right=.905, bottom=.12, top=.84, hspace=.35)
fig.suptitle("ANET target and actual holdings",x=.09,y=.98,ha="left",fontsize=15)
for ax,(arm,title,color) in zip(axes,[("baseline","Selection only","#38566b"),("filtered","1.983 valuation filter","#087568")]):
    data=trace[(trace.arm==arm) & (trace.day<=end)].sort_values("day")
    for d in data.loc[data.veto,"day"]:
        ax.axvspan(d,d+pd.Timedelta(days=7),facecolor="#f5e7ce",zorder=-2)
    ax.step(data.day,data.desired,where="post",linestyle="--",color="#917248",lw=1.6,label="Target shares")
    ax.step(data.day,data.qty_after,where="post",color=color,lw=2.2,label="Shares held")
    for field,offset,col in [("desired",8,"#78582d"),("qty_after",-9,color)]:
        val=data.iloc[-1][field]
        ax.plot(end,val,"o",ms=3,color=col)
        ax.annotate(str(int(val)),(end,val),xytext=(6,offset),textcoords="offset points",color=col,va="center")
    ax.set_ylim(-12,438)
    ax.set_yticks([0,100,200,300,400])
    ax.set_ylabel("Shares")
    ax.set_title(title,loc="left",fontsize=10)
    ax.grid(axis="y",color="#e3e8e9",lw=.65)
    ax.spines[["top","right"]].set_visible(False)
axes[0].legend(loc="lower left",ncol=2,frameon=False,fontsize=9)
axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
axes[-1].set_xlabel("2022   |   Shaded weeks indicate an ANET veto")
axes[-1].set_xlim(trace.day.min()-pd.Timedelta(days=2),end+pd.Timedelta(days=5))
fig.savefig(out/"anet_target_and_fills.png",dpi=170)
