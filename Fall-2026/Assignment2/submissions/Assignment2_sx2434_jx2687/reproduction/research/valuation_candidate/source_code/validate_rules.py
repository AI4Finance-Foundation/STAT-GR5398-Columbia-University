"""Compare only predeclared rules. Restrict portfolio periods to dev/val."""
import argparse
from pathlib import Path
from datetime import timedelta
import pandas as pd
import numpy as np
from shared.data_view import DataBundle,DataView
from shared.guard import student_call
from shared.metrics import performance
from shared.periods import PERIODS
from selection.config import load
from selection import MonthlyScores
from portfolio.backtester import run_backtest, BacktestConfig
from valuation.student_valuation import valuations, rejected

OUT=Path("../results")
class Scores:
    def __init__(self,base,flags):
        self.base,self.flags=base,flags
    def get(self,day):
        items=self.base.get(day)
        self.cutoff=self.base.cutoff
        return tuple(p for p in items if p.symbol not in self.flags[day])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("period",choices=["dev","val"])
    period=ap.parse_args().period
    _,_,params,prices,_,univ=load("selection/configs/growth.json")
    b=DataBundle.load(prices=prices)
    def calc(day, tickers):
        with student_call():
            return valuations(day,tickers,DataView(b,day))
    msft=calc("2021-06-30",["MSFT"]).iloc[0]
    assert msft.history_n==50
    np.testing.assert_allclose(msft.hist_mid,171.60135193,atol=1e-5)
    scores=MonthlyScores()
    start,end=PERIODS[period]
    cal=prices.sessions.to_series().loc[str(start):str(end)]
    weeks=cal.groupby([cal.dt.isocalendar().year,cal.dt.isocalendar().week]).first()
    rules=["baseline","dcf_peer","two_of_three","dcf_confirmation"]
    decisions={k:{} for k in rules}
    for day in weeks:
        dt=day.date()
        prior=prices.last_session_on_or_before(dt-timedelta(days=1))
        df=calc(prior,[p.symbol for p in scores.get(dt) if p.bucket=="growth_tech"])
        for rule in rules:
            decisions[rule][dt]=set() if rule=="baseline" else rejected(df,rule)
    stats=[]; curves={}
    for rule in rules:
        res=run_backtest(prices,Scores(scores,decisions[rule]),params,univ,BacktestConfig(start=start,end=end))
        metrics=performance(res.equity)
        curves[rule]=res.equity
        stats.append(dict(rule=rule,**metrics,trades=len(res.trades),turnover=round(res.trades.notional.abs().sum()/res.equity.mean()/(len(res.equity)/252),2)))
    table=pd.DataFrame(stats).set_index("rule")
    pd.DataFrame(curves).to_csv(OUT/f"{period}_curves.csv")
    table.to_csv(OUT/f"{period}_rules.csv")
    if period=="dev":
        prior=pd.read_csv("../../Patrick_dev/results/iteration_01/comparison.csv").set_index("method")
        for key in prior.index:
            for metric in ("sharpe","cagr_pct","max_drawdown_pct","trades"):
                assert np.isclose(prior.at[key,metric],table.at[key,metric]),(key,metric)
    print(table.to_string(),flush=True)
if __name__=="__main__":
    main()
