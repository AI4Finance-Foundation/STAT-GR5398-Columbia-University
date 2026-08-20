from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from src.backtest import benchmark_returns, equal_weight_returns, weights_to_returns
from src.data import daily_returns, load_prices, unique_tickers
from src.metrics import drawdown, equity_curve, performance_metrics
from src.plotting import plot_active_group, plot_drawdown, plot_equity, plot_weights
from src.report_utils import write_readme, write_reports
from src.strategy import run_adaptive_rotation
ROOT=Path(__file__).resolve().parent

def load_config(path):
    try:
        import yaml
        return yaml.safe_load(((ROOT/path) if not Path(path).is_absolute() else Path(path)).read_text(encoding="utf-8"))
    except ImportError:
        return {"start_date":"2015-01-01","end_date":None,"data_source":"yfinance","cache_path":"data/cache/prices.csv","rebalance_frequency":"W-FRI","ir_lookback_weeks":12,"ir_benchmark":"SPY","residual_momentum_lookback_days":63,"regime_ma_window":200,"top_k_assets":2,"risk_free_rate":0.0,"drawdown_threshold":-0.12,"transaction_cost_bps":2.0,"defensive_assets":["TLT","IEF","SHY"],"benchmark":"SPY","qqq_rolling_windows":[21,63,126],"qqq_top_k":5,"assignment1_selected_path":"../../../Assignment1/submissions/Assignment1_rayzhao_rz2759/outputs_step2/stock_selected.csv","groups":{"GROWTH_TECH":["AAPL","MSFT","NVDA","META","AMZN","GOOGL","TSLA"],"REAL_ASSETS":["XOM","CVX","FCX","BHP","GLD","SLV"],"DEFENSIVE":["TLT","IEF","XLU","XLV","IAU","SHY","UUP"],"DEFAULT":["SPY","QQQ","IAU","XLU","XLV"]}}

def run_pipeline(cfg,mode,run_all):
    res=ROOT/"results"; fig=ROOT/"figures"
    for d in [res,fig,ROOT/"blog",ROOT/"data/cache"]: d.mkdir(parents=True,exist_ok=True)
    tickers=unique_tickers(cfg["groups"],["QQQ","SPY"]); prices=load_prices(cfg,ROOT,tickers); ret=daily_returns(prices).dropna(how="all")
    outputs={}; wm={}; dm={}; turn={}; reb={}
    if run_all or mode in {"original","all"}:
        w,d=run_adaptive_rotation(prices,ret,cfg,ROOT,False); r,dw,t=weights_to_returns(ret,w,cfg.get("transaction_cost_bps",0)); outputs["original"]=r; wm["original"]=dw; dm["original"]=d; turn["original"]=t; reb["original"]=len(w); d[orig_cols()].to_csv(res/"weekly_decisions_original.csv",index=False); dw.to_csv(res/"weights_original.csv",index_label="date")
    if run_all or mode in {"modified","all"}:
        w,d=run_adaptive_rotation(prices,ret,cfg,ROOT,True); r,dw,t=weights_to_returns(ret,w,cfg.get("transaction_cost_bps",0)); outputs["modified"]=r; wm["modified"]=dw; dm["modified"]=d; turn["modified"]=t; reb["modified"]=len(w); d[mod_cols()].to_csv(res/"weekly_decisions_modified.csv",index=False); dw.to_csv(res/"weights_modified.csv",index_label="date")
    if run_all or mode in {"benchmarks","all"}:
        outputs["qqq"]=benchmark_returns(ret,"QQQ"); outputs["spy"]=benchmark_returns(ret,"SPY"); outputs["equal_weight_optional"]=equal_weight_returns(ret,tickers); turn.update({"qqq":0,"spy":0,"equal_weight_optional":0}); reb.update({"qqq":0,"spy":0,"equal_weight_optional":0})
    if run_all:
        names=["original","modified","qqq","spy","equal_weight_optional"]; returns=pd.concat([outputs[n].rename(n) for n in names],axis=1).dropna()
        eq=returns.apply(equity_curve); dd=eq.apply(drawdown); eq.reset_index(names="date").to_csv(res/"equity_curves.csv",index=False); dd.reset_index(names="date").to_csv(res/"drawdowns.csv",index=False)
        m=pd.DataFrame.from_dict({n:performance_metrics(returns[n],reb.get(n,0),turn.get(n,0)) for n in names},orient="index"); m.to_csv(res/"metrics.csv")
        plot_equity(eq,fig/"equity_curve_original_modified_qqq.png"); plot_drawdown(dd,fig/"drawdown_original_modified_qqq.png"); plot_weights(wm["original"],fig/"rolling_weights_original.png","Rolling Weights: Original"); plot_weights(wm["modified"],fig/"rolling_weights_modified.png","Rolling Weights: Modified"); plot_active_group(dm["original"],dm["modified"],fig/"active_group_history.png"); write_reports(ROOT,cfg,m); write_readme(ROOT,cfg,m)

def orig_cols(): return ["rebalance_date","regime_state","selected_group","group_ir_growth_tech","group_ir_real_assets","group_ir_defensive","group_ir_default","selected_assets","residual_momentum_scores","risk_state"]
def mod_cols(): return ["rebalance_date","regime_state","selected_group","group_ir_growth_tech","group_ir_real_assets","group_ir_defensive","group_ir_default","qqq_rolling_candidates","qqq_rolling_scores","selected_assets","residual_momentum_scores","risk_state"]
if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/assignment2_config.yaml"); p.add_argument("--run-all",action="store_true"); p.add_argument("--mode",choices=["all","original","modified","benchmarks"],default="all"); a=p.parse_args(); run_pipeline(load_config(a.config),a.mode,a.run_all)
