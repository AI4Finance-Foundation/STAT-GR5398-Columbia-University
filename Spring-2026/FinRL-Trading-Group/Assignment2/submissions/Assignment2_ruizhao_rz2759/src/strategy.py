from __future__ import annotations
from pathlib import Path
import pandas as pd
from .signals import group_information_ratios, qqq_rolling_candidates, regime_state, residual_momentum_scores, weekly_returns

def run_adaptive_rotation(prices,ret,config,root:Path,modified=False):
    groups={k.upper():v for k,v in config["groups"].items()}; wret=weekly_returns(ret,config.get("rebalance_frequency","W-FRI"))
    dates=wret.index; regs=regime_state(prices,dates,int(config.get("regime_ma_window",200)))
    all_t=sorted({t for v in groups.values() for t in v}); cur={t:0 for t in all_t}; nav=peak=1.0; rows=[]; dec=[]
    a1=(root/config.get("assignment1_selected_path","")).resolve()
    for i,d in enumerate(dates):
        if i>0:
            prev=wret.iloc[i-1]; nav*=1+sum(cur.get(t,0)*float(prev.get(t,0) or 0) for t in all_t); peak=max(peak,nav)
        dd=nav/peak-1
        ir=group_information_ratios(wret,groups,d,int(config.get("ir_lookback_weeks",12)),str(config.get("ir_benchmark","SPY")).lower())
        cand=["DEFENSIVE","DEFAULT"] if regs.loc[d]=="risk_off" else list(groups)
        sg=max(cand,key=lambda g: ir.get(g,-999) if pd.notna(ir.get(g)) else -999); assets=list(groups[sg])
        qnames=[]; qscores={}
        if modified and sg=="GROWTH_TECH":
            qnames,qscores=qqq_rolling_candidates(prices,d,assets,[int(x) for x in config.get("qqq_rolling_windows",[21,63,126])],int(config.get("qqq_top_k",5)),a1)
            if qnames: assets=[t for t in assets if t in qnames]
        risk="normal"
        if dd<=float(config.get("drawdown_threshold",-0.12)):
            assets=[t for t in config.get("defensive_assets",["TLT","IEF","SHY"]) if t in prices.columns]; sg="DEFENSIVE"; risk=f"drawdown_control_{dd:.2%}"
        scores=residual_momentum_scores(ret,assets,d,int(config.get("residual_momentum_lookback_days",63)))
        chosen=sorted(scores,key=lambda t:scores[t],reverse=True)[:int(config.get("top_k_assets",2))]
        if not chosen:
            fb=[t for t in groups["DEFAULT"] if t in prices.columns]; scores=residual_momentum_scores(ret,fb,d,int(config.get("residual_momentum_lookback_days",63)))
            chosen=(sorted(scores,key=lambda t:scores[t],reverse=True) or [t for t in ["QQQ","SPY"] if t in prices.columns])[:int(config.get("top_k_assets",2))]; risk+="|fallback_default"
        cur={t:0 for t in all_t}
        for t in chosen: cur[t]=1/len(chosen)
        rows.append({"date":d,**cur})
        dec.append({"rebalance_date":d,"regime_state":regs.loc[d],"selected_group":sg,"group_ir_growth_tech":ir.get("GROWTH_TECH"),"group_ir_real_assets":ir.get("REAL_ASSETS"),"group_ir_defensive":ir.get("DEFENSIVE"),"group_ir_default":ir.get("DEFAULT"),"original_candidate_assets":",".join(groups.get(sg,[])),"qqq_rolling_candidates":",".join(qnames),"qqq_rolling_scores":fmt(qscores),"selected_assets":",".join(chosen),"residual_momentum_scores":fmt(scores),"risk_state":risk})
    return pd.DataFrame(rows).set_index("date").sort_index(),pd.DataFrame(dec)
def fmt(scores): return ";".join(f"{k}:{v:.6f}" for k,v in sorted(scores.items()))

