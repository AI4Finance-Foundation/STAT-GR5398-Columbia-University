#!/usr/bin/env python3
"""DEV-only timing component and fixed-parameter sensitivity experiments.
Uses official evaluator and portfolio runner; never accepts VAL or TEST dates.
Parameters are restored after each variant; no source files are rewritten.
"""
from pathlib import Path
import argparse
from datetime import date
import pandas as pd
from timing import student_timing as m
from shared.data_view import DataBundle
from shared.metrics import performance, crisis_table
from evaluate_timing import exposure_path, simulate
from run_portfolio import run_one
from selection.config import load
from selection.scores import MonthlyScores

VARIANTS = {
    'rule': {'ENABLE_RL':False, 'ENABLE_BULL_FLOOR':False},
    'rolling_gru_policy': {'USE_FROZEN_DIRECT_Q':False,'ENABLE_BULL_FLOOR':False},
    'direct_q': {'USE_GRU_UPSIDE_SLEEVE':False,'ENABLE_BULL_FLOOR':False},
    'hybrid': {'ENABLE_BULL_FLOOR':False},
    'hybrid_bull_80': {'ENABLE_BULL_FLOOR':True,'BULL_EXPOSURE_FLOOR':.80},
    'hybrid_bull_90': {'ENABLE_BULL_FLOOR':True,'BULL_EXPOSURE_FLOOR':.90},
    'hybrid_bull_100': {'ENABLE_BULL_FLOOR':True,'BULL_EXPOSURE_FLOOR':1.0},
    'hybrid_vol_25': {'TARGET_ANNUAL_VOL':.25,'ENABLE_BULL_FLOOR':True,'BULL_EXPOSURE_FLOOR':.80},
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--portfolio', action='store_true',help='Also run official selection portfolio for each variant')
    p.add_argument('--variants',nargs='+',choices=VARIANTS,default=list(VARIANTS))
    a=p.parse_args()
    start,end=date(2017,1,3),date(2021,12,31)
    _,options,params,prices,_,universe=load('selection/configs/growth.json')
    bundle=DataBundle.load(prices=prices)
    ctx=dict(options=options,params=params,prices=prices,universe=universe,scores=MonthlyScores(),bundle=bundle)
    sessions=prices.sessions[(prices.sessions>=pd.Timestamp(start))&(prices.sessions<=pd.Timestamp(end))]
    spy=prices.close['SPY'].loc[:sessions[-1]]
    bh=(spy/spy.loc[sessions[0]]).loc[sessions]
    keys=set().union(*(v.keys() for v in VARIANTS.values()))|{'ENABLE_FINGPT'}
    original={k:getattr(m,k) for k in keys}
    rows=[]; crises=[]; portfolios=[]
    out=Path('output');out.mkdir(exist_ok=True)
    try:
        for name in a.variants:
            for k,v in original.items():setattr(m,k,v)
            for k,v in VARIANTS[name].items():setattr(m,k,v)
            m.ENABLE_FINGPT=False
            expo=exposure_path(m.exposure,bundle,sessions)
            eq=simulate(expo,spy)
            row=dict(variant=name,**performance(eq),avg_exposure=round(expo.mean(),3),time_in_market_pct=round(100*(expo>.5).mean(),1),exposure_changes=int((expo.diff().abs()>1e-9).sum()))
            rows.append(row);print(row,flush=True)
            crises.append(crisis_table(eq,bh).assign(variant=name))
            if a.portfolio:
                stats=run_one('dev_internal_'+name,start,end,valuation=False,timing='student',stops=False,ctx=ctx)
                portfolios.append(dict(variant=name,**stats));print(stats,flush=True)
            pd.DataFrame(rows).to_csv(out/'timing_internal_dev.csv',index=False)
            pd.concat(crises).to_csv(out/'timing_internal_crises_dev.csv',index=False)
            if portfolios:pd.DataFrame(portfolios).to_csv(out/'timing_internal_portfolio_dev.csv',index=False)
    finally:
        for k,v in original.items():setattr(m,k,v)

if __name__=='__main__':main()
