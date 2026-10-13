"""Quarterly index sleeve backtest on CRSP CIZ total returns.

Ranking uses the prior-session quote; executions require an actual current quote. Securities exiting coverage
are converted into cash on the dated event recognition, with a conservative
liquidation fee on proceeds. The supplied CIZ exit return is used only once.
"""
from pathlib import Path
import sys,json,os
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DATA=Path(os.environ.get('FINRL_DATA_DIR',str(ROOT/'data-source')))
RUN=Path(os.environ.get('FINRL_RUN_DIR',str(ROOT/'run')));OUT=RUN/'backtest'
sys.path.insert(0,str(Path(__file__).resolve().parent/'support'))
from portfolio_math import metrics

def rebalance(w,target,fee,cash,frozen):
    stock=np.ones(len(w),bool);stock[cash]=False
    fixed=np.where(frozen,w,0.)
    active=np.where(frozen,0.,target)
    if active.sum()>0: active/=active.sum()
    else: active[cash]=1.
    def plan(cost): return fixed+active*max(0.,1-cost-fixed.sum())
    lo=0.;hi=2*fee
    for _ in range(50):
        cost=(lo+hi)/2
        needed=fee*np.abs(plan(cost)[stock]-w[stock]).sum()
        if needed>cost:lo=cost
        else:hi=cost
    cost=(lo+hi)/2;dollars=plan(cost)
    turnover=np.abs(dollars[stock]-w[stock]).sum()
    assert abs(cost-fee*turnover)<1e-10
    assert np.allclose(dollars[frozen],w[frozen],atol=1e-12)
    return dollars/(1-cost),cost,float(turnover)

def main():
    OUT.mkdir(exist_ok=True)
    flags={'experiment_specs':json.loads((ROOT/'strategy_specs.json').read_text()),'remove_rights':os.environ.get('FINRL_REMOVE_RIGHTS')=='1','unquoted_rights_zero_sensitivity':os.environ.get('FINRL_UNQUOTED_RIGHTS_ZERO')=='1'}
    if (flags['remove_rights'] or flags['unquoted_rights_zero_sensitivity']) and RUN.resolve()==(ROOT/'run').resolve():raise ValueError('Sensitivity requires a distinct FINRL_RUN_DIR')
    prior=OUT/'backtest_protocol.json'
    if prior.exists() and json.loads(prior.read_text())!=flags:raise ValueError('Do not overwrite outputs from a different protocol')
    prior.write_text(json.dumps(flags,indent=2))
    ret=pd.read_parquet(DATA/'returns.parquet');ret.columns=ret.columns.astype(int)
    entry=pd.read_parquet(DATA/'entry_flags.parquet');entry.columns=entry.columns.astype(int)
    full_entry=entry.copy()
    days=ret.index[(ret.index>='2020-05-01')&(ret.index<='2025-09-30')]
    ret=ret.loc[days].copy();entry=entry.loc[days]
    events=pd.read_csv(DATA/'event_recognition.csv',parse_dates=['recognition_date','execution_date','crsp_return_date'])
    pred=pd.read_csv(RUN/'predictions.csv',parse_dates=['decision_date','available'])
    rel=pd.read_csv(Path(os.environ['FINRL_REFERENCE_RUN'])/'predictions.csv',parse_dates=['decision_date','available'])
    rel=rel[['security_id','decision_date','rf','best']].rename(columns={'rf':'relative_rf','best':'relative_best'})
    pred=pred.merge(rel,on=['security_id','decision_date'],validate='one_to_one')
    assert pred[['relative_rf','relative_best']].notna().all().all()
    seed_predictions=pd.read_csv(RUN/'rf_seed_predictions.csv.gz',parse_dates=['decision_date'],float_precision='round_trip')
    seed_predictions=seed_predictions.pivot(index=['ticker','decision_date'],columns='seed',values='score').rename(columns=lambda s:'seed_score_'+str(s)).reset_index()
    pred=pred.merge(seed_predictions,on=['ticker','decision_date'],validate='one_to_one')
    for seed in [42,0,1,2,3,4,5,6,7,8,9]:
        pred['best_seed_'+str(seed)]=np.where(pred.winner.eq('rf'),pred['seed_score_'+str(seed)],pred.best)
    pred['permno']=pred.security_id.str.removeprefix('CRSP_').astype(int)
    cols=ret.columns.to_list();cash=len(cols);idx={p:i for i,p in enumerate(cols)}
    rr=np.column_stack([ret.to_numpy(),np.zeros(len(ret))])
    rights={} if os.environ.get('FINRL_REMOVE_RIGHTS')=='1' else {75107:pd.Timestamp('2022-11-04'),19502:pd.Timestamp('2025-03-11')}
    specs=json.loads((ROOT/'strategy_specs.json').read_text())
    navs={};summary=[];held_events=[];counts=[];gaps=[];frozen_rows=[]
    exit_map={}
    for e in events.itertuples(index=False):
        pos=days.searchsorted(e.execution_date) if pd.notna(e.execution_date) else len(days)
        if pos<len(days) and days[pos]==e.execution_date and e.permno in idx:
            exit_map.setdefault(pos,[]).append(e)
    for name,score,core,fee,delay,frac,core_perm in specs:
        schedule={};targets=[]
        if core: 
            v=np.zeros(len(cols)+1);v[idx[core_perm]]=1.;schedule[0]=v
        for day,g in pred.groupby('decision_date'):
            known=g.permno.map(lambda p:p in rights and rights[p]<day)
            prior_day=full_entry.index[full_entry.index<day][-1]
            ok=g.permno.map(lambda p:p in full_entry.columns and bool(full_entry.at[prior_day,p]))
            candidates=g[ok & ~known & g.available.lt(day)].sort_values([score,'ticker','permno'],ascending=[False,True,True])
            n=max(1,int(np.ceil(len(candidates)*frac)))
            chosen=candidates.head(n)
            target=np.zeros(len(cols)+1);target[idx[core_perm]]=core
            if len(chosen):
                for p in chosen.permno:target[idx[p]]+=(1-core)/len(chosen)
            else:target[cash]=1-core
            pos=days.searchsorted(day)+delay
            if pos<len(days) and day>=days[0]:
                # Keep the decision's ranked names; an unavailable quote on the
                # actual execution day leaves that allocation as cash.
                for j in np.flatnonzero(target[:cash]>0):
                    pending=events[events.permno.eq(cols[j]) & events.crsp_return_date.le(days[pos]) & (events.execution_date.gt(days[pos]) | events.execution_date.isna())]
                    if not bool(entry.at[days[pos],cols[j]]) or len(pending):
                        target[cash]+=target[j];target[j]=0.
                schedule[pos]=target
            counts.append(dict(strategy=name,date=day,official_members=len(g),entry_eligible=int(ok.sum()),
                known_rights_restricted=int((ok & known).sum()),selected=len(chosen)))
            for p,t in zip(chosen.permno,chosen.ticker):targets.append(dict(decision_date=day,permno=p,ticker=t))
        w=np.zeros(len(cols)+1);w[cash]=1.;rows=[];holdings=[]
        for i,day in enumerate(days):
            returns=rr[i].copy()
            if i==0:returns[:]=0.
            held=w>1e-12
            if np.isnan(returns[held]).any():
                bad=[cols[j] for j in np.flatnonzero(held & np.isnan(returns))]
                gaps.append(dict(strategy=name,date=str(day.date()),permnos=bad))
                (OUT/'fatal_held_return_gaps.json').write_text(json.dumps(gaps,indent=2))
                raise RuntimeError('Unresolved held CRSP return: '+str(gaps[-1]))
            gross=float(w @ np.nan_to_num(returns));drift=w*(1+np.nan_to_num(returns))/(1+gross)
            exit_cost=0.;exit_turnover=0.
            for ev in exit_map.get(i,[]):
                j=idx[ev.permno]
                if drift[j]>1e-12:
                    if bool(ev.unknown_complete_proceeds) and os.environ.get('FINRL_UNQUOTED_RIGHTS_ZERO')!='1':
                        raise RuntimeError('Held unvalued proceeds '+name+' '+str(ev.permno))
                    proceeds=drift[j];fee_amount=fee*proceeds
                    drift[j]=0.;drift[cash]+=proceeds-fee_amount
                    exit_cost+=fee_amount;exit_turnover+=proceeds
                    held_events.append(dict(strategy=name,date=day,permno=ev.permno,proceeds_weight=proceeds,
                        liquidation_fee=fee_amount,recognition_date=ev.recognition_date,
                        unquoted_rights_zero_sensitivity=bool(ev.unknown_complete_proceeds)))
            if exit_cost:drift/=1-exit_cost
            trade_cost=0.;turnover=exit_turnover
            if i in schedule:
                frozen=np.r_[~entry.loc[day,cols].to_numpy(dtype=bool),False]
                for ev in events.itertuples(index=False):
                    if ev.permno in idx and ev.crsp_return_date<=day and (pd.isna(ev.execution_date) or day<ev.execution_date):
                        frozen[idx[ev.permno]]=True
                frozen_rows.append(dict(strategy=name,date=day,frozen_held_names=int((frozen & (drift>1e-12)).sum()),frozen_dollar_share=float(drift[frozen].sum())))
                w,trade_cost,t=rebalance(drift,schedule[i],fee,cash,frozen);turnover+=t
                for j in np.flatnonzero(w>1e-12):
                    holdings.append(dict(date=day,permno='CASH' if j==cash else cols[j],weight=float(w[j])))
            else:w=drift
            assert np.isclose(w.sum(),1,atol=1e-10)
            pending=sum(w[idx[ev.permno]] for ev in events.itertuples(index=False) if ev.permno in idx and ev.crsp_return_date<=day and (pd.isna(ev.execution_date) or day<ev.execution_date))
            if i==len(days)-1 and pending>1e-12: raise RuntimeError('Unresolved pending terminal holding at sample end')
            net=(1+gross)*(1-exit_cost)*(1-trade_cost)-1
            rows.append(dict(date=day,gross_ret=gross,net_ret=net,turnover=turnover,
                cost=1-(1-exit_cost)*(1-trade_cost),cash_weight=w[cash],spy_weight=w[idx[84398]],qqq_weight=w[idx[86755]],pending_exit_weight=pending))
        print('Completed '+name,flush=True)
        z=pd.DataFrame(rows).set_index('date');z['nav']=(1+z.net_ret).cumprod()
        z.to_csv(OUT/(name+'_daily.csv'));pd.DataFrame(holdings).to_csv(OUT/(name+'_holdings.csv'),index=False)
        pd.DataFrame(targets).to_csv(OUT/(name+'_selected.csv'),index=False)
        navs[name]=z.nav;summary.append(dict(strategy=name,**metrics(z.net_ret),annual_turnover=float(z.turnover.mean()*252)))
    for name,perm in [('SPY',84398),('QQQ',86755)]:
        r=ret[perm].copy();r.iloc[0]=0.;assert r.notna().all()
        navs[name]=(1+r).cumprod();summary.append(dict(strategy=name,**metrics(r),annual_turnover=0.))
    pd.DataFrame(summary).to_csv(OUT/'comparison.csv',index=False)
    pd.DataFrame(navs).to_csv(OUT/'nav.csv')
    pd.DataFrame(held_events).to_csv(OUT/'held_exit_events.csv',index=False)
    pd.DataFrame(counts).to_csv(OUT/'selection_counts.csv',index=False)
    pd.DataFrame(frozen_rows).to_csv(OUT/'frozen_rebalance_checks.csv',index=False)
    print(pd.DataFrame(summary).to_string(index=False),flush=True)

if __name__=='__main__':main()
