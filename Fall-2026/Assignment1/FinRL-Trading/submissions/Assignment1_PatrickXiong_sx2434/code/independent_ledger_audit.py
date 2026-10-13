"""Independent raw-source returns and a dollar-position ledger, no backtest imports."""
from pathlib import Path
import os,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
INP=Path(os.environ.get('FINRL_INPUT_DIR',str(ROOT/'inputs')))
RUN=Path(os.environ.get('FINRL_RUN_DIR',str(ROOT/'run')))
OUT=RUN/'backtest'
DATA=Path(os.environ.get('FINRL_DATA_DIR',str(ROOT/'data-source')))

def fee_dollars(values,target,nav,rate):
    # Piecewise linear closed-form financing, independent of the production
    # weight-based bisection. Cash is excluded from trading turnover.
    sign=np.sign(target*nav-values)
    for _ in range(20):
        fee=rate*(nav*(sign*target).sum()-(sign*values).sum())/(1+rate*(sign*target).sum())
        next_sign=np.sign(target*(nav-fee)-values)
        if np.array_equal(sign,next_sign):break
        sign=next_sign
    assert np.isclose(fee,rate*np.abs(target*(nav-fee)-values).sum(),rtol=1e-10,atol=1e-12)
    return max(0.,float(fee))

def main():
    d=pd.read_parquet(INP/'crsp_daily_raw.parquet').drop_duplicates(['permno','dlycaldt'])
    quoted=d.set_index(['dlycaldt','permno'])[['dlyclose','tradingstatusflg']]
    source_dates={}
    e=pd.read_csv(INP/'crsp_delisting.csv');e.columns=e.columns.str.lower()
    for c in ['delistingdt','deldlydt','delnextdt','delamtdt']:e[c]=pd.to_datetime(e[c])
    events=e.set_index('permno')
    calendar=pd.DatetimeIndex(d[d.permno.eq(84398)].dlycaldt.sort_values().unique())
    times=calendar[(calendar>='2020-05-01')&(calendar<='2025-09-30')]
    reference=pd.read_csv(OUT/'nav.csv',index_col=0,parse_dates=True)
    checks=[];source_returns={};recognition={}
    for p,q in d.groupby('permno'):
        q=q.set_index('dlycaldt').sort_index();r=q.dlyret.reindex(calendar).copy()
        r.loc[q.index[q.dlyretmissflg.eq('NT')]]=0.
        if q.iloc[0].dlyretmissflg=='NS':r.at[q.index[0]]=0.
        for day in q.index[q.dlyretmissflg.eq('MP')]:
            before=q.loc[q.index<day,'dlyprc'].dropna();later=q[(q.index>day)&q.dlyret.notna()]
            if len(before) and len(later) and np.isclose(later.iloc[0].dlyprevprc,before.iloc[-1],rtol=1e-6,atol=1e-6) and np.isclose(later.iloc[0].dlyfacprc,1):r.at[day]=0.
        if p in events.index:
            ev=events.loc[p]
            seen=max(t for t in [ev.deldlydt,ev.delnextdt,ev.delamtdt] if pd.notna(t))
            k=calendar.searchsorted(seen)
            actual=calendar[k] if k<len(calendar) else None
            value=q.dlyret.get(ev.deldlydt,np.nan)
            if pd.isna(value) and ev.delpaymenttype=='PRCF' and (pd.isna(ev.deldivamt) or ev.deldivamt==0) and pd.notna(ev.delnextprc) and ev.deldtprc>0:value=ev.delnextprc/ev.deldtprc-1
            r.loc[(calendar>=ev.deldlydt)&(calendar<seen)]=0.
            if actual is not None:
                r.at[actual]=value;r.loc[calendar>actual]=0.;recognition[p]=actual
        source_returns[p]=r
        if p in events.index:source_dates[p]=events.at[p,'deldlydt']
    full_rs=pd.DataFrame(source_returns,index=calendar)
    prepared=pd.read_parquet(DATA/'returns.parquet')
    prepared.columns=prepared.columns.astype(int);full_rs=full_rs.loc[prepared.index,prepared.columns]
    assert np.array_equal(np.isnan(full_rs.to_numpy()),np.isnan(prepared.to_numpy()))
    source_delta=float(np.nanmax(np.abs(full_rs.to_numpy()-prepared.to_numpy())))
    assert source_delta<1e-12,source_delta
    labels=pd.read_csv(DATA/'model_panel.csv.gz',parse_dates=['decision_date','label_end'],float_precision='round_trip')
    label_checks=[]
    for case in labels[labels.target.notna() & labels.label_status.eq('observed_CRSP_total_return')].itertuples(index=False):
        a=full_rs.loc[(full_rs.index>case.decision_date)&(full_rs.index<=case.label_end),case.permno]
        assert a.notna().all()
        raw_target=float(np.log(max(float(np.prod(1+a.to_numpy())),1e-8)))
        error=abs(raw_target-case.target);assert error<1e-12,(case.permno,str(case.decision_date),error)
        label_checks.append(error)
    rs=full_rs.loc[times]
    forecasts=pd.read_csv(RUN/'predictions.csv',parse_dates=['decision_date','available'])
    forecasts['permno']=forecasts.security_id.str.removeprefix('CRSP_').astype(int)
    rights={75107:pd.Timestamp('2022-11-04'),19502:pd.Timestamp('2025-03-11')}
    for name,core,rate,delay in [
        ('SPY80_Best20_quarterly',.8,.001,0),
        ('SPY80_Best20_quarterly_fee20bps_delay1',.8,.002,1),
        ('Best_top5_quarterly',0.,.001,0),
        ('SPY80_RF20_quarterly',.8,.001,0),
        ('RF_top5_quarterly',0.,.001,0)]:
        chosen=pd.read_csv(OUT/(name+'_selected.csv'),parse_dates=['decision_date'])
        score='rf' if 'RF' in name else 'best'
        for decision,g in forecasts.groupby('decision_date'):
            before=calendar[calendar<decision][-1];eligible=[]
            for p in g.permno:
                q=quoted.loc[(before,p)] if (before,p) in quoted.index else None
                eligible.append(q is not None and q.dlyclose>0 and q.tradingstatusflg=='A' and not (p in rights and rights[p]<decision))
            ranks=g[np.array(eligible) & g.available.lt(decision)].sort_values([score,'ticker','permno'],ascending=[False,True,True])
            expected=ranks.head(max(1,int(np.ceil(.05*len(ranks))))).permno.to_list()
            assert expected==chosen[chosen.decision_date.eq(decision)].permno.to_list(),(name,str(decision),'selection')
        schedules={}
        if core:schedules[times[0]]={84398:1.}
        for decision,g in chosen.groupby('decision_date'):
            k=times.searchsorted(decision)+delay
            if k>=len(times) or decision<times[0]:continue
            day=times[k];targets={84398:core}
            for p in g.permno:
                row=quoted.loc[(day,p)] if (day,p) in quoted.index else None
                live=row is not None and row.dlyclose>0 and row.tradingstatusflg=='A'
                pending=p in source_dates and source_dates[p]<=day and (p not in recognition or day<recognition[p])
                if live and not pending:targets[p]=targets.get(p,0)+(1-core)/len(g)
            schedules[day]=targets
        dollars={};cash=1.;nav=[]
        for i,day in enumerate(times):
            if i:
                for p in list(dollars):
                    if dollars[p]>1e-14:
                        ret=rs.at[day,p];assert np.isfinite(ret),(name,str(day),p)
                        dollars[p]*=1+ret
            for p in list(dollars):
                if recognition.get(p)==day:
                    cash+=dollars.pop(p)*(1-rate)
            total=cash+sum(dollars.values())
            if day in schedules:
                target=schedules[day];perms=sorted(set(dollars)|set(target))
                frozen={}
                for p in perms:
                    row=quoted.loc[(day,p)] if (day,p) in quoted.index else None
                    live=row is not None and row.dlyclose>0 and row.tradingstatusflg=='A'
                    pending=p in source_dates and source_dates[p]<=day and (p not in recognition or day<recognition[p])
                    if not live or pending:frozen[p]=dollars.get(p,0.)
                budget=total-sum(frozen.values())
                active=[p for p in perms if p not in frozen]
                # Re-normalize stocks together with the target cash fraction.
                original_cash=1-sum(target.values())
                scale=original_cash+sum(target.get(p,0) for p in active)
                fractions=np.array([target.get(p,0)/scale if scale>0 else 0 for p in active])
                values=np.array([dollars.get(p,0) for p in active])
                paid=fee_dollars(values,fractions,budget,rate);remaining=budget-paid
                dollars=dict(frozen)
                dollars.update({p:float(f*remaining) for p,f in zip(active,fractions) if f>0})
                cash=remaining-sum(f*remaining for f in fractions)
            nav.append(cash+sum(dollars.values()))
        delta=float(np.max(np.abs(np.array(nav)-reference[name].to_numpy())))
        assert delta<1e-10,(name,delta)
        checks.append(dict(strategy=name,max_daily_nav_difference=delta,days=len(times)))
    pd.DataFrame(checks).to_csv(RUN/'independent_ledger_checks.csv',index=False)
    result=dict(independent_raw_CRSP_return_reconstruction=True,independent_dollar_ledger=True,
        production_backtest_functions_imported=False,independent_selection_from_raw_prior_quotes=True,
        maximum_prepared_return_difference=source_delta,independent_observed_label_cases=len(label_checks),
        maximum_observed_label_difference=max(label_checks),strategies=len(checks),days=len(times),
        maximum_daily_nav_difference=max(x['max_daily_nav_difference'] for x in checks))
    (RUN/'independent_ledger_checks.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
