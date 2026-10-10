"""Common-window portfolio comparison with explicit cash and financing of fees."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from audit_and_train import membership,save_json
from market_data import prices as load_prices, canonical_predictions, aliases


def metrics(r):
    nav=(1+r).cumprod();days=(r.index[-1]-r.index[0]).days
    sd=r.std(ddof=1)
    return dict(cumulative=float(nav.iloc[-1]-1),cagr=float(nav.iloc[-1]**(365.25/days)-1),
        volatility=float(sd*np.sqrt(252)),sharpe_rf0=float(r.mean()/sd*np.sqrt(252)) if sd else 0.,
        max_drawdown=float((nav/nav.cummax().clip(lower=1)-1).min()))


def finance_trade(drift,target,missing,fee):
    """Dollar holdings in units of pre-trade NAV, with costs funded by sales.

Frozen unquoted positions retain their dollar value, not just their weight.
The residual is cash if no quoted targets remain.
"""
    frozen=np.where(missing,drift,0.)
    tradable=np.where(missing,0.,target)
    total=tradable.sum()
    if total>0: tradable=tradable/total
    def plan(cost):
        return frozen+tradable*max(0.,1-cost-frozen.sum())
    lo=0.;hi=2*fee
    for _ in range(45):
        mid=(lo+hi)/2
        needed=fee*np.abs(plan(mid)-drift).sum()
        if needed>mid: lo=mid
        else: hi=mid
    cost=(lo+hi)/2;dollars=plan(cost);turnover=np.abs(dollars-drift).sum()
    assert abs(cost-fee*turnover)<1e-10
    weights=dollars/(1-cost)
    assert weights.sum()<=1+1e-10 and weights.min()>=-1e-12
    return weights,cost,turnover


def simulate(pred,px,at,dates,frac=.25,sector_equal=False,delay=0,fee=.001,score='best',name='strategy',attribution=None):
    columns=list(px.columns);positions={t:i for i,t in enumerate(columns)}
    quote=px.reindex(dates).to_numpy()
    ret=px.pct_change(fill_method=None).reindex(dates).fillna(0.).to_numpy()
    w=np.zeros(len(columns));records=[];selected=[];missing_rows=[];decisions={};nav_before=1.0
    for date,rows in pred.groupby('decision_date'):
        index=dates.searchsorted(date)+delay
        if 0<=index<len(dates) and date>=dates[0]: decisions[index]=rows
    for i,date in enumerate(dates):
        gross=float(w@ret[i]);drift=w*(1+ret[i])/(1+gross)
        if attribution is not None:
            for j in np.flatnonzero(w>1e-12):attribution.append(dict(date=date,ticker=columns[j],start_weight=float(w[j]),stock_return=float(ret[i,j]),gross_pnl_initial_nav=float(nav_before*w[j]*ret[i,j])))
        missing=(w>1e-12)&np.isnan(quote[i])
        if missing.any():
            raise ValueError('Held security lacks price; terminal/corporate-action handling required: '+str(date)+' '+','.join(columns[j] for j in np.flatnonzero(missing)))
        cost=0.;turnover=0.
        if i in decisions:
            rows=decisions[i]
            members=at(date)
            valid=[t in positions and np.isfinite(quote[i,positions[t]]) and t in members for t in rows.ticker]
            rows=rows.loc[valid].sort_values([score,'ticker'],ascending=[False,True])
            target=np.zeros(len(columns))
            if sector_equal:
                groups=list(rows.groupby('gsector'))
                for _,group in groups:
                    chosen=group.head(max(1,int(np.ceil(len(group)*frac))))
                    for t in chosen.ticker: target[positions[t]]=1/len(groups)/len(chosen)
            elif len(rows):
                chosen=rows.head(max(1,int(np.ceil(len(rows)*frac))))
                for t in chosen.ticker: target[positions[t]]=1/len(chosen)
            w,cost,turnover=finance_trade(drift,target,missing,fee)
            for j in np.flatnonzero(w>1e-12):
                selected.append(dict(strategy=name,date=date,ticker=columns[j],weight=float(w[j])))
        else: w=drift
        fee_pnl=nav_before*(1+gross)*cost
        nav_before*=((1+gross)*(1-cost))
        records.append(dict(date=date,fee_pnl_initial_nav=fee_pnl,gross_ret=gross,net_ret=(1+gross)*(1-cost)-1,turnover=turnover,cost=cost,
                            n_holdings=int((w>1e-12).sum()),cash_weight=float(max(0,1-w.sum())),missing_held=int(missing.sum())))
    daily=pd.DataFrame(records).set_index('date');daily['nav']=(1+daily.net_ret).cumprod()
    return daily,pd.DataFrame(selected),pd.DataFrame(missing_rows)


def verify_engine():
    # A 10% price move on the entry day must not be earned before entering.
    dates=pd.date_range('2020-01-01',periods=3)
    px=pd.DataFrame({'A':[100.,110.,121.],'B':[100.,100.,100.]},index=dates)
    pred=pd.DataFrame(dict(ticker=['A'],decision_date=[dates[1]],gsector=['test'],best=[1.]))
    d,_,_=simulate(pred,px,lambda d:{'A','B'},dates,fee=0)
    assert np.allclose(d.net_ret,[0,0,.1])
    w,cost,turnover=finance_trade(np.array([0.,0.]),np.array([1.,0.]),np.array([False,False]),.01)
    assert abs(cost-.01/1.01)<1e-10 and abs(w[0]-1)<1e-10
    w,cost,turnover=finance_trade(np.array([1.,0.]),np.array([0.,1.]),np.array([False,False]),.01)
    assert abs(cost-2*.01/1.01)<1e-10
    w,cost,turnover=finance_trade(np.array([.4,.6]),np.array([0.,1.]),np.array([True,False]),.01)
    assert abs(w[0]*(1-cost)-.4)<1e-10
    bad=px.copy();bad.loc[dates[2],'A']=np.nan
    try:simulate(pred,bad,lambda d:{'A','B'},dates,fee=0)
    except ValueError:pass
    else:raise AssertionError('missing held price must fail')
    return ['entry_day_return_excluded','entry_fee_financed','two_sided_switch_cost','missing_held_price_fails']


def baseline_predictions(cache,baseline,out):
    f=pd.read_csv(cache/'prepared/final_ratios.csv',parse_dates=['date','reportdate'])
    ambiguous=f.duplicated(['tic','reportdate'],keep=False)
    f.loc[ambiguous,['tic','date','reportdate']].to_csv(out/'baseline_ambiguous_mappings.csv',index=False)
    f=f.loc[~ambiguous]
    frames=[]
    for path in baseline.glob('sector*/df_predict_best.csv'):
        x=pd.read_csv(path,index_col=0).rename_axis('reportdate').stack().rename('best').reset_index().rename(columns={'level_1':'tic'})
        x['reportdate']=pd.to_datetime(x.reportdate)
        frames.append(x)
    b=pd.concat(frames,ignore_index=True).merge(f[['tic','date','reportdate','gsector_name']],on=['tic','reportdate'],how='inner',validate='many_to_one')
    b['decision_date']=(b.date.dt.to_period('Q')+1).dt.start_time+pd.DateOffset(months=2)
    b['available']=b.reportdate+pd.offsets.BDay(1)
    b=b.loc[b.available<=b.decision_date].rename(columns={'tic':'ticker','gsector_name':'gsector'})
    assert not b.duplicated(['ticker','decision_date']).any()
    b.to_csv(out/'baseline_signals_aligned.csv',index=False)
    return b


def summarize(name,daily,rows):
    for sample,part in [('full',daily),('pre_2024',daily.loc[:'2023-12-31']),('2024_onward_descriptive',daily.loc['2024-01-01':])]:
        if len(part)>20:
            rows.append(dict(strategy=name,sample=sample,**metrics(part.net_ret),annual_turnover=float(part.turnover.mean()*252),
                missing_held=int(part.missing_held.sum()),start=str(part.index[0].date()),end=str(part.index[-1].date())))


def bootstrap_excess(r,benchmark):
    # Descriptive moving-block bootstrap, not a guarantee or a model-selection test.
    values=(r-benchmark).to_numpy();n=len(values);rng=np.random.default_rng(42);draws=[]
    for _ in range(2000):
        starts=rng.integers(0,n,size=int(np.ceil(n/63)))
        idx=np.concatenate([(np.arange(s,s+63)%n) for s in starts])[:n]
        draws.append(values[idx].mean()*252)
    return dict(annualized_arithmetic_excess=float(values.mean()*252),ci95_low=float(np.quantile(draws,.025)),ci95_high=float(np.quantile(draws,.975)),block_days=63,replicates=2000)


def main(a):
    out=a.output;out.mkdir(parents=True,exist_ok=True)
    checks=verify_engine()
    p=load_prices(a.cache,canonical=True)
    px=p.pivot(index='datadate',columns='tic',values='adj_close').sort_index()
    at=membership(a.cache)
    original=canonical_predictions(baseline_predictions(a.cache,a.baseline,out))
    prior=canonical_predictions(pd.read_csv(a.prior/'predictions.csv',parse_dates=['decision_date','available']))
    audited=canonical_predictions(pd.read_csv(a.audited/'predictions.csv',parse_dates=['decision_date','available']))
    start=audited.decision_date.min() if a.window=='full' else pd.Timestamp('2020-09-01')
    dates=px.index[(px.index>=start)&(px.index<='2025-09-30')]
    # Frozen common price window; the first overnight experiment began earlier.
    variants={
        'original_signals_top25':(original,.25,False),
        'overnight_signals_top25':(prior,.25,False),
        'overnight_signals_top10':(prior,.10,False),
        'audited_top25':(audited,.25,False),
        'audited_top10':(audited,.10,False),
        'audited_sector_top25':(audited,.25,True),
        'audited_all_candidates_equal':(audited,1.,False)}
    if a.window=='full':
        variants={k:v for k,v in variants.items() if k.startswith('audited_')}
    # A separate no-model benchmark: all contemporaneously quoted index members.
    universe=[]
    for date in sorted(audited.decision_date.unique()):
        date=pd.Timestamp(date)
        for t in sorted(at(date)&set(px.columns)):
            if date in px.index and pd.notna(px.at[date,t]): universe.append(dict(ticker=t,decision_date=date,best=0.,gsector='all'))
    variants['quoted_index_members_equal']=(pd.DataFrame(universe),1.,False)
    results={};rows=[];holdings=[];missing=[];attribution=[]
    for name,(pred,frac,sector) in variants.items():
        daily,selected,m=simulate(pred,px,at,dates,frac=frac,sector_equal=sector,name=name,attribution=attribution if name=='audited_top10' else None)
        daily.to_csv(out/(name+'_daily.csv'));results[name]=daily
        holdings.append(selected);missing.append(m);summarize(name,daily,rows)
    for ticker in ['SPY','QQQ']:
        series=pd.read_csv(a.prior/(ticker+'_adjusted_close.csv'),index_col=0,parse_dates=True).iloc[:,0]
        assert series.reindex(dates).notna().all()
        r=series.pct_change(fill_method=None).reindex(dates);r.iloc[0]=0
        daily=pd.DataFrame(dict(net_ret=r,turnover=0.,missing_held=0));daily['nav']=(1+r).cumprod()
        results[ticker]=daily;summarize(ticker,daily,rows)
    # Newly accepted recovery is applied consistently to every strategy above.
    pd.DataFrame(rows).to_csv(out/'comparison_metrics.csv',index=False)
    pd.concat(holdings,ignore_index=True).to_csv(out/'rebalance_holdings.csv',index=False)
    nonempty=[m for m in missing if len(m)]
    pd.concat(nonempty,ignore_index=True).to_csv(out/'missing_held_prices.csv',index=False) if nonempty else pd.DataFrame(columns=['strategy','date','ticker','weight']).to_csv(out/'missing_held_prices.csv',index=False)
    nav=pd.DataFrame({k:v.nav for k,v in results.items()});nav.to_csv(out/'nav_comparison.csv')
    annual=[]
    for name,d in results.items():
        for year,part in d.groupby(d.index.year):
            annual.append(dict(strategy=name,year=year,return_value=float((1+part.net_ret).prod()-1),trading_days=len(part),partial_year=year in [2020,2025]))
    pd.DataFrame(annual).to_csv(out/'annual_returns.csv',index=False)
    sensitivity=[]
    for name,frac in [('audited_top25',.25),('audited_top10',.10)]:
        for delay in [0,1,5]:
            for fee in [.0005,.001,.002,.005]:
                daily,_,_=simulate(audited,px,at,dates,frac=frac,delay=delay,fee=fee,name=name)
                sensitivity.append(dict(strategy=name,delay_sessions=delay,fee_bps=fee*10000,**metrics(daily.net_ret),missing_held=int(daily.missing_held.sum())))
    pd.DataFrame(sensitivity).to_csv(out/'cost_delay_sensitivity.csv',index=False)
    excess=[]
    for name in ['audited_top25','audited_top10','audited_sector_top25']:
        excess.append(dict(strategy=name,**bootstrap_excess(results[name].net_ret,results['SPY'].net_ret)))
    pd.DataFrame(excess).to_csv(out/'excess_return_bootstrap.csv',index=False)
    save_json(out/'verification.json',dict(engine_checks=checks,common_start=dates[0],common_end=dates[-1],trading_dates=len(dates),
        fee_bps=10,annualization_days=252,risk_free_rate=0,benchmark_explicit_trading_fee=0,
        missing_held_rows=sum(len(m) for m in missing),tested_sensitivity_scenarios=len(sensitivity),
        window=a.window,first_available_prediction=audited.decision_date.min(),prior_2024_results_already_seen=True,strict_untouched_holdout=False,
        causal_attribution_of_individual_training_changes=False))
    detail=pd.DataFrame(attribution)
    sectors=audited.sort_values('decision_date').drop_duplicates('ticker',keep='last').set_index('ticker').gsector
    detail['sector']=detail.ticker.map(sectors).fillna('unknown')
    detail.to_csv(out/'top10_daily_contributions.csv',index=False)
    contributions=detail.groupby(['ticker','sector']).agg(gross_pnl_initial_nav=('gross_pnl_initial_nav','sum'),held_days=('date','count'),mean_weight_when_held=('start_weight','mean')).reset_index().sort_values('gross_pnl_initial_nav',ascending=False)
    contributions.to_csv(out/'top10_stock_contributions.csv',index=False)
    detail.groupby('sector').gross_pnl_initial_nav.sum().sort_values(ascending=False).to_csv(out/'top10_sector_contributions.csv')
    fees=results['audited_top10'].fee_pnl_initial_nav.sum()
    assert abs(detail.gross_pnl_initial_nav.sum()-fees-(results['audited_top10'].nav.iloc[-1]-1))<1e-9
    save_json(out/'contribution_reconciliation.json',dict(gross_pnl=float(detail.gross_pnl_initial_nav.sum()),fee_pnl=float(fees),net_pnl=float(results['audited_top10'].nav.iloc[-1]-1),attribution='dollar PnL divided by initial NAV; arithmetic sum reconciles net change; sector classification remains retrospective'))
    # Disclosed-data buffer sensitivity, using frozen forecasts (not a retrain).
    strict=audited.loc[audited.available+pd.offsets.BDay(5)<=audited.decision_date]
    d,_,_=simulate(strict,px,at,dates,frac=.10,name='top10_five_day_disclosure_buffer')
    d.to_csv(out/'top10_five_day_disclosure_buffer_daily.csv')
    save_json(out/'disclosure_buffer_sensitivity.json',dict(**metrics(d.net_ret),retained_predictions=len(strict),original_predictions=len(audited),method='five additional business days before decision; frozen forecasts, no new model fitting'))
    plot_results(out,results)
    print(pd.DataFrame(rows).query("sample=='full'")[['strategy','cumulative','cagr','max_drawdown']].to_string(index=False),flush=True)


def plot_results(out,results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True,gridspec_kw={'height_ratios':[2,1]})
    labels={'audited_top25':'Audited top 25%','audited_top10':'Audited top 10%','audited_sector_top25':'Audited sector balanced','SPY':'SPY','QQQ':'QQQ'}
    for name,label in labels.items():
        nav=results[name].nav
        axes[0].plot(nav.index,nav,label=label,lw=1.5)
        axes[1].plot(nav.index,100*(nav/nav.cummax().clip(lower=1)-1),lw=1)
    axes[0].set_ylabel('Portfolio value (initial = 1)');axes[1].set_ylabel('Drawdown (%)')
    axes[0].set_title('Fixed-candidate comparison | coverage-limited universe')
    axes[0].legend(ncol=2,frameon=False);axes[0].grid(alpha=.15);axes[1].grid(alpha=.15)
    fig.tight_layout();fig.savefig(out/'nav_drawdown.png',dpi=170);plt.close(fig)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--window',choices=['full','common'],default='full')
    for name in ['cache','baseline','prior','audited','output']: ap.add_argument('--'+name,required=True,type=Path)
    main(ap.parse_args())
