"""Construct a dated CRSP universe, price features and observed outcome labels.

All index members are retained for prediction. Missing outcomes only affect
supervised fitting. CRSP CIZ returns include delisting; do not add DLRET again.
Later OTC quotes are recognized on their observed date, not the earlier exit.
"""
from pathlib import Path
import sys, json, hashlib, os
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INP = Path(os.environ.get('FINRL_INPUT_DIR',str(ROOT/'inputs')))
DATA = Path(os.environ.get('FINRL_DATA_DIR',str(ROOT/'data-source')))
RUN = Path(os.environ.get('FINRL_RUN_DIR',str(ROOT/'run')))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_features import calculate, FEATURES

def main():
    DATA.mkdir(exist_ok=True); RUN.mkdir(exist_ok=True)
    (DATA/'inputs').mkdir(exist_ok=True)
    d = pd.read_parquet(INP/'crsp_daily_raw.parquet')
    relevant = ['dlyret','dlyretx','dlyprc','dlyclose','dlydelflg','tradingstatusflg',
                'dlyretmissflg','dlyprevprc','dlyfacprc']
    dup = d[d.duplicated(['permno','dlycaldt'],keep=False)]
    assert not dup.groupby(['permno','dlycaldt'])[relevant].nunique(dropna=False).gt(1).any().any()
    d = d.drop_duplicates(['permno','dlycaldt']).sort_values(['permno','dlycaldt'])
    e = pd.read_csv(INP/'crsp_delisting.csv',low_memory=False)
    e.columns = e.columns.str.lower()
    for col in ['delistingdt','deldlydt','delnextdt','delamtdt']:
        e[col] = pd.to_datetime(e[col])
    assert not e.duplicated('permno').any()
    events = e.set_index('permno')
    member = pd.read_csv(INP/'crsp_sp500_members.csv.gz',low_memory=False,
        usecols=lambda c:c.lower() in ['permno','dlycaldt','ticker','mbrstartdt','mbrenddt'])
    member.columns = member.columns.str.lower()
    member['dlycaldt'] = pd.to_datetime(member.dlycaldt)
    member = member.drop_duplicates(['permno','dlycaldt'])
    for c in ['mbrstartdt','mbrenddt']: member[c]=pd.to_datetime(member[c])
    spells=member[['permno','mbrstartdt','mbrenddt']].drop_duplicates()
    names=pd.read_csv(INP/'crsp_names_all.csv',
        parse_dates=['secinfostartdt','secinfoenddt'],low_memory=False)
    names_by={p:g.sort_values('secinfostartdt') for p,g in names.groupby('permno')}
    missing = sorted(set(member.permno) - set(d.permno))
    if missing:
        (INP/'additional_permnos.txt').write_text(' '.join(map(str,missing)))
        raise RuntimeError('Need additional historical-member PERMNOs: '+str(missing))
    spy = d[d.permno.eq(84398)].set_index('dlycaldt').sort_index()
    calendar = pd.DatetimeIndex(spy.index)
    schedule=pd.Series(calendar,index=calendar).loc[
        (calendar>='2015-09-01') & calendar.month.isin([3,6,9,12])
    ].groupby(lambda x:(x.year,x.month)).first().to_list()
    final_month=(schedule[-1]+pd.DateOffset(months=3)).replace(day=1)
    final_end=pd.offsets.BDay().rollforward(final_month)
    date_ends=pd.DataFrame({'decision_date':schedule,'label_end':schedule[1:]+[final_end]})
    data_by = {}; ret_by = {}; entry_by = {}; resolved_by = {}; unknown_by = {}; event_clocks={}
    event_rows = []; frames = []; stale_audit=[]
    for perm, q in d.groupby('permno',sort=True):
        q = q.set_index('dlycaldt').sort_index()
        r = q.dlyret.reindex(calendar).copy()
        # X/NT identifies explicit non-trading records: a stale, non-executable
        # mark until the next observed return. Other missing returns stay missing.
        nt = q.dlyretmissflg.eq('NT').reindex(calendar,fill_value=False)
        r.loc[nt] = 0.
        # A missing quote is a non-executable stale mark. When trading resumes,
        # verify that CRSP's next return uses the last observed pre-gap price.
        # The missing-price day stays missing in the predictor's feature path.
        mp=q.dlyretmissflg.eq('MP') & q.dlyret.isna()
        for gap_day in q.index[mp]:
            prior=q.loc[q.index<gap_day,'dlyprc'].dropna()
            later=q.loc[(q.index>gap_day)&q.dlyret.notna()]
            complete=False
            if len(prior) and len(later):
                resume=later.iloc[0]
                complete=bool(np.isclose(resume.dlyprevprc,prior.iloc[-1],rtol=1e-6,atol=1e-6)
                    and np.isclose(resume.dlyfacprc,1.))
                if complete:r.at[gap_day]=0.
                stale_audit.append(dict(permno=int(perm),gap_day=str(gap_day.date()),
                    resume_day=str(later.index[0].date()),prior_quote=float(prior.iloc[-1]),
                    resume_previous_quote=resume.dlyprevprc,full_gap_return_verified=complete))
        first = q.index[0]
        if pd.isna(r.at[first]) and q.at[first,'dlyretmissflg'] == 'NS':
            r.at[first] = 0.
        exit_day = None; unknown = False
        if perm in events.index:
            ev = events.loc[perm]
            dated = [ev.deldlydt]
            if pd.notna(ev.delnextdt): dated.append(ev.delnextdt)
            if pd.notna(ev.delamtdt): dated.append(ev.delamtdt)
            observed = max(x for x in dated if pd.notna(x))
            pos = calendar.searchsorted(observed)
            exit_day = calendar[pos] if pos < len(calendar) else None
            source_event = ev.deldlydt
            event_clocks[perm]=(source_event,observed)
            terminal_return = q.dlyret.get(source_event,np.nan)  # raw, before stale-mark repair
            reconstructed = False
            if (pd.isna(terminal_return) and ev.delpaymenttype == 'PRCF'
                    and (pd.isna(ev.deldivamt) or ev.deldivamt==0) and pd.notna(ev.delnextprc)
                    and pd.notna(ev.deldtprc) and ev.deldtprc > 0):
                previous=q.loc[q.index<source_event,'dlyprc'].dropna()
                assert len(previous) and np.isclose(previous.iloc[-1],ev.deldtprc,rtol=1e-6,atol=1e-6)
                terminal_return = ev.delnextprc / ev.deldtprc - 1
                reconstructed = True
            if exit_day is not None:
                r.loc[(calendar >= source_event) & (calendar < exit_day)] = 0.
                r.at[exit_day] = terminal_return
                r.loc[calendar > exit_day] = 0.
            else:
                r.loc[calendar >= source_event] = 0.
            # Cash plus unpriced/nontransferable rights is not a complete label.
            unknown = ev.delretmisstype == 'MV' or pd.isna(terminal_return)
            event_rows.append(dict(permno=int(perm),delisting_date=str(ev.delistingdt.date()),
                crsp_return_date=str(source_event.date()),recognition_date=str(observed.date()),
                execution_date=str(exit_day.date()) if exit_day is not None else None,
                terminal_return=terminal_return,reconstructed_from_dated_quote=reconstructed,
                unknown_complete_proceeds=bool(unknown),payment_type=ev.delpaymenttype,
                successor_permno=int(ev.delpermno) if pd.notna(ev.delpermno) else None))
        # Missing returns create missing wealth observations; a later valid
        # trailing ratio can be used only after that missing observation leaves
        # its full window. Outcome products independently check every return.
        wealth = (1+r).cumprod(skipna=True) * 100.
        wealth.loc[q.index[mp]]=np.nan
        before_first = calendar < first
        wealth.loc[before_first] = np.nan
        quotes = pd.DataFrame({'adj_close':wealth,'source':'CRSP_CIZ'},index=calendar)
        sec = 'CRSP_'+str(int(perm))
        data_by[perm] = quotes; ret_by[perm] = r
        entry = q.dlyclose.gt(0) & q.tradingstatusflg.eq('A')
        # Do not use the ex-post last-record flag for executable-price eligibility.
        entry_by[perm] = entry.reindex(calendar,fill_value=False)
        resolved_by[perm] = exit_day; unknown_by[perm] = unknown
        z = pd.DataFrame({'date':calendar,'ticker':sec,'adj_close':wealth.to_numpy(),
                          'close':q.dlyclose.reindex(calendar).to_numpy(),
                          'source':'CRSP_CIZ','stock_splits':0.})
        if perm == 84398: z['ticker'] = 'SPY'
        if perm == 86755: z['ticker'] = 'QQQ'
        frames.append(z)
    pd.concat(frames,ignore_index=True).to_parquet(DATA/'inputs/prices.parquet',index=False)
    pd.DataFrame(ret_by,index=calendar).to_parquet(DATA/'returns.parquet')
    pd.DataFrame(entry_by,index=calendar).to_parquet(DATA/'entry_flags.parquet')
    pd.DataFrame(event_rows).to_csv(DATA/'event_recognition.csv',index=False)
    pd.DataFrame(stale_audit).to_csv(DATA/'missing_quote_resume_audit.csv',index=False)
    rows=[]; features=[]; checks=[]; future_checks=[]
    for date,end in date_ends.itertuples(index=False,name=None):
        current=spells[spells.mbrstartdt.le(date)&(spells.mbrenddt.ge(date)|spells.mbrenddt.isna())].copy()
        assert not current.duplicated('permno').any()
        tickers=[]
        for perm in current.permno:
            nm=names_by[perm]
            live=nm[nm.secinfostartdt.le(date)&nm.secinfoenddt.ge(date)]
            ticker=live.iloc[-1].ticker if len(live) else np.nan
            cls=live.iloc[-1].shareclass if len(live) else np.nan
            if ticker in ['BRK','BF'] and pd.notna(cls):ticker=str(ticker)+'.'+str(cls)
            tickers.append(ticker if pd.notna(ticker) else 'CRSP_'+str(int(perm)))
        current['ticker']=tickers
        current=current.sort_values(['ticker','permno'])
        assert len(current) >= 450, ('Unexpected index coverage',str(date),len(current))
        for v in current.itertuples(index=False):
            perm = v.permno; sec='CRSP_'+str(int(perm))
            q=data_by[perm]; f,z=calculate(q,calendar,date)
            a,_=calculate(q.loc[q.index<date],calendar,date)
            scaled=q.loc[q.index<date].copy();scaled['adj_close']*=7.3
            b,_=calculate(scaled,calendar,date)
            fv,av,bv=[np.array([s[k] for k in FEATURES],float) for s in [f,a,b]]
            assert np.array_equal(np.isnan(fv),np.isnan(av)) and np.array_equal(np.isnan(fv),np.isnan(bv))
            assert np.allclose(fv,av,equal_nan=True,atol=1e-12,rtol=0)
            assert np.allclose(fv,bv,equal_nan=True,atol=1e-12,rtol=0)
            checks.append(float(np.nanmax(np.abs(fv-bv))) if np.isfinite(fv).any() else 0.)
            future_checks.append(float(np.nanmax(np.abs(fv-av))) if np.isfinite(fv).any() else 0.)
            label=np.nan; reason='right_censored'; exit_day=resolved_by[perm]
            interval=ret_by[perm].loc[(calendar>date)&(calendar<=end)]
            if end <= calendar[-1]:
                reason='missing_observed_return'
                if len(interval) and interval.notna().all() and entry_by[perm].get(date,False):
                    label=float(np.log(max(float(np.prod(1+interval.to_numpy())),1e-8)))
                    reason='observed_CRSP_total_return'
                if unknown_by[perm] and exit_day is not None and date < exit_day <= end:
                    label=np.nan;reason='unknown_complete_delisting_proceeds'
                if perm in event_clocks:
                    source_event,recognition=event_clocks[perm]
                    if date>=source_event:
                        label=np.nan
                        if entry_by[perm].get(date,False):reason='security_already_exited_at_decision'
                    if date < source_event <= end and recognition > end:
                        label=np.nan;reason='terminal_quote_not_yet_observed_at_label_end'
            rows.append(dict(ticker=v.ticker if pd.notna(v.ticker) else sec,security_id=sec,
                permno=int(perm),provider_symbol_candidate=sec,decision_date=date,label_end=end,
                target=label,entry_observed=bool(entry_by[perm].get(date,False)),
                label_status=reason,training_label_eligible=bool(np.isfinite(label)),
                identity_tier='CRSP_PERMNO',gap_class=''))
            features.append(dict(ticker=rows[-1]['ticker'],security_id=sec,decision_date=date,
                last_observation=z.adj_close.last_valid_index(),**f))
    panel=pd.DataFrame(rows); feat=pd.DataFrame(features)
    assert not panel.duplicated(['permno','decision_date']).any()
    collisions=panel.duplicated(['ticker','decision_date'],keep=False)
    panel.loc[collisions].to_csv(DATA/'ticker_collisions_preserved.csv',index=False)
    panel.loc[collisions,'ticker']=panel.loc[collisions,'ticker']+' ['+panel.loc[collisions,'security_id']+']'
    feat['ticker']=panel.ticker.to_numpy()
    assert not panel.duplicated(['ticker','decision_date']).any()
    panel.to_csv(DATA/'model_panel.csv.gz',index=False)
    feat.to_csv(RUN/'price_features_candidate.csv.gz',index=False)
    feature_check=dict(rows=len(panel),features=FEATURES,future_deletion_max_difference=max(future_checks),
        uniform_adjustment_scale_max_difference=max(checks),source_transitions_require_review=False,
        official_CRSP_index_membership=True,all_official_cases_retained_asserted_separately_in_source_case_audit=True,
        terminal_log_floor=1e-8,terminal_floor_observed_cases=int((panel.target<=np.log(1e-8)).sum()),
        open_ended_membership_spells=int(spells.mbrenddt.isna().sum()),
        label_status_counts=panel.label_status.value_counts().to_dict(),
        member_counts={str(k.date()):int(v) for k,v in panel.groupby('decision_date').size().items()},
        entry_gaps=int((~panel.entry_observed).sum()),
        membership_constructed_from_effective_intervals_not_quote_presence=True,
        ticker_collision_rows_preserved=int(collisions.sum()),
        receipt_scope='Frozen CRSP source and finite construction checks, not universal vendor certification')
    (RUN/'feature_checks.json').write_text(json.dumps(feature_check,indent=2,default=str))
    print(json.dumps({k:v for k,v in feature_check.items() if k!='member_counts'},indent=2),flush=True)

if __name__=='__main__': main()
