"""One fixed price-only repair experiment, preserving every inference case.

This run removes issuer-link inputs. It is a development experiment, not an
untouched holdout or a universal certification of external source histories.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
from pathlib import Path
import sys, json, hashlib, time
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import joblib
from build_features import FEATURES
from causal_policy import causal_partition,verify_known_horizon_outcomes,label_observation_clock

CODE=Path(__file__).resolve().parent
P=Path(os.environ.get('FINRL_RUN_DIR',str(CODE.parent/'run'))).resolve()
OLD=Path(os.environ.get('FINRL_DATA_DIR',str(CODE.parent/'data-source'))).resolve()
sys.path.insert(0,str(CODE/'support'))
from missing_policy import TrainOnlyImputer
from fit_boundary_log import FitLog, describe, compare_with_expected
SEEDS = [42,0,1,2,3,4,5,6,7,8,9]

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def status(stage,**kwargs):
    v=dict(stage=stage,time=datetime.now(ZoneInfo('America/New_York')).isoformat(),**kwargs)
    (P/'status.json').write_text(json.dumps(v,indent=2,default=str))
    print(json.dumps(v,default=str),flush=True)

def main():
    validate_only='--validate-inputs-only' in sys.argv
    if not validate_only and (P/'predictions.csv').exists(): raise ValueError('Completed predictions exist; do not overwrite')
    checks=json.loads((P/'feature_checks.json').read_text())
    assert checks['future_deletion_max_difference']==0 and checks['uniform_adjustment_scale_max_difference']<1e-12
    assert not checks['source_transitions_require_review']
    assert not any(x in FEATURES for x in ['ticker','security_id','identity_tier','status','available','target','label_end'])
    protocol=dict(created_at=datetime.now(ZoneInfo('America/New_York')).isoformat(),
        price_features=FEATURES,seeds=SEEDS,
        reason='Index enhancement requires stock-relative-to-QQQ forecasts. Subtract the contemporaneous QQQ label from each observed stock label to remove market-wide quarter effects; unchanged six prior-price features and all11 seeds.',
        feature_rules='Strictly preceding market-session realized-total-return wealth ratios; each feature window requires one source and complete positive prices.',
        missing_policy='Training-only feature medians plus missing flags; no row filter on issuer status or feature coverage. Supervised eligibility is dated public-rights policy plus initial quote; any other missing realized-horizon label is fatal, not silently dropped.',
        rf='200 trees per seed, depth6, minimum leaf20; arithmetic average of ALL11 validation and inference predictions.',
        comparison_estimators='Fixed XGB200/depth3/lr0.05; LGBM200/leaves15/lr0.05; same fixed candidate parameters as previous private experiment.',
        model_selection='Lowest validation MSE on three realized quarters within a nominal four-quarter span; no test-period portfolio-return selection.',
        portfolio='Primary:80%QQQ+20%equal-weight top5% historical S&P500 stocks by validation-selected model; quarterly. RF and pure-selection comparators separately disclosed.',
        window='May1 2020 to Sep30 2025 with QQQ fallback before June1; also report June30 descriptive slice.',
        costs='10bps each traded dollar; stress20bps and one-session delay.',
        labels='Observed log stock wealth minus observed log QQQ wealth over the SAME endpoints, only used after purged realization dates. Original stock targets retained separately; main-run unknown CVRs stay unknown. The separate labelled stress explicitly replaces two unknown outcomes with the hypothetical log floor.',
        holdout_claim=False,hypothetical_label_floor_stress=os.environ.get('FINRL_LABEL_FLOOR_STRESS')=='1',feature_sha256=sha(P/'price_features_candidate.csv.gz'),
        source_panel_sha256=sha(OLD/'model_panel.csv.gz'),
        known_rights_policy='Dated public nonmarket-right facts applied during trading; developed after unvalued exposure in earlier run; same fixed rule in this benchmark-relative replay.',
        adoption='WRDS replay of the previously developed QQQ80/top5 enhancement; fixed choices inherited, no untouched-holdout claim.')
    if not validate_only:
        (P/'protocol.json').write_text(json.dumps(protocol,indent=2))
        (P/'models').mkdir(exist_ok=True)
    old=pd.read_csv(OLD/'model_panel.csv.gz',low_memory=False,float_precision='round_trip',parse_dates=['decision_date','label_end'])
    f=pd.read_csv(P/'price_features_candidate.csv.gz',float_precision='round_trip',parse_dates=['decision_date','last_observation'])
    d=old.drop(columns=[x for x in FEATURES if x in old]).merge(f[['ticker','decision_date','last_observation']+FEATURES],on=['ticker','decision_date'],validate='one_to_one')
    d['stock_total_return_target']=d.target
    sp=pd.read_parquet(OLD/'inputs/prices.parquet',columns=['date','adj_close'],filters=[('ticker','==','QQQ')]).set_index('date').adj_close
    sp.index=pd.to_datetime(sp.index)
    d['qqq_target']=np.log(d.label_end.map(sp)/d.decision_date.map(sp))
    if d.loc[d.target.notna(),'qqq_target'].isna().any(): raise ValueError('Missing benchmark endpoint for an observed stock label')
    d['target']=d['stock_total_return_target']-d.qqq_target
    assert d.target.notna().equals(d.stock_total_return_target.notna())
    event_clock=pd.read_csv(OLD/'event_recognition.csv',parse_dates=['crsp_return_date','recognition_date'])
    try:repair_clock=pd.read_csv(OLD/'missing_quote_resume_audit.csv',parse_dates=['gap_day','resume_day'])
    except pd.errors.EmptyDataError:repair_clock=pd.DataFrame(columns=['permno','gap_day','resume_day'])
    d['label_available_date']=label_observation_clock(d,event_clock,repair_clock)
    stress=os.environ.get('FINRL_LABEL_FLOOR_STRESS')=='1'
    d['supervised_decision_eligible']=verify_known_horizon_outcomes(d,sp.index.max(),stress)
    d['quarter']=d.decision_date
    # Available clock belongs to PRICE features, never to old issuer filings.
    d['available']=d.last_observation.fillna(d.decision_date-pd.Timedelta(days=1))
    assert (d.available<d.decision_date).all()
    if validate_only:
        archive=pd.read_csv(P/'model_panel.csv.gz',parse_dates=['decision_date'],float_precision='round_trip')
        assert d[['security_id','decision_date']].equals(archive[['security_id','decision_date']])
        a=d[FEATURES+['target']].to_numpy();b=archive[FEATURES+['target']].to_numpy()
        assert np.array_equal(np.isnan(a),np.isnan(b))
        delta=float(np.nanmax(np.abs(a-b)));assert delta<1e-12
        dates=sorted(d.decision_date.unique());windows=0
        for date in dates:
            date=pd.Timestamp(date)
            if date<pd.Timestamp('2020-06-01'):continue
            lower=pd.Timestamp(dates[dates.index(date)-4]);tr,va,inf=causal_partition(d,lower,date,stress)
            recorded=pd.read_csv(P/'expected_fit_manifest.csv')
            for role,z in [('outer_train',tr),('outer_validation',va),('inference',inf)]:
                cutoff=lower if role=='outer_train' else date
                witness=describe(str(date.date())+'/rf_seed_42',role,z,cutoff,lower=lower if role=='outer_validation' else None)
                row=recorded[recorded.window.eq(witness['window']) & recorded.role.eq(role)].iloc[0]
                assert witness['key_sha256']==row.key_sha256 and witness['rows']==row.rows
            windows+=1
        print(json.dumps({'actual_trainer_source_loading_exercised':True,'windows':windows,'maximum_feature_target_difference_from_archived_panel':delta,'causal_split_rows_identical':True,'model_fits_executed':0,'archived_results_modified':False},indent=2))
        return
    d.to_csv(P/'model_panel.csv.gz',index=False)
    dates=sorted(d.decision_date.unique())
    log=FitLog(P/'model_fit_execution.jsonl',P.name)
    expected=[];frames=[];scores=[];seed_frames=[]
    started=time.time()
    for i,date in enumerate(dates):
        date=pd.Timestamp(date)
        if date<pd.Timestamp('2020-06-01'): continue
        lower=pd.Timestamp(dates[i-4])
        tr,va,inf=causal_partition(d,lower,date,stress)
        im=TrainOnlyImputer(tuple(FEATURES)).fit(tr[FEATURES])
        xt,xv,xi=[im.transform(x[FEATURES]) for x in [tr,va,inf]]
        frame=inf[['ticker','security_id','decision_date','available','entry_observed']].reset_index(drop=True).copy()
        pv=[];pi=[];mses={}
        for seed in SEEDS:
            name='rf_seed_'+str(seed);window=str(date.date())+'/'+name
            status('training',decision=str(date.date()),model=name,training=len(tr),validation=len(va),inference=len(inf))
            expected += [describe(window,'outer_train',tr,lower),describe(window,'outer_validation',va,date,lower=lower),describe(window,'inference',inf,date)]
            log.record(window,'outer_train',tr,lower,context={'features':FEATURES,'panel_sha256':sha(P/'model_panel.csv.gz')})
            m=RandomForestRegressor(n_estimators=200,max_depth=6,min_samples_leaf=20,max_features=1.,n_jobs=4,random_state=seed).fit(xt,tr.target)
            log.record(window,'outer_validation',va,date,lower=lower);pv.append(m.predict(xv))
            log.record(window,'inference',inf,date);pi.append(m.predict(xi))
            joblib.dump(dict(imputer=im,model=m),P/'models'/(str(date.date())+'_'+name+'.joblib'),compress=3)
            sf=frame[['ticker','decision_date']].copy();sf['seed']=seed;sf['score']=pi[-1];seed_frames.append(sf)
        frame['rf']=np.mean(pi,axis=0);mses['rf']=float(mean_squared_error(va.target,np.mean(pv,axis=0)))
        for name,m in [('xgb',XGBRegressor(n_estimators=200,max_depth=3,learning_rate=.05,subsample=1.,colsample_bytree=1.,n_jobs=4,random_state=42,tree_method='hist')),
                       ('gbm',LGBMRegressor(n_estimators=200,num_leaves=15,learning_rate=.05,min_child_samples=20,n_jobs=4,random_state=42,verbosity=-1))]:
            window=str(date.date())+'/'+name
            expected += [describe(window,'outer_train',tr,lower),describe(window,'outer_validation',va,date,lower=lower),describe(window,'inference',inf,date)]
            log.record(window,'outer_train',tr,lower,context={'features':FEATURES})
            m.fit(xt,tr.target)
            log.record(window,'outer_validation',va,date,lower=lower);mses[name]=float(mean_squared_error(va.target,m.predict(xv)))
            log.record(window,'inference',inf,date);frame[name]=m.predict(xi)
            joblib.dump(dict(imputer=im,model=m),P/'models'/(str(date.date())+'_'+name+'.joblib'),compress=3)
        winner=min(mses,key=mses.get);frame['winner']=winner;frame['best']=frame[winner]
        frames.append(frame)
        scores += [dict(decision_date=str(date.date()),model=k,validation_mse=v,winner=k==winner) for k,v in mses.items()]
    out=pd.concat(frames,ignore_index=True)
    needed=d[d.decision_date>='2020-06-01'][['ticker','decision_date']]
    assert set(map(tuple,out[['ticker','decision_date']].to_numpy()))==set(map(tuple,needed.to_numpy()))
    assert out[['rf','xgb','gbm','best']].notna().all().all()
    out.to_csv(P/'predictions.csv',index=False)
    pd.concat(seed_frames,ignore_index=True).to_csv(P/'rf_seed_predictions.csv.gz',index=False)
    pd.DataFrame(scores).to_csv(P/'model_scores.csv',index=False)
    pd.DataFrame(expected).to_csv(P/'expected_fit_manifest.csv',index=False)
    audit=compare_with_expected(pd.read_csv(P/'expected_fit_manifest.csv'),P/'model_fit_execution.jsonl')
    if audit: raise ValueError('Executed fit boundaries differ from protocol: '+str(audit[:3]))
    (P/'fit_verification.json').write_text(json.dumps(audit,indent=2,default=str))
    status('training_complete',windows=len(frames),model_fits=len(frames)*13,inference=len(out),elapsed_seconds=time.time()-started,
        winner_counts=out.groupby('decision_date').winner.first().value_counts().to_dict(),prediction_sha256=sha(P/'predictions.csv'))

if __name__=='__main__': main()
