"""Verify saved estimator inputs, train-only medians, and forecast averaging.
No estimator is fitted by this audit.
"""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import joblib
P=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
sys.path.insert(0,str(Path(__file__).resolve().parent/'support'))
F=['momentum_21','momentum_63','momentum_126','momentum_252','volatility_63','drawdown_126']
N=[c for f in F for c in [f,f+'_missing']]
SEEDS=[42,0,1,2,3,4,5,6,7,8,9]
d=pd.read_csv(P/'model_panel.csv.gz',low_memory=False,float_precision='round_trip',parse_dates=['decision_date','label_end','available'])
sys.path.insert(0,str(Path(__file__).resolve().parent))
from causal_policy import causal_partition
stress=__import__('os').environ.get('FINRL_LABEL_FLOOR_STRESS')=='1'
p=pd.read_csv(P/'predictions.csv',float_precision='round_trip',parse_dates=['decision_date','available'])
seed_csv=pd.read_csv(P/'rf_seed_predictions.csv.gz',float_precision='round_trip',parse_dates=['decision_date'])
dates=sorted(d.decision_date.unique());rows=[];numeric=[]
for day,inf in d[d.decision_date>='2020-06-01'].groupby('decision_date'):
    lower=pd.Timestamp(dates[dates.index(day)-4]);tr,va,inf=causal_partition(d,lower,day,stress)
    # The recorded availability masks must leave the ordered training
    # values and labels unchanged. This check does not refit anything.
    prefix=d[(d.decision_date<lower)&(d.label_end<lower)]
    tr2=prefix[prefix.target.notna()]
    assert np.array_equal(tr.target.to_numpy(),tr2.target.to_numpy())
    assert np.array_equal(tr[F].to_numpy(),tr2[F].to_numpy(),equal_nan=True)
    pred=p[p.decision_date==day].set_index('ticker').loc[inf.ticker]
    rf=[];mse={}
    from sklearn.metrics import mean_squared_error
    val_rf=[]
    for name in ['rf_seed_'+str(s) for s in SEEDS]+['xgb','gbm']:
        artifact=joblib.load(P/'models'/(str(day.date())+'_'+name+'.joblib'))
        im,m=artifact['imputer'],artifact['model']
        assert list(im.features)==F
        for f in F:assert im.medians_[f]==float(tr[f].median())
        assert list(m.feature_names_in_)==N
        y=m.predict(im.transform(inf[F]));v=m.predict(im.transform(va[F]))
        if name.startswith('rf_seed_'):
            seed=int(name.removeprefix('rf_seed_'))
            recorded=seed_csv[seed_csv.decision_date.eq(day)&seed_csv.seed.eq(seed)].set_index('ticker').loc[inf.ticker].score.to_numpy()
            assert np.max(np.abs(y-recorded))<1e-12
            rf.append(y);val_rf.append(v)
        else:
            # XGB predicts float32. CSV renders the shortest float32-decimal,
            # whose parsed float64 need not equal that float32's exact binary
            # value. Cast back to the estimator dtype before exact comparison.
            delta=float(np.max(np.abs(y-pred[name].to_numpy())))
            numeric.append(dict(decision_date=str(day.date()),model=name,max_csv_float64_difference=delta,native_prediction_dtype=str(y.dtype)))
            assert np.array_equal(y,pred[name].to_numpy().astype(y.dtype)) or delta<1e-12
            mse[name]=float(mean_squared_error(va.target,v))
    avg=np.mean(rf,axis=0);difference=float(np.max(np.abs(avg-pred.rf.to_numpy())))
    assert difference<1e-12
    mse['rf']=float(mean_squared_error(va.target,np.mean(val_rf,axis=0)))
    # Tie order mirrors protocol RF, XGB, GBM; deterministic.
    winner=min(['rf','xgb','gbm'],key=lambda x:mse[x]);assert pred.winner.eq(winner).all()
    assert (tr.label_end<lower).all() and (va.label_end<day).all()
    assert (tr.available<tr.decision_date).all() and (inf.available<inf.decision_date).all()
    rows.append(dict(decision_date=str(day.date()),winner=winner,rf_average_max_diff=difference,
        training_rows=len(tr),validation_rows=len(va),inference_rows=len(inf),saved_estimators_checked=13))
pd.DataFrame(rows).to_csv(P/'saved_model_input_checks.csv',index=False)
pd.DataFrame(numeric).to_csv(P/'saved_model_csv_precision_checks.csv',index=False)
out=dict(windows=len(rows),saved_estimators_checked=sum(x['saved_estimators_checked'] for x in rows),
    inference_rows=len(p),no_financial_issuer_or_identifier_columns_in_models=True,
    all_training_medians_match=True,all_purge_and_price_availability_guards_pass=True,
    recorded_realized_clock_training_matrix_consistency=True,
    rf_average_max_difference=max(x['rf_average_max_diff'] for x in rows),
    all_validation_selected_winners_match=True,no_refits=True)
(P/'saved_model_input_checks.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
