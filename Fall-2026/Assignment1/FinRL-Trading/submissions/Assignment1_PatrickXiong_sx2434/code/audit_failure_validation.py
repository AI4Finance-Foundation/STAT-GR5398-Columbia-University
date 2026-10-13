"""Report the influence of two retained 2023 bank failures on prior validation."""
from pathlib import Path
import os,sys,json
import numpy as np,pandas as pd,joblib
from sklearn.metrics import mean_squared_error
ROOT=Path(__file__).resolve().parents[1];RUN=Path(os.environ.get('FINRL_RUN_DIR',str(ROOT/'run')))
sys.path.insert(0,str(Path(__file__).resolve().parent/'support'))
from build_features import FEATURES
p=pd.read_csv(RUN/'model_panel.csv.gz',parse_dates=['decision_date','label_end'],float_precision='round_trip')
dates=sorted(p.decision_date.unique());rows=[];changed=[]
for day in dates:
 day=pd.Timestamp(day)
 if day<pd.Timestamp('2020-06-01'):continue
 lower=pd.Timestamp(dates[dates.index(day)-4]);va=p[(p.decision_date>=lower)&(p.decision_date<day)&(p.label_end<day)&p.target.notna()]
 affected=va.permno.isin([11786,90090])&va.decision_date.eq(pd.Timestamp('2023-03-01'));mse={};trim={};rf=[]
 if not affected.any():continue
 for family in ['rf_seed_'+str(s) for s in [42,0,1,2,3,4,5,6,7,8,9]]+['xgb','gbm']:
  a=joblib.load(RUN/'models'/(str(day.date())+'_'+family+'.joblib'));y=a['model'].predict(a['imputer'].transform(va[FEATURES]))
  if family.startswith('rf'):rf.append(y);continue
  mse[family]=float(mean_squared_error(va.target,y));trim[family]=float(mean_squared_error(va.loc[~affected,'target'],y[~affected]))
 y=np.mean(rf,axis=0);mse['rf']=float(mean_squared_error(va.target,y));trim['rf']=float(mean_squared_error(va.loc[~affected,'target'],y[~affected]))
 winner=min(['rf','xgb','gbm'],key=lambda f:mse[f]);without=min(['rf','xgb','gbm'],key=lambda f:trim[f])
 changed.append(dict(date=str(day.date()),actual_winner=winner,without_failure_winner=without))
 for f in ['rf','xgb','gbm']:rows.append(dict(date=day,model=f,validation_rows=len(va),bank_failure_rows=int(affected.sum()),full_mse=mse[f],mse_without_bank_failure_rows=trim[f],actual_winner=winner,without_failure_winner=without))
pd.DataFrame(rows).to_csv(RUN/'bank_failure_validation_mse.csv',index=False)
(RUN/'bank_failure_validation_checks.json').write_text(json.dumps({'affected_windows':len(changed),'selection_changes_if_failures_removed':sum(x['actual_winner']!=x['without_failure_winner'] for x in changed),'windows':changed,'scope':'Saved-model validation sensitivity; the adopted protocol retains both failure outcomes and does not refit or drop them.'},indent=2))
