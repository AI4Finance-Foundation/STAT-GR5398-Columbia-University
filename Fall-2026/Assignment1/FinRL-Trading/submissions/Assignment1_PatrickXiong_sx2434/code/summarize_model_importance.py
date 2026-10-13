"""Compute the report's selected-family normalized tree importance from a replay."""
from pathlib import Path
import argparse,sys
import numpy as np,pandas as pd,joblib
a=argparse.ArgumentParser();a.add_argument('--run-dir',required=True);a.add_argument('--output-dir',required=True);x=a.parse_args()
R=Path(x.run_dir);O=Path(x.output_dir);O.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(Path(__file__).resolve().parent/'support'))
F=['momentum_21','momentum_63','momentum_126','momentum_252','volatility_63','drawdown_126'];rows=[];chosen=[]
for date,g in pd.read_csv(R/'model_scores.csv').groupby('decision_date'):
 winner=g.loc[g.winner,'model'].iloc[0]
 for family,suffixes in [('rf',['rf_seed_'+str(s) for s in [42,0,1,2,3,4,5,6,7,8,9]]),('xgb',['xgb']),('gbm',['gbm'])]:
  values=[]
  for suffix in suffixes:
   m=joblib.load(R/'models'/(date+'_'+suffix+'.joblib'))['model'];v=m.booster_.feature_importance(importance_type='gain') if family=='gbm' else m.feature_importances_;v=np.array(v,float);v=v/v.sum();values.append([v[2*i:2*i+2].sum() for i in range(6)])
  vector=np.mean(values,axis=0);rows.append(dict(decision_date=date,family=family,winner=family==winner,**dict(zip(F,vector))))
  if family==winner:chosen.append(vector)
pd.DataFrame(rows).to_csv(O/'feature_importance_by_window.csv',index=False)
pd.Series(np.mean(chosen,axis=0),index=F).sort_values(ascending=False).rename('normalized_tree_importance').to_csv(O/'selected_feature_importance.csv',index_label='feature')
