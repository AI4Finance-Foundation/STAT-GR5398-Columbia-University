import sys
from pathlib import Path
sys.path.insert(0,str(Path('work/retraining/improved').resolve()))
import numpy as np,pandas as pd
from selection.selector import _fit_partition_features
train=pd.DataFrame({'a':[1.,2.,3.,np.nan],'b':[2.,np.nan,4.,6.]})
val=pd.DataFrame({'a':[1e12],'b':[-1e12]})
x,first=_fit_partition_features(train,[train,val],['a','b'])
changed=val*1e12
y,second=_fit_partition_features(train,[train,changed],['a','b'])
assert np.array_equal(x,y),'Inference outliers changed fitted training preprocessing'
assert np.isfinite(x).all() and np.isfinite(first).all()
assert (train.isna().sum()==pd.Series({'a':1,'b':1})).all(),'Input mutated'
print('PASS: inference perturbation does not change training representation; missing values finite; no input mutation.')
boolean_train=train.assign(sector_example=[True,False,True,False])
boolean_val=val.assign(sector_example=[True])
transformed, inferred=_fit_partition_features(boolean_train,[boolean_train,boolean_val],['a','b','sector_example'])
assert transformed.dtype.kind=='f' and np.isfinite(transformed).all()
assert np.isfinite(inferred).all()
print('PASS: boolean sector dummies are converted before quantile calculation.')
