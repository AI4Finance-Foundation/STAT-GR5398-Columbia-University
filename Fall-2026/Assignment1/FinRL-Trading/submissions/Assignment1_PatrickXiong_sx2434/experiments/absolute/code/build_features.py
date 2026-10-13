"""Six strictly prior-session, scale-invariant price features."""
import numpy as np
import pandas as pd

FEATURES = ['momentum_21', 'momentum_63', 'momentum_126', 'momentum_252',
            'volatility_63', 'drawdown_126']

def calculate(q, calendar, day):
    # A close on the decision day is never used to predict a trade at that close.
    history = calendar[calendar < day][-253:]
    z = q.reindex(history)
    x = z.adj_close
    values = {}
    for n in [21, 63, 126, 252]:
        a = x.iloc[-(n+1):]
        valid = (len(a) == n+1 and a.notna().all() and (a > 0).all()
                 and z.loc[a.index,'source'].nunique() == 1)
        values['momentum_'+str(n)] = float(np.log(a.iloc[-1]/a.iloc[0])) if valid else np.nan
    a = x.iloc[-64:]
    valid = (len(a) == 64 and a.notna().all() and (a > 0).all()
             and z.loc[a.index,'source'].nunique() == 1)
    values['volatility_63'] = float(np.log(a).diff().dropna().std(ddof=1)*np.sqrt(252)) if valid else np.nan
    a = x.iloc[-127:]
    valid = (len(a) == 127 and a.notna().all() and (a > 0).all()
             and z.loc[a.index,'source'].nunique() == 1)
    values['drawdown_126'] = float((a/a.cummax()-1).min()) if valid else np.nan
    return values, z
