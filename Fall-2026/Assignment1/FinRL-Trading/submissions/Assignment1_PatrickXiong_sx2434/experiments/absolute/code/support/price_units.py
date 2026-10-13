"""Convert a frozen quoted close to units of the security on an anchor date."""
import numpy as np
import pandas as pd
from lifecycle_returns import UnresolvedData

def contemporaneous_close(close, date, basis):
    if basis.get('convention') not in {'raw', 'split_adjusted'}:
        raise UnresolvedData('Unknown price-unit convention')
    close=float(close);date=pd.Timestamp(date)
    if not np.isfinite(close) or close<=0:raise UnresolvedData('Invalid price')
    if basis['convention']=='raw':return close
    if not basis.get('split_history_complete') or not basis.get('basis_date'):
        raise UnresolvedData('Incomplete split-factor horizon')
    anchor=pd.Timestamp(basis['basis_date'])
    if anchor<date:raise UnresolvedData('Price basis precedes quote')
    events=basis.get('split_events')
    if not isinstance(events,list):raise UnresolvedData('Missing split history')
    factor=1.;seen=set()
    for e in events:
        d=pd.Timestamp(e['date']);ratio=float(e['new_per_old'])
        if d in seen or not np.isfinite(ratio) or ratio<=0 or d>anchor:
            raise UnresolvedData('Invalid or duplicate split event')
        seen.add(d)
        if date<d:factor*=ratio
    return close*factor
