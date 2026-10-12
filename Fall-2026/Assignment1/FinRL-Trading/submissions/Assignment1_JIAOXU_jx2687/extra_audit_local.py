import pandas as pd,numpy as np,sqlite3,json
from pathlib import Path
import argparse
a=argparse.ArgumentParser();a.add_argument('--source',required=True);a.add_argument('--output',default='results');args=a.parse_args()
s=Path(args.source);out=Path(args.output)
f=pd.read_csv(s/'data/fundamental_data_full.csv').sort_values(['ticker','datadate']);f['next_price']=f.groupby('ticker').trade_price.shift(-1);test=f.dropna(subset=['next_price','trade_price','y_return']);err=(np.log(test.next_price/test.trade_price)-test.y_return).abs()
p=pd.read_csv(s/'prepared/final_ratios.csv');missing=pd.read_csv(out/'missing_price_tickers.csv');p[p.tic.isin(missing.tic)].groupby('tic').agg(prepared_rows=('tic','size'),first_quarter=('date','min'),last_quarter=('date','max')).to_csv(out/'historical_missing_examples.csv')
c=sqlite3.connect(f'file:{s}/data/finrl_trading.db?mode=ro',uri=True)
snap=pd.read_sql('select date,tickers from sp500_components_details',c);pool=set(pd.read_sql('select distinct ticker from price_data',c).ticker)
checks=[]
for row in snap.itertuples():
 ticks=set(row.tickers.split(',')); checks.append({'date':row.date,'members':len(ticks),'members_without_price':sorted(ticks-pool),'price_pool_extras':sorted(pool-ticks)})
a={'label_reconstruction_rows':len(test),'max_log_label_error':float(err.max()),'zero_labels':int((f.y_return==0).sum()),'report_before_quarter_rows':int((pd.to_datetime(p.reportdate)<pd.to_datetime(p.date)).sum()),'price_pool_snapshot_checks':checks}
(out/'additional_audit.json').write_text(json.dumps(a,indent=2));print(json.dumps(a,indent=2)[:1800])

