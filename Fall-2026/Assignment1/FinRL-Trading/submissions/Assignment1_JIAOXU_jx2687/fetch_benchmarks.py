"""Public market-data download only. No broker interface."""
import argparse
from pathlib import Path
import yfinance as yf
p=argparse.ArgumentParser();p.add_argument('--output',default='benchmark_data');a=p.parse_args();d=Path(a.output);d.mkdir(parents=True,exist_ok=True)
for tic in ['SPY','QQQ']:
 data=yf.download(tic,start='2020-04-01',end='2025-10-01',auto_adjust=False,progress=False,multi_level_index=False)
 assert len(data)>1300 and 'Adj Close' in data
 data.to_csv(d/f'{tic}_download.csv');print(tic,len(data))
