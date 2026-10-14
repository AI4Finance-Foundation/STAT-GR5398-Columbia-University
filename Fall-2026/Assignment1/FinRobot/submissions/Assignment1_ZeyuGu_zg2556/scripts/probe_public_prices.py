"""Freeze a public Yahoo Finance quote response, independent of paid FMP/WRDS."""
import json
from pathlib import Path
from fetch_sec_evidence import Snapshot

ROOT=Path(__file__).resolve().parents[1]
snap=Snapshot(ROOT/'evidence/public_market/yahoo_2026-10-11',False,'AcademicResearch/1.0 (ZeyuGu-zg2556)')
for ticker in ('AVGO','QCOM'):
    url=f'https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?range=1y&interval=1d'
    raw=snap.get(url,ticker+'.json')
    if raw:
        obj=json.loads(raw)['chart']['result'][0]
        print(ticker,obj['meta'])
        print('Latest bars',list(zip(obj['timestamp'],obj['indicators']['quote'][0]['close']))[-3:])
