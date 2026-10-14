"""Inspect public SEC annual ingredients; never opens private WRDS exports."""
import json
from pathlib import Path
from datetime import date
from fetch_sec_evidence import Snapshot
from bs4 import BeautifulSoup

ROOT=Path(__file__).resolve().parents[1]
TAGS=['RevenueFromContractWithCustomerExcludingAssessedTax','Revenues','CostOfRevenue','CostOfGoodsAndServicesSold','SellingGeneralAndAdministrativeExpense','OperatingIncomeLoss','NetIncomeLoss','DepreciationDepletionAndAmortization','DepreciationDepletionAndAmortizationPropertyPlantAndEquipment','DepreciationAmortizationAndAccretionNet','OtherDepreciationAndAmortization','AdjustmentForAmortization','AmortizationOfIntangibleAssets','EarningsPerShareBasic','NetCashProvidedByUsedInOperatingActivities','PaymentsToAcquirePropertyPlantAndEquipment','CashAndCashEquivalentsAtCarryingValue','LongTermDebtCurrent','LongTermDebtNoncurrent','StockholdersEquity','ResearchAndDevelopmentExpense']
for ticker,acc in [('AVGO','0001730168-25-000121'),('QCOM','0000804328-25-000085')]:
    obj=json.loads((ROOT/f'evidence/sec/snapshot_2026-10-11/{ticker}/companyfacts.json').read_text())
    print('\nPUBLIC SEC',ticker,acc)
    for tag in TAGS:
        obs=[]
        for unit,rows in obj['facts']['us-gaap'].get(tag,{}).get('units',{}).items():
            for i,r in enumerate(rows):
                if r['accn']==acc and (not r.get('start') or 350<=(date.fromisoformat(r['end'])-date.fromisoformat(r['start'])).days<=380):
                    obs.append({**r,'unit':unit,'index':i})
        print(tag,obs)
    for ns,tag in [('dei','EntityCommonStockSharesOutstanding'),('us-gaap','CommonStockSharesOutstanding')]:
        rows=obj['facts'].get(ns,{}).get(tag,{}).get('units',{}).get('shares',[])
        selected=sorted([dict(r,index=i) for i,r in enumerate(rows) if r['filed']<='2026-10-11'],key=lambda r:(r['end'],r['filed']),reverse=True)[:2]
        print(ns+':'+tag,selected)

snapshot=Snapshot(ROOT/'evidence/public_market/snapshot_2026-10-11',False,'AcademicResearch/1.0 (ZeyuGu-zg2556)')
for ticker,url in [('AVGO','https://investors.broadcom.com/financial-information/stock-information/historical-price-lookup'),('QCOM','https://investor.qualcomm.com/overview/default.aspx')]:
    raw=snapshot.get(url,ticker+'.html')
    if raw:
        soup=BeautifulSoup(raw,'html.parser')
        for node in soup(['script','style']):node.decompose()
        txt=soup.get_text('\n',strip=True)
        (snapshot.directory/(ticker+'.txt')).write_text(txt,encoding='utf-8')
        print(ticker,'market text',txt[-16000:])
