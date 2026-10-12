from pathlib import Path
import json,re
import pandas as pd
from bs4 import BeautifulSoup
ROOT=Path(__file__).resolve().parents[1]
lineage=pd.read_csv(ROOT/'audit/sec_fact_lineage.csv')
metrics=['revenue','cost_of_revenue','sga','gross_profit','eps','net_income','operating_income','cfo','capex','cash']
# Independently read from NVIDIA annual report printed pp. 51, 53, 55.
nvda={'revenue':215938e6,'cost_of_revenue':62475e6,'sga':4579e6,'gross_profit':153463e6,
      'eps':4.90,'net_income':120067e6,'operating_income':130387e6,'cfo':102718e6,'capex':6042e6,'cash':10605e6}
source='https://s201.q4cdn.com/141608511/files/doc_financials/2026/ar/2026-annual-report-web-Hyperlinks.pdf'
rows=[]
for metric in metrics:
 l=lineage[(lineage.ticker=='NVDA')&(lineage.period_end=='2026-01-25')&(lineage.metric==metric)].iloc[0]
 rows.append(dict(ticker='NVDA',metric=metric,sec_value=l.value,filing_value=nvda[metric],delta=l.value-nvda[metric],
    source=source,locator='Printed pp. 51 (income), 53 (balance sheet), 55 (cash flow)',
    method='Manual reading of annual-report financial tables, programmatic comparison'))
soup=BeautifulSoup((ROOT/'data/sec/AMD_10K.html').read_text(),'html.parser')
contexts={x['id']:x for x in soup.find_all('xbrli:context')}
for metric in metrics:
 l=lineage[(lineage.ticker=='AMD')&(lineage.period_end=='2025-12-27')&(lineage.metric==metric)].iloc[0]
 matches=[]
 for x in soup.find_all('ix:nonfraction',attrs={'name':'us-gaap:'+l.tag}):
  c=contexts[x['contextref']]
  if c.find('xbrli:segment'):continue
  end=c.find('xbrli:enddate') or c.find('xbrli:instant');start=c.find('xbrli:startdate')
  if not end or end.text!='2025-12-27':continue
  if metric not in ['cash'] and (not start or start.text!='2024-12-29'):continue
  v=float(x.get_text().replace(',','').replace('$','').replace('(','').replace(')','').replace('\u2212','-'))*10**int(x.get('scale',0))
  if x.get('sign')=='-':v=-v
  matches.append((v,x['id'],x['contextref']))
 assert matches,metric
 v,identifier,context=matches[0]
 rows.append(dict(ticker='AMD',metric=metric,sec_value=l.value,filing_value=v,delta=l.value-v,
     source='https://ir.amd.com/financial-information/sec-filings/content/0000002488-26-000018/amd-20251227.htm',
     locator=f'Inline XBRL fact {identifier}, context {context}',method='Original issuer-hosted 10-K consolidated inline XBRL'))
pd.DataFrame(rows).to_csv(ROOT/'audit/filing_crosscheck.csv',index=False)
assert all(abs(r['delta'])<.01 for r in rows)
print(f'{len(rows)} original-filing cross-checks matched; FMP-vs-filing audit remains pending.')
