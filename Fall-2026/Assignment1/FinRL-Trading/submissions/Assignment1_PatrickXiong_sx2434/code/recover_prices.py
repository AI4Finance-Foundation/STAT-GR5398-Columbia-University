"""Bounded public-price recovery with explicit acceptance/rejection evidence.

Current SEC/Yahoo identities and overlapping adjusted returns are checked. This
is not a certification of historical vintages or missing terminal proceeds.
"""
import argparse,json,hashlib,time,re,concurrent.futures
from pathlib import Path
import pandas as pd
import numpy as np
import requests

ALIASES={
 'PKI':('RVTY','https://ir.revvity.com/news/investor-news/news-details/2023/Launching-Revvity-A-Scientific-Solutions-Company-Powering-Innovation-from-Discovery-to-Cure/default.aspx'),
 'ABC':('COR','https://investor.cencora.com/news/news-details/2023/Cencora-Reports-Fiscal-2023-Fourth-Quarter-and-Year-End-Results/default.aspx'),
 'ANTM':('ELV','https://www.elevancehealth.com/newsroom/anthem-inc-shareholders-approve-corporate-rebranding-new-name'),
 'BLL':('BALL','https://www.ball.com/newswire/article/124123/ball-board-declares-quarterly-dividend-stock-ticker-symbol-changing-to-ball'),
}
CAUTION={'AIV':'2020 real-estate spin-off; distribution treatment not independently established',
 'BBWI':'L Brands split and rename; pre-split issuer/adjustment continuity needs review',
 'CZR':'2020 merger and symbol reuse; earlier history may represent predecessor',
 'FISV':'ticker transition history not independently verified',
 'UA':'2016 share-class creation changes historical ticker meaning',
 'PARA':'2025 merger and terminal consideration not established',
 'SBNY':'failed-bank/OTC history incomplete; cannot use returned fragment as full history'}

def get(url,params=None):
 for attempt in range(3):
  try:
   r=requests.get(url,params=params,headers={'User-Agent':'Mozilla/5.0'},timeout=25);r.raise_for_status();return r.json()
  except Exception:
   if attempt==2:raise
   time.sleep(1+attempt)

def main(a):
 a.output.mkdir(parents=True,exist_ok=True);(a.output/'raw').mkdir(exist_ok=True);(a.output/'accepted').mkdir(exist_ok=True)
 secfile=a.output/'sec_company_tickers.json'
 sec=json.loads(secfile.read_text()) if secfile.exists() else get('https://www.sec.gov/files/company_tickers.json')
 secfile.write_text(json.dumps(sec))
 sec={x['ticker']:x for x in sec.values()}
 inv=pd.read_csv(a.probe/'recovery_inventory.csv');records=[];accepted=[]
 expected=pd.read_csv(a.cache/'prepared/daily.csv',usecols=['datadate']).datadate.unique();expected=pd.DatetimeIndex(sorted(expected))
 def one(row):
  t=row.original_ticker;symbol=ALIASES[t][0] if t in ALIASES else row.provider_symbol
  rec=dict(original_ticker=t,provider_symbol=symbol,status='rejected',reason='',sec_cik='',sec_title='',provider_name='',source_url=f'https://query1.finance.yahoo.com/v8/finance/chart/{symbol}',identity_source='https://www.sec.gov/files/company_tickers.json',alias_evidence=ALIASES[t][1] if t in ALIASES else '',terminal_proceeds_verified=False)
  if t in CAUTION:rec['reason']=CAUTION[t];return rec,None
  if row.rows==0 and t not in ALIASES:rec['reason']='no cached public history; merger/delisting/rename unresolved';return rec,None
  if symbol not in sec:rec['reason']='no exact current SEC symbol identity; not silently remapped';return rec,None
  try:
   rawfile=a.output/'raw'/f'{symbol}.json'
   js=json.loads(rawfile.read_text()) if rawfile.exists() else get(rec['source_url'],{'period1':1434326400,'period2':1759276800,'interval':'1d','events':'div,splits'})
   (a.output/'raw'/f'{symbol}.json').write_text(json.dumps(js));data=js['chart']['result'][0];meta=data['meta'];q=data['indicators']['quote'][0]
   dates=pd.to_datetime(data['timestamp'],unit='s',utc=True).tz_convert(meta.get('exchangeTimezoneName','America/New_York')).tz_localize(None).normalize()
   frame=pd.DataFrame({'adj_close':data['indicators']['adjclose'][0]['adjclose'],'close':q['close']},index=dates).sort_index().dropna()
   frame.index.name='datadate';frame=frame.loc[~frame.index.duplicated()]
   company=sec[symbol];rec.update(sec_cik=company['cik_str'],sec_title=company['title'],provider_name=meta.get('longName',meta.get('shortName','')),first_date=str(frame.index.min().date()),last_date=str(frame.index.max().date()),rows=len(frame),adjustment_source='Yahoo adjclose (not raw close)',dividend_events=len(data.get('events',{}).get('dividends',{})),split_events=len(data.get('events',{}).get('splits',{})))
   if meta['symbol']!=symbol or meta.get('currency')!='USD' or meta.get('instrumentType')!='EQUITY':raise ValueError('provider symbol/currency/instrument mismatch')
   stop={'INC','CORP','CORPORATION','COMPANY','CO','CLASS','THE','PLC','GROUP','LTD','LIMITED','HOLDINGS','HOLDING','A','B'}
   tokens=lambda s:set(re.findall(r'[A-Z]+',s.upper().replace("'",'')))-stop
   if not tokens(company['title'])&tokens(rec['provider_name']):raise ValueError('SEC/provider issuer-name disagreement')
   if (frame[['adj_close','close']]<=0).any().any():raise ValueError('nonpositive price')
   wanted=expected[(expected>=frame.index.min())&(expected<=pd.Timestamp('2025-09-30'))]
   if not frame.index.intersection(wanted).equals(wanted):raise ValueError('missing in-life market session')
   if frame.index.max()!=pd.Timestamp('2025-09-30'):raise ValueError('does not reach fixed experiment endpoint')
   prior=a.probe/(row.provider_symbol+'.csv')
   if prior.exists():
    old=pd.read_csv(prior,index_col=0,parse_dates=True)['Close'];joined=pd.concat([old,frame.adj_close],axis=1).dropna();dif=joined.pct_change().diff(axis=1).iloc[:,1].abs().dropna();rec['overlap_days']=len(joined);rec['max_overlap_return_difference']=float(dif.max())
    if len(joined)<200 or dif.max()>1e-5:raise ValueError('cached/new adjusted-return disagreement')
   else:rec['overlap_days']=0;rec['max_overlap_return_difference']=None
   rec['status']='accepted';rec['reason']='exact current issuer and full in-life adjusted series; '+('official same-issuer rename' if t in ALIASES else 'unchanged symbol, no identified identity conflict')
   rec['historical_identity_scope']='current SEC issuer + provider historical series; not independent archival certification'
   rec['sha256']=hashlib.sha256((a.output/'raw'/f'{symbol}.json').read_bytes()).hexdigest()
   frame.to_csv(a.output/'accepted'/f'{t}.csv');return rec,frame
  except Exception as e:rec['reason']=str(e);return rec,None
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  for rec,frame in pool.map(one,inv.itertuples(index=False)):
   records.append(rec)
   if frame is not None:accepted.append(rec['original_ticker'])
 pd.DataFrame(records).to_csv(a.output/'recovery_decisions.csv',index=False)
 aliases={k:v[0] for k,v in ALIASES.items() if k in accepted};aliases.update({'BRK.B':'BRK-B','BF.B':'BF-B'})
 (a.output/'security_aliases.json').write_text(json.dumps(aliases,indent=2))
 manifest={'created_at':pd.Timestamp.now(tz='America/New_York').isoformat(),'accepted':accepted,'rejected':[x['original_ticker'] for x in records if x['status']!='accepted'],'all_historical_issuers_certified':False,'delisting_coverage_complete':False,'method':'bounded current issuer checks, official renames, complete in-life prices, raw adjclose and prior-return consistency; no imputed terminal proceeds'}
 manifest['accepted_csv_sha256']={t:hashlib.sha256((a.output/'accepted'/f'{t}.csv').read_bytes()).hexdigest() for t in accepted}
 (a.output/'recovery_manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser()
 for n in ['cache','probe','output']:ap.add_argument('--'+n,type=Path,required=True)
 main(ap.parse_args())
