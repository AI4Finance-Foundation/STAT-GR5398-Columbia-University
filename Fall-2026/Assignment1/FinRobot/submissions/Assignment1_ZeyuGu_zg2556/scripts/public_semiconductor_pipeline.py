"""Offline SEC/Yahoo adapter for AVGO/QCOM; no WRDS values or fabricated API data.

Uses genuine frozen public responses, then the pinned upstream extraction,
forecast, peer projection, sensitivity, catalyst and news implementations.
This source-availability compatibility path is separate from Option A.
"""
from __future__ import annotations
import argparse
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import asdict
from datetime import datetime, date, timezone, timedelta
import io
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch
from zoneinfo import ZoneInfo
import pandas as pd
from audit_io import sha256, write_json, utc_now
from fetch_sec_evidence import InlineFacts, annual_fact
from sec_news_fallback import extract
from snapshot_pipeline import clean, record_source_files
from independent_valuation import load_inputs,recompute,run_engine,reconciliation,assumptions_rows,write_csv,write_notebook,UPSTREAM_SHA
from experiment import setup,hashes

ROOT=Path(__file__).resolve().parents[1]
SEC=ROOT/'evidence/sec/snapshot_2026-10-11'
ASOF=datetime.fromisoformat('2026-10-11T21:10:00+00:00')
CONFIG={'AVGO':('Broadcom Inc.',['NVDA','AMD']),'QCOM':('QUALCOMM Incorporated',['AMD','INTC'])}

def read_verified(path):
    raw=path.read_bytes()
    meta=json.loads(Path(str(path)+'.meta.json').read_text())
    if meta['status']!=200 or sha256(raw)!=meta['sha256']:raise ValueError('Invalid frozen public response: '+str(path))
    return raw

def build(ticker,output,upstream):
    from modules.financial_data_processor import extract_historical_metrics_from_api_data,normalize_forecast_config,calculate_growth_and_forecasts
    from modules.market_data_api import project_ebitda_for_peers
    from modules.sensitivity_analyzer import SensitivityAnalyzer
    import modules.news_integrator as news_module
    import modules.catalyst_analyzer as catalyst_module
    company,peers=CONFIG[ticker]
    analysis=output/'analysis';analysis.mkdir(parents=True)
    source_hashes={}
    def public(path):
        raw=read_verified(path);source_hashes[str(path.relative_to(ROOT)).replace('\\','/')]=sha256(raw);return raw
    facts=json.loads(public(SEC/ticker/'companyfacts.json'))
    submissions=json.loads(public(SEC/ticker/'submissions.json'))
    recent=submissions['filings']['recent']
    filings=[dict(zip(recent,values)) for values in zip(*recent.values())]
    eligible=[r for r in filings if r['filingDate']<=ASOF.date().isoformat()]
    filing=max((r for r in eligible if r['form']=='10-K'),key=lambda r:r['reportDate'])
    filing_path=SEC/ticker/filing['accessionNumber']/filing['primaryDocument']
    inline=InlineFacts(public(filing_path).decode('utf-8'))
    ends=sorted({x['end'] for x in facts['facts']['us-gaap'].get('RevenueFromContractWithCustomerExcludingAssessedTax' if ticker=='AVGO' else 'Revenues',{})['units']['USD'] if x['accn']==filing['accessionNumber'] and x.get('start') and 350<=(date.fromisoformat(x['end'])-date.fromisoformat(x['start'])).days<=380})
    provenance=[]
    def annual(tags,end,field,unit='USD'):
        obs=annual_fact(facts,tags,filing,end=end,unit=unit)
        if not obs:raise ValueError(f'Missing SEC {ticker} {field} {end}')
        matches=inline.matches(obs)
        if not matches:raise ValueError('Companyfacts has no matching actual filing fact: '+field)
        provenance.append({'field':field,'period_end':end,'value':obs['val'],'unit':unit,
            'accession':filing['accessionNumber'],'filed':obs['filed'],'tag':obs['tag'],
            'companyfacts_pointer':f"/facts/us-gaap/{obs['tag']}/units/{unit.replace('/', '~1')}/{obs['json_index']}",
            'filing_file':str(filing_path.relative_to(ROOT)).replace('\\','/'),'ixbrl_matches':json.dumps(matches)})
        return obs['val']
    def custom_amortization(end):
        candidates=[]
        for fact in inline.facts:
            ctx=inline.contexts.get(fact.get('contextref'),{})
            if fact.get('name')=='avgo:Amortizationofintangibleandrightofuseassets' and ctx.get('end')==end and not ctx.get('dimensions') and ctx.get('start') and 350<=(date.fromisoformat(end)-date.fromisoformat(ctx['start'])).days<=380:
                val=float(fact['raw_text'].replace(',',''))*10**int(fact.get('scale','0'))
                if fact.get('sign')=='-':val=-val
                candidates.append((val,fact,ctx))
        if not candidates or len({x[0] for x in candidates})!=1:raise ValueError('Ambiguous AVGO custom amortization')
        val,fact,ctx=candidates[0]
        provenance.append({'field':'amortization_intangibles_and_ROU','period_end':end,'value':val,'unit':'USD','accession':filing['accessionNumber'],'filed':filing['filingDate'],'tag':fact['name'],'companyfacts_pointer':'Not in companyfacts; exact iXBRL fact','filing_file':str(filing_path.relative_to(ROOT)).replace('\\','/'),'ixbrl_matches':json.dumps([dict(fact,context=ctx)])})
        return val
    income=[];cash=[]
    for end in ends:
        row={'symbol':ticker,'date':end,'year':int(end[:4]),'fiscalYear':int(end[:4]),'period':'FY','reportedCurrency':'USD','filingDate':filing['filingDate'],'source':'SEC 10-K comparative annual statements'}
        for field,tags in {'revenue':['RevenueFromContractWithCustomerExcludingAssessedTax','Revenues'],'costOfRevenue':['CostOfRevenue'],'sellingGeneralAndAdministrativeExpenses':['SellingGeneralAndAdministrativeExpense'],'operatingIncome':['OperatingIncomeLoss'],'eps':['EarningsPerShareBasic'],'researchAndDevelopmentExpenses':['ResearchAndDevelopmentExpense']}.items():
            row[field]=annual(tags,end,field,'USD/shares' if field=='eps' else 'USD')
        da=(annual(['Depreciation'],end,'depreciation')+custom_amortization(end)) if ticker=='AVGO' else annual(['DepreciationDepletionAndAmortization'],end,'depreciation_and_amortization')
        row['ebitda']=row['operatingIncome']+da
        row['ebitdaDefinition']='GAAP operating income + cash-flow D&A; AVGO includes intangible/ROU amortization; operating EBITDA proxy, not issuer adjusted EBITDA or FMP EBITDA'
        income.append(row)
        ocf=annual(['NetCashProvidedByUsedInOperatingActivities'],end,'operatingCashFlow')
        capex=annual(['PaymentsToAcquirePropertyPlantAndEquipment','PaymentsToAcquireProductiveAssets'],end,'capitalExpenditure')
        cash.append({'date':end,'year':int(end[:4]),'operatingCashFlow':ocf,'capitalExpenditure':capex,'freeCashFlow':ocf-capex,'reportedCurrency':'USD','definition':'Total GAAP OCF minus capital expenditures; unlevered FCF not established'})
    write_csv(analysis/'sec_fact_provenance.csv',provenance)
    inc=pd.DataFrame(income);cf=pd.DataFrame(cash)
    inc.to_csv(analysis/'income_statement_raw_data.csv',index=False);cf.to_csv(analysis/'cash_flow_raw_data.csv',index=False)
    # No undocumented PE/EV multiples. Omitting the EV/EBITDA column invokes and
    # explicitly records the genuine baseline engine 12x +/-3x fallback.
    ratios=pd.DataFrame([{'date':end,'year':int(end[:4]),'source_status':'Historical market ratios unavailable from these public inputs'} for end in ends])
    ratios.to_csv(analysis/'ratios_raw_data.csv',index=False);ratios.to_csv(analysis/'key_metrics_raw_data.csv',index=False)
    historical=extract_historical_metrics_from_api_data({'income_statement':inc,'cash_flow':cf,'ratios':ratios,'key_metrics':ratios})
    legacy={'revenue_base_year':str(max(int(e[:4]) for e in ends))+'A','revenue_growth_assumptions':{'2025E':.05,'2026E':.06,'2027E':.04},'ebitda_growth_factor':1.05,'margin_improvement':{'Contribution Margin':.01,'EBITDA Margin':.01},'sga_margin_change':-.005}
    fc=normalize_forecast_config(historical,legacy);final=calculate_growth_and_forecasts(historical,fc)
    final.to_csv(analysis/'financial_metrics_and_forecasts.csv',index=False)
    write_json(analysis/'forecast_config_original_labels.json',legacy);write_json(analysis/'forecast_config.json',fc)
    peer_ebitda={};peer_multiples={}
    for peer in peers:
        source=ROOT/'snapshots'/('INTC_news_v2' if peer=='INTC' else peer)/'analysis/income_statement_raw_data.csv'
        source_hashes[str(source.relative_to(ROOT)).replace('\\','/')]=sha256(source.read_bytes())
        frame=pd.read_csv(source).sort_values('year');peer_ebitda[peer]=pd.Series(frame['ebitda'].values,index=frame['year'])
        source=source.parent/'key_metrics_raw_data.csv'
        source_hashes[str(source.relative_to(ROOT)).replace('\\','/')]=sha256(source.read_bytes())
        frame=pd.read_csv(source).sort_values('year');peer_multiples[peer]=pd.Series(frame['enterpriseValueOverEBITDA'].values,index=frame['year'])
    peer_ebitda[ticker]=pd.Series(inc['ebitda'].values,index=inc['year'])
    hist_peers=pd.DataFrame(peer_ebitda).sort_index();hist_peers.index.name='year'
    projected=project_ebitda_for_peers(hist_peers,1);projected.to_csv(analysis/'peer_ebitda_comparison.csv')
    multiples=pd.DataFrame(peer_multiples).sort_index();multiples[ticker]=float('nan');multiples.index.name='year';multiples.to_csv(analysis/'peer_ev_ebitda_comparison.csv')
    quarter=max((r for r in eligible if r['form']=='10-Q'),key=lambda r:r['reportDate'])
    qpath=SEC/ticker/quarter['accessionNumber']/quarter['primaryDocument'];raw=public(qpath)
    soup,fulltext=extract(raw)
    # The news substitute is an actual recent public filing, not invented press.
    tables=[table.get_text('\n',strip=True) for table in soup.find_all('table')]
    selected=next(t for t in tables if len(t)>300 and 'Net income' in t and ('Earnings per share' in t or 'Net revenue' in t or 'Revenues' in t))
    offset=fulltext.find(selected)
    if offset<0:raise ValueError('Exact filing excerpt offset unavailable')
    selected=selected[:8000]
    (analysis/'filing_news_extracted.txt').write_text(fulltext,encoding='utf-8')
    publication=datetime.fromisoformat(quarter['acceptanceDateTime'].replace('Z','+00:00'))
    if not ASOF-timedelta(days=90)<=publication<=ASOF:raise ValueError('Filing is outside frozen news window')
    article={'symbol':ticker,'title':f"{company} Form 10-Q financial disclosure for period ended {quarter['reportDate']}",'publishedDate':publication.isoformat(),'text':selected,'url':f"https://www.sec.gov/Archives/edgar/data/{submissions['cik']}/{quarter['accessionNumber'].replace('-','')}/{quarter['primaryDocument']}",'source_type':'SEC 10-Q literal income-statement excerpt; not general news','accession':quarter['accessionNumber'],'source_document_sha256':sha256(raw),'excerpt_start':offset,'excerpt_end':offset+len(selected),'excerpt_sha256':sha256(selected),'source_path':str(qpath.relative_to(ROOT)).replace('\\','/')}
    write_json(analysis/'news_article_provenance.json',[article]);write_json(analysis/'news_status.json',{'available':True,'fmp_entitled':False,'alternate_source':'Recent SEC 10-Q income-statement excerpt','days_back':90,'limitations':'One recent filing; does not cover press, product news, or all material events; attempted 8-K downloads returned 403.'})
    class FrozenTime(datetime):
        @classmethod
        def now(cls,tz=None):return ASOF.astimezone(tz) if tz else ASOF.astimezone(ZoneInfo('America/New_York')).replace(tzinfo=None)
    with patch.object(news_module,'datetime',FrozenTime),patch.object(catalyst_module,'datetime',FrozenTime):
        ni=news_module.NewsIntegrator(ticker,'',company_name=company);ni.set_news_data([article]);ni.process_news(90)
        articles=ni.get_news_for_catalyst_analysis()
        for a in articles:a.update({k:v for k,v in article.items() if k not in a})
        ca=catalyst_module.CatalystAnalyzer(ticker,'',company_name=company);cats=ca.identify_catalysts(articles)
        catalyst={'catalysts':[asdict(c) for c in cats],'top_catalysts':ca.get_top_catalysts(5),'summary':ca.generate_catalyst_summary()}
        enhanced={'ticker':ticker,'articles':articles,'summary':ni.generate_news_summary(),'source_limitation':article['source_type']}
    write_json(analysis/'company_news.json',articles);write_json(analysis/'enhanced_news.json',enhanced);write_json(analysis/'catalyst_analysis.json',catalyst)
    (analysis/'news_summary.md').write_text(enhanced['summary'],encoding='utf-8');(analysis/'catalyst_summary.md').write_text(catalyst['summary'],encoding='utf-8')
    native=SensitivityAnalyzer(final)
    sensitivity={'revenue_sensitivity':native.analyze_revenue_sensitivity().to_dict(),'margin_sensitivity':native.analyze_margin_sensitivity().to_dict(),'combined_sensitivity':native.generate_sensitivity_table().to_dict()}
    native.calculate_confidence_interval('Revenue');native.calculate_confidence_interval('EBITDA');sensitivity.update(confidence_intervals=native.confidence_intervals,summary=native.generate_sensitivity_summary())
    write_json(analysis/'sensitivity_analysis.json',clean(sensitivity));(analysis/'sensitivity_summary.md').write_text(sensitivity['summary'],encoding='utf-8')
    market_path=ROOT/f'evidence/public_market/yahoo_2026-10-11/{ticker}.json';market=json.loads(public(market_path))['chart']['result'][0]
    if market['meta']['currency']!='USD':raise ValueError('Price currency mismatch')
    quotes=market['indicators']['quote'][0];prices=[]
    for i,ts in enumerate(market['timestamp']):
        dt=datetime.fromtimestamp(ts,timezone.utc)
        if dt<=ASOF and quotes['close'][i] is not None:prices.append({'date':dt.astimezone(ZoneInfo('America/New_York')).date().isoformat(),'timestamp_utc':dt.isoformat(),'close':quotes['close'][i],'volume':quotes['volume'][i]})
    prices.sort(key=lambda p:p['date']);write_csv(analysis/'historical_prices.csv',prices)
    latest=prices[-1]
    if latest['date']!='2026-10-09':raise ValueError('Expected Oct 9 completed daily bar')
    shares_rows=facts['facts']['dei']['EntityCommonStockSharesOutstanding']['units']['shares']
    si,shares=max(((i,r) for i,r in enumerate(shares_rows) if r['filed']<=ASOF.date().isoformat() and r['end']<=ASOF.date().isoformat()),key=lambda pair:(pair[1]['end'],pair[1]['filed']))
    notes=('Subject annual data: public SEC 10-K FY2023-2025 comparative statements, as filed in FY2025 10-K; not a historical backtest. '
        'Price: Yahoo Finance Oct 9, 2026 raw daily close (USD); shares: latest SEC cover-page count dated '+shares['end']+'. '
        'Historical subject PE and EV/EBITDA, analyst grade/target, float, and forward PE are unavailable. Engine EV/EBITDA uses explicit DEFAULT 12x +/-3x, not observed multiples. '
        'Engine net debt=10% of modeled EV; FCF=GAAP OCF-capex, not established unlevered FCF; WACC=10%, growth=10% then5%, terminal=2.5%. '
        'Historic subject EBITDA=GAAP operating income+cash-flow D&A (AVGO includes ROU amortization); peer EBITDA is FMP-defined and not identical. '
        'Forecast EBITDA=gross profit-SG&A excludes R&D. Peer projection uses upstream one-step growth at global last-year+1; periods differ. '
        'News consists of one literal SEC 10-Q table, not comprehensive press coverage. Catalyst dates/probabilities and native sensitivity bands are upstream heuristics, not observations.')
    (analysis/'source_notes.txt').write_text(notes,encoding='utf-8')
    metrics={'share_price':latest['close'],'shares_outstanding':shares['val'],'target_price':None,'rating':None,'market_cap':None,'volume':None,'fwd_pe':None,'sector':None}
    write_json(output/'market_snapshot.json',{'ticker':ticker,'as_of_utc':ASOF.isoformat(),'metrics':metrics,'provenance_notes':notes,'price_bar':latest,'shares_observation':dict(shares,json_pointer=f'/facts/dei/EntityCommonStockSharesOutstanding/units/shares/{si}')})
    inputs={'ticker':ticker,'run_id':output.name,'current_price':latest['close'],'shares_outstanding':shares['val'],'currency':'USD','free_cash_flow':cash[-1]['freeCashFlow'],'free_cash_flow_period':ends[-1],'peer_data':{p:{'ev_ebitda':float(multiples[p].dropna().iloc[-1])} for p in peers},'source_metadata':{'shares':dict(shares,json_pointer=f'/facts/dei/EntityCommonStockSharesOutstanding/units/shares/{si}'),'price':{'path':str(market_path.relative_to(ROOT)).replace('\\','/'),'json_pointer':'/chart/result/0/indicators/quote/0/close','bar':latest},'free_cash_flow':'analysis/cash_flow_raw_data.csv: latest operatingCashFlow-capitalExpenditure','limitations':notes}}
    write_json(output/'valuation_inputs.json',inputs)
    start=next((p for p in prices if p['date']=='2025-10-10'),prices[0])
    write_json(analysis/'price_performance.json',{'source':'Yahoo public chart API; real saved response','actual_start':start['date'],'actual_end':latest['date'],'start_close':start['close'],'end_close':latest['close'],'price_return_pct':(latest['close']/start['close']-1)*100,'adjustment_caveat':'Raw close field; not adjclose, not dividend-reinvested return; split treatment unverified'})
    write_json(output/'fallback_register.json',[{'field':'valuation.target_multiple','value':12,'status':'EXPLICIT_UPSTREAM_DEFAULT_NOT_OBSERVATION'},{'field':'valuation.multiple_standard_deviation','value':3,'status':'EXPLICIT_UPSTREAM_DEFAULT_NOT_OBSERVATION'},{'field':'all_source_and_model_caveats','value':notes,'status':'DISCLOSED'}])
    write_json(output/'missing_data.json',[{'field':f,'status':'unavailable_no_substitute'} for f in ['historical_PE','historical_EV_EBITDA','FMP_rating','FMP_consensus_target','current_share_count','broad_news_feed']])
    write_json(analysis/'analysis_summary.json',{'company_ticker':ticker,'company_name':company,'data_source':'Public SEC annual financials + Yahoo price; FMP peer snapshot; no WRDS','analysis_date':ASOF.isoformat(),'years_analyzed':len(ends),'available_years':[e[:4]+'A' for e in ends],'forecast_years':list(fc['revenue_growth_assumptions']),'forecast_config':fc,'peer_tickers':peers,'enhanced_news_enabled':True,'sensitivity_analysis_enabled':True,'catalyst_analysis_enabled':True,'news_days_back':90,'source_compatibility_departure':notes})
    context=load_inputs(analysis,output/'valuation_inputs.json');independent=recompute(context);engine=run_engine(context,upstream/'finrobot_equity/core/src');rows=reconciliation(context,independent,engine)
    val=output/'valuation';val.mkdir()
    write_json(val/'engine_valuation.json',clean(engine));write_json(val/'independent_valuation.json',{'input_sources':context['sources'],'independent_results':independent})
    write_csv(val/'valuation_reconciliation.csv',rows);write_csv(val/'valuation_assumptions.csv',assumptions_rows(context,independent));write_csv(val/'dcf_cash_flows.csv',independent['DCF']['cash_flows']);write_notebook(val/'independent_valuation.ipynb',context)
    if any(r['verdict']!='MATCH' for r in rows):raise ValueError('Valuation reconciliation failed')
    return source_hashes

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--upstream',type=Path,required=True);parser.add_argument('--ticker',choices=CONFIG,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    setup(args.upstream)
    if args.output.exists() and any(args.output.iterdir()):raise FileExistsError('Use a new output; immutable snapshots are never overwritten')
    args.output.mkdir(parents=True,exist_ok=True);log=io.StringIO();start=time.perf_counter()
    manifest={'run_id':args.output.name,'status':'running','upstream_sha':UPSTREAM_SHA,'created_utc':utc_now(),'as_of_utc':ASOF.isoformat(),'model_calls':0,'data_source':'SEC/Yahoo/FMP cached peers; WRDS excluded','adapter_sha256':sha256(Path(__file__).read_bytes()),'source_sha256':record_source_files(args.upstream)}
    try:
        with redirect_stdout(log),redirect_stderr(log):manifest['raw_input_hashes']=build(args.ticker,args.output.resolve(),args.upstream.resolve())
        manifest['status']='frozen'
    except Exception as exc:manifest.update(status='failed',error_type=type(exc).__name__,error=str(exc))
    finally:
        (args.output/'pipeline.log').write_text(log.getvalue(),encoding='utf-8');manifest['runtime_seconds']=time.perf_counter()-start;manifest['file_hashes']=hashes(args.output);write_json(args.output/'manifest.json',manifest)
    print(json.dumps({k:manifest[k] for k in ('run_id','status','runtime_seconds')}))
    if manifest['status']!='frozen':raise SystemExit(manifest['error'])

if __name__=='__main__':main()
