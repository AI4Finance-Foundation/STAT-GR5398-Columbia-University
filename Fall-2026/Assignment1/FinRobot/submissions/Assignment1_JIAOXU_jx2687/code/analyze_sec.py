"""SEC-backed companion experiment; this does NOT impersonate an FMP/LLM run."""
from pathlib import Path
import sys, json, math, hashlib, time, platform, argparse
from datetime import date
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CUTOFF = '2026-10-11'
COMPANIES = {'NVDA':(1045810,'NVIDIA Corporation'), 'AMD':(2488,'Advanced Micro Devices'),
             'INTC':(50863,'Intel Corporation'), 'AVGO':(1730168,'Broadcom Inc.'),
             'QCOM':(804328,'QUALCOMM Incorporated')}
TAGS = {
 'revenue':['RevenueFromContractWithCustomerExcludingAssessedTax','Revenues','SalesRevenueNet'],
 'net_income':['NetIncomeLoss','ProfitLoss'], 'operating_income':['OperatingIncomeLoss'],
 'gross_profit':['GrossProfit'], 'cost_of_revenue':['CostOfRevenue','CostOfGoodsAndServicesSold'],
 'sga':['SellingGeneralAndAdministrativeExpense'], 'rd':['ResearchAndDevelopmentExpense'],
 'cfo':['NetCashProvidedByUsedInOperatingActivities'],
 'capex':['PaymentsToAcquirePropertyPlantAndEquipment','PaymentsToAcquireProductiveAssets'],
 'da':['DepreciationDepletionAndAmortization','DepreciationDepletionAndAmortizationPropertyPlantAndEquipment'],
 'cash':['CashAndCashEquivalentsAtCarryingValue'], 'assets':['Assets'],
 'liabilities':['Liabilities'], 'equity':['StockholdersEquity'],
 'debt_current':['LongTermDebtCurrent'], 'debt_noncurrent':['LongTermDebtNoncurrent'],
 'debt_total':['LongTermDebt'], 'shares':['CommonStockSharesOutstanding'],
 'eps':['EarningsPerShareDiluted']}
INSTANT = {'cash','assets','liabilities','equity','debt_current','debt_noncurrent','debt_total','shares'}


def candidates(d,metric,end=None,asof=CUTOFF):
    unit='shares' if metric=='shares' else 'USD/shares' if metric=='eps' else 'USD'
    result=[]
    for rank,tag in enumerate(TAGS[metric]):
        for f in d['facts']['us-gaap'].get(tag,{}).get('units',{}).get(unit,[]):
            if f.get('form') not in ['10-K','10-K/A'] or f['filed']>asof or f['end']>asof: continue
            if end and f['end']!=end: continue
            if metric not in INSTANT:
                if not f.get('start'): continue
                duration=(date.fromisoformat(f['end'])-date.fromisoformat(f['start'])).days
                if not 330 <= duration <= 380: continue
            result.append(dict(f,tag=tag,unit=unit,tag_rank=rank))
    return result


def pick(d,metric,end,asof=CUTOFF,earliest=False):
    c=candidates(d,metric,end,asof)
    if not c: return None
    # Highest-priority available tag, latest restatement for current analysis.
    if earliest: return sorted(c,key=lambda x:(x['filed'],x['tag_rank'],x['accn']))[0]
    rank=min(x['tag_rank'] for x in c);c=[x for x in c if x['tag_rank']==rank]
    return sorted(c,key=lambda x:(x['filed'],x['accn']))[0 if earliest else -1]


def filing_link(cik,f):
    return f"https://www.sec.gov/Archives/edgar/data/{cik}/{f['accn'].replace('-','')}/{f['accn']}-index.html"


def independent_dcf(fcf,shares,wacc=.1,g1=.1,g2=.05,terminal=.025,years=10):
    flows=[fcf*(1+g1)**min(i,5)*(1+g2)**max(i-5,0) for i in range(1,years+1)]
    pv=sum(v/(1+wacc)**i for i,v in enumerate(flows,1))
    tv=flows[-1]*(1+terminal)/(wacc-terminal)/(1+wacc)**years
    return {'ev':pv+tv,'pv_cashflows':pv,'pv_terminal':tv,'unfloored_price':.9*(pv+tv)/shares,
            'target_price':max(0,.9*(pv+tv)/shares),'fcf':fcf,'shares':shares}


def main(upstream):
    started=time.perf_counter()
    sys.path.insert(0,str(Path(upstream).resolve()/'finrobot_equity/core/src'))
    from modules.financial_data_processor import (extract_historical_metrics_from_api_data,
         calculate_growth_and_forecasts,build_data_driven_forecast_config)
    from modules.valuation_engine import ValuationEngine
    from modules.sensitivity_analyzer import SensitivityAnalyzer
    for p in ['analysis','audit','charts','reports']: (ROOT/p).mkdir(exist_ok=True)
    lineage=[];summary=[];backtest=[];valuation=[];dataaudit=[];qa=[];fcfaudit=[]
    for ticker,(cik,name) in COMPANIES.items():
        d=json.loads((ROOT/'data/sec'/f'{ticker}.json').read_text())
        years=sorted(set(x['end'] for x in candidates(d,'revenue')))
        latest=years[-1]; history=[];observations={}
        target=ROOT/'analysis'/ticker;target.mkdir(exist_ok=True)
        for end in years[-7:]:
            vals={};facts={}
            for metric in TAGS:
                f=pick(d,metric,end); facts[metric]=f; vals[metric]=float(f['val']) if f else None
                if f: lineage.append(dict(ticker=ticker,metric=metric,period_end=end,value=f['val'],unit=f['unit'],
                     tag=f['tag'],period_start=f.get('start',''),filed=f['filed'],accession=f['accn'],
                     source_url=filing_link(cik,f),json_pointer=f"facts/us-gaap/{f['tag']}/units/{f['unit']}"))
            if vals['gross_profit'] is None and vals['cost_of_revenue'] is not None:
                vals['gross_profit']=vals['revenue']-vals['cost_of_revenue']
            ebitda_proxy=None if vals['da'] is None or vals['operating_income'] is None else vals['operating_income']+vals['da']
            history.append(dict(date=end,year=int(end[:4]),revenue=vals['revenue'],costOfRevenue=vals['cost_of_revenue'],
               grossProfit=vals['gross_profit'],sellingGeneralAndAdministrativeExpenses=vals['sga'],
               ebitda=ebitda_proxy,eps=vals['eps']))
            observations[end]=(vals,facts)
        historical=extract_historical_metrics_from_api_data({'income_statement':pd.DataFrame(history)})
        historical.to_csv(target/'sec_historical_metrics.csv',index=False)
        config=build_data_driven_forecast_config(historical)
        (target/'forecast_assumptions.json').write_text(json.dumps(config,indent=2))
        improved=calculate_growth_and_forecasts(historical,config)
        improved.to_csv(target/'financial_metrics_and_forecasts.csv',index=False)
        last=f'{int(latest[:4])}A'
        baseline_config=dict(revenue_base_year=last,revenue_growth_assumptions={'2025E':.05,'2026E':.06,'2027E':.04},
               margin_improvement={'Contribution Margin':.01,'EBITDA Margin':.01},sga_margin_change=-.005)
        baseline=calculate_growth_and_forecasts(historical,baseline_config)
        baseline.to_csv(target/'upstream_default_forecasts.csv',index=False)
        analyzer=SensitivityAnalyzer(improved)
        analyzer.analyze_revenue_sensitivity().to_csv(target/'upstream_revenue_sensitivity.csv',index=False)
        analyzer.analyze_margin_sensitivity().to_csv(target/'upstream_margin_sensitivity.csv',index=False)
        vals,facts=observations[latest]
        assert vals['revenue']>0 and vals['shares']>0 and vals['capex'] is not None and vals['cfo'] is not None
        fcf=vals['cfo']-vals['capex'];shares=vals['shares']
        v=ValuationEngine({'free_cash_flow':fcf,'shares_outstanding':shares}).calculate_dcf_valuation()
        own=independent_dcf(fcf,shares)
        assert math.isclose(v.target_price,own['target_price'],rel_tol=1e-10,abs_tol=1e-8)
        valuation.append(dict(ticker=ticker,period_end=latest,base_cashflow=fcf,shares=shares,
          engine_price=v.target_price,recomputed_price=own['target_price'],delta=v.target_price-own['target_price'],
          ev=own['ev'],terminal_share=own['pv_terminal']/own['ev'] if own['ev'] else None,
          wacc=.1,growth_first5=.1,growth_next5=.05,terminal_growth=.025,projection_years=10,
          assumed_net_debt=own['ev']*.1,status='Arithmetic reconciled; economic valuation not validated',
          limitation='CFO-capex is levered cashflow proxy, not FCFF; 10% EV net debt is synthetic; no current quote'))
        sensitivity=[]
        for w in [.08,.09,.1,.11,.12]:
            for g in [.015,.025,.035]:
                r=independent_dcf(fcf,shares,wacc=w,terminal=g)
                sensitivity.append(dict(wacc=w,terminal_growth=g,price=r['target_price'],unfloored_price=r['unfloored_price']))
        pd.DataFrame(sensitivity).to_csv(target/'full_dcf_sensitivity.csv',index=False)
        # Revenue +/-5% and EBITDA-margin +/-2 percentage points: explicitly a 12x EV/EBITDA scenario.
        lastf=list(config['revenue_growth_assumptions'])[-1]
        finalrev=float(improved.loc[improved.metrics=='Revenue',lastf].iloc[0])
        finalmargin=improved.loc[improved.metrics=='EBITDA Margin',lastf].iloc[0]
        if isinstance(finalmargin,str) and '%' in finalmargin:
            m=float(finalmargin.strip('%'))/100
            grid=[dict(revenue_delta=dr,margin_delta=dm,forecast_year=lastf,
               scenario_price=max(0,.9*12*finalrev*(1+dr)*(m+dm)/shares),
               assumptions='12x multiple; synthetic net debt=10% EV; forecast EBITDA excludes R&D')
               for dr in [-.05,0,.05] for dm in [-.02,0,.02]]
            pd.DataFrame(grid).to_csv(target/'target_price_sensitivity.csv',index=False)
        ni_margin=vals['net_income']/vals['revenue']
        priorend=years[-2];prior=pick(d,'revenue',priorend)
        yoy=vals['revenue']/prior['val']-1
        summary.append(dict(ticker=ticker,name=name,period_end=latest,annual_filed=facts['revenue']['filed'],
            revenue_bn=vals['revenue']/1e9,revenue_yoy=yoy,net_margin=ni_margin,fcf_bn=fcf/1e9,
            cfo_bn=vals['cfo']/1e9,capex_bn=vals['capex']/1e9,cagr=config['growth_used'],
            mechanical_dcf=v.target_price,source=filing_link(cik,facts['revenue']),
            peers=','.join(x for x in COMPANIES if x!=ticker),
            latest_quarter_end=max(x['end'] for x in d['facts']['us-gaap'][facts['revenue']['tag']]['units']['USD'] if x['filed']<=CUTOFF)))
        for metric in ['revenue','cost_of_revenue','sga','gross_profit','eps']:
            mapped={'revenue':'Revenue','cost_of_revenue':'Cost of Operations','sga':'SG&A','gross_profit':'Contribution Profit','eps':'EPS'}[metric]
            pv=historical.loc[historical.metrics==mapped,last].iloc[0]
            f=facts[metric]
            if f:
                dataaudit.append(dict(ticker=ticker,metric=mapped,year=last,pipeline_value=pv,filing_value=f['val'],
                    delta=float(pv)-f['val'] if pd.notna(pv) else None,verdict='SEC-adapter check; FMP comparison pending',
                    explanation='Executed upstream extractor with SEC-normalized input; not independent FMP verification',source_url=filing_link(cik,f)))
        for metric in ['net_income','operating_income','cfo','capex','cash']:
            f=facts[metric]
            dataaudit.append(dict(ticker=ticker,metric=metric,year=last,pipeline_value=None,
                filing_value=f['val'] if f else None,delta=None,verdict='FMP pipeline value unavailable',
                explanation='SEC reference populated; no claim of agreement',source_url=filing_link(cik,f) if f else ''))
        fcfaudit.append(dict(ticker=ticker,end=latest,cfo=vals['cfo'],capex=vals['capex'],fcf=fcf,
                            cfo_source=filing_link(cik,facts['cfo']),capex_tag=facts['capex']['tag']))
        # Rolling-origin evaluation: earliest original 10-K target, only filings available at training origin.
        for i in range(3,len(years)):
            end=years[i];trainend=years[i-1]
            targetfact=pick(d,'revenue',end,earliest=True)
            origin=pick(d,'revenue',trainend,earliest=True)['filed']
            trains=[pick(d,'revenue',e,asof=origin) for e in years[i-3:i]]
            if any(x is None for x in trains): continue
            if any(int(b['end'][:4])-int(a['end'][:4])!=1 for a,b in zip(trains,trains[1:])):continue
            rev=[x['val'] for x in trains];raw=(rev[-1]/rev[0])**.5-1
            growth=float(np.clip(raw,-.5,.5)); forecast=rev[-1]*(1+growth); default=rev[-1]*1.05
            actual=targetfact['val']
            backtest.append(dict(ticker=ticker,forecast_end=end,origin_date=origin,
               training_ends=';'.join(x['end'] for x in trains),training_accessions=';'.join(x['accn'] for x in trains),
               max_training_filed=max(x['filed'] for x in trains),target_filed=targetfact['filed'],target=actual,
               baseline_5pct=default,improved_cagr=forecast,raw_growth=raw,growth_used=growth,
               baseline_ape=abs(default-actual)/actual,improved_ape=abs(forecast-actual)/actual,
               source_url=filing_link(cik,targetfact)))
        qa.append(dict(ticker=ticker,latest_annual=latest,lineage_present=True,dcf_reconciled=True,
             forecast_years_future=all(int(y[:-1])>int(latest[:4]) for y in config['revenue_growth_assumptions']),
             actual_report_type='SEC companion, not FinRobot eight-agent report',agent_status='not_run_missing_keys'))
    frames={'sec_fact_lineage':lineage,'coverage_summary':summary,'forecast_backtest':backtest,
       'valuation_audit':valuation,'data_audit':dataaudit,'fcf_calculation':fcfaudit,'qa':qa}
    for key,data in frames.items():pd.DataFrame(data).to_csv(ROOT/'audit'/f'{key}.csv',index=False)
    b=pd.DataFrame(backtest);assert (b.max_training_filed<=b.origin_date).all() and (b.target_filed>b.origin_date).all()
    performance=b.groupby('ticker')[['baseline_ape','improved_ape']].mean().reset_index()
    performance.to_csv(ROOT/'audit/forecast_scorecard.csv',index=False)
    sections=['tagline','company_overview','investment_overview','valuation_overview','risks','competitor_analysis','major_takeaways','news_summary']
    pd.DataFrame([dict(ticker=t,section=s,total_claims=None,supported=None,unsupported=None,contradicted=None,
         grounding_rate=None,status='not_run_missing_OpenAI_and_FMP') for t in ['NVDA','AMD','INTC'] for s in sections]).to_csv(ROOT/'audit/grounding_status.csv',index=False)
    pd.DataFrame([dict(ticker='NVDA',run=i,rating=None,target_price=None,top_3_risks=None,status='not_run_missing_keys') for i in [1,2,3]]).to_csv(ROOT/'audit/stability_status.csv',index=False)
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'data/sec').glob('*.json')}
    manifest=dict(asof=CUTOFF,python=sys.version,platform=platform.platform(),runtime_seconds=time.perf_counter()-started,
       model_used=None,llm_calls=0,llm_cost_usd=0,fmp_calls=0,status='API-independent companion completed; course end-to-end pipeline blocked',
       source_hashes=hashes,baseline_mape=float(b.baseline_ape.mean()),improved_mape=float(b.improved_ape.mean()),backtest_n=len(b))
    (ROOT/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({k:v for k,v in manifest.items() if k!='source_hashes'},indent=2))
    print(pd.DataFrame(summary).to_string(index=False));print(performance.to_string(index=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--upstream',required=True);main(parser.parse_args().upstream)
