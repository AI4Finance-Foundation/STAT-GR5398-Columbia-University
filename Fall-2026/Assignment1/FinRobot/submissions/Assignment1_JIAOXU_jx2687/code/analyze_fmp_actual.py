from pathlib import Path
import sys,json,math
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'upstream/finrobot_equity/core/src'))
from modules.financial_data_processor import build_data_driven_forecast_config,calculate_growth_and_forecasts
from modules.sensitivity_analyzer import SensitivityAnalyzer
from modules.valuation_engine import ValuationEngine
from analyze_sec import independent_dcf
TICKERS=['NVDA','AMD','INTC','AAPL','MSFT']
rows=[];forecasts=[];values=[];audits=[]
lineage=pd.read_csv(ROOT/'audit/sec_fact_lineage.csv')
for ticker in TICKERS:
 path=ROOT/'actual_runs/baseline'/ticker/'analysis'
 income=pd.read_csv(path/'income_statement_raw_data.csv').sort_values('date',ascending=False)
 cf=pd.read_csv(path/'cash_flow_raw_data.csv').sort_values('date',ascending=False)
 bal=pd.read_csv(path/'balance_sheet_raw_data.csv').sort_values('date',ascending=False)
 metrics=pd.read_csv(path/'financial_metrics_and_forecasts.csv');cols=[c for c in metrics if c.endswith('A')]
 hist=metrics[['metrics']+cols].copy()
 numeric_rows=~hist.metrics.str.contains('Margin|Growth',regex=True)
 for col in cols:hist.loc[numeric_rows,col]=pd.to_numeric(hist.loc[numeric_rows,col],errors='coerce')
 cfg=build_data_driven_forecast_config(hist)
 forecast=calculate_growth_and_forecasts(hist,cfg)
 dest=ROOT/'actual_runs/improved'/ticker/'analysis';dest.mkdir(parents=True,exist_ok=True)
 forecast.to_csv(dest/'financial_metrics_and_forecasts.csv',index=False);(dest/'forecast_assumptions.json').write_text(json.dumps(cfg,indent=2))
 sensitivity=SensitivityAnalyzer(forecast);sensitivity.analyze_revenue_sensitivity().to_csv(dest/'revenue_sensitivity.csv',index=False)
 sensitivity.analyze_margin_sensitivity().to_csv(dest/'margin_sensitivity.csv',index=False)
 latest=income.iloc[0];end=latest['date'];year=str(end)[:4]+'A'
 cfo=cf[cf.date==end].iloc[0];balance=bal[bal.date==end].iloc[0]
 shares=float(latest.weightedAverageShsOutDil);fcf=float(cfo.freeCashFlow)
 result=ValuationEngine({'free_cash_flow':fcf,'shares_outstanding':shares}).calculate_dcf_valuation()
 own=independent_dcf(fcf,shares);assert abs(result.target_price-own['target_price'])<1e-8
 values.append(dict(ticker=ticker,end=end,base_fcf=fcf,share_basis='FY weighted-average diluted shares; not current outstanding',shares=shares,
   engine_price=result.target_price,recomputed_price=own['target_price'],difference=result.target_price-own['target_price'],
   actual_balance_net_debt=balance.netDebt,synthetic_net_debt=.1*own['ev'],wacc=.1,terminal_growth=.025,
   source='FMP stable statements, preserved raw CSVs',status='Arithmetic verified; mechanical valuation only'))
 rows.append(dict(ticker=ticker,annual_end=end,fiscal_year=latest.fiscalYear,revenue=latest.revenue,
  net_income=latest.netIncome,ebitda=latest.ebitda,fcf=fcf,model=None,llm_calls=0,
  revenue_growth_used=cfg['growth_used'],peers_present=(path/'peer_ev_ebitda_comparison.csv').exists(),news_access='HTTP402',
  source='FMP stable endpoints; raw data saved'))
 for method,frame in [('baseline',metrics),('improved',forecast)]:
  for y in [c for c in frame if c.endswith('E')]:forecasts.append(dict(ticker=ticker,method=method,forecast_year=y,strictly_future=int(y[:-1])>int(end[:4])))
 if ticker in ['NVDA','AMD']:
  facts=lineage[(lineage.ticker==ticker)&(lineage.period_end==end)]
  fact={x.metric:float(x.value) for _,x in facts.iterrows()}
  priorends=sorted(lineage[(lineage.ticker==ticker)&(lineage.metric=='revenue')].period_end.unique())
  prior=facts.iloc[0].source_url
  prev=lineage[(lineage.ticker==ticker)&(lineage.metric=='revenue')&(lineage.period_end==priorends[-2])].iloc[0].value
  if ticker=='AMD':
   from bs4 import BeautifulSoup
   filing=BeautifulSoup((ROOT/'data/sec/AMD_10K.html').read_text(),'html.parser')
   context=filing.find(lambda t:t.name and t.name.lower().endswith('context') and t.get('id')=='c-1')
   assert '2025-12-27' in context.get_text()
   amounts=[]
   for tag in ['us-gaap:OtherDepreciationAndAmortization','us-gaap:AdjustmentForAmortization']:
    item=filing.find(lambda t:t.get('name')==tag and t.get('contextref')=='c-1')
    amounts.append(float(item.get_text().replace(',',''))*10**int(item.get('scale','0')))
   fact['da']=sum(amounts)
  reference={'Revenue':fact['revenue'],'Cost of Operations':fact['cost_of_revenue'],'SG&A':fact['sga'],
   'Contribution Profit':fact['gross_profit'],'Contribution Margin':100*fact['gross_profit']/fact['revenue'],
   'SG&A Margin':100*fact['sga']/fact['revenue'],'Revenue Growth':100*(fact['revenue']/prev-1),
   'EPS':4.93 if ticker=='NVDA' else 2.67,
   'EBITDA':fact['operating_income']+fact['da'],'EBITDA Margin':100*(fact['operating_income']+fact['da'])/fact['revenue']}
  for metric,ref in reference.items():
   val=metrics.loc[metrics.metrics==metric,year].iloc[0];numeric=float(str(val).replace('%',''));delta=numeric-ref
   rounding='Margin/Growth' in metric or 'Margin' in metric or metric=='Revenue Growth'
   tol=.051 if rounding else .01
   verdict='Matched within published rounding' if abs(delta)<=tol else 'Definition difference; reconciliation required'
   explanation='Compared with original filing/SEC facts; percentages rounded to 0.1 percentage point.' if rounding else 'Direct filing match.'
   if metric=='EPS':explanation='Pipeline uses basic EPS (not diluted): original filing basic figure checked.'
   if 'EBITDA' in metric:explanation='Reference is operating income + filing D&A, an operating-EBITDA proxy. FMP EBITDA includes different non-operating adjustments; delta is not automatically an error and is not fully reconciled.'
   audits.append(dict(ticker=ticker,metric=metric,fiscal_year=latest.fiscalYear,period_end=end,pipeline_value=numeric,
    filing_reference_value=ref,delta=delta,unit='percentage points' if rounding else 'USD/share' if metric=='EPS' else 'USD',
    verdict=verdict,explanation=explanation,source_url=prior,raw_pipeline_file=str(path.relative_to(ROOT)/'financial_metrics_and_forecasts.csv')))
(ROOT/'audit/fmp_coverage.csv').write_text(pd.DataFrame(rows).to_csv(index=False))
pd.DataFrame(forecasts).to_csv(ROOT/'audit/fmp_temporal_validity.csv',index=False)
pd.DataFrame(values).to_csv(ROOT/'audit/fmp_valuation_audit.csv',index=False)
pd.DataFrame(audits).to_csv(ROOT/'audit/fmp_data_audit.csv',index=False)
print('Actual FMP coverage',len(rows),'data-audit rows',len(audits),'valuation checks',len(values))
print(pd.DataFrame(forecasts).groupby('method').strictly_future.mean().to_string())
print(pd.DataFrame(audits).groupby('verdict').size().to_string())
