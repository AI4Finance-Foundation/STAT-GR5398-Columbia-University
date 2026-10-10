"""Build the working report and blog from verified full/common-window artifacts."""
import argparse,csv,json
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image

ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--run',default='audited_run');ap.add_argument('--comparison',default='comparison_full');ap.add_argument('--common',default='comparison_common');a=ap.parse_args()
r=a.root.resolve();run=r/a.run;comp=r/a.comparison;common=r/a.common
read=lambda p:list(csv.DictReader(p.open(newline='')))
js=lambda p:json.loads(p.read_text())
pct=lambda x:f'{float(x)*100:.2f}%'
num=lambda x:f'{float(x):.2f}'
metrics={x['strategy']:x for x in read(comp/'comparison_metrics.csv') if x['sample']=='full'}
cm={x['strategy']:x for x in read(common/'comparison_metrics.csv') if x['sample']=='full'}
verify=js(comp/'verification.json');cv=js(common/'verification.json');audit=js(run/'audit.json')
# Report building must follow numerical verification, including for fresh directories.
for location in [comp,common]:assert js(location/'artifact_verification.json')['passed']
flow={x['stage']:x for x in read(run/'sample_flow.csv')};scores=read(run/'model_scores.csv')
coverage=read(run/'universe_coverage.csv');pred=read(run/'predictions.csv');sens=read(comp/'cost_delay_sensitivity.csv')
recovery=js(r/'evidence/price_recovery/recovery_manifest.json');financial=js(r/'evidence/financial/financial_spotcheck_summary.json')
contrib=read(comp/'top10_stock_contributions.csv');sector=read(comp/'top10_sector_contributions.csv');recon=js(comp/'contribution_reconciliation.json')
original_baseline=js(r/'reference/original/baseline_run_metrics.json')
bootstrap=read(comp/'excess_return_bootstrap.csv');lag=js(comp/'disclosure_buffer_sensitivity.json')
importance=read(run/'feature_importance_normalized.csv');plan=js(r/'candidate_plan.json')
start=str(verify['common_start'])[:10];end=str(verify['common_end'])[:10]
beaten=float(metrics['audited_top10']['cumulative'])>float(metrics['SPY']['cumulative'])
labels={'audited_top10':'Top 10% (fixed candidate)','audited_top25':'Top 25% rule baseline', 'audited_sector_top25':'Sector-balanced top 25%', 'audited_all_candidates_equal':'All model candidates, equal','quoted_index_members_equal':'Quoted index members, equal','SPY':'SPY','QQQ':'QQQ'}
dest=r/'output/pdf';dest.mkdir(parents=True,exist_ok=True)
styles=getSampleStyleSheet();styles.add(ParagraphStyle(name='Body',fontName='Helvetica',fontSize=9,leading=12,spaceAfter=6,textColor=colors.HexColor('#233549')));styles.add(ParagraphStyle(name='Note',parent=styles['Body'],fontSize=8,leading=10));styles.add(ParagraphStyle(name='Section',fontName='Helvetica-Bold',fontSize=12,leading=15,spaceBefore=8,spaceAfter=6,textColor=colors.HexColor('#173d55')));styles.add(ParagraphStyle(name='Title2',fontName='Helvetica-Bold',fontSize=22,leading=26,spaceAfter=12));styles.add(ParagraphStyle(name='Cell',fontName='Helvetica',fontSize=8,leading=10))
story=[]
def p(t,style='Body'):story.append(Paragraph(t,styles[style]))
def h(t):p(t,'Section')
def table(headers,rows,widths):
 data=[[Paragraph(escape(str(v)),styles['Cell']) for v in row] for row in [headers]+rows]
 t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT');t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dce8ef')),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f2f6f8')])]))
 story.append(t);story.append(Spacer(1,6))
def footer(canvas,doc):
 canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#536273'));canvas.drawString(44,27,'STAT GR5398 | Assignment 1 | Working draft | 8 October 2026');canvas.drawRightString(A4[0]-44,27,str(doc.page))
p('FinRL stock selection:<br/>data repair and fixed-candidate evaluation','Title2')
p('Patrick Xiong | Columbia UNI: sx2434 | Research draft','Note')
p(f'The predeclared top-10% candidate {"exceeds" if beaten else "does not exceed"} SPY on the available full trading window, {start} to {end}: {pct(metrics["audited_top10"]["cumulative"])} versus {pct(metrics["SPY"]["cumulative"])} cumulative return after strategy trading costs. This is a coverage-limited historical result. Missing failed/acquired-company histories and unverified original statement vintages still prevent a claim that the sample is free of survivorship or look-ahead bias.')
h('1. What changed and why')
p('The course top-25% rule remains the baseline. The proposed submission rule is top 10%, chosen using the 7 October development results and fixed before this repaired run. We do not treat the new run or 2024 onward as an untouched holdout, and do not change the threshold after seeing this run.')
table(['Change','Evidence / implication'],[
 ['Price coverage',f'{len(recovery["accepted"])} previously missing ticker series accepted using current SEC/Yahoo identity checks, complete in-life adjusted prices and, where available, overlapping return checks. {len(recovery["rejected"])} original missing symbols remain unresolved or excluded.'],
 ['Ticker identity','Official same-issuer changes and share-class notation use a canonical security identifier. One financial observation and one complete price source per security avoid duplicate counting and mixed-provider series.'],
 ['Financial joins','Raw payloads now match the aggregate income filing date, then the actual fiscal period. Treating fiscal Q1 as calendar Q1 matched the wrong records for several firms.'],
 ['Financial spot checks',f'{financial["sample_rows"]} fixed sample rows; {financial["matched_raw_payload_rows"]} had raw payloads. EPS matched 30/30 and current ratio 29/29 after correcting the join. Quarter-price / quarterly-EPS explains the supplied PE definition.'],
 ['Trade accounting','Quarterly close trades, next-session return accrual, drifted holdings and 10 bps per dollar bought/sold. A missing held quote raises an error; no indefinite forward-fill or invented terminal proceeds.']
],[126,381])
h('2. Data and model protocol')
p(f'The prepared inputs contain {int(flow["raw"]["rows"]):,} rows. Within the price window the historical membership filter changes {int(flow["decision_in_price_window"]["rows"]):,} rows / {flow["decision_in_price_window"]["tickers"]} tickers to {int(flow["historical_member"]["rows"]):,} rows / {flow["historical_member"]["tickers"]} tickers. Timely quoted inputs contain {int(flow["quoted_at_entry"]["rows"]):,} rows before rolling model eligibility; final predictions contain {len(pred):,} rows.')
p('RF, XGBoost and LightGBM use the same four fixed parameter candidates per model and seed 42. Expanding chronological training folds fit median imputation only on the training fold. Completed-outcome purges separate training, outer validation and inference. Lowest outer-validation MSE chooses the model for each sector/quarter. The 52 financial features exclude absolute adjusted-price levels. No extra hyperparameter search was introduced.')
story.append(PageBreak())
h('3. Full trading window and the original common window')
p(f'Full available window: {start} to {end}, {verify["trading_dates"]:,} trading dates. The first day is the earliest saved audited inference, rather than the latest start among older comparison versions.')
table(['Portfolio','Cumulative','CAGR','Vol.','Sharpe*','Max DD','TO/year'],[[labels[k],pct(metrics[k]['cumulative']),pct(metrics[k]['cagr']),pct(metrics[k]['volatility']),num(metrics[k]['sharpe_rf0']),pct(metrics[k]['max_drawdown']),num(metrics[k]['annual_turnover'])+'x'] for k in labels],[143,65,54,55,47,65,78])
p('* Daily Sharpe uses zero risk-free rate and 252-day annualization. Turnover counts both buys and sells. SPY/QQQ use adjusted closes over identical dates, no explicit benchmark trading fee, with fund expenses embedded in observed prices. These are equity backtests without leverage.','Note')
table(['Original common interval','Top 10%','Top 25%','SPY','QQQ'],[[str(cv['common_start'])[:10]+' to '+str(cv['common_end'])[:10]]+[pct(cm[k]['cumulative']) for k in ['audited_top10','audited_top25','SPY','QQQ']]],[223,71,71,71,71])
story.append(Image(str(comp/'nav_drawdown.png'),width=455,height=318.5))
p('Both tables use the repaired data. The 7 October results are retained separately and must not be presented as an isolated causal comparison: data coverage, fiscal matching, security identity and price-source choices changed together.','Note')
story.append(PageBreak())
h('4. What drove the candidate result')
p('Contributions below are dollar PnL divided by initial portfolio NAV, summed through time. They are additive percentage points of the final NAV change, not each stock\'s buy-and-hold return or a causal estimate of alpha.')
table(['Largest stock contributions','Sector','Contribution'],[[x['ticker'],x['sector'],pct(x['gross_pnl_initial_nav'])+' pts'] for x in contrib[:6]],[108,285,114])
table(['Sector','Contribution'],[[x['sector'],pct(x['gross_pnl_initial_nav'])+' pts'] for x in sector[:6]],[350,157])
p(f'All stock contributions sum to {pct(recon["gross_pnl"])} of initial NAV. Trading costs remove {pct(recon["fee_pnl"])}; the reconciled net change is {pct(recon["net_pnl"])}. Sector names are retrospective classifications and require that qualification.')
h('Model comparison and feature importance')
modelrows=[]
for model,label in [('rf','Random Forest'),('xgb','XGBoost'),('gbm','LightGBM')]:
 sub=[x for x in scores if x['model']==model];top=sorted([x for x in importance if x['model']==model],key=lambda x:float(x['importance']),reverse=True)[:3]
 modelrows.append([label,sum(x['winner']=='True' for x in sub),f'{sum(float(x["mse"]) for x in sub)/len(sub):.6f}',', '.join(x['feature'] for x in top)])
table(['Model','Wins','Mean val. MSE','Top normalized importances'],modelrows,[115,50,95,247])
p('Importances are normalized within each fitted model/window. RF impurity, XGBoost gain and LightGBM split counts differ; their magnitudes are not directly comparable. Validation MSE wins are not evidence of statistically significant superiority.','Note')
h('Uncertainty and disclosure delay')
b=next(x for x in bootstrap if x['strategy']=='audited_top10')
p(f'Top-10% annualized arithmetic excess return over SPY is {pct(b["annualized_arithmetic_excess"])}. The descriptive 63-day block-bootstrap 95% interval is {pct(b["ci95_low"])} to {pct(b["ci95_high"])}. This conditions on the dataset and excludes strategy-selection and missing-data uncertainty; statistical significance is not an added course requirement.')
p(f'Requiring five extra business days of disclosure lead time, while keeping forecasts fixed, retains {lag["retained_predictions"]:,}/{lag["original_predictions"]:,} predictions and yields {pct(lag["cumulative"])} cumulative return. This is an eligibility sensitivity, not a separately retrained model.')
story.append(PageBreak())
h('5. Costs, failures and remaining work')
sm={(x['strategy'],int(x['delay_sessions']),round(float(x['fee_bps']))):x for x in sens}
table(['Portfolio / delay','5 bps','10 bps','20 bps','50 bps'],[[('Top 10%' if k=='audited_top10' else 'Top 25%')+' / '+str(d)+' sessions']+[pct(sm[k,d,c]['cagr']) for c in [5,10,20,50]] for k in ['audited_top10','audited_top25'] for d in [0,1,5]],[223,71,71,71,71])
p('Entries are annualized returns over the full window. Delayed variants begin from cash and use then-current membership and executable quotes; they do not credit the entry-day move.')
h('What did not work')
p(f'The original notebook reported {pct(original_baseline["cumulative_return"])} cumulative return while combining inconsistent stock/benchmark return conventions and missing active-stock returns. Its headline return is retained as reproduction history rather than a comparable investment score. The first raw-date audit used fiscal/calendar quarter keys that did not consistently identify the aggregate statement. Partial public quote downloads could not safely restore failed-bank, share-class-change or merger outcomes; uncertain series were excluded instead of assigning terminal value zero or treating stale quotes as realizable cash.')
h('Data gaps that remain')
p(f'At the first audited coverage date, {coverage[0]["quoted_members"]}/{coverage[0]["members"]} historical members have quotes; at the last coverage date, {coverage[-1]["quoted_members"]}/{coverage[-1]["members"]}. The decision ledger records every accepted and rejected recovery. Missing examples include SIVB, FRC, SBNY, TWTR and ATVI. We have no certified stockholder terminal proceeds for these names. Therefore the graded survivorship-free sample requirement remains unresolved even where the observed return exceeds SPY.')
p('Current SEC identity checks and provider-adjusted histories do not prove every historical security mapping or corporate-action adjustment. Financial filing-date checks establish recorded availability within the supplied snapshot, not original unrevised financial vintages. SEC/source coverage is missing for part of the sample. No real fill, market-impact or capacity study was performed.')
h('Reproduction and submission status')
p('README gives one end-to-end command for source verification, recovery, financial checks, training, both windows, numerical checks and PDF creation. The verifier takes arbitrary run/comparison paths; sample_flow is produced during training. Mutable public recovery prices can change, so their retrieval metadata, source hashes and return-consistency checks must accompany reproduced results.')
p('Patrick Xiong (sx2434). Medium article published on 9 October; link in README. The course PR is a draft while historical data repairs remain pending WRDS approval. Numerical results on these pages are the preserved 8 October experiment; they have not been rerun on WRDS data. Code, report and derived outputs are packaged separately from raw inputs.')
h('Sources')
p('<link href="https://github.com/AI4Finance-Foundation/STAT-GR5398-Columbia-University/tree/master/Fall-2026/Assignment1/FinRL-Trading">Course instructions</link>; <link href="https://github.com/AI4Finance-Foundation/FinRL-Trading/tree/dd1fdcf5db69b4cf42e90a185211defa8d5ae273/data">pinned course dataset</link>; <link href="https://www.sec.gov/files/company_tickers.json">SEC company tickers</link>; Yahoo Finance adjusted-price chart histories; official corporate-action references in evidence/root_corporate_action_sources.json. Failed-bank context: <link href="https://www.fdic.gov/resources/resolutions/bank-failures/in-brief/2023">FDIC 2023 failures</link>. These sources do not supply a complete historical equity-delisting-return dataset.','Note')
doc=SimpleDocTemplate(str(dest/'FinRL_Assignment1_Research_Draft.pdf'),pagesize=A4,rightMargin=44,leftMargin=44,topMargin=36,bottomMargin=49,title='FinRL Assignment 1 - Data repair and fixed candidate evaluation');doc.build(story,onFirstPage=footer,onLaterPages=footer)
blog=f'''# A FinRL backtest after repairing prices, identities and financial dates

Unpublished research draft, 8 October 2026. Patrick Xiong (sx2434).

I used the course's supervised stock-selection workflow: financial features predict the next quarter's return with Random Forest, XGBoost and LightGBM. The proposed strategy keeps the highest-ranked 10% of eligible stocks, equally weights them and rebalances quarterly. I selected this candidate after the 7 October development experiments, before the repaired results below. Top 25% remains the course baseline; this is not an untouched holdout.

## What the audit changed

Point-in-time index membership did not restore companies absent from the daily price table. The new recovery accepted {len(recovery['accepted'])} missing ticker series and excluded {len(recovery['rejected'])} unresolved or ambiguous cases. Current SEC issuer identities, official rename announcements, provider metadata and adjusted-return consistency support these decisions. They do not certify a complete survivorship-free sample.

A second problem was financial-period matching. Fiscal quarter labels do not always denote the same calendar quarter. Matching the actual income filing and fiscal period restored consistency in the fixed 48-row spot check: 30 income EPS values and 29 current ratios matched their supplied payloads. The aggregate PE uses quarter-date price divided by quarterly EPS and consequently differs from the provider's PE definition. Original unrevised filing vintages remain unverified.

Canonical security identities now avoid duplicate company/quarter observations and choose one complete price series for both model labels and portfolio execution. Missing held prices stop the simulation rather than being silently forward-filled indefinitely.

## The measured result

Across the full available trading window, {start} to {end}, top 10% returned {pct(metrics['audited_top10']['cumulative'])} cumulatively, top 25% {pct(metrics['audited_top25']['cumulative'])}, SPY {pct(metrics['SPY']['cumulative'])}, and QQQ {pct(metrics['QQQ']['cumulative'])}. The candidate's annualized return was {pct(metrics['audited_top10']['cagr'])}, volatility {pct(metrics['audited_top10']['volatility'])}, zero-risk-free-rate Sharpe {num(metrics['audited_top10']['sharpe_rf0'])}, maximum drawdown {pct(metrics['audited_top10']['max_drawdown'])}, and annual two-sided turnover {num(metrics['audited_top10']['annual_turnover'])}x.

Each dollar bought or sold costs 10 basis points, including initial purchases. Trades occur at the designated close; new positions earn returns from the next session. Fees are paid from portfolio assets. The report retains the original 2020-09-01 common-window comparison, tests costs and execution delays, and explains stock/sector contributions.

## What this result establishes

The fixed candidate {'beats' if beaten else 'does not beat'} SPY within this available full-window dataset. Missing failed/acquired-company histories, incomplete financial vintages and the development choices limit any broader claim of alpha. Top-10% annualized arithmetic excess has a descriptive bootstrap interval of {pct(b['ci95_low'])} to {pct(b['ci95_high'])}; the interval does not account for all those data and selection issues.

The deliverable is a reproducible, coverage-limited research experiment with recorded failures and source decisions. Completing the assignment's bias-free sample requirement remains a separate unresolved item. The accompanying README specifies the commands, source snapshots, recovery evidence and limitations.

Sources: [course instructions](https://github.com/AI4Finance-Foundation/STAT-GR5398-Columbia-University/tree/master/Fall-2026/Assignment1/FinRL-Trading), [pinned data](https://github.com/AI4Finance-Foundation/FinRL-Trading/tree/dd1fdcf5db69b4cf42e90a185211defa8d5ae273/data), [SEC company tickers](https://www.sec.gov/files/company_tickers.json), issuer announcements in the evidence manifest, and Yahoo Finance adjusted prices.
'''
(r/'Medium_draft.md').write_text(blog)
print(dest/'FinRL_Assignment1_Research_Draft.pdf')
