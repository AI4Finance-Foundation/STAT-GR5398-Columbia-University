from pathlib import Path
import json,hashlib,configparser,zipfile
import pandas as pd
R=Path(__file__).resolve().parents[1]
rows=[]
for p in (R/'actual_runs').rglob('*.json'):
 try:d=json.loads(p.read_text())
 except:continue
 usages=d.get('provider_usage',[])
 if isinstance(d.get('usage'),dict):usages=[d['usage']]
 for u in usages:
  if not isinstance(u,dict):continue
  inp=u.get('input_tokens',u.get('prompt_tokens',0));out=u.get('output_tokens',u.get('completion_tokens',0));cache=u.get('input_tokens_details',u.get('prompt_tokens_details',{})).get('cached_tokens',0)
  rows.append(dict(file=str(p.relative_to(R)),input_tokens=inp,cached_input_tokens=cache,output_tokens=out,usd=(inp-cache)*2/1e6+cache*.5/1e6+out*8/1e6))
cost=pd.DataFrame(rows);cost.to_csv(R/'audit/api_cost_ledger.csv',index=False)
frames=[]
for t in ['NVDA','AMD','INTC','AAPL','MSFT','GOOGL']:
 folder=R/f'actual_runs/baseline/{t}'
 m=json.loads((folder/'runner_manifest.json').read_text());frames.append(dict(ticker=t,sections=len(list((folder/'agent_evidence').glob('*.json'))),step1_seconds=m['step_seconds'][0],step2_seconds=m['step_seconds'][1],html_reports=len(list((folder/'report').glob('*.html')))))
execution=pd.DataFrame(frames);execution.to_csv(R/'audit/actual_execution_summary.csv',index=False)
ground=pd.read_csv(R/'audit/grounding_rates.csv');claim=pd.read_csv(R/'audit/grounding_claims.csv')
section=ground.groupby('section')[['total','supported','unsupported','contradicted']].sum();section['grounding_rate']=section.supported/section.total;section.to_csv(R/'audit/grounding_section_scorecard.csv')
statement='''# FinRobot Assignment 1: current verified handoff

Deadline: October12,2026; exact hour/timezone unconfirmed. CourseWorks course timezone America/New_York does not establish submission cutoff. Course originals are in course/. Original code commit2717499b8e30f242640af08c4ad9afd1113c2d45, finrobot_equity, not the optional old notebook.

## Results and limitations

All five course starter names NVDA, AMD, INTC, AAPL, GOOGL ran the actual two-step CLI; formal same-sector Technology universe is NVDA, AMD, INTC, AAPL, MSFT. Every name has eight actual typed-agent evidence files and generated HTML reports, model gpt-4.1-2025-04-14. Two peers are configured per name; semiconductor versus hardware/software comparability is a limitation. Financial histories and quote snapshots have distinct retrieval dates recorded in FMP cache; do not call annual metrics live TTM.

Actual FMP financial audit:20 comparisons (NVDA,AMD):16 matches and4 EBITDA/margin definition differences. Source lineages, original NVIDIA annual report and original AMD issuer-hosted10K are saved. AMD D&A750+2254=3004million; FMP EBITDA includes a different non-operating adjustment basis. No unexplained discrepancy was hidden.

DCF: five independent arithmetic recomputations agree;45 full WACC/terminal-growth sensitivity cases saved. CFO minus capex is not unleveredFCFF; synthetic net debt10%EV and weighted-average diluted shares are material weaknesses. Intel negative raw DCF is floored at0 by the helper; this is an assumption failure, not a fair-value claim. Financial conclusions:NVDA growth/margin strength versus high valuation and baseline forecast-label errors;AMD expanding revenue with EBITDA-definition caveats;Intel negativeFCF invalidates a simplistic constant-growthDCF;Apple andMicrosoft positiveFCF do not establish an attractive investment without defensible discount/growth/debt assumptions. No Buy/Sell recommendation is justified by this mechanical audit.

Forecast improvement: actual-FMP future-year validity8/15 to15/15; SEC companion68 rolling-origin forecasts pooledMAPE13.7372% to15.8580%(worse). This is temporal consistency improvement, not validated predictive improvement. Forecast floor/cap and flat margins are assumptions, not learned macroeconomic relationships.

Grounding: claims were extracted and classified across all8 sections of NVDA,AMD,INTC. The same model performs provisional auditing; literal evidence excerpts are checked programmatically and nonliteral citations flagged separately. Formatting differences alone do not determine support. This does not independently prove entailment or extraction completeness. Counts and rates are actual computed automated evaluation results, not instructor-provided metrics beyond the required supported/total formula. Manual adjudication remains recommended. An actual news-agent output asserts headlines/analyst consensus despite no accessible news in its prompt. Agent instructions request regulatory/supply-chain context without supplying source evidence or an implemented browser tool; this structurally invites unsupported prior knowledge. Model prose should be reviewed before use.

Stability: NVDA has three real runs using identical saved financial user prompts and unchanged system instructions; full section outputs preserved. Rating/target absence is reported as missing, not zero variance. Top3 risks extracted with exact quotes; semantic clustering requires review. HTML rating/target from Step2 market-fetch/default fields are separate from agent decisions; free-float95% and a trailingPE labelledforwardPE are upstream weaknesses. Never treat these defaults as measured financial facts.

Model usage costs use officialGPT4.1 rates ($2 input,$0.50cachedinput,$8output per milliontokens). API cost ledger covers saved report/repetition/evaluation responses. Several tiny preflight/debug responses were not fully recorded; total ledger is not a billing reconciliation. Actual provider charges may differ; billing dashboard is authoritative.

## Reproduce

Use the separate work/finrobot_env Python3.12.14; requirements.lock.txt contains installed versions. Private credentials are outside this deliverable. From workspace root:

```sh
work/finrobot_env/bin/python outputs/finrobot_assignment1/code/run_data_only.py --config work/private/config.ini
work/finrobot_env/bin/python outputs/finrobot_assignment1/code/analyze_fmp_actual.py
work/finrobot_env/bin/python outputs/finrobot_assignment1/code/run_actual_pipeline.py --config work/private/config.ini
work/finrobot_env/bin/python outputs/finrobot_assignment1/code/run_actual_pipeline.py --config work/private/config.ini --ticker GOOGL
work/finrobot_env/bin/python outputs/finrobot_assignment1/code/evaluate_agents.py work/private/config.ini
```

Scripts cache requests and evaluation artifacts; preserve them rather than rerun blindly. prepare_upstream.py applies improvement.patch,compatibility.patch,agent_alignment.patch to pinned originals on clean extraction. All three patches apply sequentially. Compatibility adapter is necessary because new FMP accounts cannot use legacyv3; no fabricated financial rows are introduced. Typed-agent wiring records actual prompts/output schemas/usage and disables tracing. Provider failure raises rather than writing canned successful text. NewsHTTP402 is disclosed; no claim of retrieving paid news.

## Assignment requirement checklist

| Requirement | Status | Evidence |
|---|---|---|
| Course-specified finrobot_equity; starter five | Complete | course/FinRobot_README.md Part0; actual_execution_summary.csv |
| Five companies same sector; two peers each | Complete with comparability limitation | Part1; formal five actual_runs/baseline peer CSVs |
| Two companies, ten metrics vs filings | Complete audit; four unresolved definition differences | Part2.1; fmp_data_audit.csv,filing_crosscheck.csv |
| Forecast/valuation audit and independent sensitivity | Complete with disclosed assumptions | Part2.2; fmp_valuation_audit.csv,fmp_full_dcf_sensitivity.csv |
| Eight sections across three companies grounding | Automated provisional evaluation complete; independent review incomplete | Part2.3; grounding_claims.csv,grounding_rates.csv |
| Same inputs three-run stability | Complete actual runs; absent decisions disclosed | Part2.3; stability_actual.csv,stability_summary.json,evaluation/NVDA_run* |
| One improvement and before/after | Complete; accuracy result negative | Part3B; improvement.patch,fmp_temporal_validity.csv,forecast_scorecard.csv |
| Research PDF with methodology/cost/runtime/judgment | Complete working research report | Part4; research_report.pdf,api_cost_ledger.csv,actual_execution_summary.csv |
| Medium student analysis publication | Not published | Part4; medium_draft.md is an outline, not a publication |
| Name_UNI submission folder and course PR | Not submitted; identity not provided | Part8; user prohibits pushing/submission |
| Exact cutoff hour/timezone | Unable to confirm | Course original states date only; CourseWorks inspection findings |

Remaining personal steps:review your conclusions/AI assistance disclosure,write/publish the Medium article under your own account if course policy permits,fill Name_UNI and follow course submission instructions. No GitHub push,submission,email orpublication was performed. Original FinRL project and its Positron kernel were untouched.
'''
statement+=f'\nSaved API ledger subtotal: ${cost.usd.sum():.4f}; recorded input tokens:{cost.input_tokens.sum()},output tokens:{cost.output_tokens.sum()}. Actual claim labels:{claim.classification.value_counts().to_dict()}.\n\n'+section.to_markdown()+'\n'
(R/'CURRENT_STATUS.md').write_text(statement)
(R/'README.md').write_text(statement)
(R/'quality_validation.md').write_text('Actual provider integration successful. Review CURRENT_STATUS.md for scope. Notebook clean execution and4 forecast tests passed; three patches apply. Key values trace to preservedFMP/SEC/issuer files. Grounding evaluation is provisional automated, not independent adjudication. Raw generated baseline HTML intentionally preserves upstream errors for audit; findings identify unsupported news and default-field limitations. Cost ledger excludes small diagnostic calls. Security scan performed before ZIP regeneration.\n')
# Main PDF replaces outdated blocked-status report; SEC supplement stays separate.
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
styles=getSampleStyleSheet();story=[Paragraph('FinRobot: audit before investment use',styles['Title']),Spacer(1,12)]
for para in statement.split('\n\n'):
 if para.startswith('|') or para.startswith('```') or para.startswith('Scripts cache'):continue
 if para.startswith('#'):story.append(Paragraph(para.lstrip('# '),styles['Heading2']))
 else:story.append(Paragraph(para.replace('&','&amp;').replace('<','&lt;').replace('\n','<br/>'),styles['BodyText']))
 story.append(Spacer(1,8))
story.append(PageBreak());story.append(Paragraph('Actual section grounding rates',styles['Heading2']))
data=[['Section','Claims','Supported','Rate']]+[[str(i),str(x.total),str(x.supported),f'{x.grounding_rate:.1%}'] for i,x in section.iterrows()]
t=Table(data,colWidths=[165,70,80,70]);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.lightblue),('GRID',(0,0),(-1,-1),.3,colors.grey),('BOTTOMPADDING',(0,0),(-1,-1),7)]));story.append(t)
story+=[Spacer(1,15),Paragraph('Course and data sources',styles['Heading2']),Paragraph('Assignment: https://github.com/AI4Finance-Foundation/STAT-GR5398-Columbia-University/tree/master/Fall-2026/Assignment1/FinRobot<br/>FMP stable API: https://financialmodelingprep.com/stable/<br/>SEC CompanyFacts: https://data.sec.gov/api/xbrl/companyfacts/<br/>Original issuer filing URLs and fiscal-period dates are recorded in audit/filing_crosscheck.csv and audit/sec_fact_lineage.csv. Provider retrieval timestamps are preserved under actual_runs/fmp_cache.',styles['BodyText'])]
SimpleDocTemplate(str(R/'research_report.pdf'),leftMargin=48,rightMargin=48,topMargin=48,bottomMargin=48).build(story)
print('Final actual execution summary and PDF saved',execution.to_dict('records'))

# Build the final structured PDF rather than the interim handoff layout.
import subprocess,sys
subprocess.run([sys.executable,str(R/'code/build_actual_pdf.py')],check=True)
