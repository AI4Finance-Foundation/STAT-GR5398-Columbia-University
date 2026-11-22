"""Inventory code constants and input quality; no fabricated financial observations."""
from pathlib import Path
import ast,csv,json
r=Path(__file__).resolve().parents[2];source=r/'work/retraining/baseline/valuation/student_valuation.py';tree=ast.parse(source.read_text());config={}
for node in tree.body:
 if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CONFIG' for t in node.targets):config={k.arg:ast.literal_eval(k.value) for k in node.value.keywords}
labels={
'erp':('decimal annual rate','fixed research assumption','Stated 5% ERP, not an observed historical ERP series; economic appropriateness unvalidated'),
'credit_spread':('decimal annual rate','fixed credit-cost proxy','Stated DGS10 + spread; not issuer-specific traded debt yield'),
 'terminal_growth':('decimal annual growth','fixed steady-state assumption','Not a forecast observation; must stay below WACC for valid perpetuity'),
 'years':('years','research horizon','Ten-year fade horizon, not selected on test returns'),
 'history_months':('month-end snapshots','research window','Actual usable snapshots may be fewer; requires known filings and prices'),
 'min_history':('snapshots','coverage threshold','Minimum usable historical multiple snapshots'),
 'min_peers':('other companies','coverage threshold','Broad sector comparability remains a limitation'),
 'min_beta_weeks':('paired weekly returns','coverage threshold','Minimum is not proof of a full five-year sample'),
 'growth_floor':('decimal annual growth','heuristic clipping','Reduces extreme historical growth; may suppress real contraction'),
 'growth_cap':('decimal annual growth','heuristic clipping','May suppress genuine high growth'),
 'max_method_spread':('highest/lowest method midpoint','conservative filter threshold','Only a research agreement rule, not a probability'),
 'minimum_methods':('valid methods','coverage rule','Three implemented, at least two valid; missing estimates abstain'),
 'candidate_limit':('weekly growth_tech rank count','scope rule','None means all supplied growth_tech candidates; not the whole stock market')}
rows=[]
for name,value in config.items():
 unit,kind,note=labels[name];rows.append({'parameter_or_input':name,'value_or_rule':json.dumps(value),'unit':unit,'source_or_location':'student_valuation.py CONFIG','quality_label':kind,'limitation':note})
allocation=json.loads((r/'work/retraining/baseline/selection/configs/growth.json').read_text())['parameters']
for key in ['picks_per_bucket','min_weight_pct','max_weight_pct','incumbent_multiplier']:
 rows.append({'parameter_or_input':'course_allocation.'+key,'value_or_rule':json.dumps(allocation[key]),'unit':'course configuration','source_or_location':'selection/configs/growth.json parameters','quality_label':'unchanged course control','limitation':'Allocation normalization, whole shares, price drift and stale holdings can change realized weights; not optimized in this study'})
extra=[
('minimum_gordon_spread','0.015','annual decimal difference','unchanged valuation/finrobot/valuation_thresholds.py','course engine admissibility gate','WACC>g alone is insufficient in this engine; sub-floor spread invalidates DCF rather than inflating value'),
('terminal_maintenance_anchor','min(da_pct_revenue,capex_pct_revenue); terminal D&A=last projected revenue*anchor; terminal CapEx=D&A*(1+g)','ratios/per-share currency','unchanged valuation/finrobot/dcf.py _terminal_fcf','inherited maintenance-investment approximation','Can cause terminal FCF jump; no asset roll-forward or terminal ROIC constraint'),
('risk_free_rate','latest known DGS10 / 100; clipped [0,0.15]','annual decimal','DataView.macro(DGS10)','observed course input with safety clipping','Day-level visibility; no intraday release audit'),
('beta','weekly covariance(stock,SPY)/variance(SPY); Blume adjustment; clip [0,5]','coefficient','DataView.close lookback=1260; W-FRI returns','estimated proxy','Pair count varies; price and corporate-action quality limitations'),
('cost_of_debt','risk_free_rate + credit_spread; clip [0,0.2]','annual decimal','DCFInputs construction','fixed-spread proxy','Not issuer-specific debt yield'),
('debt_ratio','latest debt_to_capital; clip [0,0.95]','decimal weight','last known financial quarter','book-capital proxy','Not market-value debt/equity weights'),
('tax_rate','median last-four quarterly effective_tax_rate; clip [0,0.35]','decimal','known financial quarters','historical normalization assumption','Not forecast marginal tax rate'),
('ebitda_margin','median last-four quarterly margin; clip [0,1]','decimal','known financial quarters','historical normalization assumption','May differ materially from revenue-weighted annual margin'),
('da_pct_revenue','median(ebitda_margin-operating_margin); clip [0,0.5]','decimal','known financial quarters','derived approximation','Not a PP&E depreciation schedule'),
('capex_pct_revenue','sum(abs(quarterly capex_per_share))/TTM revenue; clip [0,1]','decimal','known financial quarters','historical ratio assumption','Fixed explicit-period ratio; terminal engine resets investment'),
('nwc_pct_revenue and terminal_nwc_pct_revenue','0','decimal','DCFInputs construction','missing-data assumption','Does not prove zero working-capital reinvestment'),
('growth_path','linear interpolation from clipped historical TTM growth to terminal growth','annual decimals','eight consecutive known quarters; numpy.linspace','historical extrapolation','Not analyst consensus; path not independently forecast-validated'),
('net_debt','latest interest_debt_per_share - cash_per_share','per-share currency','last known quarter','observed-derived input','Per-share and adjusted-price bases not fully reconciled'),
('shares_outstanding','1','per-share convention','course database and DCFInputs','unit convention','Not total company share count; dilution not separately modeled'),
('DCF sensitivity','WACC +/- 0.01; terminal growth [0.02,0.025,0.03]','annual decimals','calculate_sensitivity call','assumption grid','Valid positive cells bound the method range; not confidence limits'),
('peer P/E eligibility','0 < recomputed price / positive TTM EPS < 200','multiple','same-sector current index members, excluding self','heuristic screen','Stored pe/ev_multiple never used by valuation; broad peers imperfect'),
('method quantiles','[0.25,0.50,0.75]','quantiles','history and peer methods','descriptive historical distribution','Not calibrated forecast probabilities'),
('valuation aggregation','median method mids; union of method intervals','per-share currency','audit_value','research aggregation rule','Wide ranges can inflate price-in-range hit rates'),
('filing visibility','accepted_date, then filing_date, then quarter-end + 2 months + 1 day','dates','_available_dates and course DataView','conservative date convention','Fiscal labels/revisions and intraday timing not fully verified'),
('transaction costs','0.0006 each side','notional fraction','unchanged course backtester','course execution approximation','Includes commission/slippage approximation, not actual fills'),
('timing','full; no sentiment or stops','exposure mode','unchanged course baseline','experimental control','Actual cash and exposure can drift; stale positions can consume buffer')]
for name,value,unit,location,quality,note in extra:rows.append({'parameter_or_input':name,'value_or_rule':value,'unit':unit,'source_or_location':location,'quality_label':quality,'limitation':note})
out=r/'outputs/retraining/analysis';out.mkdir(exist_ok=True)
with (out/'assumptions_and_quality.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('Saved',len(rows),'source-labeled assumptions/inputs. Quality tags describe provenance, not probability of correctness.')
