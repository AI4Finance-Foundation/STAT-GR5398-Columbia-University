"""Rebuild the FinRobot research PDF from saved results, without running a backtest."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZipFile

import numpy as np
import pandas as pd
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / 'source_code'
WIDTH, HEIGHT, LEFT = 612, 792, 45
BODY = WIDTH - 2 * LEFT
INK = '#193238'
TEAL = '#007F72'
BLUE = '#496783'
RUST = '#AA4B30'
GREY = '#4D6468'
LIGHT = '#D7E3E0'
PALE = '#EDF5F2'
PAPER = '#FFFFFF'
SPLITS = {'dev': 'Development', 'val': 'Validation', 'test': 'Holdout'}
NAMES = {'A_selection': 'A   Selection', 'B_valuation': 'B   + valuation',
         'C_timing_full': 'C   + full timing', 'D_both_full': 'D   Both'}
MODELS = {'student': 'Composite', 'trailing_pe': 'Course P/E', 'student_dcf': 'DCF',
          'student_ev': 'Own EV/EBITDA', 'student_peer': 'Peer P/E'}
DISCLOSURE_URL = 'https://github.com/AI4Finance-Foundation/STAT-GR5398-Columbia-University/issues/21'


def num(value, digits=2, sign=False):
    if pd.isna(value):
        return 'n/a'
    return f'{value:+.{digits}f}' if sign else f'{value:.{digits}f}'


def pct(value, digits=1):
    return num(value * 100, digits) + '%' if pd.notna(value) else 'n/a'


class Paper:
    def __init__(self, path):
        self.c = Canvas(str(path), pagesize=(WIDTH, HEIGHT), invariant=True, pageCompression=1)
        self.c.setTitle('Can valuation improve a growth portfolio?')
        self.c.setAuthor('Zeyu Gu | STAT GR5398')
        self.page = 0
        self.y = 0

    def text(self, x, y, text, font='Helvetica', size=10, color=INK, right=False):
        self.c.setFillColor(HexColor(color)); self.c.setFont(font, size)
        (self.c.drawRightString if right else self.c.drawString)(x, y, str(text))

    def line(self, x1, y1, x2, y2, color=LIGHT, width=0.6):
        self.c.setStrokeColor(HexColor(color)); self.c.setLineWidth(width); self.c.line(x1, y1, x2, y2)

    def page_start(self, section, title, subtitle=None):
        if self.page:
            self.page_end()
        self.page += 1
        self.text(LEFT, 754, 'STAT GR5398  /  FINROBOT', 'Helvetica-Bold', 8.4, TEAL)
        self.text(WIDTH-LEFT, 754, section.upper(), 'Helvetica-Bold', 8.4, GREY, True)
        self.line(LEFT, 741, WIDTH-LEFT, 741, TEAL, 1)
        self.y = 714
        self.par(title, size=27, leading=30, font='Times-Bold', gap=10)
        if subtitle:
            self.par(subtitle, size=11.5, leading=16, color=GREY, gap=14)

    def page_end(self):
        if self.y < 53:
            raise ValueError(f'Content overflows page {self.page} by {53-self.y:.1f} pt')
        self.line(LEFT, 40, WIDTH-LEFT, 40)
        self.text(LEFT, 25, 'Zeyu Gu   /   zg2556     •     Columbia University, Fall 2026', size=8, color=GREY)
        self.text(WIDTH-LEFT, 25, str(self.page), size=8.5, color=GREY, right=True)
        self.c.showPage()

    def close(self):
        self.page_end(); self.c.save()

    def par(self, text, size=10.5, leading=14.4, font='Helvetica', color=INK, gap=8):
        style = ParagraphStyle('body', fontName=font, fontSize=size, leading=leading, textColor=HexColor(color))
        p = Paragraph(text, style); _, h = p.wrap(BODY, 690)
        self.y -= h; p.drawOn(self.c, LEFT, self.y); self.y -= gap

    def head(self, text):
        self.y -= 3; self.par(text, 15, 18, 'Times-Bold', gap=5)

    def note(self, text, gap=10):
        self.par(text, size=8.9, leading=12, color=GREY, gap=gap)

    def table(self, heads, rows, widths, right=(), highlights=(), fs=9.1, row_h=20):
        x = LEFT; y = self.y
        if sum(widths) != BODY:
            raise ValueError(f'Table width mismatch {sum(widths)}')
        self.c.setFillColor(HexColor(INK)); self.c.rect(LEFT, y-row_h, BODY, row_h, fill=1, stroke=0)
        for i, (head, w) in enumerate(zip(heads, widths)):
            self.text(x+w-5 if i in right else x+5, y-row_h+6, head, 'Helvetica-Bold', fs-0.3, PAPER, i in right); x+=w
        y-=row_h
        for j,row in enumerate(rows):
            if j in highlights or j%2 == 1:
                self.c.setFillColor(HexColor(PALE if j in highlights else '#F6F8F7'))
                self.c.rect(LEFT, y-row_h, BODY, row_h, fill=1, stroke=0)
            x=LEFT
            for i, (cell, w) in enumerate(zip(row, widths)):
                font = 'Helvetica-Bold' if j in highlights else 'Helvetica'
                if self.c.stringWidth(str(cell),font,fs)>w-9:
                    raise ValueError(f'Table text too wide: {cell} ({w:.1f} pt)')
                self.text(x+w-5 if i in right else x+5, y-row_h+6, cell,font,fs,TEAL if j in highlights else INK,i in right);x+=w
            y-=row_h
        self.line(LEFT,y,LEFT+BODY,y);self.y=y-9

    def chart(self, lines, colors, height=123):
        """Overlay equities rebased at first visible close; draw axis labels and legend."""
        top=self.y-19; bottom=top-height; x0=LEFT+37; x1=LEFT+BODY-4
        vals=pd.concat(lines,axis=1).dropna(); vals=vals.div(vals.iloc[0])*100
        ymin=np.floor(vals.min().min()/25)*25; ymax=np.ceil(vals.max().max()/25)*25
        if ymin==ymax: ymax=ymin+25
        for v in np.linspace(ymin,ymax,5):
            y=bottom+(v-ymin)/(ymax-ymin)*height
            self.line(x0,y,x1,y); self.text(x0-6,y-3,num(v,0),size=8,color=GREY,right=True)
        for col, color in colors.items():
            indexes=range(len(vals)); points=vals[col].to_numpy()
            path=self.c.beginPath()
            for i in indexes:
                x=x0+i/(len(vals)-1)*(x1-x0); y=bottom+(points[i]-ymin)/(ymax-ymin)*height
                (path.moveTo if i==0 else path.lineTo)(x,y)
            self.c.setStrokeColor(HexColor(color)); self.c.setLineWidth(1.35); self.c.drawPath(path)
        x=x0
        for label, color in colors.items():
            self.line(x,top+12,x+17,top+12,color,2);self.text(x+22,top+9,label,size=8.5,color=GREY);x+=self.c.stringWidth(label,'Helvetica',8.5)+45
        for i in [0,len(vals)//2,len(vals)-1]:
            x=x0+i/(len(vals)-1)*(x1-x0)
            self.text(x,bottom-14,vals.index[i].strftime('%b %Y'),size=8,color=GREY,right=(i==len(vals)-1))
        self.y=bottom-27


def read_runs(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def discover(out, log):
    splits=['dev','val']
    test_outputs = [p for pat in ('test_*', 'valuation_test_*', 'ablation_test.csv', 'portfolio_by_year_test.csv')
                    for p in out.glob(pat) if p.name != 'test_runs.jsonl']
    if test_outputs and not log:
        raise ValueError('Holdout files lack their run ledger. Archive source_code/output/test_runs.jsonl first.')
    if log:
        required = ['ablation_test.csv', 'portfolio_by_year_test.csv']
        required += [f'valuation_test_{kind}.csv' for kind in ('summary', 'audit', 'inputs', 'observations', 'ic_monthly', 'yearly')]
        required += [f'test_{arm}/{name}' for arm in NAMES for name in ('equity.csv', 'trades.csv', 'stats.json')]
        missing = [name for name in required if not (out / name).is_file()]
        if missing:
            raise ValueError('Incomplete holdout archive. Recover these saved files before rendering: ' + ', '.join(missing))
        splits.append('test')
    data={}
    for p in splits:
        csv=lambda name: pd.read_csv(out / name)
        ablation=csv(f'ablation_{p}.csv').set_index('run'); ablation.index=ablation.index.str.removeprefix(p+'_')
        if set(ablation.index) != set(NAMES):
            raise ValueError(f'The {p} file must be the final full-timing ablation')
        data[p]={'port':ablation,'annual':csv(f'portfolio_by_year_{p}.csv'),'obs':csv(f'valuation_{p}_observations.csv'),
                 'sums':csv(f'valuation_{p}_summary.csv').set_index(['model','horizon']),
                 'yearly':csv(f'valuation_{p}_yearly.csv'),'audit':csv(f'valuation_{p}_audit.csv').iloc[0],
                 'curves':{r: pd.read_csv(out/f'{p}_{r}'/'equity.csv',index_col=0, parse_dates=True)['equity'] for r in NAMES}}
        data[p]['cost'] = {arm: 0.0006 * pd.read_csv(out / f'{p}_{arm}' / 'trades.csv').notional.abs().sum()
                           for arm in ('A_selection', 'B_valuation')}
        if p!='test': data[p]['sens']=csv(f'sensitivity_{p}.csv')
        if p == 'dev':
            data[p]['weekly_marks'] = csv('weekly_marks_dev.csv')
    return data


def benchmarks(source, d):
    lo,hi=d['curves']['A_selection'].index[[0,-1]]
    # Read only the two benchmark symbols, and only through the released split.
    import pyarrow.parquet as pq
    p=source/'data/prices/daily_adjusted.parquet'; typ=pq.ParquetFile(p).schema_arrow.field('date').type
    boundary = lambda x: x.date() if str(typ).startswith('date') else (x.strftime('%Y-%m-%d') if str(typ) in ('string','large_string') else x.to_pydatetime())
    raw=pd.read_parquet(p,columns=['date','symbol','close'], filters=[('symbol','in',['SPY','QQQ']),('date','>=',boundary(lo)),('date','<=',boundary(hi))])
    raw['date']=pd.to_datetime(raw.date); curves=raw.pivot(index='date',columns='symbol',values='close').sort_index()
    for ticker in ('SPY','QQQ'):
        s=curves[ticker].dropna(); ret=s.pct_change().dropna(); total=s.iloc[-1]/s.iloc[0]-1; cagr=(1+total)**(252/len(ret))-1; vol=ret.std()*np.sqrt(252); dd=(s/s.cummax()-1).min()
        yield ticker, {'total_return_pct':total*100,'cagr_pct':cagr*100,'volatility_pct':vol*100,'sharpe':ret.mean()*252/vol,'max_drawdown_pct':dd*100,'calmar':cagr/abs(dd),'turnover_per_year':np.nan},s


def cover(book, data, threshold, freeze, log):
    book.page_start('Research report', 'Can valuation improve a growth portfolio?', 'A point-in-time test of a three-method valuation filter on the course growth-stock selector.')
    v=data['val']; a=v['port'].loc['A_selection']; b=v['port'].loc['B_valuation']
    book.head('What did the valuation filter add?')
    book.par(f'The selected rule excluded a top-ranked candidate when its estimated fair value was more than {pct(threshold,0)} below its price and at least two methods agreed within 2.5×. In validation, the filter returned {num(b.total_return_pct)}% versus {num(a.total_return_pct)}% for selection alone. Its annual turnover was {num(b.turnover_per_year)}×, compared with {num(a.turnover_per_year)}× for selection.')
    rows=[]
    for p,d in data.items():
        x,y=d['port'].loc[['A_selection','B_valuation']].itertuples()
        rows.append([SPLITS[p], num(x.total_return_pct),num(y.total_return_pct),num(y.total_return_pct-x.total_return_pct,2,True),num(y.sharpe-x.sharpe,3,True),num(y.max_drawdown_pct-x.max_drawdown_pct,2,True)])
    book.table(['Period','A return %','B return %','Delta, pp','Delta Sharpe','Delta MDD, pp'],rows,[96,79,79,81,91,96],right=(1,2,3,4,5),highlights=(1,))
    book.note('A uses the course selector. B adds the valuation filter. Returns use each portfolio’s first closing net asset value as the base. A positive change in maximum drawdown means a shallower loss.')
    chart_period = 'test' if 'test' in data else 'dev'
    book.head(f'{SPLITS[chart_period]} equity, indexed to 100')
    curves = data[chart_period]['curves']
    book.chart({'Selection':curves['A_selection'],'Filtered':curves['B_valuation']},{'Selection':BLUE,'Filtered':TEAL},height=122)
    ic = v['sums'].loc['student', 252]
    matched = v['sums'].loc['student_matched', 252]
    reference = v['sums'].loc['trailing_pe_matched', 252]
    book.par(f'On the same {int(matched.valued_with_return)} validation stock-date pairs, one-year rank IC was {num(matched.mean_ic,3)} for the composite and {num(reference.mean_ic,3)} for course P/E. The composite interval captured {pct(ic.interval_coverage)} of {int(ic.interval_pairs)} realized prices against its 50% target.')
    if 'test' in data:
        holdout=data['test']['port']
        baseline,filtered=holdout.loc['A_selection'],holdout.loc['B_valuation']
        sig=data['test']['sums'].loc['student',252]
        book.par(f'In the holdout, B gained {num(filtered.total_return_pct-baseline.total_return_pct)} pp over A and reduced maximum drawdown by {num(filtered.max_drawdown_pct-baseline.max_drawdown_pct)} pp. One-year interval coverage fell to {pct(sig.interval_coverage)}. The configuration was frozen locally before the only holdout run. The hash was disclosed on GitHub after the run. TA confirmation is pending (p. 13).',size=10.1)
    edition='Holdout results included' if 'test' in data else 'Development and validation edition'
    book.note(f'{edition}. Logged holdout runs: {len(log)}. '+(f'Configuration commit: <font name="Courier">{escape(freeze)}</font>.' if freeze else 'Configuration commit is recorded with the validation decision.'))


def data_page(book, data):
    book.page_start('Data and bias audit', 'What could the filter know?',
                    'Each monthly or weekly decision uses the prior equity session as its valuation as-of date. The course DataView truncates observations there. Historical publication times require a separate check.')
    book.head('Prices, filings and index membership')
    book.par('The filter uses the decision-date adjusted close and the S&amp;P 500/Nasdaq-100 membership in force then. The store supplies the latest visible filing version per fiscal quarter. The filter requires both the fiscal date and publication date to precede the decision date. It uses acceptance date when recorded, otherwise filing date, and drops rows with neither.')
    book.par('The course guard confines model I/O to DataView. Historical prices and filed fundamentals determine each mark. The evaluator alone reads forward outcomes. The filter uses neither today’s peers and analyst estimates nor historical LLM predictions, withheld labels or unverified vendor P/E and EV multiples.')
    fields = [
        ('Candidate-months', 'candidates'), ('Distinct stocks', 'distinct_tickers'),
        ('Outside decision-date index union', 'not_in_pit_universe'),
        ('Missing decision-close quote', 'missing_exact_spot'),
        ('Fewer than 4 visible quarters', 'fewer_than_four_quarters'),
        ('Latest 4 quarterly EPS incomplete', 'latest_four_eps_incomplete'),
        ('Last fiscal quarter older than 180 days', 'latest_fiscal_quarter_over_180d'),
        ('At least one visible undated filing', 'candidates_with_undated_visible_filing'),
        ('Missing 10-year Treasury yield', 'missing_risk_free'),
        ('One-year label outside period', 'unmatured_252'),
        ('Missing close at mature one-year horizon', 'missing_matured_end_price_252'),
    ]
    labels = list(data)
    rows = [[label, *[f'{int(data[p]["audit"][key]):,}' for p in labels]] for label, key in fields]
    widths = [BODY - 76 * len(labels)] + [76] * len(labels)
    book.table(['Input or outcome', *[SPLITS[p] for p in labels]], rows, widths,
               right=tuple(range(1, len(labels) + 1)), row_h=17.5, fs=9)
    book.note('Counts use candidate-months unless labeled otherwise. Undated filings can enter the course view through its modeled lag. The filter drops those rows. Label exclusions count all candidates before screening for a usable valuation.')
    book.head('Observation date versus release time')
    dates = sum(int(d['audit']['months']) for d in data.values())
    same_day = all(d['audit']['missing_risk_free'] == 0 and d['audit']['max_risk_free_age_days'] == 0
                   for d in data.values())
    timing = (f'On all {dates} monthly marks, the DGS10 observation date equaled the valuation as-of date, the equity session preceding the monthly evaluation.' if same_day
              else 'DGS10 is taken at its latest nonmissing observation date.')
    book.par(timing + ' The bundle lacks publication timestamps and historical vintages. The Federal Reserve posts its H.15 release at 4:15 p.m. A date match cannot establish that FRED supplied the value at the equity close, or before the next open. A future study could impose a release lag. <link href="https://www.federalreserve.gov/releases/h15/" color="#007F72">Federal Reserve H.15 release</link>.', size=10.0)
    book.note('Prices share one refreshed, dividend-adjusted basis. Historical per-share accounts may use a different split or dividend basis. Missing endpoint quotes can selectively remove delisted or renamed companies from the outcome sample.')


def dcf_page(book):
    book.page_start('Model specification', 'Build the DCF per share',
                    'The course bundle has quarterly per-share accounts. Setting shares outstanding to one keeps the projected cash flows and equity estimate on that basis.')
    book.head('Growth from filed revenue')
    book.par('Sum four quarters for trailing-year revenue. DCF needs eight consecutive quarters and positive revenue in both years. Start from year-over-year growth. With sixteen consecutive quarters and a positive base, average that growth rate and the three-year CAGR with equal weights. This blend smooths a single reporting cycle. Retain the course example’s −10% to +30% growth clip and ten-year linear fade to 2.5% terminal growth.')
    book.par('FinRobot discounts end-of-year cash flows and a Gordon perpetuity. Its maintenance anchor is the smaller of capex and D&amp;A as revenue shares. The engine requires positive terminal cash flow, positive equity value and at least 1.5 pp between WACC and terminal growth. Subtract net debt from enterprise value.')
    book.head('Course example parameters and student input checks')
    rows = [
        ['EBITDA margin', 'Median of four margins, bounded at 0–100%. The study requires it to be positive.'],
        ['D&A / revenue', 'Median EBITDA minus operating margin by quarter, bounded at 0–50%.'],
        ['Capex / revenue', 'Four-quarter absolute capex per share / TTM revenue, bounded at 0–100%.'],
        ['Working capital', 'Zero NWC / revenue, since the bundle lacks the required series.'],
        ['Tax rate', 'Median of four effective rates, bounded at 0–35%.'],
        ['Net debt', 'Latest interest-bearing debt per share less cash per share.'],
        ['Risk-free rate', 'Latest dated DGS10 / 100, bounded at 0–15% for CAPM.'],
        ['Cost of debt', 'DGS10 / 100 plus a fixed 2 pp credit premium, bounded at 0–20%.'],
        ['Debt weight', 'Latest debt-to-capital ratio, bounded at 0–95%.'],
        ['Cost of equity', 'Risk-free rate plus adjusted beta times a fixed 5% equity premium.'],
    ]
    book.table(['Input', 'Definition'], rows, [119, 403], fs=9.0, row_h=18.5)
    book.par('Beta uses up to 1,260 closes and at least 52 aligned completed weekly returns against SPY. Divide covariance by positive SPY variance. FinRobot shrinks beta above one by one third toward one, and the example bounds the result at 0–5. WACC weights CAPM equity cost and after-tax debt cost. Insufficient data or an invalid engine valuation leaves DCF unavailable.', size=10.1)
    book.par('Use the example’s nine stress combinations of WACC ±1 pp and 2.0%, 2.5%, 3.0% terminal growth. The study takes P25 and P75 of positive finite cells and expands the band to include the central DCF. An empty valid grid gives no DCF. These scenarios hold the operating margin and revenue path fixed.', size=10.1)


def methods_page(book, data, threshold):
    book.page_start('Model specification', 'From three marks to one decision')
    book.head('Own-history EV/EBITDA')
    book.par('Sum four consecutive quarters of revenue per share times EBITDA margin. For each TTM window, take the first close strictly after all four quarters were known. Discard windows already superseded by a newer public quarter. Divide that close plus contemporaneous net debt per share by positive TTM EBITDA per share. The course specifies five years of history and multiple percentiles. The study requires four distinct usable price dates and retains multiples between 0 (exclusive) and 100×. Apply P25/median/P75 to current TTM EBITDA, subtract current net debt, reject a nonpositive center and floor the lower bound at zero.')
    book.head('Same-sector trailing P/E')
    book.par('The course example supplies the same-sector median, P25/P75 band and 200× cap. Peers must be current index members with positive prices and positive trailing earnings. This study requires four consecutive quarters of complete EPS, a latest quarter within 180 days, a decision-date close, and five peers after excluding the target. These limits reduce stale accounts and thin peer groups. Multiply each peer P/E percentile by the target’s TTM EPS.')
    book.head('Composite and backfill')
    book.par(f'Every target needs a positive current close, four consecutive quarters and a latest quarter within 180 days. The study takes the median method center as fair value. Pool the available lows, centers and highs equally, then use their P25/P75 expanded to contain fair value. Screen the first 20 eligible growth names. To remove one, the study requires at least two methods, max/min center at most 2.5× and fair/spot − 1 below −{pct(threshold,0)}. The consensus limits are fixed heuristics. Validation selected the discount threshold. A missing or disputed mark leaves the candidate eligible for the next-open ranking.')
    monthly = data['dev']['obs']
    screen = monthly.loc[(monthly.model == 'student') & (monthly.n_methods == 3)
                         & (monthly.method_spread <= 2.5) & (monthly.mispricing < -threshold)
                         & monthly.price_252.notna()]
    example = screen.sort_values(['as_of', 'rank']).iloc[0]
    book.head(f'Monthly mark and weekly decision  /  {example.ticker}')
    rows = [[name, num(example[f'{key}_low']), num(example[f'{key}_value']), num(example[f'{key}_high'])]
            for name, key in [('DCF', 'dcf'), ('Own EV/EBITDA', 'ev'), ('Peer P/E', 'peer')]]
    rows.append(['Composite', num(example.low), num(example.fair_value), num(example.high)])
    book.table([f'Monthly mark as of {example.as_of}', 'Low', 'Center', 'High'], rows,
               [255, 89, 89, 89], right=(1, 2, 3), highlights=(3,), row_h=17.5)
    weekly = data['dev']['weekly_marks']
    next_date = weekly.loc[weekly.rebalance > example.decision_date, 'rebalance'].min()
    mark = weekly.loc[(weekly.rebalance == next_date) & (weekly.ticker == example.ticker)].iloc[0]
    excluded = mark.n_methods >= 2 and mark.method_spread <= 2.5 and mark.mispricing < -threshold
    disposition = 'excluded the candidate' if excluded else 'kept the candidate eligible'
    book.par(f'The {example.decision_date} monthly evaluation used spot {num(example.price)} and implied {pct(example.mispricing)} mispricing. The price 252 sessions later was {num(example.price_252)}. At the next weekly rebalance on {mark.rebalance}, the screen used the {mark.as_of} close. It {disposition} with {pct(mark.mispricing)} mispricing and {num(mark.method_spread)}× disagreement across {int(mark.n_methods)} methods.', size=10.1)


def signals_page(book, p, d):
    book.page_start('Single-module evaluation', f'{SPLITS[p]} signal quality', 'The experiment values the top 20 growth candidates on each month’s first trading session. The previous equity session is the cutoff for dated inputs.')
    sums=d['sums']; rows=[]
    for h in (63,126,252):
        for model in ('student','trailing_pe'):
            r=sums.loc[model,h]
            rows.append([h,MODELS[model],int(r.valued_with_return),int(r.ic_months),num(r.mean_ic,3),num(r.t_ordinary),num(r.t_hac)])
    book.table(['Sessions','Model','Pairs','Months','Mean IC','Naive t','HAC t'],rows,[63,110,70,68,75,68,68],right=(2,3,4,5,6),highlights=(0,2,4),row_h=19)
    book.note('IC is the equal-weighted average of monthly Spearman correlations. Months need at least five valid pairs and variation in both variables. The ordinary t treats months as independent. Bartlett HAC uses 2, 5 or 11 monthly lags for overlapping horizons. N/a denotes insufficient months for the stated lag.')
    book.head('Hold the candidate set fixed')
    rows=[]
    for h in (63,126,252):
        x,y=sums.loc['student_matched',h],sums.loc['trailing_pe_matched',h]
        rows.append([h,int(x.valued_with_return),num(x.mean_ic,3),num(y.mean_ic,3),num(x.mean_ic-y.mean_ic,3,True)])
    book.table(['Sessions','Matched pairs','Composite IC','Course P/E IC','IC difference'],rows,[75,106,111,117,113],right=(1,2,3,4),row_h=18)
    if p in ('val', 'test'):
        matched,reference = sums.loc['student_matched',252],sums.loc['trailing_pe_matched',252]
        comparison = (f'On the same {int(matched.valued_with_return)} one-year stock-date pairs, '
                      f'IC was {num(matched.mean_ic,3)} for the composite and {num(reference.mean_ic,3)} for course P/E. ')
        if p == 'val':
            comparison += 'Course P/E ranks ahead on this shared sample. The model-specific samples above give the reverse ordering.'
        else:
            comparison += 'The modest ranking advantage applies to stocks both methods could value.'
        book.note(comparison)
    else:
        book.note('Matching uses the identical stock and decision date in both models. Differences in coverage can otherwise change the companies included in a comparison.')
    book.head('One-year bands and individual methods')
    rows=[]
    for model in ('student','trailing_pe','student_dcf','student_ev','student_peer'):
        r=sums.loc[model,252]
        rows.append([MODELS[model],f'{int(r.valued)}/{int(r.candidates)}',num(r.mean_ic,3),int(r.interval_pairs),pct(r.interval_coverage),pct(r.median_signed_error),pct(r.median_abs_error)])
    book.table(['Model','Valued / asked','IC','Band pairs','Inside band','Med. error','Med. |err.|'],rows,[98,81,51,66,81,74,71],right=(1,2,3,4,5,6),highlights=(0,),fs=8.5,row_h=18)
    r=sums.loc['student',252]; miss=50-100*r.interval_coverage
    hits = round(r.interval_coverage * r.interval_pairs)
    book.par(f'The composite band contained {hits} of {int(r.interval_pairs)} one-year prices ({pct(r.interval_coverage)}), {abs(miss):.1f} pp {"below" if miss>=0 else "above"} the 50% target. Median signed error was {pct(r.median_signed_error)} (fair value divided by realized price, minus one). The scenario range understated the observed uncertainty in future prices.',size=10.1)
    book.note('Both endpoint prices must be observed on the exact SPY session; the horizon must mature inside the same split. This prevents validation labels from reaching the holdout. It also changes the effective sample relative to the course script, which accepts the last observed price within a forward window.')


def portfolio_page(book, p, d, source):
    abb=d['port']; a,b=abb.loc['A_selection'],abb.loc['B_valuation']; bp=list(benchmarks(source,d))
    book.page_start('Portfolio evaluation',f'{SPLITS[p]} portfolio outcomes',f'{a.start} to {a.end}. The four ablation cells replay identical precomputed monthly scores, execution rules and trading costs.')
    rows=[]
    for label,r in [(NAMES[k],abb.loc[k]) for k in NAMES]+[(t,r) for t,r,_ in bp]:
        rows.append([label,*[num(r[key],3 if key=='sharpe' else 2) for key in ('total_return_pct','cagr_pct','volatility_pct','sharpe','max_drawdown_pct','turnover_per_year')]])
    book.table(['Strategy','Return %','CAGR %','Vol. %','Sharpe','MDD %','Turnover ×'],rows,[111,72,67,62,62,69,79],right=(1,2,3,4,5,6),highlights=(1,),row_h=19)
    book.note('Return is the last closing NAV divided by the first, minus one. Sharpe uses daily close-to-close returns (zero risk-free rate, 252 sessions per year). Turnover is traded value divided by average equity and years. SPY and QQQ use adjusted closes.')
    book.par(f'B minus A changes cumulative return by {num(b.total_return_pct-a.total_return_pct,2,True)} pp, Sharpe by {num(b.sharpe-a.sharpe,3,True)}, and maximum drawdown by {num(b.max_drawdown_pct-a.max_drawdown_pct,2,True)} pp. Annual turnover changes by {num(b.turnover_per_year-a.turnover_per_year,2,True)}×. Full exposure makes C and A identical, and D and B identical. The corresponding timing contrast and valuation–timing interaction are zero.')
    if p == 'val':
        book.note('The last DISCK close in the course data was $24.42 on April 8, 2022. For the remaining 433 validation sessions, the course backtester carried 477 shares in A and 487 in B at that close. Both reported drawdowns include this fixed valuation.')
    if p=='test':
        costs=d['cost']; ca=costs['A_selection']; cb=costs['B_valuation']
        book.note(f'At 6 bp per traded dollar, the recorded trade notionals imply ${ca:,.0f} of costs in A and ${cb:,.0f} in B, from $100,000 of starting capital. These are whole-period dollar amounts derived from rounded ledger notionals. The NAVs already include them.')
    book.head('Relative to the equity benchmarks')
    book.chart({'Selection':d['curves']['A_selection'],'Filtered':d['curves']['B_valuation'],**{t:s for t,_,s in bp}},{'Selection':BLUE,'Filtered':TEAL,'SPY':'#A58040','QQQ':'#994D67'},height=90 if p=='test' else 100)
    book.head('Year-by-year change')
    yearly=d['annual'].pivot(index='year',columns='run', values=['total_return_pct','max_drawdown_pct','sharpe']); rows=[]
    for year,series in yearly.iterrows():
        ar=series['total_return_pct',p+'_A_selection']; br=series['total_return_pct',p+'_B_valuation']
        rows.append([year,num(ar),num(br),num(br-ar,2,True),num(series['max_drawdown_pct',p+'_B_valuation']),num(series['sharpe',p+'_B_valuation'],3)])
    book.table(['Year','A return %','B return %','Delta, pp','B MDD %','B Sharpe'],rows,[65,94,94,89,93,87],right=(1,2,3,4,5),row_h=17,fs=8.9)
    chart_note='Chart base is 100 at each split’s first close. Annual calculations include the previous year’s final close within that split.'
    if p=='test':
        chart_note+=' B lagged A in 2024. Its largest annual return advantage came in 2025. The 2026 row ends October 9.'
    book.note(chart_note)


def robustness_page(book,data,threshold):
    book.page_start('Robustness','Threshold choice and stability')
    book.head('Thresholds specified before development comparisons')
    rows=[]
    for p in ('dev','val'):
        for r in data[p]['sens'].itertuples():
            rows.append([SPLITS[p],pct(r.gap,0),num(r.total_return_pct),num(r.sharpe,3),num(r.max_drawdown_pct),num(r.turnover_per_year),int(r.exclusions)])
    book.table(['Period','Gap','Return %','Sharpe','MDD %','Turnover ×','Exclusions'],rows,[98,53,73,65,70,77,86],right=(1,2,3,4,5,6),highlights=tuple(i for i,row in enumerate(rows) if row[0]=='Validation' and row[1]==pct(threshold,0)),row_h=19)
    book.par('The rule chose the highest validation Sharpe rounded to three decimals. Ties went to lower annual turnover, then proximity to 25%, then the lower gap. The 15%, 25% and 35% candidates were fixed before development comparisons. Candidate exclusions count names across weekly screens. Repeated exclusions can refer to the same company.',size=10.2)
    book.head('How the sign of the signal changed')
    rows=[]
    for p,d in data.items():
        y=d['yearly']; y=y[y.model=='student'];
        for yr,g in y.groupby('decision_year'):
            g=g.set_index('horizon'); rows.append([SPLITS[p],int(yr),num(g.at[63,'mean_ic'],3),num(g.at[126,'mean_ic'],3),num(g.at[252,'mean_ic'],3),int(g.at[252,'ic_months']),pct(g.at[252,'interval_coverage'])])
    book.table(['Period','Year','63 IC','126 IC','252 IC','252 IC months','Band hits %'],rows,[95,48,58,64,64,99,94],right=(1,2,3,4,5,6),row_h=18,fs=8.7)
    book.note('Years refer to the portfolio decision date. A missing IC can reflect too few paired stocks or labels reaching beyond the split. One-year band hits use only observations with a realized price and both interval bounds.')
    book.head('Why ranking and portfolio returns can disagree')
    book.par('A rank IC measures the entire monthly top-20 ordering. The weekly decision changes only a subset with corroborated downside and trades a ranked replacement. Position weights, compounding and transaction costs determine the portfolio return. The filter reduced the 2022 decline while lagging the 2023 rebound. It also gave up return over the full development period. The validation gain alone provides little evidence for a persistent premium.',size=10.2)


def bonus_page(book, bonus):
    book.page_start('Selection retraining bonus','Does a longer scoring window help?', 'A separate paired experiment varies how much labeled history the selector uses to choose and combine its base models.')
    arm=pd.read_csv(bonus/'portfolio_comparison.csv').set_index('arm'); ranks=pd.read_csv(bonus/'monthly_rank_comparison.csv');rows=[]
    book.par('The same-environment control retrains the original growth.json. The alternative changes only the internal validation window from three to six distinct labeled reporting dates per bucket. All six regressors, the stacking procedure and allocation rules stay fixed. Each arm refits monthly from January through June 2018 and writes its own scores.parquet. The official scores remain the input to the main ablation.')
    for key,name in [('control','Original 3 dates'),('six_dates','Longer 6 dates')]:
        r=arm.loc[key];rows.append([name,num(r.total_return_pct),num(r.cagr_pct),num(r.volatility_pct),num(r.sharpe,3),num(r.max_drawdown_pct),num(r.turnover_per_year)])
    book.table(['Refit','Return %','CAGR %','Vol. %','Sharpe','MDD %','Turnover ×'],rows,[111,70,66,62,62,67,84],right=(1,2,3,4,5,6),highlights=(1,))
    x,y=arm.loc['control'],arm.loc['six_dates'];book.par(f'From {x.start} through {x.end}, the alternative returned {num(y.total_return_pct-x.total_return_pct,2,True)} pp relative to its control and changed Sharpe by {num(y.sharpe-x.sharpe,3,True)}. Maximum drawdown changed by {num(y.max_drawdown_pct-x.max_drawdown_pct,2,True)} pp. This half-year paired refit is a development demonstration; the window is too short to establish performance through a full market cycle.')
    book.head('Which rankings actually changed?')
    rows=[[str(r.month)[:7],str(r.cutoff)[:10],int(r.growth_names),num(r.spearman,3),int(r.shared_top10),num(r.control_vs_released_spearman,3)] for r in ranks.itertuples()]
    book.table(['Month','Cutoff','Names','Arm rank corr.','Top 10 overlap','Control vs release'],rows,[68,85,52,105,93,119],right=(2,3,4,5),row_h=19,fs=8.7)
    book.note('Spearman correlations use the same growth-tech names in both arms at the same cutoff. The last column measures the change attributable to rerunning the original configuration in this environment. Overlap counts compare raw top-ten rankings before investability and weighting.')
    book.head('What still needs testing')
    book.par('The regressors select and weight their ensemble using validation errors. More reporting dates can reduce the influence of one reporting cycle, while making weights react more slowly to a regime change. Extending both refits to additional development years would test that trade-off without changing this assignment’s frozen valuation filter.')


def finish(book,data,threshold,log,freeze,out):
    book.page_start('Interpretation and reproducibility','Judgment and next experiment')
    book.head('What did not work')
    d=data['dev']['sums']; v=data['val']['sums'];diff1=100*(.5-d.loc['student',252].interval_coverage);diff2=100*(.5-v.loc['student',252].interval_coverage)
    book.par(f'Composite one-year intervals missed the 50% target by {diff1:.1f} pp on development and {diff2:.1f} pp on validation. DCF in particular produced wide errors relative to future prices while its WACC–growth scenarios covered only {pct(v.loc["student_dcf",252].interval_coverage)} of eligible one-year validation prices. The grid varies discounting and terminal growth together with fixed operating margins. It cannot represent the full distribution of future multiples or operating outcomes.')
    book.par('A prospective paper-trading study could record each forecast before the trade, then measure coverage when its one-year target matures. An interval calibrated on earlier, fully matured forecast errors could be evaluated alongside the frozen interval.')
    reference=[]
    for period in ('dev','val'):
        for label, slug in [('Composite','eval'),('Course P/E','example')]:
            reference_file=out/f'valuation_{slug}_{period}.txt'
            if not reference_file.exists(): reference_file=HERE/'output'/reference_file.name
            if reference_file.exists():
                result=reference_file.read_text()
                ranks=re.findall(r'rank IC\s+\d+ sessions: mean ([+\-\d.]+)',result)
                coverage=re.search(r'inside \[low, high\]: (\d+%) of (\d+)',result)
                if len(ranks)==3 and coverage:
                    reference.append([SPLITS[period],label,*ranks,*coverage.groups()])
    if reference:
        book.head('Cross-check with the course evaluator')
        book.table(['Period','Model','63 IC','126 IC','252 IC','1y hits','Pairs'],reference,[92,100,65,65,65,70,65],right=(2,3,4,5,6),fs=8.5,row_h=16)
        book.note('The unmodified evaluator ran on a price bundle truncated at each split end. Its last-available quote rule admits missing exact endpoints. The strict endpoint tables earlier in this report show the resulting changes in signal and coverage.', gap=6)
    val_port, dev_port = data['val']['port'], data['dev']['port']
    book.par(f'With the filter, validation maximum drawdown narrowed from {abs(val_port.loc["A_selection","max_drawdown_pct"]):.2f}% to {abs(val_port.loc["B_valuation","max_drawdown_pct"]):.2f}%. On development, Sharpe fell from {dev_port.loc["A_selection","sharpe"]:.3f} to {dev_port.loc["B_valuation","sharpe"]:.3f}.', size=10.1, gap=4)
    book.head('Personal judgment')
    personal=HERE/'MY_JUDGMENT.txt'
    if personal.is_file() and personal.read_text().strip():
        for para in personal.read_text().strip().split('\n\n'):
            book.par(escape(para),font='Times-Italic',size=12.4,leading=15,color=TEAL,gap=4)
    else: book.par('The author’s personal judgment is pending for this review edition.')
    book.head('Test protocol and evidence')
    if 'test' not in data:
        book.par(f'The course holdout covers January 2, 2024 through October 9, 2026. The current research edition contains development and validation evidence. The run log records {len(log)} holdout executions. '+('No holdout stability claim is available.' if not log else 'The logged execution has incomplete saved outputs.'))
    else:
        book.par(f'The configuration was committed locally before holdout run #{log[-1]["n"]}, the only execution in the ledger. The course had not received the hash before that run. After the test, the hash was disclosed in <link href="{DISCLOSURE_URL}" color="{TEAL}"><u>course GitHub issue #21</u></link>. The TA’s response is pending. The selected threshold stayed fixed.')
    if freeze: book.note(f'Frozen valuation configuration: <font name="Courier">{escape(freeze)}</font>.', gap=6)
    book.par('The official harness uses $100,000 initial capital, a 5% cash buffer, 1 bp commission plus 5 bp slippage on each trade, 5–15% score-based weights and whole shares. Orders fill at the open of the first session of each ISO week. Small adjustments can be skipped below both $1,000 and a 3 pp weight change. Partial sells have a five-session minimum holding period. Stops are disabled. Portfolio NAV includes simulated trading costs.')
    check=HERE/'check_submission.txt'
    if check.is_file():
        matches=re.findall(r'\d+ FAIL, \d+ WARN, \d+ PASS',check.read_text())
        if matches: book.note('Official submission check archived with the package: '+matches[-1]+'.', gap=6)
    book.note('Sources: course Design Brief and Assignment 2 README, official growth.json, DataView, FinRobot dcf.py / wacc.py and the fixed course bundle. Numeric evidence: valuation_*_summary.csv, valuation_*_audit.csv, valuation_*_observations.csv, ablation_*.csv, portfolio_by_year_*.csv, sensitivity_*.csv and bonus/results/.')


def selection_feature_page(book, results, original_scores):
    book.page_start('Selection retraining bonus', 'Does a steadier price trend help?',
                    'A second paired experiment adds one price feature to the original selector. Both arms use the original three-date validation window and refit monthly from January through June 2018.')
    book.head('What changes in the model?')
    book.par('Two stocks can finish the year with the same return after following very different paths. For each stock, sum its last four consecutive quarterly log price changes and divide by the sum of their absolute values. A steady rise scores +1, a steady fall scores −1, and offsetting moves approach zero. Flat paths and unusable histories receive zero.')
    book.par('The input uses the course’s dated quarterly prices. If a row’s modeled trading date follows the monthly cutoff, the course substitutes the last available close. Both arms use the same universe, allocation, weekly trading schedule and costs. This initial six-month comparison used only development dates and was run after the formal holdout. The formal portfolio used the released course scores.')
    book.head('Selection-only replay under identical trading rules')
    stats = pd.read_csv(results / 'portfolio_comparison.csv').set_index('arm')
    control, variant = stats.loc['control'], stats.loc['path_efficiency']
    rows = []
    for label, row in [('Original model', control), ('With trend feature', variant)]:
        rows.append([label, num(row.total_return_pct), num(row.sharpe, 3),
                     num(row.max_drawdown_pct), int(row.trades), num(row.turnover_per_year)])
    book.table(['Model', 'Return %', 'Sharpe', 'MDD %', 'Trades', 'Turnover ×'],
               rows, [130, 83, 71, 83, 65, 90], right=(1, 2, 3, 4, 5), highlights=(1,))
    book.par(f'Adding the feature changed cumulative return by {num(variant.total_return_pct-control.total_return_pct,2,True)} pp, Sharpe by {num(variant.sharpe-control.sharpe,3,True)} and signed maximum drawdown by {num(variant.max_drawdown_pct-control.max_drawdown_pct,2,True)} pp. Annualized turnover rose from {num(control.turnover_per_year)}× to {num(variant.turnover_per_year)}×.')
    book.head('Did the monthly rankings change?')
    ranks = pd.read_csv(results / 'monthly_rank_comparison.csv')
    rows = [[str(r.month)[:7], str(r.cutoff)[:10], int(r.growth_names), num(r.spearman, 3),
             f'{int(r.shared_top10)}/10'] for r in ranks.itertuples()]
    book.table(['Score month', 'Data cutoff', 'Matched stocks', 'Rank correlation', 'Top-ten overlap'],
               rows, [95, 103, 106, 113, 105], right=(2, 3, 4), row_h=18)
    def may_top_ten(path):
        frame = pd.read_parquet(path, columns=['month', 'bucket', 'symbol', 'rank_score'])
        may = frame.loc[(frame.bucket == 'growth_tech') & (pd.to_datetime(frame.month) == pd.Timestamp('2018-05-01'))]
        return set(may.sort_values(['rank_score', 'symbol'], ascending=[False, True]).head(10).symbol)
    prior = may_top_ten(original_scores)
    revised = may_top_ten(results/'path_efficiency/scores.parquet')
    book.note(f'In May, {", ".join(sorted(revised-prior))} entered the raw top ten and {", ".join(sorted(prior-revised))} dropped out. Incumbent retention and position limits apply after this ranking.')
    book.par('The feature changed the top ten and increased turnover in this half-year sample. The next page follows the same two models through seven years of monthly refits.')


def long_selection_page(book, results):
    metrics = pd.read_csv(results / 'period_metrics.csv').set_index(['period', 'arm'])
    annual = pd.read_csv(results / 'year_differences.csv').set_index('year')
    stability = pd.read_csv(results / 'stability.csv').set_index('sample')
    converted = pd.read_csv(results / 'corporate_action/period_metrics.csv').set_index(['period', 'arm'])
    book.page_start('Selection retraining bonus', 'Did the feature hold up over seven years?',
                    'Each month from January 2017 to December 2023, both selectors were fitted through the prior trading close in the same software environment. The original and feature models each had 84 monthly fits.')
    book.head('Selection alone under the course trading rules')
    rows = []
    for period, dates in [('dev', '2017–2021'), ('val', '2022–2023')]:
        for arm, name in [('control', 'Original'), ('path_efficiency', 'With feature')]:
            row = metrics.loc[period, arm]
            rows.append([dates, name, num(row.total_return_pct), num(row.cagr_pct), num(row.sharpe, 3),
                         num(-row.drawdown_depth_pct), num(row.turnover_per_year)])
    book.table(['Period', 'Refit', 'Return %', 'CAGR %', 'Sharpe', 'MDD %', 'Turnover ×'],
               rows, [75, 103, 72, 66, 62, 69, 75], right=(2, 3, 4, 5, 6), highlights=(1, 3))
    book.note('Both use the same weekly growth-stock allocation and pay 6 bp per traded dollar. Return is cumulative from the first close. Turnover is annualized gross buying and selling divided by average equity.')
    book.head('Annual net return difference in percentage points')
    spreads = annual['feature_minus_control_return_pp']
    pad = 28
    step = (BODY - 2 * pad) / len(spreads)
    high, low = spreads.max(), min(spreads.min(), 0)
    scale = 81 / (high - low)
    zero = book.y - 15 - high * scale
    book.line(LEFT + pad, zero, WIDTH - LEFT - pad, zero, color=GREY)
    for i, (year, spread) in enumerate(spreads.items()):
        center = LEFT + pad + step * (i + 0.5)
        height = spread * scale
        color = TEAL if spread >= 0 else RUST
        book.c.setFillColor(HexColor(color))
        book.c.rect(center - 12, min(zero, zero + height), 24, abs(height), fill=1, stroke=0)
        book.text(center - 13, zero + height + (4 if spread >= 0 else -10),
                  num(spread, 2, sign=True), 'Helvetica-Bold', 8.7, color)
        book.text(center - 11, zero + low * scale - 25, str(year), size=8.5, color=GREY)
    book.y = zero + low * scale - 36
    dev = stability.loc['dev']
    val = stability.loc['val']
    excluded = stability.loc['dev_ex_2018h1']
    dev_years = spreads.loc[2017:2021]
    book.par(f'The feature led in {int((dev_years > 0).sum())} of {len(dev_years)} development years '
             f'and {int(dev.positive_months)} of {int(dev.months)} months. Excluding the first half of 2018, '
             f'which motivated this follow-up, it still led in {int(excluded.positive_months)} of {int(excluded.months)} months.')
    course_gain = val.feature_return_pct - val.control_return_pct
    converted_gain = (converted.loc[('val', 'path_efficiency'), 'total_return_pct']
                      - converted.loc[('val', 'control'), 'total_return_pct'])
    book.head('What persisted in validation?')
    book.par(f'The feature reduced the two-year cumulative loss by {num(course_gain)} pp, with a shallower '
             f'drawdown and lower turnover. It led in {int(val.positive_months)} of {int(val.months)} months. '
             f'It lost less in 2022 and lagged the 2023 rebound by {num(-spreads.loc[2023])} pp.')
    book.par(f'The course replay carried the last DISCK quote after April 8, 2022. A separate replay converting '
             f'each DISCK share into one WBD share before the next open increased the validation return advantage '
             f'to {num(converted_gain)} pp. These comparisons revisit a previously examined validation period. '
             'The formal valuation study retained the issued scores and its single recorded holdout run.')
    book.note('Reproduction code and source tables are in <font name="Courier">bonus/path_efficiency/long_run/</font>.')


def sentiment_timing_page(book, stocks, training, impact):
    book.page_start('Supplementary timing research', 'Does a daily news factor improve timing?',
                    'The paired control feeds 0.5 into the course timing engine’s 12% news term. The variant feeds a dated sentiment score to the same engine. Both portfolios use the same selection, trading costs and other risk factors.')
    book.head('From historical headlines to a position')
    book.par(f'A locally trained GPT-2 small adapter follows the FinGPT financial-sentiment task. The 2019 base model and 2013 Financial PhraseBank predate the eligible news. A rank-two update learns the negative, neutral and positive answer scores from {training["train_n"]} labeled sentences. The remaining model weights stay fixed.')
    book.par(f'The sample contains {training["release_count"]} Federal Reserve headlines across {training["release_days"]} release days. Each headline scores P(positive) + 0.5 × P(neutral). Headlines on the same day are averaged. Other calendar days receive 0.5. The weekly decision averages the seven preceding days and excludes its own date. News is neutral before March 2019.')
    book.head('Paired stock and SPY replays')
    rows = []
    for period, asset in [('dev', 'stocks'), ('val', 'stocks'), ('val', 'SPY')]:
        pair = stocks[(stocks.period == period) & (stocks.portfolio == asset)].set_index('arm')
        for arm, label in [('engine', 'Neutral 0.5'), ('student', 'Dated news')]:
            r = pair.loc[arm]
            rows.append([SPLITS[period], asset.upper() if asset == 'SPY' else 'Stocks', label,
                         num(r.total_return_pct), num(r.sharpe, 3), num(r.max_drawdown_pct)])
    book.table(['Period', 'Portfolio', 'News input', 'Return %', 'Sharpe', 'MDD %'],
               rows, [97, 81, 108, 88, 70, 78], right=(3, 4, 5), highlights=(1, 3, 5), row_h=18)
    book.head('Language accuracy and investment performance')
    before, after = (training['classification'][key]['metrics'] for key in ('base', 'fine_tuned'))
    stock_val = stocks[(stocks.period == 'val') & (stocks.portfolio == 'stocks')].set_index('arm')
    book.par(f'On {training["label_holdout_n"]} held-out labeled sentences, macro F1 rose from {num(before["macro avg"]["f1-score"],3)} to {num(after["macro avg"]["f1-score"],3)}. On the growth-stock validation portfolio, Sharpe moved from {num(stock_val.at["engine","sharpe"],3)} to {num(stock_val.at["student","sharpe"],3)}. The dated factor gave no measured validation improvement on either portfolio.')
    val_impact = impact.set_index('period').loc['val']
    book.par(f'Headlines were present in {int(val_impact.weeks_with_headlines)} of {int(val_impact.weeks)} validation decision windows. The target exposure changed in {int(val_impact.weeks_with_exposure_change)} weeks. Its absolute change averaged {num(val_impact.mean_abs_exposure_delta_bp)} basis points of exposure across all weeks. Stock orders differed on {int(val_impact.stock_order_days_different)} dates.')
    book.note('The labels are balanced financial-news sentences. Federal Reserve releases cover a single institution and include many routine notices. Neutral calendar days dilute the signal before it enters the 12% term.')


def news_diagnosis_page(book, topics, calendar, statements, pilot):
    book.page_start('Supplementary news research', 'What would improve the news signal?',
                    'Two explanations suggest different experiments. The weekly average may dilute useful headlines, or the headlines themselves may omit the decision that matters.')
    book.head('Do the headlines tell us what the Fed decided?')
    monetary = topics.loc[(topics.period == 'val') & (topics.url_topic == 'monetary')].iloc[0]
    calendar_url = escape(calendar.calendar_url.iloc[0])
    book.par(f'The archive captured {int(calendar.matched_statement_url.sum())} of the {len(calendar)} scheduled statement URLs on the <link href="{calendar_url}" color="{TEAL}"><u>2022–2023 FOMC calendar</u></link>. Of {int(monetary.split_headlines)} validation headlines, {int(monetary.headlines)} came from the monetary-policy URL category, including {int(monetary.titles_containing_minutes_of)} labeled as minutes. Minutes become available on their own publication dates.')
    dates = statements.set_index('date')
    examples = [('2019-05-01', 'Held at 2.25–2.50%'), ('2019-07-31', 'Cut to 2.00–2.25%')]
    rows = [[day, action, num(dates.at[day,'value'], 6)] for day, action in examples]
    book.table(['Release date', 'Target rate in statement body', 'Model score on title'],
               rows, [112, 238, 172], right=(2,), row_h=19)
    links = ' · '.join(f'<link href="{escape(dates.at[day,"url"])}" color="{TEAL}"><u>{day}</u></link>' for day, _ in examples)
    book.note(f'Both titles read “Federal Reserve issues FOMC statement.” All {len(statements)} archived 2019 instances received the same score. The decision paragraphs identify the different actions. Federal Reserve statements: {links}.')
    book.head('What happened when empty days stopped diluting the average?')
    book.par('An exploratory replay on development dates only (2017–2021) changes one input. Within the same seven-day window, average just the dates with a release. An empty window still receives 0.5. The same stock rankings, next-open trades, costs and 12% news weight apply to all three arms.')
    pilot = pilot.set_index('policy')
    rows = []
    for policy, name in [('neutral','Neutral control'), ('seven_calendar_days','All 7 days'), ('release_days_only','Release days only')]:
        row = pilot.loc[policy]
        rows.append([name, num(row.mean_abs_exposure_delta_bp), num(row.total_return_pct),
                     num(row.sharpe, 3), num(row.max_drawdown_pct), num(row.turnover_per_year)])
    book.table(['News average', 'Shift, bp', 'Return %', 'Sharpe', 'MDD %', 'Turnover ×'],
               rows, [133, 79, 82, 65, 71, 92], right=(1,2,3,4,5), row_h=18)
    book.note('Shift is the mean absolute change in target exposure relative to the neutral control, over all weekly decisions. The larger changes in exposure coincided with a lower cumulative return. This development replay supplies no new validation evidence.')
    book.head('Comparing titles and full statements')
    book.par('The next page uses the official FOMC calendars to recover missing decisions. It pairs each title with its own full statement, uses the same classifier on both and follows the resulting portfolios through the same market dates. The comparison measures what changes when the model can read the actual decision and guidance.')


def statement_page(book, comparison, differences, events, statements):
    book.page_start('Supplementary news research', 'Does reading the statement help?',
                    'At each scheduled FOMC decision, score the headline and the full statement separately. Each portfolio uses the same stock rankings, meeting dates and weekly execution.')
    differences = differences.loc[differences.asset == 'stocks'].set_index('period')
    stock = comparison.loc[comparison.asset == 'stocks'].set_index(['period', 'policy'])
    eligible = events
    if set(eligible.date) != {row['date'] for row in statements if row['date'] >= '2019-03-01'} or eligible.title_score.nunique() != 1:
        raise ValueError('Check eligible meeting dates and title scores before publishing the comparison.')
    calendar = '<link href="https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm" color="' + TEAL + '"><u>official FOMC calendars</u></link>'
    book.par(f'The {calendar} supplied {len(statements)} scheduled statements in 2017–2023. The GPT-2 vintage rule permits scoring from March 1, 2019, leaving {len(eligible)} eligible decisions. Every eligible headline received {eligible.title_score.iloc[0]:.4f}. Full-text scores ranged from {eligible.body_score.min():.4f} to {eligible.body_score.max():.4f}.')
    book.head('Stock portfolio outcomes after trading costs')
    rows = []
    for period, years in [('dev','2017–21'), ('val','2022–23')]:
        for arm, label in [('neutral','Neutral (0.5)'), ('title','Headline'), ('body','Full statement')]:
            row = stock.loc[period,arm]
            rows.append([years, label, num(row.total_return_pct), num(row.max_drawdown_pct),
                         num(row.sharpe,3), num(row.turnover_per_year)])
    book.table(['Period','News input','Return %','MDD %','Sharpe','Turnover ×'], rows,
               [74,132,82,82,71,81], right=(2,3,4,5), highlights=(2,5), row_h=18)
    book.note('Each period starts with fresh capital. Return and maximum drawdown use daily closing equity. The stock simulation pays 6 bp per dollar traded and fills orders at the weekly open following the signal cutoff.')
    book.head('How much did the extra text change?')
    rows = []
    for period,label in [('dev','Development'), ('val','Validation')]:
        row = differences.loc[period]
        rows.append([label, int(row.eligible_meetings), int(row.different_exposure_weeks),
                     num(row.return_body_minus_title_pp,3,True), num(row.max_drawdown_body_minus_title_pp,3,True)])
    book.table(['Period','Meetings','Different targets','Return change, pp','MDD change, pp'],
               rows,[102,67,113,119,121], right=(1,2,3,4), row_h=19)
    book.note('Differences use unrounded daily equity. Positive return change means higher ending wealth. Positive MDD change means a shallower peak-to-trough loss. Different targets counts weekly rebalances where the title and full-text arms chose different exposures.')
    book.head('A development gain that faded in validation')
    book.par('On meeting weeks, the generic headline consistently shifted the news score below neutral. The full statements scored closer to neutral on average. Text changed the target at 13 of 16 validation meetings, yet the stock portfolio earned slightly less and experienced a slightly deeper drawdown. All three arms reached their validation trough on December 28, 2022.')
    book.note('Both inputs use the same frozen classifier. Scores weight text chunks by their token counts. Empty weeks revert to 0.5. The other five factors and the 12% news weight match across arms. This historical follow-up uses previously examined markets. The 2024–2026 formal holdout belongs to the original valuation study.')


def stance_page(book, comparison, differences, events, annual):
    book.page_start('Supplementary news research', 'Does the change in Fed guidance help?',
                    'A paired test of the existing full-statement sentiment model and a rule that reads changes in the policy decision and forward guidance.')
    book.par('Each scheduled statement gets two votes. A rate hike scores +1, a cut scores −1 and a hold scores zero. Stronger or softer forward guidance relative to the preceding statement scores +1 or −1. Identical guidance categories score zero. The news input is 0.50 minus 0.25 times the sum. A hawkish change therefore lowers equity exposure before the engine scales and caps it.')
    by_date = events.set_index('date')
    book.head('Two hikes with different implications')
    rows = []
    for day in ('2023-02-01', '2023-03-22'):
        row = by_date.loc[day]
        rows.append([day, num(row.rate_change_bp,0,True),
                     f'{int(row.previous_guidance_level):+d} to {int(row.guidance_level):+d}',
                     num(row.wording_component,0,True), num(row.stance_score)])
    book.table(['Statement','Rate move, bp','Guidance levels','Wording vote','News input'],
               rows, [109,102,111,108,92], right=(1,2,3,4), row_h=19)
    feb = 'https://www.federalreserve.gov/newsevents/pressreleases/monetary20230201a.htm'
    mar = 'https://www.federalreserve.gov/newsevents/pressreleases/monetary20230322a.htm'
    book.note(f'In <link href="{feb}" color="{TEAL}"><u>February</u></link> the Fed expected successive increases. In <link href="{mar}" color="{TEAL}"><u>March</u></link> the language left open whether further tightening would be needed. The softer guidance offset the rate hike in the rule. The comparison uses the immediately preceding published decision and excludes dissents from the guidance vote.')
    book.head('Validation stock portfolio, 2022–2023')
    stock = comparison.loc[comparison.period == 'val'].set_index('policy')
    rows = []
    for arm, label in [('neutral','Neutral input'), ('body','Full-text sentiment'), ('stance','Policy-change rule')]:
        row = stock.loc[arm]
        rows.append([label, num(row.total_return_pct), num(row.max_drawdown_pct), num(row.sharpe,3),
                     num(row.turnover_per_year), int(row.trades)])
    book.table(['Policy','Return %','MDD %','Sharpe','Turnover ×','Trades'],rows,
               [145,82,82,67,84,62], right=(1,2,3,4,5), highlights=(2,), row_h=18)
    val = differences.set_index('period').loc['val']
    book.par(f'Across {int(val.eligible_meetings)} validation meetings, the rule changed target exposure relative to the body model at {int(val.different_exposure_weeks)} rebalances. Cumulative return changed by {num(val.return_stance_minus_body_pp,3,True)} percentage points and signed maximum drawdown by {num(val.max_drawdown_stance_minus_body_pp,3,True)} points. A positive drawdown change means a shallower loss.')
    years = annual.set_index('year')
    book.note(f'The rule’s annual return relative to the body model changed by {num(years.at[2022,"stance_minus_body_pp"],2,True)} pp in 2022 and {num(years.at[2023,"stance_minus_body_pp"],2,True)} pp in 2023. Annual returns use the preceding year-end close when available. The stock scores, 12% news term, weekly execution and 6 bp trading cost match across arms.')
    book.head('What can explain the difference?')
    body_move = stock.at['body','mean_abs_exposure_from_neutral_bp_meeting_weeks']
    rule_move = stock.at['stance','mean_abs_exposure_from_neutral_bp_meeting_weeks']
    book.par(f'On meeting weeks the average absolute target shift from the neutral engine was {num(body_move,1)} bp for full text and {num(rule_move,1)} bp for the rule. Performance reflects both the direction and size of those trades. Measuring a policy surprise would also require investors’ expectations before the announcement.')
    counts = events.loc[events.date.str.startswith(('2022','2023')), 'stance_label'].value_counts()
    book.note(f'The {int(val.eligible_meetings)} validation decisions included {counts.get("hawkish",0)} hawkish, {counts.get("neutral",0)} neutral or offsetting, and {counts.get("dovish",0)} dovish signals. The rule was specified before this replay of previously examined markets. This sample gives limited evidence about an easing cycle.')


def amplitude_page(book, calibration, comparison, direction):
    book.page_start('Supplementary news research', 'Smaller trades, same policy signal',
                    'How much of the validation shortfall survives when the hawkish and dovish trades are sized to match the full-text model during development?')
    arms = comparison.set_index(['period','policy'])
    body, stance, scaled = (arms.loc['val',name] for name in ('body','stance','scaled'))
    dev = calibration['development']
    book.head('Set the size before the validation replay')
    book.par(f'At {dev["event_weeks"]} development meetings, the mean absolute target shift from the neutral engine was {num(dev["body_mean_abs_target_shift_bp"])} basis points for the body model and {num(dev["stance_mean_abs_target_shift_bp"])} for the policy rule. Their ratio, {num(calibration["scale"],4)}, sets the fraction of each policy tilt to keep. The neutral target and the direction of each active tilt are preserved. The fit uses targets only.')
    book.head('Validation stocks after costs, 2022–2023')
    rows = []
    for name,label in [('neutral','Neutral input'),('body','Full-text model'),('stance','Original rule'),('scaled','Scaled rule')]:
        row = arms.loc['val',name]
        rows.append([label, num(arms.loc[('dev',name),'mean_abs_target_shift_bp'],2),
                     num(row.mean_abs_target_shift_bp,3), num(row.total_return_pct,3),
                     num(abs(row.max_drawdown_pct),3), num(row.sharpe,3)])
    book.table(['Target policy','Dev shift, bp','Val shift, bp','Return %','DD depth %','Sharpe'], rows,
               [121,84,81,82,86,68], right=(1,2,3,4,5), highlights=(3,), row_h=19)
    book.note('Shift is the average absolute distance from the neutral exposure target over meeting weeks. The average includes weeks at the exposure cap. Drawdown is the largest percentage loss from a previous closing peak. All arms use the same stock scores and weekly execution with 6 bp of costs per traded dollar.')
    book.head('Size and direction')
    rows = []
    for value, label in [(stance,'Original policy rule'),(body,'Full-text model')]:
        rows.append([label, num(scaled.total_return_pct-value.total_return_pct,3,True),
                     num(abs(scaled.max_drawdown_pct)-abs(value.max_drawdown_pct),3,True)])
    book.table(['Scaled rule compared with','Return difference, pp','Drawdown depth difference, pp'],
               rows,[180,148,194], right=(1,2), row_h=18)
    ratio = scaled.mean_abs_target_shift_bp / body.mean_abs_target_shift_bp
    recovery = (scaled.total_return_pct-stance.total_return_pct) / (body.total_return_pct-stance.total_return_pct)
    book.note(f'Shrinking the trades closed {recovery:.0%} of the original return gap to the body model. A negative drawdown-depth difference means a smaller loss. Despite the development match, the scaled rule’s average validation tilt was still {ratio:.2f} times the body model’s.')
    diag = direction.set_index('period')
    development, validation = diag.loc['dev'], diag.loc['val']
    book.par(f'Only {int(development.actionable_meetings)} development meetings produced an active policy tilt, both increases. In validation, all {int(validation.actionable_meetings)} active signals reduced exposure. The neutral portfolio’s tradable stocks fell at {int(validation.right)} of those meetings and rose at {int(validation.wrong)}, measured from the trade open to the next weekly open. Their average return was {num(-validation.mean_signed_basket_return_pct,2,True)}%. The rule therefore cut exposure ahead of gains on average.')
    book.note('The direction check uses invested stocks with quoted opening prices and excludes DISCK when its open is missing. The net portfolio replay retains the course’s whole-share, cost and carry rules. Nine active cuts and different validation trade sizes limit how much of the remaining gap can be assigned to direction.')


def direction_swap_page(book, comparison, weekly, audit):
    book.page_start('Supplementary news research', 'Does changing only direction help?',
                    'The full-text model determines the size of each meeting’s news trade. The policy interpretation determines its sign. A neutral policy call removes the news trade.')
    scores = comparison.set_index('policy')
    body, active, swap = (scores.loc[policy] for policy in ('body','body_active','swap'))
    events = weekly.loc[weekly.meetings > 0]
    count = events.action.value_counts()
    book.head('Execution on the same FOMC decisions')
    book.par(f'For each of the {len(events)} validation statements, measure the full-text target’s absolute distance from the neutral-news target. Subtract that distance on a hawkish call and add it on a dovish call. A neutral call restores the ordinary neutral-news target, allowing other factors and the selector to rebalance as usual.')
    book.par(f'The rule reversed {count.get("reverse",0)} nonzero body tilts, retained {count.get("retain",0)}, and canceled {count.get("abstain",0)} on neutral calls. At {count.get("zero_magnitude",0)} meetings the body model already matched neutral. The reversal and retention trades used the body model’s original magnitude at each meeting.')
    book.head('Validation stock portfolio after costs')
    rows = []
    for arm,label in [('neutral','Neutral input'),('body','Full text, all calls'),
                      ('body_active','Full text, active calls'),('swap','Policy sign, active calls')]:
        row = scores.loc[arm]
        rows.append([label, num(row.mean_abs_target_shift_bp,3), num(row.total_return_pct,5),
                     num(abs(row.max_drawdown_pct),5), num(row.sharpe,4), int(row.trades)])
    book.table(['Targets','Avg shift, bp','Return %','DD depth %','Sharpe','Trades'],
               rows,[164,80,80,91,59,48],right=(1,2,3,4,5),highlights=(3,),row_h=19)
    book.note('Avg shift is the absolute distance from neutral averaged over all 16 meetings. Each portfolio uses the same stock rankings, trading rules and 6 bp cost per dollar traded. Maximum drawdown depth measures the largest peak-to-trough decline in closing equity.')
    book.head('Separate the neutral calls from the sign reversals')
    rows = []
    for label, new, old in [('Skip neutral calls',active,body),('Reverse direction on active calls',swap,active),
                            ('Both changes vs. original',swap,body)]:
        rows.append([label, num(new.total_return_pct-old.total_return_pct,5,True),
                     num(abs(new.max_drawdown_pct)-abs(old.max_drawdown_pct),5,True)])
    book.table(['Change','Return difference, pp','DD depth difference, pp'],rows,
               [212,143,167],right=(1,2),highlights=(2,),row_h=18)
    book.note('A positive return change or a negative drawdown-depth change favors the new policy. The middle row compares the same active dates at equal size and holds neutral dates fixed.')
    result = audit.iloc[0]
    book.head('Did the reversals anticipate the weekly stock moves?')
    book.par(f'All {int(result.reversed)} reversals turned an increase into a reduction. Measured open to open, the neutral portfolio’s tradable stock basket fell {int(result.swap_correct_on_flips)} times and rose {int(result.swap_wrong_on_flips)} times on those dates. The pure reversals earned {num(100*(swap.total_return_pct-active.total_return_pct),2)} bp of additional cumulative return in the whole-share portfolio. Maximum drawdown was marginally deeper.')
    book.note('Whole-share fills and the positions carried to later rebalances affect the net portfolio results. This previously examined, tightening-heavy period included no net dovish call. The near-even directional tally supplies little evidence for other cycles.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=SOURCE,help='Course source_code with the issued data directory')
    parser.add_argument('--out',type=Path,help='Saved experimental output directory')
    parser.add_argument('--bonus',type=Path,help='Paired-retraining results directory')
    parser.add_argument('--feature-bonus', type=Path, default=HERE/'bonus/path_efficiency/results',
                        help='Independent selection-feature retraining results')
    parser.add_argument('--long-bonus', type=Path,
                        help='Seven-year selection-feature tables (default: long_run/results beside the feature results)')
    parser.add_argument('--timing-archive', type=Path, default=HERE/'timing_extensions.zip',
                        help='Packaged optional timing experiments and dev/val tables')
    parser.add_argument('--freeze',help='Recorded configuration hash')
    parser.add_argument('--pdf',type=Path,default=HERE/'Research_Report.pdf')
    args=parser.parse_args(); source=args.source.resolve()
    out=args.out or (HERE/'output' if (HERE/'output/ablation_val.csv').is_file() else source/'output')
    bonus=args.bonus or next((p for p in (HERE/'bonus/results',HERE/'selection_bonus/results',HERE.parents[1]/'work/bonus/results') if (p/'portfolio_comparison.csv').is_file()),None)
    if not bonus: parser.error('Specify the completed paired comparison directory with --bonus')
    freeze=args.freeze or ((HERE/'CONFIG_COMMIT.txt').read_text().strip() if (HERE/'CONFIG_COMMIT.txt').exists() else '')
    log=read_runs(out/'test_runs.jsonl')
    if log and (len(log) != 1 or log[0]['n'] != 1 or log[0]['git_head'] != freeze):
        raise ValueError('The recorded run number or code hash differs from the frozen single-run protocol.')
    data=discover(out,log)
    match=re.search(r'^FILTER_THRESHOLD\s*=\s*([\d.]+)',(HERE/'student_valuation.py').read_text(),re.M)
    threshold=float(match.group(1)); book=Paper(args.pdf)
    cover(book,data,threshold,freeze,log);data_page(book,data);dcf_page(book);methods_page(book,data,threshold)
    for p,d in data.items(): signals_page(book,p,d)
    for p,d in data.items(): portfolio_page(book,p,d,source)
    robustness_page(book,data,threshold);bonus_page(book,bonus);finish(book,data,threshold,log,freeze,out)
    selection_feature_page(book, args.feature_bonus, bonus/'control/scores.parquet')
    long_selection_page(book, args.long_bonus or args.feature_bonus.parent/'long_run/results')
    with ZipFile(args.timing_archive) as archive:
        def archived_csv(name):
            with archive.open('extensions/' + name) as file:
                return pd.read_csv(file)
        with archive.open('extensions/fingpt/results/training.json') as file:
            training = json.load(file)
        sentiment_timing_page(book, archived_csv('fingpt/results/paired_comparison.csv'), training,
                              archived_csv('fingpt/results/signal_impact.csv'))
        news_diagnosis_page(book, archived_csv('fingpt/results/corpus_topics.csv'),
                            archived_csv('fingpt/results/corpus_calendar_coverage.csv'),
                            archived_csv('fingpt/results/corpus_repeated_fomc.csv'),
                            archived_csv('fingpt/release_day_pilot/comparison.csv'))
        study = 'fingpt/statement_comparison/'
        with archive.open('extensions/' + study + 'statements.jsonl') as file:
            statements = [json.loads(line) for line in file]
        statement_page(book, archived_csv(study + 'results/comparison.csv'),
                       archived_csv(study + 'results/body_vs_title.csv'), archived_csv(study + 'meeting_scores.csv'), statements)
        stance = 'fingpt/stance_comparison/'
        stance_page(book, archived_csv(stance + 'results/comparison.csv'),
                    archived_csv(stance + 'results/stance_vs_body.csv'), archived_csv(stance + 'meeting_scores.csv'),
                    archived_csv(stance + 'results/validation_years.csv'))
        amplitude = 'fingpt/stance_amplitude/'
        with archive.open('extensions/' + amplitude + 'calibration.json') as file:
            calibration = json.load(file)
        amplitude_page(book, calibration, archived_csv(amplitude + 'results/comparison.csv'),
                       archived_csv(amplitude + 'results/direction_summary.csv'))
        swap = 'fingpt/direction_swap/results/'
        direction_swap_page(book, archived_csv(swap + 'comparison.csv'), archived_csv(swap + 'weekly_decisions.csv'),
                            archived_csv(swap + 'audit_summary.csv'))
    book.close();print(f'{args.pdf} ({book.page} pages)')


if __name__=='__main__':
    main()
