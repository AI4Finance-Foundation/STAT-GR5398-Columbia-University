"""
scripts/run_selection_backtest.py

Simple, fast stock selection + backtest using processed fundamentals in outputs/final_ratios.csv

Strategy implemented (deterministic, no ML to speed up):
- At each report date in `final_ratios.csv` (within 2018-01-01 to 2025-12-31), compute a composite score per stock as the average of z-scores
  of a small set of quality/value features (OPM, ROA, ROE, EPS, BPS, inverse PE where available).
- Select top 25% stocks by score each rebalance date.
- Equal-weight long-only portfolio held until next rebalance.
- Use daily adjusted close returns from `data/daily.csv` (close / ajexdi) to compute daily returns.
- Apply turnover fee 0.1% at rebalances (same as notebook default 0.001).

Outputs written to `outputs/`:
- portfolio_nav.csv : daily NAV series
- stock_selected.csv : list of selected stocks per trade date
- backtest_report.json : metrics (cumulative return, annualized return, annualized vol, sharpe, max drawdown)

This script is intentionally conservative and fast compared to full ML training.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
FINAL_RATIOS = os.path.join(OUT_DIR, 'final_ratios.csv')
PRICE_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'daily.csv')

START = '2018-01-01'
END = '2025-12-31'
FEE_RATE = 0.001  # 0.1%
TOP_PCT = 0.25

os.makedirs(OUT_DIR, exist_ok=True)


def load_data():
    fr = pd.read_csv(FINAL_RATIOS)
    fr['date'] = pd.to_datetime(fr['date'])
    fr = fr[(fr['date'] >= START) & (fr['date'] <= END)]

    price_cols = ['tic', 'datadate', 'prccd', 'ajexdi']
    dp = pd.read_csv(PRICE_FILE, usecols=price_cols)
    dp['datadate'] = pd.to_datetime(dp['datadate'])
    # adjusted close
    # In this repo, `ajexdi` comes from yfinance's Adj Close column.
    # If `ajexdi` looks like an adjusted close (ratio close_adj/close ~ 1), use it directly.
    # Otherwise, treat it as an adjustment factor and multiply.
    ratio = (dp['ajexdi'] / dp['prccd']).replace([np.inf, -np.inf], np.nan).median()
    if pd.notna(ratio) and 0.5 <= ratio <= 1.5:
        dp['close_adj'] = dp['ajexdi']
    else:
        dp['close_adj'] = dp['prccd'] * dp['ajexdi']
    return fr, dp


def compute_scores(fr):
    # features to use (if missing, skip);
    features = ['OPM', 'ROA', 'ROE', 'EPS', 'BPS', 'pe']
    # We'll use inverse PE (lower PE better)
    fr2 = fr.copy()
    fr2['ipe'] = 1.0 / fr2['pe'].replace(0, np.nan)
    # compute per-date z-scores for available features
    score_rows = []
    grouped = fr2.groupby('date')
    for date, g in grouped:
        df = g.set_index('tic')
        vals = pd.DataFrame(index=df.index)
        # collect available features
        comp_feats = []
        for f in ['OPM','ROA','ROE','EPS','BPS','ipe']:
            if f in df.columns:
                s = df[f]
                if s.notna().sum() < 3:
                    continue
                z = (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0)!=0 else 1)
                vals[f] = z
                comp_feats.append(f)
        if vals.shape[1] == 0:
            continue
        vals['score'] = vals.mean(axis=1)
        for tic, row in vals.iterrows():
            score_rows.append({'date': date, 'tic': tic, 'score': row['score']})
    scores = pd.DataFrame(score_rows)
    return scores


def select_top(scores):
    # For each date, pick top TOP_PCT tickers
    sel = {}
    for date, g in scores.groupby('date'):
        g2 = g.sort_values('score', ascending=False)
        k = max(1, int(len(g2) * TOP_PCT))
        sel[date] = list(g2.head(k)['tic'].values)
    return sel


def build_weights_indexed(sel, all_dates, all_tics):
    # create weights DataFrame indexed by all_dates and columns all_tics
    w = pd.DataFrame(0.0, index=all_dates, columns=all_tics)
    sorted_dates = sorted(sel.keys())
    for i, d in enumerate(sorted_dates):
        tickers = sel[d]
        if len(tickers) == 0:
            continue
        weight = 1.0 / len(tickers)
        # effective from date d until next rebalance
        start = d
        end = sorted_dates[i+1] if i+1 < len(sorted_dates) else all_dates[-1] + pd.Timedelta(days=1)
        mask = (w.index >= start) & (w.index < end)
        for t in tickers:
            if t in w.columns:
                w.loc[mask, t] = weight
    return w


def run_backtest():
    fr, dp = load_data()
    scores = compute_scores(fr)
    sel = select_top(scores)

    # build price pivot
    price_pivot = dp.pivot_table(values='close_adj', index='datadate', columns='tic', aggfunc='mean')
    price_pivot = price_pivot.sort_index()
    # restrict to date range
    price_pivot = price_pivot[(price_pivot.index >= START) & (price_pivot.index <= END)]

    # daily returns
    # avoid division by zero or infinities: replace non-positive prices with NaN before pct_change
    price_clean = price_pivot.replace([0, -0], np.nan)
    ret = price_clean.pct_change()
    # drop columns that are all-NaN
    ret = ret.dropna(axis=1, how='all')
    # fill remaining NaNs with 0 (no return)
    ret = ret.fillna(0.0)
    # cap extreme returns to avoid numerical overflow
    ret = ret.clip(lower=-0.5, upper=0.5)

    all_dates = ret.index
    all_tics = ret.columns

    weights = build_weights_indexed(sel, all_dates, all_tics)

    # align columns
    weights = weights.reindex(columns=all_tics, fill_value=0.0)

    # gross returns
    gross_ret = (weights * ret).sum(axis=1)

    # turnover: sum abs diff across columns per date
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    cost = FEE_RATE * turnover
    cost.iloc[0] = 0.0

    net_ret = gross_ret - cost
    nav = (1 + net_ret).cumprod()
    nav.iloc[0] = 1.0

    # metrics
    total_days = (nav.index[-1] - nav.index[0]).days
    cumulative_return = nav.iloc[-1] - 1.0
    ann_return = nav.iloc[-1] ** (252.0 / len(nav)) - 1.0
    ann_vol = net_ret.std(ddof=0) * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
    # max drawdown
    rolling_max = nav.cummax()
    drawdown = (nav - rolling_max) / rolling_max
    max_dd = drawdown.min()

    # prepare outputs
    nav_df = pd.DataFrame({'date': nav.index, 'nav': nav.values})
    nav_df.to_csv(os.path.join(OUT_DIR, 'portfolio_nav.csv'), index=False)

    # stock selected list
    rows = []
    for d, tics in sel.items():
        for t in tics:
            rows.append({'trade_date': d, 'tic': t})
    sel_df = pd.DataFrame(rows)
    sel_df.to_csv(os.path.join(OUT_DIR, 'stock_selected.csv'), index=False)

    report = {
        'cumulative_return': float(cumulative_return),
        'annualized_return': float(ann_return),
        'annualized_volatility': float(ann_vol),
        'sharpe_ratio': float(sharpe) if not np.isnan(sharpe) else None,
        'max_drawdown': float(max_dd),
        'start_date': str(nav.index[0].date()),
        'end_date': str(nav.index[-1].date()),
        'n_rebalances': len(sel)
    }

    with open(os.path.join(OUT_DIR, 'backtest_report.json'), 'w') as f:
        json.dump(report, f, indent=2)

    print('Backtest complete. Metrics:')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    run_backtest()
