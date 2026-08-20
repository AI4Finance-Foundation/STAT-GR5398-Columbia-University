"""
scripts/run_ml_selection.py

Light-weight ML-driven stock selection and backtest.
- Uses `outputs/final_ratios.csv` as input (preprocessed fundamentals)
- Uses LightGBM to predict next-quarter return (y_return) in a rolling fashion
- At each report date, trains on past data, predicts current date, selects top 25% by predicted return
- Equal-weight long-only portfolio until next rebalance
- Writes outputs to outputs/: portfolio_nav_ml.csv, stock_selected_ml.csv, backtest_ml_report.json

This is a faster, single-universe ML alternative to the notebook sector-by-sector training.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

try:
    from lightgbm import LGBMRegressor
    ML_BACKEND = 'lgbm'
except Exception:
    # Fall back to scikit-learn's RandomForest if LightGBM isn't installed.
    from sklearn.ensemble import RandomForestRegressor
    ML_BACKEND = 'rf'

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
FINAL_RATIOS = os.path.join(OUT_DIR, 'final_ratios.csv')
PRICE_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'daily.csv')

START = '2018-01-01'
END = '2025-12-31'
FEE_RATE = 0.001
TOP_PCT = 0.25

os.makedirs(OUT_DIR, exist_ok=True)


def load_data():
    fr = pd.read_csv(FINAL_RATIOS)
    fr['date'] = pd.to_datetime(fr['date'])
    fr = fr[(fr['date'] >= START) & (fr['date'] <= END)]

    price_cols = ['tic', 'datadate', 'prccd', 'ajexdi']
    dp = pd.read_csv(PRICE_FILE, usecols=price_cols)
    dp['datadate'] = pd.to_datetime(dp['datadate'])
    # If ajexdi is already adjusted close (yfinance Adj Close), use it directly.
    # Otherwise, treat it as a factor and multiply.
    ratio = (dp['ajexdi'] / dp['prccd']).replace([np.inf, -np.inf], np.nan).median()
    if pd.notna(ratio) and 0.5 <= ratio <= 1.5:
        dp['close_adj'] = dp['ajexdi']
    else:
        dp['close_adj'] = dp['prccd'] * dp['ajexdi']
    return fr, dp


def prepare_features(fr):
    # features: use numeric columns except identifiers and target
    exclude = {'date','gvkey','tic','gsector','y_return','reportdate'}
    features = [c for c in fr.columns if c not in exclude and np.issubdtype(fr[c].dtype, np.number)]
    return features


def run_ml_selection():
    fr, dp = load_data()
    features = prepare_features(fr)
    print(f"Using features: {features}")

    # pivot price
    price_pivot = dp.pivot_table(values='close_adj', index='datadate', columns='tic', aggfunc='mean').sort_index()
    price_pivot = price_pivot[(price_pivot.index >= START) & (price_pivot.index <= END)]
    # compute daily returns and cap extreme moves to avoid numerical explosions
    ret = price_pivot.pct_change()
    ret = ret.clip(-0.5, 0.5).fillna(0.0)

    # dates to trade
    trade_dates = sorted(fr['date'].unique())

    df_predict = pd.DataFrame(columns=sorted(fr['tic'].unique()))
    selected_dict = {}

    # Rolling training: for each trade date i starting from 8 (to have some history)
    min_history = 8
    for idx in range(min_history, len(trade_dates)):
        train_dates = trade_dates[:idx]
        trade_date = trade_dates[idx]
        # train on all rows with date in train_dates
        train_df = fr[fr['date'].isin(train_dates)].dropna(subset=features + ['y_return'])
        if train_df.shape[0] < 50:
            continue
        X_train = train_df[features]
        y_train = train_df['y_return']

        if ML_BACKEND == 'lgbm':
            model = LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        # prepare trade data
        trade_df = fr[fr['date'] == trade_date]
        X_trade = trade_df[features].fillna(0.0)
        preds = model.predict(X_trade)
        trade_df = trade_df.copy()
        trade_df['pred'] = preds

        # save preds into df_predict
        row = pd.Series(index=df_predict.columns, dtype=float)
        for tic, p in zip(trade_df['tic'], trade_df['pred']):
            row.at[tic] = p
        df_predict = pd.concat([df_predict, row.to_frame().T])

        # select top pct
        k = max(1, int(len(trade_df) * TOP_PCT))
        sel = trade_df.sort_values('pred', ascending=False).head(k)['tic'].tolist()
        selected_dict[trade_date] = sel

    # build weights and backtest
    all_dates = ret.index
    all_tics = ret.columns
    weights = pd.DataFrame(0.0, index=all_dates, columns=all_tics)
    sorted_dates = sorted(selected_dict.keys())
    # helper: align rebalance date to next trading day in price index
    def align_trade_date(d):
        pos = all_dates.get_indexer([d], method='bfill')[0]
        if pos == -1:
            return None
        return all_dates[pos]

    for i, d in enumerate(sorted_dates):
        tickers = selected_dict[d]
        if len(tickers) == 0:
            continue
        weight = 1.0 / len(tickers)
        start = align_trade_date(d)
        if start is None:
            continue
        # end is the aligned date of next rebalance or final date + 1
        next_d = sorted_dates[i+1] if i+1 < len(sorted_dates) else None
        end = align_trade_date(next_d) if next_d is not None else (all_dates[-1] + pd.Timedelta(days=1))
        if end is None:
            end = all_dates[-1] + pd.Timedelta(days=1)
        mask = (weights.index >= start) & (weights.index < end)
        for t in tickers:
            if t in weights.columns:
                weights.loc[mask, t] = weight

    weights = weights.reindex(columns=all_tics, fill_value=0.0)
    # ensure weights and returns align; missing returns are treated as 0
    ret = ret.reindex(columns=weights.columns).fillna(0.0)
    gross_ret = (weights * ret).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    cost = FEE_RATE * turnover
    cost.iloc[0] = 0.0
    net_ret = gross_ret - cost
    nav = (1 + net_ret).cumprod()
    nav.iloc[0] = 1.0

    # metrics
    cumulative_return = float(nav.iloc[-1] - 1.0)
    ann_return = float(nav.iloc[-1] ** (252.0 / len(nav)) - 1.0)
    ann_vol = float(net_ret.std(ddof=0) * np.sqrt(252))
    sharpe = ann_return / ann_vol if ann_vol > 0 else None
    rolling_max = nav.cummax()
    drawdown = (nav - rolling_max) / rolling_max
    max_dd = float(drawdown.min())

    # outputs
    pd.DataFrame({'date': nav.index, 'nav': nav.values}).to_csv(os.path.join(OUT_DIR, 'portfolio_nav_ml.csv'), index=False)
    rows = []
    for d, tics in selected_dict.items():
        for t in tics:
            rows.append({'trade_date': d, 'tic': t})
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, 'stock_selected_ml.csv'), index=False)

    report = {
        'cumulative_return': cumulative_return,
        'annualized_return': ann_return,
        'annualized_volatility': ann_vol,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'start_date': str(nav.index[0].date()),
        'end_date': str(nav.index[-1].date()),
        'n_rebalances': len(selected_dict)
    }

    with open(os.path.join(OUT_DIR, 'backtest_ml_report.json'), 'w') as f:
        json.dump(report, f, indent=2)

    print('ML Backtest complete. Metrics:')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    run_ml_selection()
