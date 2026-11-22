"""Score PIT valuations on the course's monthly top 20, using split-local labels.

From source_code/, a batch runner supplies the loaded course bundle and scores.
Only the evaluator sees outcomes. Within each split a 63/126/252-session return
requires positive, finite closes on both exact dates. Unmatured or missing
outcomes are left missing. Returns, coverage, mispricing, and errors are fractions.
"""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_valuation import HORIZONS, check_frame, top_candidates
from shared.data_view import DataView, HIDDEN_FUNDAMENTAL_COLUMNS
from shared.guard import student_call
from shared.periods import PERIODS
from valuation import student_valuation
from valuation.examples import trailing_pe_value

METHODS = ('dcf', 'ev', 'peer')


def _candidate_inputs(view, tickers, members, cutoff):
    """Audit what the model could see. Filing counts here concern candidates only."""
    as_of = view.as_of
    if pd.Timestamp(cutoff) > as_of:
        raise ValueError(f'Selection cutoff after valuation date {as_of.date()}')
    filings = view.fundamentals(tickers)
    if set(filings) & set(HIDDEN_FUNDAMENTAL_COLUMNS):
        raise ValueError('Label columns are visible to the student')
    duplicates = filings.duplicated(['ticker', 'datadate']).sum()
    if duplicates:
        raise ValueError('DataView delivered duplicate company-quarter filings')
    for col in ('accepted_date', 'filing_date', 'datadate'):
        filings[col] = pd.to_datetime(filings[col], errors='coerce')
    visible_date = filings.accepted_date.fillna(filings.filing_date)
    if (visible_date > as_of).any() or (filings.datadate > as_of).any():
        raise ValueError('DataView delivered a future filing')
    current = filings.sort_values('datadate').groupby('ticker').tail(1).set_index('ticker')
    closing = view.close(tickers, lookback=252)
    rf = view.macro('DGS10').dropna()
    if closing.index.max() > as_of or (len(rf) and rf.index.max() > as_of):
        raise ValueError('Market inputs extend beyond the decision date')
    out = pd.DataFrame(index=pd.Index(tickers, name='ticker'))
    out['rank'] = np.arange(1, len(out) + 1)
    out['in_pit_universe'] = out.index.isin(members)
    if not out.in_pit_universe.all():
        raise ValueError('Candidate outside the point-in-time index universe')
    out['universe_size'] = len(members)
    out['selection_cutoff'] = pd.Timestamp(cutoff)
    out['visible_quarters'] = filings.groupby('ticker').size().reindex(tickers, fill_value=0)
    out['latest_fiscal_period'] = current.datadate
    out['latest_quarter_age_days'] = (as_of - out.latest_fiscal_period).dt.days
    out['latest_quarter_accepted'] = current.accepted_date
    out['latest_quarter_filed'] = current.filing_date
    out['latest_four_complete_eps'] = (filings.sort_values('datadate').groupby('ticker').tail(4)
                                       .groupby('ticker').EPS.count().reindex(tickers, fill_value=0).eq(4))
    out['undated_visible_filings'] = (visible_date.isna().groupby(filings.ticker).sum()
                                      .reindex(tickers, fill_value=0))
    out['last_observed_close_date_252s'] = pd.to_datetime(closing.apply(lambda x: x.last_valid_index()))
    out['last_close_age_days_252s'] = (as_of - out.last_observed_close_date_252s).dt.days
    out['price'] = closing.loc[as_of].where(lambda s: s.gt(0) & np.isfinite(s))
    out['pit_last_close'] = pd.Series(view.last_close(tickers), dtype=float)
    out['dgs10_latest_date'] = rf.index[-1] if len(rf) else pd.NaT
    out['dgs10_latest_pct'] = rf.iloc[-1] if len(rf) else np.nan
    return out


def _evaluate_month(day, end, bundle, scores):
    prices = bundle.prices
    as_of = prices.last_session_on_or_before(day.date() - timedelta(days=1))
    view = DataView(bundle, as_of)
    members = view.members()
    tickers = top_candidates(scores, day.date(), members, 20)
    if not tickers or len(tickers) != len(set(tickers)):
        raise ValueError(f'Invalid course top-20 candidate list on {day.date()}')
    inputs = _candidate_inputs(view, tickers, members, scores.cutoff)
    inputs['decision_date'] = day
    inputs['as_of'] = view.as_of
    base = inputs[['decision_date', 'as_of', 'rank', 'price', 'pit_last_close']].copy()
    position = prices.sessions.get_loc(view.as_of)
    for horizon in HORIZONS:
        offset = position + horizon
        matured = offset < len(prices.sessions) and prices.sessions[offset] <= end
        base[f'matured_{horizon}'] = matured
        base[f'outcome_date_{horizon}'] = prices.sessions[offset] if matured else pd.NaT
        if matured:
            future = prices.close.loc[prices.sessions[offset]].reindex(tickers)
            base[f'price_{horizon}'] = future.where(future.gt(0) & np.isfinite(future))
        else:
            base[f'price_{horizon}'] = np.nan
        base[f'ret_{horizon}'] = base[f'price_{horizon}'] / base.price - 1

    rows = []
    for name, fn in [('student', student_valuation.value), ('trailing_pe', trailing_pe_value)]:
        with student_call():
            raw = fn(as_of, tickers, view)
        if raw.ticker.duplicated().any():
            raise ValueError(f'Duplicate {name} valuations on {as_of}')
        clean = check_frame(raw, as_of, tickers, view).set_index('ticker')
        if len(clean) and (not np.isfinite(clean.to_numpy(dtype=float)).all() or
                           (clean[['low', 'fair_value', 'high']] <= 0).any().any()):
            raise ValueError(f'Invalid finite/positive valuation contract in {name} on {as_of}')
        extra = [key for key in raw.columns if key not in clean.columns and key != 'ticker']
        enriched = clean.join(raw.set_index('ticker')[extra]) if extra else clean
        joined = base.join(enriched)
        joined['model'] = name
        rows.append(joined.reset_index())
    return inputs.reset_index(), pd.concat(rows, ignore_index=True)


def _model_views(obs):
    student = obs.loc[obs.model == 'student']
    example = obs.loc[obs.model == 'trailing_pe']
    yield 'student', student
    yield 'trailing_pe', example
    for method in METHODS:
        if f'{method}_value' not in student.columns:
            continue
        frame = student.copy()
        frame['fair_value'] = frame[f'{method}_value']
        frame['mispricing'] = frame.fair_value / frame.pit_last_close - 1
        frame['low'] = frame.get(f'{method}_low', np.nan)
        frame['high'] = frame.get(f'{method}_high', np.nan)
        frame['method_spread'] = np.nan
        yield f'student_{method}', frame
    # Compare signal quality on identical candidate-months; otherwise P/E's
    # positive-earnings restriction would also change the sample composition.
    both = obs.groupby(['decision_date', 'ticker']).fair_value.count()
    common = both[both.eq(2)].index
    for name, frame in [('student', student), ('trailing_pe', example)]:
        index = pd.MultiIndex.from_frame(frame[['decision_date', 'ticker']])
        yield f'{name}_matched', frame[index.isin(common)]


def _monthly_ic(frame, horizon):
    rows = []
    for day, section in frame.groupby('decision_date'):
        pairs = section[['mispricing', f'ret_{horizon}']].dropna()
        valid = len(pairs) >= 5 and pairs.nunique().min() > 1
        ic = pairs.iloc[:, 0].corr(pairs.iloc[:, 1], method='spearman') if valid else np.nan
        rows.append({'decision_date': day, 'as_of': section.as_of.iloc[0], 'pairs': len(pairs), 'ic': ic})
    return pd.DataFrame(rows, columns=['decision_date', 'as_of', 'pairs', 'ic'])


def _hac_t(monthly, horizon):
    """Bartlett/Newey-West statistic. Use actual monthly lags when IC is missing."""
    valid = monthly.dropna(subset=['ic'])
    series = pd.Series(valid.ic.to_numpy(), index=pd.DatetimeIndex(valid.decision_date).to_period('M'))
    lag = {63: 2, 126: 5, 252: 11}[horizon]
    n = len(series)
    if n <= lag + 1:
        return np.nan
    centered = series - series.mean()
    variance = (centered ** 2).sum() / n
    for k in range(1, lag + 1):
        lagged = centered.copy()
        lagged.index += k
        variance += 2 * (1 - k / (lag + 1)) * (centered * lagged).sum() / n
    return series.mean() / np.sqrt(variance / n) if variance > 0 else np.nan


def _summarize(model, frame, monthly, horizon):
    ic = monthly.ic.dropna()
    actual = frame[f'price_{horizon}']
    fair = frame.fair_value
    bounds = frame.low.notna() & frame.high.notna() & actual.notna() & fair.notna()
    pct_error = fair / actual - 1
    return {'model': model, 'horizon': horizon, 'candidates': len(frame), 'valued': int(fair.notna().sum()),
            'valuation_coverage': fair.notna().mean(), 'matured_candidates': int(frame[f'matured_{horizon}'].sum()),
            'valued_with_return': int((fair.notna() & frame[f'ret_{horizon}'].notna()).sum()),
            'ic_months': len(ic), 'mean_ic': ic.mean(),
            't_ordinary': ic.mean() / ic.std() * np.sqrt(len(ic)) if len(ic) >= 2 and ic.std() > 0 else np.nan,
            't_hac': _hac_t(monthly, horizon), 'hac_lags': {63: 2, 126: 5, 252: 11}[horizon],
            'interval_pairs': int(bounds.sum()),
            'interval_coverage': (actual[bounds].between(frame.low[bounds], frame.high[bounds]).mean()
                                  if bounds.any() else np.nan),
            'median_signed_error': pct_error.median(), 'median_abs_error': pct_error.abs().median(),
            'median_mispricing': frame.mispricing.median(), 'median_method_spread': frame.method_spread.median()}


def run_evaluation(period, bundle, scores, out: Path):
    """Write reproducible estimates and diagnostics. Caller logs any holdout execution."""
    start, end = map(pd.Timestamp, PERIODS[period])
    sessions = bundle.prices.sessions
    window = sessions[(sessions >= start) & (sessions <= end)]
    month_starts = window.to_series().groupby(window.to_period('M')).first()
    inputs, observations = [], []
    for day in month_starts:
        audit, rows = _evaluate_month(day, end, bundle, scores)
        inputs.append(audit)
        observations.append(rows)
    inputs = pd.concat(inputs, ignore_index=True)
    obs = pd.concat(observations, ignore_index=True)
    monthly_ics, summary, yearly = [], [], []
    for model, cohort in _model_views(obs):
        for horizon in HORIZONS:
            monthly = _monthly_ic(cohort, horizon)
            monthly_ics.append(monthly.assign(model=model, horizon=horizon))
            summary.append(_summarize(model, cohort, monthly, horizon))
            for year, part in cohort.groupby(cohort.decision_date.dt.year):
                month_part = monthly[monthly.decision_date.dt.year.eq(year)]
                yearly.append(_summarize(model, part, month_part, horizon) | {'decision_year': year})

    audit = {'period': period, 'months': len(month_starts), 'candidates': len(inputs),
             'distinct_tickers': inputs.ticker.nunique(), 'not_in_pit_universe': (~inputs.in_pit_universe).sum(),
             'min_universe_size': inputs.universe_size.min(), 'max_universe_size': inputs.universe_size.max(),
             'missing_exact_spot': inputs.price.isna().sum(), 'missing_14d_spot': inputs.pit_last_close.isna().sum(),
             'stale_spot_accepted_by_14d_lookup': (inputs.price.isna() & inputs.pit_last_close.notna()).sum(),
             'no_visible_quarters': inputs.visible_quarters.eq(0).sum(),
             'fewer_than_four_quarters': inputs.visible_quarters.lt(4).sum(),
             'latest_four_eps_incomplete': (~inputs.latest_four_complete_eps).sum(),
             'latest_fiscal_quarter_over_180d': inputs.latest_quarter_age_days.gt(180).sum(),
             'latest_quarter_missing_acceptance': (inputs.visible_quarters.gt(0) & inputs.latest_quarter_accepted.isna()).sum(),
             'candidates_with_undated_visible_filing': inputs.undated_visible_filings.gt(0).sum(),
             'max_model_score_age_days': (inputs.as_of - inputs.selection_cutoff).dt.days.max(),
             'missing_risk_free': inputs.dgs10_latest_date.isna().sum(),
             'max_risk_free_age_days': (inputs.as_of - inputs.dgs10_latest_date).dt.days.max()}
    spot = obs.loc[obs.model.eq('student')]
    for horizon in HORIZONS:
        audit[f'unmatured_{horizon}'] = (~spot[f'matured_{horizon}']).sum()
        audit[f'missing_matured_end_price_{horizon}'] = (spot[f'matured_{horizon}'] & spot[f'price_{horizon}'].isna()).sum()
    summary = pd.DataFrame(summary)
    outputs = {'inputs': inputs, 'observations': obs, 'ic_monthly': pd.concat(monthly_ics, ignore_index=True),
               'summary': summary, 'yearly': pd.DataFrame(yearly), 'audit': pd.DataFrame([audit])}
    out = Path(out)
    out.mkdir(exist_ok=True, parents=True)
    for name, table in outputs.items():
        table.to_csv(out / f'valuation_{period}_{name}.csv', index=False, float_format='%.10g')
    return summary
