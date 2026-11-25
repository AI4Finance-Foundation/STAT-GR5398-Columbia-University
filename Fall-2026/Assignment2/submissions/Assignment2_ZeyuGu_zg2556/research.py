"""Predeclared FinRobot filter experiment and single-batch frozen evaluation.

Run from the course source_code directory. The quoted discount is 1 − fair/price.
"""
import argparse
from datetime import timedelta
import fcntl
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd

sys.path.insert(0, str(Path.cwd()))
from portfolio.backtester import BacktestConfig, run_backtest
from run_portfolio import CONFIG, run_one
from selection import MonthlyScores
from selection.config import load
from shared.data_view import DataBundle, DataView
from shared.guard import student_call
from shared.metrics import performance
from shared.periods import PERIODS, record_test_run
from valuation import student_valuation as model

OUT = Path('output')
GAPS = (0.15, 0.25, 0.35)


def context():
    _, options, params, prices, _, universe = load(CONFIG)
    return dict(options=options, params=params, prices=prices, universe=universe,
                scores=MonthlyScores(), bundle=DataBundle.load(prices=prices))


class ReplayFilter:
    """Reuse the same weekly PIT marks for every threshold in the sensitivity study."""
    def __init__(self, scores, rejected):
        self.scores = scores
        self.rejected = rejected

    @property
    def cutoff(self):
        return self.scores.cutoff

    def get(self, day):
        return tuple(x for x in self.scores.get(day) if x.symbol not in self.rejected[day])


def sensitivity(period, ctx):
    start, end = PERIODS[period]
    sessions = ctx['prices'].sessions
    window = sessions[(sessions >= pd.Timestamp(start)) & (sessions <= pd.Timestamp(end))]
    decisions = window.to_series().groupby(window.to_period('W-SUN')).first()
    marks, rejected = [], {g: {} for g in GAPS}
    for day in decisions:
        as_of = ctx['prices'].last_session_on_or_before(day.date() - timedelta(days=1))
        candidates = ctx['scores'].get(day.date())
        view = DataView(ctx['bundle'], as_of)
        with student_call():
            frame = model.screen_values(as_of, candidates, view)
        marks.append(frame.assign(as_of=as_of, rebalance=day.date()))
        for gap in GAPS:
            rejected[gap][day.date()] = model.removed_symbols(frame, gap)
    pd.concat(marks, ignore_index=True).to_csv(OUT/f'weekly_marks_{period}.csv', index=False)
    rows = []
    for gap in GAPS:
        replay = ReplayFilter(ctx['scores'], rejected[gap])
        cfg = BacktestConfig(start=start, end=end)
        res = run_backtest(ctx['prices'], replay, ctx['params'], ctx['universe'], cfg,
                           global_unified=ctx['options'].inference_mode == 'global-unified')
        traded = res.trades.notional.abs().sum()
        turnover = traded / res.equity.mean() / (len(res.equity) / 252)
        counts = [len(v) for v in rejected[gap].values()]
        rows.append(dict(period=period, gap=gap, **performance(res.equity), turnover_per_year=round(turnover, 2),
                         exclusions=sum(counts), decisions=len(counts), active_weeks=sum(n > 0 for n in counts)))
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT/f'sensitivity_{period}.csv', index=False)
    if period == 'val':
        ranked = frame.assign(distance=(frame.gap - .25).abs().round(3)).sort_values(
            ['sharpe', 'turnover_per_year', 'distance', 'gap'], ascending=[False, True, True, True])
        print('Selected gap:', ranked.iloc[0].gap, flush=True)


def ablation(period, ctx):
    start, end = PERIODS[period]
    runs = [('A_selection', False, 'none'), ('B_valuation', True, 'none'),
            ('C_timing_full', False, 'full'), ('D_both_full', True, 'full')]
    results, curves, years = [], {}, []
    for label, valuation, timing in runs:
        name = f'{period}_{label}'
        print(f'Running {name}', flush=True)
        row = run_one(name, start, end, valuation=valuation, timing=timing, stops=False, ctx=ctx)
        curve = pd.read_csv(OUT/name/'equity.csv', index_col=0, parse_dates=True)['equity']
        for year, part in curve.groupby(curve.index.year):
            # Include the last price of the previous year to capture the first session's return.
            previous = curve.loc[curve.index < part.index[0]].tail(1)
            years.append(dict(run=name, year=year, **performance(pd.concat([previous, part]))))
        curves[label] = curve
        results.append(row)
    pd.testing.assert_series_equal(curves['A_selection'], curves['C_timing_full'], check_exact=True)
    pd.testing.assert_series_equal(curves['B_valuation'], curves['D_both_full'], check_exact=True)
    pd.DataFrame(results).set_index('run').to_csv(OUT/f'ablation_{period}.csv')
    pd.DataFrame(years).to_csv(OUT/f'portfolio_by_year_{period}.csv', index=False)
    if period in ('dev', 'val'):
        f = pd.read_csv(OUT/f'sensitivity_{period}.csv')
        row = f.loc[(f.gap-model.FILTER_THRESHOLD).abs() < 1e-9].iloc[0]
        for metric in ('sharpe', 'total_return_pct', 'max_drawdown_pct', 'turnover_per_year'):
            if abs(float(row[metric]) - float(results[1][metric])) > 1e-6:
                raise RuntimeError(f'Weekly replay disagrees with live student hook on {metric}.')


def verify_freeze(commit):
    if not commit:
        raise SystemExit('Pass the full configuration hash after course preregistration.')
    module = Path('valuation/student_valuation.py')
    submitted = Path(__file__).resolve().with_name('student_valuation.py')
    repo = Path(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], text=True).strip())
    frozen_path = submitted.relative_to(repo).as_posix()
    frozen = subprocess.check_output(['git', 'show', f'{commit}:{frozen_path}'])
    for name in ('research.py', 'valuation_evaluation.py'):
        script = submitted.with_name(name)
        tracked = subprocess.check_output(['git', 'show', f'{commit}:{script.relative_to(repo).as_posix()}'])
        if script.read_bytes() != tracked:
            raise SystemExit(f'{name} differs from the preregistration.')
    if frozen != module.read_bytes():
        raise SystemExit('The active valuation module differs from the preregistered commit.')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    if head != commit:
        raise SystemExit('Run the holdout in a checkout at the registered commit.')
    history = OUT/'test_runs.jsonl'
    if history.exists() and history.read_text().strip():
        raise SystemExit('A test run already exists. Read its saved outputs.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['sensitivity', 'evaluate', 'portfolio', 'holdout'])
    parser.add_argument('period', choices=['dev', 'val', 'test'])
    parser.add_argument('--registered-commit')
    args = parser.parse_args()
    if os.getenv('PYTHONHASHSEED') != '0':
        raise SystemExit('Set PYTHONHASHSEED=0 before running Python.')
    if (args.action == 'holdout') != (args.period == 'test'):
        parser.error('test must run as a single holdout batch')
    OUT.mkdir(exist_ok=True)
    if args.action == 'holdout':
        # Serialize verification and logging so two simultaneous invocations cannot spend the holdout twice.
        lock = (OUT/'.holdout.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify_freeze(args.registered_commit)
    ctx = context()
    if args.action == 'holdout':
        record_test_run('research.py', sys.argv[1:], *PERIODS['test'])
    if args.action == 'sensitivity':
        sensitivity(args.period, ctx)
    if args.action in ('evaluate', 'holdout'):
        from valuation_evaluation import run_evaluation
        run_evaluation(args.period, ctx['bundle'], ctx['scores'], OUT)
    if args.action in ('portfolio', 'holdout'):
        ablation(args.period, ctx)


if __name__ == '__main__':
    main()
