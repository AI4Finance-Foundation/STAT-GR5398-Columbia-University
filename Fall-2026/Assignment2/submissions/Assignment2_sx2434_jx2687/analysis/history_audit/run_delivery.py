"""Reproduce the two requested development-period snapshots and diagnostics."""
from __future__ import annotations
import argparse
from datetime import date
from pathlib import Path
import sys
import json
import hashlib
import time

import numpy as np
import pandas as pd

from historical_ev_ebitda import historical_ev_ebitda_value


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-root', type=Path, required=True,
                    help='Course Assignment2/source_code directory containing shared/ and data/')
    ap.add_argument('--out', type=Path, default=Path(__file__).resolve().parent)
    args = ap.parse_args()
    source = args.source_root.resolve()
    sys.path.insert(0, str(source))
    from shared.data_view import DataBundle, DataView
    from shared.guard import student_call
    from selection.scores import MonthlyScores
    from evaluate_valuation import top_candidates
    from check_submission import scan

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    bundle = DataBundle.load()
    t0 = time.perf_counter()
    logs = []

    def check(name, condition, detail=''):
        if not condition:
            raise AssertionError(name + ': ' + detail)
        logs.append('PASS ' + name + (': ' + detail if detail else ''))

    msft_view = DataView(bundle, '2021-06-30')
    with student_call():
        msft, msft_hist = historical_ev_ebitda_value('2021-06-30', ['MSFT'], msft_view, return_history=True)
    m = msft.iloc[0]
    check('MSFT valid months reference', m.valid_months == 50, str(m.valid_months))
    check('MSFT median reference rounded to 2dp', round(m.multiple_p50, 2) == 15.78, str(m.multiple_p50))
    check('MSFT fair value reference rounded to cents', round(m.fair_value, 2) == 171.60, str(m.fair_value))
    check('Caller cutoff unchanged', msft_view.as_of == pd.Timestamp('2021-06-30'))
    msft.to_csv(out/'msft_2021-06-30.csv', index=False)
    msft_hist.to_csv(out/'msft_monthly_details.csv', index=False)

    valuation_date, rebalance = date(2019, 8, 2), date(2019, 8, 5)
    view = DataView(bundle, valuation_date)
    scores = MonthlyScores()
    tickers = top_candidates(scores, rebalance, view.members(), 20)
    check('Exactly 20 unique course candidates', len(tickers) == len(set(tickers)) == 20)
    check('AMD included', 'AMD' in tickers, 'rank ' + str(tickers.index('AMD')+1))
    preds = {p.symbol: p for p in scores.get(rebalance)}
    candidates = pd.DataFrame([dict(rank=i+1, ticker=t, bucket=preds[t].bucket,
                                    rank_score=preds[t].rank_score,
                                    weight_score=preds[t].weight_score,
                                    rebalance_date=str(rebalance), valuation_date=str(valuation_date),
                                    model_cutoff=str(scores.cutoff)) for i,t in enumerate(tickers)])
    candidates.to_csv(out/'candidates_2019-08-05.csv', index=False)
    with student_call():
        batch, history = historical_ev_ebitda_value(valuation_date, tickers, view, return_history=True)
        repeated = historical_ev_ebitda_value(valuation_date, tickers, view)
    check('Repeatability', batch.equals(repeated))
    check('All candidates retained including failures', batch.ticker.tolist() == tickers)
    batch.insert(0, 'rank', range(1, len(batch)+1))
    batch.to_csv(out/'candidates_valuation_2019-08-02.csv', index=False)
    history.to_csv(out/'candidate_monthly_details.csv', index=False)

    # Independent calculation: reconstruct the earlier four-year-end MSFT case
    # directly from the course views, without using the module's snapshot helper.
    annual = []
    for day in ['2017-12-29', '2018-12-31', '2019-12-31', '2020-12-31']:
        dv = DataView(bundle, day)
        q = dv.fundamentals(['MSFT']).sort_values('datadate').tail(4)
        e = sum(float(r.revenue_per_share)*float(r.ebitda_margin) for r in q.itertuples())
        nd = float(q.iloc[-1].interest_debt_per_share-q.iloc[-1].cash_per_share)
        annual.append((dv.last_close(['MSFT'])['MSFT']+nd)/e)
    annual_value = float(np.median(annual)*m.ttm_ebitda_per_share-m.net_debt_per_share)
    check('Earlier four-year-end reference', abs(annual_value-168.7835222208331)<1e-8)

    # Independently rebuild every valid monthly multiple from DataView, and
    # verify all quantiles and fair values from these independently rebuilt data.
    max_multiple_error = 0.0
    for all_hist, as_of, res in [(msft_hist, '2021-06-30', msft),
                                 (history, '2019-08-02', batch)]:
        valid = all_hist[all_hist.valid]
        independent = {}
        for day, rows in valid.groupby('month_end'):
            dv = DataView(bundle, day)
            fs = dv.fundamentals(rows.ticker.tolist())
            ps = dv.last_close(rows.ticker.tolist())
            for r in rows.itertuples():
                q = fs[fs.ticker == r.ticker].sort_values('datadate').tail(4)
                e = sum(float(x.revenue_per_share)*float(x.ebitda_margin) for x in q.itertuples())
                nd = float(q.iloc[-1].interest_debt_per_share-q.iloc[-1].cash_per_share)
                mult = (ps[r.ticker]+nd)/e
                max_multiple_error = max(max_multiple_error, abs(mult-r.multiple))
                independent.setdefault(r.ticker, []).append(mult)
        current = DataView(bundle, as_of).fundamentals(res.ticker.tolist())
        for r in res.itertuples():
            if r.status != 'ok':
                continue
            q = current[current.ticker == r.ticker].sort_values('datadate').tail(4)
            e = sum(float(x.revenue_per_share)*float(x.ebitda_margin) for x in q.itertuples())
            nd = float(q.iloc[-1].interest_debt_per_share-q.iloc[-1].cash_per_share)
            qs = np.quantile(independent[r.ticker], [.25,.5,.75])
            check(f'{as_of} {r.ticker} quantiles and three prices',
                  np.allclose(qs, [r.multiple_p25,r.multiple_p50,r.multiple_p75], rtol=0, atol=1e-10)
                  and np.allclose(qs*e-nd, [r.low,r.fair_value,r.high], rtol=0, atol=1e-8))
        check(f'{as_of} historical cutoff and availability',
              (pd.to_datetime(valid.month_end) < pd.Timestamp(as_of)).all()
              and (pd.to_datetime(valid.month_end) >= pd.Timestamp(as_of)-pd.DateOffset(years=5)).all()
              and (pd.to_datetime(valid.latest_accepted_date) <= pd.to_datetime(valid.month_end)).all())
    check('Independent rebuild of all valid monthly multiples', max_multiple_error < 1e-10,
          'max absolute difference '+str(max_multiple_error))

    # Boundary tests for the 12-month gate, negative EBITDA and revised filings.
    from test_historical_ev_ebitda import run_boundary_tests
    logs.extend(run_boundary_tests())
    scan(Path(__file__).with_name('historical_ev_ebitda.py'))
    print('\n'.join(logs))
    summary = dict(msft=msft.replace({np.nan:None}).to_dict('records')[0],
                   annual_msft_value=annual_value,
                   candidate_count=len(batch), valued_count=int(batch.status.eq('ok').sum()),
                   failures=batch[batch.status.ne('ok')].replace({np.nan:None}).to_dict('records'),
                   rebalance_date=str(rebalance), as_of=str(valuation_date), model_cutoff=str(scores.cutoff),
                   candidate_reference='No separate reference results supplied for the 20-stock batch.',
                   checks=len(logs), elapsed_seconds=time.perf_counter()-t0,
                   source_sha256={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [source/'data/fundamentals.db',source/'data/prices/daily_adjusted.parquet',
                                 source/'data/selection/monthly_scores.parquet',source/'shared/data_view.py',
                                 source/'shared/store.py']})
    (out/'run_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    (out/'validation.txt').write_text('\n'.join(logs)+'\n')
    print(batch[['rank','ticker','price','valid_months','multiple_p50','fair_value','status','reason']].to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
