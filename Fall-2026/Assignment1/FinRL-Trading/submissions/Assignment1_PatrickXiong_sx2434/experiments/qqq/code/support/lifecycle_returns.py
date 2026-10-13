"""Life-scoped prices and cash consideration; no survival-to-end requirement.

This component is for the next pipeline. No missing terminal event is converted
to zero return, a forced sale at the last quote, or an inferred cash payment.
Economic value is distinct from executable price and settled cash.
"""
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd


class UnresolvedData(ValueError):
    pass


def verify_evidence(record, root, role):
    required = ('source_url', 'local_file', 'sha256', 'reviewer', 'reviewed_at')
    if any(not record.get(k) for k in required) or record.get('review_status') != 'verified':
        raise UnresolvedData('Missing reviewed evidence: ' + role)
    if role not in record.get('roles', []):
        raise UnresolvedData('Evidence does not establish ' + role)
    root = Path(root).resolve()
    p = (root / record['local_file']).resolve()
    if root not in p.parents or not p.is_file():
        raise UnresolvedData('Evidence file outside root or missing')
    if hashlib.sha256(p.read_bytes()).hexdigest() != record['sha256']:
        raise UnresolvedData('Evidence hash mismatch')


def cash_exit_path(quotes, sessions, start, end, event, evidence_root, *, allow_candidate_demo=False):
    """One unit of total-return wealth, with cash entitlement after last trade.

`close` on the final trading day MUST be on the merger's per-share basis.
`adj_close` supplies only pre-event total-return ratios. It is NEVER directly
compared with a raw merger payment. Unknown settlement stays a receivable.
This supports an unconditional all-cash cancellation only; mixed/stock/CVR
events must use a different, fully documented accounting path.
"""
    for role in ('identity', 'termination', 'cash_consideration'):
        record = event.get('evidence_by_role', {}).get(role, event.get('evidence', {}))
        verify_evidence(record, evidence_root, role)
    if event.get('kind') != 'cash':
        raise UnresolvedData('Cash-only event required')
    basis = event.get('share_basis', {})
    candidate = basis.get('status') == 'candidate'
    if candidate and not allow_candidate_demo:
        raise UnresolvedData('Candidate share basis cannot enter a production backtest')
    if candidate:
        if not basis.get('reason'):
            raise UnresolvedData('Candidate demonstration requires an explicit limitation')
    else:
        verify_evidence(basis.get('evidence', {}), evidence_root, 'final_share_basis')
    amount = event.get('cash_per_share')
    if amount is None or not np.isfinite(amount) or amount < 0:
        raise UnresolvedData('Invalid consideration')
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    last = pd.Timestamp(event['last_trade'])
    recognized = pd.Timestamp(event['recognition_session'])
    settlement = pd.Timestamp(event['settlement_session']) if event.get('settlement_session') else None
    if not start <= last < recognized <= end:
        raise UnresolvedData('Invalid lifecycle interval')
    if settlement is not None:
        verify_evidence(event['settlement_evidence'], evidence_root, 'settlement')
        if settlement < recognized:
            raise UnresolvedData('Settlement precedes recognition')
    idx = pd.DatetimeIndex(sessions)
    q = quotes.copy()
    q.index = pd.DatetimeIndex(q.index)
    if idx.tz is not None or q.index.tz is not None or idx.has_duplicates or q.index.has_duplicates:
        raise UnresolvedData('Use unique, timezone-naive session dates')
    if not idx.is_monotonic_increasing or not idx.equals(idx.normalize()) or not q.index.equals(q.index.normalize()):
        raise UnresolvedData('Session dates must be sorted and normalized')
    idx = idx[(idx >= start) & (idx <= end)]
    if not len(idx) or start not in idx or last not in idx or recognized not in idx:
        raise UnresolvedData('Required date is not in exchange calendar')
    if idx[idx > last][0] != recognized:
        raise UnresolvedData('Unvalued gap between final trade and consideration')
    if (q.index > last).any():
        raise UnresolvedData('Post-termination quotes require identity investigation')
    live = idx[idx <= last]
    v = q.reindex(live)[['close', 'adj_close']]
    if not np.isfinite(v.to_numpy()).all() or (v <= 0).any().any():
        raise UnresolvedData('Missing/invalid in-life quote')
    result = pd.DataFrame(index=idx)
    result.index.name = 'date'
    result['eligible_for_backtest'] = not candidate
    result['wealth'] = np.nan
    result.loc[live, 'wealth'] = v.adj_close / v.adj_close.iloc[0]
    # Normalize entitlement by final RAW quote, not dividend-adjusted quote.
    terminal = result.loc[last, 'wealth'] * amount / v.loc[last, 'close']
    result.loc[idx >= recognized, 'wealth'] = terminal
    result['state'] = 'live'
    result.loc[idx >= recognized, 'state'] = 'cash_receivable'
    if settlement is not None:
        result.loc[idx >= settlement, 'state'] = 'cash'
    result['can_buy_security'] = result.state.eq('live')
    result['settled_cash'] = result.wealth.where(result.state.eq('cash'), 0.)
    result['cash_receivable'] = result.wealth.where(result.state.eq('cash_receivable'), 0.)
    result['return'] = result.wealth.pct_change(fill_method=None)
    result.iloc[0, result.columns.get_loc('return')] = 0.
    # Once a documented zero payout destroys wealth, subsequent returns are 0.
    result.loc[(result.wealth == 0) & (result.wealth.shift() == 0), 'return'] = 0.
    if not np.isfinite(result[['wealth', 'return']].to_numpy()).all():
        raise UnresolvedData('Undefined realized wealth path')
    return result


def assert_complete_census(census, expected_keys):
    """A known unavailable security remains a recorded case, never vanishes."""
    if census.duplicated(['ticker', 'decision_date']).any():
        raise UnresolvedData('Duplicate historical universe cases')
    actual = set(zip(census.ticker, census.decision_date))
    expected = [tuple(k) for k in expected_keys]
    if len(expected) != len(set(expected)) or actual != set(expected):
        raise UnresolvedData('Historical universe denominator changed')
    allowed = {'verified_live', 'verified_not_tradable'}
    bad = census.loc[~census.status.isin(allowed)]
    if len(bad):
        raise UnresolvedData(f'{len(bad)} universe cases are unresolved')
