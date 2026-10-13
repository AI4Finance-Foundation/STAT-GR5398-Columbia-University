"""Execution-time record of what each model fit actually saw.

The preserved run saved only row counts and boundary maxima per window, so its
splits can be reconstructed by design but not proven as executed. For the next
run, the trainer calls `FitLog.record` immediately before every fit, validation
score and prediction. Each call
  * checks the time boundary for that role and refuses to continue on a breach,
  * appends one JSON line with the exact row keys (as a hash), counts and extremes.

`compare_with_expected` then checks a finished log against an independently
reconstructed manifest, execution by execution.

Row frames need: security_id, quarter, available, label_end, decision_date.

Bounds by role (all dates are session dates):
  outer_train / inner_train   cutoff = first decision of the held-out rows
  outer_validation            cutoff = inference decision (upper); lower = outer-train cutoff
  inner_validation            cutoff = fold boundary (lower);     upper = outer-train cutoff
  inference                   cutoff = the decision being predicted
"""
import hashlib
import json
from datetime import datetime, timezone

import pandas as pd

RULES = {
    'outer_train': 'label_end < cutoff, available < cutoff, available <= own decision',
    'inner_train': 'label_end < cutoff, available < cutoff, available <= own decision',
    'outer_validation': 'own decision >= lower, label_end < cutoff, available <= own decision',
    'inner_validation': 'own decision >= cutoff, available <= own decision, label_end < upper, available < upper',
    'inference': 'available <= cutoff, own decision == cutoff, target not realised by cutoff (label_end missing or later)',
}
REQUIRED_BOUND = {'outer_validation': 'lower', 'inner_validation': 'upper'}
CHECKED_FIELDS = ('window', 'role', 'rule', 'cutoff', 'lower', 'upper', 'rows', 'key_sha256', 'max_label_end',
                  'max_available', 'min_decision', 'max_decision', 'breaches')


class BoundaryViolation(RuntimeError):
    pass


def role_kind(role):
    """'inner_0_train' -> 'inner_train'."""
    return '_'.join(p for p in role.split('_') if not p.isdigit())


def key_hash(frame):
    """Order-independent digest of the exact (security, quarter) rows."""
    keys = sorted(f'{a}|{str(b)[:10]}' for a, b in zip(frame['security_id'], frame['quarter']))
    if len(set(keys)) != len(keys):
        raise BoundaryViolation('duplicate security/quarter rows in one split')
    return hashlib.sha256('\n'.join(keys).encode()).hexdigest()


def breaches(role, frame, cutoff, lower=None, upper=None):
    """Number of rows breaking the rule for this role (0 means the split is admissible)."""
    kind = role_kind(role)
    if kind not in RULES:
        raise BoundaryViolation(f'unknown role {role}')
    needed = REQUIRED_BOUND.get(kind)
    if needed and {'lower': lower, 'upper': upper}[needed] is None:
        raise BoundaryViolation(f'{role} needs an explicit {needed} bound')
    cutoff = pd.Timestamp(cutoff)
    available, label_end, decision = (pd.to_datetime(frame[c]) for c in ('available', 'label_end', 'decision_date'))
    published_at_own_decision = available <= decision
    if kind in ('outer_train', 'inner_train'):
        good = (label_end < cutoff) & (available < cutoff) & published_at_own_decision
    elif kind == 'outer_validation':
        good = (decision >= pd.Timestamp(lower)) & (label_end < cutoff) & published_at_own_decision
    elif kind == 'inner_validation':
        ceiling = pd.Timestamp(upper)
        good = (decision >= cutoff) & published_at_own_decision & (label_end < ceiling) & (available < ceiling)
    else:  # inference: a missing future target is expected and is never a reason to drop the row
        good = (available <= cutoff) & (decision == cutoff) & (label_end.isna() | (label_end > cutoff))
    return int((~good).sum())


def describe(window, role, frame, cutoff, lower=None, upper=None):
    def extreme(column, how):
        values = pd.to_datetime(frame[column]).dropna()
        return str(getattr(values, how)().date()) if len(values) else None

    def day(value):
        return None if value is None else str(pd.Timestamp(value).date())
    return dict(window=window, role=role, rule=RULES[role_kind(role)], cutoff=day(cutoff), lower=day(lower), upper=day(upper),
                rows=int(len(frame)), key_sha256=key_hash(frame),
                max_label_end=extreme('label_end', 'max'), max_available=extreme('available', 'max'),
                min_decision=extreme('decision_date', 'min'), max_decision=extreme('decision_date', 'max'),
                breaches=breaches(role, frame, cutoff, lower, upper))


class FitLog:
    """Append-only JSON-lines log. Use one per training run."""

    def __init__(self, path, run_id):
        self.path, self.run_id, self.sequence = path, run_id, 0

    def record(self, window, role, frame, cutoff, lower=None, upper=None, context=None):
        """Call right before the rows are handed to fit/score/predict. Raises on a breach.

        `context` is a free-form dict stored under its own key; it can never replace a checked field.
        """
        checked = describe(window, role, frame, cutoff, lower, upper)
        self.sequence += 1
        entry = dict(checked, run_id=self.run_id, sequence=self.sequence,
                     logged_at=datetime.now(timezone.utc).isoformat(), context=dict(context or {}))
        with open(self.path, 'a') as handle:
            handle.write(json.dumps(entry, default=str) + '\n')
        if checked['breaches']:
            raise BoundaryViolation(f"{window} {role}: {checked['breaches']} rows break '{checked['rule']}'")
        return entry


def read_log(log_path):
    with open(log_path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def compare_with_expected(expected, log_path):
    """Differences between an expected manifest (DataFrame) and an execution log.

    Every logged entry is inspected. A split executed more than once is reported,
    and each execution is checked on its own; nothing is collapsed to the last entry.
    """
    entries = read_log(log_path)
    by_split = {}
    for entry in entries:
        by_split.setdefault((entry['window'], entry['role']), []).append(entry)
    problems = []
    for entry in entries:
        if entry['breaches']:
            problems.append((entry['window'], entry['role'], f"boundary breach in execution {entry.get('sequence')}"))
    expected_keys = set()
    for row in expected.itertuples():
        key = (row.window, row.role)
        if key in expected_keys:
            problems.append((row.window, row.role, 'duplicate expected split'))
            continue
        expected_keys.add(key)
        runs = by_split.get(key, [])
        if not runs:
            problems.append((row.window, row.role, 'not executed'))
            continue
        if len(runs) > 1:
            problems.append((row.window, row.role, f'executed {len(runs)} times'))
        for entry in runs:
            same = all(entry.get(field) == getattr(row, field) for field in ('key_sha256', 'rows', 'cutoff'))
            bounds = all((entry.get(field) or None) == (None if pd.isna(getattr(row, field)) else getattr(row, field))
                         for field in ('lower', 'upper'))
            if not same:
                problems.append((row.window, row.role, f"different rows or cutoff than expected in execution {entry.get('sequence')}"))
            elif not bounds:
                problems.append((row.window, row.role, f"different bounds than expected in execution {entry.get('sequence')}"))
    problems += [(w, r, 'executed but not expected') for (w, r) in by_split if (w, r) not in expected_keys]
    return problems
