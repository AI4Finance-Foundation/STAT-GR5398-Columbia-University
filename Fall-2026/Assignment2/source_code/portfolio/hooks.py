"""Connect the student modules to the backtester.

backtester.run_backtest() asks a predictions object for each week's
candidates (.get(day), .cutoff) and calls exposure_fn(prev_day) for the
budget multiplier. These wrappers give each call a DataView limited to the
decision date, run it under the look-ahead guard, and validate the output.
"""
from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace

from shared.data_view import DataBundle, DataView
from shared.guard import student_call


class FilteredPredictions:
    """Each week: this month's scores -> filter_candidates -> backtester.

    The filter may only remove rows. target_weights() then re-takes the top 10,
    so a removed name is replaced by the next-ranked one.
    """

    def __init__(self, base, filter_fn, bundle: DataBundle):
        self._base, self._fn, self._bundle = base, filter_fn, bundle
        self.calls = self.removed = 0

    @property
    def cutoff(self):
        return self._base.cutoff

    def get(self, day):
        preds = self._base.get(day)
        as_of = self._bundle.prices.last_session_on_or_before(day - timedelta(days=1))
        with student_call():
            kept = tuple(self._fn(as_of, preds, DataView(self._bundle, as_of)))
        extra = set(kept) - set(preds)
        if extra:
            raise ValueError(f"filter_candidates({as_of}) returned {len(extra)} rows that were not in its "
                             f"input, e.g. {next(iter(extra))}. Only remove rows.")
        self.calls += 1
        self.removed += len(preds) - len(kept)
        return kept


def exposure_fn(fn, bundle: DataBundle):
    """fn(as_of, data) -> float in [0, 1], in the form backtester.run_backtest expects."""
    def call(prev_day):
        with student_call():
            x = float(fn(prev_day, DataView(bundle, prev_day)))
        if not 0.0 <= x <= 1.0:
            raise ValueError(f"exposure({prev_day}) returned {x}; it must be between 0.0 and 1.0")
        call.log.append(x)
        return SimpleNamespace(exposureMultiplier=x)
    call.log = []
    return call
