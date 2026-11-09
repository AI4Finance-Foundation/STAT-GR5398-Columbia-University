"""Course selection model: monthly ML predictions -> eligible ranked picks -> target weights.

Two modes share everything except how the list is drawn:

  Growth          inference_mode="mixed-alpha", all bucket_alpha = 0: one model
                  per GICS bucket, rank inside a bucket, take N per bucket
                  (e.g. picks_per_bucket={"growth_tech": 10}).
  Global Balanced inference_mode="global-unified": one model on every stock,
                  scores z-scored inside each bucket, take the global top N.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Dict, Literal, Optional, Sequence, Tuple

from .allocation import ScoreAllocation
from .ranking import rank_with_incumbents, select_candidates

BUCKETS = ("growth_tech", "cyclical", "real_assets", "defensive")


@dataclass(frozen=True)
class ModelOptions:
    db_path: str = "data/fundamentals.db"
    universes: Tuple[str, ...] = ("sp500", "nasdaq100")
    inference_mode: Literal["mixed-vintage", "latest-snapshot", "mixed-alpha", "global-unified"] = "mixed-alpha"
    bucket_alpha: Dict[str, float] = field(default_factory=lambda: dict.fromkeys(BUCKETS, 0.0))
    # Upper bound only: the effective cutoff is min(val_cutoff, quarter before the
    # inference quarter), which rolls forward every quarter. A far-future value
    # means "always roll".
    val_cutoff: str = "2099-12-31"
    val_quarters: int = 3

    def __post_init__(self):
        if set(self.bucket_alpha) != set(BUCKETS):
            raise ValueError("bucket_alpha must specify all four model buckets")
        if any(isinstance(v, bool) or not math.isfinite(v) or not 0 <= v <= 1 for v in self.bucket_alpha.values()):
            raise ValueError("bucket_alpha values must be fractions in [0, 1]")


@dataclass(frozen=True)
class Parameters:
    picks_per_bucket: Dict[str, int] = field(default_factory=lambda: {"growth_tech": 10})
    global_picks: int = 0                 # global-unified: 0 means 10
    exposure_rate: float = 1.0            # total budget before the exposure engine
    weight_mode: Literal["equal", "score", "score-global"] = "score-global"
    min_weight_pct: float = 0.05
    max_weight_pct: float = 0.15
    incumbent_multiplier: int = 2
    blacklist: Tuple[str, ...] = ()

    def __post_init__(self):
        if not self.picks_per_bucket or any(type(n) is not int or n < 0 for n in self.picks_per_bucket.values()):
            raise ValueError("Bucket quotas must be nonnegative integers")
        if self.min_weight_pct > self.max_weight_pct:
            raise ValueError("Minimum weight exceeds maximum weight")


@dataclass(frozen=True)
class Prediction:
    symbol: str
    bucket: str
    rank_score: float      # used for ranking (bucket percentile or global z-score)
    weight_score: float    # used for weighting (raw bucket ensemble prediction)


class MonthlyPredictions:
    """Train once per month on information completed before that month.

    ``cutoff_for(day)`` must return the last trading session before the 1st
    of ``day``'s month. ``price_as_of(symbols, day)`` returns closes on or
    before ``day``; it caps the price momentum of rows whose pricing date
    (quarter end + ~2 months) is still after the cutoff, so a backtest never
    sees a close that had not happened yet.
    """

    def __init__(self, options: ModelOptions, *, cutoff_for: Callable[[date], date],
                 price_as_of: Optional[Callable[[Sequence[str], date], Dict[str, float]]] = None):
        from .selector import MLStockSelector
        self.options, self.cutoff_for, self.price_as_of = options, cutoff_for, price_as_of
        self._factory = MLStockSelector
        self._month, self._rows, self.cutoff, self.selector = None, (), None, None

    def get(self, day: date) -> Tuple[Prediction, ...]:
        month = day.replace(day=1)
        if month == self._month:
            return self._rows
        cutoff = self.cutoff_for(day)
        selector = self._factory(
            db_path=self.options.db_path, universes=list(self.options.universes),
            inference_mode=self.options.inference_mode, bucket_alpha=dict(self.options.bucket_alpha),
            val_cutoff=self.options.val_cutoff, val_quarters=self.options.val_quarters,
            multi_universe_buckets=[], api_key="", auto_sync_missing=False, read_only=True,
            price_as_of=self.price_as_of,
        )
        frame = selector.run_monthly(cutoff)
        required = {"tic", "bucket", "pred_ensemble_avg"}
        if frame is None or frame.empty or not required <= set(frame.columns):
            raise ValueError(f"Monthly model produced no usable predictions at {cutoff}")
        rows = []
        for row in frame.to_dict("records"):
            rank = float(row["pred_ensemble_avg"])
            weight = float(row.get("pred_ensemble_avg_bucket", rank))
            if not (math.isfinite(rank) and math.isfinite(weight)):
                raise ValueError("Monthly model produced invalid predictions")
            rows.append(Prediction(str(row["tic"]), str(row["bucket"]), rank, weight))
        self._month, self._rows, self.cutoff, self.selector = month, tuple(rows), cutoff, selector
        return self._rows


def target_weights(predictions, parameters: Parameters, eligible, positions, *, global_unified: bool = False):
    """Pure decision: {symbol: weight}. ``eligible`` = symbols tradeable today."""
    eligible = set(eligible) - set(parameters.blacklist)
    candidates = [{"symbol": row.symbol, "bucket": row.bucket,
                   "pred_ensemble_avg": row.rank_score, "pred_raw_score": row.weight_score}
                  for row in predictions if row.symbol in eligible]
    rules = ScoreAllocation(parameters.min_weight_pct, parameters.max_weight_pct)
    if global_unified:
        count = parameters.global_picks or 10
        candidates.sort(key=lambda row: row["pred_ensemble_avg"], reverse=True)
        picks = rank_with_incumbents(candidates[:count], eligible, positions, count, parameters.incumbent_multiplier)
        return rules.weights(picks, parameters.exposure_rate)
    buckets = {}
    for bucket, count in parameters.picks_per_bucket.items():
        rows = sorted((row for row in candidates if row["bucket"] == bucket),
                      key=lambda row: row["pred_ensemble_avg"], reverse=True)
        # The top-N window is taken before incumbent preference, so a holding
        # ranked between N and 2N is not retained (incumbent_multiplier has no
        # effect in practice).
        picks = rank_with_incumbents(rows[:count], eligible, positions, count, parameters.incumbent_multiplier)
        if picks:
            buckets[bucket] = picks
    chosen = select_candidates(buckets, global_count=parameters.global_picks, fallbacks=(), available=eligible)
    if not chosen.picks:
        return {}
    if parameters.global_picks or parameters.weight_mode == "score-global":
        return rules.weights(chosen.picks, parameters.exposure_rate)
    if parameters.weight_mode == "score":
        return rules.buckets(chosen.buckets, parameters.picks_per_bucket, parameters.exposure_rate)
    weight = min(parameters.exposure_rate / len(chosen.picks), parameters.max_weight_pct)
    return {symbol: weight for symbol, _ in chosen.picks}
