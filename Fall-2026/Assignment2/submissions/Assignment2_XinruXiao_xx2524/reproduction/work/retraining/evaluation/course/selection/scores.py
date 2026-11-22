"""Replay the precomputed monthly scores instead of retraining every month.

MonthlyPredictions (strategy.py) trains six models each month, 1 to 4
minutes per month. build_scores.py runs it once for every month and saves
the result to data/selection/monthly_scores.parquet; MonthlyScores reads that
file and behaves exactly like MonthlyPredictions (same .get(day) and .cutoff).
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from .config import DATA
from .strategy import Prediction

SCORES_PATH = DATA / "selection" / "monthly_scores.parquet"


class MonthlyScores:
    def __init__(self, path: Path = SCORES_PATH):
        if not Path(path).is_file():
            raise FileNotFoundError(f"{path} is missing: unpack the course data bundle into data/, "
                                    f"or rebuild it with python selection/build_scores.py")
        df = pd.read_parquet(path)
        self._rows: Dict[date, Tuple[Prediction, ...]] = {}
        self._cutoffs: Dict[date, date] = {}
        for month, g in df.groupby("month"):
            key = pd.Timestamp(month).date().replace(day=1)
            self._rows[key] = tuple(Prediction(r.symbol, r.bucket, float(r.rank_score), float(r.weight_score))
                                    for r in g.itertuples(index=False))
            self._cutoffs[key] = pd.Timestamp(g["cutoff"].iloc[0]).date()
        self.cutoff = None

    @property
    def months(self):
        return sorted(self._rows)

    def get(self, day: date) -> Tuple[Prediction, ...]:
        """Every scored stock for the month containing ``day``."""
        key = day.replace(day=1)
        if key not in self._rows:
            raise LookupError(f"No scores for {key:%Y-%m}; available {self.months[0]:%Y-%m} to {self.months[-1]:%Y-%m}")
        self.cutoff = self._cutoffs[key]
        return self._rows[key]
