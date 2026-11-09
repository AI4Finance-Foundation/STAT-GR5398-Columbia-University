"""Daily prices bundled under data/prices/.

daily_adjusted.parquet  split- and dividend-adjusted OHLCV for every stock that
                        was an S&P 500 / Nasdaq-100 member over the window,
                        plus SPY, QQQ, SMH, GLD, SLV and ^VIX
market_series.parquet   closes of the 11 sector ETFs, ^VIX and SPY, used by the
                        exposure engine

The trading calendar is taken from SPY's sessions, so no exchange-calendar
dependency is needed.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd


class PriceStore:
    def __init__(self, path: str | Path):
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        self.frame = df
        self.close = df.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
        self.open = df.pivot_table(index="date", columns="symbol", values="open", aggfunc="last").sort_index()
        self.sessions = pd.DatetimeIndex(self.close["SPY"].dropna().index)

    # ---- calendar ----------------------------------------------------
    def last_session_on_or_before(self, day: date) -> Optional[date]:
        pos = self.sessions.searchsorted(pd.Timestamp(day), side="right") - 1
        return self.sessions[pos].date() if pos >= 0 else None

    def last_session_before_month(self, day: date) -> Optional[date]:
        """The cutoff a monthly model uses: last session before the 1st of day's month."""
        first = pd.Timestamp(day.year, day.month, 1)
        pos = self.sessions.searchsorted(first, side="left") - 1
        return self.sessions[pos].date() if pos >= 0 else None

    # ---- lookups -----------------------------------------------------
    def close_on_or_before(self, symbols: Iterable[str], day: date, max_days: int = 14) -> Dict[str, float]:
        """Last close on or before ``day`` within ``max_days``; absent if none."""
        ts = pd.Timestamp(day)
        window = self.close.loc[ts - pd.Timedelta(days=max_days): ts]
        out: Dict[str, float] = {}
        for symbol in symbols:
            if symbol in window.columns:
                s = window[symbol].dropna()
                if not s.empty:
                    out[symbol] = float(s.iloc[-1])
        return out

    def price_as_of_lookup(self):
        """Callable for MLStockSelector(price_as_of=...)."""
        return lambda symbols, as_of: self.close_on_or_before(symbols, as_of)
