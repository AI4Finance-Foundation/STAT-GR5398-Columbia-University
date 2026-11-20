"""Point-in-time data access for the valuation and timing modules.

``DataBundle`` loads the data once. ``DataView(bundle, as_of)`` is what your
functions receive: every method returns only rows dated on or before
``as_of``, so a look-ahead through the view is impossible.

    data.close(["SPY", "NVDA"], lookback=260)   adjusted daily closes, sessions x symbols
    data.last_close(["NVDA"])                    latest close on or before as_of
    data.market("^VIX")                          VIX, SPY and 11 sector ETF closes
    data.macro("DGS10")                          FRED: DGS10, T10Y2Y, BAA10Y, BAMLH0A0HYM2
    data.fundamentals(["NVDA"])                  filings known by as_of, one row per quarter
    data.members()                               S&P 500 and Nasdaq-100 members on as_of
    data.extra("news_sentiment")                 your own student_data/news_sentiment.parquet
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, Iterable, Optional, Set

import pandas as pd

from selection.config import DATA, load_exposure_inputs

from .guard import allow_io
from .prices import PriceStore
from .store import FundamentalStore
from .universe import TradeableUniverse

ROOT = DATA.parent
STUDENT_DATA = ROOT / "student_data"

# Columns dated after the filing or used as training labels. trade_price is the
# close about two months after quarter end (often later than the decision
# date), adj_close_q leaked in Assignment 1, y_return is next quarter's return.
HIDDEN_FUNDAMENTAL_COLUMNS = ("trade_price", "adj_close_q", "y_return")


@dataclass
class DataBundle:
    prices: PriceStore
    market_frame: pd.DataFrame
    macro_series: Dict[str, pd.Series]
    store: FundamentalStore
    universe: TradeableUniverse
    _extra: Dict[str, pd.DataFrame] = field(default_factory=dict)

    @classmethod
    def load(cls, prices: Optional[PriceStore] = None,
             universes: Iterable[str] = ("sp500", "nasdaq100")) -> "DataBundle":
        prices = prices or PriceStore(DATA / "prices" / "daily_adjusted.parquet")
        market, macro = load_exposure_inputs()
        macro = {k: v.dropna() for k, v in macro.items()}
        db = str(DATA / "fundamentals.db")
        return cls(prices, market, macro, FundamentalStore(db), TradeableUniverse.from_db_path(db, universes))

    def extra_frame(self, name: str) -> pd.DataFrame:
        if name not in self._extra:
            path = STUDENT_DATA / f"{name}.parquet"
            if not path.is_file():
                raise FileNotFoundError(f"{path} does not exist")
            df = pd.read_parquet(path)
            if "date" not in df.columns:
                raise ValueError(f"{path} needs a 'date' column: the day each row became known")
            df["date"] = pd.to_datetime(df["date"])
            self._extra[name] = df.sort_values("date")
        return self._extra[name]


class DataView:
    """Everything a valuation or timing function may see on ``as_of``."""

    def __init__(self, bundle: DataBundle, as_of):
        self._b = bundle
        self.as_of = pd.Timestamp(as_of).normalize()

    @property
    def as_of_date(self) -> date:
        return self.as_of.date()

    # ---- prices ------------------------------------------------------
    def close(self, symbols: Optional[Iterable[str]] = None, lookback: Optional[int] = None) -> pd.DataFrame:
        frame = self._b.prices.close.loc[: self.as_of]
        if symbols is not None:
            frame = frame.reindex(columns=list(symbols))
        if lookback is not None:
            frame = frame.tail(int(lookback))
        return frame.copy()

    def last_close(self, symbols: Iterable[str]) -> Dict[str, float]:
        return self._b.prices.close_on_or_before(list(symbols), self.as_of_date)

    def market(self, symbol: str) -> pd.Series:
        m = self._b.market_frame
        if symbol not in set(m["symbol"]):
            raise KeyError(f"{symbol} is not in market_series; available: {sorted(set(m['symbol']))}")
        s = m[(m["symbol"] == symbol) & (m["date"] <= self.as_of)]
        return pd.Series(s["close"].values, index=pd.DatetimeIndex(s["date"]), name=symbol)

    def macro(self, series_id: str) -> pd.Series:
        if series_id not in self._b.macro_series:
            raise KeyError(f"{series_id} not found; available: {sorted(self._b.macro_series)}")
        return self._b.macro_series[series_id].loc[: self.as_of].copy()

    # ---- filings and index membership --------------------------------
    def fundamentals(self, tickers: Optional[Iterable[str]] = None) -> pd.DataFrame:
        """One row per (ticker, quarter) for filings accepted by as_of.

        Per-share and ratio columns are single-quarter values; add the last
        four quarters for trailing-twelve-month figures.
        """
        with allow_io():
            df = self._b.store.get_fundamental_data(
                tickers=sorted(set(tickers)) if tickers is not None else None,
                as_of_date=self.as_of.strftime("%Y-%m-%d"))
        df = df.rename(columns={"tic": "ticker"})
        return df.drop(columns=[c for c in HIDDEN_FUNDAMENTAL_COLUMNS if c in df.columns])

    def members(self) -> Set[str]:
        with allow_io():
            return set(self._b.universe.members_at(self.as_of.strftime("%Y-%m-%d")))

    # ---- your own dated data ------------------------------------------
    def extra(self, name: str) -> pd.DataFrame:
        """Rows of student_data/<name>.parquet whose 'date' is on or before as_of."""
        with allow_io():
            df = self._b.extra_frame(name)
        return df[df["date"] <= self.as_of].copy()

    # ---- raw inputs for timing/engine.py --------------------------------
    def market_frame(self) -> pd.DataFrame:
        m = self._b.market_frame
        return m[m["date"] <= self.as_of].copy()

    def macro_dict(self) -> Dict[str, pd.Series]:
        return {k: v.loc[: self.as_of].copy() for k, v in self._b.macro_series.items()}
