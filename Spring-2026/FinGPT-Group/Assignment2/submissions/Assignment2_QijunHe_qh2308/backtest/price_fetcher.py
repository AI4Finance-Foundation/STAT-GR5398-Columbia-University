"""
Price fetching helpers for backtesting.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import pandas as pd
import yfinance as yf

_RETURN_CACHE: dict[tuple[str, str, str], Optional[float]] = {}
_DISK_CACHE: dict[str, Optional[float]] = {}
_DISK_CACHE_LOADED = False
_CACHE_PATH_ENV = "FINGPT_YF_CACHE_PATH"
_DEFAULT_CACHE_PATH = os.path.join("output", "yfinance_return_cache.json")


def _cache_key(ticker: str, start_date: str, end_date: str) -> str:
    return f"{ticker}|{start_date}|{end_date}"


def _cache_path() -> str:
    return os.getenv(_CACHE_PATH_ENV, _DEFAULT_CACHE_PATH)


def _load_disk_cache() -> None:
    global _DISK_CACHE_LOADED  # noqa: PLW0603
    if _DISK_CACHE_LOADED:
        return
    _DISK_CACHE_LOADED = True

    path = _cache_path()
    if not os.path.exists(path):
        return

    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            for key, value in payload.items():
                _DISK_CACHE[str(key)] = float(value) if value is not None else None
    except Exception:
        # Corrupt cache should never block backtest execution.
        _DISK_CACHE.clear()


def _save_disk_cache() -> None:
    path = _cache_path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_DISK_CACHE, handle)


def _store_cached_value(
    ticker: str,
    start_date: str,
    end_date: str,
    value: Optional[float],
) -> Optional[float]:
    key = (ticker, start_date, end_date)
    _RETURN_CACHE[key] = value
    _DISK_CACHE[_cache_key(ticker, start_date, end_date)] = value
    _save_disk_cache()
    return value


def get_realized_return(
    ticker: str,
    start_date: str,
    end_date: str,
) -> Optional[float]:
    """
    Fetch weekly OHLC and compute close-to-close realized return.
    """
    _load_disk_cache()
    key = (ticker, start_date, end_date)
    if key in _RETURN_CACHE:
        return _RETURN_CACHE[key]

    disk_key = _cache_key(ticker, start_date, end_date)
    if disk_key in _DISK_CACHE:
        cached = _DISK_CACHE[disk_key]
        _RETURN_CACHE[key] = cached
        return cached

    if not ticker or not start_date or not end_date:
        return _store_cached_value(ticker, start_date, end_date, None)

    try:
        hist = yf.download(
            tickers=ticker,
            start=start_date,
            end=end_date,
            interval="1d",          # daily bars: 5 trading days in any 7-day window
            auto_adjust=True,
            progress=False,
        )
    except Exception:
        return _store_cached_value(ticker, start_date, end_date, None)

    if hist is None or hist.empty:
        return _store_cached_value(ticker, start_date, end_date, None)

    # Flatten MultiIndex columns (yfinance ≥0.2.38 with a single ticker returns
    # columns like ("Close", "AXP"); older versions return flat "Close").
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.get_level_values(0)

    if "Close" not in hist.columns:
        return _store_cached_value(ticker, start_date, end_date, None)

    closes = hist["Close"].dropna()
    if closes.empty:
        return _store_cached_value(ticker, start_date, end_date, None)

    first_close = float(closes.iloc[0])
    last_close = float(closes.iloc[-1])
    if first_close == 0.0:
        return _store_cached_value(ticker, start_date, end_date, None)

    realized_return = (last_close - first_close) / first_close
    return _store_cached_value(ticker, start_date, end_date, realized_return)


def direction_from_return(realized_return: float, threshold: float = 0.001) -> str:
    """
    Convert realized return into up/down/neutral direction label.
    """
    if realized_return > threshold:
        return "up"
    if realized_return < -threshold:
        return "down"
    return "neutral"


def get_cache_stats() -> dict[str, int]:
    """
    Return simple stats for in-memory + disk yfinance return cache.
    """
    _load_disk_cache()
    disk_entries = len(_DISK_CACHE)
    memory_entries = len(_RETURN_CACHE)
    return {
        "disk_entries": disk_entries,
        "memory_entries": memory_entries,
    }

