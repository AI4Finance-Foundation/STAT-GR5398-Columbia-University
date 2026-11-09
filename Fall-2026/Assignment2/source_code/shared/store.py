"""Read-only access to the bundled fundamentals database (data/fundamentals.db).

Tables
  fundamental_data     one row per (ticker, quarter, filing version); quarterly
                       ratios, filing/accepted dates, trade_price and y_return
  universe_membership  index constituent snapshots (sp500, nasdaq100) by date
  symbol_changes       ticker renames with their effective dates

Everything here is point-in-time: ``get_fundamental_data(as_of_date=...)``
returns only filings accepted by that date (latest visible version per
quarter), and ``get_universe_members_at`` returns the index members on that
date under the tickers in force then.
"""
from __future__ import annotations

import bisect
import logging
import sqlite3
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Set

import pandas as pd

logger = logging.getLogger(__name__)

TEXT_COLUMNS = {"ticker", "datadate", "filing_date", "accepted_date", "gsector"}

FUNDAMENTAL_COLUMNS = [
    "ticker", "datadate", "gsector", "adj_close_q", "trade_price",
    "filing_date", "accepted_date",
    "EPS", "BPS", "DPS",
    "pe", "ps", "pb", "peg",
    "price_to_fcf", "price_to_ocf", "ev_multiple",
    "roe", "net_income_ratio",
    "gross_margin", "operating_margin", "ebitda_margin",
    "pretax_margin", "effective_tax_rate",
    "ebt_per_ebit", "net_income_per_ebt",
    "cur_ratio", "quick_ratio", "cash_ratio",
    "debt_ratio", "debt_to_equity", "debt_to_assets",
    "debt_to_capital", "lt_debt_to_capital",
    "financial_leverage", "interest_coverage",
    "debt_service_coverage", "debt_to_mktcap",
    "solvency_ratio",
    "acc_rec_turnover", "asset_turnover",
    "fixed_asset_turnover", "inventory_turnover",
    "payables_turnover", "wc_turnover",
    "fcf_per_share", "ocf_per_share", "cash_per_share",
    "capex_per_share",
    "fcf_to_ocf", "ocf_ratio", "ocf_to_sales",
    "ocf_coverage", "st_ocf_coverage", "capex_coverage",
    "revenue_per_share", "tangible_bvps", "interest_debt_per_share",
    "dividend_payout", "dividend_yield", "div_capex_coverage",
    "y_return",
]



class FundamentalStore:
    """Read-only view of the fundamentals SQLite database."""

    def __init__(self, db_path: str, *, read_only: bool = True):
        if not Path(db_path).is_file():
            raise FileNotFoundError(f"Fundamentals database is missing: {db_path}")
        self._db_path = db_path
        self._universe_cache: Dict[str, pd.DataFrame] = {}
        self._symbol_changes: Optional[list] = None

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(Path(self._db_path).resolve().as_uri() + "?mode=ro", uri=True, timeout=30)

    def get_fundamental_data(
        self,
        tickers: Optional[list] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        as_of_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load one filing per (ticker, datadate) from SQLite.

        When ``as_of_date`` is supplied, return the latest filing version that
        was visible by that date. Rows without filing metadata are kept only
        after their modeled tradedate, using the same conservative lag as the
        selector.
        """
        conditions = []
        params: list[Any] = []
        if tickers:
            placeholders = ",".join(["?"] * len(tickers))
            conditions.append(f"ticker IN ({placeholders})")
            params.extend(tickers)
        if start_date:
            conditions.append("datadate >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("datadate <= ?")
            params.append(end_date)
        if as_of_date:
            conditions.append("datadate <= ?")
            params.append(as_of_date)
            conditions.append(
                """
                (
                    (COALESCE(accepted_date, '') != '' AND accepted_date <= ?)
                    OR (
                        COALESCE(accepted_date, '') = ''
                        AND COALESCE(filing_date, '') != ''
                        AND filing_date <= ?
                    )
                    OR (
                        COALESCE(accepted_date, '') = ''
                        AND COALESCE(filing_date, '') = ''
                        AND CASE
                            WHEN substr(datadate, 6, 5) = '03-31' THEN date(substr(datadate, 1, 4) || '-06-01')
                            WHEN substr(datadate, 6, 5) = '06-30' THEN date(substr(datadate, 1, 4) || '-09-01')
                            WHEN substr(datadate, 6, 5) = '09-30' THEN date(substr(datadate, 1, 4) || '-12-01')
                            WHEN substr(datadate, 6, 5) = '12-31' THEN date((CAST(substr(datadate, 1, 4) AS INTEGER) + 1) || '-03-01')
                            ELSE NULL
                        END <= date(?)
                    )
                )
                """
            )
            params.extend([as_of_date, as_of_date, as_of_date])

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        cols = ", ".join(FUNDAMENTAL_COLUMNS)
        sql = f"""
        WITH ranked AS (
            SELECT
                {cols},
                ROW_NUMBER() OVER (
                    PARTITION BY ticker, datadate
                    ORDER BY
                        COALESCE(accepted_date, '') DESC,
                        COALESCE(filing_date, '') DESC,
                        rowid DESC
                ) AS rn
            FROM fundamental_data
            {where_clause}
        )
        SELECT {cols}
        FROM ranked
        WHERE rn = 1
        ORDER BY ticker, datadate
        """
        with self._connect() as conn:
            df = pd.read_sql(sql, conn, params=params)

        if df.empty:
            return pd.DataFrame(columns=["tic", *[c for c in FUNDAMENTAL_COLUMNS if c != "ticker"]])

        df = df.rename(columns={"ticker": "tic"})
        # Coerce every numeric column on the way out. SQLite affinity lets
        # junk like '' live in REAL columns; coercing here (errors="coerce"
        # -> NaN) makes every reader immune to dirty rows regardless of how
        # or when they were written.
        for col in FUNDAMENTAL_COLUMNS:
            if col not in TEXT_COLUMNS and col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        logger.info("Loaded %d deduplicated fundamental records (%d tickers)", len(df), df["tic"].nunique())
        return df

    def _load_universe_snapshots(self, universe: str) -> pd.DataFrame:
        if universe in self._universe_cache:
            return self._universe_cache[universe]
        with self._connect() as conn:
            df = pd.read_sql(
                """
                SELECT snapshot_date AS date, tickers
                FROM universe_membership
                WHERE universe = ?
                ORDER BY snapshot_date
                """,
                conn,
                params=[universe],
            )
        if df.empty:
            return pd.DataFrame(columns=["date", "tickers"])
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        self._universe_cache[universe] = df
        return df

    SYMBOL_CHANGE_WINDOW = timedelta(days=90)

    def _load_symbol_changes(self) -> list:
        if self._symbol_changes is None:
            try:
                with self._connect() as conn:
                    rows = conn.execute(
                        "SELECT change_date, old_symbol, new_symbol FROM symbol_changes "
                        "ORDER BY change_date, old_symbol"
                    ).fetchall()
            except sqlite3.OperationalError as exc:
                # A store written before this table existed (or a frozen test
                # snapshot) has no ticker changes: membership reads as stored.
                if "no such table" not in str(exc):
                    raise
                rows = []
            self._symbol_changes = [(pd.Timestamp(d), old, new) for d, old, new in rows]
            self._symbol_change_dates = [effective for effective, _, _ in self._symbol_changes]
        return self._symbol_changes

    def _apply_symbol_changes(self, members: Set[str], as_of: pd.Timestamp) -> Set[str]:
        """Membership as of ``as_of`` under the tickers in force on that date.

        A change applies from its effective date for SYMBOL_CHANGE_WINDOW, and
        only when the snapshot still lists the old ticker but not the new one
        (once the source catches up the stored list is already right). Earlier
        dates keep the old ticker: point-in-time reads see what traded then.
        """
        changes = self._load_symbol_changes()
        if not changes:
            return members
        # Backtests read membership once per session: touch only the window.
        dates = self._symbol_change_dates
        lo = bisect.bisect_left(dates, as_of - self.SYMBOL_CHANGE_WINDOW)
        hi = bisect.bisect_right(dates, as_of)
        mapped = set(members)
        for _, old, new in changes[lo:hi]:
            if old in mapped and new not in mapped:
                mapped.discard(old)
                mapped.add(new)
        if mapped != members:
            logger.debug("Membership at %s read under ticker changes: -%s +%s",
                         as_of.date(), sorted(members - mapped), sorted(mapped - members))
        return mapped

    def get_universe_members_at(
        self,
        universe: str,
        as_of_date: str,
        api_key: str = "",
    ) -> Set[str]:
        """Point-in-time membership, under the tickers in force on ``as_of_date``."""
        members = self._stored_members_at(universe, as_of_date, api_key=api_key)
        return self._apply_symbol_changes(members, pd.to_datetime(as_of_date)) if members else members

    def _stored_members_at(
        self,
        universe: str,
        as_of_date: str,
        api_key: str = "",
    ) -> Set[str]:
        df = self._load_universe_snapshots(universe)
        if df.empty:
            return set()
        target = pd.to_datetime(as_of_date)
        valid = df[df["date"] <= target]
        if valid.empty:
            # Fail closed: falling back to the NEWEST snapshot would
            # rank today's survivors in a historical backtest. Historical
            # membership must be backfilled instead.
            raise LookupError(
                f"No {universe} membership snapshot on or before {as_of_date} "
                f"(earliest stored: {df['date'].min().date()}). Point-in-time "
                f"constituents must cover the requested date; narrow the window."
            )
        tickers_str = valid.iloc[-1]["tickers"]
        return {t.strip() for t in str(tickers_str).split(",") if t.strip()}

    def get_universe_members_in_range(self, universe: str, start: str, end: str) -> Set[str]:
        """Metadata loading union for [start, end), never a buy-eligibility set."""
        if pd.Timestamp(end) <= pd.Timestamp(start):
            raise ValueError("Universe interval must have positive length")
        members = self.get_universe_members_at(universe, start)
        if not members:
            raise LookupError(f"No {universe} membership at interval start {start}")
        snapshots = self._load_universe_snapshots(universe)
        inside = snapshots[(snapshots["date"] > pd.Timestamp(start)) & (snapshots["date"] < pd.Timestamp(end))]
        for snapshot_date, tickers in zip(inside["date"], inside["tickers"]):
            members |= self._apply_symbol_changes(
                {symbol.strip() for symbol in str(tickers).split(",") if symbol.strip()}, snapshot_date)
        # A member renamed inside the interval trades under both tickers in it.
        for effective, old, new in self._load_symbol_changes():
            if pd.Timestamp(start) <= effective < pd.Timestamp(end) and old in members:
                members.add(new)
        return members

    def get_sp500_members_at(self, as_of_date: str) -> Set[str]:
        return self.get_universe_members_at("sp500", as_of_date)

    def get_latest_sp500_members(self) -> Set[str]:
        df = self._load_universe_snapshots("sp500")
        if df.empty:
            return set()
        tickers_str = df.iloc[-1]["tickers"]
        return {t.strip() for t in str(tickers_str).split(",") if t.strip()}

    def get_data_date_range(self) -> tuple:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT MIN(datadate), MAX(datadate)
                FROM (
                    SELECT ticker, datadate, accepted_date
                    FROM fundamental_data
                    GROUP BY ticker, datadate, accepted_date
                )
                """
            ).fetchone()
        return (row[0], row[1]) if row else (None, None)

    def get_ticker_count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(DISTINCT ticker) FROM fundamental_data").fetchone()
        return row[0] if row else 0
