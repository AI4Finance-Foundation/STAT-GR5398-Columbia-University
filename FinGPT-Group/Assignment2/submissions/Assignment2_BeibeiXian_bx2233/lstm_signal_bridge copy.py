from __future__ import annotations

import re
import math
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Dow30 company-name → ticker lookup
# ---------------------------------------------------------------------------
_DOW30_NAME_TO_TICKER: Dict[str, str] = {
    "apple":            "AAPL",
    "microsoft":        "MSFT",
    "amazon":           "AMZN",
    "american express": "AXP",
    "amgen":            "AMGN",
    "boeing":           "BA",
    "caterpillar":      "CAT",
    "salesforce":       "CRM",
    "cisco":            "CSCO",
    "chevron":          "CVX",
    "disney":           "DIS",
    "walt disney":      "DIS",
    "dow":              "DOW",
    "goldman sachs":    "GS",
    "home depot":       "HD",
    "honeywell":        "HON",
    "ibm":              "IBM",
    "intel":            "INTC",
    "johnson & johnson": "JNJ",
    "johnson":          "JNJ",
    "jpmorgan":         "JPM",
    "jp morgan":        "JPM",
    "coca-cola":        "KO",
    "coca cola":        "KO",
    "mcdonald":         "MCD",
    "3m":               "MMM",
    "merck":            "MRK",
    "nike":             "NKE",
    "procter":          "PG",
    "procter & gamble": "PG",
    "travelers":        "TRV",
    "unitedhealth":     "UNH",
    "united health":    "UNH",
    "visa":             "V",
    "verizon":          "VZ",
    "walmart":          "WMT",
    "walgreens":        "WBA",
    "walgreen":         "WBA",
}

_TICKER_RE = re.compile(
    r'(?:\(|\bTicker:\s*)([A-Z]{1,5})(?:\)|\b)',
    re.IGNORECASE,
)

_DATE_RES = [
    re.compile(r'(\d{4}-\d{2}-\d{2})'),        
    re.compile(r'(\d{2}/\d{2}/\d{4})'),         
    re.compile(r'(\w+ \d{1,2},?\s+\d{4})'),     
]

# Known Dow30 tickers 
_DOW30_TICKERS = set(_DOW30_NAME_TO_TICKER.values())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _extract_ticker(text: str) -> Optional[str]:
    """Try to extract the ticker symbol from a FinGPT input string."""
    # 1. Explicit parenthetical like (AAPL) or (MCD)
    for m in _TICKER_RE.finditer(text or ""):
        cand = m.group(1).upper()
        if cand in _DOW30_TICKERS:
            return cand

    # 2. Company name lookup (lowercase, partial match)
    ltext = (text or "").lower()
    # Sort by length descending so "johnson & johnson" beats "johnson"
    for name, tick in sorted(
        _DOW30_NAME_TO_TICKER.items(), key=lambda x: -len(x[0])
    ):
        if name in ltext:
            return tick

    return None


def _extract_date(text: str) -> Optional[pd.Timestamp]:
    """Try to extract the earliest date mentioned in a FinGPT input string."""
    dates_found = []
    for pat in _DATE_RES:
        for m in pat.finditer(text or ""):
            try:
                dates_found.append(pd.to_datetime(m.group(1)))
            except Exception:
                pass
    if dates_found:
        return min(dates_found)
    return None


def _tanh_normalize(z: float) -> float:
    """Squeeze a z-score into (-1, +1) via tanh(z/2)."""
    return math.tanh(z / 2.0)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class LSTMSignalBridge:
    
    def __init__(self, csv_path: str, date_window_days: int = 7):
        self.date_window = timedelta(days=date_window_days)
        self._df = self._load(csv_path)
        # Build a quick-lookup index: ticker -> sorted (date, pred_ret_5d)
        self._idx: Dict[str, pd.DataFrame] = {}
        for tick, grp in self._df.groupby("ticker"):
            self._idx[tick] = grp.sort_values("date").reset_index(drop=True)

    # ---- private -----------------------------------------------------------

    @staticmethod
    def _load(path: str) -> pd.DataFrame:
        df = pd.read_csv(path, parse_dates=["date"])
        required = {"date", "ticker", "pred_ret_5d"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")
        df = df.dropna(subset=["date", "ticker", "pred_ret_5d"])
        df["ticker"] = df["ticker"].str.upper().str.strip()
        return df

    def _lookup(self, ticker: str, date: pd.Timestamp) -> Optional[float]:
        """Return the nearest LSTM pred_ret_5d for (ticker, date)."""
        sub = self._idx.get(ticker.upper())
        if sub is None or len(sub) == 0:
            return None
        diffs = (sub["date"] - date).abs()
        idx_min = diffs.idxmin()
        if diffs[idx_min] <= self.date_window:
            return float(sub.loc[idx_min, "pred_ret_5d"])
        return None

    # ---- public ------------------------------------------------------------

    def get_scores_for_fingpt(
        self,
        input_texts: List[str],
        target_texts: Optional[List[str]] = None,
        fallback: float = 0.0,
        verbose: bool = True,
    ) -> List[float]:
        """
        For each FinGPT test sample, parse (ticker, date), look up the LSTM
        prediction, normalise to (-1, +1) via tanh(z/2), and return the list.

        Parameters
        ----------
        input_texts  : list of raw `input` strings from the FinGPT dataset
        target_texts : (optional) ground-truth strings; not used for lookup
                       but helps track parse success in verbose mode
        fallback     : value to use when no LSTM prediction is found (default 0)
        verbose      : print parse-success summary

        Returns
        -------
        List[float] of normalised LSTM scores, same length as input_texts.
        """
        scores: List[float] = []
        n_found = 0
        n_no_ticker = 0
        n_no_date = 0
        n_no_match = 0

        for i, text in enumerate(input_texts):
            ticker = _extract_ticker(text)
            date = _extract_date(text)

            if ticker is None:
                n_no_ticker += 1
                scores.append(fallback)
                continue
            if date is None:
                n_no_date += 1
                scores.append(fallback)
                continue

            raw = self._lookup(ticker, date)
            if raw is None:
                n_no_match += 1
                scores.append(fallback)
                continue

            scores.append(_tanh_normalize(raw))
            n_found += 1

        if verbose:
            n = len(input_texts)
            print(
                f"[LSTMSignalBridge] Matched {n_found}/{n} samples "
                f"(no-ticker={n_no_ticker}, no-date={n_no_date}, "
                f"no-LSTM-match={n_no_match})"
            )

        return scores

    def describe(self) -> pd.DataFrame:
        """Summary of the loaded LSTM predictions."""
        return pd.DataFrame(
            {
                "n_rows": len(self._df),
                "n_tickers": self._df["ticker"].nunique(),
                "date_min": self._df["date"].min(),
                "date_max": self._df["date"].max(),
                "pred_mean": self._df["pred_ret_5d"].mean(),
                "pred_std": self._df["pred_ret_5d"].std(),
            },
            index=[0],
        )

    def dow30_coverage(self) -> pd.DataFrame:
        """Show which Dow30 tickers are present in the LSTM CSV."""
        rows = []
        for tick in sorted(_DOW30_TICKERS):
            sub = self._idx.get(tick)
            rows.append(
                {
                    "ticker": tick,
                    "n_pred": len(sub) if sub is not None else 0,
                    "date_min": sub["date"].min() if sub is not None else None,
                    "date_max": sub["date"].max() if sub is not None else None,
                }
            )
        return pd.DataFrame(rows).sort_values("n_pred", ascending=False)
