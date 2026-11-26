"""Valuation module: is a stock overpriced or underpriced?

The valuation group edits only this file. Implement two functions.

value(as_of, tickers, data) -> DataFrame
    One row per ticker you can value, with columns
        ticker, fair_value, low, high, mispricing, method_spread
    fair_value     your per-share estimate
    low, high      a range you believe contains the value (low <= fair_value <= high)
    mispricing     fair_value / price - 1, where price = data.last_close([ticker])
                   positive = undervalued, negative = overvalued
    method_spread  highest / lowest fair value across your methods (1.0 for one method)
    Leave out tickers you cannot value (negative earnings, missing data).

filter_candidates(as_of, candidates, data) -> candidates
    Called once a week with every stock the selection model scored that month
    (a tuple of selection.Prediction(symbol, bucket, rank_score, weight_score)).
    Return the ones you keep. The portfolio then takes the 10 best-ranked
    growth_tech names among them, so a removed name is replaced by the next one.
    You may only remove rows. Only growth_tech names can enter the portfolio,
    so valuing other buckets is wasted time.

Rules
  * Read data only through ``data`` (shared.data_view.DataView). It returns
    nothing dated after as_of. Reading files or the network inside these
    functions raises an error.
  * Do not use an LLM to produce numbers for past dates: it has seen the future.
  * Do not use the database columns pe and ev_multiple; their basis is not
    verified. Compute multiples from EPS, revenue_per_share and prices.

Useful calls
    data.fundamentals(tickers)    filings known by as_of; per-share values are single-quarter
    data.last_close(tickers)      latest close on or before as_of
    data.close(tickers, 1260)     about 5 years of daily closes (beta, history bands)
    data.members()                index members on as_of, for peer groups
    data.macro("DGS10")           10-year Treasury yield, percent

Building blocks
    valuation/finrobot/           FinRobot's DCF engine: calculate_dcf, calculate_sensitivity, adjust_beta_blume
    valuation/examples.py         a worked trailing-P/E method and a per-share DCF

Run
    python evaluate_valuation.py --period dev               rank IC and range coverage
    python run_portfolio.py --period dev --valuation        effect on the portfolio
    python check_submission.py --valuation
"""
import numpy as np
import pandas as pd

# Import at module load: imports read files, which the look-ahead guard blocks inside value()/filter_candidates().
from selection import BUCKET  # "growth_tech": the only bucket the portfolio buys
from valuation.finrobot import DCFInputs, adjust_beta_blume, calculate_dcf, calculate_sensitivity

COLUMNS = ["ticker", "fair_value", "low", "high", "mispricing", "method_spread"]

# ---- shared interface for each valuation method ----
# Every method returns one row per input ticker with these columns.
#   fair_value, low, high   per-share values; NaN when missing
#   missing                 True when the method could not value the ticker
#   reason                  why it is missing ("" when valued); DCF uses MISSING_REASONS
METHOD_COLUMNS = ["ticker", "fair_value", "low", "high", "missing", "reason"]
MISSING_REASONS = ("insufficient_data", "loss_making", "negative_value", "basis_mismatch")
# basis_mismatch: per-share filings and adjusted prices are on different share bases
# (e.g. KLAC ~10x, ADRs like BIDU/NTES). Detected by basis_ok() below.


def _all_missing(tickers, reason: str) -> pd.DataFrame:
    tickers = list(tickers)
    return pd.DataFrame({
        "ticker": tickers,
        "fair_value": np.nan,
        "low": np.nan,
        "high": np.nan,
        "missing": True,
        "reason": reason,
    }, columns=METHOD_COLUMNS)


# ---- data checks used by the DCF (data audit, 11/19) ----
# Basis check: (our price / latest quarterly revenue_per_share) / the filing's own ps.
# Same basis gives ~0.9 (dev, 3,171 checks: 1st-99th pct 0.17-1.50); a split or ADR mismatch gives ~0.1.
# The ps column is used only for this consistency check, never as a valuation input.
BASIS_RATIO_RANGE = (0.4, 2.5)


def basis_ok(last_filing, price) -> bool:
    """False when the per-share filings and the adjusted price are on different share bases."""
    rps, ps = last_filing["revenue_per_share"], last_filing["ps"]
    if not (rps > 0 and ps > 0):
        return True  # cannot check; other checks still apply
    return BASIS_RATIO_RANGE[0] <= price / rps / ps <= BASIS_RATIO_RANGE[1]


# Basis change (11/25): basis_ok() checks only the latest filing, so a ticker whose filings switch share
# basis mid-window (KLAC from the 2024-09-30 filing, about 10x) passes while its TTM sums and its history
# mix bases. The same ratio, priced at each filing's known date, is stable from quarter to quarter on a
# fixed basis: across the whole database only 3 consecutive moves exceed 4x (KLAC 2024-09, CRWD 2024-09,
# RRC 2020-03), none before 2024 in growth_tech, so dev and val are unchanged.
BASIS_JUMP = 4.0


def basis_stable(filings, closes) -> bool:
    """False when the basis ratio of basis_ok() moves more than BASIS_JUMP-fold between consecutive filings."""
    if closes is None or len(filings) < 2:
        return True
    closes = closes.dropna()
    if closes.empty:
        return True
    if "known_date" in filings:
        known = pd.to_datetime(filings["known_date"], errors="coerce")
    else:
        known = pd.to_datetime(filings["accepted_date"], errors="coerce").fillna(
            pd.to_datetime(filings["filing_date"], errors="coerce"))
    pos = pd.DatetimeIndex(closes.index).searchsorted(pd.DatetimeIndex(known.dt.normalize()), side="right") - 1
    rps = pd.to_numeric(filings["revenue_per_share"], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    ps = pd.to_numeric(filings["ps"], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    ratios = np.array([closes.iloc[i] / r / q for i, r, q in zip(pos, rps, ps) if i >= 0 and r > 0 and q > 0])
    if len(ratios) < 2:
        return True
    jumps = ratios[1:] / ratios[:-1]
    return bool(np.all((jumps <= BASIS_JUMP) & (jumps >= 1 / BASIS_JUMP)))


# Filing-date check (data audit): for about 20 tickers (ASML, BIDU, CPAY, GEN, COHR, ...) the database's
# accepted_date sits within a day of the quarter end, weeks before any real release, so the course store
# shows those quarters early. Their true release dates are not in the data, and a fixed delay is not
# safe (11/24 used quarter end + 45 days; the 11/25 review found BIDU 2019-Q4 released on 2020-02-27,
# after that). Such filings are treated as missing.


def known_filings(filings: pd.DataFrame, as_of=None) -> pd.DataFrame:
    """Filings from data.fundamentals(), without those whose accepted date cannot be trusted."""
    if filings.empty:
        return filings
    period_end = pd.to_datetime(filings["datadate"], errors="coerce")
    known = pd.to_datetime(filings["accepted_date"], errors="coerce").fillna(
        pd.to_datetime(filings["filing_date"], errors="coerce"))
    return filings.loc[~(known - period_end).dt.days.between(-1, 1)].copy()


def contiguous_quarters(dates: pd.Series, n: int) -> bool:
    """True when the last n filings (sorted by datadate) have no missing quarter.

    The database skips quarters for some tickers (e.g. CRM 2020-06), so "the last 4 rows"
    is not always a year. n quarters should span about (n - 1) * 91 days.
    """
    return len(dates) >= n and (dates.iloc[-1] - dates.iloc[-n]).days <= (n - 1) * 91 + 30


# =============================================================================
# DCF
# =============================================================================
# Fixed assumptions, chosen before looking at results; every one goes in the report.
ERP = 0.05                    # equity risk premium
CREDIT_SPREAD = 0.02          # cost of debt = DGS10 + spread
TERMINAL_GROWTH = 0.025
NWC_PCT = 0.0                 # working capital is not in the database
DEFAULT_TAX = 0.21            # used only when effective_tax_rate is missing
YEARS, HOLD_YEARS = 10, 5     # growth held for 5 years, then fades linearly to terminal by year 10
GROWTH_CLIP = (-0.10, 0.40)   # one odd year cannot set a 100% growth path
# Filings for most tickers start 2016-06-30 (not 2006 as the README says), so history is short
# early in dev. Base growth uses the longest window available:
#   16+ quarters: 3-year CAGR of TTM revenue
#   8-15:         TTM year-over-year
#   5-7:          latest quarter vs the same quarter a year earlier
MIN_QUARTERS = 5
CAGR_QUARTERS = 16
TTM_YOY_QUARTERS = 8
MAX_FILING_AGE_DAYS = 270     # a latest quarter older than this is stale
MIN_BETA_WEEKS = 52           # fewer weekly returns -> beta = 1.0
BETA_LOOKBACK = 252 * 5
WACC_STEP, TG_STEP = 0.02, 0.01    # range = WACC +/-2pp x terminal growth +/-1pp (calibrated on dev, experiments/calibrate_range.py)


def _clip(x, lo, hi, default=None):
    if x is None or not np.isfinite(x):
        return default
    return float(np.clip(x, lo, hi))


def _growth_path(base):
    """Hold the base growth HOLD_YEARS years, then fade linearly to terminal growth by year YEARS."""
    g0 = _clip(base, *GROWTH_CLIP)
    fade = np.linspace(g0, TERMINAL_GROWTH, YEARS - HOLD_YEARS + 1)[1:]
    return [g0] * HOLD_YEARS + [float(g) for g in fade]


def _weekly_betas(tickers, data):
    """Blume-adjusted beta of weekly returns on SPY, up to 5 years; 1.0 when history is short."""
    px = data.close(list(tickers) + ["SPY"], lookback=BETA_LOOKBACK)
    wk = px.resample("W-FRI").last().pct_change(fill_method=None)
    betas = {}
    for t in tickers:
        pair = wk[[t, "SPY"]].dropna()
        if len(pair) < MIN_BETA_WEEKS or pair["SPY"].var() == 0:
            betas[t] = 1.0
            continue
        raw = np.cov(pair[t], pair["SPY"])[0, 1] / pair["SPY"].var()
        betas[t] = _clip(adjust_beta_blume(raw), 0.0, 5.0, default=1.0)
    return betas


def _dcf_inputs(f, as_of, price, rf, beta, closes=None):
    """DCFInputs fields for one ticker, or (None, reason). ``f`` = its filings sorted by datadate."""
    if len(f) < MIN_QUARTERS or price is None or not np.isfinite(price):
        return None, "insufficient_data"
    if not basis_ok(f.iloc[-1], price) or not basis_stable(f.tail(CAGR_QUARTERS), closes):
        return None, "basis_mismatch"
    dates = pd.to_datetime(f["datadate"])
    if (as_of - dates.iloc[-1]).days > MAX_FILING_AGE_DAYS:
        return None, "insufficient_data"
    if not contiguous_quarters(dates, 4):
        return None, "insufficient_data"
    q = f.tail(4)
    rev = f["revenue_per_share"].to_numpy(dtype=float)
    revenue = rev[-4:].sum()
    if not revenue > 0:
        return None, "insufficient_data"
    debt = max(float(q["interest_debt_per_share"].iloc[-1]), 0.0)
    net_debt = q["interest_debt_per_share"].iloc[-1] - q["cash_per_share"].iloc[-1]
    if not np.isfinite(net_debt):
        return None, "insufficient_data"
    margin = q["ebitda_margin"].median()
    if not margin > 0:
        return None, "loss_making"

    # base growth: the longest window available (see MIN_QUARTERS) with no missing quarter in it
    if contiguous_quarters(dates, CAGR_QUARTERS) and rev[-16:-12].sum() > 0:
        base = (revenue / rev[-16:-12].sum()) ** (1 / 3) - 1
    elif contiguous_quarters(dates, TTM_YOY_QUARTERS) and rev[-8:-4].sum() > 0:
        base = revenue / rev[-8:-4].sum() - 1
    elif contiguous_quarters(dates, 5) and rev[-5] > 0:
        base = rev[-1] / rev[-5] - 1
    else:
        return None, "insufficient_data"

    return {
        "revenue_base": float(revenue),
        "revenue_growth_rates": _growth_path(base),
        "ebitda_margin": _clip(margin, 0.0, 1.0),
        "da_pct_revenue": _clip((q["ebitda_margin"] - q["operating_margin"]).median(), 0.0, 0.5, default=0.0),
        "capex_pct_revenue": _clip(q["capex_per_share"].abs().sum() / revenue, 0.0, 1.0, default=0.0),
        "nwc_pct_revenue": NWC_PCT,
        "tax_rate": _clip(q["effective_tax_rate"].median(), 0.0, 0.35, default=DEFAULT_TAX),
        "risk_free_rate": _clip(rf, 0.0, 0.15),
        "beta": beta,
        "equity_risk_premium": ERP,
        "cost_of_debt": _clip(rf + CREDIT_SPREAD, 0.0, 0.20),
        # Market-value weights, not the book debt_to_capital the course table suggests: buybacks shrink
        # book equity (AAPL book D/C ~0.6 vs market ~0.05), which over-weights cheap debt and pushes
        # WACC toward terminal growth (AAPL fair value came out at 4-5x price with book weights).
        "debt_ratio": _clip(debt / (debt + price), 0.0, 0.95, default=0.0),
        "terminal_growth_rate": TERMINAL_GROWTH,
        "shares_outstanding": 1.0,
        "net_debt": float(net_debt),
    }, ""


def dcf_inputs_table(as_of, tickers, data) -> pd.DataFrame:
    """One row per ticker: its DCF inputs, or the reason it has none. For checking and the report."""
    tickers = list(dict.fromkeys(tickers))
    as_of = pd.Timestamp(as_of)
    dgs10 = data.macro("DGS10")
    if dgs10.empty or not tickers:
        return pd.DataFrame({"ticker": tickers, "reason": "insufficient_data"})
    rf = float(dgs10.iloc[-1]) / 100
    betas = _weekly_betas(tickers, data)
    prices = data.last_close(tickers)
    fund = known_filings(data.fundamentals(tickers), as_of).sort_values(["ticker", "datadate"])
    groups = dict(tuple(fund.groupby("ticker")))
    history = data.close(tickers, lookback=BETA_LOOKBACK)
    rows = []
    for t in tickers:
        inputs, reason = _dcf_inputs(groups.get(t, fund.iloc[:0]), as_of, prices.get(t), rf, betas[t],
                                     history[t] if t in history else None)
        rows.append({"ticker": t, "reason": reason, **(inputs or {})})
    return pd.DataFrame(rows)


def _run_dcf(inputs: dict):
    """(fair_value, low, high, reason) from FinRobot's engine; never raises.

    FinRobot raises ValueError on purpose when a DCF is meaningless (terminal growth >= WACC,
    WACC - growth spread too small, terminal FCF <= 0, equity <= 0): those map to
    "negative_value". Inputs it refuses to accept map to "insufficient_data".
    Range: min / max over the WACC x terminal-growth sensitivity grid (cells FinRobot
    refuses are skipped), always widened to include the base case.
    """
    try:
        dcf_in = DCFInputs(**inputs)
    except ValueError:  # pydantic validation error
        return np.nan, np.nan, np.nan, "insufficient_data"
    try:
        res = calculate_dcf(dcf_in)
        w, g = res.wacc, inputs["terminal_growth_rate"]
        grid = calculate_sensitivity(dcf_in, wacc_range=[w - WACC_STEP, w, w + WACC_STEP],
                                     tg_range=[g - TG_STEP, g, g + TG_STEP])
    except (ValueError, ZeroDivisionError, OverflowError):
        return np.nan, np.nan, np.nan, "negative_value"
    fair = float(res.implied_price)
    if not (np.isfinite(fair) and fair > 0):
        return np.nan, np.nan, np.nan, "negative_value"
    cells = [p for row in grid["implied_prices"] for p in row if p is not None and np.isfinite(p)]
    return fair, min(cells + [fair]), max(cells + [fair]), ""


def dcf_value(as_of, tickers, data) -> pd.DataFrame:
    """Per-share DCF, one row per ticker in METHOD_COLUMNS format."""
    table = dcf_inputs_table(as_of, tickers, data)
    fields = [c for c in table.columns if c not in ("ticker", "reason")]
    rows = []
    for r in table.to_dict("records"):
        if r["reason"]:
            rows.append({"ticker": r["ticker"], "fair_value": np.nan, "low": np.nan, "high": np.nan,
                         "missing": True, "reason": r["reason"]})
            continue
        fair, low, high, reason = _run_dcf({k: r[k] for k in fields})
        rows.append({"ticker": r["ticker"], "fair_value": fair, "low": low, "high": high,
                     "missing": bool(reason), "reason": reason})
    return pd.DataFrame(rows, columns=METHOD_COLUMNS)


# =============================================================================
# EV/EBITDA own-history band, peer P/E, and value()
# Same output as dcf_value() (METHOD_COLUMNS). No method raises: a ticker it cannot value is
# marked missing with a reason.
# =============================================================================
MIN_PRICE_OBSERVATIONS = 252     # a peer needs about a year of prices
MIN_PEERS = 5                    # fewer same-sector peers -> insufficient_sector_peers
MAX_PEER_MULTIPLE = 200.0        # peers with P/E above this are dropped
MAX_QUARTER_AGE_DAYS = 183       # a latest quarter older than this is stale
HISTORICAL_SESSIONS = 5 * 252    # own-history window for EV/EBITDA
MIN_MULTIPLE_OBSERVATIONS = 252  # daily multiples needed for a band
MIN_EV_QUARTERS = 8
EV_BAND_Q = (0.10, 0.90)         # low / high quantiles of own past multiples (calibrated on dev: value() coverage 55% vs 50% target)
PE_BAND_Q = (0.10, 0.90)         # low / high quantiles of the peers' P/E (same calibration)


def _empty_methods():
    return pd.DataFrame(columns=METHOD_COLUMNS)


def _dated_filings(filings, as_of):
    """Filings known by as_of, one per (ticker, quarter): the latest known version of each quarter."""
    cutoff = pd.Timestamp(as_of).normalize()
    filings = filings.copy()
    if not filings.empty:
        filings["datadate"] = pd.to_datetime(filings["datadate"], errors="coerce")
        accepted = pd.to_datetime(filings["accepted_date"], errors="coerce")
        filed = pd.to_datetime(filings["filing_date"], errors="coerce")
        filings["known_date"] = accepted.fillna(filed)
        filings = filings.loc[
            (filings["datadate"] <= cutoff) & (filings["known_date"] <= cutoff)
        ].sort_values(["ticker", "datadate", "known_date"])
        filings["quarter"] = filings["datadate"].dt.to_period("Q")
        filings = filings.drop_duplicates(["ticker", "quarter"], keep="last")
    return filings


def _peer_snapshot(as_of, symbols, data):
    """Per symbol: TTM EPS, sector, trailing P/E, or the reason it cannot be used."""
    cutoff = pd.Timestamp(as_of).normalize()
    history = data.close(symbols, lookback=2 * MIN_PRICE_OBSERVATIONS)
    counts = history.where(np.isfinite(history) & (history > 0)).count()
    prices = data.last_close(symbols)
    filings = _dated_filings(known_filings(data.fundamentals(symbols), cutoff), cutoff)
    groups = dict(tuple(filings.groupby("ticker"))) if not filings.empty else {}
    rows = []
    for ticker in symbols:
        row = {
            "ticker": ticker, "price": prices.get(ticker, np.nan),
            "price_observations": int(counts.get(ticker, 0)),
            "quarter_count": 0, "sector": None, "eps_ttm": np.nan,
            "latest_quarter": pd.NaT, "max_known_date": pd.NaT,
            "trailing_multiple": np.nan, "missing_reason": "",
        }
        f = groups.get(ticker)
        if f is None or len(f) < 4:
            row["quarter_count"] = 0 if f is None else len(f)
            row["missing_reason"] = "insufficient_dated_quarters"
        else:
            q = f.tail(4)
            row["quarter_count"] = len(f)
            row["latest_quarter"] = q["datadate"].iloc[-1]
            row["max_known_date"] = q["known_date"].max()
            sector = q["gsector"].iloc[-1]
            row["sector"] = str(sector).strip() if pd.notna(sector) else None
            earnings = pd.to_numeric(q["EPS"], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
            if not np.all(np.diff(q["quarter"].astype("int64")) == 1):
                row["missing_reason"] = "nonconsecutive_quarters"
            elif (cutoff - row["latest_quarter"]).days > MAX_QUARTER_AGE_DAYS:
                row["missing_reason"] = "stale_fundamentals"
            elif not np.isfinite(earnings).all():
                row["missing_reason"] = "missing_eps"
            else:
                row["eps_ttm"] = float(earnings.sum())
                if row["eps_ttm"] <= 0:
                    row["missing_reason"] = "nonpositive_ttm_eps"
                elif not row["sector"]:
                    row["missing_reason"] = "missing_sector"
                elif not np.isfinite(row["price"]) or row["price"] <= 0:
                    row["missing_reason"] = "missing_price"
                elif row["price_observations"] < MIN_PRICE_OBSERVATIONS:
                    row["missing_reason"] = "insufficient_price_history"
                elif not basis_ok(q.iloc[-1], row["price"]) or not basis_stable(
                        q, history[ticker] if ticker in history else None):
                    row["missing_reason"] = "basis_mismatch"  # 11/24, dev: KLAC/APH/BIDU/NTES got ~+700% mispricing
                else:
                    row["trailing_multiple"] = row["price"] / row["eps_ttm"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("ticker")


def peer_pe_estimates(as_of, tickers, data) -> pd.DataFrame:
    """Trailing P/E against same-sector index members on as_of, with diagnostics and missing reasons.

    fair_value = peer median P/E x own TTM EPS; low / high from the peers' PE_BAND_Q quantiles.
    """
    tickers = list(dict.fromkeys(tickers))
    if not tickers:
        return pd.DataFrame(columns=["ticker", "fair_value", "low", "high", "price", "missing_reason"])
    members = set(data.members())
    snapshot = _peer_snapshot(as_of, sorted(members | set(tickers)), data)
    reference = snapshot.loc[
        snapshot.index.isin(members)
        & snapshot["missing_reason"].eq("")
        & snapshot["trailing_multiple"].between(0, MAX_PEER_MULTIPLE, inclusive="right")
    ]
    qs = [PE_BAND_Q[0], 0.5, PE_BAND_Q[1]]
    rows = []
    for ticker in tickers:
        row = snapshot.loc[ticker].to_dict()
        row.update({"ticker": ticker, "fair_value": np.nan, "low": np.nan,
                    "high": np.nan, "peer_count": 0, "peer_max_known_date": pd.NaT,
                    "multiple_low": np.nan, "multiple_mid": np.nan, "multiple_high": np.nan})
        if ticker not in members:
            row["missing_reason"] = "not_current_member"
        if not row["missing_reason"]:
            peers = reference.loc[
                reference["sector"].eq(row["sector"]) & (reference.index != ticker)
            ]
            row["peer_count"] = len(peers)
            if len(peers) < MIN_PEERS:
                row["missing_reason"] = "insufficient_sector_peers"
            else:
                bands = peers["trailing_multiple"].quantile(qs)
                row["peer_max_known_date"] = peers["max_known_date"].max()
                row["multiple_low"], row["multiple_mid"], row["multiple_high"] = (float(bands.loc[q]) for q in qs)
                row["low"], row["fair_value"], row["high"] = (float(bands.loc[q] * row["eps_ttm"]) for q in qs)
                estimates = [row["low"], row["fair_value"], row["high"]]
                if not np.isfinite(estimates).all() or min(estimates) <= 0:
                    row["missing_reason"] = "invalid_estimate"
        rows.append(row)
    return pd.DataFrame(rows)


def _ebitda_state(filings):
    """TTM EBITDA and net debt per share from the last 4 known quarters, or the reason they are unusable."""
    latest = filings.tail(4)
    state = {"latest_quarter": pd.NaT, "max_known_date": pd.NaT,
             "ebitda_ttm": np.nan, "net_debt_per_share": np.nan, "missing_reason": ""}
    if not latest.empty:
        state["latest_quarter"] = latest["datadate"].iloc[-1]
        state["max_known_date"] = latest["known_date"].max()
    if len(latest) < 4:
        state["missing_reason"] = "insufficient_dated_quarters"
        return state
    if not np.all(np.diff(latest["quarter"].astype("int64")) == 1):
        state["missing_reason"] = "nonconsecutive_quarters"
        return state
    revenue = pd.to_numeric(latest["revenue_per_share"], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    margin = pd.to_numeric(latest["ebitda_margin"], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(revenue).all() or not np.isfinite(margin).all() or (revenue <= 0).any():
        state["missing_reason"] = "missing_ebitda_inputs"
        return state
    state["ebitda_ttm"] = float(np.dot(revenue, margin))
    if not np.isfinite(state["ebitda_ttm"]) or state["ebitda_ttm"] <= 0:
        state["missing_reason"] = "nonpositive_ttm_ebitda"
        return state
    balance = pd.to_numeric(latest[["interest_debt_per_share", "cash_per_share"]].iloc[-1],
                            errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(balance).all() or (balance < 0).any():
        state["missing_reason"] = "missing_net_debt_inputs"
        return state
    state["net_debt_per_share"] = float(balance[0] - balance[1])
    return state


def _financial_events(filings):
    """The EBITDA state as it was known after each filing date."""
    f = filings.copy()
    f["available_date"] = f[["known_date", "datadate"]].max(axis=1)
    events = []
    for day in sorted(f["available_date"].unique()):
        visible = f.loc[f["available_date"] <= day]
        state = _ebitda_state(visible)
        events.append({"available_date": pd.Timestamp(day), **state})
    return pd.DataFrame(events)


def _ev_observations(as_of, prices, events):
    """Daily past EV/EBITDA: each close before as_of against the filings known on that day."""
    cutoff = pd.Timestamp(as_of).normalize()
    quotes = prices.loc[prices.index < cutoff].tail(HISTORICAL_SESSIONS)
    quotes = quotes.rename("price").rename_axis("price_date").reset_index()
    quotes["price_date"] = quotes["price_date"].astype("datetime64[ns]")
    events = events.copy()
    events["available_date"] = events["available_date"].astype("datetime64[ns]")
    matched = pd.merge_asof(quotes.sort_values("price_date"), events.sort_values("available_date"),
                            left_on="price_date", right_on="available_date", direction="backward")
    age = (matched["price_date"] - matched["latest_quarter"]).dt.days
    enterprise_value = matched["price"] + matched["net_debt_per_share"]
    valid = (matched["missing_reason"].eq("") & age.between(0, MAX_QUARTER_AGE_DAYS)
             & np.isfinite(matched["price"]) & (matched["price"] > 0)
             & np.isfinite(enterprise_value) & (enterprise_value > 0)
             & np.isfinite(matched["ebitda_ttm"]) & (matched["ebitda_ttm"] > 0))
    matched["historical_multiple"] = np.nan
    matched.loc[valid, "historical_multiple"] = (
        enterprise_value.loc[valid] / matched.loc[valid, "ebitda_ttm"]
    )
    return matched


def historical_ev_estimates(as_of, tickers, data) -> pd.DataFrame:
    """EV/EBITDA against the stock's own daily history, per-share inputs, with diagnostics and missing reasons.

    fair_value = median past multiple x current TTM EBITDA - net debt; low / high from EV_BAND_Q quantiles.
    """
    tickers = list(dict.fromkeys(tickers))
    if not tickers:
        return pd.DataFrame(columns=["ticker", "fair_value", "low", "high", "price", "missing_reason"])
    cutoff = pd.Timestamp(as_of).normalize()
    members = set(data.members())
    history = data.close(tickers, lookback=HISTORICAL_SESSIONS + 1)
    prices = data.last_close(tickers)
    filings = _dated_filings(known_filings(data.fundamentals(tickers), cutoff), cutoff)
    groups = dict(tuple(filings.groupby("ticker"))) if not filings.empty else {}
    qs = [EV_BAND_Q[0], 0.5, EV_BAND_Q[1]]
    rows = []
    for ticker in tickers:
        f = groups.get(ticker)
        row = {"ticker": ticker, "fair_value": np.nan, "low": np.nan, "high": np.nan,
               "price": prices.get(ticker, np.nan), "quarter_count": 0 if f is None else len(f),
               "latest_quarter": pd.NaT, "max_known_date": pd.NaT,
               "ebitda_ttm": np.nan, "net_debt_per_share": np.nan,
               "history_observations": 0, "history_start": pd.NaT, "history_end": pd.NaT,
               "multiple_low": np.nan, "multiple_mid": np.nan, "multiple_high": np.nan,
               "missing_reason": ""}
        if ticker not in members:
            row["missing_reason"] = "not_current_member"
        elif f is None or len(f) < MIN_EV_QUARTERS:
            row["missing_reason"] = "insufficient_ev_quarters"
        elif not np.isfinite(row["price"]) or row["price"] <= 0:
            row["missing_reason"] = "missing_price"
        elif not basis_ok(f.iloc[-1], row["price"]) or not basis_stable(
                f, history[ticker] if ticker in history else None):
            # a constant basis gap does not cancel once net debt is nonzero (11/25 review), and a change
            # of basis inside the window would mix bases in the history
            row["missing_reason"] = "basis_mismatch"
        else:
            events = _financial_events(f)
            row.update(events.iloc[-1].drop(labels="available_date").to_dict())
            if not row["missing_reason"] and (cutoff - row["latest_quarter"]).days > MAX_QUARTER_AGE_DAYS:
                row["missing_reason"] = "stale_fundamentals"
            if not row["missing_reason"]:
                observations = _ev_observations(cutoff, history[ticker], events)
                valid = observations.loc[np.isfinite(observations["historical_multiple"])
                                         & (observations["historical_multiple"] > 0)]
                row["history_observations"] = len(valid)
                if len(valid) < MIN_MULTIPLE_OBSERVATIONS:
                    row["missing_reason"] = "insufficient_multiple_history"
                else:
                    row["history_start"], row["history_end"] = valid["price_date"].min(), valid["price_date"].max()
                    bands = valid["historical_multiple"].quantile(qs)
                    row["multiple_low"], row["multiple_mid"], row["multiple_high"] = (float(bands.loc[q]) for q in qs)
                    row["low"], row["fair_value"], row["high"] = (
                        float(bands.loc[q] * row["ebitda_ttm"] - row["net_debt_per_share"]) for q in qs
                    )
                    estimates = [row["low"], row["fair_value"], row["high"]]
                    if not np.isfinite(estimates).all() or min(estimates) <= 0:
                        row["missing_reason"] = "nonpositive_equity_estimate"
        rows.append(row)
    return pd.DataFrame(rows)


def _as_method(estimates: pd.DataFrame) -> pd.DataFrame:
    """Diagnostic estimates -> METHOD_COLUMNS."""
    reason = estimates["missing_reason"].fillna("").astype(str)
    return pd.DataFrame({"ticker": estimates["ticker"], "fair_value": estimates["fair_value"].astype(float),
                         "low": estimates["low"].astype(float), "high": estimates["high"].astype(float),
                         "missing": reason.ne(""), "reason": reason})[METHOD_COLUMNS]


def ev_ebitda_value(as_of, tickers, data) -> pd.DataFrame:
    tickers = list(dict.fromkeys(tickers))
    return _as_method(historical_ev_estimates(as_of, tickers, data)) if tickers else _empty_methods()


def peer_pe_value(as_of, tickers, data) -> pd.DataFrame:
    tickers = list(dict.fromkeys(tickers))
    return _as_method(peer_pe_estimates(as_of, tickers, data)) if tickers else _empty_methods()


# ---- value(): combine the three methods ----
MIN_METHODS = 1   # one valued method is enough, for coverage; method_spread records disagreement
RANGE_COMBINE = "median"   # "median": median of the methods' lows / highs; "envelope": lowest low to highest high
VALUE_METHODS = (("dcf", dcf_value), ("ev", ev_ebitda_value), ("peer_pe", peer_pe_value))


def method_table(as_of, tickers, data) -> pd.DataFrame:
    """Every method's result, long format (method, ticker, fair_value, low, high, missing, reason). For checks and the report."""
    parts = [fn(as_of, tickers, data).assign(method=name) for name, fn in VALUE_METHODS]
    return pd.concat(parts, ignore_index=True)


def combined_table(as_of, tickers, data) -> pd.DataFrame:
    """value()'s COLUMNS plus n_methods (how many methods valued the ticker)."""
    tickers = list(dict.fromkeys(tickers))
    if not tickers:
        return pd.DataFrame(columns=COLUMNS + ["n_methods"])
    m = method_table(as_of, tickers, data)
    m = m[~m["missing"].astype(bool) & (m["fair_value"] > 0)]
    prices = data.last_close(tickers)
    rows = []
    for t, g in m.groupby("ticker"):
        price = prices.get(t)
        if len(g) < MIN_METHODS or price is None or not (np.isfinite(price) and price > 0):
            continue
        fair = float(g["fair_value"].median())
        if RANGE_COMBINE == "envelope":
            low, high = float(g["low"].min()), float(g["high"].max())
        else:
            low, high = float(g["low"].median()), float(g["high"].median())
        rows.append({"ticker": t, "fair_value": fair, "low": min(low, fair), "high": max(high, fair),
                     "mispricing": fair / price - 1,
                     "method_spread": float(g["fair_value"].max() / g["fair_value"].min()),
                     "n_methods": len(g)})
    return pd.DataFrame(rows, columns=COLUMNS + ["n_methods"])


def value(as_of, tickers, data) -> pd.DataFrame:
    """Median of the available methods.

    fair_value = median of the methods' fair values; low / high per RANGE_COMBINE, widened if
    needed so low <= fair_value <= high. method_spread = highest / lowest fair value.
    Tickers with fewer than MIN_METHODS valued methods are left out.
    """
    return combined_table(as_of, tickers, data)[COLUMNS]


# =============================================================================
# filter_candidates
# =============================================================================
# Rules fixed before running them on dev (11/19); the final one is chosen once on val.
#   none     keep everything (group A behaviour)
#   abs      drop if combined mispricing < OVERVALUED
#   agree    drop if mispricing < OVERVALUED and >= 2 methods valued it and they agree within MAX_SPREAD
#   worst_k  drop the DROP_WORST_K most overvalued of the current top 10 (only names with mispricing < 0)
# Second round (registered 11/21, six literature variants of worst_k, experiments/variants.py):
#   worst_k_rates  worst_k, but only while the 10-year Treasury yield is above its level
#                  RATE_LOOKBACK_DAYS earlier; otherwise nothing is dropped. Growth stocks are
#                  long-duration assets, hurt most when discount rates rise (Dechow, Sloan & Soliman 2004).
# A stock with no valuation is always kept.
FILTER_RULES = ("none", "abs", "agree", "worst_k", "worst_k_rates", "cheap_k")
FILTER_RULE = "worst_k_rates"  # 11/19: worst_k won on val. 11/21: worst_k_rates was the only variant to pass
                               # the dev screen and beat worst_k's val Calmar (0.002 vs -0.003, a noise-sized margin)
OVERVALUED = -0.50
MAX_SPREAD = 2.0
DROP_WORST_K = 2
FILTER_DEPTH = 15          # value the top 15 growth_tech names: the 10 held plus room for backfills
RATE_LOOKBACK_DAYS = 182   # "rates rising" = DGS10 today above DGS10 about six months earlier
_VALUE_CACHE: dict = {}    # (ticker, year, month) -> (as_of computed on, combined row or None)


def _monthly_values(as_of, tickers, data) -> pd.DataFrame:
    """mispricing / method_spread / n_methods per ticker; fair values cached per month for speed.

    A cached value is reused only when it was computed on or before this as_of (never later data).
    Mispricing always uses today's price.
    """
    ym = (as_of.year, as_of.month)
    need = [t for t in tickers if (t, *ym) not in _VALUE_CACHE or _VALUE_CACHE[(t, *ym)][0] > as_of]
    if need:
        tab = combined_table(as_of, need, data).set_index("ticker")
        for t in need:
            _VALUE_CACHE[(t, *ym)] = (as_of, tab.loc[t].to_dict() if t in tab.index else None)
    prices = data.last_close(tickers)
    rows = []
    for t in tickers:
        v, p = _VALUE_CACHE[(t, *ym)][1], prices.get(t)
        if v is None or p is None or not (np.isfinite(p) and p > 0):
            continue
        rows.append({"ticker": t, "mispricing": v["fair_value"] / p - 1,
                     "method_spread": v["method_spread"], "n_methods": v["n_methods"]})
    out = pd.DataFrame(rows, columns=["ticker", "mispricing", "method_spread", "n_methods"]).set_index("ticker")
    return out.astype(float)  # an empty week must still compare and sort as numbers


def rates_rising(as_of, data) -> bool:
    """DGS10 on as_of above its last value on or before as_of - RATE_LOOKBACK_DAYS."""
    r = data.macro("DGS10").dropna()
    past = r.loc[: pd.Timestamp(as_of) - pd.Timedelta(days=RATE_LOOKBACK_DAYS)]
    return bool(len(r) and len(past) and r.iloc[-1] > past.iloc[-1])


def filter_candidates(as_of, candidates, data):
    """Remove growth_tech names the valuation calls clearly overvalued, per FILTER_RULE."""
    if FILTER_RULE == "none":
        return candidates
    as_of = pd.Timestamp(as_of)
    members = data.members()
    growth = sorted((c for c in candidates if c.bucket == BUCKET and c.symbol in members),
                    key=lambda c: c.rank_score, reverse=True)
    ranked = [c.symbol for c in growth[:FILTER_DEPTH]]
    if not ranked:
        return candidates
    m = _monthly_values(as_of, ranked, data)
    if FILTER_RULE == "abs":
        drop = set(m.index[m["mispricing"] < OVERVALUED])
    elif FILTER_RULE == "agree":
        drop = set(m.index[(m["mispricing"] < OVERVALUED) & (m["n_methods"] >= 2)
                           & (m["method_spread"] <= MAX_SPREAD)])
    elif FILTER_RULE in ("worst_k", "worst_k_rates"):
        top = m["mispricing"].reindex(ranked[:10]).dropna()
        drop = set(top[top < 0].nsmallest(DROP_WORST_K).index)
        if FILTER_RULE == "worst_k_rates" and not rates_rising(as_of, data):
            drop = set()
    elif FILTER_RULE == "cheap_k":
        # negative rank IC: in this universe the cheapest-looking names do worst (registered 11/25)
        top = m["mispricing"].reindex(ranked[:10]).dropna()
        drop = set(top[top > 0].nlargest(DROP_WORST_K).index)
    else:
        raise ValueError(f"FILTER_RULE must be one of {FILTER_RULES}")
    return tuple(c for c in candidates if c.symbol not in drop)
