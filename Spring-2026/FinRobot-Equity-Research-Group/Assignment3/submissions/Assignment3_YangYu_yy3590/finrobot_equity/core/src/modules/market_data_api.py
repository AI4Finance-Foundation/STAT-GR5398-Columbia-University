#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import datetime
import os
import json
import hashlib
from pathlib import Path
from typing import Any

try:
    import yfinance as yf
except Exception:  # pragma: no cover - optional dependency
    yf = None

try:
    import requests
except Exception:  # pragma: no cover - optional dependency
    requests = None


class _RequestExceptionBase(Exception):
    pass


if requests is None:
    class _RequestsExceptions:
        RequestException = _RequestExceptionBase

    class _RequestsFallback:
        exceptions = _RequestsExceptions()

        @staticmethod
        def get(*args, **kwargs):
            raise ImportError("requests is not installed")

    requests = _RequestsFallback()

# Assuming common_utils.py is in the same parent directory (src/modules)
from .common_utils import get_api_key, load_config 

fmp_base_url = "https://financialmodelingprep.com/stable"

_FMP_RUNTIME = {
    "force_refresh": False,
    "allow_stale_fallback": True,
    "cache_dir": None,
    "cache_policy": "same-day",
    "events": [],
}


def _utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _today_local_str() -> str:
    return datetime.datetime.now().astimezone().strftime("%Y-%m-%d")


def _default_cache_dir() -> Path:
    # modules -> src -> core -> finrobot_equity -> FinRobot
    try:
        project_root = Path(__file__).resolve().parents[4]
    except Exception:
        project_root = Path.cwd()
    return project_root / "output" / "cache" / "fmp"


def configure_fmp_runtime(
    force_refresh: bool = False,
    cache_dir: str | None = None,
    allow_stale_fallback: bool = True,
    cache_policy: str = "same-day",
    clear_events: bool = True,
) -> None:
    if cache_policy not in {"same-day", "reuse-until-forced"}:
        raise ValueError("cache_policy must be 'same-day' or 'reuse-until-forced'")
    _FMP_RUNTIME["force_refresh"] = bool(force_refresh)
    _FMP_RUNTIME["allow_stale_fallback"] = bool(allow_stale_fallback)
    _FMP_RUNTIME["cache_dir"] = cache_dir
    _FMP_RUNTIME["cache_policy"] = cache_policy
    if clear_events:
        _FMP_RUNTIME["events"] = []


def get_fmp_event_cursor() -> int:
    return len(_FMP_RUNTIME["events"])


def _normalized_params(params: dict[str, Any] | None) -> dict[str, Any]:
    if not params:
        return {}
    normalized: dict[str, Any] = {}
    for key in sorted(params.keys()):
        val = params[key]
        if isinstance(val, (str, int, float, bool)) or val is None:
            normalized[str(key)] = val
        else:
            normalized[str(key)] = str(val)
    return normalized


def _cache_root_path() -> Path:
    configured = _FMP_RUNTIME.get("cache_dir")
    root = Path(configured) if configured else _default_cache_dir()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _cache_file_path(endpoint: str, ticker: str, params: dict[str, Any]) -> Path:
    endpoint_key = endpoint.strip("/").replace("/", "__")
    cache_key_payload = {
        "endpoint": endpoint_key,
        "ticker": ticker.upper(),
        "params": _normalized_params(params),
    }
    digest = hashlib.sha256(
        json.dumps(cache_key_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()[:20]
    safe_ticker = ticker.upper().replace("/", "_")
    return _cache_root_path() / endpoint_key / f"{safe_ticker}_{digest}.json"


def _read_cache(cache_path: Path) -> dict | None:
    if not cache_path.exists():
        return None
    try:
        with cache_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict) and "data" in payload and "meta" in payload:
            return payload
    except Exception:
        return None
    return None


def _write_cache(cache_path: Path, payload: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _record_fmp_event(
    endpoint: str,
    ticker: str,
    origin: str,
    cache_path: Path,
    params: dict[str, Any],
    detail: str = "",
) -> None:
    _FMP_RUNTIME["events"].append(
        {
            "timestamp_utc": _utc_now_iso(),
            "endpoint": endpoint.strip("/"),
            "ticker": ticker.upper(),
            "origin": origin,
            "cache_path": str(cache_path),
            "cache_date_local": _today_local_str(),
            "params": _normalized_params(params),
            "cache_policy": _FMP_RUNTIME.get("cache_policy", "same-day"),
            "force_refresh": bool(_FMP_RUNTIME.get("force_refresh")),
            "detail": detail,
        }
    )


def _is_same_day_cache(payload: dict) -> bool:
    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    cache_date_local = meta.get("cache_date_local")
    if cache_date_local:
        return str(cache_date_local) == _today_local_str()
    return False


def _as_of_timestamp(as_of_date: str | datetime.datetime | pd.Timestamp | None) -> pd.Timestamp | None:
    if as_of_date is None:
        return None
    ts = pd.to_datetime(as_of_date, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid as_of_date: {as_of_date}")
    return pd.Timestamp(ts).tz_localize(None).normalize()


def _apply_as_of_filter(
    df: pd.DataFrame | None,
    as_of_date: str | datetime.datetime | pd.Timestamp | None,
    *,
    date_col: str = "date",
    period: str = "annual",
    publication_lag_days: int | None = None,
) -> pd.DataFrame | None:
    """Filter a statement/news frame to only records visible as of a given date."""
    if df is None or df.empty:
        return df

    cutoff = _as_of_timestamp(as_of_date)
    if cutoff is None:
        return df

    result = df.copy()
    if date_col not in result.columns:
        return result

    period_lag = publication_lag_days
    if period_lag is None:
        period_lag = 90 if str(period).lower() == "annual" else 45 if str(period).lower() == "quarterly" else 0

    statement_date = pd.to_datetime(result[date_col], errors="coerce")
    available_date = statement_date + pd.to_timedelta(period_lag, unit="D")

    # If the source already includes a filing date, use it as an additional
    # conservative visibility gate.
    for filing_col in ("filingDate", "fillingDate", "acceptedDate", "publishedDate"):
        if filing_col in result.columns:
            filing_date = pd.to_datetime(result[filing_col], errors="coerce")
            available_date = pd.concat([available_date, filing_date], axis=1).max(axis=1)

    mask = available_date <= cutoff
    result = result.loc[mask].copy()
    if result.empty:
        return result

    result[date_col] = pd.to_datetime(result[date_col], errors="coerce")
    result = result.sort_values(date_col, ascending=False).reset_index(drop=True)
    return result


def summarize_fmp_events(since_cursor: int = 0) -> dict:
    events = _FMP_RUNTIME["events"][since_cursor:]
    counts = {
        "fresh_fetch": 0,
        "cache_hit": 0,
        "fallback_stale_cache": 0,
        "no_data": 0,
    }
    endpoints: dict[str, int] = {}
    for item in events:
        origin = item.get("origin", "no_data")
        if origin in counts:
            counts[origin] += 1
        else:
            counts["no_data"] += 1
        endpoint = item.get("endpoint", "unknown")
        endpoints[endpoint] = endpoints.get(endpoint, 0) + 1

    non_zero_origins = [k for k, v in counts.items() if v > 0 and k != "no_data"]
    if not events:
        primary_origin = "no_data"
    elif len(non_zero_origins) == 1:
        primary_origin = non_zero_origins[0]
    elif len(non_zero_origins) == 0:
        primary_origin = "no_data"
    else:
        primary_origin = "mixed"

    return {
        "total_calls": len(events),
        "primary_origin": primary_origin,
        "origin_counts": counts,
        "endpoint_counts": endpoints,
        "events": events,
    }


def _fmp_get_json(
    endpoint: str,
    ticker: str,
    api_key: str,
    params: dict[str, Any] | None = None,
    timeout: float | None = None,
) -> tuple[Any, dict]:
    base_params = dict(params or {})
    query = dict(base_params)
    query["apikey"] = api_key
    cache_path = _cache_file_path(endpoint=endpoint, ticker=ticker, params=base_params)
    cached_payload = _read_cache(cache_path)

    if not _FMP_RUNTIME["force_refresh"] and cached_payload:
        cache_policy = _FMP_RUNTIME.get("cache_policy", "same-day")
        can_reuse = cache_policy == "reuse-until-forced" or _is_same_day_cache(cached_payload)
        if can_reuse:
            cache_date = cached_payload.get("meta", {}).get("cache_date_local", "unknown")
            detail_prefix = "exact_key_cache" if cache_policy == "reuse-until-forced" else "same_day_cache"
            meta = dict(cached_payload.get("meta", {}))
            meta["origin"] = "cache_hit"
            meta["cache_policy"] = cache_policy
            _record_fmp_event(
                endpoint=endpoint,
                ticker=ticker,
                origin="cache_hit",
                cache_path=cache_path,
                params=base_params,
                detail=f"{detail_prefix}:{cache_date}",
            )
            return cached_payload.get("data"), meta

    url = f"{fmp_base_url}/{endpoint.strip('/')}"
    try:
        if timeout is not None:
            response = requests.get(url, params=query, timeout=timeout)
        else:
            response = requests.get(url, params=query)
        response.raise_for_status()
        data = response.json()
        payload = {
            "meta": {
                "endpoint": endpoint.strip("/"),
                "ticker": ticker.upper(),
                "params": _normalized_params(base_params),
                "fetched_at_utc": _utc_now_iso(),
                "cache_date_local": _today_local_str(),
                "origin": "fresh_fetch",
                "cache_policy": _FMP_RUNTIME.get("cache_policy", "same-day"),
            },
            "data": data,
        }
        _write_cache(cache_path, payload)
        _record_fmp_event(
            endpoint=endpoint,
            ticker=ticker,
            origin="fresh_fetch",
            cache_path=cache_path,
            params=base_params,
            detail="network_fetch",
        )
        return data, payload["meta"]
    except Exception as e:
        if cached_payload and _FMP_RUNTIME["allow_stale_fallback"]:
            cache_date = cached_payload.get("meta", {}).get("cache_date_local", "unknown")
            _record_fmp_event(
                endpoint=endpoint,
                ticker=ticker,
                origin="fallback_stale_cache",
                cache_path=cache_path,
                params=base_params,
                detail=f"cache_date={cache_date}; {type(e).__name__}: {e}",
            )
            stale_meta = dict(cached_payload.get("meta", {}))
            stale_meta["fallback_reason"] = f"{type(e).__name__}: {e}"
            stale_meta["origin"] = "fallback_stale_cache"
            stale_meta["cache_policy"] = _FMP_RUNTIME.get("cache_policy", "same-day")
            return cached_payload.get("data"), stale_meta
        _record_fmp_event(
            endpoint=endpoint,
            ticker=ticker,
            origin="no_data",
            cache_path=cache_path,
            params=base_params,
            detail=f"{type(e).__name__}: {e}",
        )
        raise

def fetch_yfinance_volume(ticker: str, start_date: str, end_date: str) -> pd.DataFrame | None:
    """Fetches historical trading volume data using yfinance."""
    if yf is None:
        print("yfinance is not installed; volume fetch is unavailable.")
        return None
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if stock_data.empty:
            print(f"No data returned from yfinance for {ticker} between {start_date} and {end_date}")
            return None
        stock_data = stock_data[["Volume"]]
        stock_data.reset_index(inplace=True)
        stock_data["Date"] = pd.to_datetime(stock_data["Date"])
        return stock_data
    except Exception as e:
        print(f"Error fetching yfinance volume for {ticker}: {e}")
        return None

def fetch_fmp_enterprise_value(ticker: str, api_key: str, limit: int = 2000) -> pd.DataFrame | None:
    """Fetches historical enterprise value from Financial Modeling Prep API."""
    try:
        data, _ = _fmp_get_json(
            endpoint="enterprise-value",
            ticker=ticker,
            api_key=api_key,
            params={"symbol": ticker, "limit": limit},
        )
        if not data:
            print(f"No EV data returned from FMP for {ticker}.")
            return None
        
        df = pd.DataFrame(data)
        if "date" not in df.columns or "enterpriseValue" not in df.columns:
            print(f"FMP EV data for {ticker} does not contain expected columns ('date', 'enterpriseValue'). Response: {data}")
            return None
            
        df["date"] = pd.to_datetime(df["date"])
        df = df[["date", "enterpriseValue"]].sort_values(by="date").reset_index(drop=True)
        return df
    except requests.exceptions.RequestException as e:
        print(f"Error fetching FMP EV for {ticker}: {e}")
        return None
    except (KeyError, ValueError) as e:
        print(f"Error processing FMP EV data for {ticker}: {e}. Response: {data if 'data' in locals() else 'N/A'}")
        return None

def get_fmp_ratios_and_key_metrics(
    ticker: str,
    api_key: str,
    period: str = "annual",
    limit: int = 5,
    as_of_date: str | datetime.datetime | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Fetches financial ratios and key metrics from FMP API."""
    ratios_df, key_metrics_df = None, None
    try:
        # Ratios
        ratios_data, _ = _fmp_get_json(
            endpoint="ratios",
            ticker=ticker,
            api_key=api_key,
            params={"symbol": ticker, "period": period, "limit": limit},
        )
        if ratios_data:
            ratios_df = pd.DataFrame(ratios_data)
            ratios_df["date"] = pd.to_datetime(ratios_df["date"])
            ratios_df["year"] = ratios_df["date"].dt.year
            ratios_df = _apply_as_of_filter(ratios_df, as_of_date, period=period)

        # Key Metrics
        key_metrics_data, _ = _fmp_get_json(
            endpoint="key-metrics",
            ticker=ticker,
            api_key=api_key,
            params={"symbol": ticker, "period": period, "limit": limit},
        )
        if key_metrics_data:
            key_metrics_df = pd.DataFrame(key_metrics_data)
            key_metrics_df["date"] = pd.to_datetime(key_metrics_df["date"])
            key_metrics_df["year"] = key_metrics_df["date"].dt.year
            key_metrics_df = _apply_as_of_filter(key_metrics_df, as_of_date, period=period)

    except requests.exceptions.RequestException as e:
        print(f"Error fetching FMP ratios/key metrics for {ticker}: {e}")
    except (KeyError, ValueError) as e:
        print(f"Error processing FMP ratios/key metrics data for {ticker}: {e}")
        
    return ratios_df, key_metrics_df

def get_fmp_income_statement(
    ticker: str,
    api_key: str,
    period: str = "annual",
    limit: int = 5,
    as_of_date: str | datetime.datetime | pd.Timestamp | None = None,
) -> pd.DataFrame | None:
    """Fetches income statement data from FMP API."""
    try:
        data, _ = _fmp_get_json(
            endpoint="income-statement",
            ticker=ticker,
            api_key=api_key,
            params={"symbol": ticker, "period": period, "limit": limit},
        )
        if not data:
            print(f"No income statement data from FMP for {ticker}.")
            return None
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df["year"] = df["date"].dt.year
        df = _apply_as_of_filter(df, as_of_date, period=period)
        return df
    except requests.exceptions.RequestException as e:
        print(f"Error fetching FMP income statement for {ticker}: {e}")
        return None
    except (KeyError, ValueError) as e:
        print(f"Error processing FMP income statement data for {ticker}: {e}")
        return None

def get_fmp_balance_sheet(
    ticker: str,
    api_key: str,
    period: str = "annual",
    limit: int = 5,
    as_of_date: str | datetime.datetime | pd.Timestamp | None = None,
) -> pd.DataFrame | None:
    """Fetches balance sheet data from FMP API."""
    try:
        data, _ = _fmp_get_json(
            endpoint="balance-sheet-statement",
            ticker=ticker,
            api_key=api_key,
            params={"symbol": ticker, "period": period, "limit": limit},
        )
        if not data:
            print(f"No balance sheet data from FMP for {ticker}.")
            return None
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df["year"] = df["date"].dt.year
        df = _apply_as_of_filter(df, as_of_date, period=period)
        return df
    except requests.exceptions.RequestException as e:
        print(f"Error fetching FMP balance sheet for {ticker}: {e}")
        return None
    except (KeyError, ValueError) as e:
        print(f"Error processing FMP balance sheet data for {ticker}: {e}")
        return None

def get_fmp_cash_flow_statement(
    ticker: str,
    api_key: str,
    period: str = "annual",
    limit: int = 5,
    as_of_date: str | datetime.datetime | pd.Timestamp | None = None,
) -> pd.DataFrame | None:
    """Fetches cash flow statement data from FMP API."""
    try:
        data, _ = _fmp_get_json(
            endpoint="cash-flow-statement",
            ticker=ticker,
            api_key=api_key,
            params={"symbol": ticker, "period": period, "limit": limit},
        )
        if not data:
            print(f"No cash flow data from FMP for {ticker}.")
            return None
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df["year"] = df["date"].dt.year
        df = _apply_as_of_filter(df, as_of_date, period=period)
        return df
    except requests.exceptions.RequestException as e:
        print(f"Error fetching FMP cash flow for {ticker}: {e}")
        return None
    except (KeyError, ValueError) as e:
        print(f"Error processing FMP cash flow data for {ticker}: {e}")
        return None

def get_comprehensive_financial_data(
    ticker: str,
    api_key: str,
    period: str = "annual",
    limit: int = 5,
    as_of_date: str | datetime.datetime | pd.Timestamp | None = None,
) -> dict:
    """Fetches all three financial statements for a company."""
    print(f"Fetching comprehensive financial data for {ticker}...")
    
    financial_data = {
        'income_statement': get_fmp_income_statement(ticker, api_key, period, limit, as_of_date=as_of_date),
        'balance_sheet': get_fmp_balance_sheet(ticker, api_key, period, limit, as_of_date=as_of_date),
        'cash_flow': get_fmp_cash_flow_statement(ticker, api_key, period, limit, as_of_date=as_of_date),
        'ratios': None,
        'key_metrics': None
    }
    
    # Also get ratios and key metrics
    ratios_df, key_metrics_df = get_fmp_ratios_and_key_metrics(ticker, api_key, period, limit, as_of_date=as_of_date)
    financial_data['ratios'] = ratios_df
    financial_data['key_metrics'] = key_metrics_df
    
    return financial_data

def combine_peer_financial_data(
    tickers: list[str],
    api_key: str,
    years_limit: int = 5,
    as_of_date: str | datetime.datetime | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combines EBITDA and EV/EBITDA for a list of peer tickers."""
    all_peers_data = {}
    for ticker in tickers:
        income_df = get_fmp_income_statement(ticker, api_key, limit=years_limit, as_of_date=as_of_date)
        _, key_metrics_df = get_fmp_ratios_and_key_metrics(ticker, api_key, limit=years_limit, as_of_date=as_of_date)
        
        ticker_data = {}
        expected_ebitda = "ebitda"
        if income_df is not None and not income_df.empty:
            if expected_ebitda not in income_df.columns:
                print(f"Warning: EBITDA data for {ticker} is missing expected columns")
            else:
                for _, row in income_df.iterrows():
                    year = row["year"]
                    val = row.get(expected_ebitda)
                    if pd.notna(val):
                        ticker_data.setdefault(year,{})["EBITDA"] = val

        expected_evebitda = "evToEBITDA"
        if key_metrics_df is not None and not key_metrics_df.empty:
            if expected_evebitda not in key_metrics_df.columns:
                print(f"Warning: EV/EBITDA data for {ticker} is missing expected columns")
            else:
                for _, row in key_metrics_df.iterrows():
                    year = row["year"]
                    val = row.get(expected_evebitda)
                    if pd.notna(val):
                        ticker_data.setdefault(year,{})["EV/EBITDA"] = val
        
        if ticker_data:
            all_peers_data[ticker] = ticker_data

    ebitda_records = []
    for ticker, yearly_data in all_peers_data.items():
        for year, metrics in yearly_data.items():
            if "EBITDA" in metrics and metrics["EBITDA"] is not None:
                ebitda_records.append({"ticker": ticker, "year": year, "EBITDA": metrics["EBITDA"]})
    df_ebitda_all = pd.DataFrame(ebitda_records)
    df_ebitda_pivot = pd.DataFrame()
    if not df_ebitda_all.empty:
        df_ebitda_pivot = df_ebitda_all.pivot(index="year", columns="ticker", values="EBITDA").sort_index()

    ev_ebitda_records = []
    for ticker, yearly_data in all_peers_data.items():
        for year, metrics in yearly_data.items():
            if "EV/EBITDA" in metrics and metrics["EV/EBITDA"] is not None:
                ev_ebitda_records.append({"ticker": ticker, "year": year, "EV/EBITDA": metrics["EV/EBITDA"]})
    df_ev_ebitda_all = pd.DataFrame(ev_ebitda_records)
    df_ev_ebitda_pivot = pd.DataFrame()
    if not df_ev_ebitda_all.empty:
        df_ev_ebitda_pivot = df_ev_ebitda_all.pivot(index="year", columns="ticker", values="EV/EBITDA").sort_index()
        
    return df_ebitda_pivot, df_ev_ebitda_pivot

def project_ebitda_for_peers(df_ebitda_historical: pd.DataFrame, num_projection_years: int = 1) -> pd.DataFrame:
    """Projects EBITDA for future years based on average historical YoY growth."""
    df_projected = df_ebitda_historical.copy()
    if df_projected.empty:
        return df_projected

    last_historical_year = df_projected.index.max()
    
    for company in df_projected.columns:
        historical_values = df_projected[company].dropna()
        if len(historical_values) < 2:
            print(f"Not enough historical EBITDA data for {company} to project.")
            continue
        
        growth_rates = historical_values.pct_change().dropna()
        if growth_rates.empty or all(g == 0 for g in growth_rates):
            avg_growth_rate = 0 
        else:
            avg_growth_rate = growth_rates.mean()

        current_ebitda = historical_values.iloc[-1]
        for i in range(1, num_projection_years + 1):
            projection_year = last_historical_year + i
            current_ebitda = current_ebitda * (1 + avg_growth_rate)
            df_projected.loc[projection_year, company] = current_ebitda
            
    return df_projected.sort_index()

def get_fmp_current_price(ticker: str, api_key: str) -> float | None:
    """Fetches the latest stock price from Financial Modeling Prep API."""
    try:
        data, _ = _fmp_get_json(
            endpoint="quote-short",
            ticker=ticker,
            api_key=api_key,
            params={"symbol": ticker},
        )
        if data and isinstance(data, list) and len(data) > 0:
            quote_data = data[0]
            if "price" in quote_data and quote_data["price"] is not None:
                return float(quote_data["price"])
            else:
                data_full, _ = _fmp_get_json(
                    endpoint="quote",
                    ticker=ticker,
                    api_key=api_key,
                    params={"symbol": ticker},
                )
                if data_full and isinstance(data_full, list) and len(data_full) > 0:
                    quote_data_full = data_full[0]
                    if "price" in quote_data_full and quote_data_full["price"] is not None:
                        return float(quote_data_full["price"])
                    elif "previousClose" in quote_data_full and quote_data_full["previousClose"] is not None:
                        print(f"Current price not available for {ticker} via /quote or /quote-short, using previous close: {quote_data_full['previousClose']}")
                        return float(quote_data_full["previousClose"])
                print(f"Price data not found in FMP quote for {ticker}. Response: {quote_data}")
                return None
        else:
            print(f"No data or unexpected format returned from FMP /quote-short for {ticker}. Response: {data}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching FMP current price for {ticker}: {e}")
        return None
    except (KeyError, ValueError, TypeError) as e:
        print(f"Error processing FMP current price data for {ticker}: {e}. Response: {data if 'data' in locals() else 'N/A'}")
        return None

def get_analyst_insights(ticker: str, api_key: str = None) -> tuple[str | None, float | None]:
    """
    Fetches analyst rating and target price using FMP API.
    
    Args:
        ticker: Stock ticker symbol
        api_key: FMP API key (optional, will try to use existing functions if not provided)
    
    Returns:
        Tuple of (rating, target_price)
    """
    rating = None
    target_price = None
    
    try:
        # Try to get rating from FMP
        if api_key:
            rating = get_fmp_analyst_rating(ticker, api_key)
            target_price = get_fmp_target_price(ticker, api_key)
            
            if rating:
                print(f"[INFO] For {ticker} - Rating: {rating}")
            if target_price:
                print(f"[INFO] For {ticker} - Target Price: {target_price}")
        else:
            print(f"[WARN] No API key provided for get_analyst_insights. Skipping.")
            
    except Exception as e:
        print(f"[ERROR] Error in get_analyst_insights for {ticker}: {e}")
        
    return rating, target_price

def get_fmp_target_price(ticker: str, api_key: str) -> float | None:
    """Fetches the latest analyst target price from Financial Modeling Prep API."""
    try:
        data, _ = _fmp_get_json(
            endpoint="price-target-summary",
            ticker=ticker,
            api_key=api_key,
            params={"symbol": ticker},
        )
        payload_priority = [
            "lastQuarterAvgPriceTarget",
            "lastMonthAvgPriceTarget",
            "lastYearAvgPriceTarget",
            "allTimeAvgPriceTarget"
        ]

        if data and isinstance(data, list) and len(data) > 0:
            # Legacy Parsing
            if data[0].get('publishedDate') and data[0].get('priceTarget'):
                for item in data:
                    try:
                        item['parsedDate'] = pd.to_datetime(item.get('publishedDate'))
                    except Exception as e:
                        print(f"Warning: Could not parse publishedDate '{item.get('publishedDate')}' for {ticker}: {e}")
                        item['parsedDate'] = None
                valid_data = [item for item in data if item['parsedDate'] is not None]
                if not valid_data:
                    print(f"No valid published dates found in FMP target price data for {ticker}.")
                    return None
                latest_target_info = sorted(valid_data, key=lambda x: x['parsedDate'], reverse=True)[0]
                if "priceTarget" in latest_target_info and latest_target_info["priceTarget"] is not None:
                    return float(latest_target_info["priceTarget"])
                else:
                    print(f"Price target not found in the latest FMP data for {ticker}. Data: {latest_target_info}")
                    return None
            # New Parsing
            else:
                return next((float(data[0].get(v)) for v in payload_priority if data[0].get(v)), None)
                
        else:
            print(f"No target price data or unexpected format returned from FMP for {ticker}. Response: {data}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching FMP target price for {ticker}: {e}")
        return None
    except (KeyError, ValueError, TypeError) as e:
        print(f"Error processing FMP target price data for {ticker}: {e}. Response: {data if 'data' in locals() else 'N/A'}")
        return None

def get_fmp_analyst_rating(ticker: str, api_key: str) -> str | None:
    """Fetches the latest analyst rating (e.g., Buy, Hold, Sell) from Financial Modeling Prep API."""
    try:
        data, _ = _fmp_get_json(
            endpoint="grades",
            ticker=ticker,
            api_key=api_key,
            params={"symbol": ticker},
        )
        
        if data and isinstance(data, list) and len(data) > 0:
            for item in data:
                try:
                    d = pd.to_datetime(item.get('date'), errors='coerce')
                    if pd.isna(d):
                        d = pd.to_datetime(item.get('publishedDate'), errors='coerce')
                    item['parsedDate'] = d
                except Exception as e:
                    print(f"Warning: Could not parse rating date for {ticker}: {e}")
                    item['parsedDate'] = None
            valid_data = [item for item in data if pd.notna(item['parsedDate'])]
            if not valid_data:
                print(f"No valid dates found in FMP analyst rating data for {ticker}.")
                return None
            latest_rating_info = sorted(valid_data, key=lambda x: x['parsedDate'], reverse=True)[0]
            if "newGrade" in latest_rating_info and latest_rating_info["newGrade"]:
                return str(latest_rating_info["newGrade"])
            elif "action" in latest_rating_info and latest_rating_info["action"]:
                 print(f"newGrade not found, using action: {latest_rating_info['action']} for {ticker}")
                 return str(latest_rating_info["action"])
            else:
                print(f"Analyst rating (newGrade or action) not found in the latest FMP data for {ticker}. Data: {latest_rating_info}")
                return None
                    
        else:
            print(f"No analyst rating data or unexpected format returned from FMP for {ticker}. Response: {data}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching FMP analyst rating for {ticker}: {e}")
        return None
    except (KeyError, ValueError, TypeError, AttributeError) as e:
        print(f"Error processing FMP analyst rating data for {ticker}: {e}. Response: {data if 'data' in locals() else 'N/A'}")
        return None

def get_fmp_company_profile(ticker: str, api_key: str) -> dict | None:
    """Fetches comprehensive company profile data from FMP API."""
    try:
        data, _ = _fmp_get_json(
            endpoint="profile",
            ticker=ticker,
            api_key=api_key,
            params={"symbol": ticker},
        )
        if data and isinstance(data, list) and len(data) > 0:
            return data[0]  # Profile returns a list with one item
        else:
            print(f"No profile data returned for {ticker}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching company profile for {ticker}: {e}")
        return None
    except (KeyError, ValueError, TypeError) as e:
        print(f"Error processing company profile data for {ticker}: {e}")
        return None

def get_fmp_market_cap(ticker: str, api_key: str) -> float | None:
    """Fetches current market capitalization from FMP API."""
    try:
        data, _ = _fmp_get_json(
            endpoint="market-capitalization",
            ticker=ticker,
            api_key=api_key,
            params={"symbol": ticker},
        )
        if data and isinstance(data, list) and len(data) > 0:
            market_cap = data[0].get('marketCap')
            return float(market_cap) if market_cap else None
        else:
            print(f"No market cap data returned for {ticker}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching market cap for {ticker}: {e}")
        return None
    except (KeyError, ValueError, TypeError) as e:
        print(f"Error processing market cap data for {ticker}: {e}")
        return None

def get_comprehensive_company_metrics(ticker: str, api_key: str) -> dict:
    """Fetches all key company metrics needed for equity report from FMP API."""
    print(f"Fetching comprehensive company metrics for {ticker}...")
    event_cursor = get_fmp_event_cursor()
    
    metrics = {
        'share_price': None,
        'target_price': None,
        'market_cap': None,
        'volume': None,
        'fwd_pe': None,
        'pb_ratio': None,
        'dividend_yield': None,
        'free_float': None,
        'roe': None,
        'net_debt_to_equity': None,
        'rating': None,
        'beta': None,
        'sector': None,
        'industry': None,
        'exchange': None,
        '52w_range': None,
        'shares_outstanding': None,
        '_fmp_cache': {},
    }
    
    # 1. Get current price and basic quote data
    current_price = get_fmp_current_price(ticker, api_key)
    if current_price:
        metrics['share_price'] = current_price
    
    # 2. Get target price and rating
    target_price = get_fmp_target_price(ticker, api_key)
    if target_price:
        metrics['target_price'] = target_price
    
    rating = get_fmp_analyst_rating(ticker, api_key)
    if rating:
        metrics['rating'] = rating
    
    # 3. Get company profile data
    profile = get_fmp_company_profile(ticker, api_key)
    if profile:
        if profile.get('mktCap'):
            metrics['market_cap'] = profile.get('mktCap', 0) / 1e9 # Convert to billions
        elif profile.get('marketCap'):
            metrics['market_cap'] = profile.get('marketCap', 0) / 1e9
        else:
            None
        if profile.get('volAvg'):
            metrics['volume'] = profile.get('volAvg', 0) / 1e6 # Convert to millions
        elif profile.get('averageVolume'):
            metrics['volume'] = profile.get('averageVolume', 0) / 1e6
        else:
            None
        metrics['beta'] = profile.get('beta')
        metrics['sector'] = profile.get('sector', 'N/A')
        metrics['industry'] = profile.get('industry', 'N/A')
        metrics['exchange'] = profile.get('exchangeShortName', 'N/A')
    
    # 4. Get detailed quote data (volume, 52w range, shares outstanding)
    try:
        quote_data, _ = _fmp_get_json(
            endpoint="quote",
            ticker=ticker,
            api_key=api_key,
            params={"symbol": ticker},
        )
        if quote_data and isinstance(quote_data, list) and len(quote_data) > 0:
            quote = quote_data[0]
            # Volume
            if not metrics['volume']:
                avg_volume = quote.get('avgVolume')
                if avg_volume:
                    metrics['volume'] = avg_volume / 1e6  # Convert to millions
            # 52-week range
            year_high = quote.get('yearHigh')
            year_low = quote.get('yearLow')
            if year_high is not None and year_low is not None:
                metrics['52w_range'] = f"${year_low:.2f} - ${year_high:.2f}"
            # Shares outstanding
            shares_out = quote.get('sharesOutstanding')
            if shares_out:
                metrics['shares_outstanding'] = float(shares_out)
    except Exception as e:
        print(f"Warning: Could not fetch quote data: {e}")
    
    # 5. Get financial ratios
    try:
        ratios_df, key_metrics_df = get_fmp_ratios_and_key_metrics(ticker, api_key, limit=1)
        
        if ratios_df is not None and not ratios_df.empty:
            latest_ratios = ratios_df.iloc[0]
            metrics['pb_ratio'] = latest_ratios.get('priceToBookRatio')
            metrics['roe'] = latest_ratios.get('returnOnEquity')
            if latest_ratios.get('returnOnEquity'):
                metrics['roe'] = latest_ratios['returnOnEquity'] * 100  # Convert to percentage
            metrics['net_debt_to_equity'] = latest_ratios.get('debtEquityRatio')
            # Try to get forward P/E from ratios
            metrics['fwd_pe'] = latest_ratios.get('priceEarningsRatio')
        
        if key_metrics_df is not None and not key_metrics_df.empty:
            latest_key_metrics = key_metrics_df.iloc[0]
            # Override with key metrics if available
            if latest_key_metrics.get('peRatio'):
                metrics['fwd_pe'] = latest_key_metrics['peRatio']
            if latest_key_metrics.get('pbRatio'):
                metrics['pb_ratio'] = latest_key_metrics['pbRatio']
    except Exception as e:
        print(f"Warning: Could not fetch financial ratios: {e}")
    
    # 6. Get dividend yield from profile or financial data
    if profile and profile.get('lastDiv'):
        if current_price and profile['lastDiv'] > 0:
            # Calculate dividend yield: (annual dividend / current price) * 100
            annual_dividend = profile['lastDiv']  # Assuming this is annual
            metrics['dividend_yield'] = (annual_dividend / current_price) * 100
    
    # 7. Get shares outstanding for free float calculation (approximate)
    try:
        if profile and profile.get('sharesOutstanding'):
            # Most companies have high free float, use a reasonable estimate
            metrics['free_float'] = 95.0  # Default assumption for large companies
    except Exception as e:
        print(f"Warning: Could not calculate free float: {e}")
    
    # Fill in any remaining None values with sensible defaults
    if metrics['free_float'] is None:
        metrics['free_float'] = 95.0
    if metrics['sector'] is None:
        metrics['sector'] = 'Technology'  # Default for many stocks
    if metrics['rating'] is None:
        metrics['rating'] = 'N/A'

    metrics["_fmp_cache"] = summarize_fmp_events(event_cursor)
    
    print(f"Successfully fetched metrics for {ticker}")
    return metrics


def get_technical_indicators(ticker: str, api_key: str) -> dict:
    """从 FMP 获取历史价格并计算 SMA50/200、RSI14、MACD、成交量信号。"""
    result = {
        'sma50': None, 'sma200': None, 'rsi14': None,
        'macd': None, 'macd_signal': None, 'macd_histogram': None,
        'avg_volume_20d': None, 'latest_volume': None,
        'price': None,
        'ma_signal': 'N/A', 'rsi_signal': 'N/A',
        'macd_signal_label': 'N/A', 'volume_signal': 'N/A',
        'overall_signal': 'N/A',
    }
    try:
        import numpy as np
        data, _ = _fmp_get_json(
            endpoint="historical-price-eod/full",
            ticker=ticker,
            api_key=api_key,
            params={"symbol": ticker, "timeseries": 250},
            timeout=15,
        )

        prices = []
        if isinstance(data, list):
            if data and isinstance(data[0], dict):
                for d in data:
                    if d.get('date') and d.get('close') and d.get('volume'):
                        prices.append({
                            'date': d.get('date'),
                            'close': d.get('close'),
                            'volume': d.get('volume')
                        })
        elif isinstance(data, dict):
            prices = data.get('historical', [])
        
        if len(prices) < 50:
            return result

        df = pd.DataFrame(prices).sort_values('date').reset_index(drop=True)
        close = df['close'].astype(float)
        volume = df['volume'].astype(float)

        result['price'] = close.iloc[-1]

        # SMA
        sma50 = close.rolling(50).mean().iloc[-1]
        sma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else None
        result['sma50'] = round(sma50, 2)
        if sma200 is not None:
            result['sma200'] = round(sma200, 2)

        # MA signal
        price = close.iloc[-1]
        if sma200 is not None:
            if price > sma50 > sma200:
                result['ma_signal'] = 'Bullish'
            elif price < sma50 < sma200:
                result['ma_signal'] = 'Bearish'
            elif price > sma200:
                result['ma_signal'] = 'Neutral-Bullish'
            else:
                result['ma_signal'] = 'Neutral-Bearish'
        elif price > sma50:
            result['ma_signal'] = 'Bullish'
        else:
            result['ma_signal'] = 'Bearish'

        # RSI 14
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        rsi_val = rsi.iloc[-1]
        result['rsi14'] = round(rsi_val, 1)
        if rsi_val > 70:
            result['rsi_signal'] = 'Overbought'
        elif rsi_val < 30:
            result['rsi_signal'] = 'Oversold'
        elif rsi_val > 55:
            result['rsi_signal'] = 'Bullish'
        elif rsi_val < 45:
            result['rsi_signal'] = 'Bearish'
        else:
            result['rsi_signal'] = 'Neutral'

        # MACD (12, 26, 9)
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9).mean()
        histogram = macd_line - signal_line
        result['macd'] = round(macd_line.iloc[-1], 2)
        result['macd_signal'] = round(signal_line.iloc[-1], 2)
        result['macd_histogram'] = round(histogram.iloc[-1], 2)
        if macd_line.iloc[-1] > signal_line.iloc[-1] and histogram.iloc[-1] > 0:
            result['macd_signal_label'] = 'Bullish'
        elif macd_line.iloc[-1] < signal_line.iloc[-1] and histogram.iloc[-1] < 0:
            result['macd_signal_label'] = 'Bearish'
        else:
            result['macd_signal_label'] = 'Neutral'

        # Volume
        avg_vol = volume.tail(20).mean()
        latest_vol = volume.iloc[-1]
        result['avg_volume_20d'] = round(avg_vol)
        result['latest_volume'] = round(latest_vol)
        vol_ratio = latest_vol / avg_vol if avg_vol > 0 else 1
        if vol_ratio > 1.5:
            result['volume_signal'] = 'High Activity'
        elif vol_ratio < 0.5:
            result['volume_signal'] = 'Low Activity'
        else:
            result['volume_signal'] = 'Normal'

        # Overall signal
        signals = [result['ma_signal'], result['rsi_signal'],
                   result['macd_signal_label']]
        bullish = sum(1 for s in signals if 'Bullish' in s or s == 'Oversold')
        bearish = sum(1 for s in signals if 'Bearish' in s or s == 'Overbought')
        if bullish >= 2:
            result['overall_signal'] = 'Bullish'
        elif bearish >= 2:
            result['overall_signal'] = 'Bearish'
        else:
            result['overall_signal'] = 'Neutral'

        print(f"✅ Computed technical indicators for {ticker}: {result['overall_signal']}")
    except Exception as e:
        print(f"⚠️ Could not compute technical indicators: {e}")
    return result


def get_company_news(
    ticker: str,
    api_key: str,
    days_back: int = 5,
    limit: int = 50,
    as_of_date: str | datetime.datetime | pd.Timestamp | None = None,
) -> list[dict] | None:
    """
    Fetches recent company news from FMP API.
    
    Args:
        ticker: Stock ticker symbol
        api_key: FMP API key
        days_back: Number of days to look back for news (default: 5)
        limit: Maximum number of news articles to fetch (default: 50)
    
    Returns:
        List of dictionaries containing filtered news data, or None if error occurs
    """
    try:
        from datetime import datetime, timedelta
        
        # Calculate date range
        end_date = _as_of_timestamp(as_of_date).to_pydatetime() if as_of_date is not None else datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Format dates for API
        from_date = start_date.strftime('%Y-%m-%d')
        to_date = end_date.strftime('%Y-%m-%d')
        
        params = {
            'tickers': ticker,
            'from': from_date,
            'to': to_date,
            'limit': limit,
        }
        
        print(f"Fetching news for {ticker} from {from_date} to {to_date}...")
        data, _ = _fmp_get_json(
            endpoint="news/stock",
            ticker=ticker,
            api_key=api_key,
            params=params,
        )
        
        if not data:
            print(f"No news data returned from FMP for {ticker}.")
            return None
        
        # Filter to keep only required fields
        filtered_news = []
        as_of_cutoff = _as_of_timestamp(as_of_date) if as_of_date is not None else None
        if as_of_cutoff is not None:
            as_of_cutoff = as_of_cutoff + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)

        for article in data:
            published_date = pd.to_datetime(article.get("publishedDate"), errors="coerce")
            if as_of_cutoff is not None and pd.notna(published_date) and published_date > as_of_cutoff:
                continue
            filtered_article = {
                'symbol': article.get('symbol'),
                'title': article.get('title'),
                'publishedDate': article.get('publishedDate'),
                'text': article.get('text'),
                'site': article.get('site'),
                'url': article.get('url')
            }
            filtered_news.append(filtered_article)
        
        print(f"Successfully fetched {len(filtered_news)} news articles for {ticker}")
        return filtered_news
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching news for {ticker}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error fetching news for {ticker}: {e}")
        return None


if __name__ == "__main__":
    print("Testing market_data_api.py...")
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path_test = os.path.join(current_script_dir, "..", "..", "config", "config.ini")
    
    fmp_api_key_for_test = "YOUR_FMP_KEY_HERE" 
    if not os.path.exists(config_path_test):
        print(f"Test config file not found at {config_path_test}. Creating a dummy one for structure test.")
        os.makedirs(os.path.dirname(config_path_test), exist_ok=True)
        with open(config_path_test, "w") as f:
            f.write("[API_KEYS]\n")
            f.write("fmp_api_key = YOUR_FMP_KEY_HERE\n")
        print("Please put a valid FMP API key in the dummy config.ini to run live tests.")
    else:
        try:
            test_config = load_config(config_path_test)
            fmp_api_key_for_test = get_api_key(test_config, section="API_KEYS", key="fmp_api_key")
        except Exception as e:
            print(f"Error loading test config: {e}")

    if fmp_api_key_for_test != "YOUR_FMP_KEY_HERE" and fmp_api_key_for_test:
        print(f"\nUsing FMP API Key: {fmp_api_key_for_test[:5]}... for live FMP tests")
        
        print("\nTesting get_comprehensive_financial_data for AAPL...")
        financial_data = get_comprehensive_financial_data("AAPL", fmp_api_key_for_test)
        for statement_type, df in financial_data.items():
            if df is not None and not df.empty:
                print(f"{statement_type}: {len(df)} years of data")
            else:
                print(f"{statement_type}: No data")
    else:
        print("\nSkipping live FMP API tests. Please provide a valid API key in config.ini.")

    print("\nmarket_data_api.py tests complete.")
