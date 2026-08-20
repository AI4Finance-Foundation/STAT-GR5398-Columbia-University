from __future__ import annotations

from pathlib import Path

import pandas as pd


def unique_tickers(groups, extras=()):
    out = set(extras)
    for names in groups.values():
        out.update(names)
    return sorted(out)


def load_prices(config, root: Path, tickers):
    cache = root / config.get("cache_path", "data/cache/prices.csv")
    cache.parent.mkdir(parents=True, exist_ok=True)
    required = sorted(set(tickers))
    min_obs = int(config.get("min_non_null_observations", 252))

    if cache.exists():
        cached = _trim(pd.read_csv(cache, index_col=0, parse_dates=True).sort_index(), config.get("start_date"), config.get("end_date"))
        ok, message = validate_required_tickers(cached, required, min_obs=min_obs, raise_on_error=False)
        if ok:
            return cached.reindex(columns=required)
        (root / "data/cache/cache_validation_error.txt").write_text(message + "\n", encoding="utf-8")

    prices = _download_yfinance_prices(required, config, root)

    local = _local_prices(root, required, config)
    for col in local.columns:
        if col not in prices.columns or prices[col].dropna().empty:
            prices[col] = local[col]

    prices = _trim(prices.sort_index().reindex(columns=required), config.get("start_date"), config.get("end_date"))
    validate_required_tickers(prices, required, min_obs=min_obs, raise_on_error=True, root=root)
    prices.to_csv(cache)
    _write_data_note(root, prices, required, config)
    missing_file = root / "data/cache/missing_tickers.csv"
    if missing_file.exists():
        missing_file.unlink()
    return prices


def validate_required_tickers(prices, required_tickers, min_obs=252, raise_on_error=True, root: Path | None = None):
    required = sorted(set(required_tickers))
    missing_cols = [t for t in required if t not in prices.columns]
    all_nan = [t for t in required if t in prices.columns and prices[t].dropna().empty]
    too_short = [t for t in required if t in prices.columns and 0 < int(prices[t].notna().sum()) < int(min_obs)]
    problems = []
    if missing_cols:
        problems.append("missing columns: " + ", ".join(missing_cols))
    if all_nan:
        problems.append("all-NaN columns: " + ", ".join(all_nan))
    if too_short:
        problems.append(f"fewer than {min_obs} non-null observations: " + ", ".join(too_short))
    if not problems:
        return True, "all required tickers present"

    message = "Price validation failed; " + "; ".join(problems)
    if root is not None:
        bad = sorted(set(missing_cols + all_nan + too_short))
        (root / "data/cache/missing_tickers.csv").write_text("ticker\n" + "\n".join(bad) + "\n", encoding="utf-8")
        (root / "data/cache/data_source_note.txt").write_text(
            "full required universe was NOT available; final backtest should not be treated as complete\n",
            encoding="utf-8",
        )
    if raise_on_error:
        raise RuntimeError(message)
    return False, message


def _download_yfinance_prices(tickers, config, root: Path):
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("yfinance is required to download the full Assignment 2 universe.") from exc

    start = config.get("start_date")
    end = config.get("end_date")
    errors = []

    def normalize(raw, requested):
        if raw is None or raw.empty:
            return pd.DataFrame()
        if isinstance(raw.columns, pd.MultiIndex):
            if "Adj Close" in raw.columns.get_level_values(0):
                out = raw["Adj Close"].copy()
            elif "Close" in raw.columns.get_level_values(0):
                out = raw["Close"].copy()
            else:
                return pd.DataFrame()
        else:
            field = "Adj Close" if "Adj Close" in raw.columns else "Close" if "Close" in raw.columns else None
            if field is None:
                return pd.DataFrame()
            out = raw[[field]].rename(columns={field: requested[0]})
        out.index = pd.to_datetime(out.index).tz_localize(None).normalize()
        return out

    try:
        raw = yf.download(tickers=tickers, start=start, end=end, auto_adjust=False, progress=False, threads=True)
        prices = normalize(raw, tickers)
    except Exception as exc:
        errors.append(f"bulk download failed: {exc}")
        prices = pd.DataFrame()

    missing = [t for t in tickers if t not in prices.columns or prices[t].dropna().empty]
    retry_frames = [prices] if not prices.empty else []
    for ticker in missing:
        try:
            raw_one = yf.download(tickers=[ticker], start=start, end=end, auto_adjust=False, progress=False, threads=False)
            one = normalize(raw_one, [ticker])
            if ticker in one.columns and not one[ticker].dropna().empty:
                retry_frames.append(one[[ticker]])
            else:
                errors.append(f"{ticker}: no adjusted close returned")
        except Exception as exc:
            errors.append(f"{ticker}: {exc}")

    if errors:
        (root / "data/cache/yfinance_error.txt").write_text("\n".join(errors) + "\n", encoding="utf-8")
    if not retry_frames:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="date"))
    out = pd.concat(retry_frames, axis=1)
    out = out.loc[:, ~out.columns.duplicated()].sort_index()
    return out


def _local_prices(root, tickers, config):
    a1 = (root / "../../../Assignment1/submissions/Assignment1_rayzhao_rz2759").resolve()
    frames = []
    f = a1 / "nasdaq_stock.csv"
    if f.exists():
        df = pd.read_csv(f, usecols=["tic", "datadate", "ajexdi", "prccd"])
        df = df[df.tic.isin(tickers)].copy()
        if not df.empty:
            df["date"] = pd.to_datetime(df.datadate).dt.normalize()
            df["price"] = pd.to_numeric(df.prccd, errors="coerce") / pd.to_numeric(df.ajexdi, errors="coerce")
            frames.append(df.pivot_table(index="date", columns="tic", values="price", aggfunc="last"))
    b = a1 / "backtest_result/original_equal/benchmark_qqq_spy_cache.csv"
    if b.exists():
        bm = pd.read_csv(b, parse_dates=["Date"]).rename(columns={"Date": "date"}).set_index("date")
        frames.append(bm[[c for c in ["QQQ", "SPY"] if c in tickers and c in bm.columns]])
    if not frames:
        return pd.DataFrame()
    px = pd.concat(frames, axis=1).loc[:, lambda x: ~x.columns.duplicated()].sort_index()
    return _trim(px.reindex(columns=[t for t in tickers if t in px.columns]), config.get("start_date"), config.get("end_date"))


def _write_data_note(root: Path, prices, required, config):
    note = (
        "final backtest uses the full required Adaptive Rotation universe with adjusted close prices; "
        f"source=yfinance with local Assignment 1 cache used only for empty columns if needed; "
        f"date_range={prices.index.min().date()} to {prices.index.max().date()}; "
        f"tickers={', '.join(required)}\n"
    )
    (root / "data/cache/data_source_note.txt").write_text(note, encoding="utf-8")


def _trim(px, start, end):
    if start:
        px = px.loc[pd.Timestamp(start):]
    if end:
        px = px.loc[:pd.Timestamp(end)]
    return px.dropna(how="all")


def daily_returns(prices):
    return prices.sort_index().ffill().pct_change(fill_method=None)
