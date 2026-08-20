"""Integrated backtest runner for FinRL + FinRobot comparison."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .active_setup import ActiveResearchSetup
from .research_automation import ensure_research_snapshot
from .research_flags import extract_research_flags
from .weights import apply_research_adjustment, load_base_weights


@dataclass(frozen=True)
class BacktestSummary:
    start: str
    end: str
    research_frequency: str
    base_weights_path: str
    finrobot_run_root: str
    output_dir: str


def _infer_research_dir(finrobot_run_root: Path, ticker: str, as_of_date: str) -> Path:
    candidates = [
        finrobot_run_root / as_of_date / ticker.upper() / "analysis",
        finrobot_run_root / ticker.upper() / as_of_date / "analysis",
        finrobot_run_root / ticker.upper() / "analysis",
        finrobot_run_root / "analysis",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No FinRobot research directory found for {ticker} under {finrobot_run_root}")


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _integration_output_dir(output_dir: str | Path | None, integration_run_id: str | None) -> tuple[str, Path]:
    run_id = integration_run_id or f"integration_{_utc_now_compact()}"
    if output_dir:
        base = Path(output_dir)
        if base.name == run_id:
            return run_id, base
        return run_id, base / run_id
    return run_id, Path("output") / "integration" / run_id


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_price_frame(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    current = Path(__file__).resolve()
    finrl_root = None
    for parent in current.parents:
        candidate = parent / "FinRL-Trading"
        if candidate.exists():
            finrl_root = candidate
            break

    try:
        if finrl_root is not None:
            sys.path.insert(0, str(finrl_root))
            sys.path.insert(0, str(finrl_root / "src"))
        from src.data.data_fetcher import fetch_price_data

        raw = fetch_price_data(tickers, start, end)
    except Exception as exc:
        if finrl_root is None:
            raise ImportError(f"Could not import FinRL price fetcher and no FinRL-Trading root was found: {exc}") from exc
        print(f"Warning: FinRL price fetcher unavailable; falling back to local CSV prices: {exc}")
        return _load_price_frame_from_csv(finrl_root / "data" / "fmp_daily", tickers, start, end)

    if raw.empty:
        if finrl_root is not None:
            csv_prices = _load_price_frame_from_csv(finrl_root / "data" / "fmp_daily", tickers, start, end)
            if not csv_prices.empty:
                return csv_prices
        return raw

    if {"datadate", "tic", "adj_close"}.issubset(raw.columns):
        wide = raw.pivot(index="datadate", columns="tic", values="adj_close").sort_index()
        wide.index = pd.to_datetime(wide.index)
        return wide.ffill().dropna(how="all")
    raise ValueError("Unexpected price data shape from FinRL fetcher")


def _load_price_frame_from_csv(data_dir: Path, tickers: list[str], start: str, end: str) -> pd.DataFrame:
    frames: dict[str, pd.Series] = {}
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    for ticker in sorted({str(t).upper() for t in tickers if str(t).upper() not in {"CASH", "USD"}}):
        path = data_dir / f"{ticker}_daily.csv"
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        columns = {c.lower(): c for c in df.columns}
        date_col = columns.get("date") or columns.get("datadate")
        price_col = columns.get("adj_close") or columns.get("adjusted_close") or columns.get("close")
        if not date_col or not price_col:
            continue
        dates = pd.to_datetime(df[date_col], errors="coerce")
        prices = pd.to_numeric(df[price_col], errors="coerce")
        series = pd.Series(prices.values, index=dates, name=ticker).dropna()
        series = series[(series.index >= start_ts) & (series.index <= end_ts)].sort_index()
        if not series.empty:
            frames[ticker] = series

    if not frames:
        return pd.DataFrame()
    return pd.DataFrame(frames).sort_index().ffill().dropna(how="all")


def _to_return_curve(prices: pd.DataFrame) -> pd.DataFrame:
    returns = prices.pct_change().fillna(0.0)
    return (1.0 + returns).cumprod()


def _summarize_curves(curves: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    if curves.empty:
        return {}

    out: dict[str, dict[str, float | None]] = {}
    for col in curves.columns:
        series = pd.to_numeric(curves[col], errors="coerce").dropna()
        if series.empty:
            out[col] = {
                "total_return": None,
                "annualized_return": None,
                "annualized_volatility": None,
                "max_drawdown": None,
            }
            continue

        total_return = float(series.iloc[-1] / series.iloc[0] - 1.0) if series.iloc[0] else None
        years = max((series.index[-1] - series.index[0]).days / 365.25, 1.0 / 365.25)
        daily_returns = series.pct_change().dropna()
        annualized_return = (1.0 + total_return) ** (1.0 / years) - 1.0 if total_return is not None else None
        annualized_volatility = float(daily_returns.std() * (252 ** 0.5)) if not daily_returns.empty else 0.0
        drawdown = series / series.cummax() - 1.0
        out[col] = {
            "total_return": total_return,
            "annualized_return": float(annualized_return) if annualized_return is not None else None,
            "annualized_volatility": annualized_volatility,
            "max_drawdown": float(drawdown.min()) if not drawdown.empty else None,
        }

    if {"finrl_only", "finrl_plus_finrobot"}.issubset(out):
        base_ret = out["finrl_only"].get("total_return")
        integrated_ret = out["finrl_plus_finrobot"].get("total_return")
        out["comparison"] = {
            "finrobot_total_return_delta": (
                float(integrated_ret - base_ret)
                if integrated_ret is not None and base_ret is not None
                else None
            )
        }
    return out


def _latest_key_on_or_before(keys: list[str], date_key: str) -> str:
    if not keys:
        raise ValueError("No date keys available")
    decision_ts = pd.Timestamp(date_key)
    candidates = [key for key in keys if pd.Timestamp(key) <= decision_ts]
    return max(candidates, key=lambda key: pd.Timestamp(key)) if candidates else min(keys, key=lambda key: pd.Timestamp(key))


def run_integrated_backtest(
    *,
    start: str,
    end: str,
    research_frequency: str,
    finrl_weights: str | Path,
    finrobot_run_root: str | Path | None,
    output_dir: str | Path | None,
    llm_settings,
    active_setup: ActiveResearchSetup | None = None,
    integration_run_id: str | None = None,
    research_mode: str = "existing",
    config_file: str | Path | None = None,
    fmp_cache_policy: str = "reuse-until-forced",
    force_refresh_fmp: bool = False,
    fmp_cache_dir: str | Path | None = None,
    company_names: dict[str, str] | None = None,
    peer_map: dict[str, list[str]] | None = None,
    python_executable: str | None = None,
    force_refresh_flags: bool = False,
    no_op_band: tuple[float, float] = (0.90, 1.10),
    max_relative_adjustment: float = 0.20,
    defensive_stock_threshold: float = 0.50,
    defensive_cash_threshold: float = 0.20,
    defensive_max_relative_adjustment: float = 0.10,
    stock_boost_cash_funding_limit: float = 0.0,
    stock_boost_etf_funding_limit: float = 0.0,
    enable_enhanced_news: bool = False,
    enable_catalyst_analysis: bool = False,
    news_days_back: int = 5,
    news_limit: int = 50,
) -> dict[str, Any]:
    run_id, output_dir = _integration_output_dir(output_dir, integration_run_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    finrobot_run_root_path = Path(finrobot_run_root) if finrobot_run_root else output_dir / "research_snapshots"

    config_dir = output_dir / "config"
    flags_cache_dir = output_dir / "flags_cache"
    weights_dir = output_dir / "weights"
    audit_dir = output_dir / "audit"
    for directory in [config_dir, flags_cache_dir, weights_dir, audit_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    if active_setup is not None:
        _write_json(config_dir / "active_research_setup_snapshot.json", active_setup.to_dict())

    base_weights = load_base_weights(finrl_weights)
    stock_tickers = sorted(
        {
            str(row["ticker"]).upper()
            for _, row in base_weights.iterrows()
            if str(row.get("asset_type", "")).lower() == "stock"
        }
    )

    freq_dates = [pd.Timestamp(start)]
    if research_frequency == "monthly":
        freq_dates = list(pd.date_range(start=start, end=end, freq="MS"))
        if not freq_dates or freq_dates[0] != pd.Timestamp(start):
            freq_dates = [pd.Timestamp(start)] + freq_dates
    elif research_frequency == "weekly":
        freq_dates = list(pd.date_range(start=start, end=end, freq="W-MON"))
        if not freq_dates or freq_dates[0] != pd.Timestamp(start):
            freq_dates = [pd.Timestamp(start)] + freq_dates
    elif research_frequency == "once":
        freq_dates = [pd.Timestamp(start)]
    else:
        raise ValueError("research-frequency must be one of monthly, weekly, once")

    research_snapshots: dict[str, dict] = {}
    flags_by_date: dict[str, dict] = {}
    audit_by_date: dict[str, dict] = {}
    snapshot_status_by_date: dict[str, dict] = {}
    rebalance_rows: list[pd.DataFrame] = []

    for decision_date in freq_dates:
        as_of_date = decision_date.strftime("%Y-%m-%d")
        daily_flags = {}
        daily_audit = {}
        daily_snapshots = {}
        for ticker in stock_tickers:
            if research_mode == "existing" and finrobot_run_root:
                research_dir = _infer_research_dir(finrobot_run_root_path, ticker, as_of_date)
                daily_snapshots[ticker] = {
                    "ticker": ticker,
                    "as_of_date": as_of_date,
                    "analysis_dir": str(research_dir),
                    "status": "external",
                    "generated": False,
                }
            elif active_setup is not None:
                snapshot = ensure_research_snapshot(
                    integration_output_dir=output_dir,
                    as_of_date=as_of_date,
                    ticker=ticker,
                    mode=research_mode,
                    active_setup=active_setup,
                    base_config_path=config_file,
                    fmp_cache_policy=fmp_cache_policy,
                    force_refresh_fmp=force_refresh_fmp,
                    fmp_cache_dir=fmp_cache_dir,
                    company_names=company_names,
                    peer_map=peer_map,
                    python_executable=python_executable,
                    enable_enhanced_news=enable_enhanced_news,
                    enable_catalyst_analysis=enable_catalyst_analysis,
                    news_days_back=news_days_back,
                    news_limit=news_limit,
                )
                if snapshot.status == "failed":
                    raise RuntimeError(f"Research snapshot generation failed for {ticker} {as_of_date}: {snapshot.detail}")
                research_dir = snapshot.analysis_dir
                daily_snapshots[ticker] = snapshot.to_dict()
            else:
                if research_mode != "existing":
                    raise ValueError("research generation requires an active research setup")
                research_dir = _infer_research_dir(finrobot_run_root_path, ticker, as_of_date)
                daily_snapshots[ticker] = {
                    "ticker": ticker,
                    "as_of_date": as_of_date,
                    "analysis_dir": str(research_dir),
                    "status": "external",
                    "generated": False,
                }
            flags, meta = extract_research_flags(
                ticker=ticker,
                as_of_date=as_of_date,
                research_dir=research_dir,
                llm_settings=llm_settings,
                cache_dir=flags_cache_dir,
                force_refresh=force_refresh_flags,
            )
            daily_flags[ticker] = flags.to_dict()
            daily_audit[ticker] = {
                "flags": flags.to_dict(),
                "llm_meta": meta,
                "decision_date": as_of_date,
                "research_snapshot": daily_snapshots[ticker],
            }
        research_snapshots[as_of_date] = daily_flags
        flags_by_date[as_of_date] = daily_flags
        audit_by_date[as_of_date] = daily_audit
        snapshot_status_by_date[as_of_date] = daily_snapshots

    if "date" in base_weights.columns and base_weights["date"].notna().any():
        snapshot_dates = sorted(base_weights["date"].dropna().unique())
    else:
        snapshot_dates = [pd.Timestamp(start)]

    def _base_snapshot_for(date_key: str) -> pd.DataFrame:
        if "date" not in base_weights.columns or base_weights["date"].isna().all():
            return base_weights.drop(columns=["date"], errors="ignore").copy()
        decision_ts = pd.Timestamp(date_key)
        candidates = [d for d in snapshot_dates if pd.Timestamp(d) <= decision_ts]
        chosen = max(candidates) if candidates else min(snapshot_dates)
        snap = base_weights.loc[base_weights["date"] == chosen].copy()
        if snap.empty:
            snap = base_weights.drop(columns=["date"], errors="ignore").copy()
        return snap

    flag_dates_sorted = sorted(flags_by_date.keys())
    rebalance_dates = (
        [pd.Timestamp(d).strftime("%Y-%m-%d") for d in snapshot_dates]
        if "date" in base_weights.columns and base_weights["date"].notna().any()
        else flag_dates_sorted
    )
    for as_of_date in rebalance_dates:
        snapshot_base = _base_snapshot_for(as_of_date)
        flag_date = _latest_key_on_or_before(flag_dates_sorted, as_of_date)
        adjusted_weights, audit = apply_research_adjustment(
            snapshot_base,
            flags_by_date[flag_date],
            no_op_band=no_op_band,
            max_relative_adjustment=max_relative_adjustment,
            defensive_stock_threshold=defensive_stock_threshold,
            defensive_cash_threshold=defensive_cash_threshold,
            defensive_max_relative_adjustment=defensive_max_relative_adjustment,
            stock_boost_cash_funding_limit=stock_boost_cash_funding_limit,
            stock_boost_etf_funding_limit=stock_boost_etf_funding_limit,
        )
        adjusted_weights["as_of_date"] = as_of_date
        adjusted_weights["source_rebalance_date"] = as_of_date
        adjusted_weights["source_research_date"] = flag_date
        rebalance_rows.append(adjusted_weights)
        audit_entry = audit_by_date.setdefault(as_of_date, {})
        audit_entry["source_research_date"] = flag_date
        audit_entry["portfolio"] = audit.get("__portfolio__", {})
        audit_entry["positions"] = audit

    latest_as_of = max(rebalance_dates, key=lambda key: pd.Timestamp(key)) if rebalance_dates else None
    latest_adjusted = rebalance_rows[-1].copy() if rebalance_rows else pd.DataFrame()
    latest_audit = audit_by_date[latest_as_of] if latest_as_of else {}

    base_weights_path = weights_dir / "base_weights_normalized.csv"
    base_weights.to_csv(base_weights_path, index=False)
    adjusted_weights_path = weights_dir / "adjusted_weights.csv"
    latest_adjusted.to_csv(adjusted_weights_path, index=False)
    if rebalance_rows:
        weight_history_df = pd.concat(rebalance_rows, ignore_index=True)
        weight_history_path = weights_dir / "weight_history.csv"
        weight_history_df.to_csv(weight_history_path, index=False)
    else:
        weight_history_path = None

    flags_path = audit_dir / "research_flags.json"
    _write_json(flags_path, research_snapshots)
    audit_path = audit_dir / "decision_audit.json"
    _write_json(audit_path, audit_by_date)
    snapshots_path = audit_dir / "research_snapshots.json"
    _write_json(snapshots_path, snapshot_status_by_date)

    universe = sorted(set(base_weights["ticker"].astype(str).str.upper().tolist() + ["SPY"]))
    price_error = None
    try:
        price_frame = _load_price_frame(universe, start, end)
    except Exception as exc:
        price_error = str(exc)
        print(f"Warning: could not load price data for equity curves: {exc}")
        price_frame = pd.DataFrame()
    if not price_frame.empty:
        daily_returns = price_frame.pct_change().fillna(0.0)

        base_snapshots = []
        if "date" in base_weights.columns and base_weights["date"].notna().any():
            for d in sorted(base_weights["date"].dropna().unique()):
                snap = base_weights.loc[base_weights["date"] == d].copy()
                snap["as_of_date"] = pd.Timestamp(d).strftime("%Y-%m-%d")
                base_snapshots.append(snap)
        else:
            snap = base_weights.copy()
            snap["as_of_date"] = start
            base_snapshots.append(snap)
        base_snapshots_df = pd.concat(base_snapshots, ignore_index=True)

        adj_snapshots_df = weight_history_df if rebalance_rows else pd.DataFrame()

        def _build_daily_weights(snapshot_df: pd.DataFrame, value_col: str) -> pd.DataFrame:
            out = pd.DataFrame(0.0, index=price_frame.index, columns=price_frame.columns)
            if snapshot_df.empty:
                return out
            snapshot_df = snapshot_df.copy()
            snapshot_df["as_of_date"] = pd.to_datetime(snapshot_df["as_of_date"])
            for dt in price_frame.index:
                eligible = snapshot_df[snapshot_df["as_of_date"] <= dt]
                if eligible.empty:
                    chosen = snapshot_df.iloc[0]
                    chosen_df = snapshot_df.iloc[[0]]
                else:
                    chosen_dt = eligible["as_of_date"].max()
                    chosen_df = snapshot_df[snapshot_df["as_of_date"] == chosen_dt]
                row = pd.Series(0.0, index=price_frame.columns)
                for _, position in chosen_df.iterrows():
                    ticker = str(position["ticker"]).upper()
                    if ticker in row.index:
                        row[ticker] = float(position[value_col])
                out.loc[dt] = row.reindex(out.columns).fillna(0.0)
            return out.ffill().fillna(0.0)

        base_weights_daily = _build_daily_weights(base_snapshots_df.rename(columns={"weight": "base_weight"}), "base_weight")
        adjusted_weights_daily = _build_daily_weights(adj_snapshots_df, "adjusted_weight")

        base_curve = (1.0 + (daily_returns * base_weights_daily).sum(axis=1)).cumprod()
        adjusted_curve = (1.0 + (daily_returns * adjusted_weights_daily).sum(axis=1)).cumprod()
        spy_curve = (1.0 + daily_returns["SPY"]).cumprod() if "SPY" in daily_returns.columns else pd.Series(index=price_frame.index, dtype=float)

        curves = pd.DataFrame(
            {
                "finrl_only": base_curve,
                "finrl_plus_finrobot": adjusted_curve,
                "spy": spy_curve,
            }
        ).ffill()
        curves_path = output_dir / "equity_curves.csv"
        curves.to_csv(curves_path)
    else:
        curves = pd.DataFrame()
        if price_error is None:
            price_error = "price data unavailable or empty; equity_curves.csv was not generated"

    summary = {
        "integration_run_id": run_id,
        "window": {"start": start, "end": end},
        "research_frequency": research_frequency,
        "research_mode": research_mode,
        "news_controls": {
            "enable_enhanced_news": enable_enhanced_news,
            "enable_catalyst_analysis": enable_catalyst_analysis,
            "news_days_back": news_days_back,
            "news_limit": news_limit,
        },
        "overlay_controls": {
            "no_op_band": list(no_op_band),
            "max_relative_adjustment": max_relative_adjustment,
            "defensive_stock_threshold": defensive_stock_threshold,
            "defensive_cash_threshold": defensive_cash_threshold,
            "defensive_max_relative_adjustment": defensive_max_relative_adjustment,
            "stock_boost_cash_funding_limit": stock_boost_cash_funding_limit,
            "stock_boost_etf_funding_limit": stock_boost_etf_funding_limit,
        },
        "active_setup": active_setup.to_dict() if active_setup else None,
        "base_weights_path": str(base_weights_path),
        "adjusted_weights_path": str(adjusted_weights_path),
        "weight_history_path": str(weight_history_path) if weight_history_path else None,
        "research_flags_path": str(flags_path),
        "decision_audit_path": str(audit_path),
        "research_snapshots_path": str(snapshots_path),
        "equity_curves_path": str(output_dir / "equity_curves.csv") if not curves.empty else None,
        "equity_curves_error": price_error,
        "performance": _summarize_curves(curves),
        "final_cash_weight": float(latest_audit.get("portfolio", {}).get("cash_weight", 0.0)),
    }
    summary_path = output_dir / "metrics.json"
    _write_json(summary_path, summary)
    _write_json(output_dir / "integrated_backtest_summary.json", summary)
    return summary
