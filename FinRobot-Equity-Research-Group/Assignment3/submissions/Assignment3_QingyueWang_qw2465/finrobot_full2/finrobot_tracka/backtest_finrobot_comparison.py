#!/usr/bin/env python
"""Backtest FinRL-only and point-in-time FinRobot against SPY."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

from build_finrobot_multipliers import (
    ROOT,
    clip01,
    pct_to_float,
    score_from_catalysts,
    score_from_risks,
)


DEFAULT_FINRL = ROOT / "finrl2" / "finrl_stock_selection.csv"
DEFAULT_IMPROVED_PANEL = ROOT / "finrobot_tracka" / "finrobot2_rebalanced_panel.csv"
DEFAULT_OUTPUT_DIR = ROOT / "finrobot_tracka" / "backtest_results"
DEFAULT_OUTPUT_ROOT = ROOT / "finrobot_outputs"


def legacy_financial_scores(analysis_dir: Path) -> dict:
    df = pd.read_csv(analysis_dir / "financial_metrics_and_forecasts.csv")
    latest_actual_cols = [c for c in df.columns if c.endswith("A")]
    latest = sorted(latest_actual_cols)[-1]

    def metric(name: str):
        row = df[df["metrics"] == name]
        if row.empty:
            return None
        return pct_to_float(row[latest].iloc[0])

    revenue_growth = metric("Revenue Growth")
    contribution_margin = metric("Contribution Margin")
    ebitda_margin = metric("EBITDA Margin")
    pe_ratio = metric("PE Ratio")
    eps = metric("EPS")

    growth_score = clip01(((revenue_growth or 0.0) + 0.10) / 0.50)
    margin_score = clip01(((ebitda_margin or 0.0) + (contribution_margin or 0.0)) / 0.90)
    profitability_score = 1.0 if (eps is not None and eps > 0) else 0.35
    valuation_score = clip01(1.0 - ((pe_ratio or 80.0) - 20.0) / 100.0)
    business_quality_score = 0.35 * growth_score + 0.40 * margin_score + 0.25 * profitability_score
    return {
        "business_quality_score": round(business_quality_score, 4),
        "valuation_score": round(valuation_score, 4),
    }


def legacy_risk_score(risks_txt: Path) -> dict:
    text = risks_txt.read_text(errors="replace").lower() if risks_txt.exists() else ""
    risk_terms = [
        "competition",
        "valuation",
        "regulatory",
        "cyclical",
        "margin",
        "supply",
        "geopolitical",
        "dependence",
        "concentration",
        "execution",
    ]
    hits = sum(1 for term in risk_terms if term in text)
    return {"risk_score": round(clip01(1.0 - hits / len(risk_terms)), 4)}


def legacy_decision(score: float) -> tuple[str, float]:
    if score >= 0.78:
        return "strong_keep", 1.20
    if score >= 0.62:
        return "keep", 1.00
    if score >= 0.48:
        return "reduce", 0.70
    return "reject", 0.00


def build_legacy_panel(finrl: pd.DataFrame, output_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = finrl[finrl["selected"] == 1].copy() if "selected" in finrl.columns else finrl.copy()
    rows = []
    for ticker in sorted(selected["ticker"].unique()):
        analysis_dir = output_root / ticker / "analysis"
        row = {"ticker": ticker}
        row.update(legacy_financial_scores(analysis_dir))
        row.update(score_from_catalysts(analysis_dir / "catalyst_analysis.json"))
        row.update(legacy_risk_score(analysis_dir / "risks.txt"))
        score = (
            0.45 * row["business_quality_score"]
            + 0.25 * row["valuation_score"]
            + 0.20 * row["catalyst_score"]
            + 0.10 * row["risk_score"]
        )
        decision, multiplier = legacy_decision(score)
        row["finrobot_score"] = round(score, 4)
        row["finrobot_decision"] = decision
        row["finrobot_multiplier"] = multiplier
        rows.append(row)

    multipliers = pd.DataFrame(rows)
    panel = selected.merge(
        multipliers[["ticker", "finrobot_score", "finrobot_decision", "finrobot_multiplier"]],
        on="ticker",
        how="left",
    )
    panel["initial_weight"] = panel.groupby("date")["ticker"].transform(lambda s: 1.0 / len(s))
    panel["after_finrobot_raw_weight"] = panel["initial_weight"] * panel["finrobot_multiplier"]
    raw_sum = panel.groupby("date")["after_finrobot_raw_weight"].transform("sum")
    panel["after_finrobot_weight"] = panel["after_finrobot_raw_weight"] / raw_sum
    return multipliers, panel


def download_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"].copy()
    else:
        prices = data[["Close"]].rename(columns={"Close": tickers[0]})
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    return prices.sort_index().dropna(how="all")


def weights_by_day(panel: pd.DataFrame, dates: pd.DatetimeIndex, weight_col: str) -> pd.DataFrame:
    wide = panel.pivot_table(index="date", columns="ticker", values=weight_col, aggfunc="sum")
    wide.index = pd.to_datetime(wide.index)
    daily = wide.reindex(dates.union(wide.index)).sort_index().ffill().reindex(dates).fillna(0.0)
    denom = daily.sum(axis=1).replace(0.0, math.nan)
    return daily.div(denom, axis=0).fillna(0.0)


def strategy_returns(panel: pd.DataFrame, returns: pd.DataFrame, weight_col: str) -> pd.Series:
    weights = weights_by_day(panel, returns.index, weight_col)
    aligned_returns = returns.reindex(columns=weights.columns).fillna(0.0)
    return (weights.shift(1).fillna(0.0) * aligned_returns).sum(axis=1)


def metrics(ret: pd.Series) -> dict:
    ret = ret.dropna()
    equity = (1.0 + ret).cumprod()
    total_return = equity.iloc[-1] - 1.0
    volatility = ret.std() * math.sqrt(252)
    sharpe = (ret.mean() * 252 / volatility) if volatility > 0 else 0.0
    max_drawdown = (equity / equity.cummax() - 1.0).min()
    return {
        "return": total_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "volatility": volatility,
    }


def contribution_table(
    before_panel: pd.DataFrame,
    after_panel: pd.DataFrame,
    returns: pd.DataFrame,
    before_col: str,
    after_col: str,
) -> pd.DataFrame:
    before_weights = weights_by_day(before_panel, returns.index, before_col)
    after_weights = weights_by_day(after_panel, returns.index, after_col)
    weight_change = after_weights.mean() - before_weights.mean()
    ticker_returns = returns.reindex(columns=after_weights.columns).fillna(0.0)
    realized_return = ticker_returns.mean() * 252
    out = pd.DataFrame(
        {
            "avg_weight_before": before_weights.mean(),
            "avg_weight_after": after_weights.mean(),
            "avg_weight_change": weight_change,
            "annualized_ticker_return": realized_return,
            "approx_active_contribution": weight_change * realized_return,
        }
    )
    return out.sort_values("approx_active_contribution", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finrl-csv", default=str(DEFAULT_FINRL))
    parser.add_argument("--improved-panel", default=str(DEFAULT_IMPROVED_PANEL))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--end", default="2026-05-01")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    finrl = pd.read_csv(args.finrl_csv)
    selected = finrl[finrl["selected"] == 1].copy() if "selected" in finrl.columns else finrl.copy()
    selected["initial_weight"] = selected.groupby("date")["ticker"].transform(lambda s: 1.0 / len(s))
    improved_panel = pd.read_csv(args.improved_panel)

    tickers = sorted(set(selected["ticker"]) | {"SPY"})
    prices = download_prices(tickers, args.start, args.end)
    prices.to_csv(output_dir / "prices.csv")
    returns = prices.pct_change().dropna(how="all")
    asset_returns = returns.drop(columns=["SPY"], errors="ignore")

    strategy_returns_map = {
        "FinRL-only": strategy_returns(selected, asset_returns, "initial_weight"),
        "FinRL + no-lookahead FinRobot": strategy_returns(improved_panel, asset_returns, "after_finrobot_weight"),
        "SPY": returns["SPY"],
    }
    returns_df = pd.DataFrame(strategy_returns_map).loc[returns.index]
    equity = (1.0 + returns_df.fillna(0.0)).cumprod()
    summary = pd.DataFrame({name: metrics(series) for name, series in returns_df.items()}).T

    returns_df.to_csv(output_dir / "strategy_daily_returns.csv")
    equity.to_csv(output_dir / "equity_curve.csv")
    summary.to_csv(output_dir / "performance_summary.csv")

    attribution = contribution_table(
        selected,
        improved_panel,
        asset_returns,
        "initial_weight",
        "after_finrobot_weight",
    )
    attribution.to_csv(output_dir / "finrl_vs_no_lookahead_finrobot_attribution.csv")

    ax = equity.plot(figsize=(10, 6), linewidth=1.8)
    ax.set_title("Track A Equity Curve vs SPY")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    ax.grid(True, alpha=0.25)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(output_dir / "equity_curve.png", dpi=180)

    print("Performance summary")
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))
    print("\nTop FinRL-vs-no-lookahead-FinRobot attribution changes")
    print(attribution.head(10).to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\nWrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
