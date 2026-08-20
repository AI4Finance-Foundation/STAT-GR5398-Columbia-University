#!/usr/bin/env python
"""Create FinRobot research multipliers from real FinRobot outputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FINRL = ROOT / "finrl" / "finrl_stock_selection.csv"
DEFAULT_OUTPUT_ROOT = ROOT / "finrobot_outputs"
DEFAULT_RESULT = ROOT / "finrobot_tracka" / "finrobot_multipliers.csv"
DEFAULT_PANEL = ROOT / "finrobot_tracka" / "finrobot_rebalanced_panel.csv"
DEFAULT_NEWS_SIGNALS = ROOT / "finrobot_tracka" / "historical_news_signals.csv"


def pct_to_float(value) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    if text in {"", "N/A", "nan"}:
        return None
    if text.endswith("%"):
        try:
            return float(text[:-1]) / 100.0
        except ValueError:
            return None
    try:
        return float(text)
    except ValueError:
        return None


def clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def score_range(value: float | None, low: float, high: float, neutral: float = 0.5) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return neutral
    if high == low:
        return neutral
    return clip01((value - low) / (high - low))


def inverse_score_range(value: float | None, low: float, high: float, neutral: float = 0.5) -> float:
    base = score_range(value, low, high, neutral)
    return neutral if value is None or (isinstance(value, float) and math.isnan(value)) else 1.0 - base


def latest_row(csv_path: Path) -> pd.Series | None:
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if df.empty:
        return None
    if "year" in df.columns:
        return df.sort_values("year").iloc[-1]
    if "calendarYear" in df.columns:
        return df.sort_values("calendarYear").iloc[-1]
    return df.iloc[0]


def row_for_year(csv_path: Path, year: int) -> pd.Series | None:
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if df.empty or "year" not in df.columns:
        return None
    rows = df[pd.to_numeric(df["year"], errors="coerce") == year]
    if rows.empty:
        return None
    return rows.iloc[0]


def latest_available_filing(analysis_dir: Path, as_of_date: pd.Timestamp) -> tuple[int | None, pd.Timestamp | None]:
    statement = analysis_dir / "income_statement_raw_data.csv"
    if not statement.exists():
        return None, None
    df = pd.read_csv(statement)
    if "year" not in df.columns or "acceptedDate" not in df.columns:
        return None, None
    df["accepted_ts"] = pd.to_datetime(df["acceptedDate"], errors="coerce").dt.tz_localize(None)
    available = df[df["accepted_ts"] <= as_of_date].copy()
    if available.empty:
        return None, None
    row = available.sort_values(["year", "accepted_ts"]).iloc[-1]
    return int(row["year"]), row["accepted_ts"]


def pct_rank(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    ranks = values.rank(pct=True, method="average")
    if not higher_is_better:
        ranks = 1.0 - ranks + (1.0 / max(values.notna().sum(), 1))
    return ranks.fillna(0.5).clip(0.0, 1.0)


def score_from_financials(analysis_dir: Path, as_of_date: pd.Timestamp) -> dict:
    year, accepted_ts = latest_available_filing(analysis_dir, as_of_date)
    if year is None:
        return {
            "latest_actual_year": None,
            "financials_available_as_of": None,
            "latest_revenue_growth": None,
            "latest_ebitda_margin": None,
            "latest_contribution_margin": None,
            "latest_pe_ratio": None,
            "latest_ev_ebitda": None,
            "latest_fcf_yield": None,
            "latest_fcf_margin": None,
            "latest_ocf_sales_ratio": None,
            "latest_debt_to_equity": None,
            "latest_current_ratio": None,
            "latest_roe": None,
            "latest_eps": None,
            "growth_score_abs": 0.5,
            "margin_score_abs": 0.5,
            "earnings_score_abs": 0.5,
            "cashflow_score_abs": 0.5,
            "balance_sheet_score_abs": 0.5,
            "roe_score_abs": 0.5,
            "absolute_quality_score": 0.5,
        }

    df = pd.read_csv(analysis_dir / "financial_metrics_and_forecasts.csv")
    latest = f"{year}A"

    def metric(name: str):
        if latest not in df.columns:
            return None
        row = df[df["metrics"] == name]
        if row.empty:
            return None
        return pct_to_float(row[latest].iloc[0])

    revenue_growth = metric("Revenue Growth")
    contribution_margin = metric("Contribution Margin")
    ebitda_margin = metric("EBITDA Margin")
    pe_ratio = metric("PE Ratio")
    eps = metric("EPS")

    key = row_for_year(analysis_dir / "key_metrics_raw_data.csv", year)
    ratios = row_for_year(analysis_dir / "ratios_raw_data.csv", year)

    fcf_yield = pct_to_float(key.get("freeCashFlowYield")) if key is not None else None
    ev_ebitda = pct_to_float(key.get("enterpriseValueOverEBITDA")) if key is not None else None
    fcf_margin = pct_to_float(ratios.get("freeCashFlowOperatingCashFlowRatio")) if ratios is not None else None
    operating_cash_flow_sales = pct_to_float(ratios.get("operatingCashFlowSalesRatio")) if ratios is not None else None
    debt_to_equity = pct_to_float(ratios.get("debtEquityRatio")) if ratios is not None else None
    current_ratio = pct_to_float(ratios.get("currentRatio")) if ratios is not None else None
    roe = pct_to_float(ratios.get("returnOnEquity")) if ratios is not None else None

    growth_score = score_range(revenue_growth, -0.05, 0.30)
    margin_score = clip01(
        0.60 * score_range(ebitda_margin, 0.05, 0.45)
        + 0.40 * score_range(contribution_margin, 0.20, 0.65)
    )
    earnings_score = 0.5 if eps is None else (1.0 if eps > 0 else 0.25)
    cashflow_score = clip01(
        0.55 * score_range(fcf_margin, 0.05, 0.35)
        + 0.45 * score_range(operating_cash_flow_sales, 0.05, 0.30)
    )
    balance_sheet_score = clip01(
        0.50 * inverse_score_range(debt_to_equity, 0.0, 2.0)
        + 0.50 * score_range(current_ratio, 0.8, 2.0)
    )
    roe_score = score_range(roe, 0.0, 0.30)

    absolute_quality_score = (
        0.22 * growth_score
        + 0.24 * margin_score
        + 0.18 * earnings_score
        + 0.18 * cashflow_score
        + 0.10 * balance_sheet_score
        + 0.08 * roe_score
    )

    return {
        "latest_actual_year": latest,
        "financials_available_as_of": accepted_ts.date().isoformat() if accepted_ts is not None else None,
        "latest_revenue_growth": revenue_growth,
        "latest_ebitda_margin": ebitda_margin,
        "latest_contribution_margin": contribution_margin,
        "latest_pe_ratio": pe_ratio,
        "latest_ev_ebitda": ev_ebitda,
        "latest_fcf_yield": fcf_yield,
        "latest_fcf_margin": fcf_margin,
        "latest_ocf_sales_ratio": operating_cash_flow_sales,
        "latest_debt_to_equity": debt_to_equity,
        "latest_current_ratio": current_ratio,
        "latest_roe": roe,
        "latest_eps": eps,
        "growth_score_abs": round(growth_score, 4),
        "margin_score_abs": round(margin_score, 4),
        "earnings_score_abs": round(earnings_score, 4),
        "cashflow_score_abs": round(cashflow_score, 4),
        "balance_sheet_score_abs": round(balance_sheet_score, 4),
        "roe_score_abs": round(roe_score, 4),
        "absolute_quality_score": round(absolute_quality_score, 4),
    }


def score_from_catalysts(analysis_dir: Path, as_of_date: pd.Timestamp) -> dict:
    articles_path = analysis_dir / "company_news.json"
    articles = []
    if articles_path.exists():
        data = json.loads(articles_path.read_text())
        articles = data if isinstance(data, list) else data.get("articles", [])

    dated_articles = []
    for article in articles:
        published = pd.to_datetime(article.get("publishedDate"), errors="coerce")
        if pd.isna(published):
            continue
        published = published.tz_localize(None)
        if published <= as_of_date:
            dated_articles.append(article)

    pos = sum(1 for c in dated_articles if str(c.get("sentiment", "")).lower() == "positive")
    neg = sum(1 for c in dated_articles if str(c.get("sentiment", "")).lower() == "negative")
    neu = sum(1 for c in dated_articles if str(c.get("sentiment", "")).lower() == "neutral")
    total = pos + neg + neu
    if total == 0:
        return {
            "positive_catalysts": 0,
            "negative_catalysts": 0,
            "neutral_catalysts": 0,
            "catalyst_articles_used": 0,
            "catalyst_score": 0.5,
            "catalyst_evidence": "neutral_no_point_in_time_news",
        }
    directional_score = (pos + 1) / (pos + neg + 2)
    coverage_score = min(total / 8.0, 1.0)
    catalyst_score = clip01(0.70 * directional_score + 0.30 * coverage_score)
    return {
        "positive_catalysts": pos,
        "negative_catalysts": neg,
        "neutral_catalysts": neu,
        "catalyst_articles_used": total,
        "catalyst_score": round(catalyst_score, 4),
        "catalyst_evidence": "company_news_published_before_rebalance",
    }


def score_from_risks() -> dict:
    return {
        "risk_term_hits": 0,
        "risk_weighted_hits": 0.0,
        "risk_score": 0.5,
        "risk_evidence": "neutral_no_point_in_time_risk_text",
    }


def score_from_historical_news(news_row: pd.Series | None) -> dict:
    if news_row is None:
        return {
            "positive_catalysts": 0,
            "negative_catalysts": 0,
            "neutral_catalysts": 0,
            "catalyst_articles_used": 0,
            "catalyst_score": 0.5,
            "catalyst_evidence": "neutral_no_historical_news_signal",
            "risk_term_hits": 0,
            "risk_weighted_hits": 0.0,
            "risk_score": 0.5,
            "risk_evidence": "neutral_no_historical_news_signal",
            "news_count": 0,
            "news_start_date": None,
            "news_end_date": None,
            "top_news_titles": "",
        }

    return {
        "positive_catalysts": int(news_row.get("positive_news", 0) or 0),
        "negative_catalysts": int(news_row.get("negative_news", 0) or 0),
        "neutral_catalysts": int(news_row.get("neutral_news", 0) or 0),
        "catalyst_articles_used": int(news_row.get("news_count", 0) or 0),
        "catalyst_score": float(news_row.get("historical_catalyst_score", 0.5) or 0.5),
        "catalyst_evidence": "historical_fmp_news_published_before_rebalance",
        "risk_term_hits": int(news_row.get("risk_news_count", 0) or 0),
        "risk_weighted_hits": float(news_row.get("risk_weighted_hits", 0.0) or 0.0),
        "risk_score": float(news_row.get("historical_risk_score", 0.5) or 0.5),
        "risk_evidence": "historical_fmp_news_published_before_rebalance",
        "news_count": int(news_row.get("news_count", 0) or 0),
        "news_start_date": news_row.get("news_start_date"),
        "news_end_date": news_row.get("news_end_date"),
        "top_news_titles": news_row.get("top_news_titles", ""),
    }


def decision_and_multiplier(score: float) -> tuple[str, float]:
    if score >= 0.80:
        return "strong_keep", 1.20
    if score >= 0.60:
        return "keep", 1.00
    if score >= 0.45:
        return "reduce", 0.70
    return "reject", 0.00


def add_relative_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    grouped = out.groupby("date", group_keys=False)
    out["growth_score_rel"] = grouped["latest_revenue_growth"].transform(pct_rank)
    out["margin_score_rel"] = grouped["latest_ebitda_margin"].transform(pct_rank)
    out["cashflow_score_rel"] = grouped["latest_fcf_yield"].transform(pct_rank)
    out["roe_score_rel"] = grouped["latest_roe"].transform(pct_rank)
    out["leverage_score_rel"] = grouped["latest_debt_to_equity"].transform(
        lambda s: pct_rank(s, higher_is_better=False)
    )

    out["quality_relative_score"] = (
        0.25 * out["growth_score_rel"]
        + 0.25 * out["margin_score_rel"]
        + 0.20 * out["cashflow_score_rel"]
        + 0.15 * out["roe_score_rel"]
        + 0.15 * out["leverage_score_rel"]
    )
    out["business_quality_score"] = (
        0.60 * out["absolute_quality_score"]
        + 0.40 * out["quality_relative_score"]
    ).round(4)

    pe_score = grouped["latest_pe_ratio"].transform(lambda s: pct_rank(s, higher_is_better=False))
    ev_ebitda_score = grouped["latest_ev_ebitda"].transform(lambda s: pct_rank(s, higher_is_better=False))
    fcf_yield_score = grouped["latest_fcf_yield"].transform(pct_rank)
    out["valuation_score"] = (
        0.40 * pe_score
        + 0.35 * ev_ebitda_score
        + 0.25 * fcf_yield_score
    ).round(4)

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finrl-csv", default=str(DEFAULT_FINRL))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--result-csv", default=str(DEFAULT_RESULT))
    parser.add_argument("--panel-csv", default=str(DEFAULT_PANEL))
    parser.add_argument("--news-signals-csv", default=str(DEFAULT_NEWS_SIGNALS))
    args = parser.parse_args()

    finrl = pd.read_csv(args.finrl_csv)
    selected = finrl[finrl["selected"] == 1].copy() if "selected" in finrl.columns else finrl.copy()
    rows = []
    output_root = Path(args.output_root)
    selected["date"] = pd.to_datetime(selected["date"]).dt.tz_localize(None)
    news_lookup = {}
    news_path = Path(args.news_signals_csv)
    if news_path.exists():
        news_signals = pd.read_csv(news_path)
        news_signals["date"] = pd.to_datetime(news_signals["date"]).dt.tz_localize(None)
        news_lookup = {
            (row["date"], row["ticker"]): row
            for _, row in news_signals.iterrows()
        }
    for selected_row in selected[["date", "ticker"]].drop_duplicates().itertuples(index=False):
        rebalance_date = selected_row.date
        ticker = selected_row.ticker
        analysis_dir = output_root / ticker / "analysis"
        metrics_csv = analysis_dir / "financial_metrics_and_forecasts.csv"
        if not metrics_csv.exists():
            raise FileNotFoundError(f"Missing FinRobot analysis for {ticker}: {metrics_csv}")

        row = {"date": rebalance_date, "ticker": ticker}
        row.update(score_from_financials(analysis_dir, rebalance_date))
        row.update(score_from_historical_news(news_lookup.get((rebalance_date, ticker))))
        row["evidence_path"] = str(analysis_dir.relative_to(ROOT))
        rows.append(row)

    multipliers = add_relative_scores(pd.DataFrame(rows))
    final_rows = []
    for row in multipliers.to_dict("records"):
        finrobot_score = (
            0.45 * row["business_quality_score"]
            + 0.25 * row["valuation_score"]
            + 0.20 * row["catalyst_score"]
            + 0.10 * row["risk_score"]
        )
        decision, multiplier = decision_and_multiplier(finrobot_score)
        row["finrobot_score"] = round(finrobot_score, 4)
        row["finrobot_decision"] = decision
        row["finrobot_multiplier"] = multiplier
        row["finrobot_rationale"] = (
            f"quality={row['business_quality_score']:.2f}, valuation={row['valuation_score']:.2f}, "
            f"catalyst={row['catalyst_score']:.2f}, risk={row['risk_score']:.2f}"
        )
        final_rows.append(row)

    multipliers = pd.DataFrame(final_rows).sort_values(["date", "finrobot_score", "ticker"], ascending=[True, False, True])
    Path(args.result_csv).parent.mkdir(parents=True, exist_ok=True)
    multipliers.to_csv(args.result_csv, index=False)

    panel = selected.merge(
        multipliers[
            [
                "date",
                "ticker",
                "financials_available_as_of",
                "latest_actual_year",
                "business_quality_score",
                "valuation_score",
                "catalyst_score",
                "risk_score",
                "news_count",
                "positive_catalysts",
                "negative_catalysts",
                "neutral_catalysts",
                "catalyst_articles_used",
                "risk_term_hits",
                "risk_weighted_hits",
                "news_start_date",
                "news_end_date",
                "top_news_titles",
                "finrobot_score",
                "finrobot_decision",
                "finrobot_multiplier",
                "finrobot_rationale",
                "evidence_path",
            ]
        ],
        on=["date", "ticker"],
        how="left",
    )
    panel["initial_weight"] = panel.groupby("date")["ticker"].transform(lambda s: 1.0 / len(s))
    panel["after_finrobot_raw_weight"] = panel["initial_weight"] * panel["finrobot_multiplier"]
    raw_sum = panel.groupby("date")["after_finrobot_raw_weight"].transform("sum")
    panel["after_finrobot_weight"] = (panel["after_finrobot_raw_weight"] / raw_sum).where(raw_sum > 0, panel["initial_weight"])
    panel["date"] = panel["date"].dt.date.astype(str)
    panel.to_csv(args.panel_csv, index=False)

    print(f"Wrote {args.result_csv}")
    print(f"Wrote {args.panel_csv}")
    print(multipliers[["date", "ticker", "latest_actual_year", "finrobot_score", "finrobot_decision", "finrobot_multiplier"]].to_string(index=False))


if __name__ == "__main__":
    main()
