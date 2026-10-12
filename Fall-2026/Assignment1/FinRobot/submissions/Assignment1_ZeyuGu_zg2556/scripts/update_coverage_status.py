"""Rebuild coverage status from saved real evidence; performs no network calls.

Research selection/classification fields are retained from coverage_universe.csv.
The intended five-company universe is distinct from successfully frozen coverage.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACTIVE_SNAPSHOTS = {"NVDA": "NVDA", "AMD": "AMD", "INTC": "INTC_news_v2",
                    "AAPL": "AAPL_news_v2", "GOOGL": "GOOGL",
                    "AVGO":"AVGO_public_v1", "QCOM":"QCOM_public_v1"}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def observed(ticker, entitlements):
    status = entitlements[ticker]["status"]
    result = {"fmp_income_http_status": status,
              "fmp_data_status": "HTTP_200_FINANCIAL_SNAPSHOT_FROZEN" if status == 200 else "HTTP_402_ENTITLEMENT_BLOCKED",
              "entitlement_evidence": "evidence/stable_endpoint_preflight/universe_entitlements.json",
              "fmp_news_status": "HTTP_402_NOT_ENTITLED" if ticker in ACTIVE_SNAPSHOTS else "NOT_CAPTURED_FOR_BLOCKED_TICKER",
              "peer_tickers": "", "peer_count": 0, "actual_peer_tickers": "", "actual_peer_count": 0,
              "snapshot_directory": "", "snapshot_run_id": "", "snapshot_manifest_sha256": "",
              "price_performance_status": "NOT_CAPTURED_PENDING_FINANCIAL_ACCESS",
              "wrds_status": "NOT_NEEDED_FOR_CURRENT_FMP_SNAPSHOT" if ticker in ACTIVE_SNAPSHOTS else "PENDING_USER_CONNECTION_AND_ENTITLEMENT_PROBE"}
    if ticker not in ACTIVE_SNAPSHOTS:
        result["coverage_status"] = "PLANNED_FORMAL_COMPANY_NOT_RUN"
        return result
    path = ROOT / "snapshots" / ACTIVE_SNAPSHOTS[ticker]
    manifest = read_json(path / "manifest.json")
    public = ticker in {"AVGO","QCOM"}
    summary = read_json(path / "analysis" / "analysis_summary.json")
    peers = summary["peer_tickers"] if public else manifest["parameters"]["peers"]
    price = read_json(path / "analysis" / "price_performance.json")
    news = read_json(path / "analysis" / "company_news.json")
    result.update(peer_tickers=";".join(peers), peer_count=len(peers), actual_peer_tickers=";".join(peers),
                  actual_peer_count=len(peers), snapshot_directory=path.relative_to(ROOT).as_posix(),
                  snapshot_run_id=manifest["run_id"],
                  snapshot_manifest_sha256=hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest(),
                  snapshot_as_of_utc=manifest["as_of_utc"] if public else manifest["parameters"]["as_of_utc"],
                  coverage_status="BASELINE_AND_FINAL_REAL_EIGHT_AGENT_RUNS_FROM_FROZEN_INPUTS",
                  financial_source="SEC 10-K comparative annual statements, public adapter" if public else "FMP frozen annual statements",
                  market_source="Yahoo public chart API" if public else "FMP frozen historical price",
                  wrds_status="EXCLUDED_FROM_MODEL_AND_SUBMISSION; private local exports only",
                  alternate_news_count=len(news),
                  alternate_news_source="Literal recent SEC 10-Q income statement table; not press/news coverage" if public else "Official issuer earnings release; curated single article, not comprehensive news feed",
                  news_window_days=summary["news_days_back"] if public else manifest["parameters"]["news_days_back"],
                  price_performance_status="HTTP_200_FROZEN_COMMON_WINDOW_CLOSE_PRICE_RETURN",
                  price_window_start=price["actual_start"], price_window_end=price["actual_end"],
                  start_close_usd=price["start_close"], end_close_usd=price["end_close"],
                  close_price_return_pct=price["price_return_pct"],
                  price_return_formula="100 * (end_close / start_close - 1)",
                  price_adjustment_caveat="Raw provider close field; no dividend reinvestment or adjustment applied by this analysis; provider split/adjustment convention not independently verified; not total return",
                  price_evidence=(path / "analysis" / "price_performance.json").relative_to(ROOT).as_posix())
    expected = 100 * (price["end_close"] / price["start_close"] - 1)
    if abs(expected - price["price_return_pct"]) > 1e-9:
        raise ValueError(f"Saved return does not reconcile: {ticker}")
    return result


def main():
    entitlements = {row["ticker"]: row for row in read_json(ROOT / "evidence" / "stable_endpoint_preflight" / "universe_entitlements.json")}
    with (ROOT / "coverage_universe.csv").open(encoding="utf-8-sig", newline="") as handle:
        formal = list(csv.DictReader(handle))
    for row in formal:
        row["research_desirable_peer_tickers"] = row.get("research_desirable_peer_tickers", row["peer_tickers"])
        row["research_desirable_peer_count"] = len(row["research_desirable_peer_tickers"].split(";"))
        row["universe_role"] = "FORMAL_SAME_INDUSTRY_REAL_RUN"
        row.update(observed(row["ticker"], entitlements))
        row["selection_price_caveat"] = "All five observed raw-close returns are positive (+11.37% to +187.87%); varied magnitudes, not balanced winners/losers. Two providers and unverified split conventions limit comparison."
        if row['ticker']=='AVGO': row['peer_basis']='NVDA and AMD: AI/data-center compute and accelerator platforms; network/custom-silicon exposure differs. Frozen FMP peers used because subject FMP access is denied.'
        if row['ticker']=='QCOM': row['peer_basis']='AMD and INTC: client and edge processor platforms (Snapdragon versus x86); handset/licensing mix differs. Frozen FMP peers used because subject FMP access is denied.'
        for variant in ('baseline_1','final_v6'):
            run=ROOT/'runs'/f"{row['ticker']}_{variant}"
            row[variant+'_run_state']=read_json(run/'manifest.json')['state']
            row[variant+'_report']=(run/'output'/row['ticker']/'report'/f"Audited_Equity_Report_{row['ticker']}.html").relative_to(ROOT).as_posix()
    save_csv(ROOT / "coverage_universe.csv", formal)
    onboarding = []
    for ticker in ("NVDA", "AMD", "INTC", "AAPL", "GOOGL"):
        snapshot = ROOT / "snapshots" / ACTIVE_SNAPSHOTS[ticker]
        manifest = read_json(snapshot / "manifest.json")
        is_formal = ticker in {"NVDA", "AMD", "INTC"}
        row = {"ticker": ticker, "company_name": manifest["parameters"]["company_name"],
               "universe_role": "ONBOARDING_REUSES_FORMAL_SNAPSHOT" if is_formal else "ONBOARDING_ONLY_NOT_FORMAL_SEMICONDUCTOR_COVERAGE",
               "formal_industry_count_eligible": is_formal,
               "peer_mapping_purpose": "Semiconductor compute comparables" if is_formal else "API demonstration only; cross-industry peer choices are not economically comparable and do not qualify as formal same-industry peers",
               "result_reuse": "Same snapshot reused for onboarding and formal baseline; not an additional independent generation" if is_formal else "Separate onboarding baseline; excluded from formal three-company verifier comparison"}
        row.update(observed(ticker, entitlements))
        onboarding.append(row)
    save_csv(ROOT / "onboarding_universe.csv", onboarding)
    print("Updated five formal real runs (three FMP, two public SEC adapters) and five onboarding rows from saved evidence.")


if __name__ == "__main__":
    main()
