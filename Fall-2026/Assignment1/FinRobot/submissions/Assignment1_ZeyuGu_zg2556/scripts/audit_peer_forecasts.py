"""Reconcile native peer projections from genuine frozen FMP raw responses.

This diagnoses a separate, inconsistent forecast path without changing any saved
baseline prompt or table. It is an audit, not a replacement peer forecast model.
"""
import argparse
import csv
import json
from pathlib import Path
import statistics
from urllib.parse import parse_qs, urlsplit
from independent_valuation import write_csv


def raw_income(snapshot: Path, ticker: str):
    for meta_path in (snapshot / "http").glob("*/response.json"):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        url = urlsplit(meta["source_url"])
        if url.path == "/stable/income-statement" and parse_qs(url.query).get("symbol") == [ticker] and meta["status_code"] == 200:
            return json.loads((meta_path.parent / "body.bin").read_bytes()), meta, meta_path.parent.name
    raise ValueError(f"Missing raw peer income statement for {ticker}")


def read_table(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for ticker in ("NVDA", "AMD", "INTC", "AAPL", "GOOGL"):
        snapshot = args.snapshots_root / ticker
        matrix_path = snapshot / "analysis" / "peer_ebitda_comparison.csv"
        if not matrix_path.exists():
            continue
        matrix = read_table(matrix_path)
        year_column = next(iter(matrix[0]))
        last_row = max(matrix, key=lambda row: int(row[year_column]))
        projection_year = int(last_row[year_column])
        manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
        for projected_company in list(matrix[0])[1:]:
            raw, metadata, key = raw_income(snapshot, projected_company)
            historical = sorted([(int(item["date"][:4]), float(item["ebitda"])) for item in raw if item.get("ebitda") is not None])
            if len(historical) < 2 or any(value == 0 for _, value in historical[:-1]):
                raise ValueError("Native mean-growth audit needs at least two nonzero historical bases")
            growths = [historical[i][1] / historical[i-1][1] - 1 for i in range(1, len(historical))]
            average = statistics.mean(growths)
            recomputed = historical[-1][1] * (1 + average)
            actual_peer_value = float(last_row[projected_company])
            own_path = args.snapshots_root / projected_company / "analysis" / "financial_metrics_and_forecasts.csv"
            own_value = None
            if own_path.exists():
                own = {row["metrics"]: row for row in read_table(own_path)}
                candidate = own["EBITDA"].get(f"{projection_year}E")
                if candidate:
                    own_value = float(candidate)
            rows.append({"report_ticker": ticker, "report_snapshot_run_id": manifest["run_id"],
                "projected_company": projected_company, "source_url": metadata["source_url"], "raw_response_key": key,
                "fetched_at_utc": metadata["fetched_at_utc"], "raw_body_sha256": metadata["body_sha256"],
                "historical_years": json.dumps([year for year, _ in historical]),
                "historical_ebitda_values": json.dumps([value for _, value in historical]),
                "historical_yoy_growth_fractions": json.dumps(growths), "arithmetic_mean_yoy_growth": average,
                "latest_company_actual_year": historical[-1][0], "peer_projection_label": projection_year,
                "label_gap_years": projection_year - historical[-1][0], "growth_steps_actually_applied": 1,
                "native_peer_ebitda": actual_peer_value, "independent_peer_formula_ebitda": recomputed,
                "formula_difference": recomputed - actual_peer_value,
                "own_company_forecast_same_year": own_value,
                "peer_minus_own_forecast": actual_peer_value - own_value if own_value is not None else None,
                "peer_relative_to_own_forecast_pct": (actual_peer_value / own_value - 1) * 100 if own_value else None,
                "finding": "Peer table uses arithmetic-mean historical EBITDA growth; main forecast uses revenue and gross-margin-minus-SG&A. Projection label is latest year across all peers +1, even when a company has an older last actual."})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output, rows)
    print(f"Audited {len(rows)} actual peer projections without changing the baseline tables.")


if __name__ == "__main__":
    main()
