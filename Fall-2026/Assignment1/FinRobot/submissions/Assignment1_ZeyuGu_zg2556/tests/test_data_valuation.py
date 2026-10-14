"""Mathematical regression fixtures ONLY; these are not experiment evidence."""
import os
from pathlib import Path
import sys
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
UPSTREAM_SRC = os.environ.get("FINROBOT_UPSTREAM_SRC")
if not UPSTREAM_SRC:
    raise RuntimeError("Set FINROBOT_UPSTREAM_SRC to patched finrobot_equity/core/src")
sys.path.insert(0, UPSTREAM_SRC)
import pandas as pd
from independent_valuation import ev_ebitda, dcf, recompute, run_engine, reconciliation
from target_price_sensitivity import compute_grid, summarize
from modules.financial_data_processor import normalize_forecast_config, calculate_growth_and_forecasts, extract_historical_metrics_from_api_data


class DataValuationRegressionTests(unittest.TestCase):
    def history(self, year=2026):
        return pd.DataFrame({"metrics": ["Revenue", "Cost of Operations", "SG&A", "Contribution Profit", "Contribution Margin", "EBITDA", "EBITDA Margin", "SG&A Margin", "Revenue Growth", "EPS", "PE Ratio"],
                             f"{year}A": [1000, 600, 100, 400, "40.0%", 220, "22.0%", "10.0%", "N/A", 2, 15]})

    def config(self, year=2026):
        return {"revenue_base_year": f"{year}A", "revenue_growth_assumptions": {"2025E": .05, "2026E": .06, "2027E": .04},
                "margin_improvement": {"Contribution Margin": .01, "EBITDA Margin": .01}, "sga_margin_change": -.005}

    def test_forecasts_follow_latest_actual_and_preserve_growth_values(self):
        history, config = self.history(), self.config()
        normalized = normalize_forecast_config(history, config)
        self.assertEqual(normalized["revenue_growth_assumptions"], {"2027E": .05, "2028E": .06, "2029E": .04})
        self.assertEqual(config["revenue_growth_assumptions"], {"2025E": .05, "2026E": .06, "2027E": .04})
        self.assertEqual(normalize_forecast_config(history, normalized), normalized)
        forecast = calculate_growth_and_forecasts(history, config)
        self.assertNotIn("2026E", forecast)
        revenue = forecast.set_index("metrics").loc["Revenue", "2029E"]
        self.assertAlmostEqual(revenue, 1000 * 1.05 * 1.06 * 1.04)

    def test_legacy_2024_origin_remains_unchanged(self):
        normalized = normalize_forecast_config(self.history(2024), self.config(2024))
        self.assertEqual(normalized["revenue_growth_assumptions"], self.config(2024)["revenue_growth_assumptions"])

    def test_backtest_refuses_future_actuals(self):
        config = self.config()
        config["revenue_base_year"] = "2025A"
        with self.assertRaisesRegex(ValueError, "truncate"):
            calculate_growth_and_forecasts(self.history(), config)

    def test_quarterly_collapse_rejected(self):
        data = {"income_statement": pd.DataFrame([{"date": "2026-03-31", "year": 2026}, {"date": "2026-06-30", "year": 2026}])}
        with self.assertRaisesRegex(ValueError, "quarterly"):
            extract_historical_metrics_from_api_data(data)

    def context(self, ebitda=211.125):
        analysis = [{"metrics": "Revenue", "2026A": "1000", "2027E": "1050"},
                    {"metrics": "EBITDA", "2026A": "200", "2027E": str(ebitda)},
                    {"metrics": "EBITDA Margin", "2026A": "20.0%", "2027E": f"{ebitda / 1050 * 100:.1f}%"}]
        return {"inputs": {"ticker": "REGRESSION_FIXTURE", "run_id": "unit-test-only", "shares_outstanding": 20, "current_price": 95},
                "analysis": analysis, "metrics": {row["metrics"]: row for row in analysis}, "key_metrics": [],
                "period": "2027E", "ebitda": ebitda, "target_multiple": 12.0, "historical_multiple_std": 3., "sources": {}}

    def test_independent_ev_and_dcf_reconcile_with_actual_engine(self):
        for ebitda in (211.125, -25):
            with self.subTest(ebitda=ebitda):
                context = self.context(ebitda)
                independent = recompute(context)
                engine = run_engine(context, Path(UPSTREAM_SRC))
                records = reconciliation(context, independent, engine)
                self.assertEqual(len(records), 12)
                self.assertTrue(all(row["verdict"] == "MATCH" for row in records))
                engine_ev = next(result for result in engine["results"] if result["method"] == "EV/EBITDA")
                self.assertEqual(engine_ev["assumptions"]["EBITDA Period"], "2027E")
                self.assertIn("FALLBACK", engine_ev["assumptions"]["Multiple Source"])

    def test_zero_shock_preserves_value_despite_displayed_margin_rounding(self):
        context = self.context()
        rows = compute_grid(context, [-.05, 0, .05], [-.02, 0, .02])
        center = next(row for row in rows if row["revenue_relative_delta"] == row["margin_delta_fraction"] == 0)
        self.assertAlmostEqual(center["target_price"], center["base_target_price"])
        self.assertNotEqual(center["native_rounding_price_difference"], 0)
        base = center["target_price"]
        plus_five = next(row for row in rows if row["revenue_relative_delta"] == .05 and row["margin_delta_fraction"] == 0)
        self.assertAlmostEqual(plus_five["target_price"], 1.05 * base)
        plus_pp = next(row for row in rows if row["revenue_relative_delta"] == 0 and row["margin_delta_fraction"] == .02)
        self.assertAlmostEqual(plus_pp["scenario_ebitda"] - context["ebitda"], 1050 * .02)
        self.assertEqual(summarize(context, rows)["local_revenue_elasticity"], 1)

    def test_prices_floor_at_zero_and_dcf_denominator_is_guarded(self):
        self.assertEqual(ev_ebitda(-100, 12, 3, 20)["target_price"], 0)
        with self.assertRaisesRegex(ValueError, "WACC"):
            dcf(100, 20, {"wacc": .02, "terminal_growth": .025})


if __name__ == "__main__":
    unittest.main()
