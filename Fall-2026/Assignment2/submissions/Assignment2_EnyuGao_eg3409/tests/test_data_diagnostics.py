from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.data_diagnostics import (
    alias_candidates,
    first_monthly_sessions,
    normalized_symbol,
    transition_point,
    weekly_price_diagnostics,
)
from shared.data_view import DataBundle, DataView
from valuation.student_valuation import (
    COLUMNS,
    _latest_four_visible_quarters,
    build_ttm_fundamentals,
    estimate_da_pct_revenue,
    filter_candidates,
    statutory_federal_tax_rate,
    tax_rate_for_dcf,
    value,
)


def synthetic_quarters() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["A"] * 4,
            "datadate": pd.to_datetime(["2020-03-31", "2020-06-30", "2020-09-30", "2020-12-31"]),
            "accepted_date": pd.to_datetime(["2020-05-01", "2020-08-01", "2020-11-01", "2021-02-01"]),
            "filing_date": pd.to_datetime(["2020-05-02", "2020-08-02", "2020-11-02", "2021-02-02"]),
            "gsector": [45] * 4,
            "EPS": [1.0] * 4,
            "revenue_per_share": [10.0] * 4,
            "fcf_per_share": [1.0] * 4,
            "ocf_per_share": [2.0] * 4,
            "capex_per_share": [0.5] * 4,
            "ebitda_margin": [0.20] * 4,
            "operating_margin": [0.15] * 4,
            "effective_tax_rate": [0.21] * 4,
            "cash_per_share": [2.0] * 4,
            "interest_debt_per_share": [5.0] * 4,
            "debt_to_capital": [0.3] * 4,
        }
    )


class TTMRegressionTests(unittest.TestCase):
    def test_dated_duplicate_wins_over_missing_filing_timestamps(self):
        frame = synthetic_quarters()
        undated = frame.iloc[[-1]].copy()
        undated["accepted_date"] = pd.NaT
        undated["filing_date"] = pd.NaT
        undated["EPS"] = 99.0

        latest = _latest_four_visible_quarters(pd.concat([frame, undated], ignore_index=True))
        repeated = _latest_four_visible_quarters(pd.concat([frame, undated], ignore_index=True))

        self.assertEqual(len(latest), 4)
        self.assertEqual(latest.loc[latest["datadate"].eq(pd.Timestamp("2020-12-31")), "EPS"].iloc[0], 1.0)
        pd.testing.assert_frame_equal(latest, repeated)

    def test_visible_quarter_eligibility_is_distinct_from_complete_aggregation(self):
        frame = synthetic_quarters()
        frame.loc[1, "EPS"] = np.nan

        latest = _latest_four_visible_quarters(frame)
        ttm = build_ttm_fundamentals(frame)

        self.assertEqual(latest["ticker"].nunique(), 1)
        self.assertEqual(len(ttm), 1)
        self.assertTrue(np.isnan(ttm.iloc[0]["eps_ttm"]))


class PriceHistoryRegressionTests(unittest.TestCase):
    def test_symbol_normalization_and_alias_candidates_are_conservative(self):
        self.assertEqual(normalized_symbol("BRK.B"), "BRKB")
        self.assertEqual(alias_candidates("BRK.B", {"BRK-B", "BRKBX", "AAPL"}), ["BRK-B"])

    def test_missing_short_and_spy_pairing_groups(self):
        dates = pd.date_range("2020-01-03", periods=80, freq="W-FRI")
        closes = pd.DataFrame(index=dates)
        closes["MISSING"] = np.nan
        closes["SHORT"] = np.nan
        closes.loc[dates[:20], "SHORT"] = np.arange(20) + 10
        closes["PAIRING"] = np.arange(80) + 100
        closes["SPY"] = np.nan
        closes.loc[dates[-40:], "SPY"] = np.arange(40) + 300

        result = weekly_price_diagnostics(closes, ["MISSING", "SHORT", "PAIRING"]).set_index("ticker")

        self.assertEqual(result.at["MISSING", "group"], "A")
        self.assertEqual(result.at["SHORT", "group"], "B")
        self.assertEqual(result.at["PAIRING", "group"], "C")


class DevelopmentGapIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = DataBundle.load()

    def test_reindexed_frame_length_is_not_price_availability(self):
        frame = DataView(self.bundle, "2021-12-31").close(["AAL"], lookback=1260)

        self.assertEqual(len(frame), 1260)
        self.assertEqual(int(frame["AAL"].notna().sum()), 0)

    def test_unpriceable_symbol_can_exist_in_development_scores(self):
        scores = pd.read_parquet(Path("data/selection/monthly_scores.parquet"))
        scores["month"] = pd.to_datetime(scores["month"])
        dev = scores.loc[scores["month"].between("2017-01-01", "2021-12-01", inclusive="both")]

        self.assertEqual(int(dev["symbol"].eq("AAL").sum()), 60)
        self.assertEqual(int((dev["symbol"].eq("OKTA") & dev["bucket"].eq("growth_tech")).sum()), 12)


class AssumptionPolicyTests(unittest.TestCase):
    def test_da_floor_requires_complete_paired_history(self):
        negative = pd.DataFrame(
            {"ebitda_margin": [0.10, 0.11, 0.12, 0.13], "operating_margin": [0.12, 0.13, 0.14, 0.15]}
        )
        incomplete = negative.copy()
        incomplete.loc[0, "ebitda_margin"] = np.nan

        self.assertEqual(estimate_da_pct_revenue(negative), 0.0)
        self.assertIsNone(estimate_da_pct_revenue(incomplete))

    def test_tax_policy_positive_zero_negative_missing_and_historical_rates(self):
        self.assertEqual(statutory_federal_tax_rate("2017-12-31"), 0.35)
        self.assertEqual(statutory_federal_tax_rate("2018-12-31"), 0.21)
        self.assertEqual(tax_rate_for_dcf(0.25, "2021-12-31"), 0.25)
        self.assertEqual(tax_rate_for_dcf(0.50, "2021-12-31"), 0.35)
        self.assertEqual(tax_rate_for_dcf(0.0, "2017-12-31"), 0.35)
        self.assertEqual(tax_rate_for_dcf(-0.10, "2021-12-31"), 0.21)
        self.assertEqual(tax_rate_for_dcf(np.nan, "2021-12-31"), 0.21)


class CoverageRampRegressionTests(unittest.TestCase):
    def test_monthly_calendar_uses_first_observed_session(self):
        dates = pd.to_datetime(["2017-01-03", "2017-01-04", "2017-02-01", "2017-02-02", "2017-03-02"])
        spy = pd.Series([1, 2, 3, 4, 5], index=dates)

        sessions = first_monthly_sessions(spy, "2017-01-01", "2017-03-31")

        self.assertEqual(list(sessions), list(pd.to_datetime(["2017-01-03", "2017-02-01", "2017-03-02"])))

    def test_threshold_dates_and_not_reached(self):
        frame = pd.DataFrame({"as_of": ["2017-01-03", "2017-02-01"], "coverage": [40, 60]})

        self.assertEqual(transition_point(frame, "coverage", 50, ">="), "2017-02-01")
        self.assertEqual(transition_point(frame, "coverage", 100, ">="), "not_reached")


class PublicSchemaRegressionTests(unittest.TestCase):
    def test_public_valuation_functions_handle_empty_inputs(self):
        candidates = ()
        result = value("2021-12-31", [], None)

        self.assertEqual(list(result.columns), COLUMNS)
        self.assertTrue(result.empty)
        self.assertIs(filter_candidates("2021-12-31", candidates, None), candidates)


if __name__ == "__main__":
    unittest.main()
