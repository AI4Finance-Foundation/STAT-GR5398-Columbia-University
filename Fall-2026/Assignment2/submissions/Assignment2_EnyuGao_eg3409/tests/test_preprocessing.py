from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from shared.data_view import DataBundle, DataView
from shared.guard import LookAheadGuardError, student_call
from valuation.student_valuation import build_ttm_fundamentals


class TTMPreprocessingTests(unittest.TestCase):
    def test_latest_four_distinct_visible_quarters_are_aggregated(self):
        dates = pd.to_datetime(["2020-03-31", "2020-06-30", "2020-09-30", "2020-12-31", "2021-03-31"])
        frame = pd.DataFrame(
            {
                "ticker": ["A"] * 5 + ["A"] + ["B"] * 3,
                "datadate": list(dates) + [pd.Timestamp("2021-03-31")] + list(dates[-3:]),
                "accepted_date": pd.to_datetime(
                    ["2020-05-01", "2020-08-01", "2020-11-01", "2021-02-01", "2021-05-01"]
                    + ["2021-05-02"]
                    + ["2020-11-01", "2021-02-01", "2021-05-01"]
                ),
                "filing_date": pd.to_datetime(
                    ["2020-05-02", "2020-08-02", "2020-11-02", "2021-02-02", "2021-05-02"]
                    + ["2021-05-03"]
                    + ["2020-11-02", "2021-02-02", "2021-05-02"]
                ),
                "gsector": [45] * 6 + [20] * 3,
                "EPS": [1, 2, 3, 4, 5, 6, 1, 1, 1],
                "revenue_per_share": [10, 10, 10, 10, 10, 12, 5, 5, 5],
                "fcf_per_share": [1, 1, 1, 1, 1, 1, 1, 1, 1],
                "ocf_per_share": [2, 2, 2, 2, 2, 2, 2, 2, 2],
                "capex_per_share": [0.5] * 9,
                "ebitda_margin": [0.2, 0.2, 0.2, 0.2, 0.2, 0.25, 0.1, 0.1, 0.1],
                "operating_margin": [0.15] * 9,
                "effective_tax_rate": [0.21] * 9,
                "cash_per_share": [2] * 9,
                "interest_debt_per_share": [5] * 9,
                "debt_to_capital": [0.3] * 9,
            }
        )

        result = build_ttm_fundamentals(frame).set_index("ticker")

        self.assertEqual(list(result.index), ["A"])
        self.assertEqual(result.at["A", "quarter_count"], 4)
        self.assertEqual(result.at["A", "eps_ttm"], 15)
        self.assertEqual(result.at["A", "revenue_ttm"], 42)
        self.assertEqual(result.at["A", "ebitda_ttm"], 9)
        self.assertEqual(result.at["A", "net_debt_per_share"], 3)

    def test_missing_quarterly_component_is_not_imputed(self):
        frame = pd.DataFrame(
            {
                "ticker": ["A"] * 4,
                "datadate": pd.date_range("2020-03-31", periods=4, freq="3ME"),
                "EPS": [1, 1, 1, 1],
                "revenue_per_share": [10, 10, 10, 10],
                "fcf_per_share": [1, np.nan, 1, 1],
                "ocf_per_share": [2, 2, 2, 2],
                "capex_per_share": [0.5, 0.5, 0.5, 0.5],
                "ebitda_margin": [0.2] * 4,
                "operating_margin": [0.15] * 4,
            }
        )

        result = build_ttm_fundamentals(frame).iloc[0]

        self.assertTrue(np.isnan(result["fcf_ttm"]))
        self.assertEqual(result["eps_ttm"], 4)


class PointInTimeIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = DataBundle.load()

    def test_data_view_hides_leakage_columns_and_future_filings(self):
        as_of = pd.Timestamp("2021-07-26")
        frame = DataView(self.bundle, as_of).fundamentals(["AAPL"])

        self.assertFalse({"trade_price", "adj_close_q", "y_return"} & set(frame.columns))
        accepted = pd.to_datetime(frame["accepted_date"], errors="coerce").dropna()
        filings = pd.to_datetime(frame["filing_date"], errors="coerce").dropna()
        self.assertTrue(accepted.le(as_of).all())
        self.assertTrue(filings.le(as_of).all())
        self.assertLess(pd.to_datetime(frame["datadate"]).max(), pd.Timestamp("2021-06-30"))

        visible_next_day = DataView(self.bundle, "2021-07-27").fundamentals(["AAPL"])
        self.assertEqual(pd.to_datetime(visible_next_day["datadate"]).max(), pd.Timestamp("2021-06-30"))

    def test_guard_blocks_direct_file_io_but_allows_data_view(self):
        with student_call():
            with self.assertRaises(LookAheadGuardError):
                open("data/DATA_MANIFEST.json")
            visible = DataView(self.bundle, "2021-12-31").fundamentals(["AAPL"])
        self.assertFalse(visible.empty)


if __name__ == "__main__":
    unittest.main()
