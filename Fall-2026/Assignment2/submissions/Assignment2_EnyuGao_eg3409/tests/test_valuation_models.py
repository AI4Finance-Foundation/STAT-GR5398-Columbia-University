from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from valuation.student_valuation import (
    dcf_valuation,
    historical_ev_ebitda_valuation,
    peer_pe_valuation,
)


class SyntheticDataView:
    def __init__(self, *, price_target: bool = True):
        self.as_of = pd.Timestamp("2021-12-31")
        self._members = {f"P{i}" for i in range(6)} | {"TGT"}
        quarter_dates = pd.date_range("2018-03-31", periods=16, freq="3ME")
        rows = []
        for ticker_index, ticker in enumerate(sorted(self._members)):
            for i, quarter in enumerate(quarter_dates):
                revenue = (10.0 + ticker_index * 0.2) * (1.02 ** i)
                rows.append(
                    {
                        "ticker": ticker,
                        "datadate": quarter,
                        "accepted_date": quarter + pd.Timedelta(days=30),
                        "filing_date": quarter + pd.Timedelta(days=31),
                        "gsector": 45,
                        "EPS": 1.0 + ticker_index * 0.1,
                        "revenue_per_share": revenue,
                        "fcf_per_share": revenue * 0.10,
                        "ocf_per_share": revenue * 0.14,
                        "capex_per_share": revenue * 0.04,
                        "ebitda_margin": 0.25,
                        "operating_margin": 0.18,
                        "effective_tax_rate": 0.21,
                        "cash_per_share": 2.0,
                        "interest_debt_per_share": 1.0,
                        "debt_to_capital": 0.10,
                        "pe": 9999.0,
                        "ev_multiple": 9999.0,
                    }
                )
        self._fundamentals = pd.DataFrame(rows)

        dates = pd.bdate_range("2016-01-04", "2021-12-31")
        step = np.arange(len(dates))
        spy_returns = 0.00025 + 0.002 * np.sin(step / 13)
        spy = 100 * np.cumprod(1 + spy_returns)
        prices = pd.DataFrame(index=dates)
        prices["SPY"] = spy
        target_returns = 1.2 * spy_returns + 0.0005 * np.cos(step / 17)
        prices["TGT"] = 50 * np.cumprod(1 + target_returns) if price_target else np.nan
        for i in range(6):
            peer_returns = (0.8 + i * 0.05) * spy_returns
            prices[f"P{i}"] = (40 + i * 8) * np.cumprod(1 + peer_returns)
        self._prices = prices

    def fundamentals(self, tickers=None):
        if tickers is None:
            return self._fundamentals.copy()
        return self._fundamentals.loc[self._fundamentals["ticker"].isin(set(tickers))].copy()

    def close(self, symbols=None, lookback=None):
        frame = self._prices
        if symbols is not None:
            frame = frame.reindex(columns=list(symbols))
        if lookback is not None:
            frame = frame.tail(int(lookback))
        return frame.copy()

    def last_close(self, symbols):
        result = {}
        for symbol in symbols:
            if symbol in self._prices:
                values = self._prices[symbol].dropna()
                if not values.empty:
                    result[symbol] = float(values.iloc[-1])
        return result

    def macro(self, series_id):
        if series_id != "DGS10":
            raise KeyError(series_id)
        return pd.Series([2.0], index=[self.as_of], name="DGS10")

    def members(self):
        return set(self._members)


class ValuationModelTests(unittest.TestCase):
    def setUp(self):
        self.data = SyntheticDataView()

    def test_dcf_uses_historical_inputs_and_sensitivity_range(self):
        result = dcf_valuation("2021-12-31", ["TGT"], self.data)

        self.assertEqual(len(result), 1)
        row = result.iloc[0]
        self.assertEqual(row["method"], "dcf")
        self.assertGreater(row["fair_value"], 0)
        self.assertLessEqual(row["low"], row["fair_value"])
        self.assertGreaterEqual(row["high"], row["fair_value"])
        self.assertAlmostEqual(row["risk_free_rate"], 0.02)
        self.assertAlmostEqual(row["tax_rate"], 0.21)
        self.assertAlmostEqual(row["da_pct_revenue"], 0.07)
        self.assertGreaterEqual(row["beta_weeks"], 52)

    def test_historical_ev_ebitda_builds_own_point_in_time_band(self):
        result = historical_ev_ebitda_valuation("2021-12-31", ["TGT"], self.data)

        self.assertEqual(len(result), 1)
        row = result.iloc[0]
        self.assertEqual(row["method"], "historical_ev_ebitda")
        self.assertGreaterEqual(row["history_count"], 8)
        self.assertLessEqual(row["multiple_p25"], row["multiple_median"])
        self.assertLessEqual(row["multiple_median"], row["multiple_p75"])
        self.assertLessEqual(row["low"], row["fair_value"])
        self.assertLessEqual(row["fair_value"], row["high"])

    def test_peer_pe_uses_same_sector_trailing_pe_and_excludes_target(self):
        result = peer_pe_valuation("2021-12-31", ["TGT"], self.data)

        self.assertEqual(len(result), 1)
        row = result.iloc[0]
        self.assertEqual(row["method"], "peer_pe")
        self.assertEqual(row["peer_count"], 6)
        self.assertGreater(row["eps_ttm"], 0)
        self.assertLessEqual(row["low"], row["fair_value"])
        self.assertLessEqual(row["fair_value"], row["high"])

    def test_unpriceable_target_is_omitted_by_all_methods(self):
        data = SyntheticDataView(price_target=False)

        self.assertTrue(dcf_valuation("2021-12-31", ["TGT"], data).empty)
        self.assertTrue(historical_ev_ebitda_valuation("2021-12-31", ["TGT"], data).empty)
        self.assertTrue(peer_pe_valuation("2021-12-31", ["TGT"], data).empty)

    def test_nonpositive_target_metrics_are_omitted(self):
        negative_ebitda = SyntheticDataView()
        target = negative_ebitda._fundamentals["ticker"].eq("TGT")
        negative_ebitda._fundamentals.loc[target, "ebitda_margin"] = -0.10
        self.assertTrue(
            historical_ev_ebitda_valuation(
                "2021-12-31", ["TGT"], negative_ebitda
            ).empty
        )

        negative_eps = SyntheticDataView()
        target = negative_eps._fundamentals["ticker"].eq("TGT")
        negative_eps._fundamentals.loc[target, "EPS"] = -1.0
        self.assertTrue(peer_pe_valuation("2021-12-31", ["TGT"], negative_eps).empty)

    def test_unverified_database_multiple_columns_do_not_affect_results(self):
        baseline_ev = historical_ev_ebitda_valuation("2021-12-31", ["TGT"], self.data)
        baseline_pe = peer_pe_valuation("2021-12-31", ["TGT"], self.data)
        self.data._fundamentals["pe"] = -123456.0
        self.data._fundamentals["ev_multiple"] = -123456.0

        changed_ev = historical_ev_ebitda_valuation("2021-12-31", ["TGT"], self.data)
        changed_pe = peer_pe_valuation("2021-12-31", ["TGT"], self.data)

        pd.testing.assert_frame_equal(baseline_ev, changed_ev)
        pd.testing.assert_frame_equal(baseline_pe, changed_pe)


if __name__ == "__main__":
    unittest.main()
