from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from analysis.engine_rejection_breakdown import classify_engine_rejection
from tests.test_valuation_models import SyntheticDataView
from valuation.finrobot import adjust_beta_blume, calculate_dcf as engine_calculate_dcf
from valuation.student_valuation import (
    COLUMNS,
    DEFAULT_DCF_CONFIG,
    LEGACY_DCF_CONFIG,
    LOCKED_DCF_CONFIG,
    _historical_growth_path,
    _historical_ev_ebitda_multiples,
    _weekly_beta,
    capex_spending_magnitude,
    dcf_valuation,
    dcf_valuation_diagnostics,
    estimate_da_pct_revenue,
    filter_candidates,
    statutory_federal_tax_rate,
    tax_rate_for_dcf,
    value,
)


def growth_frame(first: float, second: float, *, irregular: bool = False) -> pd.DataFrame:
    dates = list(pd.date_range("2019-03-31", periods=8, freq="3ME"))
    if irregular:
        dates[-1] = dates[-1] + pd.DateOffset(months=3)
    return pd.DataFrame(
        {
            "ticker": "TGT",
            "datadate": dates,
            "accepted_date": pd.to_datetime(dates) + pd.Timedelta(days=30),
            "filing_date": pd.to_datetime(dates) + pd.Timedelta(days=31),
            "revenue_per_share": [first] * 4 + [second] * 4,
        }
    )


class WeeklyPriceView:
    def __init__(self, stock: pd.Series, market: pd.Series):
        self.frame = pd.concat(
            [stock.rename("TGT"), market.rename("SPY")], axis=1, sort=True
        )

    def close(self, symbols=None, lookback=None):
        frame = self.frame.reindex(columns=list(symbols)) if symbols is not None else self.frame
        return frame.tail(lookback).copy() if lookback is not None else frame.copy()


class DCFConfigurationTests(unittest.TestCase):
    def test_configs_are_immutable_and_locked_is_default(self):
        self.assertIs(DEFAULT_DCF_CONFIG, LOCKED_DCF_CONFIG)
        with self.assertRaises(FrozenInstanceError):
            LOCKED_DCF_CONFIG.beta_min = 0.0

    def test_growth_clips_both_bounds_and_has_five_year_half_life(self):
        upper_path, upper, _ = _historical_growth_path(
            growth_frame(10.0, 20.0), config=LOCKED_DCF_CONFIG
        )
        lower_path, lower, _ = _historical_growth_path(
            growth_frame(10.0, 1.0), config=LOCKED_DCF_CONFIG
        )

        self.assertAlmostEqual(upper, 0.35)
        self.assertAlmostEqual(lower, -0.05)
        self.assertAlmostEqual(upper_path[0], upper)
        self.assertAlmostEqual(
            upper_path[5] - LOCKED_DCF_CONFIG.terminal_growth,
            (upper - LOCKED_DCF_CONFIG.terminal_growth) / 2,
        )
        self.assertNotAlmostEqual(upper_path[-1], LOCKED_DCF_CONFIG.terminal_growth)
        self.assertEqual(len(upper_path), 10)
        self.assertTrue(all(np.isfinite(lower_path)))

    def test_flat_negative_and_irregular_growth_histories(self):
        flat_path, flat, _ = _historical_growth_path(
            growth_frame(10.0, 10.0), config=LOCKED_DCF_CONFIG
        )
        negative_path, negative, _ = _historical_growth_path(
            growth_frame(10.0, 9.0), config=LOCKED_DCF_CONFIG
        )
        irregular = _historical_growth_path(
            growth_frame(10.0, 12.0, irregular=True), config=LOCKED_DCF_CONFIG
        )

        self.assertAlmostEqual(flat, 0.0)
        self.assertAlmostEqual(flat_path[0], 0.0)
        self.assertLess(negative, 0.0)
        self.assertAlmostEqual(negative_path[0], negative)
        self.assertIsNone(irregular)

    def test_terminal_growth_is_independent_engine_input(self):
        data = SyntheticDataView()
        config = replace(LOCKED_DCF_CONFIG, name="tg_test", terminal_growth=0.03)
        captured = []

        def record(inputs, wacc_override=None, tg_override=None, mid_year=False):
            captured.append(inputs)
            return engine_calculate_dcf(
                inputs,
                wacc_override=wacc_override,
                tg_override=tg_override,
                mid_year=mid_year,
            )

        with patch("valuation.student_valuation.calculate_dcf", record):
            result = dcf_valuation("2021-12-31", ["TGT"], data, config=config)

        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(captured[0].terminal_growth_rate, 0.03)
        self.assertAlmostEqual(result.iloc[0]["terminal_growth"], 0.03)
        self.assertNotAlmostEqual(captured[0].revenue_growth_rates[-1], 0.03)


class BetaHardeningTests(unittest.TestCase):
    @staticmethod
    def weekly_view(n_prices: int, stock_scale: float = 1.5, constant_market: bool = False):
        dates = pd.date_range("2020-01-03", periods=n_prices, freq="W-FRI")
        market_returns = np.zeros(n_prices) if constant_market else 0.01 + 0.002 * np.sin(np.arange(n_prices))
        stock_returns = stock_scale * market_returns
        market = pd.Series(100 * np.cumprod(1 + market_returns), index=dates)
        stock = pd.Series(50 * np.cumprod(1 + stock_returns), index=dates)
        return WeeklyPriceView(stock, market)

    def test_blume_adjustment_and_locked_beta_clip(self):
        view = self.weekly_view(60, stock_scale=6.0)
        raw, adjusted, pairs = _weekly_beta("TGT", view, LOCKED_DCF_CONFIG)

        self.assertGreaterEqual(pairs, 52)
        self.assertGreater(raw, 2.5)
        self.assertAlmostEqual(adjusted, 2.5)
        self.assertGreater(adjust_beta_blume(raw), 2.5)

    def test_exactly_52_pairs_pass_and_fewer_fail(self):
        self.assertIsNotNone(_weekly_beta("TGT", self.weekly_view(53), LOCKED_DCF_CONFIG))
        self.assertIsNone(_weekly_beta("TGT", self.weekly_view(52), LOCKED_DCF_CONFIG))

    def test_missing_prices_and_zero_market_variance_fail(self):
        dates = pd.date_range("2020-01-03", periods=60, freq="W-FRI")
        missing = WeeklyPriceView(pd.Series(np.nan, index=dates), pd.Series(100.0, index=dates))
        self.assertIsNone(_weekly_beta("TGT", missing, LOCKED_DCF_CONFIG))
        self.assertIsNone(
            _weekly_beta("TGT", self.weekly_view(60, constant_market=True), LOCKED_DCF_CONFIG)
        )

    def test_independent_weekly_resampling_does_not_require_same_daily_rows(self):
        fridays = pd.date_range("2020-01-03", periods=60, freq="W-FRI")
        mondays = fridays - pd.Timedelta(days=4)
        returns = 0.01 + 0.002 * np.sin(np.arange(60))
        stock = pd.Series(50 * np.cumprod(1 + 1.2 * returns), index=mondays)
        market = pd.Series(100 * np.cumprod(1 + returns), index=fridays)
        view = WeeklyPriceView(stock, market)

        self.assertTrue(view.frame.dropna().empty)
        self.assertIsNotNone(_weekly_beta("TGT", view, LOCKED_DCF_CONFIG))


class AccountingAndSensitivityTests(unittest.TestCase):
    def test_engine_rejection_subcause_classification(self):
        self.assertEqual(
            classify_engine_rejection("WACC−terminal growth spread 1.0% is below"),
            "unstable_gordon_spread",
        )
        self.assertEqual(
            classify_engine_rejection(
                "Terminal growth rate 0.03 must be less than WACC 0.02"
            ),
            "unstable_gordon_spread",
        )
        self.assertEqual(
            classify_engine_rejection("Equity value is non-positive (-1.0)"),
            "nonpositive_equity_value",
        )
        self.assertEqual(
            classify_engine_rejection("Steady-state terminal FCF is non-positive (-1.0)"),
            "invalid_intermediate_input",
        )
        self.assertEqual(classify_engine_rejection("unexpected"), "other")

    def test_da_floor_and_missing_pairs(self):
        negative = pd.DataFrame(
            {"ebitda_margin": [0.10] * 4, "operating_margin": [0.20] * 4}
        )
        missing = negative.copy()
        missing.loc[0, "operating_margin"] = np.nan
        self.assertEqual(estimate_da_pct_revenue(negative), 0.0)
        self.assertIsNone(estimate_da_pct_revenue(missing))

    def test_tax_fallbacks_and_clip(self):
        self.assertEqual(statutory_federal_tax_rate("2017-12-31"), 0.35)
        self.assertEqual(statutory_federal_tax_rate("2018-12-31"), 0.21)
        self.assertEqual(tax_rate_for_dcf(0.50, "2021-12-31"), 0.35)
        for raw in (0.0, -0.1, np.nan):
            self.assertEqual(tax_rate_for_dcf(raw, "2021-12-31"), 0.21)
        self.assertIn("Fiscal-year transitions", statutory_federal_tax_rate.__doc__)

    def test_capex_sign_conversion(self):
        self.assertEqual(capex_spending_magnitude([-1, -2, -3, -4]), (10.0, "negative_outflow"))
        self.assertEqual(capex_spending_magnitude([1, 2, 3, 4]), (10.0, "positive_magnitude"))
        self.assertEqual(capex_spending_magnitude([0, 0, 0, 0]), (0.0, "zero"))
        self.assertEqual(capex_spending_magnitude([-1, 2, -3, 4]), (None, "mixed_sign"))
        self.assertEqual(capex_spending_magnitude([1, 2, np.nan, 4]), (None, "missing"))

    def test_erp_recalculates_wacc_and_preserves_ordered_positive_ranges(self):
        data = SyntheticDataView()
        outputs = {}
        for erp in (0.04, 0.05, 0.06):
            config = replace(LOCKED_DCF_CONFIG, name=f"erp_{erp}", equity_risk_premium=erp)
            result = dcf_valuation("2021-12-31", ["TGT"], data, config=config)
            self.assertEqual(len(result), 1)
            row = result.iloc[0]
            self.assertGreater(row["fair_value"], 0)
            self.assertLessEqual(row["low"], row["fair_value"])
            self.assertLessEqual(row["fair_value"], row["high"])
            outputs[erp] = row
        self.assertLess(outputs[0.04]["wacc"], outputs[0.05]["wacc"])
        self.assertLess(outputs[0.05]["wacc"], outputs[0.06]["wacc"])
        self.assertGreater(outputs[0.04]["fair_value"], outputs[0.06]["fair_value"])

    def test_invalid_wacc_terminal_combination_is_diagnosed(self):
        data = SyntheticDataView()
        data.macro = lambda _series: pd.Series([0.0], index=[data.as_of])
        config = replace(
            LOCKED_DCF_CONFIG,
            name="invalid_wacc",
            equity_risk_premium=0.0,
            terminal_growth=0.05,
        )
        values, diagnostics = dcf_valuation_diagnostics(
            "2021-12-31", ["TGT"], data, config=config
        )
        self.assertTrue(values.empty)
        self.assertEqual(diagnostics.iloc[0]["primary_status"], "engine_rejected")

    def test_invalid_sensitivity_grid_is_diagnosed(self):
        data = SyntheticDataView()
        empty_grid = {"implied_prices": [[None] * 3 for _ in range(3)]}
        with patch("valuation.student_valuation.calculate_sensitivity", return_value=empty_grid):
            values, diagnostics = dcf_valuation_diagnostics(
                "2021-12-31", ["TGT"], data, config=LOCKED_DCF_CONFIG
            )
        self.assertTrue(values.empty)
        self.assertEqual(
            diagnostics.iloc[0]["primary_status"], "invalid_sensitivity_grid"
        )

    def test_diagnostics_have_exactly_one_primary_status_per_attempt(self):
        data = SyntheticDataView(price_target=False)
        values, diagnostics = dcf_valuation_diagnostics(
            "2021-12-31", ["TGT", "P0"], data, config=LEGACY_DCF_CONFIG
        )
        self.assertEqual(len(diagnostics), 2)
        self.assertEqual(diagnostics["ticker"].nunique(), 2)
        self.assertEqual(len(values), diagnostics["primary_status"].eq("valued").sum())

    def test_risk_free_and_mixed_capex_exclusions_are_explicit(self):
        no_rate = SyntheticDataView()
        no_rate.macro = lambda _series: pd.Series(dtype=float)
        _, diagnostic = dcf_valuation_diagnostics(
            "2021-12-31", ["TGT"], no_rate, config=LOCKED_DCF_CONFIG
        )
        self.assertEqual(diagnostic.iloc[0]["primary_status"], "risk_free_missing")

        mixed_capex = SyntheticDataView()
        target_rows = mixed_capex._fundamentals["ticker"].eq("TGT")
        last_indices = mixed_capex._fundamentals.loc[target_rows].tail(4).index
        mixed_capex._fundamentals.loc[last_indices, "capex_per_share"] = [-1, 1, -1, 1]
        _, diagnostic = dcf_valuation_diagnostics(
            "2021-12-31", ["TGT"], mixed_capex, config=LOCKED_DCF_CONFIG
        )
        self.assertEqual(diagnostic.iloc[0]["primary_status"], "invalid_capex")
        self.assertEqual(diagnostic.iloc[0]["secondary_reason"], "mixed_sign")

    def test_public_aggregation_interfaces_preserve_empty_input_contract(self):
        candidates = ()
        self.assertEqual(value("2021-12-31", [], None).columns.tolist(), COLUMNS)
        self.assertTrue(value("2021-12-31", [], None).empty)
        self.assertIs(filter_candidates("2021-12-31", candidates, None), candidates)

    def test_historical_multiple_events_never_exceed_as_of(self):
        data = SyntheticDataView()
        fundamentals = data.fundamentals(["TGT"])
        future = fundamentals.iloc[-1].copy()
        future["datadate"] = pd.Timestamp("2021-12-31")
        future["accepted_date"] = pd.Timestamp("2022-02-01")
        future["filing_date"] = pd.Timestamp("2022-02-02")
        fundamentals = pd.concat([fundamentals, future.to_frame().T], ignore_index=True)
        history = _historical_ev_ebitda_multiples(
            "TGT",
            fundamentals,
            data.close(["TGT"])["TGT"],
            "2021-12-31",
            lookback_years=5,
        )
        self.assertFalse(history.empty)
        self.assertLessEqual(history["date"].max(), pd.Timestamp("2021-12-31"))


if __name__ == "__main__":
    unittest.main()
