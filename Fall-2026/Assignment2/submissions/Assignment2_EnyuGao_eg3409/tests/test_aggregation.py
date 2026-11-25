from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from run_portfolio import build_parser
from selection.strategy import Prediction
from tests.test_valuation_models import SyntheticDataView
from valuation.student_valuation import (
    COLUMNS,
    FILTER_RULE,
    MAX_METHOD_SPREAD_FOR_FILTER,
    MIN_METHODS_FOR_FILTER,
    OVERVALUATION_THRESHOLD,
    filter_candidates,
    filter_candidates_by_rule,
    filter_removals,
    valuation_details,
    value,
)


def method_frame(method: str, fair: float, low: float, high: float, price: float = 100.0):
    return pd.DataFrame(
        [{
            "ticker": "TGT",
            "method": method,
            "fair_value": fair,
            "low": low,
            "high": high,
            "current_price": price,
        }]
    )


class AggregationTests(unittest.TestCase):
    def test_three_methods_use_median_and_exact_public_schema(self):
        with (
            patch(
                "valuation.student_valuation.dcf_valuation",
                return_value=method_frame("dcf", 80.0, 70.0, 90.0),
            ),
            patch(
                "valuation.student_valuation.historical_ev_ebitda_valuation",
                return_value=method_frame("historical_ev_ebitda", 120.0, 110.0, 130.0),
            ),
            patch(
                "valuation.student_valuation.peer_pe_valuation",
                return_value=method_frame("peer_pe", 100.0, 90.0, 110.0),
            ),
        ):
            details = valuation_details("2021-12-31", ["TGT"], object())
            public = value("2021-12-31", ["TGT"], object())

        self.assertEqual(public.columns.tolist(), COLUMNS)
        self.assertEqual(len(public), 1)
        row = details.iloc[0]
        self.assertEqual(row["fair_value"], 100.0)
        self.assertEqual(row["low"], 90.0)
        self.assertEqual(row["high"], 110.0)
        self.assertEqual(row["mispricing"], 0.0)
        self.assertEqual(row["method_spread"], 1.5)
        self.assertEqual(row["method_count"], 3)
        self.assertEqual(row["filter_decision"], "keep")

    def test_one_method_is_valued_but_not_filter_eligible(self):
        empty = pd.DataFrame()
        with (
            patch(
                "valuation.student_valuation.dcf_valuation",
                return_value=method_frame("dcf", 60.0, 50.0, 70.0),
            ),
            patch("valuation.student_valuation.historical_ev_ebitda_valuation", return_value=empty),
            patch("valuation.student_valuation.peer_pe_valuation", return_value=empty),
        ):
            details = valuation_details("2021-12-31", ["TGT"], object())

        row = details.iloc[0]
        self.assertEqual(row["method_spread"], 1.0)
        self.assertEqual(row["method_count"], 1)
        self.assertFalse(row["filter_eligible"])
        self.assertTrue(row["overvaluation_flag"])
        self.assertEqual(row["filter_decision"], "keep_insufficient_or_disagreeing_methods")

    def test_invalid_standalone_rows_are_not_aggregated(self):
        invalid = method_frame("dcf", np.nan, 1.0, 2.0)
        with (
            patch("valuation.student_valuation.dcf_valuation", return_value=invalid),
            patch("valuation.student_valuation.historical_ev_ebitda_valuation", return_value=pd.DataFrame()),
            patch("valuation.student_valuation.peer_pe_valuation", return_value=pd.DataFrame()),
        ):
            result = value("2021-12-31", ["TGT"], object())
        self.assertTrue(result.empty)
        self.assertEqual(result.columns.tolist(), COLUMNS)

    def test_filter_removes_only_supported_overvalued_growth_names(self):
        self.assertEqual(FILTER_RULE, "absolute")
        candidates = (
            Prediction("REMOVE", "growth_tech", 9.0, 1.0),
            Prediction("DISAGREE", "growth_tech", 8.0, 1.0),
            Prediction("ONE", "growth_tech", 7.0, 1.0),
            Prediction("FAIR", "growth_tech", 6.0, 1.0),
            Prediction("UNVALUED", "growth_tech", 5.0, 1.0),
            Prediction("OTHER", "value", 4.0, 1.0),
        )
        details = pd.DataFrame(
            [
                {
                    "ticker": "REMOVE", "filter_decision": "remove_overvalued",
                    "method_count": MIN_METHODS_FOR_FILTER,
                    "method_spread": MAX_METHOD_SPREAD_FOR_FILTER,
                    "mispricing": OVERVALUATION_THRESHOLD - 0.01,
                    "filter_eligible": True,
                },
                {
                    "ticker": "DISAGREE",
                    "filter_decision": "keep_insufficient_or_disagreeing_methods",
                    "mispricing": -0.80,
                    "filter_eligible": False,
                },
                {
                    "ticker": "ONE",
                    "filter_decision": "keep_insufficient_or_disagreeing_methods",
                    "mispricing": -0.80,
                    "filter_eligible": False,
                },
                {
                    "ticker": "FAIR", "filter_decision": "keep",
                    "mispricing": -0.10, "filter_eligible": True,
                },
            ]
        )
        with patch(
            "valuation.student_valuation.valuation_details", return_value=details
        ) as mocked:
            kept = filter_candidates("2021-12-31", candidates, object())

        self.assertEqual([row.symbol for row in kept], [
            "DISAGREE", "ONE", "FAIR", "UNVALUED", "OTHER"
        ])
        self.assertIs(kept[0], candidates[1])
        self.assertEqual(
            mocked.call_args.args[1],
            ["REMOVE", "DISAGREE", "ONE", "FAIR", "UNVALUED"],
        )

    def test_high_disagreement_candidate_is_retained(self):
        candidate = Prediction("WIDE", "growth_tech", 9.0, 1.0)
        details = pd.DataFrame([{
            "ticker": "WIDE",
            "mispricing": -0.90,
            "method_count": 3,
            "method_spread": MAX_METHOD_SPREAD_FOR_FILTER + 0.01,
            "filter_eligible": False,
            "filter_decision": "keep_insufficient_or_disagreeing_methods",
        }])
        with patch("valuation.student_valuation.valuation_details", return_value=details):
            kept = filter_candidates("2021-12-31", (candidate,), object())
        self.assertEqual(kept, (candidate,))
        self.assertIs(kept[0], candidate)

    def test_filter_preserves_input_container_identity_order_and_objects(self):
        first = Prediction("FIRST", "growth_tech", 9.0, 1.0)
        removed = Prediction("REMOVE", "growth_tech", 8.0, 1.0)
        last = Prediction("LAST", "value", 7.0, 1.0)
        details = pd.DataFrame([
            {
                "ticker": "FIRST", "mispricing": 0.10,
                "filter_eligible": True, "filter_decision": "keep",
            },
            {
                "ticker": "REMOVE", "mispricing": -0.50,
                "filter_eligible": True, "filter_decision": "remove_overvalued",
            },
        ])
        with patch("valuation.student_valuation.valuation_details", return_value=details):
            tuple_result = filter_candidates(
                "2021-12-31", (first, removed, last), object()
            )
            list_result = filter_candidates(
                "2021-12-31", [first, removed, last], object()
            )
        self.assertIsInstance(tuple_result, tuple)
        self.assertIsInstance(list_result, list)
        self.assertEqual(tuple_result, (first, last))
        self.assertEqual(list_result, [first, last])
        self.assertIs(tuple_result[0], first)
        self.assertIs(tuple_result[1], last)
        self.assertIs(list_result[0], first)
        self.assertIs(list_result[1], last)

    def test_quantile_rule_exact_removals_and_small_pool(self):
        details = pd.DataFrame({
            "ticker": [f"T{i}" for i in range(10)],
            "mispricing": [-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1],
            "filter_eligible": [True] * 10,
        })
        self.assertEqual(filter_removals(details, rule="quantile"), {"T0", "T1"})
        self.assertEqual(filter_removals(details.head(4), rule="quantile"), set())

        candidates = tuple(
            Prediction(ticker, "growth_tech", 10.0 - i, 1.0)
            for i, ticker in enumerate(details["ticker"])
        )
        audit_rows = []
        with patch("valuation.student_valuation.valuation_details", return_value=details):
            kept = filter_candidates_by_rule(
                "2021-12-31", candidates, object(), rule="quantile",
                audit_rows=audit_rows,
            )
        self.assertEqual([row.symbol for row in kept], [f"T{i}" for i in range(2, 10)])
        self.assertEqual(len(audit_rows), 1)
        audit = audit_rows[0]
        self.assertEqual(audit["growth_tech_candidates"], 10)
        self.assertEqual(audit["eligible_candidates"], 10)
        self.assertEqual(audit["expected_removed_count"], 2)
        self.assertEqual(audit["removed_count"], 2)
        self.assertAlmostEqual(audit["removed_fraction"], 0.20)
        self.assertTrue(audit["all_removed_eligible"])
        self.assertTrue(audit["rule_count_check"])

    def test_absolute_weekly_audit_has_correct_sign_and_denominator(self):
        candidates = (
            Prediction("REMOVE", "growth_tech", 9.0, 1.0),
            Prediction("KEEP", "growth_tech", 8.0, 1.0),
            Prediction("WIDE", "growth_tech", 7.0, 1.0),
            Prediction("OTHER", "value", 6.0, 1.0),
        )
        details = pd.DataFrame([
            {
                "ticker": "REMOVE", "mispricing": OVERVALUATION_THRESHOLD - 0.01,
                "filter_eligible": True,
            },
            {
                "ticker": "KEEP", "mispricing": OVERVALUATION_THRESHOLD,
                "filter_eligible": True,
            },
            {
                "ticker": "WIDE", "mispricing": -0.90,
                "filter_eligible": False,
            },
        ])
        audit_rows = []
        with patch("valuation.student_valuation.valuation_details", return_value=details):
            kept = filter_candidates_by_rule(
                "2021-12-31", candidates, object(), rule="absolute",
                audit_rows=audit_rows,
            )

        self.assertEqual([row.symbol for row in kept], ["KEEP", "WIDE", "OTHER"])
        audit = audit_rows[0]
        self.assertEqual(audit["input_candidates"], 4)
        self.assertEqual(audit["growth_tech_candidates"], 3)
        self.assertEqual(audit["eligible_candidates"], 2)
        self.assertEqual(audit["threshold_eligible_candidates"], 1)
        self.assertEqual(audit["removed_count"], 1)
        self.assertAlmostEqual(audit["removed_fraction"], 1 / 3)
        self.assertTrue(audit["all_removed_eligible"])
        self.assertEqual(audit["absolute_threshold"], OVERVALUATION_THRESHOLD)
        self.assertTrue(audit["all_removed_below_threshold"])
        self.assertLess(audit["removed_mispricing_max"], OVERVALUATION_THRESHOLD)

    def test_quantile_audit_does_not_add_absolute_threshold(self):
        details = pd.DataFrame({
            "ticker": [f"T{i}" for i in range(5)],
            "mispricing": [-0.10, 0.00, 0.10, 0.20, 0.30],
            "filter_eligible": [True] * 5,
        })
        candidates = tuple(
            Prediction(ticker, "growth_tech", 10.0 - i, 1.0)
            for i, ticker in enumerate(details["ticker"])
        )
        audit_rows = []
        with patch("valuation.student_valuation.valuation_details", return_value=details):
            kept = filter_candidates_by_rule(
                "2021-12-31", candidates, object(), rule="quantile",
                audit_rows=audit_rows,
            )
        self.assertEqual([row.symbol for row in kept], ["T1", "T2", "T3", "T4"])
        self.assertFalse(audit_rows[0]["all_removed_below_minus_20"])
        self.assertTrue(audit_rows[0]["all_removed_eligible"])

    def test_filter_rule_cli_default_and_selection(self):
        parser = build_parser()
        default = parser.parse_args(["--period", "dev", "--valuation"])
        quantile = parser.parse_args([
            "--period", "dev", "--valuation", "--filter-rule", "quantile",
        ])
        self.assertEqual(default.filter_rule, "absolute")
        self.assertEqual(quantile.filter_rule, "quantile")

    def test_real_synthetic_outputs_are_finite_ordered_and_deterministic(self):
        data = SyntheticDataView()
        first = value("2021-12-31", ["TGT"], data)
        second = value("2021-12-31", ["TGT"], data)

        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(len(first), 1)
        row = first.iloc[0]
        self.assertGreater(row["fair_value"], 0)
        self.assertLessEqual(row["low"], row["fair_value"])
        self.assertLessEqual(row["fair_value"], row["high"])
        self.assertTrue(np.isfinite(row["mispricing"]))
        self.assertGreaterEqual(row["method_spread"], 1.0)


if __name__ == "__main__":
    unittest.main()
