"""Offline checker tests; altered test strings are not experimental reports."""
import sys
import unittest
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_reasoning_v9 import assess_changed, independently_compute, prompt_revenue, scalar_matches


class IndependentCAGRTests(unittest.TestCase):
    def setUp(self):
        self.prompt = (ROOT / "runs/NVDA_baseline_1/agents/risks/prompt.txt").read_text(encoding="utf-8")

    def test_actual_endpoints_and_full_table_are_distinct(self):
        expected = {"NVDA": ("68.30", "37.49"), "AMD": ("20.49", "13.59"), "INTC": ("-9.57", "-3.59")}
        for ticker, (actual, full) in expected.items():
            prompt = (ROOT / f"runs/{ticker}_baseline_1/agents/risks/prompt.txt").read_text(encoding="utf-8")
            table = prompt_revenue(prompt)
            historical = independently_compute(table["values"], table["actual"][0], table["actual"][-1])
            complete = independently_compute(table["values"], table["years"][0], table["years"][-1])
            self.assertTrue(scalar_matches(actual, historical["percent"]))
            self.assertTrue(scalar_matches(full, complete["percent"]))
            self.assertEqual(historical["elapsed_years"], 4)
            self.assertEqual(complete["elapsed_years"], 7)

    def test_wrong_arithmetic_is_not_supported(self):
        unit = {"unit_id": 0, "text": "The supplied Revenue CAGR cell is 68.3%; the 2022A to 2026A actual endpoints imply 37.49% CAGR. "}
        atoms, _ = assess_changed(unit, self.prompt, "NVDA")
        self.assertEqual([a["classification"] for a in atoms], ["Supported", "Contradicted"])

    def test_forecast_as_actual_is_contradiction(self):
        unit = {"unit_id": 0, "text": "The supplied Revenue CAGR cell is 68.3%; the 2022A to 2029E actual endpoints imply 37.49% CAGR. "}
        atoms, _ = assess_changed(unit, self.prompt, "NVDA")
        self.assertEqual(atoms[1]["classification"], "Contradicted")

    def test_unknown_changed_prose_aborts(self):
        with self.assertRaisesRegex(ValueError, "Unreviewed changed"):
            assess_changed({"unit_id": 0, "text": "Revenue has a strong CAGR."}, self.prompt, "NVDA")

    def test_zero_negative_missing_endpoints_decline(self):
        for start in (Decimal(0), Decimal(-1)):
            self.assertIsNone(independently_compute({"2020A": start, "2024A": Decimal(2)}, "2020A", "2024A"))
        self.assertIsNone(independently_compute({"2020A": Decimal(2)}, "2020A", "2024A"))

    def test_rounding_is_display_precision_not_broad_tolerance(self):
        self.assertTrue(scalar_matches("20.49", "20.491251645171"))
        self.assertFalse(scalar_matches("20.50", "20.491251645171"))
        self.assertTrue(scalar_matches("-9.57", "-9.56684999067"))


if __name__ == "__main__":
    unittest.main()
