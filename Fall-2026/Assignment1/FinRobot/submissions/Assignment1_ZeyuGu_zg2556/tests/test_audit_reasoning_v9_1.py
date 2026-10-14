"""Offline evaluator tests; altered evidence exists only in memory, never as runs."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_reasoning_v9_1 import assess_range, original_range_data


class RangeTests(unittest.TestCase):
    def setUp(self):
        self.prompt = (ROOT / "runs/AMD_baseline_1/agents/valuation_overview/prompt.txt").read_text(encoding="utf-8")
        self.text = "Across the supplied EV/EBITDA rows aligned with AMD's 2021A through 2025A actual columns, the minimum is 17.9136x (2022A) and the maximum is 57.1197x (2023A). "

    def test_real_prompt_all_five_actual_observations(self):
        rows = original_range_data(self.prompt, "AMD")
        self.assertEqual(rows["actual_years"], [2021, 2022, 2023, 2024, 2025])
        self.assertEqual(str(rows["minimum"]), "17.9136")
        self.assertEqual(str(rows["maximum"]), "57.1197")
        atoms, _ = assess_range({"unit_id": 10, "text": self.text}, self.prompt, "AMD")
        self.assertEqual([a["classification"] for a in atoms], ["Supported"] * 3)

    def test_wrong_minimum_not_grounded(self):
        text = self.text.replace("17.9136x", "45.0000x")
        atoms, _ = assess_range({"unit_id": 10, "text": text}, self.prompt, "AMD")
        self.assertEqual([a["classification"] for a in atoms], ["Supported", "Contradicted", "Supported"])

    def test_correct_value_wrong_year_not_grounded(self):
        atoms, _ = assess_range({"unit_id": 10, "text": self.text.replace("(2022A)", "(2024A)")}, self.prompt, "AMD")
        self.assertEqual(atoms[1]["classification"], "Contradicted")

    def test_nonfinite_actual_cannot_be_ignored(self):
        with self.assertRaisesRegex(ValueError, "missing/nonfinite"):
            original_range_data(self.prompt.replace("17.9136", "nan"), "AMD")

    def test_forecast_and_peer_outlier_not_subject_extreme(self):
        prompt = self.prompt.replace("|   2026 | nan      | nan       |  31.4283 |", "|   2026 | 9999999  | nan       |  31.4283 |")
        data = original_range_data(prompt, "AMD")
        self.assertEqual(str(data["maximum"]), "57.1197")
        self.assertNotIn(2026, data["actual_years"])

    def test_ties_allow_any_tied_year(self):
        prompt = self.prompt.replace("42.1013", "17.9136")
        atoms, _ = assess_range({"unit_id": 10, "text": self.text.replace("(2022A)", "(2021A)")}, prompt, "AMD")
        self.assertEqual(atoms[1]["classification"], "Supported")

    def test_unknown_prose_and_wrong_subject_abort(self):
        for text in ("AMD valuation is high.", self.text.replace("AMD's", "NVDA's")):
            with self.assertRaisesRegex(ValueError, "Unreviewed"):
                assess_range({"unit_id": 10, "text": text}, self.prompt, "AMD")


if __name__ == "__main__":
    unittest.main()
