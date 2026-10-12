"""Synthetic unit fixtures only; these tests are not experimental observations."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import stability_audit_v2 as module
from audit_io import write_json


def fixture():
    sections = {section: {"text": "SYNTHETIC TEST heading"} for section in module.FIELDS}
    sections["valuation_overview"]["text"] = "We rate it Buy with a 12-month USD target of 150."
    sections["risks"]["text"] = "Rival chips can erode share. Demand could fall. Suppliers may be late."
    raw = {"ratings": [{"section": "valuation_overview", "source_unit_id": 0, "label": "Buy", "source_kind": "writer_recommendation"}],
           "targets": [{"section": "valuation_overview", "source_unit_id": 0, "amount_text": "150", "amount": 150.0,
                        "currency": "USD", "horizon": "12-month", "source_kind": "writer_recommendation"}],
           "risks": [{"source_unit_id": 2, "taxonomy": "supply_capacity"}, {"source_unit_id": 0, "taxonomy": "competition"},
                     {"source_unit_id": 1, "taxonomy": "demand_cyclicality"}]}
    return sections, raw


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.sections, self.raw = fixture()

    def test_exact_text_is_resolved_and_ordered_independent_of_model_order(self):
        checked = module.validate_extraction(self.raw, self.sections)
        self.assertEqual(checked["writer_target"]["amount"], 150)
        self.assertEqual(checked["writer_rating"], "buy")
        self.assertEqual([row["taxonomy"] for row in checked["top_three_risks"]], ["competition", "demand_cyclicality", "supply_capacity"])
        for row in checked["all_risks"] + checked["ratings"] + checked["targets"]:
            self.assertEqual(self.sections[row["section"]]["text"][row["source_start"]:row["source_end"]], row["source_quote"])

    def test_unit_ids_reject_missing_boolean_and_out_of_range(self):
        for value in (-1, 200, True, "0"):
            with self.subTest(value=value):
                candidate = deepcopy(self.raw)
                candidate["risks"][0]["source_unit_id"] = value
                with self.assertRaises(ValueError):
                    module.validate_extraction(candidate, self.sections)

    def test_schema_has_no_quote_fields(self):
        schema = module.build_agent().output_type.model_json_schema()
        self.assertNotIn("source_quote", json.dumps(schema))
        self.assertIn("source_unit_id", json.dumps(schema))
        self.assertEqual(set(json.loads(module.build_prompt(self.sections))["report_sections"]), set(module.FIELDS))

    def test_partial_amount_and_unstated_context_rejected(self):
        self.raw["targets"][0].update(amount=50, amount_text="50")
        with self.assertRaisesRegex(ValueError, "exact numeric token"):
            module.validate_extraction(self.raw, self.sections)
        self.raw["targets"][0].update(amount=150, amount_text="150", currency="EUR")
        with self.assertRaisesRegex(ValueError, "currency"):
            module.validate_extraction(self.raw, self.sections)

    def test_attributed_rating_target_do_not_become_writer_recommendations(self):
        self.raw["ratings"][0]["source_kind"] = "attributed_external"
        self.raw["targets"][0]["source_kind"] = "ambiguous"
        checked = module.validate_extraction(self.raw, self.sections)
        self.assertIsNone(checked["writer_rating"])
        self.assertIsNone(checked["writer_target"])

    def test_same_unit_ties_deterministic_and_duplicates_rejected(self):
        self.raw["risks"] = [{"source_unit_id": 0, "taxonomy": "supply_capacity"}, {"source_unit_id": 0, "taxonomy": "competition"}]
        checked = module.validate_extraction(self.raw, self.sections)
        self.assertEqual([row["taxonomy"] for row in checked["all_risks"]], ["competition", "supply_capacity"])
        self.raw["risks"].append(self.raw["risks"][0])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            module.validate_extraction(self.raw, self.sections)


class ResumeTests(unittest.TestCase):
    def test_failed_attempt_preserved_success_reused_and_contract_changes_rejected(self):
        sections, raw = fixture()
        agent = module.build_agent()
        schema = agent.output_type.model_json_schema()
        with tempfile.TemporaryDirectory(prefix="synthetic_stability_v2_") as temp:
            args = SimpleNamespace(output=Path(temp), offline_extractions=None, max_tokens=100)
            item = {"manifest": {"run_id": "SYNTHETIC"}, "sections": sections}
            calls = []
            async def fake_call(_agent, prompt, **kwargs):
                folder = kwargs["directory"]
                folder.mkdir()
                calls.append(folder)
                result = deepcopy(raw)
                if len(calls) == 1:
                    result["risks"][0]["source_unit_id"] = 900
                write_json(folder / "request.json", {"evidence_prompt": prompt, "instructions": module.INSTRUCTIONS,
                    "model": "synthetic", "temperature": 0, "max_output_tokens": 100, "output_schema": schema})
                write_json(folder / "response.json", {"structured_output": result, "cost_usd": 0})
                write_json(folder / "http_response.json", {"status_code": 200})
                write_json(folder / "wire_response.json", {"id": "synthetic"+str(len(calls)), "choices": [{"message": {"content": json.dumps(result)}}]})
                return result
            with patch.object(module, "call_agent", side_effect=fake_call):
                with self.assertRaisesRegex(ValueError, "outside"):
                    asyncio.run(module.select_extraction(args, item, {"openai_model": "synthetic"}, None, agent, schema))
                self.assertTrue((Path(temp) / "calls/SYNTHETIC/attempt_001/validation_error.json").exists())
                checked, selected, _ = asyncio.run(module.select_extraction(args, item, {"openai_model": "synthetic"}, None, agent, schema))
                self.assertEqual(selected.name, "attempt_002")
                self.assertEqual(checked["writer_rating"], "buy")
                asyncio.run(module.select_extraction(args, item, {"openai_model": "synthetic"}, None, agent, schema))
                self.assertEqual(len(calls), 2)
                args.max_tokens = 101
                with self.assertRaisesRegex(ValueError, "contract changed"):
                    asyncio.run(module.select_extraction(args, item, {"openai_model": "synthetic"}, None, agent, schema))

    def test_offline_without_validated_attempt_never_calls_provider(self):
        sections, _ = fixture()
        with tempfile.TemporaryDirectory(prefix="synthetic_stability_offline_") as temp:
            root = Path(temp)
            (root / "output/calls/SYNTHETIC").mkdir(parents=True)
            args = SimpleNamespace(output=root / "output", offline_extractions=root / "source", max_tokens=100)
            item = {"manifest": {"run_id": "SYNTHETIC"}, "sections": sections}
            with patch.object(module, "call_agent") as call:
                with self.assertRaisesRegex(ValueError, "no API call allowed"):
                    asyncio.run(module.select_extraction(args, item, {"openai_model": "synthetic"}, None, None, {}))
                call.assert_not_called()


if __name__ == "__main__":
    unittest.main()
