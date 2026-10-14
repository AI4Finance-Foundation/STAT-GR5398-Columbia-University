"""Synthetic isolated unit fixtures only; never experimental report/audit evidence."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import asyncio

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from audit_io import sha256,write_json
from grounding_audit import FIELDS,metrics,paragraphs,source_units,validate_audit
from stability_audit import validate_extraction,validate_independent_runs,overlap,numeric_summary
import grounding_audit


class GroundingUnitTests(unittest.TestCase):
    def setUp(self):
        self.text="Revenue was 10.\n\nOutlook"
        self.evidence="TEST FIXTURE\nRevenue: 10"
        self.claim={"claim":"Revenue was 10.","claim_type":"quantitative","classification":"Supported",
            "evidence":[{"line_start":2,"line_end":2}],"reason":"Literal fixture value.",
            "numeric_status":"correct","fact_key":"fixture|revenue|test|10|units"}
        self.result={"units":[{"unit_id":0,"claims":[self.claim],"nonfactual_reason":""},
            {"unit_id":1,"claims":[],"nonfactual_reason":"Heading only."}]}

    def test_all_text_and_zero_claim_heading_retained(self):
        rows,coverage=validate_audit(self.result,self.text,self.evidence,{"section":"test"})
        self.assertEqual(len(rows),1)
        self.assertEqual(len(coverage),2)
        self.assertEqual(self.text[rows[0]["source_start"]:rows[0]["source_end"]],"Revenue was 10.")
        self.assertEqual(metrics(rows)["grounding_rate"],1)
        self.assertEqual(json.loads(rows[0]["evidence"])[0]["quote"],"Revenue: 10")

    def test_missing_paragraph_rejected(self):
        self.result["units"].pop()
        with self.assertRaisesRegex(ValueError,"omitted"): validate_audit(self.result,self.text,self.evidence,{"section":"test"})

    def test_duplicated_unit_rejected(self):
        self.result["units"][1]["unit_id"]=0
        with self.assertRaisesRegex(ValueError,"duplicated"): validate_audit(self.result,self.text,self.evidence,{"section":"test"})

    def test_posthoc_or_wrong_line_evidence_rejected(self):
        self.claim["evidence"][0]["line_end"]=3
        with self.assertRaisesRegex(ValueError,"original prompt"): validate_audit(self.result,self.text,self.evidence,{"section":"test"})

    def test_supported_with_wrong_number_rejected(self):
        self.claim["numeric_status"]="incorrect"
        with self.assertRaisesRegex(ValueError,"incorrect number"): validate_audit(self.result,self.text,self.evidence,{"section":"test"})

    def test_zero_claim_rate_is_na(self):
        self.assertEqual(metrics([])["grounding_rate"],"N/A")
        self.assertEqual(metrics([])["numeric_error_rate"],"N/A")

    def test_multiline_paragraph_offsets(self):
        text="\nTitle\n\nFirst line\nsecond line.\n\n"
        p=paragraphs(text)
        self.assertEqual([x["text"] for x in p],["Title","First line\nsecond line."])
        self.assertTrue(all(text[x["start"]:x["end"]]==x["text"] for x in p))

    def test_sentence_units_keep_decimals_abbreviations_and_whitespace(self):
        text='Revenue was 10.5 billion.  U.S. Inc. demand grew.\nNext line!  More.\n\nHeading'
        units=source_units(text)
        self.assertEqual([x["text"] for x in units],['Revenue was 10.5 billion.  ','U.S. Inc. demand grew.','Next line!  ','More.','Heading'])
        self.assertTrue(all(text[x['start']:x['end']]==x['text'] for x in units))
        covered=set().union(*(set(range(x['start'],x['end'])) for x in units))
        self.assertTrue(all(i in covered for i,c in enumerate(text) if not c.isspace()))


class StabilityExtractionTests(unittest.TestCase):
    def setUp(self):
        self.sections={s:{"text":"TEST FIXTURE"} for s in FIELDS}
        self.sections["valuation_overview"]["text"]="We rate it Buy with a 12-month USD target of 150."
        self.sections["risks"]["text"]="Rival chips can erode share. Demand could fall. Suppliers may be late."
        self.output={"ratings":[{"section":"valuation_overview","source_quote":self.sections["valuation_overview"]["text"],"label":"Buy","source_kind":"writer_recommendation"}],
            "targets":[{"section":"valuation_overview","source_quote":self.sections["valuation_overview"]["text"],"amount_text":"150","amount":150.0,"currency":"USD","horizon":"12-month","source_kind":"writer_recommendation"}],
            "risks":[{"source_quote":"Suppliers may be late.","taxonomy":"supply_capacity"},
                {"source_quote":"Rival chips can erode share.","taxonomy":"competition"},{"source_quote":"Demand could fall.","taxonomy":"demand_cyclicality"}]}

    def test_first_mention_risk_order_not_model_order(self):
        result=validate_extraction(self.output,self.sections)
        self.assertEqual([x["taxonomy"] for x in result["top_three_risks"]],["competition","demand_cyclicality","supply_capacity"])
        self.assertEqual(result["writer_rating"],"buy")
        self.assertEqual(result["writer_target"]["amount"],150)

    def test_attributed_fields_do_not_become_writer_rating(self):
        self.output["ratings"][0]["source_kind"]="attributed_external"
        self.output["targets"][0]["source_kind"]="attributed_external"
        result=validate_extraction(self.output,self.sections)
        self.assertIsNone(result["writer_rating"])
        self.assertIsNone(result["writer_target"])

    def test_numeric_substring_not_valid_amount(self):
        self.output["targets"][0].update(amount_text="50",amount=50)
        with self.assertRaisesRegex(ValueError,"exact numeric quote"): validate_extraction(self.output,self.sections)

    def test_no_inference_on_missing_recommendation(self):
        self.output["ratings"]=[]
        self.output["targets"]=[]
        result=validate_extraction(self.output,self.sections)
        self.assertEqual(result["writer_target_status"],"not_stated")

    def test_risk_metrics_and_na(self):
        self.assertEqual(overlap(["a","b","c"],["c","b","a"]),(1,-1))
        self.assertEqual(overlap(["a"],["a"]),(1,"N/A"))
        self.assertEqual(overlap([],[]),("N/A","N/A"))
        self.assertEqual(numeric_summary([1,2,3])["sample_variance"],1)
        self.assertEqual(numeric_summary([None,None,None])["sample_variance"],"N/A")


def create_synthetic_run(parent,name):
    """Only used inside TemporaryDirectory; clearly not a live provider call."""
    run=parent/name
    analysis=run/"output/TEST/analysis"
    analysis.mkdir(parents=True)
    hashes={}
    prompt="SYNTHETIC UNIT TEST EVIDENCE"
    for section,field in FIELDS.items():
        text="SYNTHETIC UNIT TEST "+section
        path=analysis/(section+".txt")
        path.write_text(text,encoding="utf-8")
        hashes[path.name]=sha256(path.read_bytes())
        folder=run/"agents"/section
        folder.mkdir(parents=True)
        (folder/"prompt.txt").write_text(prompt,encoding="utf-8")
        instruction="SYNTHETIC UNIT TEST INSTRUCTION"
        (folder/"instructions.txt").write_text(instruction,encoding="utf-8")
        call_id=name+"/"+section
        write_json(folder/"request.json",{"call_id":call_id,"instructions":instruction,"instructions_sha256":sha256(instruction),
            "evidence_prompt":prompt,"evidence_sha256":sha256(prompt),"model":"test-only","temperature":0.7,
            "max_output_tokens":10,"timestamp_utc":"2000-01-01T00:00:00+00:00"})
        write_json(folder/"response.json",{"call_id":call_id,"structured_output":{field:text},"usage":{"total_tokens":10,"input_tokens":5,"output_tokens":5},
            "attempts":1,"cost_usd":0,"runtime_seconds":1,"completed_at_utc":"2000-01-01T00:00:01+00:00"})
        write_json(folder/"http_response.json",{"status_code":200,"request_id":"synthetic-request-"+call_id})
        write_json(folder/"wire_request.json",{"model":"test-only","temperature":0.7,"messages":[{"role":"user","content":prompt}],"section":section})
        write_json(folder/"wire_response.json",{"id":"synthetic-response-"+call_id,"choices":[{"message":{"content":json.dumps({field:text})}}]})
    write_json(run/"manifest.json",{"run_id":name,"ticker":"TEST","state":"generated","variant":"synthetic_unit_fixture_only",
        "upstream_sha":"test","input_hashes":{"test":"test"},"evidence_sha256":sha256(prompt),"model":"test-only",
        "temperature":0.7,"max_tokens":10,"analysis_hashes":hashes})
    return run


class IndependenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="synthetic_unit_evaluation_")
        self.paths=[create_synthetic_run(Path(self.temp.name),"test"+str(i)) for i in range(3)]
    def tearDown(self): self.temp.cleanup()
    def test_distinct_wire_calls_allowed(self):
        self.assertEqual(len(validate_independent_runs(self.paths)),3)
    def test_copied_response_rejected(self):
        target=self.paths[1]/"agents/tagline/wire_response.json"
        target.write_bytes((self.paths[0]/"agents/tagline/wire_response.json").read_bytes())
        with self.assertRaisesRegex(ValueError,"copied wire response"): validate_independent_runs(self.paths)
    def test_changed_actual_request_rejected(self):
        target=self.paths[1]/"agents/tagline/wire_request.json"
        data=json.loads(target.read_text()); data["temperature"]=0.1; write_json(target,data)
        with self.assertRaisesRegex(ValueError,"Actual model request changed"): validate_independent_runs(self.paths)
    def test_changed_frozen_input_rejected(self):
        target=self.paths[1]/"manifest.json"
        data=json.loads(target.read_text());data["input_hashes"]={"different":"input"};write_json(target,data)
        with self.assertRaisesRegex(ValueError,"Frozen inputs"): validate_independent_runs(self.paths)
    def test_wire_and_typed_response_must_match(self):
        target=self.paths[1]/"agents/tagline/wire_response.json"
        data=json.loads(target.read_text());data["choices"][0]["message"]["content"]='{"tagline":"changed"}';write_json(target,data)
        with self.assertRaisesRegex(ValueError,"actual wire response"): validate_independent_runs(self.paths)
    def test_fewer_than_three_rejected(self):
        with self.assertRaisesRegex(ValueError,"three"): validate_independent_runs(self.paths[:2])


class ResumeTests(unittest.TestCase):
    def test_verifier_costs_include_archive_and_unresolved_attempts(self):
        with tempfile.TemporaryDirectory(prefix="synthetic_costs_test_") as temp:
            root=Path(temp)
            for folder,call,cost in [('verifier/tagline','current',.01),('attempt_history/verifier/tagline/attempt_001','prior',.02),('attempt_history/verifier/risks/attempt_001','unresolved',None)]:
                p=root/folder;p.mkdir(parents=True)
                write_json(p/'request.json',{'call_id':call})
                if cost is not None:write_json(p/'response.json',{'call_id':call,'cost_usd':cost,'runtime_seconds':1,'usage':{'input_tokens':2,'output_tokens':3}})
            responses,unresolved,records=grounding_audit.verifier_accounting(root)
            self.assertEqual(sum(r['cost_usd'] for r in responses),.03)
            self.assertEqual(len(unresolved),1)
            self.assertEqual(sum(r['archived_attempt'] for r in records),1)

    def test_revalidation_skips_success_and_preserves_failed_attempt(self):
        with tempfile.TemporaryDirectory(prefix="synthetic_resume_test_") as temp:
            root=Path(temp)
            args=SimpleNamespace(run=root/"run",evidence_run=None,config=None,budget="test",ledger=root/"ledger",
                output=root/"audit",resume=False,max_tokens=100,repair_attempts=1)
            source={s:{"text":"Heading","evidence":"Fixture data","metadata":{"ticker":"TEST","run_id":"TEST", "section":s,
                "source_sha256":sha256("Heading"),"evidence_prompt_sha256":sha256("Fixture data")}} for s in FIELDS}
            manifest={"ticker":"TEST","run_id":"TEST","model":"test-only"}
            class Schema:
                @staticmethod
                def model_json_schema(): return {"type":"object"}
            class DummyBudget:
                policy={"model":"test-only"}
                def __init__(self,*a):pass
                def close(self):pass
            calls=[]
            async def fake_call(agent,prompt,**kwargs):
                path=kwargs["directory"];path.mkdir(parents=True)
                calls.append(path)
                # First attempt deliberately omits a unit. Paid production
                # behavior is tested without any real provider or saved experiment.
                output={"units":[]} if len(calls)==1 else {"units":[{"unit_id":0,"claims":[],"nonfactual_reason":"Fixture heading"}]}
                write_json(path/"request.json",{"instructions":grounding_audit.INSTRUCTIONS,"model":"test-only","temperature":0,"max_output_tokens":100,"evidence_prompt":prompt})
                write_json(path/"response.json",{"structured_output":output,"cost_usd":0,"runtime_seconds":1,"usage":{"input_tokens":1,"output_tokens":1}})
                return output
            with patch.object(grounding_audit,"load_inputs",return_value=(manifest,source)),patch.object(grounding_audit,"build_agent",return_value=SimpleNamespace(output_type=Schema)),patch.object(grounding_audit,"credentials",return_value={"openai_model":"test-only","openai_base_url":"https://example.invalid"}),patch.object(grounding_audit,"Budget",DummyBudget),patch.object(grounding_audit,"call_agent",side_effect=fake_call):
                asyncio.run(grounding_audit.run_audit(args))
                self.assertEqual(len(calls),9)
                self.assertTrue((args.output/"calls/tagline/attempt_001/validation_error.json").exists())
                self.assertEqual(json.loads((args.output/"calls/tagline/selected_attempt.json").read_text())["directory"],"attempt_002")
                args.resume=True
                asyncio.run(grounding_audit.run_audit(args))
                self.assertEqual(len(calls),9)
                args.max_tokens=101
                with self.assertRaisesRegex(ValueError,"Resume rejected"): asyncio.run(grounding_audit.run_audit(args))


if __name__=="__main__": unittest.main()
