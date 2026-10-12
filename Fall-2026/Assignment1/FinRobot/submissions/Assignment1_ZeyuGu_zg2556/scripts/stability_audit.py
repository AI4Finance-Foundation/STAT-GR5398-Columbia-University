"""Audit three independent eight-agent runs on identical frozen evidence and settings.

No numeric recommendation is inferred when the actual prose does not state one.
FMP headline values are reported separately and never counted as LLM stability.
"""
from __future__ import annotations
import argparse
import asyncio
import itertools
import json
import math
from pathlib import Path
import re
import shutil
import statistics
from typing import Literal
from audit_io import Budget, call_agent, credentials, sha256, utc_now, write_json
from grounding_audit import FIELDS, csv_write, load_inputs, read_json

TAXONOMY = ("competition", "demand_cyclicality", "customer_concentration", "supply_capacity",
            "export_controls_geopolitics", "regulation_litigation", "technology_execution",
            "pricing_margin", "financial_liquidity", "valuation_market", "acquisitions_integration",
            "cybersecurity_ip", "macro_currency", "other")
RATING_MAP = {"strong buy":"strong_buy","buy":"buy","outperform":"buy","overweight":"buy",
              "hold":"hold","neutral":"hold","market perform":"hold","equal weight":"hold",
              "equal-weight":"hold","underperform":"sell","underweight":"sell","sell":"sell","strong sell":"strong_sell"}
INSTRUCTIONS = """Extract observations from the EXACT eight report sections supplied.
You are a measurement extractor, not an analyst. No browsing or prior knowledge.
List EVERY explicitly stated rating/recommendation and per-share target price with exact
source_quote, section, verbatim label/amount_text and stated horizon/currency.
Do not infer a rating from optimism, thesis, price upside or valuation; no stated rating -> [].
Do not infer a target from a multiple, valuation EV, current price or arithmetic; absent -> [].
Classify source_kind as writer_recommendation only when the report explicitly recommends
it itself, or attributed_external when it cites analysts, market consensus, FMP or others.
If attribution is ambiguous use ambiguous, never writer_recommendation. Preserve all
conflicting values; no cherry-picking. amount_text is an exact numeric substring, excluding
currency symbols but preserving commas; amount is that explicit numeric value (no conversion).
Currency must be explicit in the quote/context; otherwise use not_stated. Horizon is exact
stated horizon or not_stated. rating label must be an exact standard rating phrase.
For risks: list EVERY substantive distinct risk assertion in the risks section, with a
specific exact source_quote (unique within that section) and one primary taxonomy category:
""" + ", ".join(TAXONOMY) + """.
Keep risks as statements of exposure/uncertainty, not generic prose headings alone. Do not
invent risks or let financial evidence change what the generated prose actually says.
The program selects the first three distinct categories by first textual mention; you do
not rank importance. Exact quotes must be unique substrings of their stated source section.
"""


def build_agent():
    from pydantic import BaseModel
    from agents import Agent
    class Rating(BaseModel):
        section: str
        source_quote: str
        label: str
        source_kind: Literal["writer_recommendation", "attributed_external", "ambiguous"]
    class Target(BaseModel):
        section: str
        source_quote: str
        amount_text: str
        amount: float
        currency: str
        horizon: str
        source_kind: Literal["writer_recommendation", "attributed_external", "ambiguous"]
    class Risk(BaseModel):
        source_quote: str
        taxonomy: Literal["competition", "demand_cyclicality", "customer_concentration", "supply_capacity",
            "export_controls_geopolitics", "regulation_litigation", "technology_execution", "pricing_margin",
            "financial_liquidity", "valuation_market", "acquisitions_integration", "cybersecurity_ip", "macro_currency", "other"]
    class Extraction(BaseModel):
        ratings: list[Rating]
        targets: list[Target]
        risks: list[Risk]
    return Agent(name="IndependentStabilityExtractor",instructions=INSTRUCTIONS,output_type=Extraction)


def unique_span(text,quote):
    if not quote.strip():
        raise ValueError("Extraction quote must identify a unique actual source span")
    matches=list(re.finditer(re.escape(quote),text))
    if not matches:
        # A model often capitalizes a quoted mid-sentence clause. Resolve only
        # this typographic change, without altering words/numbers/punctuation.
        matches=list(re.finditer(re.escape(quote),text,re.IGNORECASE))
    if len(matches)!=1:
        raise ValueError("Extraction quote must identify a unique actual source span")
    return matches[0].start(),matches[0].end()


def validate_extraction(output,sections):
    if hasattr(output,"model_dump"): output=output.model_dump()
    ratings,targets,risks=[],[],[]
    for field in ("ratings","targets"):
        for record in output[field]:
            section=record["section"]
            if section not in FIELDS: raise ValueError("Unknown extraction section")
            start,end=unique_span(sections[section]["text"],record["source_quote"])
            result=dict(record,source_start=start,source_end=end,
                extractor_quote=record['source_quote'],source_quote=sections[section]['text'][start:end])
            if record["source_kind"] not in {"writer_recommendation","attributed_external","ambiguous"}:
                raise ValueError("Unknown source kind")
            if field=="ratings":
                label=record["label"].lower().strip()
                if label not in RATING_MAP or not re.search(r"\b"+re.escape(label)+r"\b",record["source_quote"].lower()):
                    raise ValueError("Nonliteral or nonstandard rating; record it manually without inferring")
                result["normalized_rating"]=RATING_MAP[label]
                ratings.append(result)
            else:
                token=record["amount_text"]
                try: number=float(token.replace(",",""))
                except ValueError: raise ValueError("Target amount is not a literal number") from None
                numeric_tokens=re.findall(r"(?<!\w)[-+]?\d[\d,]*(?:\.\d+)?",record["source_quote"])
                if token not in numeric_tokens or not math.isfinite(number) or number!=record["amount"] or number<0:
                    raise ValueError("Target value does not match exact numeric quote")
                if not record["currency"].strip() or not record["horizon"].strip():
                    raise ValueError("Missing explicit currency/horizon status")
                targets.append(result)
    for risk in output["risks"]:
        if risk["taxonomy"] not in TAXONOMY: raise ValueError("Unknown canonical risk category")
        start,end=unique_span(sections["risks"]["text"],risk["source_quote"])
        risks.append(dict(risk,section="risks",source_start=start,source_end=end,
            extractor_quote=risk['source_quote'],source_quote=sections['risks']['text'][start:end]))
    ordered=sorted(risks,key=lambda x:x["source_start"])
    top=[]
    for risk in ordered:
        if risk["taxonomy"] not in {x["taxonomy"] for x in top}:
            top.append(dict(risk,rank=len(top)+1))
            if len(top)==3: break
    writer_ratings={r["normalized_rating"] for r in ratings if r["source_kind"]=="writer_recommendation"}
    writer_targets={(r["amount"],r["currency"].upper(),r["horizon"].lower()) for r in targets if r["source_kind"]=="writer_recommendation"}
    return {"ratings":ratings,"targets":targets,"all_risks":ordered,"top_three_risks":top,
        "writer_rating":next(iter(writer_ratings)) if len(writer_ratings)==1 else None,
        "writer_rating_status":"explicit_unique" if len(writer_ratings)==1 else "conflicting" if writer_ratings else "not_stated",
        "writer_target":dict(zip(["amount","currency","horizon"],next(iter(writer_targets)))) if len(writer_targets)==1 else None,
        "writer_target_status":"explicit_unique" if len(writer_targets)==1 else "conflicting" if writer_targets else "not_stated",
        "risk_selection_method":"first textual mention of first three distinct taxonomy categories; not importance ranking"}


def validate_independent_runs(paths):
    if len(paths)<3: raise ValueError("At least three independent generations required")
    loaded=[]
    run_ids,response_ids,request_ids,call_ids,response_hashes=set(),set(),set(),set(),set()
    signature=None
    wire_by_section={}
    for path in paths:
        path=Path(path).resolve()
        manifest,sections=load_inputs(path)
        if manifest.get("variant")=="option_A_verifier":
            raise ValueError("Stability needs independent writer generations, not revised derivatives")
        if manifest["run_id"] in run_ids: raise ValueError("Repeated run ID")
        run_ids.add(manifest["run_id"])
        this={k:manifest[k] for k in ["ticker","upstream_sha","input_hashes","evidence_sha256","model","temperature","max_tokens"]}
        if signature is None: signature=this
        elif this!=signature: raise ValueError("Frozen inputs/model/parameters differ across runs")
        proof=[]
        costs=[]
        for section in FIELDS:
            folder=path/"agents"/section
            request,response,http,wire=map(read_json,[folder/"request.json",folder/"response.json",folder/"http_response.json",folder/"wire_response.json"])
            if http.get("status_code")!=200: raise ValueError("Generation lacks a successful HTTP completion")
            response_id=wire.get("id")
            request_id=http.get("request_id")
            call_id=request.get("call_id")
            wire_hash=sha256((folder/"wire_response.json").read_bytes())
            if not response_id or response_id in response_ids or wire_hash in response_hashes:
                raise ValueError("Missing/reused provider response ID or copied wire response")
            if request_id and request_id in request_ids: raise ValueError("Reused provider HTTP request ID")
            if not call_id or call_id in call_ids or call_id!=response.get("call_id") or manifest["run_id"] not in call_id:
                raise ValueError("Missing/reused call ID or run mismatch")
            if not response.get("completed_at_utc") or not request.get("timestamp_utc"):
                raise ValueError("Missing generation timestamps")
            if response["completed_at_utc"] < request["timestamp_utc"]:
                raise ValueError("Generation completion predates request")
            if response.get("attempts")!=1 or response.get("runtime_seconds",0)<=0 or response.get("usage",{}).get("total_tokens",0)<=0:
                raise ValueError("Generation lacks actual call usage/runtime evidence")
            try:
                wire_structured=json.loads(wire["choices"][0]["message"]["content"])
            except (KeyError,IndexError,TypeError,json.JSONDecodeError):
                raise ValueError("Wire response lacks parsable typed Chat Completions content") from None
            if wire_structured!=response.get("structured_output"):
                raise ValueError("Recorded typed output differs from actual wire response")
            if request.get("evidence_sha256")!=sections[section]["metadata"]["evidence_prompt_sha256"]:
                raise ValueError("Generation prompt mismatch")
            instruction=(folder/"instructions.txt").read_text(encoding="utf-8")
            if instruction!=request.get("instructions") or sha256(instruction)!=request.get("instructions_sha256"):
                raise ValueError("Instruction hash differs from generation request")
            wire_request=read_json(folder/"wire_request.json")
            if wire_request.get("model")!=manifest["model"] or request.get("model")!=manifest["model"]:
                raise ValueError("Manifest model differs from actual provider request")
            if request.get("temperature")!=manifest["temperature"] or request.get("max_output_tokens")!=manifest["max_tokens"]:
                raise ValueError("Manifest generation parameters differ from request record")
            # Canonical JSON equality tests every actual request parameter, model and schema,
            # while ignoring only insignificant serialized key ordering/whitespace.
            req_hash=sha256(json.dumps(wire_request,sort_keys=True,separators=(",",":")))
            if section in wire_by_section and wire_by_section[section]!=req_hash:
                raise ValueError("Actual model request changed for " + section)
            wire_by_section[section]=req_hash
            response_ids.add(response_id)
            response_hashes.add(wire_hash)
            call_ids.add(call_id)
            if request_id: request_ids.add(request_id)
            proof.append({"run_id":manifest["run_id"],"section":section,"call_id":call_id,
                "provider_response_id":response_id,"http_request_id":request_id or "NOT_RETURNED_BY_PROVIDER",
                "wire_response_sha256":wire_hash,"canonical_wire_request_sha256":req_hash,
                "instructions_sha256":sha256(instruction),"input_tokens":response["usage"]["input_tokens"],
                "output_tokens":response["usage"]["output_tokens"],"cost_usd":response["cost_usd"],
                "runtime_seconds":response["runtime_seconds"],"request_utc":request["timestamp_utc"],
                "completed_utc":response["completed_at_utc"]})
            costs.append(response)
        market_path=path/"output"/manifest["ticker"]/"analysis/market_snapshot.json"
        if not market_path.exists() and manifest.get('source_analysis'):
            snapshot_root=Path(manifest['source_analysis']).parent
            candidate=snapshot_root/'market_snapshot.json'
            if candidate.exists():
                frozen=read_json(snapshot_root/'manifest.json')
                if sha256(candidate.read_bytes()) != frozen['file_hashes']['market_snapshot.json']:
                    raise ValueError('Frozen market snapshot changed')
                market_path=candidate
        market=read_json(market_path) if market_path.exists() else None
        loaded.append({"path":path,"manifest":manifest,"sections":sections,"proof":proof,"market":market,
            "market_source_file":str(market_path),"costs":costs})
    return loaded


def overlap(left,right):
    a,b=set(left),set(right)
    shared=a&b
    jaccard=len(shared)/len(a|b) if a|b else "N/A"
    if len(shared)<2: return jaccard,"N/A"
    concordant=discordant=0
    for x,y in itertools.combinations(shared,2):
        if (left.index(x)-left.index(y))*(right.index(x)-right.index(y))>0: concordant+=1
        else: discordant+=1
    return jaccard,(concordant-discordant)/(concordant+discordant)


def numeric_summary(values):
    numeric=[x for x in values if x is not None]
    return {"observed_n":len(numeric),"missing_n":len(values)-len(numeric),
        "mean":statistics.mean(numeric) if numeric else "N/A",
        "sample_variance":statistics.variance(numeric) if len(numeric)>=2 else "N/A",
        "minimum":min(numeric) if numeric else "N/A","maximum":max(numeric) if numeric else "N/A",
        "complete_three_run_measure":len(numeric)==len(values) and len(values)>=3}


async def run(args):
    loaded=validate_independent_runs(args.runs)
    if args.resume:
        prior=read_json(args.output/'manifest.json')
        if prior['runs'] != [str(x['path']) for x in loaded]:
            raise ValueError('Resume source runs changed')
    args.output.mkdir(parents=True,exist_ok=args.resume)
    write_json(args.output/"manifest.json",{"state":"extracting","started_at_utc":utc_now(),"runs":[str(x["path"]) for x in loaded]})
    if args.offline_extractions:
        previous=read_json(args.offline_extractions/"manifest.json")
        if previous.get("state")!="complete": raise ValueError("Offline extraction audit is incomplete")
        cfg,budget={"openai_model":previous["extractor_model"]},None
    else:
        if not args.budget or not args.ledger: raise ValueError("Paid extraction requires --budget and --ledger")
        cfg,budget=credentials(args.config),Budget(args.budget,args.ledger)
        cfg["openai_model"]=cfg["openai_model"] or budget.policy["model"]
    observations,explicit,proof,extractions=[],[],[],[]
    try:
        agent=build_agent() if not args.offline_extractions else None
        for index,item in enumerate(loaded):
            run_id=item["manifest"]["run_id"]
            prompt=json.dumps({"report_sections":{s:x["text"] for s,x in item["sections"].items()}},ensure_ascii=False)
            if args.offline_extractions:
                source=args.offline_extractions/"calls"/run_id
                request=read_json(source/"request.json")
                if request["evidence_prompt"]!=prompt or request["instructions"]!=INSTRUCTIONS:
                    raise ValueError("Frozen extraction prompt/instructions differ from replay inputs")
                extracted=read_json(source/"response.json")["structured_output"]
                shutil.copytree(source,args.output/"calls"/run_id)
            elif args.resume and (args.output/'calls'/run_id/'response.json').exists():
                folder=args.output/'calls'/run_id
                request=read_json(folder/'request.json')
                if request['evidence_prompt']!=prompt or request['instructions']!=INSTRUCTIONS or request['model']!=cfg['openai_model'] or request['temperature']!=0 or request['max_output_tokens']!=args.max_tokens:
                    raise ValueError('Resumed extraction contract changed')
                saved=read_json(folder/'response.json')
                wire=read_json(folder/'wire_response.json')
                if json.loads(wire['choices'][0]['message']['content'])!=saved['structured_output']:
                    raise ValueError('Persisted extraction differs from actual wire response')
                extracted=saved['structured_output']
            else:
                extracted=await call_agent(agent,prompt,config=cfg,budget=budget,directory=args.output/"calls"/run_id,
                    call_id=f"{args.output.name}/stability_extract/{run_id}",temperature=0,max_tokens=args.max_tokens)
            checked=validate_extraction(extracted,item["sections"])
            write_json(args.output/"calls"/run_id/"checked.json",checked)
            extractions.append(checked)
            target=checked["writer_target"]
            risks=checked["top_three_risks"]
            market=item["market"] or {}
            values=market.get("metrics",market.get("fields",market))
            observations.append({"ticker":item["manifest"]["ticker"],"run_id":run_id,"model":item["manifest"]["model"],
                "writer_rating":checked["writer_rating"] or "N/A","writer_rating_status":checked["writer_rating_status"],
                "writer_target_price":target["amount"] if target else "N/A","writer_target_status":checked["writer_target_status"],
                "writer_target_currency":target["currency"] if target else "N/A","writer_target_horizon":target["horizon"] if target else "N/A",
                **{f"top_risk_{i+1}":risks[i]["taxonomy"] if len(risks)>i else "N/A" for i in range(3)},
                "risk_selection_method":checked["risk_selection_method"],"risk_count":len(risks),
                "fmp_rating":json.dumps(values.get("rating")),"fmp_target_price":json.dumps(values.get("target_price")),
                "fmp_source_file":item["market_source_file"],"fmp_field_provenance":json.dumps(market.get("provenance_notes",market.get("_provenance",{}))),
                "generation_input_tokens":sum(r["usage"]["input_tokens"] for r in item["costs"]),
                "generation_output_tokens":sum(r["usage"]["output_tokens"] for r in item["costs"]),
                "generation_cost_usd":sum(r["cost_usd"] for r in item["costs"]),
                "generation_runtime_seconds":sum(r["runtime_seconds"] for r in item["costs"])})
            for kind in ("ratings","targets","all_risks"):
                for record in checked[kind]:
                    explicit.append({"run_id":run_id,"observation_kind":kind,"section":record["section"],
                        "source_start":record["source_start"],"source_end":record["source_end"],"source_quote":record["source_quote"],
                        "record":json.dumps(record,ensure_ascii=False),"source_file":item["sections"][record["section"]]["metadata"]["source_file"]})
            proof+=item["proof"]
        pair_rows=[]
        for i,j in itertools.combinations(range(len(loaded)),2):
            left=[r["taxonomy"] for r in extractions[i]["top_three_risks"]]
            right=[r["taxonomy"] for r in extractions[j]["top_three_risks"]]
            jac,tau=overlap(left,right)
            pair_rows.append({"run_a":loaded[i]["manifest"]["run_id"],"run_b":loaded[j]["manifest"]["run_id"],
                "risk_jaccard":jac,"kendall_tau_shared_risks":tau,"shared_risk_count":len(set(left)&set(right)),
                "rank_basis":"first-mention order within shared taxonomy categories; tau N/A if fewer than two shared"})
        targets=[x["writer_target"] for x in extractions]
        horizons={(x["currency"],x["horizon"]) for x in targets if x}
        comparable=len(horizons)==1 and all(x and x["currency"]!="NOT_STATED" for x in targets)
        target_stats=numeric_summary([x["amount"] if x else None for x in targets]) if comparable else {"status":"N/A: absent, ambiguous, mixed horizon/currency or currency not stated"}
        rating_values=[x["writer_rating"] for x in extractions]
        ordinal={"strong_buy":2,"buy":1,"hold":0,"sell":-1,"strong_sell":-2}
        rating_stats=numeric_summary([ordinal.get(x) for x in rating_values])
        market_values=[(x["market"] or {}).get("metrics",(x["market"] or {}).get("fields",x["market"] or {})) for x in loaded]
        fmp_targets=[m.get("target_price") if isinstance(m.get("target_price"),(float,int)) else None for m in market_values]
        extractor_costs=[read_json(args.output/"calls"/x["manifest"]["run_id"]/"response.json") for x in loaded]
        summary={"writer_target_price":target_stats,"writer_rating_ordinal":rating_stats,
            "rating_encoding":ordinal,"rating_encoding_caution":"Ordinal score is a reporting convention, not a cardinal economic distance.",
            "writer_rating_exact_agreement":len(set(rating_values))==1 if all(rating_values) else "N/A",
            "fmp_target_price":numeric_summary(fmp_targets),
            "fmp_rating_exact_agreement":len({json.dumps(m.get('rating'),sort_keys=True) for m in market_values})==1 if all(m.get('rating') is not None for m in market_values) else "N/A",
            "fmp_interpretation":"These fields are copied from the frozen market data. Zero variance measures fixed input reuse, not LLM stability.",
            "risk_taxonomy":TAXONOMY,"all_runs_have_three_distinct_risks":all(len(x["top_three_risks"])==3 for x in extractions),
            "extraction_extra_cost_usd":sum(r["cost_usd"] for r in extractor_costs),
            "extraction_extra_runtime_seconds":sum(r["runtime_seconds"] for r in extractor_costs),
            "extraction_extra_tokens":sum(r["usage"]["input_tokens"]+r["usage"]["output_tokens"] for r in extractor_costs),
            "offline_replay":bool(args.offline_extractions),
            "incremental_extraction_cost_this_execution":0 if args.offline_extractions else sum(r["cost_usd"] for r in extractor_costs),
            "limitations":"Machine extraction, no human review. Three runs give imprecise variance estimates. Coarse risk taxonomy can overstate overlap. Saved unique provider IDs and wire records support independent calls but are not provider-signed receipts."}
        csv_write(args.output/"stability_runs.csv",observations)
        csv_write(args.output/"explicit_observations.csv",explicit)
        csv_write(args.output/"generation_independence.csv",proof)
        csv_write(args.output/"risk_pairwise.csv",pair_rows)
        write_json(args.output/"stability_summary.json",summary)
        write_json(args.output/"manifest.json",{"state":"complete","completed_at_utc":utc_now(),"run_count":len(loaded),
            "runs":[str(x["path"]) for x in loaded],"ticker":loaded[0]["manifest"]["ticker"],
            "extractor_model":cfg["openai_model"],"instructions_sha256":sha256(INSTRUCTIONS),"human_review_performed":False,
            "offline_replay_from":str(args.offline_extractions) if args.offline_extractions else None})
    finally:
        if budget is not None: budget.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs",nargs="+",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--config")
    parser.add_argument("--budget")
    parser.add_argument("--ledger",type=Path)
    parser.add_argument("--offline-extractions",type=Path,help="Reuse saved extraction responses without model/network calls; does not count as another generation.")
    parser.add_argument("--max-tokens",type=int,default=8000)
    parser.add_argument('--resume',action='store_true',help='Continue extraction over the same completed independent writer runs; does not count as a fresh generation.')
    args=parser.parse_args()
    asyncio.run(run(args))


if __name__=="__main__": main()
