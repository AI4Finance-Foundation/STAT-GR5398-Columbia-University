"""Run actual eight-agent generation, ninth-stage verification and offline rendering.

Data acquisition is separate: supply the immutable snapshot's analysis directory.
Every new run invokes the model anew. --resume continues the SAME run, preserving
validated successes and archiving failed attempts; it is not a stability replicate.
"""
from __future__ import annotations
import argparse
import asyncio
import json
import shutil
import subprocess
import sys
from pathlib import Path
from audit_io import Budget, call_agent, credentials, public_url, sha256, utc_now, write_json

UPSTREAM_SHA = "2717499b8e30f242640af08c4ad9afd1113c2d45"
SUBMISSION_ROOT = Path(__file__).resolve().parents[1]
FIELDS = {"tagline":"tagline", "company_overview":"overview", "investment_overview":"investment_update",
    "valuation_overview":"valuation_analysis", "risks":"risk_analysis", "competitor_analysis":"competitive_analysis",
    "major_takeaways":"takeaways", "news_summary":"news_summary"}


def setup(upstream):
    revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if revision != UPSTREAM_SHA:
        raise ValueError("Upstream revision differs from the audited commit")
    src = upstream / "finrobot_equity/core/src"
    sys.path.insert(0, str(src))
    return src


def hashes(root):
    return {str(path.relative_to(root)).replace("\\", "/"): sha256(path.read_bytes())
            for path in sorted(root.rglob("*")) if path.is_file() and "__pycache__" not in path.parts}


def validate_hashes(root, manifest):
    if set(hashes(root)) != set(manifest):
        raise ValueError("Frozen input file set changed")
    for relative, expected in manifest.items():
        path = root / relative
        if not path.is_file() or sha256(path.read_bytes()) != expected:
            raise ValueError("Frozen input changed: " + relative)


def load_data(analysis):
    import pandas as pd
    required = {"financial_metrics":"financial_metrics_and_forecasts.csv", "peer_ebitda":"peer_ebitda_comparison.csv",
                "peer_ev_ebitda":"peer_ev_ebitda_comparison.csv"}
    data = {key: pd.read_csv(analysis/name, index_col=0 if key.startswith("peer_") else None)
            for key, name in required.items()}
    data["company_news"] = json.loads((analysis/"company_news.json").read_text(encoding="utf-8"))
    return data


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def contract(agent, prompt, cfg, temperature, max_tokens):
    """Nonsecret, immutable identity of one intended typed call."""
    result = {"agent_name": agent.name, "model": cfg["openai_model"],
        "provider_base_url": public_url(cfg["openai_base_url"]),
        "instructions": agent.instructions, "instructions_sha256": sha256(agent.instructions),
        "evidence_prompt": prompt, "evidence_sha256": sha256(prompt),
        "temperature": temperature, "max_output_tokens": max_tokens,
        "output_schema": agent.output_type.model_json_schema(),
        "tracing_disabled": True, "sdk_retries": 0}
    if cfg.get('reasoning_effort'):
        result.update(temperature=None,reasoning_effort=cfg['reasoning_effort'],
            output_limit_parameter='max_completion_tokens',temperature_omitted=True)
    return result


def validate_request(directory, expected):
    record = read_json(directory/"request.json")
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError("Persisted generation contract changed: " + key)
    for filename, key in (("prompt.txt", "evidence_prompt"), ("instructions.txt", "instructions")):
        if (directory/filename).read_text(encoding="utf-8") != expected[key]:
            raise ValueError("Persisted generation text changed: " + filename)
    return record


def validate_success(directory, agent, expected, budget):
    """Revalidate persisted paid output using source wire bytes and settled usage."""
    from agents import AgentOutputSchema
    request = validate_request(directory, expected)
    response = read_json(directory/"response.json")
    wire = read_json(directory/"wire_request.json")
    limit_key=expected.get('output_limit_parameter','max_tokens')
    if wire.get("model") != expected["model"] or wire.get("temperature") != expected["temperature"] or wire.get(limit_key) != expected["max_output_tokens"]:
        raise ValueError("Persisted wire model/parameters changed")
    if expected.get('reasoning_effort') and (wire.get('reasoning_effort')!=expected['reasoning_effort'] or 'temperature' in wire or 'max_tokens' in wire):
        raise ValueError('Persisted reasoning parameters changed')
    if wire.get("messages") != [{"role":"system", "content":expected["instructions"]},
                                {"role":"user", "content":expected["evidence_prompt"]}]:
        raise ValueError("Persisted wire evidence/instructions changed")
    if wire.get("response_format", {}).get("json_schema", {}).get("schema") != AgentOutputSchema(agent.output_type).json_schema():
        raise ValueError("Persisted wire output schema changed")
    actual = read_json(directory/"wire_response.json")
    http = read_json(directory/"http_response.json")
    if http.get("status_code") != 200 or not actual.get("id"):
        raise ValueError("Persisted success lacks provider completion evidence")
    if json.loads(actual["choices"][0]["message"]["content"]) != response.get("structured_output"):
        raise ValueError("Persisted structured output differs from provider response")
    if response.get("call_id") != request["call_id"] or response.get("attempts") != 1:
        raise ValueError("Persisted call identity changed")
    usage, raw_usage = response["usage"], actual["usage"]
    if (usage.get("requests") != 1 or usage.get("total_tokens", 0) <= 0 or
        usage["total_tokens"] != usage["input_tokens"] + usage["output_tokens"] or
        usage["input_tokens"] != raw_usage["prompt_tokens"] or
        usage["output_tokens"] != raw_usage["completion_tokens"] or
        usage["total_tokens"] != raw_usage["total_tokens"]):
        raise ValueError("Persisted token usage changed")
    rows = [row for row in budget.rows() if row.get("call_id") == request["call_id"]]
    settles = [row for row in rows if row["event"] == "settle"]
    if len([row for row in rows if row["event"] == "reserve"]) != 1 or len(settles) != 1:
        raise ValueError("Persisted success needs exactly one reservation and settlement")
    settled = settles[0]
    if any(settled[key] != response[key] for key in ("cost_usd", "runtime_seconds")) or any(
            settled[key] != usage[key] for key in ("input_tokens", "output_tokens")):
        raise ValueError("Persisted success differs from cost ledger")
    if response.get("runtime_seconds", 0) <= 0 or not response.get("completed_at_utc"):
        raise ValueError("Persisted success lacks timing evidence")
    return agent.output_type.model_validate(response["structured_output"])


def recover_settled_response(directory, agent, expected, budget):
    """Recover a crash between ledger settlement and response.json persistence.

    Never settle unknown billing automatically. A complete provider response and
    an existing exact settlement are both required; all reused results still pass
    validate_success. The reconstructed record is explicitly marked as recovery.
    """
    if not directory.exists() or (directory/"response.json").exists() or not (directory/"request.json").exists():
        return
    request = validate_request(directory, expected)
    settlements = [row for row in budget.rows() if row.get("call_id") == request["call_id"] and row["event"] == "settle"]
    if not settlements or not (directory/"wire_response.json").exists():
        return
    if len(settlements) != 1:
        raise ValueError("Crash recovery has ambiguous cost settlements")
    settled = settlements[0]
    wire = read_json(directory/"wire_response.json")
    if read_json(directory/"http_response.json").get("status_code") != 200:
        raise ValueError("Settled crash recovery lacks successful HTTP evidence")
    raw_output = json.loads(wire["choices"][0]["message"]["content"])
    typed = agent.output_type.model_validate(raw_output)
    if typed.model_dump() != raw_output:
        raise ValueError("Recovered wire output differs from the typed schema")
    usage = wire["usage"]
    if settled["input_tokens"] != usage["prompt_tokens"] or settled["output_tokens"] != usage["completion_tokens"]:
        raise ValueError("Recovered wire usage differs from settled cost")
    write_json(directory/"response.json", {"call_id":request["call_id"], "structured_output":typed.model_dump(),
        "usage":{"requests":1, "input_tokens":usage["prompt_tokens"], "output_tokens":usage["completion_tokens"],
                 "total_tokens":usage["total_tokens"], "input_tokens_details":usage.get("prompt_tokens_details"),
                 "output_tokens_details":usage.get("completion_tokens_details")},
        "runtime_seconds":settled["runtime_seconds"], "cost_usd":settled["cost_usd"],
        "completed_at_utc":settled["timestamp_utc"], "attempts":1,
        "recovered_from_settled_wire":True})
    validate_success(directory, agent, expected, budget)
    write_json(directory/"recovery.json", {"at_utc":utc_now(), "call_id":request["call_id"],
        "reason":"Recovered persisted provider output after ledger settlement; no new generation or cost",
        "wire_response_sha256":sha256((directory/"wire_response.json").read_bytes())})


def attempt_history(run, group, section):
    return run/"attempt_history"/group/section


def archive_attempt(run, group, section):
    """Move the previous attempt intact; never overwrite or delete wire evidence."""
    current = run/group/section
    history = attempt_history(run, group, section)
    history.mkdir(parents=True, exist_ok=True)
    number = 1
    while (history/f"attempt_{number:03d}").exists():
        number += 1
    destination = history/f"attempt_{number:03d}"
    current.replace(destination)
    return destination


def attempt_number(run, group, section):
    history = attempt_history(run, group, section)
    return 1 + len(list(history.glob("attempt_*")))


def same_or_write(path, text, *, replace_original=None):
    """Only recover absent output, or replace the known unverified original."""
    if path.exists() and path.read_text(encoding="utf-8") not in {text, replace_original}:
        raise ValueError("Persisted successful section text changed: " + path.name)
    if not path.exists() or path.read_text(encoding="utf-8") != text:
        path.write_text(text, encoding="utf-8")


def frozen_copy(source, destination, expected, text_sections=()):
    """Recover an interrupted local copy without changing existing frozen data."""
    allowed = set(expected) | {section+".txt" for section in text_sections}
    extras = set(hashes(destination)) - allowed
    if extras:
        raise ValueError("Unexpected files in frozen analysis copy: " + repr(sorted(extras)))
    for relative, digest in expected.items():
        target = destination/relative
        if target.exists():
            # Generated section texts are checked against their paid response below.
            if relative in {section+".txt" for section in text_sections}:
                continue
            if sha256(target.read_bytes()) != digest:
                raise ValueError("Frozen copied input changed: " + relative)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source/relative, target)


def begin_run(path, manifest, resume, identity):
    if path.exists():
        if not resume:
            raise FileExistsError("Run exists; --resume explicitly continues this same run")
        old = read_json(path/"manifest.json")
        for key in identity:
            if old.get(key) != manifest.get(key):
                raise ValueError("Resume identity changed: " + key)
        old.setdefault("resume_events", []).append({"at_utc":utc_now(),
            "meaning":"Continuation of the same run_id; never an independent stability replicate",
            "reused_sections":[], "new_call_ids":[]})
        write_json(path/"manifest.json", old)
        return old
    if resume:
        raise FileNotFoundError("--resume requires an existing run manifest")
    path.mkdir(parents=True, exist_ok=False)
    manifest["resume_events"] = []
    write_json(path/"manifest.json", manifest)
    return manifest


def record_progress(run, manifest, section, *, reused=False, call_id=None):
    if manifest.get("resume_events"):
        event = manifest["resume_events"][-1]
        if reused and section not in event["reused_sections"]:
            event["reused_sections"].append(section)
        if call_id and call_id not in event["new_call_ids"]:
            event["new_call_ids"].append(call_id)
    manifest.setdefault("completed_sections", [])
    if section not in manifest["completed_sections"]:
        manifest["completed_sections"].append(section)
    write_json(run/"manifest.json", manifest)


async def generate(args):
    from modules.equity_agents.agent_manager import EquityResearchAgentManager
    manager = EquityResearchAgentManager()
    cfg = credentials(args.config)
    if not cfg["openai_api_key"]:
        raise ValueError("Missing OpenAI credential; no baseline was generated")
    budget = Budget(args.budget, args.ledger)
    try:
        cfg["openai_model"] = cfg["openai_model"] or budget.policy["model"]
        input_hashes = hashes(args.analysis)
        data = load_data(args.analysis)
        validate_hashes(args.analysis, input_hashes)
        prompt = manager._prepare_financial_data_prompt(data, args.company_name, args.ticker)
        # Public-source availability adapter only: disclose units, missing fields
        # and heterogeneous definitions to every writer. Prior FMP runs have no
        # such file and their immutable prompts remain byte-identical.
        source_notes = args.analysis/'source_notes.txt'
        if source_notes.exists():
            prompt += '\n## Explicit input provenance and limitations (not news)\n' + source_notes.read_text(encoding='utf-8') + '\n'
        manifest = {"run_id":args.run.name, "ticker":args.ticker,
            "company_name":args.company_name, "started_at_utc":utc_now(), "upstream_sha":UPSTREAM_SHA,
            "state":"generating", "source_analysis":str(args.analysis.resolve()), "input_hashes":input_hashes,
            "evidence_sha256":sha256(prompt), "model":cfg["openai_model"], "temperature":args.temperature,
            "max_tokens":args.max_tokens, "variant":"course_typed_eight_agent_compatibility_baseline",
            "baseline_departure":"Current upstream CLI uses a legacy plain chat wrapper. This runner uses the actual eight typed agents and their unmodified original instructions/shared evidence prompt."}
        if source_notes.exists():
            manifest['baseline_departure'] += ' Public-source adapter appends frozen source_notes.txt to disclose missing observations and incompatible definitions before generation.'
            manifest['source_notes_sha256'] = sha256(source_notes.read_bytes())
        identity = ("run_id", "ticker", "company_name", "upstream_sha", "source_analysis", "input_hashes",
                    "evidence_sha256", "model", "temperature", "max_tokens", "variant")
        manifest = begin_run(args.run, manifest, getattr(args, "resume", False), identity)
        contracts = {section:contract(agent, prompt, cfg, args.temperature, args.max_tokens)
                     for section, agent in manager.agents.items()}
        signatures = {section:sha256(json.dumps(value, sort_keys=True)) for section, value in contracts.items()}
        if "call_contracts" in manifest and manifest["call_contracts"] != signatures:
            raise ValueError("Resume instructions/schema/provider changed")
        manifest["call_contracts"] = signatures
        write_json(args.run/"manifest.json", manifest)
        analysis = args.run/"output"/args.ticker/"analysis"
        if manifest.get("analysis_hashes"):
            validate_hashes(analysis, manifest["analysis_hashes"])
        frozen_copy(args.analysis, analysis, input_hashes, FIELDS)
        for section, agent in manager.agents.items():
            validate_hashes(args.analysis, input_hashes)
            directory, expected = args.run/"agents"/section, contracts[section]
            reused, call_id = False, None
            recover_settled_response(directory, agent, expected, budget)
            if directory.exists() and (directory/"response.json").exists():
                result = validate_success(directory, agent, expected, budget)
                reused = True
            else:
                if manifest["state"] in {"generated", "rendered"}:
                    raise ValueError("Completed run lost a successful section; refusing paid replacement")
                if directory.exists():
                    if (directory/"request.json").exists():
                        validate_request(directory, expected)
                    archive_attempt(args.run, "agents", section)
                number = attempt_number(args.run, "agents", section)
                call_id = f"{args.run.name}/{section}" + (f"/attempt_{number:03d}" if number > 1 else "")
                if manifest.get("resume_events"):
                    manifest["resume_events"][-1]["new_call_ids"].append(call_id)
                    write_json(args.run/"manifest.json", manifest)
                result = await call_agent(agent, prompt, config=cfg, budget=budget, directory=directory,
                    call_id=call_id, temperature=args.temperature, max_tokens=args.max_tokens)
                result = validate_success(directory, agent, expected, budget)
            text = getattr(result, FIELDS[section])
            if not text.strip():
                raise ValueError("Empty structured text from " + section)
            same_or_write(analysis/(section+".txt"), text)
            record_progress(args.run, manifest, section, reused=reused, call_id=call_id)
        validate_hashes(args.analysis, input_hashes)
        if manifest["state"] != "rendered":
            manifest.update(state="generated", completed_at_utc=manifest.get("completed_at_utc", utc_now()))
        manifest["analysis_hashes"] = hashes(analysis)
        write_json(args.run/"manifest.json", manifest)
    finally:
        budget.close()


def repair_prompt(base_prompt, review, diagnostic):
    return (base_prompt + "\n\nVALIDATION REPAIR (not additional factual evidence):\n"
        "The previous structured review failed deterministic validation. Return a complete corrected review "
        "covering every original source unit and all its claims. Preserve supported substantive facts. "
        "Use ONLY the ORIGINAL EVIDENCE PROMPT above for facts and citations. The prior review is a "
        "candidate to correct, never a source of factual evidence.\nDIAGNOSTIC: " + diagnostic +
        "\nPREVIOUS CANDIDATE REVIEW:\n" + json.dumps(review.model_dump(), ensure_ascii=False, sort_keys=True))


async def verify_section(args, manifest, section, agent, evidence, original, cfg, budget, validate_review, base_prompt):
    directory = args.final_run/"verifier"/section
    reused_record = directory/'reused_review.json'
    if reused_record.exists():
        provenance=read_json(reused_record)
        # Cross-version reuse is an explicitly derived review, never another
        # model draw. Original prompts, calls, settlement and response IDs stay.
        for relative,digest in provenance['copied_file_hashes'].items():
            if sha256((directory/relative).read_bytes()) != digest:
                raise ValueError('Reused review bytes changed')
        if provenance['evidence_sha256']!=sha256(evidence) or provenance['original_sha256']!=sha256(original):
            raise ValueError('Reused review source/evidence changed')
        expected=contract(agent,(directory/'prompt.txt').read_text(encoding='utf-8'),cfg,0,args.max_tokens)
        review=validate_success(directory,agent,expected,budget)
        checked=validate_review(review,evidence,original)
        if (directory/'checked.json').exists() and read_json(directory/'checked.json')!=checked:
            raise ValueError('Reused review checked output changed')
        write_json(directory/'checked.json',checked)
        return checked,True,None
    planned_prompt, repair_index = base_prompt, 0
    # Reconstruct the exact repair chain from immutable archived response/diagnostics.
    for previous in sorted(attempt_history(args.final_run, "verifier", section).glob("attempt_*")):
        if (previous/"request.json").exists():
            expected = contract(agent, planned_prompt, cfg, 0, args.max_tokens)
            validate_request(previous, expected)
            if (previous/"response.json").exists():
                review = validate_success(previous, agent, expected, budget)
                if (previous/"validation_error.json").exists():
                    saved = read_json(previous/"validation_error.json")
                    try:
                        validate_review(review, evidence, original)
                    except ValueError as exc:
                        diagnostic = str(exc)
                    else:
                        raise ValueError("Archived failed verifier review unexpectedly validates")
                    expected_error = {"error_type":"ValueError", "message":diagnostic,
                        "repair_index":repair_index, "base_prompt_sha256":sha256(base_prompt),
                        "evidence_sha256":sha256(evidence), "original_sha256":sha256(original)}
                    if saved != expected_error:
                        raise ValueError("Archived verifier validation diagnostics changed")
                    repair_index += 1
                    planned_prompt = repair_prompt(base_prompt, review, diagnostic)
                else:
                    raise ValueError("Successful verifier output was archived without a validation failure")
    if repair_index > args.max_validation_repairs:
        raise ValueError("Archived verifier repair chain exceeds configured limit")
    while True:
        expected = contract(agent, planned_prompt, cfg, 0, args.max_tokens)
        reused, call_id = False, None
        recover_settled_response(directory, agent, expected, budget)
        if directory.exists() and (directory/"response.json").exists():
            review = validate_success(directory, agent, expected, budget)
            reused = True
        else:
            if manifest["state"] in {"generated", "rendered"}:
                raise ValueError("Completed verifier run lost a section; refusing paid replacement")
            if directory.exists():
                if (directory/"request.json").exists():
                    validate_request(directory, expected)
                archive_attempt(args.final_run, "verifier", section)
            number = attempt_number(args.final_run, "verifier", section)
            call_id = f"{args.final_run.name}/verifier/{section}" + (f"/attempt_{number:03d}" if number > 1 else "")
            if manifest.get("resume_events"):
                manifest["resume_events"][-1]["new_call_ids"].append(call_id)
                write_json(args.final_run/"manifest.json", manifest)
            review = await call_agent(agent, planned_prompt, config=cfg, budget=budget, directory=directory,
                call_id=call_id, temperature=0, max_tokens=args.max_tokens)
            review = validate_success(directory, agent, expected, budget)
        try:
            checked = validate_review(review, evidence, original)
        except ValueError as exc:
            diagnostic = str(exc)
            error = {"error_type":"ValueError", "message":diagnostic,
                "repair_index":repair_index, "base_prompt_sha256":sha256(base_prompt),
                "evidence_sha256":sha256(evidence), "original_sha256":sha256(original)}
            error_path = directory/"validation_error.json"
            if error_path.exists() and read_json(error_path) != error:
                raise ValueError("Persisted validation failure changed") from None
            write_json(error_path, error)
            if repair_index >= args.max_validation_repairs:
                raise ValueError("Verifier validation failed; configured repair limit reached: " + diagnostic) from None
            if manifest["state"] in {"generated", "rendered"}:
                raise ValueError("Completed verifier output no longer validates") from None
            planned_prompt = repair_prompt(base_prompt, review, diagnostic)
            repair_index += 1
            archive_attempt(args.final_run, "verifier", section)
            continue
        checked_path = directory/"checked.json"
        if checked_path.exists() and read_json(checked_path) != checked:
            raise ValueError("Persisted verifier checked output changed")
        write_json(checked_path, checked)
        return checked, reused, call_id


def adopt_prior_review(args, section, agent, evidence, original, cfg, budget, validate_review, base_prompt):
    """Select earliest prior wire-confirmed review passing the new validator.

    Selection is solely deterministic validity, never a favorable score/label.
    Source failures are not mutated. Repaired prompts retain their complete prior
    chain, and every chosen response remains linked to its actual paid call.
    """
    source=getattr(args,'reuse_review_run',None)
    destination=args.final_run/'verifier'/section
    if not source or destination.exists():return
    directories=sorted(attempt_history(source,'verifier',section).glob('attempt_*'))
    directories.append(source/'verifier'/section)
    planned_prompt=base_prompt
    trail=[]
    for previous in directories:
        if not (previous/'request.json').exists():continue
        if (previous/'reused_review.json').exists():
            # Transitive reuse still preserves the original paid call. Validate
            # exact immutable bytes and the complete original/repair prompt;
            # never reconstruct a shorter request for an inherited response.
            inherited=read_json(previous/'reused_review.json')
            if inherited['evidence_sha256']!=sha256(evidence) or inherited['original_sha256']!=sha256(original):
                raise ValueError('Inherited review evidence identity changed')
            for name,digest in inherited['copied_file_hashes'].items():
                if sha256((previous/name).read_bytes())!=digest:
                    raise ValueError('Inherited review bytes changed')
            planned_prompt=(previous/'prompt.txt').read_text(encoding='utf-8')
            if not planned_prompt.startswith(base_prompt):
                raise ValueError('Inherited review prompt lost original evidence/source prefix')
        expected=contract(agent,planned_prompt,cfg,0,args.max_tokens)
        validate_request(previous,expected)
        if not (previous/'response.json').exists():continue
        review=validate_success(previous,agent,expected,budget)
        try:checked=validate_review(review,evidence,original)
        except ValueError as exc:
            trail.append({'directory':str(previous.resolve()),'new_validator_error':str(exc),
                          'response_sha256':sha256((previous/'response.json').read_bytes())})
        else:
            destination.mkdir(parents=True,exist_ok=False)
            names=('request.json','response.json','wire_request.json','wire_response.json',
                   'http_response.json','prompt.txt','instructions.txt')
            for name in names:shutil.copy2(previous/name,destination/name)
            write_json(destination/'reused_review.json',{
                'derived_at_utc':utc_now(),'source_run':str(source.resolve()),
                'source_directory':str(previous.resolve()),'source_call_id':read_json(previous/'request.json')['call_id'],
                'original_sha256':sha256(original),'evidence_sha256':sha256(evidence),
                'copied_file_hashes':{name:sha256((destination/name).read_bytes()) for name in names},
                'preceding_candidates':trail,'selection_rule':'Earliest saved wire-confirmed review passing new deterministic validator.',
                'new_model_generation':False,'incremental_model_cost_usd':0,
                'cost_note':'The historical call is billed once in global ledger; copied response retains actual source usage/cost, not a second charge.'})
            write_json(destination/'checked.json',checked)
            return
        if (previous/'validation_error.json').exists():
            failure=read_json(previous/'validation_error.json')
            if any(failure[key]!=value for key,value in {
                'base_prompt_sha256':sha256(base_prompt),'evidence_sha256':sha256(evidence),
                'original_sha256':sha256(original)}.items()):
                raise ValueError('Prior repair diagnostic source identity changed')
            planned_prompt=repair_prompt(base_prompt,review,failure['message'])


async def verify(args):
    import inspect
    from modules.equity_agents.verifier_agent import verifier_agent, prepare_verifier_prompt, validate_review
    baseline = read_json(args.run/"manifest.json")
    if baseline["state"] not in {"generated", "rendered"}:
        raise ValueError("Baseline is incomplete")
    ticker = baseline["ticker"]
    analysis = args.run/"output"/ticker/"analysis"
    validate_hashes(analysis, baseline["analysis_hashes"])
    cfg = credentials(args.config)
    budget = Budget(args.budget, args.ledger)
    try:
        cfg["openai_model"] = cfg["openai_model"] or budget.policy["model"]
        if not 0 <= args.max_validation_repairs <= 2:
            raise ValueError("At most two verifier validation repairs are permitted")
        if getattr(args,'reasoning_effort',None):
            cfg['reasoning_effort']=args.reasoning_effort
        verifier_parameters = {"model":cfg["openai_model"], "temperature":0,
            "validator_source_sha256":sha256(Path(inspect.getfile(validate_review)).read_bytes()),
            "max_tokens":args.max_tokens, "max_validation_repairs":args.max_validation_repairs,
            "provider_base_url":public_url(cfg["openai_base_url"]),
            "instructions_sha256":sha256(verifier_agent.instructions),
            "output_schema":verifier_agent.output_type.model_json_schema()}
        if cfg.get('reasoning_effort'):
            verifier_parameters.update(temperature=None,reasoning_effort=cfg['reasoning_effort'],
                output_limit_parameter='max_completion_tokens',temperature_omitted=True)
        prior_reviews=getattr(args,'reuse_review_run',None)
        if prior_reviews:
            prior_manifest=read_json(prior_reviews/'manifest.json')
            if prior_manifest.get('parent_run_id')!=args.run.name or prior_manifest['parent_analysis_hashes']!=baseline['analysis_hashes']:
                raise ValueError('Prior review run has different baseline inputs')
            previous_parameters=prior_manifest['verifier_parameters']
            for key,value in verifier_parameters.items():
                if key!='validator_source_sha256' and previous_parameters.get(key)!=value:
                    raise ValueError('Prior paid review contract differs: '+key)
            verifier_parameters['reuse_review_run']=str(prior_reviews.resolve())
            verifier_parameters['reuse_source_manifest_sha256']=sha256((prior_reviews/'manifest.json').read_bytes())
        manifest = dict(baseline, run_id=args.final_run.name, parent_run_id=args.run.name,
            variant="option_A_verifier", state="verifying", started_at_utc=utc_now(),
            verifier_parameters=verifier_parameters, parent_analysis_hashes=baseline["analysis_hashes"])
        # Deterministic revalidation retains the paid source's presentation date;
        # started_at_utc still records when this new local derivation actually ran.
        if prior_reviews:
            manifest["report_date"] = prior_manifest.get("report_date", prior_manifest["started_at_utc"][:10])
        for key in ("completed_sections", "analysis_hashes", "report_hashes", "rendered_at_utc",
                    "completed_at_utc", "resume_events", "call_contracts"):
            manifest.pop(key, None)
        identity = ("run_id", "parent_run_id", "variant", "upstream_sha", "input_hashes", "evidence_sha256",
                    "verifier_parameters", "parent_analysis_hashes")
        manifest = begin_run(args.final_run, manifest, getattr(args, "resume", False), identity)
        final_analysis = args.final_run/"output"/ticker/"analysis"
        if manifest.get("analysis_hashes"):
            validate_hashes(final_analysis, manifest["analysis_hashes"])
        frozen_copy(analysis, final_analysis, baseline["analysis_hashes"], FIELDS)
        for section in FIELDS:
            validate_hashes(analysis, baseline["analysis_hashes"])
            prompt = (args.run/"agents"/section/"prompt.txt").read_text(encoding="utf-8")
            if sha256(prompt) != baseline["evidence_sha256"]:
                raise ValueError("Original writer prompt changed: " + section)
            original = (analysis/(section+".txt")).read_text(encoding="utf-8")
            adopt_prior_review(args,section,verifier_agent,prompt,original,cfg,budget,
                               validate_review,prepare_verifier_prompt(prompt,original))
            checked, reused, call_id = await verify_section(args, manifest, section, verifier_agent,
                prompt, original, cfg, budget, validate_review, prepare_verifier_prompt(prompt, original))
            same_or_write(final_analysis/(section+".txt"), checked["revised_text"], replace_original=original)
            record_progress(args.final_run, manifest, section, reused=reused, call_id=call_id)
        validate_hashes(analysis, baseline["analysis_hashes"])
        if manifest["state"] != "rendered":
            manifest.update(state="generated", completed_at_utc=manifest.get("completed_at_utc", utc_now()))
        manifest["analysis_hashes"] = hashes(final_analysis)
        write_json(args.final_run/"manifest.json", manifest)
    finally:
        budget.close()


def render_supplement_hashes(recorded):
    return {relative: digest for relative, digest in recorded.items()
            if relative in {"market_snapshot.json", "valuation_inputs.json"} or relative.startswith("valuation/")}


def require_render_supplements(recorded):
    missing = {"market_snapshot.json", "valuation_inputs.json", "valuation/engine_valuation.json"} - set(recorded)
    if missing:
        raise ValueError("Frozen market/valuation inputs are required; missing: " + ", ".join(sorted(missing)))


def resolve_render_snapshot(manifest):
    """Resolve saved identities locally; honor an explicit fresh-run snapshot.

    Both actual analysis bytes and all render supplements must match the saved
    identities. Equal company names or equal analysis alone are insufficient:
    two snapshots can have the same accounts but different market observations.
    """
    expected_inputs = manifest["input_hashes"]
    if not expected_inputs:
        raise ValueError("Cannot relocate a snapshot without saved input hashes")
    saved = manifest.get("render_input_hashes", {})
    expected_supplements = render_supplement_hashes(saved)
    if saved:
        require_render_supplements(expected_supplements)

    def match(source):
        location = source/"manifest.json"
        if not location.is_file():
            return None
        snapshot = read_json(location)
        if snapshot.get("status") != "frozen" or snapshot.get("upstream_sha") != UPSTREAM_SHA:
            return None
        recorded = snapshot["file_hashes"]
        inputs = {key[len("analysis/"):]: value for key, value in recorded.items() if key.startswith("analysis/")}
        supplements = render_supplement_hashes(recorded)
        if inputs != expected_inputs or (saved and supplements != expected_supplements):
            return None
        require_render_supplements(supplements)
        validate_hashes(source/"analysis", expected_inputs)
        for relative, digest in supplements.items():
            if not (source/relative).is_file() or sha256((source/relative).read_bytes()) != digest:
                raise ValueError("Frozen render supplement changed: " + str(source/relative))
        return source, snapshot, supplements

    historical = Path(manifest["source_analysis"]).parent
    if not saved:
        # A first render has no saved market identity yet. Its explicitly supplied
        # snapshot is authoritative only after validating its actual input bytes.
        result = match(historical)
        if result:
            return result
    snapshots = SUBMISSION_ROOT/"snapshots"
    candidates = []
    if snapshots.is_dir():
        for source in sorted(snapshots.iterdir()):
            if source.is_dir():
                result = match(source)
                if result:
                    if saved:
                        return result
                    candidates.append(result)
    if candidates:
        market_identities = {tuple(sorted(supplements.items())) for _, _, supplements in candidates}
        if len(market_identities) > 1:
            raise ValueError("Ambiguous frozen snapshots: identical analysis has different market/valuation inputs; "
                             "supply an accessible explicit source_analysis snapshot or saved render-input hashes")
        return candidates[0]
    # Fresh runs may deliberately use a separately located frozen snapshot.
    # Old absolute paths are a last resort, never a reason to omit inputs.
    if saved:
        result = match(historical)
        if result:
            return result
    raise ValueError("No frozen snapshot matches the saved analysis and render-input hashes; "
                     "restore the matching snapshot under this submission's snapshots directory")


def freeze_render_inputs(run, manifest):
    """Freeze presentation data without altering historical source manifests.

    A complete run's packaged inputs are self-contained. A new derived run finds
    the current snapshot by saved content hashes and copies its historical
    provenance byte-for-byte from the packaged parent. Resolution of relocated
    paths is recorded separately and never injected into the writer evidence.
    """
    destination = run/"render_inputs"
    saved = manifest.get("render_input_hashes", {})
    if saved:
        require_render_supplements(render_supplement_hashes(saved))
        current = hashes(destination)
        if set(current) - set(saved) or any(saved.get(key) != value for key, value in current.items()):
            raise ValueError("Copied render inputs changed")
        if current == saved:
            if read_json(destination/"manifest.json")["input_hashes"] != render_supplement_hashes(saved):
                raise ValueError("Render provenance does not describe the saved supplement hashes")
            write_json(run/"render_input_resolution.json", {"resolved_at_utc":utc_now(),
                "method":"self-contained saved render inputs", "resolved_directory":str(destination.resolve()),
                "historical_source_analysis":manifest["source_analysis"], "validated_hashes":saved})
            return destination/"market_snapshot.json"

    source, snapshot, supplements = resolve_render_snapshot(manifest)
    snapshot_manifest = source/"manifest.json"
    previous = destination/"manifest.json"
    provenance_source = None
    if saved:
        if "manifest.json" not in saved:
            raise ValueError("Saved render input identity lacks its provenance manifest")
        candidates = [previous]
        if manifest.get("parent_run_id"):
            candidates.append(SUBMISSION_ROOT/"runs"/manifest["parent_run_id"]/"render_inputs"/"manifest.json")
        for candidate in candidates:
            if candidate.is_file() and sha256(candidate.read_bytes()) == saved["manifest.json"]:
                provenance_source = candidate
                break
        if provenance_source is None:
            raise ValueError("Cannot restore the exact saved render provenance; retain parent render_inputs/manifest.json")
        provenance = read_json(provenance_source)
        if (provenance.get("input_hashes") != supplements or
                provenance.get("snapshot_manifest_sha256") != sha256(snapshot_manifest.read_bytes())):
            raise ValueError("Relocated snapshot differs from the preserved render provenance")
    else:
        provenance = {"source_snapshot":str(source.resolve()), "snapshot_run_id":snapshot["run_id"],
            "snapshot_manifest_sha256":sha256(snapshot_manifest.read_bytes()), "input_hashes":supplements,
            "scope":"Render-only supplementary market and valuation data; not writer evidence"}
        if previous.exists() and read_json(previous) != provenance:
            raise ValueError("Render supplement manifest changed")

    for relative, digest in supplements.items():
        copied = destination/relative
        if copied.exists() and sha256(copied.read_bytes()) != digest:
            raise ValueError("Copied render supplement changed: " + relative)
        if not copied.exists():
            copied.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source/relative, copied)
    if not previous.exists():
        if provenance_source:
            shutil.copy2(provenance_source, previous)
        else:
            write_json(previous, provenance)
    if saved:
        validate_hashes(destination, saved)
    manifest["render_input_hashes"] = hashes(destination)
    write_json(run/"render_input_resolution.json", {"resolved_at_utc":utc_now(),
        "method":"frozen snapshot matched by analysis and render hashes", "resolved_snapshot":str(source.resolve()),
        "historical_source_analysis":manifest["source_analysis"],
        "snapshot_manifest_sha256":sha256(snapshot_manifest.read_bytes()),
        "preserved_render_provenance_sha256":sha256(previous.read_bytes()),
        "validated_hashes":manifest["render_input_hashes"]})
    return destination/"market_snapshot.json"


def render(args, src):
    manifest = json.loads((args.run/"manifest.json").read_text(encoding="utf-8"))
    if manifest["state"] not in {"generated", "rendered"}:
        raise ValueError("Cannot render an incomplete generation")
    ticker = manifest["ticker"]
    analysis = args.run/"output"/ticker/"analysis"
    validate_hashes(analysis, manifest["analysis_hashes"])
    report = args.run/"output"/ticker/"report"
    command = [sys.executable, str(src/"create_equity_report.py"), "--company-ticker", ticker,
        "--company-name", manifest["company_name"], "--analysis-csv", str(analysis/"financial_metrics_and_forecasts.csv"),
        "--ratios-csv", str(analysis/"ratios_raw_data.csv"), "--output-dir", str(report),
        "--audited-render", "--skip-auto-fetch", "--peer-ebitda-csv", str(analysis/"peer_ebitda_comparison.csv"),
        "--peer-ev-ebitda-csv", str(analysis/"peer_ev_ebitda_comparison.csv"),
        "--analyst-names", "Zeyu Gu", "--analyst-emails", "zg2556@columbia.edu",
        "--report-date", manifest.get("report_date", manifest["started_at_utc"][:10]), "--closing-price-date", "See frozen market snapshot timestamp"]
    for section in FIELDS:
        command += ["--"+section.replace("_", "-")+"-file", str(analysis/(section+".txt"))]
    for name in ("sensitivity_analysis", "catalyst_analysis", "enhanced_news"):
        if (analysis/(name+".json")).exists():
            command += ["--"+name.replace("_", "-")+"-file", str(analysis/(name+".json"))]
    market = freeze_render_inputs(args.run, manifest)
    if (analysis/'source_notes.txt').exists():
        command += ['--data-source-text', (analysis/'source_notes.txt').read_text(encoding='utf-8')]
    if market is None and (analysis/"market_snapshot.json").exists():
        market = analysis/"market_snapshot.json"
    if market is not None:
        command += ["--market-snapshot", str(market)]
    frozen_valuation = args.run/"render_inputs"/"valuation"/"engine_valuation.json"
    if frozen_valuation.exists():
        command += ["--frozen-valuation", str(frozen_valuation)]
    # Explicit independent valuation/engine output is generated separately; never let
    # renderer silently calculate value with a default one-share denominator.
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
        env=dict(__import__("os").environ, PYTHONIOENCODING="utf-8"))
    (args.run/"render.log").write_text(completed.stdout+completed.stderr, encoding="utf-8")
    if completed.returncode or not (report/f"Audited_Equity_Report_{ticker}.html").exists():
        raise RuntimeError("Render failed; see render.log")
    validate_hashes(analysis, manifest['analysis_hashes'])
    rendered = json.loads((report/'rendered_section_texts.json').read_text(encoding='utf-8'))
    if set(rendered['sections']) != set(FIELDS):
        raise ValueError('Rendered section inventory differs from eight-agent audit')
    for section in FIELDS:
        expected = (analysis/(section+'.txt')).read_text(encoding='utf-8')
        record = rendered['sections'][section]
        if record['text'] != expected or record['sha256'] != sha256(expected):
            raise ValueError('Rendered text differs from audited input: ' + section)
    if rendered['html_sha256'] != sha256((report/f'Audited_Equity_Report_{ticker}.html').read_bytes()):
        raise ValueError('Rendered HTML hash mismatch')
    manifest.update(state="rendered", rendered_at_utc=utc_now(), report_hashes=hashes(report))
    write_json(args.run/"manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    commands = parser.add_subparsers(dest="action", required=True)
    narrative = commands.add_parser("narrate")
    narrative.add_argument("--analysis", type=Path, required=True)
    narrative.add_argument("--ticker", required=True)
    narrative.add_argument("--company-name", required=True)
    narrative.add_argument("--temperature", type=float, default=0.7)
    verification = commands.add_parser("verify")
    verification.add_argument("--final-run", type=Path, required=True)
    verification.add_argument("--reuse-review-run",type=Path,
        help="Explicitly derive from earliest valid saved reviews; not a new independent generation")
    verification.add_argument("--max-validation-repairs", type=int, choices=(0, 1, 2), default=2)
    verification.add_argument('--reasoning-effort',choices=('low','medium','high'),
        help='Explicit reasoning-model opt-in; omit temperature and use max_completion_tokens')
    rendering = commands.add_parser("render")
    for sub in (narrative, verification, rendering):
        sub.add_argument("--run", type=Path, required=True)
    for sub in (narrative, verification):
        sub.add_argument("--resume", action="store_true", help="Continue this same run; never a new stability replicate")
        sub.add_argument("--config")
        sub.add_argument("--budget", required=True)
        sub.add_argument("--ledger", type=Path, required=True)
        sub.add_argument("--max-tokens", type=int, default=4000 if sub is narrative else 14000)
    args = parser.parse_args()
    args.upstream = args.upstream.resolve()
    args.run = args.run.resolve()
    if hasattr(args, "analysis"):
        args.analysis = args.analysis.resolve()
    if hasattr(args, "final_run"):
        args.final_run = args.final_run.resolve()
    src = setup(args.upstream)
    if args.action == "narrate":
        asyncio.run(generate(args))
    elif args.action == "verify":
        asyncio.run(verify(args))
    else:
        render(args, src)


if __name__ == "__main__":
    main()
