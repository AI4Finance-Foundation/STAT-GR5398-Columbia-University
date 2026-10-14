"""Integrate actual offline parameter scenarios without changing Option A evidence.

Uses the preserved pre-scenario manuscripts for an idempotent document update.
The experiment and independent review must exist before this authoring step.
"""
import csv
import json
from pathlib import Path
from audit_io import sha256, utc_now, write_json

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT/'evidence/document_history/pre_forecast_parameter_scenarios'
SCENARIO = 'audits/forecast_parameter_scenarios/run_v2'


def read(relative):
    return json.loads((ROOT/relative).read_text(encoding='utf-8'))


def rows(relative):
    with (ROOT/relative).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def base(relative):
    return (ARCHIVE/relative).read_text(encoding='utf-8')


def save(relative, text):
    (ROOT/relative).write_text(text, encoding='utf-8')


def main():
    manifest = read(SCENARIO+'/manifest.json')
    assert manifest['status'] == 'PASS' and manifest['all_checks_passed']
    assert manifest['new_api_calls'] == manifest['new_model_calls'] == manifest['api_cost_usd'] == 0
    assert manifest['original_data_and_verifier_experiment_unchanged'] and manifest['network_disabled']
    for relative, digest in manifest['output_hashes'].items():
        assert sha256((ROOT/SCENARIO/relative).read_bytes()) == digest, relative
    assert sha256((ROOT/'scripts/forecast_parameter_scenarios.py').read_bytes()) == manifest['driver_sha256']
    independent = read('evidence/forecast_parameter_scenarios_review/review_manifest.json')
    preservation = read('evidence/forecast_parameter_scenarios_review/preservation_review.json')
    assert independent['passed'] and independent['failure_count'] == 0
    assert preservation['passed'] and preservation['all_protected_submitted_files_unchanged']
    annual = rows(SCENARIO+'/forecast_comparison.csv')
    targets = rows(SCENARIO+'/target_price_comparison.csv')
    assert len(annual) == 6 and len(targets) == 2
    f = lambda x: float(x)
    annual_table = '| Company / fiscal forecast | Default revenue, $bn | Adjusted revenue, $bn | Default EBITDA margin | Adjusted EBITDA margin |\n|---|---:|---:|---:|---:|\n'
    for row in annual:
        annual_table += f"| {row['ticker']} {row['period']} | {f(row['revenue_default'])/1e9:.3f} | {f(row['revenue_adjusted'])/1e9:.3f} | {f(row['ebitda_margin_default']):.1f}% | {f(row['ebitda_margin_adjusted']):.1f}% |\n"
    target_table = '| Company / EBITDA year | Default model target, USD | Adjusted model target, USD | Change |\n|---|---:|---:|---:|\n'
    for row in targets:
        target_table += f"| {row['ticker']} {row['valuation_period']} | {f(row['default_target_usd']):.2f} | {f(row['adjusted_target_usd']):.2f} | +{f(row['target_delta_pct']):.2f}% |\n"
    section = f'''### 5.1 Executed offline parameter scenarios

To test the proposed assumptions, I separately invoked the same pinned forecasting functions on frozen NVIDIA and AMD raw statements. Each company has a default replay and an analyst-deceleration case. NVIDIA uses sequential annual revenue growth of 30%, 20%, 10%; AMD uses 15%, 10%, 8%. Both annual contribution/EBITDA-margin input increments and the SG&A-margin change are set to zero. The adjusted contribution and SG&A ratios therefore stay at the processor's rounded latest-actual levels: 71.1% and 2.1% for NVIDIA, 49.5% and 12.0% for AMD. No precise SEC ratio was silently substituted for a rounded pipeline input.

{annual_table}

The rationale remains a deceleration from observed annual growth of 65.47% and 34.34%, without assuming the same immediate 5% rate for both businesses or further expense-ratio compression. These are analyst-selected conditional assumptions, not guidance, fitted predictions or validated forecast accuracy. All six forecast periods remain labeled as issuer fiscal years.

The same EV/EBITDA engine uses third-year EBITDA, the original historical mean multiple, original shares and the original rule that assumed net debt equals 10% of each scenario's EV. Debt amounts consequently change; the rule is held fixed, not the dollar debt. Neither discounting nor terminal growth enters this multiple method.

{target_table}

NVIDIA's terminal revenue increases 48.25% while its terminal EBITDA margin falls from 73.5% to 69.0%; AMD's revenue increases 18.03% while margin falls from 42.0% to 37.5%. With shares and the multiple unchanged, the target ratio equals the terminal revenue ratio times the EBITDA-margin ratio. Thus stronger growth and lower margins jointly produce the displayed target changes. Those prices are conditional model outputs, not endorsed fair values. Zero margin improvement does not repair the upstream definition: projected EBITDA still omits R&D and does not explicitly reconcile D&A, and AMD's adjusted 37.5% remains far above its 21.002% historical provider measure.

The default replays match all {manifest['baseline_fields_compared']} saved table fields. The scenario implementation records {manifest['independent_checks']} internal arithmetic checks; a separately written standard-library Decimal reference passes {independent['checks']} additional numerical and integrity checks. Full configurations, actual processor CSVs, engine output, annual deltas, assumptions, source hashes and logs are in `{SCENARIO}/`. The run took {manifest['runtime_seconds']:.2f} seconds including integrity hashing, with zero API calls, tokens or new API cost. Python network/subprocess rejection probes passed; this is not an OS sandbox. A second check confirms all {preservation['submitted_files']:,} protected original submission files are byte-identical, including snapshots, writer prompts, generated texts, verifier treatments, paired claim audits and the API ledger. This fulfills the forecast-audit parameter exercise and does not introduce a second research improvement or modify the Option A comparison.

'''
    report = base('docs/research_report.md')
    old = next(p for p in report.split('\n\n') if p.startswith('As analysis-only alternatives,'))
    report = report.replace(old, 'The analyst-selected alternative growth paths and unchanged initial expense ratios are now executed in a separate frozen-input parameter exercise in Section 5.1. Their source-based rationale is retained, but neither scenario is company guidance or a fitted forecast. The original Option A experiment continues to use its unchanged baseline assumptions.', 1)
    assert report.count('## 6. Independent valuation') == 1
    report = report.replace('## 6. Independent valuation', section+'## 6. Independent valuation', 1)
    save('docs/research_report.md', report)
    forecast_doc = base('docs/forecast_recommendations.md')
    first = next(p for p in forecast_doc.split('\n\n') if p.startswith('These recommendations are analysis-only.'))
    forecast_doc = forecast_doc.replace(first, 'The parameter alternatives below have now been executed as separate offline audit scenarios on the frozen NVDA/AMD inputs. The sole formal system improvement remains Option A, the verifier; its original inputs and outputs are unchanged. No fitted forecasting model, backtest or forecast-accuracy improvement is claimed. The saved results distinguish conditional parameter effects from the unchanged upstream EBITDA-definition limitation.', 1)
    forecast_doc = forecast_doc.replace('## Defensible changes to investigate', '## Rationale for the separately executed assumptions')
    forecast_doc = forecast_doc.replace('a transparent analyst-selected deceleration case would use', 'the separate analyst-selected deceleration case uses')
    forecast_doc = forecast_doc.replace('These are proposed assumptions,', 'These are executed scenario assumptions,')
    forecast_doc = forecast_doc.replace('The existing target-price sensitivity exercise should propagate stated revenue and margin scenarios through the same explicit valuation method. It will quantify how sensitive the point estimate is, not demonstrate that any scenario is probable. Changes to the earnings definition and growth assumptions would be a separate future improvement; they are not merged into the Option A before/after experiment, whose writer inputs must stay identical.', 'The existing local target-price sensitivity remains preserved. The new parameter scenarios separately propagate the executed paths through the same EV/EBITDA method, without changing its earnings definition. Neither calculation establishes the probability or predictive accuracy of a scenario. A repaired earnings model would be a separate future improvement; this parameter exercise is not merged into the Option A before/after experiment, whose writer inputs stay identical.')
    save('docs/forecast_recommendations.md', forecast_doc+'\n'+section.replace('### 5.1', '##'))
    blog = base('blog/medium_article.md')
    old = next(p for p in blog.split('\n\n') if p.startswith('The default revenue growth path is 5%, 6%, 4%'))
    new = '''The default revenue growth path is 5%, 6%, 4%, with gross margin rising one percentage point and SG&A/revenue falling half a point annually. Those are universal assumptions, not company guidance. I then executed a separate offline parameter exercise: NVIDIA growth of 30%, 20%, 10%, and AMD growth of 15%, 10%, 8%, with both gross and SG&A ratios held at the pipeline's rounded latest-actual levels. The rates describe deceleration from the companies' observed growth; they are assumptions to interrogate, not fitted forecasts.

'''+target_table+'''
The result illustrates why changing a growth rate is not enough. NVIDIA's terminal revenue rises 48.25%, but lower margin assumptions reduce the target increase to 39.17%. AMD's revenue rises 18.03%, while removing assumed margin expansion leaves a target increase of only 5.38%. The same shares, historical multiple and 10%-of-EV debt rule are used in both cases; the debt dollar amount is not held fixed. These conditional prices retain the original method's economic weaknesses.

More seriously, AMD's adjusted EBITDA margin is still 37.5%, compared with about 21% in the historical provider data. Holding margins flat did not restore the missing expense definition. I saved all four actual processor runs and independently checked the arithmetic. Both default replays match the original tables; networking is blocked and this adds no API cost. The new scenario outputs are separate from every writer prompt, verifier response and paired audit: the original verifier comparison is unchanged.'''
    blog = blog.replace(old, new, 1)
    save('blog/medium_article.md', blog)
    guide = f'''## Executed offline forecast-parameter scenarios

The separate audit in `{SCENARIO}/` executes NVDA growth 30%/20%/10% and AMD 15%/10%/8%, with zero annual margin changes. It retains the original rounded historical ratios, valuation inputs and EBITDA-definition limitations. Each company has an actual default replay and adjusted processor output; all {manifest['baseline_fields_compared']} default fields match the saved baseline. This is a conditional parameter exercise, not a second research improvement. It adds $0 API cost and leaves all original verifier inputs, outputs and scorecards unchanged.

{target_table}

Exact offline commands, after the clean-environment setup above (choose new output directories):

```powershell
$ScenarioId = [guid]::NewGuid().ToString('N').Substring(0,8)
& $RunPython scripts/forecast_parameter_scenarios.py --upstream $RunUpstream --output "reproduction/forecast-parameters-$ScenarioId"
& $RunPython scripts/independent_forecast_scenario_review.py --submission-root . --output-directory "reproduction/forecast-reference-$ScenarioId"
& $RunPython scripts/compare_forecast_scenario_review.py --submission-root . --scenario-output "reproduction/forecast-parameters-$ScenarioId" --reference-directory "reproduction/forecast-reference-$ScenarioId" --output-directory "reproduction/forecast-check-$ScenarioId"
```

Read `forecast_comparison.csv`, `target_price_comparison.csv`, `definition_caveats.csv`, the per-case configuration/processor CSV/engine JSON, and `manifest.json` under the recorded run. A separate reference is saved in `evidence/forecast_parameter_scenarios_review/independent_reference/`. Integrity review binds the original snapshots, prompts, responses, paired audits and API ledger. Neither replay is an independent LLM generation.

Medium and course PR preparation is in [the local handoff](docs/publication_preparation.md), with title/tags in `blog/medium_metadata.json` and the full description in `docs/course_pr_body.md`. Both remain unpublished; no public URL is invented.

'''
    readme = base('README.md')
    assert readme.count('## Artifact map and release') == 1
    save('README.md', readme.replace('## Artifact map and release', guide+'## Artifact map and release', 1))
    readiness = base('docs/submission_readiness.md')
    readiness += '\nThe forecast-parameter audit now includes actual offline default/adjusted pipeline outputs for NVDA and AMD, independently checked arithmetic and conditional EV/EBITDA targets. It preserves the original Option A experiment. See `audits/forecast_parameter_scenarios/run_v2/`, the updated Section 5.1, and the local Medium/PR preparation guide. The earlier analysis-only proposal is no longer the sole evidence for this course requirement.\n'
    save('docs/submission_readiness.md', readiness)
    pr = base('docs/course_pr_body.md')
    paragraph = f'\nAdds a separate offline forecast-audit scenario for NVDA and AMD: four actual default/adjusted processor runs, {manifest["baseline_fields_compared"]} baseline-field matches and independent arithmetic review. The conditional EV/EBITDA target changes are +39.17% and +5.38%, with growth paths, rounded margins and unchanged valuation/EBITDA limitations explicit. Frozen verifier inputs, outputs, paired audit labels and the $10.045071 API ledger are unchanged. This does not add a second research improvement.\n'
    pr = pr.replace('\nOption A adds', paragraph+'\nOption A adds', 1)
    save('docs/course_pr_body.md', pr)
    metadata = read('blog/medium_metadata.json')
    metadata['article_source_hash'] = sha256((ROOT/'blog/medium_article.md').read_bytes())
    metadata['hash_status'] = 'Pinned to the completed offline-scenario manuscript; publication remains deferred.'
    metadata['forecast_scenario_output_directory'] = SCENARIO+'/'
    metadata['course_pr']['body_sha256'] = sha256((ROOT/'docs/course_pr_body.md').read_bytes())
    write_json(ROOT/'blog/medium_metadata.json', metadata)
    summary = {'prepared_at_utc': utc_now(), 'scenario_directory': SCENARIO,
        'scenario_manifest_sha256': sha256((ROOT/SCENARIO/'manifest.json').read_bytes()),
        'forecast_rows': len(annual), 'target_price_comparison': targets,
        'baseline_fields_matched': manifest['baseline_fields_compared'], 'new_api_cost_usd': 0,
        'independent_review_checks': independent['checks'], 'protected_submission_files': preservation['submitted_files'],
        'original_verifier_comparison_changed': False, 'publication_authorized': False,
        'document_sha256': {name: sha256((ROOT/name).read_bytes()) for name in [
            'README.md', 'docs/research_report.md', 'docs/forecast_recommendations.md',
            'blog/medium_article.md', 'docs/submission_readiness.md', 'docs/course_pr_body.md']},
        'next': 'Build and visually inspect the current PDF; finish independent review and integrity comparison; update acceptance/inventory and re-export locally.'}
    write_json(ROOT/'evidence/forecast_scenario_release.json', summary)
    print(json.dumps({'source_updated': True, 'scenario': SCENARIO, 'target_rows': targets}, indent=2))


if __name__ == '__main__':
    main()
