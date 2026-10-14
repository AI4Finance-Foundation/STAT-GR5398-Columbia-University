"""Validate saved experimental evidence without new provider or model calls.

Passing these integrity/scope checks does not certify factual correctness or
publication. Those open requirements are explicit in the resulting manifest.
"""
import argparse,csv,json,re,subprocess,os
from collections import Counter
from pathlib import Path
from html.parser import HTMLParser
from audit_io import sha256,write_json,utc_now
from grounding_audit import load_audit,load_inputs,source_units,FIELDS,metrics
from update_coverage_status import ACTIVE_SNAPSHOTS
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def rows(p):return list(csv.DictReader(Path(p).open(encoding='utf-8-sig',newline='')))
class Sections(HTMLParser):
    def __init__(self):super().__init__(convert_charrefs=True);self.active=None;self.texts={};self.images=[]
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if 'data-agent-section' in a:self.active=a['data-agent-section'];self.texts[self.active]=''
        if tag=='img':self.images.append(a['src'])
    def handle_endtag(self,tag):
        if tag=='div' and self.active:self.active=None
    def handle_data(self,text):
        if self.active:self.texts[self.active]+=text

def validate_submission_relocation_evidence(version='v9_1'):
    """Validate the moved-submission proof without relying on its temporary paths.

    The tested copy contains the real runtime dependency closure, including frozen
    snapshots and saved paid calls. The older output-directory-only replay is not
    migration evidence. Only executed runtime scripts must match today's code;
    unrelated authoring utilities copied incidentally are historical inventory.
    """
    assert version in {'v8_2','v9','v9_1'}
    relative='evidence/reproducibility/true_submission_relocation/final_'+version
    directory=ROOT/relative;proof=read(directory/'manifest.json')
    assert proof['verifier_version']==version
    assert proof['status']=='passed' and proof['new_paid_calls']==0 and proof['new_paid_cost_usd']==0
    assert proof['network_disabled_during_verification_and_render'] and proof['original_and_copied_inputs_unchanged']
    assert proof['upstream_sha']=='2717499b8e30f242640af08c4ad9afd1113c2d45'
    normalized=lambda p:str(p).replace('\\','/').rstrip('/').lower()
    original=normalized(proof['source_submission']);copied=normalized(proof['copied_submission'])
    assert original!=copied and not copied.startswith(original+'/'), 'Submission itself was not relocated'
    dependencies=proof['dependency_copy_hashes']
    def digest(relative_path):
        relative_path=relative_path.replace('\\','/')
        assert not Path(relative_path).is_absolute() and '..' not in Path(relative_path).parts
        path=ROOT/relative_path
        if os.name=='nt':path=Path('\\\\?\\'+str(path.resolve()))
        return sha256(path.read_bytes())
    runtime={'scripts/experiment.py':'experiment_script_sha256',
             'scripts/revalidate_reasoning_offline.py':'replay_script_sha256',
             'scripts/validate_submission_relocation.py':'script_sha256'}
    for path,key in runtime.items():
        assert proof[key]==dependencies[path]==digest(path), 'Migration proof runtime changed: '+path
    assert dependencies['scripts/audit_io.py']==digest('scripts/audit_io.py')
    required={'config/budget.reasoning_followup.json','evidence/costs.jsonl',
              f'evidence/verifier_versions/verifier_agent_{version}.py'}
    assert required<=set(dependencies)
    validated_dependencies=0
    for path,expected in dependencies.items():
        if not path.startswith('scripts/'):
            assert digest(path)==expected, 'Migration proof input changed: '+path
            validated_dependencies+=1
    # Guard logs are saved in the submission; old temporary directories need not exist.
    isolation=[]
    for name in ['worker_isolation.json']+[f'renderer_isolation_{i}.json' for i in range(1,4)]:
        log=read(directory/name)
        assert log['probe_denials']==2 and not log['unexpected_original_root_attempts']
        assert len(log['guard_events'])==2
        assert {(item['operation'],normalized(item['path'])) for item in log['guard_events']}=={
            ('open',original+'/readme.md'),('stat',original+'/snapshots')}, 'Missing actual old-root isolation probes'
        if name=='worker_isolation.json':
            assert {normalized(path).rsplit('/',1)[-1] for path in log['renderer_logs']}=={
                f'renderer_isolation_{i}.json' for i in range(1,4)}
        isolation.append({'path':relative+'/'+name,'sha256':digest(relative+'/'+name)})
    assert len(proof['records'])==3 and {r['ticker'] for r in proof['records']}=={'NVDA','AMD','INTC'}
    records=[]
    for record in proof['records']:
        ticker=record['ticker'];run_id=ticker+'_final_reasoning_'+version;base_id=ticker+'_baseline_1'
        run=ROOT/'runs'/run_id;manifest=read(run/'manifest.json')
        baseline=read(ROOT/'runs'/base_id/'manifest.json')
        prior_id=ticker+'_final_reasoning_v8_1';prior=read(ROOT/'runs'/prior_id/'manifest.json')
        assert manifest['input_hashes']==baseline['input_hashes']==prior['input_hashes']
        for source_id in [base_id,prior_id]:
            assert 'runs/'+source_id+'/manifest.json' in dependencies
        assert record['exact_analysis_files']==len(manifest['analysis_hashes'])
        for path,expected in manifest['analysis_hashes'].items():
            assert digest(f'runs/{run_id}/output/{ticker}/analysis/{path}')==expected
        report=run/'output'/ticker/'report'
        assert record['html_sha256']==sha256((report/f'Audited_Equity_Report_{ticker}.html').read_bytes())
        sections=read(report/'rendered_section_texts.json')['sections']
        assert record['exact_sections']==8 and set(sections)==set(FIELDS)
        for section in FIELDS:
            actual=(run/'output'/ticker/'analysis'/(section+'.txt')).read_text(encoding='utf-8')
            assert sections[section]['sha256']==sha256(actual) and sections[section]['text']==actual
        pngs={str(path.relative_to(report)).replace('\\','/'):sha256(path.read_bytes()) for path in report.rglob('*.png')}
        assets={path:expected for path,expected in pngs.items() if path.startswith('assets/')}
        assert record['exact_chart_files']==pngs and len(pngs)==6
        assert record['exact_html_chart_assets']==assets and len(assets)==3
        render_hashes=record['exact_render_input_hashes']
        assert render_hashes==manifest['render_input_hashes']==baseline['render_input_hashes']
        assert {'manifest.json','market_snapshot.json','valuation_inputs.json','valuation/engine_valuation.json'}<=set(render_hashes)
        for path,expected in render_hashes.items():
            assert digest(f'runs/{run_id}/render_inputs/{path}')==expected
            assert dependencies[f'runs/{base_id}/render_inputs/{path}']==expected
        resolution=record['relocation']
        assert resolution['method']=='frozen snapshot matched by analysis and render hashes'
        assert normalized(resolution['resolved_snapshot']).startswith(copied+'/snapshots/')
        assert resolution['historical_source_analysis']==manifest['source_analysis']
        assert resolution['validated_hashes']==render_hashes
        lineage=read(run/'render_inputs/manifest.json')
        assert resolution['preserved_render_provenance_sha256']==render_hashes['manifest.json']
        assert resolution['snapshot_manifest_sha256']==lineage['snapshot_manifest_sha256']
        snapshot='snapshots/'+normalized(resolution['resolved_snapshot']).rsplit('/',1)[-1]
        # Resolve recorded snapshot names case-insensitively without reading old absolute paths.
        matches=[key for key in dependencies if key.lower()==snapshot+'/manifest.json']
        assert len(matches)==1 and dependencies[matches[0]]==resolution['snapshot_manifest_sha256']
        snapshot=matches[0].rsplit('/',1)[0]
        for path,expected in baseline['input_hashes'].items():
            assert dependencies[snapshot+'/analysis/'+path]==expected
        assert record['market_metrics']==read(run/'render_inputs/market_snapshot.json')['metrics']
        assert record['valuation_results']==read(run/'render_inputs/valuation/engine_valuation.json')['results']
        assert record['preserved_report_date']==manifest.get('report_date',manifest['started_at_utc'][:10])
        assert record['future_execution_time'][:4]=='2099' and record['future_execution_time'][:10]!=record['preserved_report_date']
        call_ids=[]
        for section in FIELDS:
            reused=read(run/'verifier'/section/'reused_review.json')
            assert not reused['new_model_generation'] and reused['incremental_model_cost_usd']==0
            original_directory=reused['source_directory'].replace('\\','/')
            marker='/runs/'+prior_id+'/'
            assert marker in original_directory
            source_relative='runs/'+prior_id+'/'+original_directory.split(marker,1)[1]
            for name,expected in reused['copied_file_hashes'].items():
                assert digest(f'runs/{run_id}/verifier/{section}/{name}')==expected==dependencies[source_relative+'/'+name]
            call_ids.append(reused['source_call_id'])
        assert record['source_paid_call_ids']==call_ids and len(set(call_ids))==8
        records.append({'ticker':ticker,'identical_sections':8,'identical_html_sha256':record['html_sha256'],
            'identical_chart_files':6,'identical_html_assets':3,'market_and_valuation_sources_identical':True,
            'preserved_report_date':record['preserved_report_date'],'source_paid_call_ids':call_ids})
    return {'passed':True,'verifier_version':version,'proof_path':relative+'/manifest.json','proof_sha256':digest(relative+'/manifest.json'),
        'scope':'Actual submission runtime dependencies moved, with original-root access blocked in worker and renderer children; not merely relocated output. Supporting documents are outside the runtime closure.',
        'new_model_calls':0,'new_model_cost_usd':0,'current_runtime_script_hashes_match':True,
        'copied_input_files_validated_against_current_release':validated_dependencies,
        'other_copied_scripts':'Historical copy inventory only; unexecuted authoring utilities are not replay dependencies.',
        'isolation_logs':isolation,'records':records,
        'temporary_directories_required_for_this_check':False}

def validate_semantic_followup():
    study=read(ROOT/'evidence/semantic_study_summary.json')
    assert study['version']=='v9_1' and study['study_scope_complete']
    assert not study['human_review_performed'] and not study['held_out_evaluation']
    for relative,expected in study['input_hashes'].items():
        assert sha256((ROOT/relative.replace('\\','/')).read_bytes())==expected
    assert sha256((ROOT/'docs/verifier_semantic_v9_followup.md').read_bytes())==study['study_sha256']
    overall=lambda path:next(row for row in rows(path) if row['ticker']=='ALL' and row['section']=='ALL')
    assert study['scorecard']==overall(ROOT/'audits/grounding/comparison_reasoning_v9_1/before_after_scorecard.csv')
    assert study['warnings']==overall(ROOT/'audits/grounding/annotation_reasoning_v9_1/annotation_scorecard.csv')
    patch_dir=ROOT/'evidence/verifier_versions/patches_v9_1';patch=read(patch_dir/'manifest.json')
    assert patch['clean_apply']=='passed' and patch['offline_test_returncode']==0
    assert patch['upstream_sha']=='2717499b8e30f242640af08c4ad9afd1113c2d45'
    for name,expected in patch['patch_sha256'].items():
        assert sha256((patch_dir/name).read_bytes())==expected
    browser=[]
    for ticker in ['NVDA','AMD','INTC']:
        run_id=ticker+'_final_reasoning_v9_1'
        qa=read(ROOT/f'evidence/browser_semantic_v9_1/{ticker}/qa.json')
        assert qa['html'].replace('\\','/').endswith(f'/runs/{run_id}/output/{ticker}/report/Audited_Equity_Report_{ticker}.html')
        assert qa['passed'] and qa['sections_identical']
        assert not qa['broken_images'] and not qa['javascript_errors'] and not qa['blocked_remote_requests']
        assert not qa.get('unexplained_missing_notices',qa['optional_missing_notices']) and qa['geometry']['scrollWidth']<=qa['geometry']['pageWidth']
        assert len(qa['images'])==3 and all(item['complete'] and item['width']>0 and item['height']>0 for item in qa['images'])
        report=ROOT/'runs'/run_id/'output'/ticker/'report'
        for image in qa['images']:
            assert not image['src'].startswith(('http:','https:')) and (report/image['src']).is_file()
        current_hash=sha256((report/f'Audited_Equity_Report_{ticker}.html').read_bytes())
        assert read(report/'rendered_section_texts.json')['html_sha256']==current_hash
        if 'html_sha256' in qa:assert qa['html_sha256']==current_hash
        browser.append({'run_id':run_id,'qa_path':f'evidence/browser_semantic_v9_1/{ticker}/qa.json',
            'html_sha256':current_hash,'local_images':len(qa['images']),
            'qa_records_html_hash':'html_sha256' in qa,'scope':'Browser DOM/text/image check; visual review is recorded separately.'})
    visual=read(ROOT/'evidence/browser_semantic_v9_1/visual_review.json')
    assert visual['passed_for_requested_visual_scope'] and not visual['human_review_performed'] and not visual['semantic_certification']
    viewed={r['path']:r['sha256'] for r in visual['actual_images_opened_with_view_image']}
    assert len(viewed)==visual['actual_images_opened_count']==32
    for relative,expected in viewed.items():assert sha256((ROOT/relative).read_bytes())==expected
    for reused in visual['exact_hash_reuse']:
        assert sha256((ROOT/reused['path']).read_bytes())==reused['sha256']==viewed[reused['visually_reviewed_identical_reference']]
    for source in visual['html_sources']:
        assert sha256((ROOT/source['path']).read_bytes())==source['sha256']
        assert sha256((ROOT/source['qa_json']).read_bytes())==source['qa_sha256']
    assert sha256((ROOT/visual['range_qa_path']).read_bytes())==visual['range_qa_sha256']
    assert read(ROOT/visual['range_qa_path'])['passed']
    assert sha256((ROOT/'scripts/qa_semantic_ranges.py').read_bytes())==visual['range_script_sha256']
    reproduced=read(ROOT/'evidence/reproducibility/semantic_release_metrics.json')
    assert reproduced['passed'] and reproduced['network_disabled'] and len(reproduced['csv_checks'])==24
    for item in reproduced['csv_checks']:assert item['byte_identical'] and sha256((ROOT/item['file']).read_bytes())==item['sha256']
    return {'study':study,'clean_patch':patch,'browser_reports':browser,'visual_review':visual,'recomputed_metrics':reproduced}

def main():
    evidence={};coverage=rows(ROOT/'coverage_universe.csv')
    assert {r['ticker'] for r in coverage}=={'NVDA','AMD','INTC','AVGO','QCOM'}
    assert all(int(r['actual_peer_count'])>=2 and r['ticker'] not in r['actual_peer_tickers'].split(';') for r in coverage)
    evidence['formal_companies']=len(coverage);evidence['all_actual_peer_counts']={r['ticker']:r['actual_peer_count'] for r in coverage}
    renders=[]
    for ticker in ACTIVE_SNAPSHOTS:
        versions = ['baseline_1','final_v6','final_v7_1'] + (['baseline_2','baseline_3'] if ticker=='NVDA' else [])
        if ticker in {'NVDA','AMD','INTC'}:
            versions.extend(['final_reasoning_v8_2','final_reasoning_v9_1'])
        for version in versions:
            run=ROOT/'runs'/f'{ticker}_{version}';m=read(run/'manifest.json');assert m['state']=='rendered'
            parent=ROOT/'runs'/f'{ticker}_baseline_1' if version.startswith('final_') else run
            _,inputs=load_inputs(run,parent);report=run/'output'/ticker/'report';path=report/f'Audited_Equity_Report_{ticker}.html'
            rendered=read(report/'rendered_section_texts.json');assert sha256(path.read_bytes())==rendered['html_sha256']
            parser=Sections();parser.feed(path.read_text(encoding='utf-8'));assert set(parser.texts)==set(FIELDS)
            for s,item in inputs.items():assert parser.texts[s]==item['text']==rendered['sections'][s]['text']
            for image in parser.images:assert not image.startswith(('http:','https:')) and (report/image).is_file()
            assert '\ufffd' not in path.read_text(encoding='utf-8')
            renders.append({'run_id':m['run_id'],'html_sha256':sha256(path.read_bytes()),'sections':8,'local_images':len(parser.images),'exact_source_text':True})
    evidence['reports']=renders
    data=rows(ROOT/'audits/data/data_audit.csv');assert len(data)>=20
    counts=Counter(r['ticker'] for r in data);assert counts['NVDA']>=10 and counts['AMD']>=10;evidence['data_audit_counts']=dict(counts)
    semantic_pairs=read(ROOT/'config/grounding_reasoning_v9_1_pairs.json')
    assert len(semantic_pairs)==3
    assert {read(ROOT/pair['after']/'manifest.json')['run_id'] for pair in semantic_pairs}=={
        ticker+'_final_reasoning_v9_1' for ticker in ['NVDA','AMD','INTC']}
    pairs=(read(ROOT/'config/grounding_v6_pairs.json')+read(ROOT/'config/grounding_v7_1_pairs.json')+
           read(ROOT/'config/grounding_reasoning_v8_2_pairs.json')+semantic_pairs);audit_records=[]
    for pair in pairs:
        before,bc=load_audit(ROOT/pair['before']);after,ac=load_audit(ROOT/pair['after']);assert before['evidence_hashes']==after['evidence_hashes']
        for side,manifest,claims in [('before',before,bc),('after',after,ac)]:
            run=ROOT/'runs'/manifest['run_id'];_,inputs=load_inputs(run,ROOT/'runs'/before['run_id'])
            assert set(manifest['source_hashes'])==set(FIELDS)
            assert {s:i['metadata']['source_sha256'] for s,i in inputs.items()}==manifest['source_hashes']
            coverage_rows=rows(ROOT/pair[side]/'text_coverage.csv')
            expected_units={(section,str(unit['unit_id'])):unit for section,item in inputs.items() for unit in source_units(item['text'])}
            assert len(coverage_rows)==len(expected_units)
            assert {(row['section'],row['unit_id']) for row in coverage_rows}==set(expected_units)
            for row in coverage_rows:
                unit=expected_units[(row['section'],row['unit_id'])]
                assert row['source_quote']==unit['text'] and int(row['source_start'])==unit['start'] and int(row['source_end'])==unit['end']
            audit_records.append({'ticker':manifest['ticker'],'side':side,'source_units':sum(len(source_units(i['text'])) for i in inputs.values()),**metrics(claims)})
    evidence['audits']=audit_records
    for ticker,snapshot in ACTIVE_SNAPSHOTS.items():
        rec=rows(ROOT/f'snapshots/{snapshot}/valuation/valuation_reconciliation.csv');assert len(rec)==12 and all(r['verdict']=='MATCH' for r in rec)
    evidence['independent_reconciliation_components']=84
    for ticker in ['NVDA','AMD','INTC','AVGO','QCOM']:
        folder='audits/sensitivity' if ticker in {'AVGO','QCOM'} else 'audits/target_price_sensitivity'
        assert len(rows(ROOT/f'{folder}/{ticker}/target_price_sensitivity.csv'))==25
    evidence['formal_target_price_scenarios']=125
    stability=read(ROOT/'audits/stability/NVDA_adjudicated/stability_summary.json');assert stability['generation_count']==3 and stability['independent_writer_calls']==24
    evidence['stability']=stability
    dom=read(ROOT/'evidence/browser_release_ui7/dom_checks.json');assert len(dom)==14
    for r in dom:
        assert r['scrollWidth']<=r['width'] and not r['overflow'] and not r['replacementCharacter']
        assert all(i['complete'] and i['width']>0 for i in r['images'])
        ticker=r['run'].split('_')[0]
        for s in r['sections']:
            actual=(ROOT/'runs'/r['run']/'output'/ticker/'analysis'/(s['section']+'.txt')).read_text(encoding='utf-8')
            assert s['text']==actual and s['hash']==sha256(actual)
    evidence['browser_reports_checked']=14
    reasoning_dom = []
    for ticker in ['NVDA','AMD','INTC']:
        check = read(ROOT/f'evidence/browser_reasoning_v8_2/{ticker}_dom_check.json')
        assert check['run_id'] == f'{ticker}_final_reasoning_v8_2'
        report = ROOT/'runs'/check['run_id']/'output'/ticker/'report'/f'Audited_Equity_Report_{ticker}.html'
        assert sha256(report.read_bytes()) == check['html_sha256']
        assert check['scrollWidth'] <= check['width'] and not check['section_overflow'] and not check['replacementCharacter']
        assert len(check['section_source_sha256']) == 8
        for section, expected in check['section_source_sha256'].items():
            actual = (ROOT/'runs'/check['run_id']/'output'/ticker/'analysis'/(section+'.txt')).read_text(encoding='utf-8')
            assert sha256(actual) == expected
        assert len(check['images']) == 3 and all(i['complete'] and i['naturalWidth'] > 0 for i in check['images'])
        reasoning_dom.append(check['run_id'])
    evidence['reasoning_browser_reports_checked'] = reasoning_dom
    study = read(ROOT/'evidence/reasoning_study_summary.json')
    assert study['study_scope_complete'] and not study['human_review_performed']
    for relative, expected in study['input_hashes'].items():
        assert sha256((ROOT/relative.replace('\\','/')).read_bytes()) == expected
    assert sha256((ROOT/'docs/verifier_reasoning_v8_followup.md').read_bytes()) == study['study_sha256']
    evidence['reasoning_study'] = study
    semantic=validate_semantic_followup()
    evidence['semantic_study']=semantic['study'];evidence['semantic_clean_patch']=semantic['clean_patch']
    evidence['semantic_browser_reports_checked']=semantic['browser_reports']
    evidence['semantic_browser_visual_review']=semantic['visual_review']
    evidence['semantic_recomputed_metrics']=semantic['recomputed_metrics']
    patch_dir = ROOT/'evidence/verifier_versions/patches_v8_2'
    patch = read(patch_dir/'manifest.json')
    assert patch['clean_apply'] == 'passed' and patch['offline_test_returncode'] == 0
    assert patch['upstream_sha'] == '2717499b8e30f242640af08c4ad9afd1113c2d45'
    for name, expected in patch['patch_sha256'].items():
        assert sha256((patch_dir/name).read_bytes()) == expected
    evidence['reasoning_clean_patch'] = patch
    evidence['relocated_reasoning_replay'] = validate_submission_relocation_evidence()
    evidence['relocated_v8_2_replay'] = validate_submission_relocation_evidence('v8_2')
    old_replay_path=ROOT/'evidence/reproducibility/reasoning_replay/manifest.json'
    evidence['historical_output_only_replay']={
        'path':'evidence/reproducibility/reasoning_replay/manifest.json',
        'sha256':sha256(old_replay_path.read_bytes()),'source_script_sha256':read(old_replay_path)['source_script_sha256'],
        'accepted_as_submission_relocation':False,
        'scope':'Historical output-directory replay. Its older script hash is not compared with the current V8.2/V9-capable replay script and is not migration acceptance evidence.'}
    historical_migration=ROOT/'evidence/reproducibility/true_submission_relocation/verified/manifest.json'
    evidence['historical_v8_2_submission_migration']={
        'path':'evidence/reproducibility/true_submission_relocation/verified/manifest.json',
        'sha256':sha256(historical_migration.read_bytes()),'accepted_as_current_runtime_proof':False,
        'scope':'Preserved successful V8.2 submission migration before the driver and harness gained V9.1 selection. Current-runtime acceptance uses final_v8_2 and final_v9_1.'}
    # The user submits only the selected PDF; source/build history stays local.
    from validate_submission_selection import validate as validate_selected_pdf
    evidence['selected_research_pdf']=validate_selected_pdf()
    evidence['pdf_pages']=evidence['selected_research_pdf']['pages']
    cost=read(ROOT/'audits/resources/cost_summary.json');assert cost['cost_usd']<20 and not cost['unsettled_call_ids'];evidence['cost']=cost
    secret=read(ROOT/'evidence/secret_scan.json');assert secret['passed'];evidence['credential_scan']=secret
    # This saved pre-export check is a historical comparison. The export script
    # rechecks all final blobs after any subsequent manifest/inventory changes.
    attributes=ROOT/'.gitattributes'
    assert '* -text' in attributes.read_text(encoding='utf-8').splitlines()
    assert '*.pdf -diff' in attributes.read_text(encoding='utf-8').splitlines()
    byte_path=ROOT/'evidence/reproducibility/course_git_bytes_after.json'
    byte_check=read(byte_path)
    assert byte_check['passed'] and byte_check['mismatch_count']==0
    assert byte_check['files']==len(byte_check['all_blob_bindings'])>0
    assert all(r['exact_bytes'] and r['raw_git_blob_sha1']==r['index_git_blob_sha1'] for r in byte_check['all_blob_bindings'])
    assert byte_check['script_sha256']==sha256((ROOT/'scripts/check_course_git_bytes.py').read_bytes())
    evidence['git_byte_preservation']={'attributes_sha256':sha256(attributes.read_bytes()),
        'pre_export_check':'evidence/reproducibility/course_git_bytes_after.json',
        'pre_export_check_sha256':sha256(byte_path.read_bytes()),'pre_export_files':byte_check['files'],
        'scope':'Saved initial byte check, not a claim that later inventory/manifest files were already checked. prepare_course_diff.py separately verifies every final staged blob and actually applied clean-index blob before successful export.'}
    from validate_forecast_scenario_release import validate as validate_forecast_parameter_followup
    evidence['forecast_parameter_scenarios']=validate_forecast_parameter_followup()
    from validate_medium_publication import validate as validate_publication
    evidence['medium_publication']=validate_publication()
    # Scope inventory uses Git ignore rules, without modifying the shared index.
    out=subprocess.run(['git','ls-files','--cached','--others','--exclude-standard','-z','--','.'],cwd=ROOT,capture_output=True,check=True).stdout.decode('utf-8')
    files=sorted(set(x for x in out.split('\0') if x));inventory=[]
    for relative in files:
        file=ROOT/relative
        if relative in {'ARTIFACTS.csv','evidence/delivery_validation.json','evidence/progress/checkpoint.json','evidence/course_submission_diff.json','docs/course_submission_diff_stat.txt'}:continue
        if os.name=='nt':file=Path('\\\\?\\'+str(file))
        assert file.is_file(), 'Inventory path could not be read: '+relative
        assert file.stat().st_size<100*1024*1024, 'GitHub per-file limit: '+relative
        inventory.append({'path':relative,'bytes':file.stat().st_size,'sha256':sha256(file.read_bytes()),'category':relative.split('/')[0]})
    with (ROOT/'ARTIFACTS.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['path','bytes','sha256','category']);w.writeheader();w.writerows(inventory)
    evidence.update(checked_utc=utc_now(),technical_scope_checks_passed=True,full_assignment_complete=False,
        publication=read(ROOT/'config/publication_status.json'),inventory_files=len(inventory),inventory_bytes=sum(r['bytes'] for r in inventory),
        inventory_exclusions='Self, validation result, mutable checkpoint and the two local diff metadata files excluded to avoid circular hashes; Git-ignored local runtime/reproduction/cache files are not release artifacts.',
        limitations=('Machine labels and visual QA do not prove truth. Historical V6 has 48 contradicted after assertions (8 unflagged); historical V8.2 has 16 (12 unflagged) under its then-current audit. '
            f"The current V9.1 study reports {semantic['study']['scorecard']['after_contradicted']} contradicted after assertions, "
            f"{semantic['study']['warnings']['unflagged_contradicted']} unflagged contradictions, "
            f"{semantic['study']['scorecard']['after_unsupported']} unsupported assertions and "
            f"{semantic['study']['scorecard']['after_numeric_unverifiable']} unverifiable numeric assertions. "
            'These are machine-assisted measurements on the development cohort, not human-validated truth or held-out evidence. Historical errors remain preserved as experimental observations. The user requested accurate delivered numbers; experimental outputs must remain identified as imperfect. Medium is published by the user and verified public; the course master PR remains pending user submission from Mac.'))
    write_json(ROOT/'evidence/delivery_validation.json',evidence)
    print(json.dumps({k:evidence[k] for k in ['technical_scope_checks_passed','full_assignment_complete','formal_companies','independent_reconciliation_components','formal_target_price_scenarios','browser_reports_checked','pdf_pages','inventory_files','inventory_bytes']},indent=2))
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--relocation-only',action='store_true',help='Read-only check of actual submission migration evidence; do not rebuild inventory or delivery manifest')
    parser.add_argument('--relocation-version',choices=['v8_2','v9','v9_1'],default='v9_1',help='Proof version for --relocation-only; full delivery checks V9.1 and V8.2')
    args=parser.parse_args()
    if args.relocation_only:print(json.dumps(validate_submission_relocation_evidence(args.relocation_version),indent=2))
    else:main()
