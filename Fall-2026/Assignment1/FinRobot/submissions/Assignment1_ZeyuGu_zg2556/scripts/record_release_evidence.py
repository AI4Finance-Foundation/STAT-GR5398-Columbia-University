"""Record completed agent visual inspection and archive small replay evidence.

Run only after the current 13 rendered PDF pages and HTML chart areas have been
visually inspected by the task agent. This is not human financial validation.
"""
import csv,json,shutil
from pathlib import Path
from audit_io import write_json,sha256,utc_now
ROOT=Path(__file__).resolve().parents[1]
def main():
    build=json.loads((ROOT/'output/pdf/research_report_build.json').read_text())
    assert build['pages']==13
    images=[Path(p) for p in build['page_images']];assert all(p.exists() for p in images)
    write_json(ROOT/'evidence/pdf_release_visual_qa.json',{'checked_utc':utc_now(),'pdf_sha256':build['pdf_sha256'],
        'source_sha256':build['source_sha256'],'pages_inspected':13,'reviewer':'Codex task agent, not a human reviewer',
        'findings':'All 13 Poppler page images inspected: readable text/tables and three figures, repeated table headers, page numbers, no clipped text, overlapping content, missing images or replacement glyphs observed.',
        'page_image_sha256':{p.name:sha256(p.read_bytes()) for p in images},
        'limitation':'Visual QA and checks of report summary values do not certify generated financial narrative; residual claim errors are disclosed.'})
    build['visual_qa_status']='All 13 current pages inspected by Codex task agent; no observed layout defects; no human financial validation.'
    write_json(ROOT/'output/pdf/research_report_build.json',build)
    for name in ['clean7','clean7_v2','clean7_v3']:
        src=ROOT/'reproduction'/name;dest=ROOT/'evidence/reproducibility'/name;dest.mkdir(parents=True,exist_ok=True)
        for file in src.iterdir():
            if file.is_file() and file.suffix in {'.log','.json'}:shutil.copyfile(file,dest/file.name)
        for m in (src/'snapshots').glob('*/manifest.json'):
            if json.loads(m.read_text())['status']=='failed':shutil.copyfile(m,dest/(m.parent.name+'_failure_manifest.json'))
    write_json(ROOT/'evidence/browser_release_ui7/visual_review.json',{'checked_utc':utc_now(),'reviewer':'Codex task agent',
        'method':'CUA browser DOM checks on 14 baseline/final pages; saved full-page screenshots; visual inspection of report header/notice and all seven distinct final chart areas. Baseline/final charts share frozen data.',
        'result':'No page-width overflow, missing linked images or replacement glyphs; all eight DOM texts hash-match source. All seven chart areas legible with actual/forecast labels.',
        'source_limitations':'AVGO/QCOM EPS-PE figure intentionally absent with visible unavailable label because subject PE is missing. Their peer-multiple plot includes only supplied peer observations; no fabricated subject series.',
        'semantic_certification':False})
    version=ROOT/'evidence/verifier_versions/manifest.json';m=json.loads(version.read_text())
    m['versions']={p.name:sha256(p.read_bytes()) for p in sorted(version.parent.glob('verifier_agent_v*.py'))}
    m['canonical_version']='6-reject-structural-only-evidence';m['canonical_pairs']='config/grounding_v6_pairs.json';m['status']='All seven final_v6 runs rendered; three full independent before/after inventories complete; residual semantic errors remain.';write_json(version,m)
    lineage=ROOT/'docs/field_lineage.csv'
    with lineage.open(encoding='utf-8-sig',newline='') as f:r=csv.DictReader(f);fields=r.fieldnames;entries=list(r)
    entries=[r for r in entries if r['field'] not in {'Public adapter annual metrics','Public adapter price/shares/default multiple'}]
    common={'layer':'deterministic public source adapter','upstream_source':'scripts/public_semiconductor_pipeline.py','upstream_locator':'build and SEC exact annual_fact/iXBRL match','upstream_commit':'2717499b8e30f242640af08c4ad9afd1113c2d45','runtime_scope':'AVGO_public_v1;QCOM_public_v1','status':'Real frozen public inputs; WRDS excluded','required_run_evidence':'raw SEC/Yahoo bytes, exact context/pointers and source hashes','audited_mode_control':'Frozen input provenance appended to both public writer prompts and displayed in final HTML','compatibility_path':'Public SEC/Yahoo input adapter after actual FMP402; same forecast/valuation/agents'}
    for field,formula,units,risk,evidence in [
      ('Public adapter annual metrics','SEC latest FY2025 10-K comparative statements FY2023-2025; EBITDA=GAAP operating income+cashflow D&A; FCF=OCF-capex','USD; exact fiscal periods and latest-10K restated comparative basis','AVGO D&A includes ROU amortization; subject EBITDA differs from FMP peers; FCF is not proven unlevered','snapshots/{AVGO,QCOM}_public_v1/analysis/financial_period_provenance.csv; source_notes.txt; raw_input_hashes'),
      ('Public adapter price/shares/default multiple','Yahoo Oct9 2026 raw close; SEC cover shares; missing subject multiple => explicit12x +/-3x engine fallback','USD/share; AVGO shares2026-08-28;QCOM2026-07-27','Stale shares; absent analyst grades/targets; default12x not an observation; net debt10%EV assumption','snapshots/{AVGO,QCOM}_public_v1/market_snapshot.json; valuation/valuation_assumptions.csv; evidence/public_market/yahoo_2026-10-11')]:
        entries.append(dict(common,field=field,origin_or_formula=formula,units_or_time_basis=units,upstream_default_or_risk=risk,runtime_evidence=evidence))
    with lineage.open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(entries)
    history=ROOT/'docs/verifier_version_history.md';text=history.read_text()
    if '## V4-V6 final experiment' not in text:
        text+='''\n\n## V4-V6 final experiment\n\nV4 permits only a finite list of complete uncovered connector/heading spans, retaining exact offsets. Omitted substantive facts still fail. See `verifier_v4_validation_change.md`. Real V4 failures showed that evidence-supported rephrasing with action=correct was being rejected solely because the model labeled the original Supported. V5 allows that action when all source/citation/numeric checks pass; unvalidated proposed rewrites still cannot replace original assertions. Saved reviews are reused only after validating source/evidence hashes, original prompt prefix, provider response and budget settlement; reuse is explicitly recorded, not a new generation.\n\nV6 adds a substantive-line guard: a company identity line, blank line, heading, separator or table header alone cannot establish a business/financial proposition. Supported or corrected claims with only such references fail closed into an unverified original quote (or withholding for rejected contradictions). This narrow guard fixes an observed citation failure; a substantive line can still be semantically irrelevant or point at a wrong company/year.\n\nAll seven `*_final_v6` outputs are real verifier treatments. Five inherit prior paid reviews and rerun deterministic validation; AVGO/QCOM have new GPT-4.1 reviews. Exact version hashes are in the source archive manifest. The three-company complete independent audit finds 358/1590 supported and48 contradicted assertions. Warning recall is93.99%, precision78.78%;74 problematic claims are not warned. No improvement is inferred from a software test alone. Baseline and treated narrative errors remain preserved as experimental evidence.\n\nFinal patches apply on a clean pinned checkout;101 tests and6 subtests passed, then seven saved data pipelines and14 saved-text renders ran with networking disabled. The final renderer adds the same research-status notice to baseline and final and displays table NaN as N/A; none of the audited section text changes.\n'''
        history.write_text(text,encoding='utf-8')
    for relative,note in [
      ('docs/evaluation_method.md','Canonical final results use the explicit machine-adjudication protocol in `machine_adjudication_protocol.md`, not the early API-judge proposals described below. Exact paired directories are in `config/grounding_v6_pairs.json`; compare/annotation outputs are `comparison_v6` and `annotation_v6`. Saved-response V6 revalidation has zero incremental API cost but original allocated review costs remain reported. `stability_adjudication.py` supplies the canonical explicit three-run reading. All old API attempts remain development evidence.'),
      ('docs/coverage_methodology.md','Current execution update: all five formal companies now have actual baseline/final reports. AVGO/QCOM use public SEC/Yahoo adapters and cached FMP peers after entitlement denial; WRDS observations are excluded. Actual peers and common-window returns are in `coverage_universe.csv`; any planned or pending source-collection language below records earlier selection work.'),
      ('docs/measured_grounding_stability_progress.md','Historical progress note retained. Current complete results supersede partial counts below:1478 baseline/1590 after atomic claims across NVDA AMD INTC and all8 sections; canonical paths in config/grounding_v6_pairs.json; stability in audits/stability/NVDA_adjudicated; no human validation.')]:
        p=ROOT/relative;old=p.read_text(encoding='utf-8')
        if not old.startswith('> Current release note:'):p.write_text('> Current release note: '+note+'\n\n'+old,encoding='utf-8')
    print('Recorded actually completed visual review, replay logs, lineage and verifier-version evidence.')
if __name__=='__main__':main()
