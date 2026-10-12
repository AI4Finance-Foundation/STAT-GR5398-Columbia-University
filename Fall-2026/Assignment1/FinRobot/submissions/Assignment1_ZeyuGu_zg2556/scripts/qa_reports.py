"""Offline HTML rendering QA. Does not certify financial or narrative correctness."""
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright
from audit_io import sha256, utc_now, write_json
from experiment import FIELDS, hashes, validate_hashes


def inspect(root, executable):
    outputs=[]
    qa=root/'evidence/render_qa'
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path=str(executable),headless=True)
        context=browser.new_context(viewport={'width':1360,'height':1000},device_scale_factor=1)
        context.route('**/*',lambda route: route.continue_() if route.request.url.startswith(('file:','data:')) else route.abort())
        for path in sorted((root/'runs').glob('*/manifest.json')):
            manifest=json.loads(path.read_text(encoding='utf-8'))
            if manifest['state']!='rendered':continue
            run=path.parent;ticker=manifest['ticker'];report=run/'output'/ticker/'report'
            analysis=run/'output'/ticker/'analysis'
            validate_hashes(analysis,manifest['analysis_hashes'])
            validate_hashes(report,manifest['report_hashes'])
            html=report/f'Audited_Equity_Report_{ticker}.html'
            out=qa/run.name;out.mkdir(parents=True,exist_ok=True)
            page=context.new_page();errors=[]
            page.on('pageerror',lambda error:errors.append(str(error)))
            page.goto(html.as_uri(),wait_until='load')
            page.screenshot(path=str(out/'desktop_top.png'))
            rows=[]
            for section in FIELDS:
                source=(analysis/(section+'.txt')).read_text(encoding='utf-8')
                locator=page.locator(f'[data-agent-section="{section}"]')
                observed=locator.text_content()
                rows.append({'section':section,'exact_text_match':observed==source,
                             'source_sha256':sha256(source),'rendered_sha256':sha256(observed)})
            images=page.locator('img').evaluate_all('(nodes)=>nodes.map(n=>({src:n.getAttribute("src"),complete:n.complete,width:n.naturalWidth,height:n.naturalHeight}))')
            for i,figure in enumerate(page.locator('figure').all(),1):
                figure.screenshot(path=str(out/f'figure_{i:02d}.png'))
            desktop_overflow=page.evaluate('document.documentElement.scrollWidth>window.innerWidth')
            page.set_viewport_size({'width':375,'height':900});page.evaluate('window.scrollTo(0,0)');page.screenshot(path=str(out/'mobile_top.png'))
            mobile_overflow=page.evaluate('document.documentElement.scrollWidth>window.innerWidth')
            check={'run_id':run.name,'checked_at_utc':utc_now(),'browser':'Microsoft Edge headless via Playwright',
               'html_sha256':sha256(html.read_bytes()),'sections':rows,'images':images,'page_errors':errors,
               'desktop_horizontal_overflow':desktop_overflow,'mobile_horizontal_overflow':mobile_overflow,
               'financial_correctness_scope':'Not certified here: numeric/factual claims are evaluated in separate audit tables.',
               'passed':all(x['exact_text_match'] for x in rows) and len(images)>=3 and
                  all(x['complete'] and x['width']>0 and x['height']>0 for x in images) and
                  not errors and not desktop_overflow and not mobile_overflow}
            write_json(out/'checks.json',check)
            page.close();outputs.append(check)
        browser.close()
    write_json(qa/'summary.json',{'at_utc':utc_now(),'reports':outputs,
        'all_passed':bool(outputs) and all(x['passed'] for x in outputs),
        'visual_review_status':'Screenshots generated; human-readable agent visual inspection is recorded separately.'})
    print(json.dumps([{'run_id':x['run_id'],'passed':x['passed'],'images':len(x['images'])} for x in outputs],indent=2))
    if not outputs or not all(x['passed'] for x in outputs):raise SystemExit(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submission',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--browser',type=Path,default=Path(r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'))
    args=parser.parse_args();inspect(args.submission.resolve(),args.browser)
