"""Read-only Chromium TextNode/Range screenshots of V9.1 changed explanations.

No HTML/DOM text, style, run evidence or upstream code is modified. Viewport and
scroll position change normally. Screenshots require a separate visual review.
"""
import argparse
import math
from pathlib import Path
from playwright.sync_api import sync_playwright
from audit_io import sha256, write_json, utc_now

ROOT = Path(__file__).resolve().parents[1]
RANGES = r"""() => {
  const found=[];
  for (const section of document.querySelectorAll('[data-agent-section]')) {
    const walker=document.createTreeWalker(section,NodeFilter.SHOW_TEXT);
    let node;
    while(node=walker.nextNode()) {
      const patterns=[['cagr',/The supplied Revenue CAGR cell[^\n]+/g],
        ['ev_ebitda_range',/Across the supplied EV\/EBITDA rows[^\n]+/g]];
      for(const [kind,pattern] of patterns) {
        for(const match of node.textContent.matchAll(pattern)) {
          const range=document.createRange();range.setStart(node,match.index);
          range.setEnd(node,match.index+match[0].length);
          const bounds=range.getBoundingClientRect();
          const rects=Array.from(range.getClientRects()).map(r=>({x:r.x+scrollX,y:r.y+scrollY,width:r.width,height:r.height,right:r.right+scrollX,bottom:r.bottom+scrollY}));
          const parent=section.getBoundingClientRect();
          found.push({section:section.dataset.agentSection,kind,text:match[0],start:match.index,
            end:match.index+match[0].length,node_length:node.length,rects,
            bounds:{x:bounds.x+scrollX,y:bounds.y+scrollY,width:bounds.width,height:bounds.height},
            parent_bounds:{x:parent.x+scrollX,y:parent.y+scrollY,width:parent.width,height:parent.height},
            all_line_boxes_inside_viewport:rects.every(r=>r.x>=-0.5&&r.right<=document.documentElement.clientWidth+0.5),
            all_line_boxes_inside_section:rects.every(r=>r.x>=parent.x+scrollX-0.5&&r.right<=parent.right+scrollX+0.5&&r.y>=parent.y+scrollY-0.5&&r.bottom<=parent.bottom+scrollY+0.5)});
        }
      }
    }
  }
  return found;
}"""


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'evidence/browser_semantic_v9_1/range_checks')
    parser.add_argument('--browser-executable',default=r'C:\Program Files\Google\Chrome\Application\chrome.exe')
    args=parser.parse_args()
    args.output_dir=args.output_dir.resolve()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError('Use a new output directory; existing visual evidence is retained')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    records=[]
    with sync_playwright() as runtime:
        browser=runtime.chromium.launch(executable_path=args.browser_executable,headless=True)
        for ticker in ['NVDA','AMD','INTC']:
            run=ROOT/'runs'/f'{ticker}_final_reasoning_v9_1'
            html=run/'output'/ticker/'report'/f'Audited_Equity_Report_{ticker}.html'
            before_hash=sha256(html.read_bytes())
            for viewport,width,height in [('desktop',1360,1000),('mobile',375,900)]:
                page=browser.new_page(viewport={'width':width,'height':height},device_scale_factor=1)
                blocked=[];errors=[]
                page.on('pageerror',lambda error:errors.append(str(error)))
                def route(request):
                    if request.request.url.startswith(('http://','https://')):
                        blocked.append(request.request.url);request.abort()
                    else:request.continue_()
                page.route('**/*',route)
                page.goto(html.resolve().as_uri(),wait_until='load')
                page.evaluate('document.fonts.ready')
                original_sections=page.locator('[data-agent-section]').evaluate_all('nodes=>nodes.map(n=>({section:n.dataset.agentSection,text:n.textContent}))')
                geometry=page.evaluate('({pageWidth:document.documentElement.clientWidth,scrollWidth:document.documentElement.scrollWidth,height:document.documentElement.scrollHeight,replacementCharacter:document.body.innerText.includes("\\ufffd")})')
                spans=page.evaluate(RANGES)
                expected={'NVDA':4,'AMD':7,'INTC':2}[ticker]
                assert len(spans)==expected, (ticker,len(spans),expected)
                directory=args.output_dir/ticker/viewport;directory.mkdir(parents=True)
                cover=directory/'cover.png';page.screenshot(path=str(cover))
                for index,span in enumerate(spans):
                    assert span['all_line_boxes_inside_viewport'] and span['all_line_boxes_inside_section']
                    assert span['rects'] and all(r['width']>0 and r['height']>0 for r in span['rects'])
                    box=span['bounds'];left=max(0,math.floor(box['x'])-8);top=max(0,math.floor(box['y'])-8)
                    # Bring the real text into the rasterized viewport before clipping.
                    # Full-page captures of offscreen 40k-pixel pages can omit glyphs.
                    page.evaluate('(y)=>scrollTo(0,y)',max(0,top-80))
                    page.evaluate('async()=>{await new Promise(requestAnimationFrame);await new Promise(requestAnimationFrame)}')
                    scroll=page.evaluate('({x:scrollX,y:scrollY})')
                    clip={'x':left-scroll['x'],'y':top-scroll['y'],
                          'width':min(width-left,math.ceil(box['width'])+16),'height':math.ceil(box['height'])+16}
                    filename=f"{index+1:02d}_{span['section']}_{span['kind']}.png"
                    page.screenshot(path=str(directory/filename),clip=clip)
                    span.update(screenshot=str((directory/filename).relative_to(ROOT)),
                                screenshot_sha256=sha256((directory/filename).read_bytes()),text_sha256=sha256(span['text']),
                                clip=clip,clip_coordinates='viewport after normal scroll',scroll=scroll)
                assert page.locator('[data-agent-section]').evaluate_all('nodes=>nodes.map(n=>({section:n.dataset.agentSection,text:n.textContent}))')==original_sections
                assert len(original_sections)==8
                for section in original_sections:
                    assert section['text']==(run/'output'/ticker/'analysis'/(section['section']+'.txt')).read_text(encoding='utf-8')
                assert not blocked and not errors and not geometry['replacementCharacter']
                assert geometry['scrollWidth']<=geometry['pageWidth']
                records.append({'ticker':ticker,'viewport':viewport,'width':width,'height':height,
                    'html_sha256':before_hash,'geometry':geometry,'remote_requests':blocked,'javascript_errors':errors,
                    'eight_sections_match_saved_text':True,'dom_text_unchanged':True,'spans':spans,
                    'cover':str(cover.relative_to(ROOT)),'cover_sha256':sha256(cover.read_bytes())})
                page.close()
            assert sha256(html.read_bytes())==before_hash
        browser.close()
    write_json(args.output_dir/'range_qa.json',{'checked_at_utc':utc_now(),'passed':True,
        'method':'Read-only TextNode TreeWalker and Range.getClientRects/getBoundingClientRect; clipped real Chromium screenshots; no DOM text/style edits.',
        'script_sha256':sha256(Path(__file__).read_bytes()),'records':records,
        'visual_inspection':'Separate model review of resulting screenshots required; this record alone is geometry evidence.'})
    print({'passed':True,'viewports':len(records),'range_screenshots':sum(len(r['spans']) for r in records)})


if __name__=='__main__':main()
