"""Check delivered HTML in a real browser with network blocked, then save images.

DOM checks prove text identity and asset availability. Screenshots still require
visual inspection; the output does not claim a human review occurred.
"""
import argparse
import json
from pathlib import Path
import hashlib
from playwright.sync_api import sync_playwright
from audit_io import write_json, utc_now


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--browser-executable", default=r"C:\Program Files\Google\Chrome\Application\chrome.exe")
    args = parser.parse_args()
    manifest = json.loads((args.report_dir/"rendered_section_texts.json").read_text(encoding="utf-8"))
    html = (args.report_dir/f"Audited_Equity_Report_{manifest['ticker']}.html").resolve()
    if hashlib.sha256(html.read_bytes()).hexdigest() != manifest["html_sha256"]:
        raise ValueError("HTML changed after rendering")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(executable_path=args.browser_executable, headless=True)
        page = browser.new_page(viewport={"width":1360, "height":1000}, device_scale_factor=1)
        blocked, errors = [], []
        page.on("pageerror", lambda error: errors.append(str(error)))
        def route(request):
            if request.request.url.startswith(("http://", "https://")):
                blocked.append(request.request.url)
                request.abort()
            else:
                request.continue_()
        page.route("**/*", route)
        page.goto(html.resolve().as_uri(), wait_until="load")
        page.evaluate("document.fonts.ready")
        sections = page.locator("[data-agent-section]").evaluate_all("nodes => nodes.map(n=>({section:n.dataset.agentSection,text:n.textContent}))")
        if len(sections) != 8 or {entry["section"] for entry in sections} != set(manifest["sections"]):
            raise ValueError("Rendered HTML does not contain exactly eight required sections")
        for section in sections:
            saved = manifest["sections"][section["section"]]
            if section["text"] != saved["text"] or hashlib.sha256(section["text"].encode()).hexdigest() != saved["sha256"]:
                raise ValueError("Rendered narrative text changed: " + section["section"])
        images = page.locator("img").evaluate_all("nodes=>nodes.map(n=>({src:n.getAttribute('src'),complete:n.complete,width:n.naturalWidth,height:n.naturalHeight}))")
        failures = [image for image in images if not image["complete"] or image["width"] <= 0]
        geometry = page.evaluate("({pageWidth:document.documentElement.clientWidth,scrollWidth:document.documentElement.scrollWidth,height:document.documentElement.scrollHeight})")
        missing = page.locator(".missing").all_text_contents()
        # The renderer uses this CSS class for its required data-availability
        # legend as well as real missing-content notices. Do not reject the
        # explicit N/A definition itself, or treat missing data as zero.
        unexplained_missing=[s for s in missing if s.strip()!='N/A denotes missing or inapplicable data, never zero.']
        page.screenshot(path=str(args.output_dir/"cover.png"))
        for section in ("company_overview", "risks", "news_summary"):
            page.locator("#"+section).scroll_into_view_if_needed()
            page.screenshot(path=str(args.output_dir/(section+".png")))
        for index in range(page.locator('figure').count()):
            page.locator('figure').nth(index).screenshot(path=str(args.output_dir/f'chart_{index+1}.png'))
        page.screenshot(path=str(args.output_dir/"full_report.png"), full_page=True)
        result = {"checked_at_utc":utc_now(), "html":str(html), "html_sha256":hashlib.sha256(html.read_bytes()).hexdigest(), "sections_identical":True,
            "images":images, "broken_images":failures, "optional_missing_notices":missing,
            "unexplained_missing_notices":unexplained_missing,
            "blocked_remote_requests":blocked, "javascript_errors":errors, "geometry":geometry,
            "visual_inspection":"screenshots_saved_pending_model_or_human_visual_review",
            "passed":len(images) >= 3 and not unexplained_missing and not failures and not blocked and not errors and geometry["scrollWidth"] <= geometry["pageWidth"]}
        write_json(args.output_dir/"qa.json", result)
        browser.close()
        if not result["passed"]:
            raise SystemExit("HTML QA found failures; inspect qa.json")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
