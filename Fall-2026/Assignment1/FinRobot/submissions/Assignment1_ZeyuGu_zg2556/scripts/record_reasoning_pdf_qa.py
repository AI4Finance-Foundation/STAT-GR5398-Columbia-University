"""Record the specific inspected 15-page follow-up, preserving prior QA."""
import json, hashlib
from pathlib import Path
from PIL import Image
from pypdf import PdfReader
from audit_io import write_json, sha256, utc_now
ROOT = Path(__file__).resolve().parents[1]

def main():
    path = ROOT / 'evidence/pdf_release_visual_qa.json'
    prior_bytes = path.read_bytes()
    prior = json.loads(prior_bytes)
    build = json.loads((ROOT / 'output/pdf/research_report_build.json').read_text(encoding='utf-8'))
    assert build['pages'] == 15
    assert build['source_sha256'] == 'cb3613cf9ceb54e7d38d8fbf5ae06a943e86fd406072e4c0b433fc5cb6a5c7e0'
    assert build['pdf_sha256'] == '73a18129b202db0d5652da60185643678eea510111f16be23a09316ddf650db2'
    assert sha256((ROOT / 'output/pdf/research_report.pdf').read_bytes()) == build['pdf_sha256']
    assert sha256((ROOT / 'docs/research_report.md').read_bytes()) == build['source_sha256']
    reference = ROOT / 'tmp/pdfs/research-cb3613cf9ceb-73a18129'
    image_hashes, header_hashes = {}, []
    for i, item in enumerate(build['page_images'], 1):
        page = Path(item)
        assert page == reference / f'page-{i:02d}.png'
        image_hashes[page.name] = sha256(page.read_bytes())
        # Pages 1-10 are byte-identical to already inspected preceding release.
        if i <= 10:
            assert image_hashes[page.name] == prior['page_image_sha256'][page.name]
        # Pages 11-15 were individually viewed in this continuation, not inferred.
        crop = Image.open(page).convert('RGB').crop((50, 20, 600, 60))
        assert crop.getextrema() != ((255, 255), (255, 255), (255, 255))
        header_hashes.append(hashlib.sha256(crop.tobytes()).hexdigest())
    assert len(set(header_hashes)) == 1
    for page in PdfReader(ROOT / 'output/pdf/research_report.pdf').pages:
        text = page.extract_text()
        assert 'STAT GR5398 | Fall 2026 | FinRobot audit' in text
        assert 'Zeyu Gu | zg2556' in text and '\ufffd' not in text
    archive = ROOT / 'evidence/progress' / ('pdf_qa_before_' + sha256(prior_bytes)[:12] + '.json')
    if not archive.exists():
        archive.write_bytes(prior_bytes)
    write_json(path, {
        'checked_utc': utc_now(), 'pdf_sha256': build['pdf_sha256'],
        'source_sha256': build['source_sha256'], 'pages_inspected': 15,
        'reviewer': 'Codex task agent; not human claim validation',
        'page_image_sha256': image_hashes, 'inspection_reference': str(reference),
        'reused_exact_image_reviews': list(range(1, 11)),
        'newly_viewed_pages': list(range(11, 16)),
        'prior_qa_archive': archive.relative_to(ROOT).as_posix(),
        'header_pixel_hash_identical_all_pages': header_hashes[0],
        'findings': 'Pages 1-10 match previously visually inspected PNGs byte-for-byte. Current pages 11-15 individually inspected: tables and figures readable, no clipped body, overlap, missing image or replacement glyph observed. All 15 header crops are identical and nonblank; header/footer text verified in PDF.',
        'limitation': 'Rendering integrity is separate from financial correctness; residual narrative errors remain explicitly reported.'
    })
    build['visual_qa_status'] = '15 pages covered: 10 exact prior image matches and 5 current individual visual inspections. No observed layout defect; no human financial validation.'
    write_json(ROOT / 'output/pdf/research_report_build.json', build)
    print('Recorded current 15-page visual QA; original evidence preserved.')

if __name__ == '__main__':
    main()
