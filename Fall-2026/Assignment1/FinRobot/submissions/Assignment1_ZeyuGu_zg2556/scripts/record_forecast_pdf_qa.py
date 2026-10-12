"""Bind the 17-page scenario manuscript to the actual full-page visual reviews."""
import json
import hashlib
from pathlib import Path
from PIL import Image
from pypdf import PdfReader
from audit_io import sha256, utc_now, write_json

ROOT = Path(__file__).resolve().parents[1]
SOURCE = '9fe1e92ebb0cd956f28cce4f3740ca6d53a1879736e82b72e32d33630e840126'
PDF = '1d77a21b0494a6017959895f12bfdf94795782d4d1b4dd2a7df69fd8b6de27fa'


def main():
    build_path = ROOT/'output/pdf/research_report_build.json'
    build = json.loads(build_path.read_text(encoding='utf-8'))
    assert build['pages'] == 17 and build['source_sha256'] == SOURCE and build['pdf_sha256'] == PDF
    assert sha256((ROOT/'docs/research_report.md').read_bytes()) == SOURCE
    assert sha256((ROOT/'output/pdf/research_report.pdf').read_bytes()) == PDF
    reviewed = {}
    sources = {}
    for name in ['pages_01_08.json', 'pages_09_17.json']:
        path = ROOT/'evidence/pdf_visual_versions/forecast_parameters'/name
        proof = json.loads(path.read_text(encoding='utf-8'))
        assert proof['passed'] and proof['source_sha256'] == SOURCE and proof['pdf_sha256'] == PDF
        sources[path.relative_to(ROOT).as_posix()] = sha256(path.read_bytes())
        for item in proof['pages']:
            assert item.get('visual_pass', item.get('visual_result') == 'PASS')
            assert item.get('actually_viewed_with') or item.get('actually_viewed_full_page')
            assert item['page'] not in reviewed
            reviewed[item['page']] = item.get('sha256', item.get('png_sha256'))
    assert set(reviewed) == set(range(1, 18))
    hashes = {}
    headers = []
    for number, value in enumerate(build['page_images'], 1):
        path = Path(value)
        actual = sha256(path.read_bytes())
        assert actual == reviewed[number]
        hashes[path.name] = actual
        crop = Image.open(path).convert('RGB').crop((50, 20, 600, 60))
        assert crop.getextrema() != ((255, 255), (255, 255), (255, 255))
        headers.append(hashlib.sha256(crop.tobytes()).hexdigest())
    assert len(set(headers)) == 1
    for page in PdfReader(ROOT/'output/pdf/research_report.pdf').pages:
        text = page.extract_text()
        assert 'STAT GR5398 | Fall 2026 | FinRobot audit' in text and 'Zeyu Gu | zg2556' in text
        assert '\ufffd' not in text
    result = {'checked_utc': utc_now(), 'pdf_sha256': PDF, 'source_sha256': SOURCE,
        'pages_inspected': 17, 'reviewer': 'Two Codex task agents viewed all pages; root additionally viewed pages 2 and 6. No human financial validation.',
        'page_image_sha256': hashes, 'inspection_reference': str(Path(build['page_images'][0]).parent),
        'newly_viewed_pages': list(range(1, 18)), 'reused_exact_image_reviews': [],
        'review_evidence_sha256': sources, 'header_pixel_hash_identical_all_pages': headers[0],
        'findings': 'All17 complete page PNGs actually viewed. New forecast tables, existing audit tables, figures and page furniture readable; no observed cutoff, overlap, missing figure or encoding defect.',
        'limitation': 'Layout review does not certify financial truth or forecast accuracy. Unsupported claims and conditional scenario assumptions remain disclosed.'}
    write_json(ROOT/'evidence/pdf_release_visual_qa.json', result)
    write_json(ROOT/'evidence/pdf_visual_versions/forecast_parameters_final.json', result)
    build['visual_qa_status'] = 'All17 current page PNGs actually viewed; no observed layout defect. No human financial certification.'
    write_json(build_path, build)
    print('Current17-page PDF and source bound to actual full-page visual review.')


if __name__ == '__main__':
    main()
