"""Bind the final wording revision to actual current and identical prior page views."""
import json,hashlib
from pathlib import Path
from PIL import Image
from pypdf import PdfReader
from audit_io import write_json,sha256,utc_now
ROOT=Path(__file__).resolve().parents[1]
SOURCE='57a922c846ee5aaeba1a4683320e7580a0c86c4da0ec153e5be944817b369131'
PDF='9494dd046102da59b1b17679f1f0c1ec41d6c9054f281ac1070969fd0be8fd72'
VIEWED=[1,2,10,11,12,13]
def main():
    build_path=ROOT/'output/pdf/research_report_build.json'
    build=json.loads(build_path.read_text(encoding='utf-8'))
    prior=json.loads((ROOT/'evidence/pdf_visual_versions/semantic_v9_1.json').read_text(encoding='utf-8'))
    assert build['pages']==16 and build['source_sha256']==SOURCE and build['pdf_sha256']==PDF
    assert sha256((ROOT/'docs/research_report.md').read_bytes())==SOURCE
    assert sha256((ROOT/'output/pdf/research_report.pdf').read_bytes())==PDF
    image_hashes={};headers=[];reused=[]
    for i,p in enumerate(build['page_images'],1):
        path=Path(p);assert path.name==f'page-{i:02d}.png'
        image_hashes[path.name]=sha256(path.read_bytes())
        if i not in VIEWED:
            assert image_hashes[path.name]==prior['page_image_sha256'][path.name];reused.append(i)
        crop=Image.open(path).convert('RGB').crop((50,20,600,60))
        assert crop.getextrema()!=((255,255),(255,255),(255,255))
        headers.append(hashlib.sha256(crop.tobytes()).hexdigest())
    assert len(set(headers))==1
    for page in PdfReader(ROOT/'output/pdf/research_report.pdf').pages:
        text=page.extract_text();assert 'STAT GR5398 | Fall 2026 | FinRobot audit' in text
        assert 'Zeyu Gu | zg2556' in text and '\ufffd' not in text
    result={'checked_utc':utc_now(),'pdf_sha256':PDF,'source_sha256':SOURCE,'pages_inspected':16,
        'reviewer':'Codex root task agent; not human claim validation','page_image_sha256':image_hashes,
        'inspection_reference':str(Path(build['page_images'][0]).parent),'newly_viewed_pages':VIEWED,
        'reused_exact_image_reviews':reused,'prior_qa':'evidence/pdf_visual_versions/semantic_v9_1.json',
        'header_pixel_hash_identical_all_pages':headers[0],
        'findings':'All6 changed current PNGs displayed and inspected;10 other current PNGs match individually viewed prior pages exactly. Tables, figures, wrapping, dates, historical/current version labels and page furniture readable; no observed clipping, overlap, missing image or replacement glyph.',
        'limitation':'Visual integrity is not semantic truth. Unsupported/unverifiable assertions remain disclosed; no human claim validation.'}
    write_json(ROOT/'evidence/pdf_release_visual_qa.json',result)
    write_json(ROOT/'evidence/pdf_visual_versions/semantic_v9_1_final.json',result)
    build['visual_qa_status']='All16 pages covered:6 changed current views and10 exact prior inspected images. No observed layout defect; no human financial validation.'
    write_json(build_path,build)
    print('Final16-page PDF matches source;6 changed views plus10 exact prior views recorded.')
if __name__=='__main__':main()
