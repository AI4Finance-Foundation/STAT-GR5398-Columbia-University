"""Record this specific, actually inspected PDF; not a generic visual approval."""
import json,hashlib
from pathlib import Path
from PIL import Image
from pypdf import PdfReader
from audit_io import write_json,sha256,utc_now
ROOT=Path(__file__).resolve().parents[1]

def main():
    build=json.loads((ROOT/'output/pdf/research_report_build.json').read_text())
    inspected=ROOT/'tmp/pdfs/research-df91ff8c1001-bd748bbe'
    assert build['pages']==14 and build['source_sha256'].startswith('df91ff8c1001')
    pages=[Path(p) for p in build['page_images']]
    header_hashes=[];image_hashes={}
    for page in pages:
        # Every reference page was visually inspected through view_image.
        assert Image.open(page).tobytes()==Image.open(inspected/page.name).tobytes()
        crop=Image.open(page).convert('RGB').crop((50,20,600,60))
        assert crop.getextrema()!=((255,255),(255,255),(255,255))
        header_hashes.append(hashlib.sha256(crop.tobytes()).hexdigest())
        image_hashes[page.name]=sha256(page.read_bytes())
    assert len(set(header_hashes))==1
    reader=PdfReader(ROOT/'output/pdf/research_report.pdf')
    for n,page in enumerate(reader.pages,1):
        text=page.extract_text()
        assert 'STAT GR5398 | Fall 2026 | FinRobot audit' in text and 'Zeyu Gu | zg2556' in text
        assert '\ufffd' not in text
    write_json(ROOT/'evidence/pdf_release_visual_qa.json',{'checked_utc':utc_now(),
        'pdf_sha256':build['pdf_sha256'],'source_sha256':build['source_sha256'],'pages_inspected':14,
        'reviewer':'Codex task agent; not human claim validation','page_image_sha256':image_hashes,
        'inspection_reference':str(inspected),'pixel_identity_with_inspected_pages':True,
        'header_pixel_hash_identical_all_pages':header_hashes[0],
        'findings':'All14 pages visually inspected; readable tables, continuation headers and three figures. No clipped body, overlapping content, missing figure or replacement glyph observed. Apparent missing repeated page furniture in tool previews was not supported by PNG pixel comparison: all14 header regions are identical and nonblank; PDF text also contains header/footer on every page.',
        'limitation':'Rendering integrity is separate from financial correctness; residual narrative errors remain explicitly reported.'})
    build['visual_qa_status']='All14 current pages visually inspected; page pixels match inspected reference; no observed layout defect. No human financial validation.'
    write_json(ROOT/'output/pdf/research_report_build.json',build)
    print('Recorded current14-page visual QA and pixel-level repeated-header check.')
if __name__=='__main__':main()
