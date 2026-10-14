"""Record the specific 16-page semantic-study PDF actually viewed by the root agent."""
import json,hashlib
from pathlib import Path
from PIL import Image
from pypdf import PdfReader
from audit_io import write_json,sha256,utc_now
ROOT=Path(__file__).resolve().parents[1]
SOURCE='a59e3d741fe7377d0283a99769d89c0da908678e1f24a743cb364b52be68e463'
PDF='9974d2792ae64546c317a06403993ae295e4264b63f6b64554b87d1ae5ad4be4'
def main():
    build_path=ROOT/'output/pdf/research_report_build.json'
    build=json.loads(build_path.read_text(encoding='utf-8'))
    assert build['pages']==16 and build['source_sha256']==SOURCE and build['pdf_sha256']==PDF
    assert sha256((ROOT/'docs/research_report.md').read_bytes())==SOURCE
    assert sha256((ROOT/'output/pdf/research_report.pdf').read_bytes())==PDF
    image_hashes={};headers=[]
    for i,p in enumerate(build['page_images'],1):
        path=Path(p);assert path.name==f'page-{i:02d}.png'
        image_hashes[path.name]=sha256(path.read_bytes())
        crop=Image.open(path).convert('RGB').crop((50,20,600,60))
        assert crop.getextrema()!=((255,255),(255,255),(255,255))
        headers.append(hashlib.sha256(crop.tobytes()).hexdigest())
    assert len(set(headers))==1
    for page in PdfReader(ROOT/'output/pdf/research_report.pdf').pages:
        text=page.extract_text();assert 'STAT GR5398 | Fall 2026 | FinRobot audit' in text
        assert 'Zeyu Gu | zg2556' in text and '\ufffd' not in text
    result={'checked_utc':utc_now(),'pdf_sha256':PDF,'source_sha256':SOURCE,'pages_inspected':16,
        'reviewer':'Codex root task agent; not human claim validation',
        'page_image_sha256':image_hashes,'inspection_reference':str(Path(build['page_images'][0]).parent),
        'newly_viewed_pages':list(range(1,17)),'reused_exact_image_reviews':[],
        'header_pixel_hash_identical_all_pages':headers[0],
        'findings':'All16 current page PNGs individually displayed and inspected in four batches. Body/table wrapping, repeated table headers, figures, page numbering and evidence/cost results readable; no clipping, overlap, missing image or replacement glyph observed. Standard-font page-header pixels identical and nonblank across all pages.',
        'limitation':'Visual integrity is not financial truth. Unverifiable and unsupported narratives remain explicitly quantified; no human claim review.'}
    old=ROOT/'evidence/pdf_release_visual_qa.json';raw=old.read_bytes()
    archive=ROOT/'evidence/progress'/('pdf_qa_before_'+sha256(raw)[:12]+'.json')
    if not archive.exists():archive.write_bytes(raw)
    write_json(old,result)
    write_json(ROOT/'evidence/pdf_visual_versions/semantic_v9_1.json',result)
    build['visual_qa_status']='All16 current page PNGs actually viewed by root task agent; no observed layout defect. No human financial validation.'
    write_json(build_path,build)
    print('Current16-page source/PDF bound to actual visual inspection; previous PDF and QA preserved.')
if __name__=='__main__':main()
