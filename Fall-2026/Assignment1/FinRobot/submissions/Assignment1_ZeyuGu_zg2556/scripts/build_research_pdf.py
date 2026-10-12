"""Build the research PDF from editable Markdown and genuine local result figures.

Offline: no model/API/data calls. It never replaces missing measurements with examples.
Requires reportlab, pypdf, Pillow and Poppler; bundled runtime supplies these packages.
"""
from __future__ import annotations
import argparse
import hashlib
import html
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image as PILImage
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, LongTable,
    TableStyle, Image, PageBreak, KeepTogether, Preformatted)

ROOT=Path(__file__).resolve().parents[1]
RUNTIME=Path.home()/'.cache/codex-runtimes/codex-primary-runtime'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def register_fonts():
    fonts=Path('C:/Windows/Fonts')
    names={'Research':'arial.ttf','Research-Bold':'arialbd.ttf','Research-Italic':'ariali.ttf','Research-BoldItalic':'arialbi.ttf'}
    for name,filename in names.items():
        path=fonts/filename
        if not path.exists():raise FileNotFoundError('Arial font is required; configure register_fonts for this platform: '+str(path))
        pdfmetrics.registerFont(TTFont(name,str(path)))
    pdfmetrics.registerFontFamily('Research',normal='Research',bold='Research-Bold',italic='Research-Italic',boldItalic='Research-BoldItalic')


def clean(text):
    return text.translate(str.maketrans({'\u2010':'-','\u2011':'-','\u2012':'-','\u2013':'-','\u2014':'-','\u2212':'-','\u00a0':' '}))


def inline(text):
    text=html.escape(clean(text),quote=False)
    text=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',lambda m:'<link href="'+html.escape(html.unescape(m[2]),quote=True)+'" color="#18496b">'+m[1]+'</link>',text)
    text=re.sub(r'`([^`]+)`',r'<font name="Research">\1</font>',text)
    text=re.sub(r'\*\*([^*]+)\*\*',r'<b>\1</b>',text)
    text=re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)',r'<i>\1</i>',text)
    return text


def make_styles():
    base=getSampleStyleSheet()
    for name in ('Normal','BodyText'):
        base[name].fontName='Research';base[name].fontSize=9.6;base[name].leading=14.0
        base[name].textColor=colors.HexColor('#162638');base[name].spaceAfter=7
        base[name].splitLongWords=True
    for name,size,leading in (('Title',25,29),('Heading1',16,20),('Heading2',12.5,16),('Heading3',10.8,14)):
        base[name].fontName='Research-Bold';base[name].fontSize=size;base[name].leading=leading
        base[name].textColor=colors.HexColor('#18496b');base[name].spaceBefore=12;base[name].spaceAfter=7
        base[name].keepWithNext=True
    base.add(ParagraphStyle('Cell',parent=base['BodyText'],fontSize=7.7,leading=10.3,spaceAfter=0))
    base.add(ParagraphStyle('Caption',parent=base['BodyText'],fontSize=8.2,leading=11,textColor=colors.HexColor('#526574'),spaceAfter=12))
    base.add(ParagraphStyle('CodeBlock',parent=base['BodyText'],fontName='Research',fontSize=7.8,leading=10,backColor=colors.HexColor('#f0f4f6'),leftIndent=8,rightIndent=8,spaceBefore=4,spaceAfter=8))
    return base


def local_image(path, width, styles, caption=''):
    with PILImage.open(path) as source:w,h=source.size
    scaled=min(width/w,370/h)
    image=Image(str(path),width=w*scaled,height=h*scaled)
    image.hAlign='CENTER'
    parts=[image]
    if caption:parts.append(Paragraph(inline(caption),styles['Caption']))
    return KeepTogether(parts)


def table(lines,width,styles):
    rows=[[v.strip() for v in row.strip().strip('|').split('|')] for row in lines]
    rows=[row for row in rows if not all(re.fullmatch(r':?-{3,}:?',cell.replace(' ','')) for cell in row)]
    columns=max(map(len,rows));rows=[row+['']*(columns-len(row)) for row in rows]
    # Set widths from content, with a lower bound for each column and a cap on
    # long source links. Every cell wraps; no table is scaled into unreadability.
    weights=[max(7,min(30,max(len(re.sub(r'\[([^]]+)\]\([^)]+\)',r'\1',row[c])) for row in rows))) for c in range(columns)]
    minimums=[max(38,min(72,max(pdfmetrics.stringWidth(word,'Research-Bold',7.7) for word in rows[0][c].split())+12)) if rows[0][c].split() else 38 for c in range(columns)]
    remaining=max(0,width-sum(minimums))
    widths=[minimums[c]+remaining*weights[c]/sum(weights) for c in range(columns)]
    if sum(widths)>width:
        widths=[w*width/sum(widths) for w in widths]
    cells=[[Paragraph(('<b>'+inline(value)+'</b>') if i==0 else inline(value),styles['Cell']) for value in row] for i,row in enumerate(rows)]
    result=LongTable(cells,colWidths=widths,repeatRows=1,hAlign='LEFT',splitByRow=1)
    result.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e7eff5')),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f7f9fb')]),
        ('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),0.7,colors.HexColor('#879dab')),
        ('LINEBELOW',(0,1),(-1,-1),0.3,colors.HexColor('#dbe3e9')),
        ('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),
        ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
    return result


def parse_markdown(text,source,width,styles):
    lines=text.splitlines();story=[];images=[];i=0;first_title=True
    while i<len(lines):
        line=lines[i].strip()
        if not line:i+=1;continue
        if line.startswith('```'):
            block=[];i+=1
            while i<len(lines) and not lines[i].strip().startswith('```'):block.append(lines[i]);i+=1
            # Paragraph wrapping keeps exact displayed command characters without
            # overflowing a fixed-width preformatted box.
            story.append(Paragraph('<br/>'.join(html.escape(clean(x)).replace(' ','&#160;') for x in block),styles['CodeBlock']))
            i+=1;continue
        if line.startswith('|') and i+1<len(lines) and re.match(r'^\s*\|?\s*:?-{3,}',lines[i+1]):
            block=[]
            while i<len(lines) and lines[i].strip().startswith('|'):block.append(lines[i]);i+=1
            story.extend([table(block,width,styles),Spacer(1,9)]);continue
        image=re.fullmatch(r'!\[([^]]*)\]\(([^)]+)\)',line)
        if image:
            location=Path(image[2]);location=location if location.is_absolute() else (source.parent/location).resolve()
            if not location.exists():raise FileNotFoundError('Missing actual report figure: '+str(location))
            story.append(local_image(location,width,styles,image[1]));images.append(location);i+=1;continue
        heading=re.match(r'^(#{1,4})\s+(.*)',line)
        if heading:
            level=len(heading[1]);style='Title' if first_title and level==1 else 'Heading'+str(min(level,3))
            story.append(Paragraph(inline(heading[2]),styles[style]));first_title=False;i+=1;continue
        if line in {'---','***'}:story.append(Spacer(1,10));i+=1;continue
        item=re.match(r'^(?:[-*]|\d+\.)\s+(.*)',line)
        if item:
            story.append(Paragraph(inline(item[1]),styles['BodyText'],bulletText='-'));i+=1;continue
        block=[inline(line)];hard_break=lines[i].endswith('  ');i+=1
        while i<len(lines) and lines[i].strip() and not re.match(r'^(#{1,4}\s|[-*]\s|\d+\.\s|\||```|!\[)',lines[i].strip()):
            block.append(('<br/>' if hard_break else ' ')+inline(lines[i].strip()))
            hard_break=lines[i].endswith('  ');i+=1
        story.append(Paragraph(''.join(block),styles['BodyText']))
    return story,images


def build(source,output,poppler,extra_figures):
    if not source.is_file():raise FileNotFoundError('Editable report manuscript is not ready: '+str(source))
    register_fonts();styles=make_styles();text=source.read_text(encoding='utf-8-sig')
    if not text.strip():raise ValueError('Refusing to build an empty report')
    output.parent.mkdir(parents=True,exist_ok=True)
    doc=SimpleDocTemplate(str(output),pagesize=A4,rightMargin=45,leftMargin=45,topMargin=52,bottomMargin=48,
        title='FinRobot: execution, evidence and verification audit',author='Zeyu Gu (zg2556)',
        subject='STAT GR5398 Fall 2026 Assignment 1 - actual frozen-input experiments')
    story,images=parse_markdown(text,source,doc.width,styles)
    available=[Path(p).resolve() for p in extra_figures if Path(p).is_file() and Path(p).resolve() not in images]
    if available:
        story.extend([PageBreak(),Paragraph('Result figures',styles['Heading1']),
            Paragraph('The following figures are generated from saved experiment results. These are scenario analyses and measured audits; no probability distribution or unperformed result is implied.',styles['BodyText'])])
        for n,path in enumerate(available,1):
            caption=f'Figure {n}. '+path.parent.name.replace('_',' ')+' - '+path.stem.replace('_',' ')
            story.append(local_image(path,doc.width,styles,caption));images.append(path)
    def footer(canvas,document):
        # Use a standard PDF font for page furniture, without font-subset state.
        canvas.saveState();canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#526574'))
        canvas.drawString(45,A4[1]-30,'STAT GR5398 | Fall 2026 | FinRobot audit')
        canvas.line(45,37,A4[0]-45,37)
        canvas.drawString(45,25,'Zeyu Gu | zg2556');canvas.drawRightString(A4[0]-45,25,str(document.page))
        canvas.restoreState()
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    reader=PdfReader(output);extracted='\n'.join(page.extract_text() or '' for page in reader.pages)
    if not extracted.strip() or '\ufffd' in extracted:raise ValueError('Extracted PDF text is empty or contains replacement characters')
    source_hash=digest(source)
    qa=ROOT/'tmp/pdfs'/('research-'+source_hash[:12]+'-'+digest(output)[:8]);qa.mkdir(parents=True,exist_ok=True)
    (qa/'extracted_text.txt').write_text(extracted,encoding='utf-8')
    result=subprocess.run([str(poppler),'-r','110','-png',str(output),str(qa/'page')],capture_output=True,text=True)
    if result.returncode:raise RuntimeError('Poppler failed: '+result.stderr)
    page_images=sorted(qa.glob('page-*.png'))
    if len(page_images)!=len(reader.pages):raise ValueError('Rendered page count does not match PDF')
    manifest={'built_at_utc':datetime.now(timezone.utc).isoformat(),'source':str(source),'source_sha256':source_hash,
        'pdf':str(output),'pdf_sha256':digest(output),'builder_sha256':digest(__file__),'pages':len(reader.pages),
        'figures':[{'path':str(p),'sha256':digest(p)} for p in images],
        'page_images':[str(p) for p in page_images],
        'visual_qa_status':'PENDING: every current page image must be visually inspected before release'}
    (output.parent/'research_report_build.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({'pages':manifest['pages'],'pdf':str(output),'qa_dir':str(qa)},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'docs/research_report.md')
    parser.add_argument('--output',type=Path,default=ROOT/'output/pdf/research_report.pdf')
    parser.add_argument('--poppler',type=Path,default=RUNTIME/'dependencies/native/poppler/Library/bin/pdftoppm.exe')
    parser.add_argument('--figure',type=Path,action='append',default=[])
    args=parser.parse_args();build(args.source.resolve(),args.output.resolve(),args.poppler,args.figure)
