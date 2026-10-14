"""Rebuild exact issuer-news excerpts from the archived public source bytes.

This is offline extraction, not an LLM summary or a replacement data API. PDF
page extraction is archived separately with its pypdf version. All source units
are corporate disclosures and have not been independently corroborated here.
"""
from pathlib import Path
from datetime import datetime, timedelta
import hashlib
import json

from bs4 import BeautifulSoup
import bs4

ROOT = Path(__file__).resolve().parent
AS_OF = datetime.fromisoformat('2026-10-11T21:10:00+00:00')

SPECS = {
    'NVDA': dict(body='source_01.html', selector='.module_body', start='', stop='Q2 Fiscal 2027 Summary',
                 title='NVIDIA Announces Financial Results for Second Quarter Fiscal 2027',
                 date='2026-08-26T00:00:00-04:00', date_text='August 26, 2026', date_selector='.module_date-text',
                 precision='date_only', site='NVIDIA Investor Relations'),
    'AMD': dict(body='source_01.html', selector='article', start='SANTA CLARA', stop='GAAP Quarterly',
                title='AMD Reports Second Quarter 2026 Financial Results',
                date='2026-08-04T16:15:00-04:00', date_text='August 04, 2026 4:15 pm EDT', date_selector='time.date',
                precision='minute', site='AMD Investor Relations'),
    'INTC': dict(body='source_02.html', selector='article', start='News Summary', stop='Q2 2026 Financial Results',
                 title='Intel Reports Second-Quarter 2026 Financial Results',
                 date='2026-07-23T16:01:00-04:00', date_text='Jul 23, 2026', date_selector='time.date',
                 precision='minute', site='Intel Investor Relations'),
    'AAPL': dict(body='source_01.html', selector='article', start='CUPERTINO, CALIFORNIA', stop='Apple will provide live streaming',
                 title='Apple reports third quarter results',
                 date='2026-07-30T00:00:00-04:00', date_text='July 30, 2026', date_selector='article',
                 precision='date_only', site='Apple Newsroom'),
    'GOOGL': dict(body='source_03.pdf', selector='PDF page 1', start='', stop='Q2 2026 Financial Highlights (unaudited)',
                  title='Alphabet Announces Second Quarter 2026 Results',
                  date='2026-07-22T00:00:00-04:00', date_text='July 22, 2026', date_selector='PDF page 1 opening dateline',
                  precision='date_only', site='Alphabet Investor Relations'),
}


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode('utf-8')).hexdigest()


def main():
    results=[]
    for ticker, spec in SPECS.items():
        folder=ROOT/ticker
        raw=(folder/spec['body']).read_bytes()
        fetch=json.loads((folder/('fetch_'+spec['body'][7:9]+'.json')).read_text(encoding='utf-8'))
        assert fetch['status_code']==200 and digest(raw)==fetch['body_sha256']
        if spec['body'].endswith('.html'):
            soup=BeautifulSoup(raw,'html.parser')
            node=soup.select_one(spec['selector'])
            assert node is not None
            full=node.get_text('\n',strip=True)
            date_node=soup.select_one(spec['date_selector'])
            assert date_node is not None
            date_evidence=date_node.get_text(' ',strip=True)
            assert spec['date_text'] in date_evidence
            if spec['precision']=='minute':
                assert date_node.get('datetime')==spec['date'][:19]
                assert 'EDT' in date_evidence
            extractor={'name':'BeautifulSoup.get_text', 'version':bs4.__version__, 'separator':'\\n', 'strip':True}
        else:
            pdf=json.loads((folder/'source_03_pages.json').read_text(encoding='utf-8'))
            full='\f'.join(pdf['pages'])
            date_evidence=pdf['pages'][0].splitlines()[1]
            assert spec['date_text'] in date_evidence
            extractor={'name':pdf['extractor'], 'version':pdf['version'], 'page_separator':'form_feed', 'excerpt_page':1}
        start=full.index(spec['start']) if spec['start'] else 0
        end=full.index(spec['stop'],start)
        excerpt=full[start:end]
        # Keep exact extracted text including original line breaks/trailing spaces.
        assert excerpt and excerpt==full[start:end]
        published=datetime.fromisoformat(spec['date'])
        assert AS_OF-timedelta(days=90) <= published <= AS_OF
        (folder/'extracted_text.txt').write_text(full,encoding='utf-8',newline='')
        (folder/'excerpt.txt').write_text(excerpt,encoding='utf-8',newline='')
        article={'title':spec['title'], 'text':excerpt, 'publishedDate':spec['date'],
            'site':spec['site'], 'url':fetch['final_url'], 'symbol':ticker,
            'source_type':'official_issuer_earnings_press_release',
            'source_body_path':f'evidence/issuer_news/{ticker}/{spec["body"]}',
            'extracted_text_path':f'evidence/issuer_news/{ticker}/extracted_text.txt',
            'source_body_sha256':digest(raw), 'extracted_text_sha256':digest(full),
            'excerpt_start':start, 'excerpt_end':end, 'excerpt_sha256':digest(excerpt),
            'offset_basis':'Unicode code points in extracted_text.txt; start inclusive and end exclusive',
            'extractor':extractor, 'extraction_selector':spec['selector'],
            'fetched_at_utc':fetch['fetched_at_utc'], 'published_date_precision':spec['precision'],
            'published_date_evidence':date_evidence, 'published_date_locator':spec['date_selector'],
            'publication_timezone_assumption':('Source exposes a date only; America/New_York midnight is used for window filtering, not claimed as actual release time.' if spec['precision']=='date_only' else 'EDT is explicitly printed by the issuer; UTC-04:00 applied.'),
            'window_as_of_utc':AS_OF.isoformat(), 'window_days_back':90,
            'selection_rule':'One official quarterly earnings press release within the common trailing-90-day window; opening narrative before financial tables.',
            'limitations':'Corporate issuer disclosure; not independent media, a comprehensive news feed, or a retrospective SEC audit fact inserted as news. Forward-looking management assertions remain attributed statements.'}
        if ticker=='GOOGL':
            article['issuer_landing_page']='https://abc.xyz/investor/news/news-details/2026/Alphabet-Announces-Second-Quarter-2026-Results-2026-Y3uQ6H4ZJa/default.aspx'
            article['link_provenance_path']='evidence/issuer_news/GOOGL/source_02.html'
        (folder/'articles.json').write_text(json.dumps([article],indent=2,ensure_ascii=False),encoding='utf-8')
        results.append({'ticker':ticker,'articles':1,'chars':len(excerpt),'publication':spec['date'],'excerpt_sha256':digest(excerpt)})
    (ROOT/'manifest.json').write_text(json.dumps({'as_of_utc':AS_OF.isoformat(),'window_days':90,
        'coverage_type':'selected corporate issuer disclosures; not full independent-media coverage',
        'sources':results},indent=2),encoding='utf-8')
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    main()
