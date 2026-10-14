"""Validate the saved public readback and its current publication bindings offline.

This does not recheck live availability or certify the scientific claims.
"""
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]


def validate():
    def read(relative):
        return json.loads((ROOT / relative).read_text(encoding='utf-8-sig'))

    def sha(relative):
        return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()

    pub = read('config/publication_status.json')
    meta = read('blog/medium_metadata.json')
    proof = read('evidence/medium_publication/verification.json')
    url = pub['medium_url']
    parsed = urlsplit(url)
    assert parsed.scheme == 'https' and parsed.netloc == 'medium.com'
    assert not parsed.query and not parsed.fragment
    assert parsed.path.startswith('/@zeyu.gu0302/')
    assert url == meta['medium_url'] == proof['url']
    assert pub['medium_status'] == proof['status'] == meta['publication_status'] == 'published_by_user_verified_public'
    assert pub['medium_published_by'] == 'user'
    assert not pub['assistant_remote_write_authorized'] and not proof['external_write_performed_by_assistant']
    assert not meta['publication_authorized'] and not meta['account_write_access_verified']
    assert pub['course_pull_request_url'] is None and meta['course_pr']['url'] is None
    assert proof['signed_out_public_read_verified'] and proof['full_body_accessible']
    for name, expected in proof['capture_sha256'].items():
        assert sha('evidence/medium_publication/' + name) == expected, name
    assert proof['body_blocks'] == len(proof['formatting_copy_blocks_match']) == 65
    assert all(row['match'] for row in proof['formatting_copy_blocks_match'])
    assert proof['list_rows'] == 10 and len(proof['headings']) == 9
    assert sha('blog/medium_article.md') == proof['article_source_sha256'] == meta['article_source_hash']
    assert sha('docs/course_pr_body.md') == meta['course_pr']['body_sha256']
    for relative in ['README.md', 'docs/course_pr_body.md', 'docs/submission_readiness.md']:
        assert url in (ROOT / relative).read_text(encoding='utf-8'), relative
    body = (ROOT / 'evidence/medium_publication/published_body.txt').read_text(encoding='utf-8')
    manuscript = (ROOT / 'blog/medium_article.md').read_text(encoding='utf-8')
    for text in ['570/1985', '1415', '461 verifiable numeric claims', '51 numeric claims remain unverifiable', 'not an autonomous portfolio recommendation']:
        assert text in body and text in manuscript, text
    assert meta['research_pdf'] == 'output/pdf/research_report_submission.pdf'
    assert meta['research_source'] is None
    return {'passed': True, 'url': url, 'checked_utc': proof['checked_utc'],
            'saved_public_body_blocks': proof['body_blocks'], 'course_pr_created': False,
            'scope': 'Offline integrity of captured signed-out public readback and publication bindings; not a new live check or semantic truth certification.'}


if __name__ == '__main__':
    print(json.dumps(validate(), indent=2))
