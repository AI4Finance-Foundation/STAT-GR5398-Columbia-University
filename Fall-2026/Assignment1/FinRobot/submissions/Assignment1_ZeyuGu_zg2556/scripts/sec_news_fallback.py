"""Freeze genuine recent SEC 8-K event disclosures as a disclosed news substitute.

This is not FMP news, a complete media feed, or evidence of absence of news.
No generated summaries: the prompt receives literal excerpts of extracted filing
text, with exact character offsets and hashes into the saved extraction.
"""
import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
from urllib.parse import urljoin, urlsplit
from bs4 import BeautifulSoup
from fetch_sec_evidence import Snapshot
from audit_io import sha256, write_json, utc_now

CIKS = {"NVDA":1045810, "AMD":2488, "INTC":50863, "AVGO":1730168, "QCOM":804328,
        "AAPL":320193, "GOOGL":1652044}


def extract(raw):
    soup = BeautifulSoup(raw, "html.parser")
    for node in soup.find_all(["script", "style", "ix:header", "head"]):
        node.decompose()
    text = soup.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return soup, text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--as-of", default="2026-10-11T21:10:00+00:00")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--tickers", nargs="+", default=list(CIKS))
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--max-filings", type=int, default=3)
    parser.add_argument("--excerpt-characters", type=int, default=10000)
    args = parser.parse_args()
    as_of = datetime.fromisoformat(args.as_of)
    if as_of.tzinfo is None:
        raise ValueError("as-of must include timezone")
    start = as_of - timedelta(days=args.days)
    snapshot = Snapshot(args.output, args.offline, "AcademicResearch/1.0 (ZeyuGu-zg2556)")
    summaries = []
    for ticker in args.tickers:
        cik = CIKS[ticker]
        url = f"https://data.sec.gov/submissions/CIK{cik:010d}.json"
        body = snapshot.get(url, f"{ticker}/submissions.json")
        if not body:
            raise ValueError("SEC submissions unavailable for " + ticker)
        data = json.loads(body)
        recent = data["filings"]["recent"]
        candidates = []
        for row in (dict(zip(recent, values)) for values in zip(*recent.values())):
            if row["form"] != "8-K":
                continue
            acceptance = row.get("acceptanceDateTime", "")
            if not acceptance:
                continue
            published = datetime.fromisoformat(acceptance.replace("Z", "+00:00"))
            if published.tzinfo is None:
                # Fail rather than inventing a publication timezone.
                continue
            if start <= published <= as_of:
                candidates.append((published, row))
        articles = []
        selected = sorted(candidates, reverse=True, key=lambda entry:entry[0])[:args.max_filings]
        for published, filing in selected:
            acc = filing["accessionNumber"]
            base = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/"
            source_url = base + filing["primaryDocument"]
            relative = f"{ticker}/{acc}/{filing['primaryDocument']}"
            raw = snapshot.get(source_url, relative)
            if not raw:
                continue
            soup, text = extract(raw)
            documents = [(source_url, relative, text, "8-K primary disclosure")]
            seen = set()
            # One press release exhibit per filing, if explicitly linked as 99.x.
            for link in soup.find_all("a", href=True):
                href = link["href"]
                label = link.get_text(" ", strip=True)
                if (re.search(r"99[._-]?1|exhibit\s*99", href+" "+label, re.I)
                        and href.lower().split("?")[0].endswith((".htm", ".html"))):
                    exhibit_url = urljoin(source_url, href)
                    if not exhibit_url.startswith(base) or exhibit_url in seen:
                        continue
                    seen.add(exhibit_url)
                    exhibit_name = Path(urlsplit(exhibit_url).path).name
                    exhibit_relative = f"{ticker}/{acc}/{exhibit_name}"
                    exhibit = snapshot.get(exhibit_url, exhibit_relative)
                    if exhibit:
                        _, exhibit_text = extract(exhibit)
                        documents.append((exhibit_url, exhibit_relative, exhibit_text, "8-K Exhibit 99.1 disclosure"))
                        break
            # Prefer an attached press release, otherwise use the actual filing.
            doc_url, doc_relative, full_text, source_type = documents[-1]
            extracted_path = args.output/(doc_relative+".extracted.txt")
            extracted_path.write_text(full_text, encoding="utf-8")
            # A primary 8-K starts with an SEC cover. Locate the first item body;
            # do not paraphrase or selectively alter the extracted source text.
            match = re.search(r"Item\s+[1-9]\.\d{2}", full_text, re.I) if len(documents)==1 else None
            offset = match.start() if match else 0
            excerpt = full_text[offset:offset+args.excerpt_characters]
            title = f"{data['name']} ({ticker}) SEC Form 8-K filed {filing['filingDate']}; Items {filing.get('items', 'not specified')}"
            articles.append({"symbol":ticker, "title":title, "publishedDate":published.isoformat(),
                "text":excerpt, "site":"SEC EDGAR issuer disclosure", "url":doc_url,
                "source_type":source_type, "sec_accession":acc, "filing_date":filing["filingDate"],
                "source_body_path":doc_relative, "extracted_text_path":str(extracted_path.relative_to(args.output)),
                "extracted_text_sha256":sha256(full_text), "excerpt_start":offset,
                "excerpt_end":offset+len(excerpt), "excerpt_sha256":sha256(excerpt),
                "source_limitation":"Issuer-filed event disclosure; not independent media reporting. Literal excerpt may be truncated; full document retained."})
        write_json(args.output/ticker/"articles.json", articles)
        summaries.append({"ticker":ticker, "eligible_8k_count":len(candidates), "selected_filings":len(selected),
            "article_count":len(articles), "source":"SEC event-disclosure substitute, not FMP news",
            "selection":"latest accepted 8-K filings, at most one primary/exhibit excerpt per filing"})
    write_json(args.output/"manifest.json", {"created_at_utc":utc_now(), "as_of":args.as_of,
        "window_start":start.isoformat(), "days":args.days, "max_filings":args.max_filings,
        "excerpt_max_characters":args.excerpt_characters, "companies":summaries,
        "reason":"FMP news endpoint returned HTTP 402 under configured account; no service purchase performed",
        "fmp_news_access":False, "not_a_complete_news_feed":True})
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
