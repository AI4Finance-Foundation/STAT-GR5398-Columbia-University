# Frozen official issuer-news fallback

These inputs contain **one selected official quarterly earnings press release for each of NVDA, AMD, INTC, AAPL and GOOGL**. They are corporate issuer disclosures, not an independent-media sample or a complete news feed. They were acquired because the original FMP news endpoint was unavailable under the configured entitlement. Restricted SEC responses were not used or bypassed.

The common eligibility window is the trailing 90 days ending `2026-10-11T21:10:00+00:00`. Every selected release falls inside it. AMD and Intel explicitly publish minute-level EDT timestamps. NVIDIA, Apple and Alphabet expose publication dates without verified times; their JSON records use midnight America/New_York solely for window filtering and clearly state this assumption. Acquisition timestamps are separately retained in UTC.

Each `TICKER/articles.json` is a list accepted by the snapshot runner. Its `text` is an **exact substring of extracted source text**, with no generated paraphrase or post-hoc financial audit facts. Excerpts use the opening earnings narrative before financial tables. Management statements and forward-looking claims remain issuer assertions. They are not independently validated facts merely because they appear in a release.

For each article the folder preserves:

- Public source response bytes and SHA-256, final/requested URL, HTTP status, content type and fetch UTC.
- The complete extracted text and an exact excerpt with Unicode start-inclusive/end-exclusive offsets and SHA-256 hashes.
- Publication-date evidence and its DOM selector or PDF-page locator.
- Selection rule, source type, extractor version, timezone assumption and coverage limitations.

Alphabet's official release landing page links directly to the archived Q4 CDN PDF. The landing HTML is retained as link provenance. Page 1 was rendered with Poppler and visually checked against the extracted earnings narrative; `source_03_page1.png` is the inspection image. PDF text extraction used bundled `pypdf 6.10.0`; HTML extraction used `beautifulsoup4 4.15.0`. The full PDF contains ten pages; only page 1's opening narrative is included in the agent news excerpt.

Rebuild the article JSON files entirely offline with:

```powershell
python evidence/issuer_news/extract_snapshot.py
```

This command reuses archived raw HTML and the saved PDF-page extraction. It validates raw-source hashes, release-date evidence, window inclusion and exact excerpt offsets. The pipeline should use the fixed `articles.json` snapshots when comparing baseline, repeated generations and verifier treatment. A new online acquisition must use a new directory and a new data snapshot; it must not silently replace these frozen inputs.

Raw webpage bodies can include public publisher scripts and form fields. No authenticated browser state, API credentials, request headers or cookies were submitted or archived by this acquisition.
