"""Read-only text comparison; writes evidence only to --output-directory.

Usage (requires pypdf):
  python compare_pdf_body.py REFERENCE.pdf CANDIDATE.pdf --output-directory OUT

This compares extracted text, not PDF bytes, glyph placement or vector geometry.
The raw text differences are classified before the more permissive canonical
normalization is accepted. See the emitted methodology and difference records.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import re
import unicodedata

from pypdf import PdfReader


BODY_START = "1.Researchquestionandprincipalfindings"
CONTINUATIONS = (
    "ContinuedonnextpageEBITDAbridge,$millionNVDAFY2026AMDFY2025",
    "ContinuedonnextpageCompanyWarningrecallonU/CWarningprecisionUnflaggedU/CUnflaggedC",
)


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def extracted_normalization(text: str) -> str:
    # These match complete lines only; the body and PDF metadata are separate.
    text = re.sub(r"(?m)^FINROBOT / EVIDENCE AUDIT\s*\n?", "", text)
    text = re.sub(r"(?m)^Zeyu Gu \| zg2556\s*\n?", "", text)
    text = re.sub(r"(?m)^\d+\s*$", "", text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u2019", "'").replace("\u2212", "-")
    text = text.replace("\u2013", "-").replace("\xad", "")
    return re.sub(r"\s+", "", text)


def canonical_body(text: str) -> str:
    text = text[text.index(BODY_START):]
    for continuation in CONTINUATIONS:
        text = text.replace(continuation, "")
    # This reproduces the original 42,757-character comparison exactly. It is
    # permissive: minus signs also disappear, so raw-difference classification
    # is essential and canonical equality alone must not establish correctness.
    return text.replace("-", "")


def classify(tag: str, a: int, b: int, c: int, d: int,
             reference: str, candidate: str) -> str:
    removed, added = reference[a:b], candidate[c:d]
    if tag == "delete" and a == 0 and b == reference.index(BODY_START) and c == 0:
        return "removed_visible_frontmatter"
    if tag == "delete" and removed in CONTINUATIONS:
        return "reference_only_repeated_longtable_continuation_header"
    if tag == "insert" and added in CONTINUATIONS:
        return "candidate_only_repeated_longtable_continuation_header"
    if (tag == "delete" and removed == "-" and a > 0 and b < len(reference)
            and reference[a-1].isalpha() and reference[b].isalpha()):
        return "reference_only_in_word_line_hyphenation"
    if (tag == "insert" and added == "-" and c > 0 and d < len(candidate)
            and candidate[c-1].isalpha() and candidate[d].isalpha()):
        return "candidate_only_in_word_line_hyphenation"
    return "UNCLASSIFIED_REQUIRES_REVIEW"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_pdf", type=Path)
    parser.add_argument("candidate_pdf", type=Path)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    entries, texts = [], []
    for path in (args.reference_pdf, args.candidate_pdf):
        reader = PdfReader(path)
        page_texts = [page.extract_text() or "" for page in reader.pages]
        raw = "\n".join(page_texts)
        footer_matches, nonfooter_integers = [], []
        for page_number, page_text in enumerate(page_texts, 1):
            lines = page_text.splitlines()
            footer_matches.append(bool(lines) and lines[-1].strip() == str(page_number))
            # Integer-only graph ticks would also disappear under the original
            # normalization. Audit them independently rather than omit them.
            nonfooter_integers.extend(line.strip() for line in lines[:-1]
                                     if re.fullmatch(r"\d+", line.strip()))
        extracted = extracted_normalization(raw)
        entries.append({
            "path": str(path.resolve()), "file_sha256": sha256(path.read_bytes()),
            "page_count": len(reader.pages), "raw_extracted_characters": len(raw),
            "all_last_lines_match_page_number": all(footer_matches),
            "nonfooter_integer_only_lines": nonfooter_integers,
            "metadata": {str(k): str(v) for k, v in (reader.metadata or {}).items()},
        })
        texts.append(extracted)
    reference, candidate = texts
    differences = []
    matcher = difflib.SequenceMatcher(None, reference, candidate, autojunk=False)
    for tag, a, b, c, d in matcher.get_opcodes():
        if tag == "equal":
            continue
        differences.append({
            "difference_number": len(differences) + 1,
            "operation": tag,
            "classification": classify(tag, a, b, c, d, reference, candidate),
            "reference_offsets": [a, b], "candidate_offsets": [c, d],
            "removed_text": reference[a:b], "added_text": candidate[c:d],
            "reference_context": reference[max(0, a-65):b+65],
            "candidate_context": candidate[max(0, c-65):d+65],
        })
    bodies = [canonical_body(text) for text in texts]
    for entry, body in zip(entries, bodies):
        entry.update({"canonical_body_characters": len(body),
                      "canonical_body_sha256": sha256(body.encode("utf-8"))})
    classified = all(d["classification"] != "UNCLASSIFIED_REQUIRES_REVIEW"
                     for d in differences)
    integer_audit_pass = (all(e["all_last_lines_match_page_number"] for e in entries)
                          and entries[0]["nonfooter_integer_only_lines"]
                          == entries[1]["nonfooter_integer_only_lines"])
    report = {
        "reference": entries[0], "candidate": entries[1],
        "file_bytes_identical": entries[0]["file_sha256"] == entries[1]["file_sha256"],
        "extracted_normalized_difference_count": len(differences),
        "all_raw_differences_classified": classified,
        "integer_only_line_audit_pass": integer_audit_pass,
        "canonical_body_equal": bodies[0] == bodies[1],
        "comparison_pass": classified and integer_audit_pass and bodies[0] == bodies[1],
        "methodology": [
            "Extract with pypdf; concatenate pages in PDF order.",
            "Remove exact running-header lines and standalone integer lines used as page numbers; apply Unicode NFKC, curly-apostrophe, en-dash/minus normalization, soft-hyphen removal and whitespace removal.",
            "Because integer-only graph tick lines are also removed by that historical normalization, separately check that every page's final line equals its page number and compare the complete remaining integer-only line sequences; this gate must also pass.",
            "Save and classify every remaining character-level difference before body normalization. In-word hyphen deletions are accepted only when the complete changed span is one hyphen between letters; numeric or substantive changes remain unclassified.",
            "For the canonical body, start at section 1, remove the two explicitly enumerated repeated longtable continuation headers, then remove hyphens. This last operation is deliberately reported as permissive and is never accepted without the prior difference classification.",
            "Canonical body equality does not mean original PDF bytes match, does not establish visual/layout or vector-geometry equivalence, and does not validate scientific claims against source evidence.",
            "Removed visible title, author/course identification, frozen clock, upstream SHA and release-status frontmatter remain a separately reported omission; they are not silently deemed unimportant.",
        ],
    }
    (args.output_directory / "comparison.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.output_directory / "all_text_differences.json").write_text(
        json.dumps(differences, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    concise = ["Character-level differences after the documented initial extraction normalization.",
               "These records do not assert identical original PDF bytes.", ""]
    for difference in differences:
        concise.extend([
            f"{difference['difference_number']:02d}. {difference['classification']}",
            f"  removed: {difference['removed_text']!r}",
            f"  added:   {difference['added_text']!r}",
            f"  reference context: {difference['reference_context']}",
            f"  candidate context: {difference['candidate_context']}", "",
        ])
    (args.output_directory / "all_text_differences.txt").write_text(
        "\n".join(concise), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["comparison_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
