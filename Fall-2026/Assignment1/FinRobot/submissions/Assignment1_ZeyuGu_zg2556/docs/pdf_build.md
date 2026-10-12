# Selected research PDF and verification

The course submission contains `output/pdf/research_report_submission.pdf`, the unchanged user-provided 16-page research report. Its SHA-256, original attachment identity, complete page review, two minor presentation findings and historical body comparison are recorded in `evidence/submission_pdf_review/selection.json`.

The user explicitly excluded the research-report LaTeX/Markdown sources and earlier report editions from the course PR. Those files remain on the author's computer. `config/research_report_submission_policy.json` lists the exclusions; `.gitignore` and the scoped exporter enforce them. The submission still includes experimental source code, frozen inputs, independent valuation notebooks, audit tables and reproduction commands. This choice does not remove experiment evidence or change any reported result.

From the submission directory, run:

```powershell
python scripts/validate_submission_selection.py
python scripts/validate_delivery.py
```

These commands validate the selected PDF and submitted evidence without requiring the excluded report sources or old PDFs. If historical report files happen to be present locally, their hashes are additionally checked. Their absence in a clean course copy is expected and is not disguised as a reproduced manuscript build.

Poppler can render the selected file for a new visual inspection:

```powershell
pdftoppm -r 120 -png output/pdf/research_report_submission.pdf selected-report-page
```

All 16 pages were inspected in the recorded review. No clipping, missing charts or unreadable result values were found. Missing visible identification/provenance frontmatter and a repeated table header on page 10 remain disclosed minor presentation findings; the selected bytes were not silently edited.

The stored comparison with an older LaTeX PDF is historical evidence. Repeating that comparison requires the author's local older PDF or a separately provided reference; it is not a clean-submission reproduction command. Historical report-authoring/build utilities likewise require the excluded local manuscript files and are not the experiment reproduction entry points. The external PDF's exact build from the current local source is not claimed. No publication or human financial validation is implied.
