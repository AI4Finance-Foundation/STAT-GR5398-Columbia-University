# HTML visual review

Seven baseline reports were rendered from preserved inputs: NVDA baseline runs 1–3 and AMD, INTC, AAPL and GOOGL run 1. Browser checks in summary.json confirm all eight exact source sections, three decoded local images per report, no browser page errors and no document-level horizontal overflow at desktop 1360px and mobile 375px widths. Market fields and frozen engine values are supplied separately from writer prompts through hashed render_inputs.

The coding agent visually inspected the NVDA desktop page and mobile top, full-size NVDA revenue chart, AMD EPS chart and INTC peer chart, and a contact sheet containing all fifteen distinct five-company figures. Titles, legends, axis labels, fiscal actual/forecast labels and chart edges were visible; no clipping, blank images or encoding corruption was found. NVDA repeated-run figures use the same frozen data. This is agent-assisted visual QA, not a claim of human review.

The screenshot checks certify rendering and source-text fidelity. They do not certify financial correctness. In particular, baseline narratives still contain unsupported claims and mischaracterized actual/forecast periods, and INTC has economically problematic negative EPS/P-E forecasts; separate financial/grounding audits evaluate those issues. No baseline prose was rewritten during rendering.

Rerun: `python scripts/qa_reports.py --submission .` from the submission root, in the dedicated environment. The default browser is locally installed Microsoft Edge; `--browser` accepts another installed Chromium executable. External HTTP requests are blocked by the QA browser route.
