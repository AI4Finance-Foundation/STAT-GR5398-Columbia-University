## Findings

**Instrument check.** The LLM-free counter reports 0 false positives on table-generated sentences and detects 50/50
injected status errors across the ten companies. On real text its precision is lower: a manual check of 10 flagged C0
sentences found 11 of 13 counted errors to be real (the two misses were a correctly derived forecast-period CAGR and a
number that coincided with a reported value). It is a corroborating measure; the judge is the primary one.

1. **Labelling the evidence does most of the work.** Annotating the prompt (C1) cuts contradicted claims from 16.3% to
   1.9% and counter-detected status errors from 1.47 to 0.03 per section (sections with any such error: 77% to 3%).
   The grounding rate rises from 58% to 83%, and unsupported claims fall from 26% to 16%: told what the data is, the
   writer uses more of it. This is not bought by writing less. Supported claims rise from 413 to 680 and sections get
   slightly longer (227 to 244 words). **The effect is the same on the held-out healthcare names** (16.6% to 1.5%
   contradicted), which were never used while designing or tuning the changes.
2. **A verifier on its own helps, but only with the context the writer lacked.** On the original prompt (C0+V) it cuts
   contradicted claims to 5.7%. In the pilot, without the report date, it repeated the writer's mistake: it rewrote
   "consensus estimates project revenues to climb to $130.5 billion in 2025" as "revenues are projected to climb
   to $130.5 billion in 2025", still presenting a reported figure as a forecast. With the date it fixes this class of
   error.
3. **On top of the annotation, the verifier adds little.** Contradicted claims go from 1.9% to 1.3%, but the intervals
   overlap (1.2–2.7% vs 0.6–2.1%) and the counter is unchanged at 0.03 errors per section. Its cost is one gpt-4.1-mini
   call per section. It is worth keeping as a cheap safety net and for writers whose prompt cannot be changed; it is
   not where the improvement comes from.
4. **How the verifier behaves.** It applied 399 corrections, mostly status (187), invented sources (88) and
   values/comparisons (95). Another 313 proposed corrections were not applied because the model did not quote the
   original sentence verbatim. That is safe by design, but it wastes 44% of its output. Some "corrections" are noise,
   such as replacing $26.9 billion with $26.914 billion. It listed 628 statements as unverified rather than deleting
   them.

**What it still gets wrong**
- Unsupported background claims remain (15% in C1+V). The company overview is still asked for products and strategy
  that the data does not contain. Fixing that needs either evidence for it (segments, filings text) or a narrower
  prompt.
- Step 2's `--enable-text-regeneration` rewrites two sections with gpt-4o-mini outside both fixes. It should be turned
  off, or routed through the same prompt and verifier.
- Judge and writer are the same model family; the judge was 91% accurate on the Part 2.3 review sample and lenient
  rather than strict. Each condition has one sample per company-section; the stability runs show section-level
  results vary a lot between samples, which is why the comparison pools 70 of them.
- Out of scope and unchanged: forecast EBITDA still omits R&D, and the valuation engine is still a fixed 10.84x
  multiple (Part 2.2).
