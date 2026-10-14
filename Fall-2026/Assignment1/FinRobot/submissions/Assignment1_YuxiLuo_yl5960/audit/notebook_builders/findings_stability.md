## Findings

1. **The headline is perfectly stable, and that is uninformative.** Rating (Buy), target ($339.17), broker grade and
   the engine's $82.24 are identical in all three runs, because none of them passes through the LLM. They come from FMP
   and from fixed formulas.
2. **The risk list is stable as a set, not as a ranking.** All three runs name the same five risk categories (Jaccard
   1.0). Valuation and growth sustainability are always #1 and #2, but #3 alternates between margins and competition,
   so the top-3 overlap is 0.5 / 0.5 / 1.0. A "top-3 risks" output is not reproducible.
3. **Error rates are stable; individual errors are not.** Over the seven Step 1 sections the grounding rate is
   43% / 44% / 46% and the contradicted rate 25% / 23% / 21%; the LLM-free counter finds 14 / 13 / 8 status errors.
   Section by section the same input gives very different results: the tagline is 100% / 75% / 50% grounded, the
   Step 1 competitor analysis 60% / 55% / 88%, and the report's gpt-4o-mini takeaways 100% / 67% / 0%.
4. **Implication.** At temperature 0.7 one run is one sample. Any section-level claim about quality needs several
   samples, which is why Part 3 is evaluated on 70 company-sections per condition. Because the error *rate* is a
   property of the prompt and the evidence, the fix belongs there, not in patching individual outputs.
