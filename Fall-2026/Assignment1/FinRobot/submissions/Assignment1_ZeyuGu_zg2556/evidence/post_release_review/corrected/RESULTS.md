# Accepted post-release measurement errata

These are new machine-adjudication versions of unchanged V8.2 reports. They correct measurement, not the verifier. No human annotation or new paid model call occurred.

The31 accepted original atoms map to32 atoms because AMD's supported PE decline is retained separately from its unsupported elevated-level judgment. All other atoms, source texts and original audits remain unchanged.

|Company|Before S/U/C/total|After S/U/C/total|
|---|---|---|
|NVDA|82/392/12/486|175/493/2/670|
|AMD|91/455/8/554|141/500/4/645|
|INTC|113/315/11/439|214/427/2/643|
|ALL|286/1162/31/1479|530/1420/8/1958|

Corrected warning recall: 99.15966387%; precision: 89.90476190%; unflagged unsupported/contradicted: 12, including 7 contradictions.

Supported lexical-key retention: 9.67741935%; evidence-span retention: 81.81818182%. Changed wording/atomization affects these proxies; they do not establish semantic factual recall.

Original source errors remain: seven explicit CAGR-scope contradictions and the AMD numerical-range error after treatment. Unsupported labels do not imply truth or successful repair.

Reproduce from the submission root in the recorded dedicated environment:

```powershell
python scripts/materialize_post_release_errata.py --validate-only
python scripts/materialize_post_release_errata.py --suffix corrected_reproduction
```

The script pins proposal/config hashes, verifies every original source unit and evidence quote, refuses existing output destinations, and compares protected input hashes after writing. `old_new_atom_mapping.csv` preserves every accepted change and its reason. `results.json` contains exact counts and metrics; the new paired config points to versioned audit directories.
