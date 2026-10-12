"""Builds audit/part3_evaluation.ipynb from the experiment outputs and executes it.

    python audit/notebook_builders/build_part3.py [--findings]

--findings appends audit/notebook_builders/findings_part3.md as the last cell.
"""
import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

AUDIT = Path(__file__).resolve().parent.parent
PATH = AUDIT / "part3_evaluation.ipynb"
FINDINGS = Path(__file__).with_name("findings_part3.md")
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
md("""# Part 3 — Option A: a verifier, and the evidence it needs

**What changed** (branch `feat/verified-sections`, see `improvement/0002-verified-sections.patch`):
1. *Evidence.* Forecast columns are labelled from the base year (FY+1..FY+3, e.g. `2027E–2029E` for NVIDIA) instead
   of a hard-coded `2025E–2027E`, and every writer prompt starts with reading notes: the report date, which columns
   are reported and which are the pipeline's own forecast (with its assumptions), the CAGR period, which peer year is
   projected, and that the data contains no guidance, consensus or news.
2. *Verifier* (`modules/claim_verifier.py`, `--enable-verifier`, default `gpt-4.1-mini`). After each section is
   written, it checks the section against the writer's data (plus the report date and notes) and replaces sentences
   that mislabel reported vs. forecast figures, misattribute the CAGR period, misquote numbers or comparisons, or cite
   sources that do not exist. Corrections are exact sentence replacements, logged in `verifier_log.json`. Statements
   the data does not cover are listed, not deleted.

**Experiment** (`audit/part3_experiment.py`): 2 x 2 on the same saved data for 10 companies x 7 sections.
`C0` original prompt / `C1` annotated prompt, each with and without the verifier. The writer is called exactly as the
pipeline calls it (gpt-4.1, temperature 0.7). Each text is scored against the prompt its writer saw by the Part 2.3
scorer (gpt-4.1 judge + rules), and by the LLM-free status-error counter that neither the writer nor the verifier
uses. The five healthcare names are a held-out set: they were never used while designing or tuning
the changes (the pilots used NVDA and AMD)."""),

code("""import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd().parent if Path.cwd().name == "audit" else Path.cwd()
sys.path.insert(0, str(ROOT / "audit"))
import grounding as g

pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 140)
S = pd.read_csv(ROOT / "audit" / "part3_sections.csv")
C = pd.read_csv(ROOT / "audit" / "part3_claims.csv").fillna("")
T = [json.loads(l) for l in open(ROOT / "audit" / "part3_texts.jsonl", encoding="utf-8")]
ORDER = ["C0", "C0+V", "C1", "C1+V"]
S["set"] = np.where(S["held_out"], "held-out (healthcare)", "development")


def relabel(fm):  # as in part3_experiment.py: C1 tables carry FY+1..FY+3 labels
    base = max(int(c[:4]) for c in fm.columns if c.endswith("A") and c[:4].isdigit())
    return fm.rename(columns={"2025E": f"{base + 1}E", "2026E": f"{base + 2}E", "2027E": f"{base + 3}E"})


# Recompute the LLM-free counter from the texts so the notebook always uses the current counter.
TABLES = {}
for t in S["ticker"].unique():
    t0 = g.load_tables(g.OUT / t)
    TABLES[(t, "C0")], TABLES[(t, "C1")] = t0, {**t0, "financial_metrics": relabel(t0["financial_metrics"])}
TEXT = {(r["ticker"], r["section"], r["condition"]): r["text"] for r in T}
counts = [g.status_errors(TEXT[(r.ticker, r.section, r.condition)], TABLES[(r.ticker, r.condition[:2])])
          for r in S.itertuples()]
for k in ("actual_as_forecast", "cagr_period", "invented_source", "total"):
    S[f"counter_{k}"] = [c[k] for c in counts]
print(f"{S[['ticker', 'section']].drop_duplicates().shape[0]} company-sections x {S['condition'].nunique()} conditions; "
      f"{len(C)} claims scored; verifier errors: {S['verifier_error'].notna().sum()}")"""),

md("""## 1. Is the counter itself trustworthy?

Before using the counter as an outcome, it has to pass the same test as any checker: on sentences generated
mechanically from each company's own table it must report nothing (no false positives), and on the same sentences
with one known error injected it must report every one."""),

code("""def oracle_and_faults(tables):
    fm = tables["financial_metrics"].set_index("metrics")
    a_cols = [c for c in fm.columns if c.endswith("A") and c[:4].isdigit()]
    e_cols = [c for c in fm.columns if c.endswith("E") and c[:4].isdigit()]
    a, last, first = a_cols[-1], int(a_cols[-1][:4]), a_cols[0]
    rev_a = float(fm.loc["Revenue", a]) / 1e9
    rev_e = float(fm.loc["Revenue", e_cols[0]]) / 1e9
    marg = str(fm.loc["EBITDA Margin", a])
    e_marg = {str(fm.loc["EBITDA Margin", c]) for c in e_cols}
    cagr = str(fm["CAGR"].iloc[0])
    oracle = [f"Revenue was ${rev_a:.1f} billion in FY{last}.",
              f"EBITDA margin was {marg} in FY{last}.",
              f"The pipeline's forecast for {e_cols[0]} revenue is ${rev_e:.1f} billion.",
              f"Revenue CAGR was {cagr} from {first[:4]} to {last}."]
    faults = {"actual presented as forecast": f"Revenue is projected to reach ${rev_a:.1f} billion in {last}.",
              "E label on a reported value": f"Revenue of ${rev_a:.1f} billion ({last}E) shows strong momentum.",
              "CAGR attributed to forecast years": f"Revenue is set to grow at a {cagr} CAGR through {last + 3}.",
              "invented consensus": f"Consensus expects revenue of ${rev_e:.1f} billion next year."}
    if marg not in e_marg:  # skip when the reported margin also appears as a forecast value
        faults["reported margin presented as estimate"] = f"EBITDA margin is expected to reach {marg} in {last}."
    return oracle, faults


rows = []
for t in S["ticker"].unique():
    tables = g.load_tables(g.OUT / t)
    oracle, faults = oracle_and_faults(tables)
    fp = sum(g.status_errors(s, tables)["total"] for s in oracle)
    hits = {k: g.status_errors(v, tables)["total"] > 0 for k, v in faults.items()}
    rows.append({"ticker": t, "false positives (oracle)": fp, "faults injected": len(hits),
                 "faults detected": sum(hits.values()), "missed": ", ".join(k for k, v in hits.items() if not v) or "-"})
selfcheck = pd.DataFrame(rows).set_index("ticker")
print("TOTAL false positives:", selfcheck["false positives (oracle)"].sum(),  # see Findings for the precision on real text
      "| detected:", selfcheck["faults detected"].sum(), "/", selfcheck["faults injected"].sum())
selfcheck"""),

md("""## 2. Results by condition

Rates are pooled over claims; the 95% interval bootstraps over the 70 company-sections. *Supported claims* is an
absolute count: a fix that deleted content could lower the error rate while making the report emptier."""),

code("""def boot(df, stat, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    units = df.groupby(["ticker", "section"])
    keys = list(units.groups)
    vals = []
    for _ in range(n):
        pick = [keys[i] for i in rng.integers(0, len(keys), len(keys))]
        vals.append(stat(pd.concat([units.get_group(k) for k in pick])))
    return np.percentile(vals, [2.5, 97.5])


def summarize(df):
    out = {}
    for cond in ORDER:
        d = df[df.condition == cond]
        rate = lambda x, col: x[col].sum() / x["claims"].sum()
        lo, hi = boot(d, lambda x: rate(x, "contradicted"))
        out[cond] = {
            "claims": d["claims"].sum(), "supported claims": d["supported"].sum(),
            "grounding rate": rate(d, "supported"), "unsupported": rate(d, "unsupported"),
            "contradicted": rate(d, "contradicted"), "contradicted 95% CI": f"{lo:.1%}-{hi:.1%}",
            "quant. claims contradicted": d["quant_contradicted"].sum() / max(d["quant_claims"].sum(), 1),
            "status errors per section (counter)": d["counter_total"].mean(),
            "sections with >=1 status error": (d["counter_total"] > 0).mean(),
            "words per section": d["words"].mean(),
        }
    return pd.DataFrame(out)


summarize(S).round(3)"""),

code("""for name, d in S.groupby("set"):
    print(f"\\n=== {name}")
    print(summarize(d).round(3).to_string())"""),

md("""## 3. Paired comparison against C0

Same company-section, same data: the mean change per section in contradicted claims, counter-detected status errors
and supported claims, with bootstrap 95% intervals."""),

code("""def paired(df, cond, col):
    a = df[df.condition == "C0"].set_index(["ticker", "section"])[col]
    b = df[df.condition == cond].set_index(["ticker", "section"])[col]
    diff = (b - a).dropna()
    rng = np.random.default_rng(1)
    bs = [diff.sample(len(diff), replace=True, random_state=int(rng.integers(1e9))).mean() for _ in range(2000)]
    return f"{diff.mean():+.2f} ({np.percentile(bs, 2.5):+.2f} to {np.percentile(bs, 97.5):+.2f})"


rows = []
for cond in ORDER[1:]:
    for name, d in [("all", S)] + list(S.groupby("set")):
        rows.append({"condition": cond, "set": name,
                     "contradicted claims / section": paired(d, cond, "contradicted"),
                     "status errors / section (counter)": paired(d, cond, "counter_total"),
                     "supported claims / section": paired(d, cond, "supported")})
pd.DataFrame(rows).set_index(["condition", "set"])"""),

code("""by_sec = S.groupby(["section", "condition"]).apply(
    lambda d: d["contradicted"].sum() / d["claims"].sum(), include_groups=False).unstack()[ORDER]
by_sec.loc["ALL"] = [S[S.condition == c]["contradicted"].sum() / S[S.condition == c]["claims"].sum() for c in ORDER]
by_sec.round(3)"""),

md("""## 4. What the verifier did"""),

code("""logs = [(r["ticker"], r["section"], r["condition"], r["verifier_log"]) for r in T if r["verifier_log"]]
corr = pd.DataFrame([{"ticker": t, "section": s, "condition": c, **x} for t, s, c, l in logs for x in l["corrections"]])
print("sections checked:", len(logs), "| corrections applied:", len(corr),
      "| proposed but not applied (sentence not found verbatim):", sum(len(l["not_applied"]) for *_, l in logs),
      "| statements listed as unverified:", sum(len(l["unverified"]) for *_, l in logs))
print(pd.crosstab(corr["problem"], corr["condition"], margins=True))"""),

code("""pd.set_option("display.max_colwidth", 200)
corr.sample(min(12, len(corr)), random_state=3)[["ticker", "section", "condition", "problem", "original", "corrected"]]"""),
]

if "--findings" in sys.argv:
    cells.append(md(FINDINGS.read_text(encoding="utf-8")))

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(AUDIT)}}).execute()
nbf.write(nb, PATH)
print("written + executed:", PATH)
