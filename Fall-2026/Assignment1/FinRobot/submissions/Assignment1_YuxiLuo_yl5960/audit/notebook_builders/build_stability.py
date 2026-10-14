"""Builds audit/2_3b_stability.ipynb and executes it.

    python audit/notebook_builders/build_stability.py [--findings]
"""
import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

AUDIT = Path(__file__).resolve().parent.parent
PATH = AUDIT / "2_3b_stability.ipynb"
FINDINGS = Path(__file__).with_name("findings_stability.md")
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
md("""# Part 2.3 (b) — Stability: does the same input give the same answer?

NVIDIA was run three times on 2026-10-10 with identical arguments (`scripts/run_pipeline.py NVDA ... --tag stab1/2/3`),
minutes apart. Compared: the page-1 rating and target, the valuation engine's target, the top-3 risks, and the grounding
of every section (Part 2.3 scorer, plus the LLM-free status-error counter)."""),

code("""import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path.cwd().parent if Path.cwd().name == "audit" else Path.cwd()
sys.path.insert(0, str(ROOT / "audit"))
import grounding as g
from openai import OpenAI
from modules.common_utils import load_config, get_api_key

pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 90)
RUNS = {f"stab{i}": g.OUT / "NVDA" / f"stab{i}" for i in (1, 2, 3)}
client = OpenAI(api_key=get_api_key(load_config(str(g.SRC.parent / "config" / "config.ini")),
                                    section="API_KEYS", key="openai_api_key"), max_retries=8)
summary = pd.read_csv(ROOT / "notes" / "run_summary.csv").set_index("run")
summary.loc[[f"NVDA/{r}" for r in RUNS],
            ["price", "page1_rating", "page1_target", "fmp_grade", "engine_target", "engine_up_down"]]"""),

md("""Every page-1 number is identical across runs because none of them comes from the LLM: price, consensus target and
broker grade are fetched from FMP, the rating is a fixed rule on target vs. price, and the engine is a closed-form
function of the forecast table (Part 2.2). They are stable by construction, so a stable headline says nothing about
whether the text underneath agrees with it."""),

code("""def top_risks(text: str, n: int = 5) -> list[str]:
    heads = re.findall(r"^\\s*(?:[-*\\u2022]|\\d+\\.)\\s*\\*\\*([^*]+?)\\*\\*", text, re.M)
    return [h.strip(" :") for h in heads][:n]


CATEGORIES = {  # keyword -> risk category, so differently worded headings can be compared
    "valuation": "valuation", "growth": "growth sustainability", "sustainab": "growth sustainability",
    "margin": "margins / profitability", "profitab": "margins / profitability",
    "compet": "competition", "peer": "competition", "execution": "execution / operations",
    "operational": "execution / operations", "concentration": "concentration / cyclicality",
    "cyclical": "concentration / cyclicality", "demand": "concentration / cyclicality",
}


def category(heading: str) -> str:
    h = heading.lower()
    return next((c for k, c in CATEGORIES.items() if k in h), "other: " + heading)


risks = {r: top_risks((d / "analysis" / "risks.txt").read_text(encoding="utf-8")) for r, d in RUNS.items()}
pd.DataFrame({r: [f"{h}  [{category(h)}]" for h in hs] for r, hs in risks.items()},
             index=[f"risk {i}" for i in range(1, 6)])"""),

code("""def jaccard(a, b):
    return len(a & b) / len(a | b)


cats = {r: [category(h) for h in hs] for r, hs in risks.items()}
pairs = [("stab1", "stab2"), ("stab1", "stab3"), ("stab2", "stab3")]
print("top-3 identical as ordered lists:", len({tuple(c[:3]) for c in cats.values()}) == 1)
print("top-3 category overlap (Jaccard):", {f"{a}-{b}": round(jaccard(set(cats[a][:3]), set(cats[b][:3])), 2) for a, b in pairs})
print("all-5 category overlap (Jaccard):", {f"{a}-{b}": round(jaccard(set(cats[a]), set(cats[b])), 2) for a, b in pairs})"""),

md("""## Grounding across runs

Same scorer as Part 2.3, applied to each run's seven Step 1 sections and the two sections the report regenerates."""),

code("""rows = []
for r, d in RUNS.items():
    claims, sections = g.score_run(client, "gpt-4.1", "NVDA", "NVIDIA Corporation", run_dir=d, skip=("news_summary",))
    tables = g.load_tables(d)
    for (s, v), grp in claims.groupby(["section", "variant"]):
        text = sections.query("section == @s and variant == @v")["text"].iloc[0]
        vc = grp["verdict"].value_counts()
        rows.append({"run": r, "section": s, "variant": v, "claims": len(grp),
                     "supported": vc.get("Supported", 0), "contradicted": vc.get("Contradicted", 0),
                     "status errors (counter)": g.status_errors(text, tables)["total"],
                     "words": len(text.split())})
stab = pd.DataFrame(rows)
stab["grounding rate"] = stab["supported"] / stab["claims"]
stab.to_csv(ROOT / "audit" / "stability_grounding.csv", index=False)
wide = stab.pivot_table(index=["section", "variant"], columns="run", values="grounding rate")
wide["range (max-min)"] = wide.max(axis=1) - wide.min(axis=1)
wide.round(2)"""),

code("""s1 = stab[stab.variant == "step1"].groupby("run")[["claims", "supported", "contradicted", "status errors (counter)"]].sum()
s1["grounding rate"] = s1["supported"] / s1["claims"]
s1["contradicted rate"] = s1["contradicted"] / s1["claims"]
s1.round(3)"""),
]

if "--findings" in sys.argv:
    cells.append(md(FINDINGS.read_text(encoding="utf-8")))

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(AUDIT)}}).execute()
nbf.write(nb, PATH)
print("written + executed:", PATH)
