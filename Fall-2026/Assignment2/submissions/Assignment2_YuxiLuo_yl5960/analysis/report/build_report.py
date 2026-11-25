"""Build the Assignment 2 report: tables and figures straight from the output files, then a PDF.

    /opt/anaconda3/bin/python analysis/report/build_report.py            # draft: report.html + report.pdf
    /opt/anaconda3/bin/python analysis/report/build_report.py --final    # refuses while any placeholder remains

Runs with Anaconda Python (matplotlib), not the course venv. It reads only files that already exist
(course output/, analysis/ logs and study/ results) and never runs a backtest or the test period.
Every number in a generated table comes from those files; the evaluation tables are recomputed from
the evaluation CSVs and checked against the evaluate_valuation logs.

report_src.html holds the prose. Markers it understands:
    <!-- TABLE:name -->   replaced by a generated table (see TABLES)
    <!-- FIG:name -->     replaced by a figure (see FIGURES), written to report/figures/
    <div class="todo">    a writing placeholder; drafts show it, --final fails on it
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WORK = HERE.parent          # the analysis folder: analysis/ when working, <submission>/analysis/ when submitted


def _find_source_code():
    """Assignment2/source_code/, found by walking up from this script: the submission layout
    (Assignment2/submissions/<team>/analysis[/...]) or the working layout (<root>/a2_work beside course-repo/)."""
    for parent in HERE.parents:
        for cand in (parent / "source_code", parent / "course-repo" / "Fall-2026" / "Assignment2" / "source_code"):
            if (cand / "run_portfolio.py").is_file():
                return cand
    sys.exit("cannot find Assignment2/source_code/ above " + str(HERE))


class _OutDirs:
    """Output files: the submission's own output/ first (it holds the test-period files, which cannot be
    regenerated), then source_code/output/ (anything reproduce.sh regenerates)."""

    def __init__(self, roots):
        self.roots = [r for r in roots if r.is_dir()]

    def __truediv__(self, rel):
        for r in self.roots:
            if (r / rel).exists():
                return r / rel
        return self.roots[-1] / rel


SRC = _find_source_code()
OUT = _OutDirs([WORK.parent / "output", SRC / "output"])
FIGS = HERE / "figures"
CHROME = os.environ.get("CHROME", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
PERIODS = ("dev", "val", "test")
PERIOD_LABEL = {"dev": "dev 2017–21", "val": "val 2022–23", "test": "test 2024–26"}
METHODS = ("dcf", "ev_ebitda", "peer_multiple")
METHOD_LABEL = {"dcf": "DCF", "ev_ebitda": "EV/EBITDA own history", "peer_multiple": "peer trailing P/E"}

# Reference palette (dataviz skill), light mode: categorical slots 1-2, neutral ink.
BLUE, ORANGE, GRAY, INK, INK2, GRID = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b", "#52514e", "#e4e3df"


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def fmt(x, nd=2, pct=False, sign=False, pp=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return ""
    s = f"{x:+,.{nd}f}" if sign else f"{x:,.{nd}f}"
    s = s.replace("-", "−")
    return s + ("%" if pct else "") + (" pp" if pp else "")


def html_table(header, rows, num_from=1, cls="", group=False, widths=None):
    """header: list of str; rows: list of lists of str. Columns from num_from on are right-aligned.
    group=True shows the first column only where it changes and draws a rule above each group."""
    def cell(tag, v, i):
        c = ' class="n"' if i >= num_from else (' class="k"' if i == 0 and group else "")
        return f"<{tag}{c}>{v}</{tag}>"
    h = "".join(cell("th", v, i) for i, v in enumerate(header))
    body, prev = [], None
    for r in rows:
        r = list(r)
        start = group and r[0] != prev
        if group:
            prev, r[0] = r[0], (r[0] if start else "")
        tr = '<tr class="grp">' if start and body else "<tr>"
        body.append(tr + "".join(cell("td", v, i) for i, v in enumerate(r)) + "</tr>")
    cg = "<colgroup>" + "".join(f'<col style="width:{w}">' for w in widths) + "</colgroup>" if widths else ""
    return f'<table class="{cls}">{cg}\n<thead><tr>{h}</tr></thead>\n<tbody>\n' + "\n".join(body) + "\n</tbody></table>"


def jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def ic_series(res, h):
    """Monthly Spearman IC of mispricing vs ret_h, exactly as evaluate_valuation.py computes it."""
    col = f"ret_{h}"
    r = res.dropna(subset=[col])
    ic = r.groupby("as_of").apply(
        lambda g: g["mispricing"].corr(g[col], method="spearman") if len(g) >= 5 else np.nan,
        include_groups=False).dropna()
    return ic


def ic_stats(res, h):
    ic = ic_series(res, h)
    t = ic.mean() / ic.std() * np.sqrt(len(ic)) if len(ic) > 1 and ic.std() > 0 else np.nan
    return ic.mean(), t, len(ic)


# ---------------------------------------------------------------------------
# data loaders
# ---------------------------------------------------------------------------
EVAL_FILES = {  # (period, which) -> (csv, log)
    ("dev", "module"): ("valuation_eval_dev.csv", "eval_dev.log"),
    ("val", "module"): ("valuation_eval_val.csv", "eval_val.log"),
    ("test", "module"): ("valuation_eval_test.csv", "test_evaluate.log"),
    ("dev", "example"): ("valuation_eval_dev_example.csv", "eval_dev_example.log"),
    ("val", "example"): ("valuation_eval_val_example.csv", "eval_val_example.log"),
}


def eval_summary(period, which):
    csv, log = EVAL_FILES[(period, which)]
    res = pd.read_csv(OUT / csv)
    text = (WORK / "logs" / log).read_text()
    asked = int(re.search(r"valued (\d+) of (\d+)", text).group(2))
    row = {"period": period, "which": which, "valued": len(res), "asked": asked}
    for h in (63, 126, 252):
        m, t, n = ic_stats(res, h)
        logged = re.search(rf"rank IC\s+{h} sessions: mean ([+-]\d\.\d+)\s+t ([+-]\d+\.\d+)\s+months (\d+)", text)
        assert logged and round(m, 3) == float(logged.group(1)) and round(t, 2) == float(logged.group(2)) \
            and n == int(logged.group(3)), f"{csv} IC{h} does not match {log}"
        row.update({f"ic{h}": m, f"t{h}": t, f"n{h}": n})
    r = res.dropna(subset=["price_252", "low", "high"])
    row["inside"] = ((r["price_252"] >= r["low"]) & (r["price_252"] <= r["high"])).mean()
    row["inside_n"] = len(r)
    row["med_mis"] = res["mispricing"].median()
    row["med_spread"] = res["method_spread"].median()
    assert f"{row['inside']:.0%} of {len(r)}" in text, f"{csv} range coverage does not match {log}"
    return row, res


def ablation(period):
    df = pd.read_csv(OUT / f"ablation_{period}.csv").set_index("run")
    a, b = df.loc[f"{period}_A_selection"], df.loc[f"{period}_B_valuation"]
    c, d = df.loc[f"{period}_C_timing_full"], df.loc[f"{period}_D_both_full"]
    cols = ["total_return_pct", "sharpe", "max_drawdown_pct", "trades"]
    assert (c[cols] == a[cols]).all() and (d[cols] == b[cols]).all(), f"{period}: C/D differ from A/B"
    return a, b


def equity(run):
    e = pd.read_csv(OUT / run / "equity.csv", index_col=0, parse_dates=True).iloc[:, 0]
    return e / e.iloc[0]


def spy(start, end):
    px = pd.read_parquet(SRC / "data" / "prices" / "daily_adjusted.parquet", columns=["date", "symbol", "close"])
    s = px[px["symbol"] == "SPY"].assign(date=lambda d: pd.to_datetime(d["date"])).set_index("date")["close"]
    s = s.loc[start:end]
    return s / s.iloc[0]


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
def t_constants():
    """Every module constant with its value and the source comment, parsed from the committed file."""
    lines = (SRC / "valuation" / "student_valuation.py").read_text().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("# Constants"))
    end = next(i for i, l in enumerate(lines) if l.startswith("_TTM_COLUMNS"))
    rows, group, i = [], "", start + 2
    while i < end:
        l = lines[i]
        if l.startswith("# ") and not l.startswith("# ---"):
            group = l[2:].split("(")[0].strip()
        m = re.match(r"^([A-Z][A-Z_]+(?:, [A-Z_]+)?)\s*=\s*(.*)$", l)
        if m:
            name, rest = m.groups()
            while rest.count("(") > rest.count(")"):
                i += 1
                rest += " " + lines[i].strip()
            value, _, comment = rest.partition("#")
            value = value.strip()
            if value.count('"') > 6:   # the per-share column tuple
                value = f'{value.count(",") + 1} columns: ' + value.strip("()").replace('"', "").replace(",", ", ")
                value = value.replace(",  ", ", ")
                rows.append([group, f"<code>{name}</code>", value, comment.strip()])
            else:
                rows.append([group, f"<code>{name}</code>", f"<code>{value}</code>", comment.strip()])
        i += 1
    return html_table(["group", "constant", "value", "source / reason"], rows, num_from=99, cls="small", group=True,
                      widths=["15%", "25%", "20%", "40%"])


def t_eval():
    rows = []
    for period in PERIODS:
        for which in ("module", "example"):
            if (period, which) not in EVAL_FILES:
                rows.append([PERIOD_LABEL[period], "course example (P/E only)", "not run (would be a 3rd test run)",
                             "", "", "", "", ""])
                continue
            s, _ = eval_summary(period, which)
            rows.append([PERIOD_LABEL[period], "this module" if which == "module" else "course example (P/E only)",
                         f"{s['valued']} / {s['asked']} ({s['valued'] / s['asked']:.0%})",
                         *(f"{fmt(s[f'ic{h}'], 3, sign=True)} ({fmt(s[f't{h}'], 2, sign=True)})" for h in (63, 126, 252)),
                         f"{s['inside']:.0%} of {s['inside_n']}", fmt(100 * s["med_mis"], 1, pct=True, sign=True)])
    return html_table(["period", "valuation", "valued", "IC63 (t)", "IC126 (t)", "IC252 (t)",
                       "price in [low, high] after 252", "median mispricing"], rows, num_from=2, group=True)


def t_ablation():
    rows = []
    for p in PERIODS:
        a, b = ablation(p)
        for name, r in (("A selection (= C)", a), ("B + filter (= D)", b)):
            rows.append([PERIOD_LABEL[p], name, fmt(r.total_return_pct, 2, pct=True), fmt(r.cagr_pct, 2, pct=True),
                         fmt(r.volatility_pct, 2, pct=True), fmt(r.sharpe, 3), fmt(r.max_drawdown_pct, 2, pct=True),
                         fmt(r.calmar, 3), f"{int(r.trades)}", fmt(r.turnover_per_year, 2),
                         fmt(r.avg_removed_per_week, 1) if name.startswith("B") else ""])
        for bm in ("SPY", "QQQ"):
            rows.append([PERIOD_LABEL[p], bm, fmt(a[f"{bm}_total_return_pct"], 2, pct=True), "", "",
                         fmt(a[f"{bm}_sharpe"], 3), "", "", "", "", ""])
    return html_table(["period", "run", "total return", "CAGR", "volatility", "Sharpe", "max drawdown", "Calmar",
                       "trades", "turnover / yr", "removed / wk"], rows, num_from=2, group=True)


def t_marginal():
    rows = []
    for p in PERIODS:
        a, b = ablation(p)
        rows.append([PERIOD_LABEL[p], fmt(b.cagr_pct - a.cagr_pct, 2, sign=True, pp=True),
                     fmt(b.sharpe - a.sharpe, 3, sign=True),
                     fmt(b.max_drawdown_pct - a.max_drawdown_pct, 2, sign=True, pp=True),
                     fmt(b.volatility_pct - a.volatility_pct, 2, sign=True, pp=True),
                     fmt(b.turnover_per_year - a.turnover_per_year, 2, sign=True),
                     fmt(b.sharpe - a[f"SPY_sharpe"], 3, sign=True)])
    return html_table(["period", "CAGR", "Sharpe", "max drawdown (+ = shallower)", "volatility",
                       "turnover / yr", "B Sharpe − SPY Sharpe"], rows)


def diag_frame(period):
    d = pd.read_csv(WORK / "sens" / "diag" / f"{period}_base.csv")
    for m in METHODS:
        d[f"mis_{m}"] = d[f"fair_{m}"] / d["price"] - 1
    fairs = d[[f"fair_{m}" for m in METHODS]]
    d["flag"] = (d["n_methods"] >= 2) & (d["price"] > fairs.max(axis=1))
    return d


def t_ic_by_year():
    rows = []
    for p in PERIODS:
        _, res = eval_summary(p, "module")
        res["year"] = (pd.to_datetime(res["as_of"]) + pd.Timedelta(days=7)).dt.year.astype(str)   # the month valued, not as_of
        diag = diag_frame(p) if p != "test" else None
        if diag is not None:
            diag["year"] = (pd.to_datetime(diag["as_of"]) + pd.Timedelta(days=7)).dt.year.astype(str)
        for y, g in res.groupby("year"):
            ic = ic_series(g, 126)
            row = [PERIOD_LABEL[p], y, f"{fmt(ic.mean(), 2, sign=True)} ({len(ic)})" if len(ic) else "n/a"]
            for m in METHODS:
                if diag is None:
                    row.append("")
                    continue
                dg = diag[diag["year"] == y].assign(mispricing=lambda x: x[f"mis_{m}"])
                icm = ic_series(dg.dropna(subset=["mispricing"]), 126)
                row.append(f"{fmt(icm.mean(), 2, sign=True)} ({len(icm)})" if len(icm) else "n/a")
            if diag is None:
                row += ["", ""]
            else:
                dg = diag[(diag["year"] == y)].dropna(subset=["ret_126"])
                f, r = dg[dg["flag"]], dg[~dg["flag"]]
                row += [f"{fmt(100 * f['ret_126'].mean(), 1, pct=True, sign=True)} (n={len(f)})" if len(f) else "",
                        fmt(100 * r["ret_126"].mean(), 1, pct=True, sign=True)]
            rows.append(row)
    return html_table(["period", "year", "combined IC126 (months)", "DCF", "EV/EBITDA", "peer P/E",
                       "flagged: next-126 return", "rest"], rows, num_from=2, cls="small", group=True)


SENS_ORDER = ["base", "erp_040", "erp_060", "tg_020", "tg_030", "gw_000", "gw_100", "gcap_025", "gcap_050",
              "nwc_000", "nwc_020", "yrs_05", "yrs_15", "beta_3y", "hist_3y", "peer_cap100", "peers_10",
              "no_align", "bands_10_90", "dcf_step_x2"]


def t_sensitivity():
    dd = {p: pd.read_csv(WORK / "sens" / f"table_diag_{p}.csv").set_index("tag") for p in ("dev", "val")}
    pp = {p: pd.read_csv(WORK / "sens" / f"table_portfolio_{p}.csv").set_index("tag") for p in ("dev", "val")}
    rows = []
    for tag in SENS_ORDER:
        row = [dd["dev"].loc[tag, "variant"].replace(" -> ", " → ")]
        for p in ("dev", "val"):
            d = dd[p].loc[tag]
            row += [f"{fmt(d['IC126'], 3, sign=True)} ({fmt(d['t126'], 2, sign=True)})", fmt(d["rank corr vs base"], 3),
                    fmt(d["flag overlap"], 2)]
            if tag in pp[p].index:
                q = pp[p].loc[tag]
                row += [fmt(q["removed/wk"], 1), fmt(q["Sharpe - A"], 3, sign=True)]
            else:
                row += ["=", "="]
        rows.append(row)
    return html_table(["assumption changed", "dev IC126 (t)", "dev rank corr", "dev flag overlap", "dev removed/wk",
                       "dev Sharpe B − A", "val IC126 (t)", "val rank corr", "val flag overlap", "val removed/wk",
                       "val Sharpe B − A"], rows, cls="small")


RULES = [("committed", "committed: ≥ 2 methods, above all"), ("min1", "≥ 1 method, above all"),
         ("dcf", "DCF alone"), ("ev_ebitda", "EV/EBITDA alone"), ("peer_multiple", "peer P/E alone"),
         ("dcf+ev_ebitda", "DCF + EV/EBITDA"), ("dcf+peer_multiple", "DCF + peer P/E"),
         ("ev_ebitda+peer_multiple", "EV/EBITDA + peer P/E")]


def study(version="v1"):
    """study/portfolio.jsonl keyed by (period, rule) for one module version (v1 = the submitted module)."""
    return {(r["period"], r["variant"]): r for r in jsonl(WORK / "study" / "portfolio.jsonl")
            if r.get("version", "v1") == version}


def t_rules():
    s = study()
    a = {p: ablation(p)[0] for p in ("dev", "val")}
    rows = [["A: no filter", *sum(([fmt(a[p].total_return_pct, 2, pct=True), fmt(a[p].sharpe, 3),
                                    fmt(a[p].max_drawdown_pct, 2, pct=True), ""] for p in ("dev", "val")), [])]]
    for key, label in RULES:
        row = [label]
        for p in ("dev", "val"):
            r = s[(p, key)]
            row += [fmt(r["total_return_pct"], 2, pct=True), fmt(r["sharpe"], 3), fmt(r["max_drawdown_pct"], 2, pct=True),
                    fmt(r["avg_removed_per_week"], 1)]
        rows.append(row)
    return html_table(["filter rule", "dev return", "dev Sharpe", "dev max DD", "dev removed/wk",
                       "val return", "val Sharpe", "val max DD", "val removed/wk"], rows)


def t_swaps():
    s = study()
    rows = []
    for p in ("dev", "val"):
        sw = s[(p, "committed")]["swaps"]
        for h in (5, 21, 63, 126):
            x = sw[f"h{h}"]
            rows.append([PERIOD_LABEL[p], str(h), str(x["n"]), fmt(100 * x["out_mean"], 2, pct=True, sign=True),
                         fmt(100 * x["in_mean"], 2, pct=True, sign=True),
                         fmt(100 * x["in_minus_out"], 2, sign=True, pp=True), fmt(x["t_nw"], 2, sign=True),
                         f"{x['in_beats_out']:.0%}"])
    return html_table(["period", "horizon (sessions)", "weeks", "removed holdings", "replacements",
                       "replacement − removed", "Newey-West t", "weeks replacement wins"], rows, num_from=1, group=True)


def t_spot():
    d = pd.read_csv(WORK / "study" / "spot_check_basis.csv")
    rng = lambda s, nd=1, pct=False: (f"{fmt(100 * s.min(), 0, sign=True)}% to {fmt(100 * s.max(), 0, sign=True)}%"
                                      if pct else f"{fmt(s.min(), nd)}–{fmt(s.max(), nd)}")
    rows = []
    for t, g in d.groupby("ticker", sort=False):
        rows.append([t, rng(g["basis"], 3), rng(g["pe_aligned"]), rng(g["pe_own"]), rng(g["pe_raw"]),
                     rng(g["mis_al"], pct=True), rng(g["mis_raw"], pct=True)])
    return html_table(["ticker", "basis factor", "P/E aligned", "P/E, filing's own basis", "P/E not aligned",
                       "mispricing aligned", "mispricing not aligned"], rows)


def t_lookahead():
    d = pd.read_csv(WORK / "study" / "lookahead_test.csv")
    n = len(d)
    log = (WORK / "study" / "logs" / "lookahead_test.log").read_text()
    lag = re.search(r"\|mispricing change\| median ([\d.]+), p95 ([\d.]+), max ([\d.]+); DCF value change median "
                    r"([\d.]+)%, max ([\d.]+)%", log)
    rows = [
        ["poison everything after as_of", f"{int(d['future_rows_poisoned'].min()):,}–{int(d['future_rows_poisoned'].max()):,}"
         f" filing rows, {int(d['future_snapshots_poisoned'].min())}–{int(d['future_snapshots_poisoned'].max())} member snapshots,"
         " all later prices and macro",
         f"values identical on {int(d['poison_identical_values'].sum())} of {n} dates; filter identical on "
         f"{int(d['poison_identical_filter'].sum())} of {n}; visible rows and members unchanged on "
         f"{int((d['poison_visible_rows_identical'] & d['poison_members_identical']).sum())} of {n}"],
        ["price basis ×10 (KLAC, NVDA)", "whole price history multiplied by 10",
         f"largest mispricing change {d['basis_max_mispricing_diff'].max():.1e}; filter identical on "
         f"{int(d['basis_identical_filter'].sum())} of {n} dates"],
        ["macro one observation later", "DGS10 and BAA10Y lagged one observation",
         f"mispricing change median {lag.group(1)}, p95 {lag.group(2)}, max {lag.group(3)}; DCF value change median "
         f"{lag.group(4)}%, max {lag.group(5)}%; filter identical on {int(d['lag_identical_filter'].sum())} of {n} dates"],
    ]
    return html_table(["test", "what was changed", "result"], rows, num_from=99)


def t_testruns():
    rows = [[str(r["n"]), r["time"].replace("T", " "), r["script"], " ".join(r["argv"]), f"<code>{r['git_head'][:7]}</code>"]
            for r in jsonl(OUT / "test_runs.jsonl")]
    return html_table(["#", "time", "script", "arguments", "commit"], rows, num_from=99)


def t_tuning():
    """The filter grid (dev) and the val finalists rerun with PYTHONHASHSEED=0 (tuning_seeded.jsonl). The original
    runs (tuning_log.jsonl, made before 8288c74 without a fixed seed) are kept as the record of the choice."""
    log = [r for r in jsonl(WORK / "tuning_seeded.jsonl") if r["overrides"]]
    a = {p: ablation(p)[0] for p in ("dev", "val")}
    rows = []
    for p in ("dev", "val"):
        rows.append([PERIOD_LABEL[p], "A (no filter)", "", "0", fmt(a[p].total_return_pct, 2, pct=True),
                     fmt(a[p].sharpe, 3), fmt(a[p].max_drawdown_pct, 2, pct=True), fmt(a[p].turnover_per_year, 2)])
        for r in sorted((r for r in log if r["period"] == p),
                        key=lambda r: (r["overrides"]["FILTER_MIN_METHODS"], r["overrides"]["FILTER_PREMIUM"])):
            o = r["overrides"]
            rows.append([PERIOD_LABEL[p], str(o["FILTER_MIN_METHODS"]), fmt(o["FILTER_PREMIUM"], 1),
                         fmt(r["avg_removed_per_week"], 1), fmt(r["total_return_pct"], 2, pct=True), fmt(r["sharpe"], 3),
                         fmt(r["max_drawdown_pct"], 2, pct=True), fmt(r["turnover_per_year"], 2)])
    return html_table(["period", "min methods", "premium", "removed / wk", "total return", "Sharpe", "max drawdown",
                       "turnover / yr"], rows, num_from=1, group=True)


# ---------------------------------------------------------------------------
# post-test exploration: DCF diagnostic and valuation_v2 (dev and val only)
# ---------------------------------------------------------------------------
V2 = [("v1", "v1 submitted"), ("v2a", "v2a corr. peers"), ("v2b", "v2b growth-adj."), ("v2c", "v2c both")]


def v2_json(p, v):
    f = WORK / "sens" / "diag" / f"{p}_base.json" if v == "v1" else WORK / "study" / "v2" / f"diag_{p}_{v}.json"
    return json.loads(f.read_text())


def v2_rows(p, v):
    if v == "v1":   # dev: the Mac rerun of the committed module (equal to the Windows file up to 5e-13)
        return pd.read_csv(WORK / "study" / "v2" / "diag_dev_v1_mac.csv") if p == "dev" else \
            pd.read_csv(WORK / "sens" / "diag" / f"{p}_base.csv")
    return pd.read_csv(WORK / "study" / "v2" / f"diag_{p}_{v}.csv")


def common_ic126(p, v):
    key = ["as_of", "ticker"]
    b, x = v2_rows(p, "v1"), v2_rows(p, v)
    c = b[key].merge(x[key])
    return len(c), ic_stats(b.merge(c), 126), ic_stats(x.merge(c), 126)


def t_v2_eval():
    rows = []
    for v, label in V2:
        row = [label]
        for p in ("dev", "val"):
            r = v2_json(p, v)["rows"]
            row += [f"{r['combined']['coverage']:.1%}",
                    f"{fmt(r['combined']['IC126'], 3, sign=True)} ({fmt(r['combined']['t126'], 2, sign=True)})"]
            if v == "v1":
                row.append("")
            else:
                n, (m0, t0, _), (m1, t1, _) = common_ic126(p, v)
                row.append(f"{fmt(m0, 3, sign=True)} → {fmt(m1, 3, sign=True)} (n={n})")
            row.append(fmt(r["peer_multiple"]["IC126"], 3, sign=True))
        rows.append(row)
    return html_table(["version", "dev coverage", "dev IC126 (t)", "dev IC126 on common rows: v1 → this",
                       "dev peer-method IC126", "val coverage", "val IC126 (t)", "val IC126 on common rows: v1 → this",
                       "val peer-method IC126"], rows, cls="small")


def t_v2_portfolio():
    a = {p: ablation(p)[0] for p in ("dev", "val")}
    rows = [["A: no filter", *sum(([fmt(a[p].total_return_pct, 2, pct=True), fmt(a[p].sharpe, 3), "",
                                    fmt(a[p].max_drawdown_pct, 2, pct=True), ""] for p in ("dev", "val")), [])]]
    for v, label in V2:
        s = study(v)
        row = [label]
        for p in ("dev", "val"):
            r = s[(p, "committed")]
            row += [fmt(r["total_return_pct"], 2, pct=True), fmt(r["sharpe"], 3), fmt(r["sharpe"] - a[p].sharpe, 3, sign=True),
                    fmt(r["max_drawdown_pct"], 2, pct=True), fmt(r["avg_removed_per_week"], 1)]
        rows.append(row)
    return html_table(["B with the submitted filter rule", "dev return", "dev Sharpe", "dev Sharpe − A", "dev max DD",
                       "dev removed/wk", "val return", "val Sharpe", "val Sharpe − A", "val max DD", "val removed/wk"],
                      rows, cls="small")


def yearly(run):
    e = pd.read_csv(OUT / run / "equity.csv", index_col=0, parse_dates=True).iloc[:, 0]
    y = e.groupby(e.index.year).last()
    prev = np.r_[e.iloc[0], y.values[:-1]]
    return pd.Series((y.values / prev - 1) * 100, index=y.index)


def t_v2_years():
    rows = []
    for p in ("dev", "val"):
        a, b1 = yearly(f"{p}_A_selection"), yearly(f"{p}_B_valuation")
        b2 = yearly(f"tuning/{p}_B_study_committed_v2c")
        for y in a.index:
            rows.append([PERIOD_LABEL[p], str(y), fmt(a[y], 2, pct=True), fmt(b1[y], 2, pct=True), fmt(b2[y], 2, pct=True),
                         fmt(b1[y] - a[y], 2, sign=True, pp=True), fmt(b2[y] - a[y], 2, sign=True, pp=True)])
    return html_table(["period", "year", "A", "B v1", "B v2c", "v1 − A", "v2c − A"], rows, num_from=2, group=True,
                      cls="small")


def dcf_diag(p):
    d = pd.read_csv(WORK / "study" / f"dcf_diagnostic_{p}.csv")
    return d


def t_dcf_quintiles():
    rows = []
    for p in ("dev", "val"):
        g = dcf_diag(p).dropna(subset=["growth"]).copy()
        g["q"] = pd.qcut(g["growth"], 5, labels=["1 (slowest)", "2", "3", "4", "5 (fastest)"])
        for q, x in g.groupby("q", observed=True):
            rows.append([PERIOD_LABEL[p], str(q), fmt(100 * x["growth"].median(), 0, pct=True, sign=True),
                         fmt(x["r_dcf"].median(), 2), fmt(x["r_ev_ebitda"].median(), 2),
                         fmt(x["r_peer_multiple"].median(), 2), f"{x['flag_committed'].mean():.0%}"])
    return html_table(["period", "growth quintile", "median growth", "DCF / price", "EV/EBITDA / price",
                       "peer P/E / price", "flagged"], rows, num_from=2, group=True, cls="small")


def t_dcf_groups():
    rows = []
    names = [("both", "flagged by both rules (removed)"), ("committed only", "submitted rule only"),
             ("EV+peer only", "EV/EBITDA + peer rule only (DCF says cheap)"), ("neither", "not flagged")]
    for p in ("dev", "val"):
        d = dcf_diag(p)
        grp = np.select([d["flag_committed"] & d["flag_evpeer"], d["flag_committed"], d["flag_evpeer"]],
                        ["both", "committed only", "EV+peer only"], "neither")
        for key, label in names:
            x = d[grp == key]
            rows.append([PERIOD_LABEL[p], label, str(len(x)), fmt(100 * x["growth"].median(), 1, pct=True, sign=True),
                         fmt(x["r_dcf"].median(), 2), fmt(100 * x["ret_63"].mean(), 1, pct=True, sign=True),
                         fmt(100 * x["ret_126"].mean(), 1, pct=True, sign=True)])
    return html_table(["period", "group", "n", "median growth", "median DCF / price", "next-63 return",
                       "next-126 return"], rows, num_from=2, group=True, cls="small")


def eval_csv(p):
    return pd.read_csv(OUT / f"valuation_eval_{p}.csv")


def t_disagreement():
    """How far apart the methods are, from the evaluation CSVs (method_spread = highest / lowest method value)."""
    rows = []
    for p in PERIODS:
        d = eval_csv(p)
        multi = d[d["method_spread"].round(9) > 1]          # rows with at least two methods
        sp = multi["method_spread"]
        rows.append([PERIOD_LABEL[p], str(len(d)), f"{len(multi) / len(d):.0%}", fmt(sp.median(), 2),
                     f"{(sp > 2).mean():.0%}", f"{(sp > 2.6).mean():.0%}", fmt(sp.quantile(0.9), 1)])
    return html_table(["period", "valued", "with ≥ 2 methods", "median spread", "spread > 2×", "spread > 2.6×",
                       "90th percentile"], rows, num_from=1)


def range_row(label, d, nominal):
    r = d.dropna(subset=["price_252", "low", "high"])
    inside = ((r["price_252"] >= r["low"]) & (r["price_252"] <= r["high"])).mean()
    above, below = (r["price_252"] > r["high"]).mean(), (r["price_252"] < r["low"]).mean()
    width = (d["high"] / d["low"] - 1).median()
    return [label, nominal, str(len(r)), f"{inside:.0%}", f"{above:.0%}", f"{below:.0%}", f"{width:.0%}"]


def t_range():
    rows = [range_row(f"{PERIOD_LABEL[p]}, submitted", eval_csv(p), "P25–P75 multiples, DCF grid") for p in PERIODS]
    for p in ("dev", "val"):
        rows.append(range_row(f"{PERIOD_LABEL[p]}, 10–90% multiple bands",
                              pd.read_csv(WORK / "sens" / "diag" / f"{p}_bands_10_90.csv"), "P10–P90 multiples, DCF grid"))
    return html_table(["sample", "range built from", "n", "price after 252 sessions inside", "above high", "below low",
                       "median width (high / low − 1)"], rows, num_from=2)


def t_seeds():
    """A under PYTHONHASHSEED 0-4 (study/seeds.jsonl); B under seed 0 (ablation) and 1-4 (tuning_seeded.jsonl final_seed*)."""
    a = {(r["period"], r["seed"]): r for r in jsonl(WORK / "study" / "seeds.jsonl")}
    b = {(r["period"], int(r["tag"][-1])): r for r in jsonl(WORK / "tuning_seeded.jsonl") if r["tag"].startswith("final_seed")}
    rows = []
    for p in ("dev", "val"):
        b[(p, 0)] = ablation(p)[1].to_dict()
        for run, d in (("A selection", a), ("B + filter", b)):
            cells = [f"{fmt(d[(p, s)]['total_return_pct'], 2, pct=True)} / {fmt(d[(p, s)]['sharpe'], 3)}" for s in range(5)]
            rows.append([PERIOD_LABEL[p], run, *cells])
    return html_table(["period", "run", *(f"seed {s}: return / Sharpe" for s in range(5))], rows, num_from=2, group=True,
                      cls="small")


TABLES = {"tuning": t_tuning, "constants": t_constants, "eval": t_eval, "ablation": t_ablation, "marginal": t_marginal,
          "ic_by_year": t_ic_by_year, "sensitivity": t_sensitivity, "rules": t_rules, "swaps": t_swaps,
          "spot": t_spot, "lookahead": t_lookahead, "testruns": t_testruns, "seeds": t_seeds,
          "v2_eval": t_v2_eval, "v2_portfolio": t_v2_portfolio, "v2_years": t_v2_years,
          "dcf_quintiles": t_dcf_quintiles, "dcf_groups": t_dcf_groups,
          "disagreement": t_disagreement, "range": t_range}


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=7.5, length=0)


def f_equity(path):
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5), gridspec_kw={"width_ratios": [5, 2, 2.8]})
    for ax, p in zip(axes, PERIODS):
        a, b = equity(f"{p}_A_selection"), equity(f"{p}_B_valuation")
        s = spy(a.index[0], a.index[-1])
        lines = ((s, GRAY, "SPY", 1.2), (a, BLUE, "A selection", 2), (b, ORANGE, "B + valuation", 2))
        for series, color, name, lw in lines:
            ax.plot(series.index, series.values, color=color, linewidth=lw, solid_capstyle="round")
        # End labels, pushed apart so they never overlap (minimum gap = 9% of the panel's y-range).
        lo = min(x.min() for x, *_ in lines)
        hi = max(x.max() for x, *_ in lines)
        gap = 0.09 * (hi - lo)
        ends = sorted(((x.values[-1], name) for x, _, name, _ in lines))
        ys = [ends[0][0]]
        for y, _ in ends[1:]:
            ys.append(max(y, ys[-1] + gap))
        for (y0, name), y in zip(ends, ys):
            ax.annotate(name, (a.index[-1], y), xytext=(4, 0), textcoords="offset points",
                        fontsize=6.5, color=INK2, va="center", annotation_clip=False)
        ax.set_title(PERIOD_LABEL[p], fontsize=8.5, color=INK, loc="left")
        style(ax)
        ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
        ax.margins(x=0.02)
    axes[0].set_ylabel("growth of 1", fontsize=7.5, color=INK2)
    fig.tight_layout(w_pad=3.2)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return "Portfolio value (start = 1) with PYTHONHASHSEED=0, from output/&lt;period&gt;_{A_selection,B_valuation}/equity.csv; SPY from the course price file."


def f_rules(path):
    s = study()
    a = {p: ablation(p)[0].sharpe for p in ("dev", "val")}
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.axhline(0, color=GRAY, linewidth=0.8)
    ax.axvline(0, color=GRAY, linewidth=0.8)
    short = {"committed": "committed", "min1": "≥1 method", "dcf": "DCF", "ev_ebitda": "EV/EBITDA",
             "peer_multiple": "peer P/E", "dcf+ev_ebitda": "DCF+EV", "dcf+peer_multiple": "DCF+peer",
             "ev_ebitda+peer_multiple": "EV+peer"}
    for key, _ in RULES:
        x, y = s[("dev", key)]["sharpe"] - a["dev"], s[("val", key)]["sharpe"] - a["val"]
        c = ORANGE if key == "committed" else BLUE
        ax.scatter([x], [y], s=34, color=c, edgecolor="white", linewidth=1.2, zorder=3)
        # committed sits just right of EV/EBITDA alone: label EV/EBITDA on its left so the two labels never meet
        off, ha = ((-5, 0), "right") if key == "ev_ebitda" else ((5, 0), "left")
        ax.annotate(short[key], (x, y), xytext=off, textcoords="offset points", fontsize=7, color=INK2, ha=ha,
                    va="center")
    ax.set_xlabel("dev: Sharpe B − A", fontsize=7.5, color=INK2)
    ax.set_ylabel("val: Sharpe B − A", fontsize=7.5, color=INK2)
    style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return "Each filter rule's Sharpe minus selection-only Sharpe, dev against val (study/portfolio.jsonl). Every rule is left of zero (worse than A on dev) and above zero (better on val)."


def f_sens(path):
    import shutil
    shutil.copy(WORK / "sens" / "fig_sharpe_vs_A.png", path)
    return "One-at-a-time sensitivity of the valuation constants: Sharpe of B minus A on dev and val (analysis/sens/fig_sharpe_vs_A.png)."


FIGURES = {"equity": f_equity, "rules": f_rules, "sens": f_sens}


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true", help="fail if any placeholder is left")
    ap.add_argument("--no-pdf", action="store_true")
    a = ap.parse_args()
    html = (HERE / "report_src.html").read_text()
    FIGS.mkdir(exist_ok=True)
    for name, fn in TABLES.items():
        marker = f"<!-- TABLE:{name} -->"
        if marker in html:
            html = html.replace(marker, fn())
    for name, fn in FIGURES.items():
        marker = f"<!-- FIG:{name} -->"
        if marker in html:
            path = FIGS / f"{name}.png"
            caption = fn(path)
            cls = ' class="half"' if name == "rules" else ""
            html = html.replace(marker, f'<figure{cls}><img src="figures/{name}.png"><figcaption>{caption}</figcaption></figure>')
    left = re.findall(r"<!-- (TABLE|FIG):(\w+) -->", html)
    if left:
        sys.exit(f"unknown markers: {left}")
    todos = len(re.findall(r'class="todo[ "]', html))   # also "todo yours" (Judgment)
    if a.final and (todos or re.search(r"\b(TODO|TBD|XXX)\b", html)):
        sys.exit(f"--final: {todos} placeholder block(s) left; fill them before building the final PDF")
    (HERE / "report.html").write_text(html)
    print(f"report.html written ({todos} placeholder blocks left)")
    if not a.no_pdf:
        pdf = HERE / ("report_final.pdf" if a.final else "report_draft.pdf")
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={pdf}", (HERE / "report.html").as_uri()],
                       check=True, capture_output=True)
        print(f"{pdf.name} written")


if __name__ == "__main__":
    main()
