"""Summarizes pipeline runs under FinRobot/output into one table (and a CSV).

Usage:  python scripts/summarize_runs.py [--csv notes/run_summary.csv]

For every run folder (output/<TICKER> or output/<TICKER>/<tag>) that has a
report, it pulls the page-1 headline numbers, the valuation engine's own target,
which text sections fell back to canned text, which models wrote the text, and
whether either API key appears anywhere in the logs.
"""
import argparse
import configparser
import csv
import re
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "FinRobot" / "output"
CONFIG = ROOT / "FinRobot" / "finrobot_equity" / "core" / "config" / "config.ini"


class _Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts, self._skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1
        self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip and data.strip():
            self.parts.append(data.strip())


def html_text(path: Path) -> str:
    parser = _Text()
    parser.feed(path.read_text(encoding="utf-8"))
    return re.sub(r"\n\s*\n+", "\n", "".join(parser.parts))


def after(label: str, text: str) -> str:
    """The first non-empty line after a line equal to `label`."""
    m = re.search(rf"^{re.escape(label)}\n(.+)$", text, re.M)
    return m.group(1).strip() if m else ""


def find(pattern: str, text: str) -> str:
    m = re.search(pattern, text)
    return m.group(1) if m else ""


def summarize(run_dir: Path, secrets: list[str]) -> dict:
    step1 = (run_dir / "step1.log").read_text(encoding="utf-8", errors="replace")
    step2_path = next((p for p in (run_dir / "step2_full.log", run_dir / "step2.log") if p.exists()), None)
    step2 = step2_path.read_text(encoding="utf-8", errors="replace") if step2_path else ""
    report = next(run_dir.glob("report/Professional_Equity_Report_*.html"), None)
    text = html_text(report) if report else ""
    logs = step1 + step2

    fallbacks = sorted(set(
        re.findall(r"Warning: (\w[\w ]*?) (?:seems too short|missing required sections), using fallback", step1)
        + re.findall(r"Error generating '(\w+)'", step1)
    ))
    regenerated = re.findall(r"Successfully regenerated '(\w+)'", step2)
    models = sorted(set(re.findall(r"Using model: ([\w.\-]+)", logs)))
    return {
        "run": str(run_dir.relative_to(OUTPUT)).replace("\\", "/"),
        "base_year": find(r"Using (\d{4}A) as base year", step1),
        "price": after("Price", text),
        "page1_rating": after("Rating", text),
        "page1_target": after("Target", text),
        "fmp_grade": find(r"Rating: (.+?) \(auto-fetched\)", step2),
        "engine_ev_ebitda": find(r"EV/EBITDA valuation: (\$[\d,.]+)", step2),
        "engine_dcf": find(r"DCF valuation: (\$[\d,.]+)", step2),
        "engine_target": after("Weighted Target Price", text),
        "engine_up_down": (after("Implied Downside", text) and "-" + after("Implied Downside", text))
                          or after("Implied Upside", text),
        "fallback_sections": ";".join(fallbacks) or "-",
        "step2_regenerated": ";".join(regenerated) or "-",
        "models": ";".join(models),
        "key_in_logs": any(s and s in logs for s in secrets),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()

    cfg = configparser.ConfigParser()
    cfg.read(CONFIG, encoding="utf-8")
    secrets = [cfg.get("API_KEYS", k, fallback="").strip() for k in ("fmp_api_key", "openai_api_key")]

    runs = sorted({p.parent for p in OUTPUT.rglob("step1.log")})
    rows = [summarize(r, secrets) for r in runs]
    if not rows:
        print("No runs found.")
        return

    cols = list(rows[0])
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    for r in rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved {args.csv}")


if __name__ == "__main__":
    main()
