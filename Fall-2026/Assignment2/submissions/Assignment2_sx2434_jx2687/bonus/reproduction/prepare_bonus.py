"""Create a stand-alone, dev/val-only working copy for the two selection runs."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import sqlite3
from tempfile import TemporaryDirectory

import pyarrow.compute as pc
import pyarrow.parquet as pq


END = "2023-12-29"
# The December 2023 model scores at the November 30 close. In growth.json's
# mixed-alpha mode its inference quarter is June 30 and the course's maximum
# eligible training/inner-validation quarter is March 31 (selector.py).
LAST_DECISION = "2023-11-30"
LAST_LABELED_QUARTER = "2023-03-31"


def copy_market_data(source, destination):
    paths = [source / "prices" / name for name in ("daily_adjusted.parquet", "market_series.parquet")]
    paths.extend(sorted((source / "macro").glob("*.parquet")))
    for path in paths:
        table = pq.read_table(path, filters=[("date", "<=", datetime.fromisoformat(END))])
        if table.num_rows == 0 or pc.max(table["date"]).as_py().date().isoformat() > END:
            raise ValueError(f"Unexpected date coverage in {path.name}")
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, target)


def copy_database(source, destination):
    # Keep original rowids: the course resolves otherwise tied filing versions
    # by descending rowid. The fallback date is the course quarter's trade date
    # for records without a filing or acceptance timestamp.
    visible = """
        datadate <= :decision AND
        COALESCE(NULLIF(accepted_date, ''), NULLIF(filing_date, ''), CASE
            WHEN substr(datadate, 6, 5) IN ('03-31', '06-30', '09-30', '12-31')
            THEN date(datadate, 'start of month', '+3 months') END) <= :decision
    """
    predicates = {"fundamental_data": visible, "universe_membership": "snapshot_date <= :end",
                  "symbol_changes": "change_date <= :end"}
    bounds = {"decision": LAST_DECISION, "label": LAST_LABELED_QUARTER, "end": END}
    with sqlite3.connect(destination.resolve().as_uri(), uri=True) as conn:
        conn.execute("ATTACH DATABASE ? AS original", (source.resolve().as_uri() + "?mode=ro",))
        for name, condition in predicates.items():
            schema = conn.execute(
                "SELECT sql FROM original.sqlite_master WHERE type='table' AND name=?", (name,)
            ).fetchone()
            if schema is None:
                raise ValueError(f"Missing course table {name}")
            conn.execute(schema[0])
            columns = [row[1] for row in conn.execute(f"PRAGMA original.table_info({name})")]
            if name == "fundamental_data" and "y_return" not in columns:
                raise ValueError("Missing course training label")
            fields = ['"' + c.replace('"', '""') + '"' for c in columns]
            selected = [("CASE WHEN datadate <= :label THEN y_return END" if c == "y_return" else f)
                        for c, f in zip(columns, fields)]
            conn.execute(f"INSERT INTO main.{name} (rowid, {', '.join(fields)}) "
                         f"SELECT rowid, {', '.join(selected)} FROM original.{name} WHERE {condition}", bounds)
        for (sql,) in conn.execute("SELECT sql FROM original.sqlite_master WHERE type='index' "
                                   "AND tbl_name='fundamental_data' AND sql IS NOT NULL"):
            conn.execute(sql)
        if conn.execute("SELECT COUNT(*) FROM main.fundamental_data").fetchone()[0] == 0:
            raise ValueError("No historical filings found")


def prepare(source, destination):
    if not source.is_dir() or destination.exists() or destination.is_relative_to(source):
        raise ValueError("Supply an existing course source_code and a new destination outside it")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Publish the directory only after all bounded extracts and config edits succeed.
    with TemporaryDirectory(prefix=".prepare_bonus_", dir=destination.parent) as temp:
        work = Path(temp) / "work"
        work.mkdir()
        for name in ("selection", "portfolio", "shared", "timing"):
            (work / name).mkdir()
            for path in (source / name).glob("*.py"):
                shutil.copy2(path, work / name / path.name)
        for filename in ("run_portfolio.py", "requirements.txt"):
            shutil.copy2(source / filename, work / filename)
        configs = work / "selection" / "configs"
        configs.mkdir()
        original = source / "selection" / "configs" / "growth.json"
        shutil.copy2(original, configs / "growth.json")
        candidate = json.loads(original.read_text())
        if candidate["model"]["val_quarters"] != 3 or candidate["model"]["inference_mode"] != "mixed-alpha":
            raise ValueError("Expected the original course growth.json")
        candidate["model"]["val_quarters"] = 4
        (configs / "growth_q4.json").write_text(json.dumps(candidate, indent=2) + "\n")
        data = work / "data"
        data.mkdir()
        copy_market_data(source / "data", data)
        copy_database(source / "data" / "fundamentals.db", data / "fundamentals.db")
        work.rename(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_code", type=Path)
    parser.add_argument("new_destination", type=Path)
    args = parser.parse_args()
    prepare(args.source_code.resolve(), args.new_destination.resolve())
    print(f"Prepared {args.new_destination.resolve()} through {END}.")
