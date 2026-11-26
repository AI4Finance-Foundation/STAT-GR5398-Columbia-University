"""Export dated course data and preserve row IDs for point-in-time replication."""
from datetime import datetime
import sqlite3

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


def copy_database(source, destination, decision=LAST_DECISION):
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
    bounds = {"decision": decision, "label": LAST_LABELED_QUARTER, "end": END}
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
