"""Local interactive WRDS setup and read-only entitlement probe.

No username/password argument is accepted. Passwords are requested with getpass,
never written into submission artifacts, and never included in exception logs.
Verified credentials may be saved only in the operating system's standard private
PostgreSQL password file. Existing unrelated entries remain byte-for-byte intact.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import getpass
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import warnings

from audit_io import write_json

HOST, PORT, DATABASE = "wrds-pgdata.wharton.upenn.edu", 9737, "wrds"
REFERENCE = "https://github.com/wharton/wrds/blob/main/wrds/sql.py"
MFA_REFERENCE = "https://wrds-www.wharton.upenn.edu/pages/about/log-in-to-wrds-using-two-factor-authentication/multi-factor-authentication-for-sasconnect-and-postgresql/"
SUBMISSION = Path(__file__).resolve().parents[1]


def pgpass_path() -> Path:
    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        if not appdata:
            raise RuntimeError("PRIVATE_STORAGE_UNAVAILABLE")
        return Path(appdata) / "postgresql" / "pgpass.conf"
    return Path.home() / ".pgpass"


def split_pgpass(line: str) -> list[str]:
    """Parse PostgreSQL backslash escapes; never log the returned fields."""
    fields, buffer, escaped = [], [], False
    for char in line.rstrip("\r\n"):
        if escaped:
            buffer.append(char)
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == ":":
            fields.append("".join(buffer))
            buffer = []
        else:
            buffer.append(char)
    if escaped:
        buffer.append("\\")
    fields.append("".join(buffer))
    return fields


def escape_pgpass(value: str) -> str:
    if any(char in value for char in ("\r", "\n", "\x00")):
        raise ValueError("INVALID_CREDENTIAL_FORMAT")
    return value.replace("\\", "\\\\").replace(":", "\\:")


def merge_pgpass(existing: str, username: str, password: str) -> str:
    lines = existing.splitlines(keepends=True)
    new_line = ":".join(escape_pgpass(item) for item in (HOST, str(PORT), DATABASE, username, password)) + "\n"
    retained = []
    for line in lines:
        fields = split_pgpass(line)
        if len(fields) == 5 and fields[:4] == [HOST, str(PORT), DATABASE, username]:
            continue
        retained.append(line)
    # Exact entry first avoids an earlier wildcard line shadowing this login.
    return new_line + "".join(retained)


def private_profile_path() -> Path:
    return pgpass_path().parent / "finrobot_wrds_profile.json"


def restrict_private_file(path: Path) -> None:
    if os.name == "nt":
        # No credentials in command arguments or output. Limit newly written file
        # to the current Windows identity (SID avoids localized account names).
        who = subprocess.run(["whoami", "/user", "/fo", "csv", "/nh"],
                             capture_output=True, text=True, check=True)
        import csv
        sid = next(csv.reader([who.stdout.strip()]))[1]
        subprocess.run(["icacls", str(path), "/inheritance:r", "/grant:r", f"*{sid}:(F)"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    else:
        path.chmod(0o600)


def save_login(username: str, password: str) -> None:
    path = pgpass_path()
    if SUBMISSION == path.resolve() or SUBMISSION in path.resolve().parents:
        raise RuntimeError("REFUSING_CREDENTIAL_IN_SUBMISSION")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Preserve unrelated entries including their newline representation.
    with path.open("r", encoding="utf-8", newline="") if path.exists() else nullcontext(None) as handle:
        existing = handle.read() if handle else ""
    content = merge_pgpass(existing, username, password)
    descriptor, temporary = tempfile.mkstemp(prefix=".finrobot-pgpass-", dir=path.parent)
    temp_path = Path(temporary)
    try:
        os.close(descriptor)
        restrict_private_file(temp_path)
        with temp_path.open("w", encoding="utf-8", newline="") as handle:
            handle.write(content)
        os.replace(temp_path, path)
        profile = private_profile_path()
        # The profile contains only the selected local username, not a password.
        profile.write_text(json.dumps({"host": HOST, "port": PORT, "database": DATABASE, "username": username}), encoding="utf-8")
        restrict_private_file(profile)
    finally:
        temp_path.unlink(missing_ok=True)


def local_username() -> str:
    profile = private_profile_path()
    if profile.exists():
        saved = json.loads(profile.read_text(encoding="utf-8"))
        if (saved.get("host"), saved.get("port"), saved.get("database")) == (HOST, PORT, DATABASE):
            return saved["username"]
    path = pgpass_path()
    if path.exists():
        matches = []
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = split_pgpass(line)
            if len(fields) == 5 and fields[:3] == [HOST, str(PORT), DATABASE] and fields[3] != "*":
                matches.append(fields[3])
        if len(set(matches)) == 1:
            return matches[0]
    raise RuntimeError("LOCAL_LOGIN_NOT_CONFIGURED")


def connect(username: str, password: str | None = None):
    import psycopg2
    arguments = {"host": HOST, "port": PORT, "dbname": DATABASE, "user": username,
                 "sslmode": "require", "connect_timeout": 120, "application_name": "finrobot_zg2556_entitlement_probe",
                 "options": "-c default_transaction_read_only=on -c statement_timeout=20000"}
    if password is not None:
        arguments["password"] = password
    else:
        arguments["passfile"] = str(pgpass_path())
    connection = psycopg2.connect(**arguments)
    connection.autocommit = True
    return connection


def classify_error(exc) -> str:
    """Only fixed categories escape this function; never export any error fragment.

    PAM errors do not distinguish wrong credentials from MFA/account policy, so
    this classification must not assert that the password was wrong.
    """
    text = str(exc).lower()
    code = getattr(exc, 'pgcode', None)
    if isinstance(exc, KeyboardInterrupt):return 'USER_CANCELLED'
    if 'could not translate host name' in text or 'name or service not known' in text:
        return 'DNS_RESOLUTION_FAILED'
    if 'pam authentication failed' in text:return 'PAM_AUTHENTICATION_REJECTED_CHECK_LOGIN_MFA_OR_ACCOUNT'
    if 'ldap authentication failed' in text:return 'LDAP_AUTHENTICATION_REJECTED_CHECK_LOGIN_OR_ACCOUNT'
    if 'password authentication failed' in text or code=='28P01':return 'DATABASE_AUTHENTICATION_REJECTED'
    if 'no pg_hba.conf entry' in text:return 'SERVER_ACCESS_POLICY_REJECTED'
    if 'permission denied' in text or code=='42501':return 'DATABASE_PRIVILEGE_DENIED'
    if 'timeout expired' in text or 'timed out' in text:return 'CONNECTION_TIMEOUT_NETWORK_OR_MFA'
    if 'connection refused' in text:return 'TCP_CONNECTION_REFUSED'
    if 'ssl' in text or 'certificate' in text:return 'TLS_CONNECTION_FAILURE'
    if 'server closed the connection' in text:return 'SERVER_CLOSED_CONNECTION'
    if 'unrecognized configuration parameter' in text:return 'SESSION_OPTION_REJECTED'
    return 'CONNECTION_OR_PROBE_FAILED_DETAILS_WITHHELD'


def probe(connection) -> dict:
    from psycopg2 import sql
    result = {"schema_inventory": [], "tables": [], "download_scope": "catalog metadata and LIMIT 1 access check only; no financial dataset downloaded"}
    with connection.cursor() as cursor:
        cursor.execute("SELECT nspname FROM pg_catalog.pg_namespace WHERE (nspname LIKE 'comp%%' OR nspname LIKE 'crsp%%') AND has_schema_privilege(oid, 'USAGE') ORDER BY nspname")
        result["schema_inventory"] = [row[0] for row in cursor.fetchall()]
    candidates = [("comp", "funda"), ("comp", "security"), ("comp", "company"),
                  ("comp_na_daily_all", "funda"), ("comp_na_daily_all", "security"),
                  ("crsp", "dsf"), ("crsp", "stocknames"), ("crsp", "msf"),
                  ("crsp_a_stock", "dsf"), ("crsp_a_stock", "stocknames"),
                  ("crsp_a_ccm", "ccmxpf_linktable")]
    for schema, table in candidates:
        entry = {"schema": schema, "table": table, "selectable": False, "columns": []}
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT column_name,data_type FROM information_schema.columns WHERE table_schema=%s AND table_name=%s ORDER BY ordinal_position", (schema, table))
                entry["columns"] = [{"name": row[0], "type": row[1]} for row in cursor.fetchall()]
                if not entry["columns"]:
                    entry["status"] = "not_visible_or_absent"
                else:
                    cursor.execute(sql.SQL("SELECT 1 FROM {}.{} LIMIT 1").format(sql.Identifier(schema), sql.Identifier(table)))
                    cursor.fetchone()
                    entry.update(selectable=True, status="select_verified")
        except Exception as exc:
            entry.update(status="select_failed", error_type=type(exc).__name__, sqlstate=getattr(exc, "pgcode", None))
        result["tables"].append(entry)
    result["mapping_next_step"] = "If Compustat funda/security is selectable, inspect observed ticker/GVKEY, fiscal-date and statement columns before mapping AVGO/QCOM; no fabricated FMP-equivalent fields."
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("setup", "probe"))
    args = parser.parse_args()
    output = SUBMISSION / "evidence" / "wrds"
    started = datetime.now(timezone.utc).isoformat()
    result = {"started_at_utc": started, "action": args.action, "host": HOST, "port": PORT,
              "database": DATABASE, "sslmode": "require", "read_only": True, "official_reference": REFERENCE,
              "status": "not_connected", "credentials_in_artifact": False,
              "connection_timeout_seconds":120,"mfa_reference":MFA_REFERENCE}
    connection = None
    try:
        if args.action == "setup":
            if not sys.stdin.isatty():
                raise RuntimeError("INTERACTIVE_LOCAL_TERMINAL_REQUIRED")
            print("Enter WRDS credentials only in this local terminal. Both entries are hidden.")
            with warnings.catch_warnings():
                warnings.simplefilter("error", getpass.GetPassWarning)
                username = getpass.getpass("WRDS username: ").strip()
                password = getpass.getpass("WRDS password: ")
            if not username or not password:
                raise RuntimeError("EMPTY_LOGIN")
            result['failed_stage'] = 'encrypted_connection_and_authentication'
            print('Connecting once. If WRDS sends a Duo Push, approve it on your own device. Waiting up to 120 seconds.')
            connection = connect(username, password)
            print("Encrypted WRDS connection verified.")
            choice = input("Save verified login in your private PostgreSQL password file for local reuse? [Y/n] ").strip().lower()
            if choice in ("", "y", "yes"):
                save_login(username, password)
                result["private_login_saved"] = True
                print("Verified login saved outside the submission; existing unrelated database entries preserved.")
            else:
                result["private_login_saved"] = False
            password = None
        else:
            connection = connect(local_username())
        result['failed_stage'] = 'read_only_catalog_probe'
        result.update(probe(connection), status="connected")
        result.pop('failed_stage',None)
        print("Read-only entitlement probe complete. Only sanitized availability metadata was saved.")
    except (Exception, KeyboardInterrupt) as exc:
        known = {"PRIVATE_STORAGE_UNAVAILABLE", "REFUSING_CREDENTIAL_IN_SUBMISSION", "LOCAL_LOGIN_NOT_CONFIGURED",
                 "INTERACTIVE_LOCAL_TERMINAL_REQUIRED", "EMPTY_LOGIN", "INVALID_CREDENTIAL_FORMAT"}
        result.update(status="failed", error_type=type(exc).__name__, sqlstate=getattr(exc, "pgcode", None),
                      reason=str(exc) if str(exc) in known else classify_error(exc))
        print('WRDS sanitized failure category: '+result['reason'])
        print("No credentials or raw server message were saved. There is no automatic login retry. See evidence/wrds/access_latest.json.")
    finally:
        if connection is not None:
            connection.close()
        result["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "access_latest.json", result)
        write_json(output / ("access_" + started.replace(":", "").replace("+", "_") + ".json"), result)
    if result["status"] != "connected":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
