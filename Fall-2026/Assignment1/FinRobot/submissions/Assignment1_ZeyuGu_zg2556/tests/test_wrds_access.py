"""Credential-format regression fixtures; never use a real login in tests."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from wrds_access import escape_pgpass, split_pgpass, merge_pgpass, HOST, PORT, DATABASE, classify_error


class PgpassTests(unittest.TestCase):
    def test_escaped_values_round_trip(self):
        fields = [HOST, str(PORT), DATABASE, "fixture:user", "fixture\\value:tail"]
        line = ":".join(escape_pgpass(value) for value in fields)
        self.assertEqual(split_pgpass(line), fields)

    def test_unrelated_entries_retained_and_exact_user_replaced(self):
        other = "# existing comment\r\nother-host:5432:other-db:user:fixture-value\r\n"
        previous = f"{HOST}:{PORT}:{DATABASE}:fixture-user:old-fixture\n"
        other_user = f"{HOST}:{PORT}:{DATABASE}:different-user:retained-fixture\n"
        combined = merge_pgpass(other + previous + other_user, "fixture-user", "replacement-fixture")
        self.assertTrue(combined.startswith(f"{HOST}:{PORT}:{DATABASE}:fixture-user:replacement-fixture\n"))
        self.assertIn(other, combined)
        self.assertIn(other_user, combined)
        self.assertNotIn(previous, combined)

    def test_newline_injection_rejected(self):
        for value in ("line\nline", "line\rline", "nul\x00byte"):
            with self.assertRaises(ValueError):
                escape_pgpass(value)

    def test_error_categories_never_return_credential_fragments(self):
        fixture='fixture-user-with-sensitive-data'
        cases={'PAM authentication failed for user '+fixture:'PAM_AUTHENTICATION_REJECTED_CHECK_LOGIN_MFA_OR_ACCOUNT',
               'connection timed out '+fixture:'CONNECTION_TIMEOUT_NETWORK_OR_MFA',
               'permission denied '+fixture:'DATABASE_PRIVILEGE_DENIED',
               'unknown '+fixture:'CONNECTION_OR_PROBE_FAILED_DETAILS_WITHHELD'}
        for message,expected in cases.items():
            result=classify_error(RuntimeError(message))
            self.assertEqual(result,expected)
            self.assertNotIn(fixture,result)


if __name__ == "__main__":
    unittest.main()
