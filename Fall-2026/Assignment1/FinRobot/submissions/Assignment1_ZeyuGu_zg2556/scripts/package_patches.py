"""Export separate compatibility and Option A patches; optionally prove clean apply.

Never changes the course checkout index or branch. Only the explicitly supplied
isolated upstream checkout gets an intent-to-add entry for the new verifier.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from audit_io import sha256, utc_now, write_json

SHA = "2717499b8e30f242640af08c4ad9afd1113c2d45"
VERIFIER = "finrobot_equity/core/src/modules/equity_agents/verifier_agent.py"


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.STDOUT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--clean-checkout", type=Path)
    parser.add_argument("--output-directory", type=Path,
        help="Optional separate development patch directory; canonical patches are left intact")
    args = parser.parse_args()
    submission = Path(__file__).resolve().parents[1]
    root = args.upstream.resolve()
    if root == submission or submission in root.parents:
        raise ValueError("Upstream clone cannot be inside the submission")
    if git(root, "rev-parse", "HEAD").decode().strip() != SHA:
        raise ValueError("Wrong upstream commit")
    git(root, "add", "--intent-to-add", "--", VERIFIER)
    changed = git(root, "diff", "--name-only", "HEAD").decode().splitlines()
    if any(not name.startswith("finrobot_equity/core/src/") for name in changed):
        raise ValueError("Unexpected changes outside equity core src")
    patches = args.output_directory.resolve() if args.output_directory else submission/"patches"
    patches.mkdir(parents=True,exist_ok=True)
    compatibility = [name for name in changed if name != VERIFIER]
    (patches/"compatibility.patch").write_bytes(git(root, "diff", "--binary", "HEAD", "--", *compatibility))
    (patches/"option_a_verifier.patch").write_bytes(git(root, "diff", "--binary", "HEAD", "--", VERIFIER))
    manifest = {"upstream_url":"https://github.com/AI4Finance-Foundation/FinRobot.git", "upstream_sha":SHA,
        "exported_at_utc":utc_now(), "changed_paths":changed,
        "patch_sha256": {path.name:sha256(path.read_bytes()) for path in patches.glob("*.patch")},
        "source_sha256": {name:sha256((root/name).read_bytes()) for name in changed}, "clean_apply": "not_tested"}
    if args.clean_checkout:
        clean = args.clean_checkout.resolve()
        if clean.exists():
            raise ValueError("Clean verification path must not already exist")
        # A clone copies only committed objects, never source working-tree edits.
        subprocess.run(["git", "clone", "--no-hardlinks", "--no-checkout", str(root), str(clean)], check=True,
                       capture_output=True)
        git(clean, "checkout", "--detach", SHA)
        for name in ("compatibility.patch", "option_a_verifier.patch"):
            git(clean, "apply", "--check", str(patches/name))
            git(clean, "apply", str(patches/name))
        line_ending_differences = []
        for name in changed:
            produced, applied = (root/name).read_bytes(), (clean/name).read_bytes()
            if produced.replace(b'\r\n', b'\n') != applied.replace(b'\r\n', b'\n'):
                raise ValueError("Applied source differs: " + name)
            if produced != applied:
                line_ending_differences.append(name)
        env = dict(os.environ, FINROBOT_UPSTREAM=str(clean),
                   FINROBOT_UPSTREAM_SRC=str(clean/"finrobot_equity/core/src"), PYTHONIOENCODING="utf-8")
        test = subprocess.run([sys.executable, "-m", "pytest", "-q", str(submission/"tests")], env=env,
            capture_output=True, text=True, encoding="utf-8")
        (patches/"clean_apply_tests.log").write_text(test.stdout+test.stderr, encoding="utf-8")
        manifest.update(clean_apply="passed" if test.returncode == 0 else "failed_tests",
            clean_checkout=str(clean), offline_test_returncode=test.returncode,
            line_ending_only_differences=line_ending_differences,
            verification_scope="Patch applies to clean pinned checkout; source content identical after CRLF/LF normalization; offline regression tests only, not a replacement for live financial experiments.")
        write_json(patches/"manifest.json", manifest)
        if test.returncode:
            raise RuntimeError("Applied checkout tests failed; see clean_apply_tests.log")
    write_json(patches/"manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
