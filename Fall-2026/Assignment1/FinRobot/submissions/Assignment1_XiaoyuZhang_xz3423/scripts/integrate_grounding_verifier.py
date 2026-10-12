from pathlib import Path
import ast
import shutil

path = Path("finrobot_equity/core/src/generate_financial_analysis.py")
source = path.read_text(encoding="utf-8")
tree = ast.parse(source)
lines = source.splitlines(keepends=True)

flag = "--verify-generated-text"
marker = "# BEGIN FINANCIAL VERIFIER INTEGRATION"

if flag in source or marker in source:
    raise SystemExit(
        "Integration appears to exist already. No changes made."
    )

# Find the existing command-line argument declaration.
flag_anchor = next(
    (i for i, line in enumerate(lines)
     if 'parser.add_argument("--generate-text-sections"' in line),
    None
)
if flag_anchor is None:
    raise RuntimeError("Could not locate CLI argument anchor.")

# Find the exact assignment to generate_text_section(...)
targets = []
for node in ast.walk(tree):
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        value = node.value
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "generate_text_section"
        ):
            targets.append(node)

if len(targets) != 1:
    raise RuntimeError(
        f"Expected exactly one text-generation assignment; "
        f"found {len(targets)}. No changes made."
    )

node = targets[0]
indent = " " * node.col_offset

integration = f'''
{indent}{marker}
{indent}if args.verify_generated_text:
{indent}    import json as _grounding_json
{indent}    from pathlib import Path as _GroundingPath
{indent}    try:
{indent}        from financial_grounding_verifier import verify as _verify_grounding
{indent}        _audit_dir = _GroundingPath(
{indent}            args.output_dir or
{indent}            f"output/{{args.company_ticker}}/analysis"
{indent}        )
{indent}        _audit_csv = _audit_dir / args.output_csv_name
{indent}        _audit_output_dir = _GroundingPath(
{indent}            args.text_output_dir or str(_audit_dir)
{indent}        )
{indent}        _audit_output_dir.mkdir(parents=True, exist_ok=True)
{indent}        _audit_path = (
{indent}            _audit_output_dir /
{indent}            f"{{text_type}}_grounding_verification.json"
{indent}        )
{indent}        _audit_result = _verify_grounding(
{indent}            generated_text, str(_audit_csv)
{indent}        )
{indent}        _audit_path.write_text(
{indent}            _grounding_json.dumps(
{indent}                _audit_result, indent=2, default=str
{indent}            ),
{indent}            encoding="utf-8"
{indent}        )
{indent}        print(f"Grounding verification saved: {{_audit_path}}")
{indent}    except Exception as _grounding_error:
{indent}        print(
{indent}            f"WARNING: grounding verification failed "
{indent}            f"for {{text_type}}: {{_grounding_error}}"
{indent}        )
{indent}# END FINANCIAL VERIFIER INTEGRATION
'''

# Insert after the complete call expression and CLI flag line.
lines.insert(node.end_lineno, integration)
lines.insert(
    flag_anchor + 1,
    '    parser.add_argument("--verify-generated-text", '
    'action="store_true", '
    'help="Save financial grounding verification JSON for AI text sections.")\n'
)

updated = "".join(lines)
compile(updated, str(path), "exec")

backup = path.with_suffix(".py.before_verifier_integration")
shutil.copy2(path, backup)
path.write_text(updated, encoding="utf-8")

print("PASS - Original script backed up")
print("PASS - CLI verification flag added")
print("PASS - Post-generation verifier inserted")
print("PASS - Python syntax validated")
print("Backup:", backup)
print("VERIFIER INTEGRATION CREATED")
