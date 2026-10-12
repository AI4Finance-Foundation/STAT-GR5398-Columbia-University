from pathlib import Path
import zipfile
root=Path(__file__).resolve().parent
for name in ["reports_and_audits.zip","code_and_upstream.zip","raw_source_data.zip"]:
    with zipfile.ZipFile(root/name) as z:
        for item in z.infolist():
            target=(root/item.filename).resolve()
            if not target.is_relative_to(root):raise ValueError("Unsafe archive path")
        z.extractall(root)
print("Evidence extracted: actual_runs/, audit/, code/, upstream/, data/ available")
