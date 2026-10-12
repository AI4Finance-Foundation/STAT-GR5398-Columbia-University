from pathlib import Path
import tarfile,subprocess
root=Path(__file__).resolve().parents[1]
if not (root/'upstream').exists():
 with tarfile.open(root/'upstream_snapshot.tar.gz') as archive:archive.extractall(root,filter='data')
 subprocess.run(['patch','-p1','-i',str(root/'improvement.patch')],cwd=root/'upstream',check=True)
 for patchname in ['compatibility.patch','agent_alignment.patch']:
  if (root/patchname).exists():
   subprocess.run(['patch','-p1','-i',str(root/patchname)],cwd=root/'upstream',check=True)
else:
 print('Existing upstream preserved. Confirm improvement.patch has already been applied.')
