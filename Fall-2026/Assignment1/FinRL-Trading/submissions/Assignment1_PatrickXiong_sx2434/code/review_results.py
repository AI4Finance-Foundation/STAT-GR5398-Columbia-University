from pathlib import Path
import nbformat
from nbclient import NotebookClient
P=Path(__file__).resolve().parents[1]
f=P/'Review_Results.ipynb';n=nbformat.read(f,as_version=4)
NotebookClient(n,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(P)}}).execute()
nbformat.write(n,f)
print('Review notebook executed successfully')
