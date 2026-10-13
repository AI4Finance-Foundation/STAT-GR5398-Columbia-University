from pathlib import Path
import os,sys,json
import nbformat
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parent
# A dedicated kernel spec under this workspace, not Positron or a global user kernel.
spec=ROOT.parents[1]/'work/fingpt/jupyter/kernels/fingpt-isolated'
spec.mkdir(parents=True,exist_ok=True)
(spec/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'FinGPT isolated experiment','language':'python'}))
os.environ['JUPYTER_PATH']=str(spec.parents[1])
os.environ['IPYTHONDIR']=str(spec.parents[2]/'ipython')
os.environ['JUPYTER_RUNTIME_DIR']=str(spec.parents[2]/'jupyter_runtime')
os.environ['JUPYTER_CONFIG_DIR']=str(spec.parents[2]/'jupyter_config')
path=ROOT/(sys.argv[1] if len(sys.argv)>1 else 'FinGPT_Assignment1_Review.ipynb')
nb=nbformat.read(path,as_version=4)
NotebookClient(nb,timeout=180,kernel_name='fingpt-isolated',resources={'metadata':{'path':str(ROOT)}}).execute()
nbformat.validate(nb)
nbformat.write(nb,path)
code=[c for c in nb.cells if c.cell_type=='code']
assert all(c.execution_count for c in code)
assert not [o for c in code for o in c.outputs if o.output_type=='error']
print('Notebook executed cleanly in fresh isolated kernel:',len(code),'code cells')
