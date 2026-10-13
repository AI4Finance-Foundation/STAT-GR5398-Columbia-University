from pathlib import Path
import os,json
from huggingface_hub import snapshot_download
r=Path(__file__).parent
p=snapshot_download('meta-llama/Llama-3.1-8B',revision='d04e592bb4f6aa9cfee91e2e20afa771667e1d4b',token=os.environ.get('HF_TOKEN'),max_workers=2,allow_patterns=['*.json','*.safetensors','tokenizer.model','LICENSE*','USE_POLICY.md','README.md'])
(r/'results/weights_ready.json').write_text(json.dumps({'status':'complete','snapshot':p}))
print('Weights ready',flush=True)
