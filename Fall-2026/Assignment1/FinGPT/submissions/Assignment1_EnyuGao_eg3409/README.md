FinGPT Forecaster

Enyu Gao(eg3409)

This project fine-tunes Qwen2.5-7B-Instruct, Llama-3.1-8B, and
DeepSeek-R1-Distill-Llama-8B on the FinGPT Dow30 forecasting dataset using 4-bit QLoRA.

## Hardware

- GPU: NVIDIA Tesla T4, 14.56 GiB
- CUDA: 13.0
- OS: Linux x86_64 (Google Colab)
- Python: 3.13.15
- PyTorch: 2.11.0+cu130

## Training

- Dataset: `FinGPT/fingpt-forecaster-dow30-202305-202405`
- Models: Qwen2.5-7B-Instruct, Llama-3.1-8B, and DeepSeek-R1-Distill-Llama-8B
- Baseline LoRA: rank 8, alpha 16, dropout 0.1
- Optimization experiment: Qwen LoRA rank 16
- Quantization: 4-bit
- Epochs: 1
- Learning rate: `5e-5`
- Maximum sequence length: 2,048

Training commands:

```bash
MODEL_KEY=qwen bash train.sh
MODEL_KEY=llama bash train.sh
MODEL_KEY=deepseek bash train.sh
MODEL_KEY=qwen RUN_NAME=qwen_r16 LORA_R=16 bash train.sh
```

The exact Colab setup, smoke tests, checkpoint recovery, training, and evaluation commands are in
[`FinGPT_colab_notebook.ipynb`](./FinGPT_colab_notebook.ipynb).

## Evaluation

All models were evaluated on the complete 300-example test set.

| Model | Valid count | Binary accuracy | MSE | ROUGE-1 | Mean inference time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen rank 8 | 297 | **0.5219** | 8.9024 | 0.4244 | **51.77 s** |
| Llama rank 8 | 296 | 0.4899 | 11.8471 | **0.4367** | 57.85 s |
| DeepSeek rank 8 | 297 | 0.5185 | **8.7744** | 0.4283 | 61.87 s |
| Qwen rank 16 | 297 | 0.5185 | 9.1111 | 0.4266 | 52.23 s |

Raw generations, summary files, metrics, and comparison tables are located in
[`evaluation/`](./evaluation/). Result plots are located in [`figures/`](./figures/), and the 24
completed human reviews are in [`hitl/hitl.json`](./hitl/hitl.json).

## LoRA Adapter

Adapter configurations, manifests, training results, and Trainer states are located in
[`finetuned_models/`](./finetuned_models/):

- `finetuned_models/qwen_baseline/`
- `finetuned_models/llama_baseline/`
- `finetuned_models/deepseek_baseline/`
- `finetuned_models/qwen_r16/`

Adapter configurations, manifests, training results, and Trainer states are included. Adapter
weight files are omitted because their combined size made the GitHub transfer unreliable. Full
base-model weights, model caches, and checkpoint model weights are also excluded.

## Report 
[PDF Report](./FinGPT_report.pdf)

Published URL: https://medium.com/@gaoenyu0801/fine-tuning-llms-for-dow-30-stock-forecasting-f18cffdcf95f
