# Verified execution commands

Executed in an isolated Colab T4 workspace. Python/package versions and hardware: cloud_runs/llama31_8b/results/cloud_environment.json. Dataset and model revisions: provenance.json and README.md.

```sh
python prepare_data.py
python train_lora.py --model "$MODEL_SNAPSHOT" --arch llama3 --run_name llama31_8b --quant8 --offline --max_length 2048
python evaluate.py --model "$MODEL_SNAPSHOT" --batch_size 16 --max_new_tokens 512
python benchmark_inference.py --model "$MODEL_SNAPSHOT"
```

The historical --quant8 option actually selects 4-bit NF4; this adaptation is disclosed. Training: 60 rows, validation: 30 rows, test: all 300 rows per variant. The actual adapter was independently checked and metrics recomputed from all 600 raw outputs. Notebook: 4 executed cells, no errors. Four-page PDF rendered and checked. Medium URL is in README. Second-model comparison is incomplete. The 80MiB adapter weights are retained locally and omitted from the public submission; see ADAPTER_WEIGHTS.md before attempting checkpoint-dependent reruns.
