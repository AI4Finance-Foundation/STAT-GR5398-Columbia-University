# STAT GR5398 Assignment1 — Llama-3.1-8B

Llama training and all300base plus300LoRA evaluations are completed and independently validated. The student explicitly chose Llama only. **The second-model requirement is NOT COMPLETED; no instructor waiver is assumed.** This is prepared work, not a submitted assignment. Student: JIAO XU; UNI: jx2687. The student reports reviewing the work; numeric HITL ratings remain blank and additional AI policy is unverified. Medium publication is complete; course submission has not occurred.

## Inspect

- Research_Report.pdf: final actual Llama report, four pages, rendered and visually checked.
- FinGPT_Assignment1_GPU_Review.ipynb: four code cells executed in a fresh isolated local kernel; no error outputs.
- results/gpu_metrics.json and gpu_metrics.csv: final metric tables and all-case direction accuracy.
- cloud_runs/llama31_8b/results/: training.json, base_answers.jsonl and lora_answers.jsonl (300 each), metrics.json, full_validation.json, single_request_benchmark.json/.jsonl, environment and cache manifests.
- cloud_runs/llama31_8b/finetuned_models/llama31_8b/: adapter configuration and validation evidence. The 80MiB weight file is retained locally, omitted from this GitHub submission due to size; original dense base model is not included.
- HITL_review_Llama.csv: six predeclared paired cases; human ratings are blank for the student to fill.
- REQUIREMENTS_CHECKLIST.md: original requirements, completed work, omissions and course-policy uncertainty.
- Medium_Reviewed.md: published AI-assisted article, reviewed by the student. Medium URL: https://medium.com/@xujiao0224/fingpt-forecaster-with-llama-3-1-8b-formatting-gains-are-not-forecasting-proof-df1b7fa0e030
- AI_DISCLOSURE.md and sources/: disclosure and exact course sources.

## Environment and data

GPU execution: Google Colab Tesla T4,15360MiB GPU RAM,~12.67GB system RAM; Python3.13.15, torch2.11.0+cu130, transformers4.57.6, peft0.18.1, datasets4.8.5, bitsandbytes0.50.2. Exact versions in cloud_runs/llama31_8b/results/cloud_environment.json. Repeated runtime resets delayed progress; persistent cache/results were finally saved to a dedicated Drive folder. No FinRL or Positron environment was modified.

Dataset: FinGPT/fingpt-forecaster-dow30-202305-202405, revision4b234310115a6f2cb5806dcd95f801b60d177882, original1230train/300test. Fixed subset60train/30validation, all300test. Raw parquet hashes and source provenance: provenance.json and run provenance.json. Model: meta-llama/Llama-3.1-8B, revisiond04e592bb4f6aa9cfee91e2e20afa771667e1d4b.

## Actual GPU commands

Executed within the independent llama31_8b workspace (the MODEL_SNAPSHOT points to the pinned authorized cache):

```sh
python prepare_data.py
python train_lora.py --model "$MODEL_SNAPSHOT" --arch llama3 --run_name llama31_8b --quant8 --offline --max_length 2048
python evaluate.py --model "$MODEL_SNAPSHOT" --batch_size 16 --max_new_tokens 512
python benchmark_inference.py --model "$MODEL_SNAPSHOT"
```

Historical --quant8 flag selects4-bit NF4 in this modified script; the saved training manifest records this explicitly. LoRA rank8/alpha16/dropout0.1, batch1,accum4,lr5e-5,one epoch,seed5398. Prompt budget896 and shortened teacher training answers; evaluation references are unchanged. Inference starts at batch16 and automatically falls back to8 for adapter inference on T4. No DeepSpeed; TensorBoard used. These adaptations are not exact compliance with the original8-bit/DeepSpeed recipe.

Local final validation/build commands from the workspace root:

```sh
work/fingpt/venv/bin/python outputs/fingpt/cloud/validate_cloud_run.py outputs/fingpt/cloud_runs/llama31_8b
work/fingpt/venv/bin/python outputs/fingpt/build_gpu_deliverables.py
work/fingpt/venv/bin/python outputs/fingpt/execute_notebook.py FinGPT_Assignment1_GPU_Review.ipynb
```

The cloud runs were resumed without retraining after interruptions. Preserve old raw records; do not run all historical Colab cells. For a clean rerun use the documented run scripts in a fresh separate directory. Hugging Face credentials must remain in Colab Secret and are not part of this package.

## Interpretation and submission

LoRA improves paired formatting coverage68.67%→83.33%, conditional teacher-direction accuracy46.60%→49.20%; midpoint MSE worsens7.1383→9.7920. All-case provided-label direction accuracy32.00%→41.33%, below the fixed always-up baseline45.67%. Lexical ROUGE overlap is not factual grounding. Missing point-in-time timestamps, hindsight-conditioned teacher explanations, pretraining overlap and small training subset prevent unbiased forecasting or profitability claims.

The local Qwen0.5B experiment is auxiliary (Local_Qwen_Research_Report.pdf); it is not the final Llama result and is excluded from the prepared Llama package. DeepSeek partial work is retained outside that package, with its full comparison requirement marked incomplete.

Formal submission directory: Fall-2026/Assignment1/FinGPT/submissions/Assignment1_JiaoXu_jx2687. Submitted work is AI-assisted and reviewed by the student. Numeric HITL ratings are unfilled; no full-compliance claim is made. The repository permits multiple Assignment1 tracks in separate folders; this does not establish personal degree-credit eligibility.

## Published blog

[https://medium.com/@xujiao0224/fingpt-forecaster-with-llama-3-1-8b-formatting-gains-are-not-forecasting-proof-df1b7fa0e030](https://medium.com/@xujiao0224/fingpt-forecaster-with-llama-3-1-8b-formatting-gains-are-not-forecasting-proof-df1b7fa0e030)

Publication verified live in Medium. The PDF has been updated to reflect publication. The blog preserves AI assistance disclosure and the incomplete second-model requirement.
