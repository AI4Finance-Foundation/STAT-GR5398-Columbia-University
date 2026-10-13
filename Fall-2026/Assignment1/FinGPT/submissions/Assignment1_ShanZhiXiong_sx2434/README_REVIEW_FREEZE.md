# FinGPT three model retrospective review materials

All three models completed990 training examples and248 optimizer steps,1,440 primary validation outputs and1,800 authorized final-test outputs. The48 matched HITL outputs remain a separate development experiment. The report is `report/output/pdf/FINGPT_THREE_MODEL_RETROSPECTIVE_FINAL_RESULTS.pdf`.

Claude Opus5.5 High independently accepted the actual run evidence with0 execution defects, reproduced all900 native prompts/1800 outputs/18 metric groups and checked the prior report. This version applies its D1-D3 text corrections using independently verified facts. The auditor explicitly requires no repeat round; the wording changes are Codex checks, not a new independent audit. Final user acceptance remains pending. It is not publicly published or formally submitted. The recurring schedule remains PAUSED. Name/UNI and a Medium public link remain for the user's later instructions.

## Final test on all300 retained labels

| Variant | n | Course valid | Course MSE | +0.5 MSE, same valid subset | Common direction correct / valid | Balanced accuracy | Always-up |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen base | 300 | 208 | 11.2356 | 7.1827 | 161 / 296 | 0.5352 | 137 |
| qwen adapter | 300 | 297 | 8.7475 | 7.6465 | 142 / 300 | 0.5008 | 137 |
| deepseek base | 300 | 0 | N/A | N/A | 130 / 265 | 0.4541 | 137 |
| deepseek adapter | 300 | 297 | 8.5724 | 7.6465 | 148 / 300 | 0.5134 | 137 |
| llama base | 300 | 220 | 9.3727 | 8.3000 | 2 / 4 | 0.0073 | 137 |
| llama adapter | 300 | 297 | 8.5152 | 7.6465 | 150 / 300 | 0.5166 | 137 |

## Previously inspected20

| Variant | n | Course valid | Course MSE | +0.5 MSE, same valid subset | Common direction correct / valid | Balanced accuracy | Always-up |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen base | 20 | 11 | 9.9091 | 7.1818 | 12 / 20 | 0.6061 | 11 |
| qwen adapter | 20 | 19 | 8.8947 | 7.6316 | 12 / 20 | 0.5758 | 11 |
| deepseek base | 20 | 0 | N/A | N/A | 10 / 17 | 0.4747 | 11 |
| deepseek adapter | 20 | 19 | 10.2632 | 7.6316 | 8 / 20 | 0.3838 | 11 |
| llama base | 20 | 12 | 6.4167 | 6.9167 | 0 / 0 | 0.0000 | 11 |
| llama adapter | 20 | 19 | 10.4211 | 7.6316 | 7 / 20 | 0.3283 | 11 |

## Other280 not claimed pristine

| Variant | n | Course valid | Course MSE | +0.5 MSE, same valid subset | Common direction correct / valid | Balanced accuracy | Always-up |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen base | 280 | 197 | 11.3096 | 7.1827 | 149 / 276 | 0.5307 | 126 |
| qwen adapter | 280 | 278 | 8.7374 | 7.6475 | 130 / 280 | 0.4964 | 126 |
| deepseek base | 280 | 0 | N/A | N/A | 120 / 248 | 0.4524 | 126 |
| deepseek adapter | 280 | 278 | 8.4568 | 7.6475 | 140 / 280 | 0.5238 | 126 |
| llama base | 280 | 208 | 9.5433 | 8.3798 | 2 / 4 | 0.0079 | 126 |
| llama adapter | 280 | 278 | 8.3849 | 7.6475 | 143 / 280 | 0.5299 | 126 |

Missing or ambiguous final-line directions count as wrong. The last standalone Prediction line is frozen. DeepSeek native base requires exactly one closed think before final-answer extraction. Course raw parser results, separate extracted sensitivity,9 ROUGE values, EOS/caps and complete denominators remain in `final_test/<model>/*_metrics.json`. The constant+0.5 baseline uses the identical valid subset for each course score. Encoded references and retained labels were not independently reconstructed as realized returns. ROUGE is not grounding.

## Class balance and parser interpretation

Test labels are137 up /163 down, versus the frozen training majority up542/990. Always-up137/300 is the pre-declared comparator. Constant-down163/300 follows from the observed label distribution and exceeds all six variants (best Qwen base161/300); it is a post-hoc label-count fact, not a new or retrospectively selected primary baseline. The three adapters predict up on245/219/207 rows and have all300 balanced accuracies0.5008/0.5134/0.5166. Exceeding always-up is not evidence of forecasting skill; Qwen adapter142/300 is below its base161/300.

Llama BASE contains234/300 Up/Down placeholders. All220 course-valid pairs parse as up, and208 have zero margins; this is template acceptance, not a meaningful prediction. The three rejected references are UNH2024-03-17, KO2024-03-31 and GS2024-04-14 (GS is in the inspected20). These fixed parser failures limit paired course-valid scores to297. All three adapters have300 individually parseable answers and297 accepted pairs. All inputs, labels and raw outputs remain unchanged, with no repair or filtering. Generated-only Llama BASE acceptance223 is distinct from paired course-valid220.

## Sources and reproduction

The exact executed final-test source, PLAN.json, dependency hashes, original human authorization, first-read receipt, derived policy, three scoped read receipts, all900 native prepared prompts, native tokenizer copies, complete1,800 token/boundary receipts, raw outputs and CPU metric recheck are preserved. `final_test/COMPLETED_FILES_SHA256.json` commits every completed remote file. `final_test/LOCAL_VERIFICATION.json` records local re-encoding/decoding and18 reproduced metric groups. These mechanical checks are authored by Codex, not independent Claude acceptance.

`code/train_lora.py`, `train.sh`, `config.json` and `utils.py` are independently reviewed fresh-reproduction wrappers. Their default is a CPU plan. Historical executed sources and selected248 adapters are preserved separately; wrappers did not produce those earlier runs. Validation commands/hardware/full source commitments are preserved in `validation_stage_history/README.md`. All three trained selected adapters are in `selected_adapters/`. Base weights and credentials are excluded.

The historical final-test command was the fixed Python worker with plan SHA `f284b5fb9c06a970990d2557057d3ed0036bf4441c8811eb709479bb5d2c5339`, original authorization SHA `56296cabe16f4b21d93cd81a091399839fb6560fbe95aa081277b6723534d24a` and the original recorded absolute authorization path in `final_report_and_verification_sources/final_test_launch_20261012T184917Z.json`. On the original server:

```sh
/root/shared-nvme/AI4FinRuntime/venvs/fingpt/bin/python -u /root/shared-nvme/AI4FinRuntime/setup/prospective_final_test_v4_20261012/final_test_worker.py --plan-sha f284b5fb9c06a970990d2557057d3ed0036bf4441c8811eb709479bb5d2c5339 --authorization /root/shared-nvme/AI4FinRuntime/setup/final_test_user_authorization_20261012T184917Z.json --authorization-sha 56296cabe16f4b21d93cd81a091399839fb6560fbe95aa081277b6723534d24a
```

That exact output directory now exists; the executor intentionally rejects reruns. A new reproduction requires a new run/output path, a new freeze and fresh authorization before test access. It is not an automatic resume. Local mechanical verification used `verify_completed_test.py --run <collected-run>`; it requires the preserved D: workspace, Transformers4.51.3, sklearn and rouge-score, and a fresh verification destination because evidence is not overwritten. The report builder uses ReportLab and `--out <fresh-report-directory>`; supplied page PNGs complement the PDF.

Training used one RTX4090 24GB, Ubuntu24.04.1, Python3.12.3, torch2.7.0a0+7c8ec84dab.nv25.3, Transformers4.51.3, PEFT0.15.2 and bitsandbytes0.47.0. LoRA r8/alpha16/dropout0.05 on q_proj/v_proj; int8 frozen bases; batch1,acc4,seed42,peak LR2e-4 cosine/warmup0.03,full4096 reference targets plus EOS. Selected248 checkpoints were frozen before test. Llama is the exact Llama-3.1-8B BASE with native BOS and plain prefix, not the Instruct model.

Inference used greedy full6144 context,max new1920,chunk-prefill512,default test batches Qwen16/DeepSeek16/Llama8. Input-length-only fallback was frozen; actual over-envelope rows were zero for all models. Final smaller batches were recorded in raw receipts. Qwen primary validation used batch1; test batch16 changes serving behavior, so this is not a same-serving speed or validation/test causal comparison.

## Human feedback and research limits

Actual human feedback contains18 selected claims plus a quick note and earlier partial review. AI-first triage,24 matched-pair AI judgments and a six-case Llama source review are retained separately. No new human verdict was fabricated. The unchanged-adapter48-output prompt experiment produced mixed direction/grounding results and did not establish stable improvement. Human final acceptance is still pending.

Financial headers in61 train and3 validation rows postdate prediction origin; all1,230 development rows lack complete independent availability evidence. This does not prove all rows leaked, and retrospective disclosure does not clean training data. The original DeepSeek adapter's unclosed reasoning supervision remains a confound. No target repair, new training or claim of a clean forecasting benchmark is made.

Prior independent reviews accepted training/validation/HITL corrections and the prospective test protocol. The actual completed-test audit is also included; its limited scope does not certify clean data, predictive ability, completed HITL or the entire assignment. `CURRENT_SCOPE.json` and later actual-results acceptance records govern current status; old generation-time pending statements in archived validation material are historical.

The local Medium draft is `MEDIUM_DRAFT.md`, unpublished with no public URL. Publication and course submission await separate user instruction.
