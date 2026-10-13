# FinGPT three model retrospective validation review

This review contains three completed LoRA training runs and 1,440 original-prompt validation outputs. It preserves the original data and reference answers and reports limitations and negative results. It is a validation-stage artifact, not the complete Assignment1 submission. No new final-test access, publication or submission has occurred.

Read `report/output/pdf/FINGPT_THREE_MODEL_RETROSPECTIVE_VALIDATION_20261012.pdf` first. Its six pages cover the hardware, environment, dataset, training, quantitative evaluation and qualitative analysis. `report/REPORT_DATA.json` and `report/METRICS.csv` retain the exact source hashes and denominators. The original R3 v1 results/materials were independently reviewed by Claude Opus5.5 High with no material defect. This v2 preserves all model outputs and human feedback and applies the four noted minor presentation/schema corrections; their exact scope is in MINOR_CORRECTIONS.json. It does not certify final test or the complete assignment.

## Results and limits

Each model trained on 990 earlier-week examples for 248 optimizer steps, selecting checkpoint248 by full240 validation token-weighted cross-entropy. Original Qwen/DeepSeek adapters were retained. The new Llama model is **meta-llama/Llama-3.1-8B BASE**, not Instruct. It uses its native BOS and a plain SYSTEM/input prefix without a chat template.

| Selected adapter | Direction correct /240 | Course-valid pairs | Course MSE |
|---|---:|---:|---:|
| Qwen2.5-7B-Instruct |145|240|7.4292|
| DeepSeek-R1-Distill-Llama-8B |145|240|7.5083|
| Llama-3.1-8B BASE |144|240|7.1375|
| Always-up / fixed +0.5 baseline |145|240|6.3458|

Direction counts use the explicitly named common final-line parser in the report. Historical broader normalization remains separate. MSE compares the course parser's encoded reference prediction, not independently fetched realized returns. Balanced accuracy shows modest descriptive differences; eight dependent validation weeks and checkpoint selection on the same rows do not establish reliable generalization.

Llama BASE produced120 EOS outputs and120 token caps; its adapter produced240 EOS outputs and0 caps. The BASE often repeats empty answer templates or continues the supplied text. Course parser acceptance is not evidence of a meaningful forecast. DeepSeek raw course-valid0 reflects parser incompatibility; its240 generated answers exist. Its original adapter still lacks an explicit closing think boundary.

Financial headers in61 training and3 validation rows postdate the prediction origin. Complete independent availability evidence is missing for all1,230 rows. This does not prove every row leaked. Disclosure and parser changes do not repair data or trained weights.

## Human feedback and AI review

The original human export contains18 selected claim ratings across HON/KO/TRV and one quick-review note. An earlier partial full-answer review is retained separately. AI second opinions did not replace human ratings.

The separate48-output experiment compares fresh original/revised prompts on six previously frozen validation cases, with unchanged Qwen/DeepSeek adapters. Guidance wording was AI-authored and informed by human feedback. Some evidence errors disappear while others persist or appear; neither adapter adopts the requested citation markers. No reliable grounding improvement is established. These48 outputs are excluded from the1,440-output main evaluation count.

The Llama AI assessment reads all six selected adapter outputs and reviews BASE structure/excerpts, with twelve checked claim/source quote anchors. Its scope is not an exhaustive factual review of all BASE continuations. Fresh48 human ratings remain0, and final human review is pending. Missing source benchmarks or portfolio weights are evidence gaps; the user should not be asked to guess them.

## Contents and provenance

- `qwen_validation240/`, `deepseek_validation240/`, `llama_validation240/`: original raw outputs, actual configuration and scores. Variant-specific parser denominators remain explicit.
- `selected_adapters/{qwen,deepseek,llama}/`: the selected adapter weights/configuration, with no base-model weights.
- `code/`: new course-named `train_lora.py`, `train.sh`, `config.json`, `utils.py` and preserved dispatch/source contracts. These wrappers did not produce the reported runs. Their default is a CPU plan, and no new reproduction was executed.
- `llama_executed_training_sources/`, `llama_executed_validation_sources/`, `llama_prepared_data/`: actual sources, frozen plans, native tokenizer bindings and full train/validation targets.
- `llama_completed_training_evidence/`: concise actual run evidence and the original112-file manifest. Intermediate checkpoint/optimizer files are not all included here; that manifest indexes the complete original run retained separately on D: and the server. It is not a claim that this smaller folder has112 members.
- `HITL_MATCHED_EXPERIMENT/`, `HITL_AI_EFFECT_REVIEW/`, `llama_source_review/`: matched48 raw records, corrected AI analysis and the new Llama source assessment.
- `human_original_claim_review/`, `human_earlier_partial_review/`, `ai_second_claim_review/`: separately labelled human and AI contributions.
- `HISTORICAL_TWO_MODEL_STAGE.zip`: the unchanged208-file frozen stage, SHA256 `c6c5a43247ae1d1a5c43bc55d979e68670310979fa399d77cc44be4110934f8c`. Its older README says Llama was deferred; that is archival history, superseded by `CURRENT_SCOPE.json` and this README. Extract to a short D: path or on Linux to avoid Windows path-length problems.
- `independent_audits/`: actual Claude Opus5.5 High reports. Earlier component checks do not certify this whole package or clean forecasting data.

The selected Llama adapter SHA256 is `e4d88dc0562bb72370258cbd2ca80079348d9d645d612283b15a06c7fc373ab9`. Model revisions and adapter/source hashes are recorded in the actual plans and report data. No credentials or private monitor configuration are included.

## Environment and exact commands

Actual execution used one RTX4090 24GB on Ubuntu24.04.1 via SSH from Windows. Training used Python3.12.3, vendor torch2.7.0a0+7c8ec84dab.nv25.3 with CUDA build12.8, transformers4.51.3, PEFT0.15.2, bitsandbytes0.47.0, accelerate1.7.0 and datasets3.6.0. The vendor Torch build is not a generic tested PyPI lock. The historical archive contains the original environment metadata and requirements notes.

All three runs used int8 frozen base weights, LoRA r8/alpha16/dropout0.05 on q/v, batch1, accumulation4, peak LR2e-4, warmup0.03/cosine, seed42 and complete4096-token targets. This is a custom single-GPU Trainer path, not a claimed execution of the starter DeepSpeed example. The first optimizer step has LR0 and the last accumulates two microbatches, as confirmed by the actual training audit.

For a **future fresh reproduction**, the original Linux runtime root must be `/root/shared-nvme/AI4FinRuntime`, with the exact pinned backend sources, prepared data and already verified HF caches installed there. Do not overwrite existing evidence to stage these prerequisites. Llama requires the user's own approved HF access; no token is part of this package. Cache downloads are not performed by these wrappers.

```sh
# CPU plans only; no data/GPU execution:
python code/train_lora.py --model qwen --out /root/shared-nvme/AI4FinRuntime/runs/FRESH_QWEN
python code/train_lora.py --model deepseek --out /root/shared-nvme/AI4FinRuntime/runs/FRESH_DEEPSEEK
python code/train_lora.py --model llama --out /root/shared-nvme/AI4FinRuntime/runs/FRESH_LLAMA

# Deliberate fresh training on the original prepared Linux environment only:
bash code/train.sh llama /root/shared-nvme/AI4FinRuntime/runs/FRESH_LLAMA --execute
```

The parent legacy dispatch serializes Qwen/DeepSeek execution and checks backend/dependency/prepared-data hashes. It does not rehash their base weights or enforce library versions. Llama's child preflight additionally checks weights/versions and reencodes all1,230 targets. These are asymmetric reproduction guarantees, disclosed in `code/CODE_PROVENANCE.json`; no wrapper path has been GPU-executed for this packaging.

The exact **executed** Llama commands were:

```sh
FINGPT_TRAINING_PLAN_SHA=f8c9d2abe365865c0b57fcdcdbaa94336f5d1406aabc0310c580fb8a15475860 /root/shared-nvme/AI4FinRuntime/venvs/fingpt/bin/python -u /root/shared-nvme/AI4FinRuntime/setup/llama_retrospective_training_v2_20261012/llama_full_int8_train.py --model llama --out /root/shared-nvme/AI4FinRuntime/import_20261009/assignment1_20261008/runs/server4090_llama_base_int8_4096_epoch_retrospective_20261012
/root/shared-nvme/AI4FinRuntime/venvs/fingpt/bin/python -u /root/shared-nvme/AI4FinRuntime/setup/llama_retrospective_validation_v1_20261012/llama_validation_worker.py --plan-sha 4409db129bd049e4cca3251182a270fb06895bf2bfa3fa4cf4c6dc656349b572
```

These historical destinations already exist and must not be rerun. Qwen/DeepSeek original commands and source/data staging notes are in the archived `REPRODUCTION_README.md`. Inference uses Qwen batch1, DeepSeek batch16/chunk512 and Llama batch8/chunk512, with context6144 and1920 greedy new tokens. Timing comparisons therefore describe different serving configurations.

`build_report.py` is the actual new local report source and reads the preserved original D: workspace paths shown at its top. On another machine, rebase only a scratch copy of those location constants, preserve pinned source hashes, and use a fresh `--out`. It requires ReportLab; the validation-loss figure uses ReportLab's standard line chart and exports SVG. Supplied PNGs are the rendered six PDF pages; the original report build was visually checked.

## Remaining work

The original300-row final test remains outside authorization. Twenty rows were inspected earlier and must be reported separately from the other280, which are not claimed fully pristine. Before any new read, model/checkpoint/source/metric/budget choices must be frozen and the user must explicitly authorize final-test access. Test outputs must not choose checkpoints, prompts or parameters.

The Medium article is a local draft with no public URL. Identity/UNI, publication and formal submission await the user's later instructions. The recurring schedule remains PAUSED. Current validation completion is not complete Assignment1 readiness.

## Latest independent acceptance

Round12 independently accepted all four minor report corrections and confirmed all main values unchanged. The PDF is preserved byte-for-byte as reviewed: its pending-review footer records its generation-time state. CURRENT_SCOPE.json and the independent audit reports record the later acceptance. Current page renders are qa/page-1..6.png and identical latest-1..6.png; historical renders are retained separately.

The prospective final-test executor is separate from the completed validation materials. E1-E5 revisions are prepared, actual Linux CPU preflight and synthetic integration checks passed, and bounded independent recheck is pending. Read prospective_test/PROTOCOL_REVIEW_NOTE.txt alongside the report: Qwen proposes test batch16 versus primary validation batch1, with0/32 byte-identical probe texts and31/32 identical parsed directions. No original validation data or metrics were replaced. Actual300 test access still requires separate approval after accepted protocol freeze.

The current prospective protocol is V4. The earlier V2 source is retained in prospective_test_history/v2 and is superseded for execution. E1-E5 were independently accepted. V4 adds the recommended input-length-only fallback: a canonical batch containing an over-validation-envelope row is generated individually for every row, identically for base and adapter. Original order and all300 rows are preserved; default batches16/16/8, hard context6144/new tokens1920, no clipping, filtering or post-test tuning. Hard-budget or schema/execution failure can still abort after authorized access; a new protocol and new approval would then be required. Bounded fallback recheck is pending. The real Linux CPU parent remained absent from compute-apps after preflight, and16 synthetic integration tests passed on both platforms; these are not GPU or real-test execution.

## Final prospective acceptance

The bounded R1 fallback recheck independently accepted V4 with zero code defects and no further review required. Its one stale wording sentence has been corrected in the current PROTOCOL_REVIEW_NOTE.txt, with the reviewed original preserved. E1-E5 remain accepted. Actual test authorization, access and GPU execution remain false. GPU fallback/remainder groups and the full real subprocess chain have not been executed; reviewed static/synthetic behavior and actual CPU preflight are distinct from actual test evaluation. The frozen plan and source hashes remain unchanged. The protocol is now prepared for requesting separate real user test authorization.
