# FinGPT Forecaster with Llama-3.1-8B: formatting gains are not forecasting proof

Unpublished AI-assisted draft. The student must review and edit before publication. The required second-model comparison remains incomplete by student decision.

The experiment uses the supplied FinGPT Dow30 data (1230training,300test rows), a fixed60-row training subset and30time-separated validation rows. Llama-3.1-8B was adapted on a Tesla T4 using NF4 QLoRA, rank8,alpha16,one epoch and masked prompt loss. Training and validation losses were1.7047/1.6787. Inputs and teacher training answers were shortened for memory; original evaluation references remained unchanged.

Both base and adapter generated answers for all300test rows with greedy512-token decoding. Valid paired formatting rose from206/300(68.67%) to250/300(83.33%). Conditional direction accuracy against teacher references was46.60% and49.20%; signed-bin midpoint MSE was7.1383 and9.7920. These conditional values use different valid subsets and do not establish a statistically significant accuracy improvement. Against provided labels, counting invalid answers as wrong, accuracy was32.00% and41.33%, below the fixed train-majority always-up baseline45.67%.

ROUGE F1 improved across the three explanation sections, but lexical teacher overlap is not evidence of factual grounding or profitable trading. On six fixed single-request cases, mean synchronized inference time was26.19s(base) and55.32s(adapter), excluding model loading and warmup. This is a small observational timing sample; differing response lengths matter.

The dataset audit removed obvious future-dated financial blocks from64training and2test inputs. Filing availability, original publication timestamps and historical company-profile snapshots could not be established. The upstream teacher pipeline conditions explanations on realized future labels; exact released-row generation provenance and pretraining overlap cannot be excluded. Results therefore describe retrospective imitation and formatting, not unbiased prospective forecasting.

DeepSeek training was completed earlier, but full evaluation was cancelled after the student chose Llama only. The second-model requirement is explicitly unfinished; no instructor waiver is presumed. Human qualitative ratings remain for the student to provide. No publication or assignment submission has occurred.

Sources: [course FinGPT assignment](https://github.com/AI4Finance-Foundation/STAT-GR5398-Columbia-University/tree/master/Fall-2026/Assignment1/FinGPT), [official Forecaster](https://github.com/AI4Finance-Foundation/FinGPT/tree/master/fingpt/FinGPT_Forecaster), [dataset](https://huggingface.co/datasets/FinGPT/fingpt-forecaster-dow30-202305-202405), [Llama model](https://huggingface.co/meta-llama/Llama-3.1-8B).

Disclosure: Codex assisted with source checking, code changes, execution, validation and drafting. The student must comply with any additional course AI policy, perform their own qualitative review and supply the actual Medium publication URL.
