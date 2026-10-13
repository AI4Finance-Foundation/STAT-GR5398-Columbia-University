# Assignment 1 FinGPT 原文核对与状态

核对日期：2026-10-11（America/New_York）。课程 Git commit：`8c38d8f0125e6b5c4b08db140046f3396e2020b0`。本表只把原文明确事项列为作业要求；不把学期研究目标加成作业硬性要求。状态于2026-10-12更新；Llama指定8B实验已完成；第二模型完整比较未完成。最终PDF和GPU_Review Notebook现为已执行的Llama8B成果；旧0.5B报告单独保留为辅助材料。

## 原文来源

- [Assignment 1 总说明](https://github.com/AI4Finance-Foundation/STAT-GR5398-Columbia-University/blob/8c38d8f0125e6b5c4b08db140046f3396e2020b0/Fall-2026/Assignment1/README.md)：track 数量、提交规则、日期。
- [FinGPT 作业原文](https://github.com/AI4Finance-Foundation/STAT-GR5398-Columbia-University/blob/8c38d8f0125e6b5c4b08db140046f3396e2020b0/Fall-2026/Assignment1/FinGPT/README.md)：Targets、3.1–3.3、4、5。
- [课程总览](https://github.com/AI4Finance-Foundation/STAT-GR5398-Columbia-University/blob/8c38d8f0125e6b5c4b08db140046f3396e2020b0/README.md)：学期方向目标；该文件标题写 Spring 2026，不应覆盖 Fall-2026 的具体作业指令。
- 原始文件保存在 `sources/`；内容哈希见 `provenance.json`。

## 当前范围变更（2026-10-12）

用户明确要求仅完成Llama。DeepSeek后续队列已取消；第二个模型的完整评估/比较要求未完成，不声称获教师豁免。以下已有DeepSeek训练材料仅作为保留的部分成果。最终报告只包含Llama实际完整结果。

## 必做项

| 项目 | 状态 | 依据和可检查成果 |
|---|---|---|
| 核对 Forecaster 全流程 | 已完成 | Targets 与 pipeline 表；本项目沿用 Dow30 prompt/answer、LoRA、结构化预测和课程指标，不替换为情绪分类 |
| 获取 Dow30 数据：prompt/answer/label/symbol/period，train/test | 已完成 | §3.1；完整 parquet 1230/300 条已用于实验；公开提交仅提供原始来源和哈希，重新下载方法在README；另有可复现 60/30/30 本地辅助子集；指定8B实验使用60/30/300划分 |
| 数据概况与质量检查 | 已完成 | §4 brief analysis；`results/data_audit.json`、Notebook；缺失、重复、时间边界、标签分布、前视风险均记录 |
| 使用 LoRA/PEFT 冻结基础模型训练 | Llama已完成；第二模型未完成 | §0、§3.2；Llama8B实际训练日志和adapter验证见 cloud_runs/llama31_8b/results/；448个LoRA张量发生更新且有限；权重80MiB未随GitHub提交 |
| 选定 Llama-3.1-8B、DeepSeek-R1-Distill-Llama-8B 的微调/比较 | 部分完成 | §3.2 逐个列出；同文又说 two models，存在内部数量不一致，按用户2026-10-11指示，采用较简单的两模型解释：Llama3.1-8B和DeepSeek8B；Qwen7B不计入本次选择范围，教师是否接受此解释无法验证。已连接用户Colab Tesla T4并验证CUDA；DeepSeek8B的60条QLoRA训练、30条验证和adapter保存已完成，本地检查点哈希校验通过；DeepSeek评估按用户2026-10-12指令停止，Drive保留160条base、0条LoRA输出；Llama一轮训练、300条base和300条LoRA评估、六例单请求耗时测试均已完成；完整指标独立重算通过，最终PDF/Notebook已检查。DeepSeek完整比较仍未完成 |
| 按 GPU 显存设置长度、batch、累积和精度 | 本地与DeepSeek CUDA配置已实际运行 | §3.2；Apple MPS、0.5B float16、LoRA float32、batch1、accum4、1536 序列上限；`train_lora.py` |
| DeepSpeed、8-bit 与 Wandb/TensorBoard 监控 | 未完整完成 | §3.2；实际T4采用4-bit NF4 double quant；首个8-bit加载失败，确切错误未保留。不使用DeepSpeed，使用TensorBoard；这些是有记录的资源适配，不能谎称完全满足原文8-bit/DeepSpeed配置 |
| 从基础模型+adapter生成测试回答 | Llama完整完成；第二模型未完成 | §3.3；cloud_runs/llama31_8b/results/base_answers.jsonl 和 lora_answers.jsonl 各300条；30家公司、10个测试周 |
| 与教师参考量化比较，报告 valid_count | Llama完整完成；第二模型未完成 | §3.3；`results/metrics.json`/`.csv`；严格格式解析、不补写生成结果，保留失败样本；教师是数据集附带参考，未重新调用 API |
| bin_acc、MSE、分段ROUGE1/2/L | Llama完整完成；第二模型未完成 | §3.3；course-compatible midpoint编码、ROUGE F1；空有效集分数为 null，明确覆盖率；不能用参考相似度证明收益 |
| 推理时间 | Llama完整完成；第二模型未完成 | §3.3、§4；GPU同步、先warmup、每条真实耗时；包含prefill+decode，不包含模型加载 |
| 主观分析、执行正确性、深度、清晰性 | 自动审查已完成；本人 HITL 未完成 | §0 HITL 与 §4；`HITL_review.csv`/`QUALITATIVE_REVIEW.md`。AI自动审查不能冒充学生的人类审核 |
| 修改 train_lora.py、train.sh、config.json、utils.py，Notebook | 已完成 | §5.1；同名文件；config保留原CUDA配置并明确本地不使用，portable脚本修复硬编码、loss masking、架构选择和评估顺序 |
| adapter_config.json + adapter weights | 本地、DeepSeek训练和Llama8B权重完成 | §5.2；`finetuned_models/qwen05_local/`；DeepSeek8B检查点见`cloud_runs/deepseek8b/finetuned_models/deepseek8b/`；模型缓存仅在`work/`，不放提交内容 |
| 原始回答和每个比较模型的指标表 | Llama完整完成；第二模型未完成 | §5.3；实际base/LoRA，不伪造三种指定模型的结果 |
| PDF research report：硬件、环境、数据、微调、评估 | Llama报告已完成；第二模型未完成 | §4、§5.4；`Research_Report.pdf`四页实际Llama报告；明确第二模型未完成与训练规模缩小 |
| README：确切命令、硬件、Medium链接 | 命令/硬件已完成；Medium发布及链接已完成 | §5.5；`README.md`；已发表博客，真实URL见README；保留AI披露和第二模型未完成说明 |
| 新文件夹 Assignment1_Name_UNI / track submissions | 未完成 | §5、总说明；用户真实Name/UNI未提供，本地不假冒身份或建正式提交目录 |
| GitHub PR 到 master | 未完成（按用户指令不提交） | 总说明“new folder (not a new branch)”及PR流程；未push、未建PR |
| 截止日期 | 已核对 | 总说明与track：Monday Oct 12, 2026；原文无具体时刻或时区，不能推断11:59pm |
| 密钥卫生 | 已检查 | 总说明/§5；HF_TOKEN由Colab Secret读取，不写代码/本地文件。凭证清理返回值曾意外回显，已清除该输出并修正代码。导出的DeepSeek文本扫描无凭证模式；最终重新扫描仍需执行 |
| AI使用规则 | 无法验证完整课程规则 | 已读仓库文件未发现禁止/申报生成式AI的完整政策；不能推断Canvas/讲课要求没有额外规定；`AI_DISCLOSURE.md`记录本次AI辅助范围 |

## 可选项与学期目标

作业 §3.1 明确标为 **Optional**：自行用 Finnhub/OpenAI 采集 NASDAQ100 等数据。不开展自建数据不会自动违反这个条款。本次使用提供的 Dow30 数据，无需这两项 API key，也无 API 费用。

课程总览的学期目标包括整合新闻、社交媒体、买方意见的数据平台、设计标注策略、情绪微调及真实/近真实市场验证。本次作业明确覆盖：现成数据准备/审计、LoRA微调、预测回答与教师参考的实证指标。**并未要求交付整学期数据平台、社交数据整合、完整新标注体系或实盘策略。** Forecaster的目标是下一周股票走势及解释，并非三类情绪分类器。

## 同时 FinGPT 和 FinRL

Assignment 1 总说明写“track(s)”，明确多track分别提交相同命名文件夹。0–2学分任选一个；2–3学分强烈鼓励全部三个。**因此同时做FinGPT和FinRL在这份作业说明中被允许；没有发现会使作业无效的条款。** 2学分处区间重叠，且“strongly encouraged”不是“必须”。课程总览“一方向”描述学期研究结构，不能擅自推断多track能增加学分或无需导师批准。

无法依据GitHub确认你的注册学分、评分方式、导师批准、学期研究方向选择或毕业学分认定。本报告不声称FinGPT一定计入学分，不代表你确认课程政策。

## 用户最后需要提供/决定的事项

1. 若要完成原文选定两模型GPU实验，Colab CUDA入口已提供并验证；DeepSeek训练已完成，评估按用户要求停止；Llama完整训练、评估和成果整理已完成，无需用户逐步执行。
2. 检查真实结果及 `HITL_review.csv`，给出本人判断；确认课程额外AI规则。
3. 填真实Name/UNI，审阅/改写博客后自行发布Medium并填README链接。
4. 最后自行按track目录提交GitHub PR；本项目没有进行任何外部提交。

## 最终运行验证（2026-10-12）

Llama300base/300LoRA原始输出与原始test符号、期间、标签、参考答案逐条一致；指标重算通过。有效配对206/250，coverage68.67%/83.33%，bin_acc46.60%/49.20%，MSE7.13835/9.792。448个adapter张量全部有限，20,971,520元素；SHA256记录在full_validation.json。Notebook四个代码单元格以独立内核无错误执行；PDF四页全部渲染并检查。第二模型、本人HITL、Name/UNI和最终提交仍未完成；Medium博客已发表并验证。

博客发表验证：https://medium.com/@xujiao0224/fingpt-forecaster-with-llama-3-1-8b-formatting-gains-are-not-forecasting-proof-df1b7fa0e030。用户确认已审阅博客稿；定性评分工作表尚未代填。

提交状态更新：JIAO XU (jx2687) 已明确授权课程PR提交；Medium已发布。个人报告审阅由用户确认，数值HITL评分仍为空。第二模型比较未完成，adapter权重因大小未上传。原始parquet因再分发权限未验证而不公开上传，README给出固定revision下载方法。
