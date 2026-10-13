# Final-test 实际结果与评审材料的独立审计（2026-10-12）

审计者：Claude（Anthropic），独立于主实现代理 Codex。
实际模型 / 推理强度：`claude-opus-5-5`，effort `high`，fast mode off（来源：本会话元数据；与此前各轮同一会话）。
请求文件：`FINAL_TEST_ACTUAL_RESULTS_AUDIT_REQUEST_20261012.md`。
被审包：`D:\AI4FinRuntime\R3_FINAL_REVIEW_20261012_v1`。

## 0. 结论

**实际运行证据：接受，0 个缺陷。** 1800 条真实输出、900 条 prompt、全部 token 回执、分批顺序、18 组指标和子进程顺序，我都在本机 CPU 上独立复现，与包内记录一致。PDF 与 README 里的数字全部与我的重算一致。

**评审材料：有两处需要改文字的实质问题，另有一处建议同时补一句。** 三处都只改文字，不改任何数字或证据。

| 编号 | 位置 | 问题 | 性质 |
|---|---|---|---|
| D1 | `MEDIUM_DRAFT.md` 第 3 行 | 写着“no final-test result is included”，但同一份草稿的倒数第二段报告了 final-test 结果 | 已核实的事实错误 |
| D2 | PDF 第 4 节、Medium（README 部分缺） | 没有交代 test 的标签分布是 137 up / 163 down。文中唯一的方向基线 always-up 137 恰好是少数类 | 影响主结论读法的遗漏 |
| D3 | PDF 第 4–5 节 | test 表里 Llama base 的 220 条 course-valid 没有“模板回显”说明 | 建议同时补一句 |

这三处改完之后不需要我再做一轮：要补的数字都列在第 6 节，且都已由本轮 probe 核实，按字符串对照即可。

**不认证的内容**：本报告不认证完整的 Assignment 1、干净数据、预测能力或已完成的 HITL。公开发布、正式提交和你的最终验收都还没有发生。

## 1. 输入与约束

- `MANIFEST.json` SHA256 `363ef533d946b583a5b714d2ca6ac4e78870294d18016241e83580cb255924ea`，与请求一致；列 380 个文件，0 不一致；磁盘上 381 个文件，多出的一个是 manifest 本身。
- plan SHA `f284b5fb…`、授权文件 SHA `56296cab…` 与请求一致。`final_test_executor/` 的 9 个文件与我上一轮接受的 fallback 包逐字节相同。
- 遵守：本机 D 盘、CPU、离线；未 SSH、未用 GPU、未联网、未安装或下载、未读凭据、未训练、未生成新的 test 输出、未发布或提交。
- 本轮读取了 test 数据和 test 输出，依据是请求中的明确授权。tokenizer 只加载了包内三份非权重副本。
- 我没有在被审包内创建或修改任何文件，没有执行包内的 builder 或 Codex 的 verifier。
- 审计工作区内 15 个旧包的 manifest 全部 0 不一致；我此前的 24 份报告文件哈希未变；human 18 条评分导出未变；没有 `__pycache__` 或 scratch 残留。
- probe 与日志：`claude_probes_final_test_20261012/`（`y01`–`y03`，以及 PDF 文本抽取 `final_pdf_text.txt`）。

## 2. 实际运行证据（全部复现）

### 2.1 授权与时间顺序

| UTC 时间 | 事件 |
|---|---|
| 10-12 16:26:24 | plan 冻结 |
| 10-12 18:49:17.73 | 授权转录（原话“赶紧收尾完工了”） |
| 10-12 18:49:17.74 | 启动，父进程 pid 851430 |
| 10-12 18:50:38 | 首次读取 test，随后三次 V3 loader 读取 |
| 10-12 18:51:10 – 19:41:57 | Qwen 子进程（pid 852004） |
| 10-12 19:00:31 | 额外的 audit-copy 读取（worker 之外，有回执） |
| 10-12 19:42:44 – 21:53:04 | DeepSeek 子进程（pid 867184） |
| 10-12 21:53:48 – 10-13 01:49:55 | Llama 子进程（pid 905576） |
| 10-13 01:50:09 | `COMPLETED.json`（1800 outputs） |

- 要求的先后顺序成立：冻结 → 授权 → 启动 → 首次读取 → 三个子进程依次 → 完成。授权晚于冻结 2 小时 22 分 52 秒。
- 三个子进程不重叠（间隔 82 秒和 71 秒），每个子进程只有一个 pid。日志中没有 Traceback 或 OOM，没有 `FAILED.json`。
- 各子进程的墙钟时间与其生成秒数之和吻合（3020/3017、7785/7788、14140/14132 秒），没有重启的迹象。
- 授权范围：`publish` 与 `submit` 都是 false；派生的 scope 授权与 hydrated policy 的哈希相互一致。

### 2.2 test 源文件

- 文件 SHA 同时等于 plan、授权、首次读取回执，以及 10 月 9 日 `split_manifest.json` 里早先记录的值。
- 300 行，300 个不同的 id，全部 schema 检查 300/300；10 个预测周 × 30 个 symbol。
- 与 train、validation 在 id 和（symbol, period）上都是 0 重叠。hydrated policy 的 id 顺序等于文件顺序。
- 标签：137 up / 163 down。先前看过的 20 行是 11 up / 9 down（16 个 symbol、8 个周）；其余 280 行是 126 up / 154 down。

### 2.3 900 条 prompt

- 我用包内 tokenizer 副本重新编码，三个模型各 300/300 与 prepared 的 `input_ids` 相同。tokenizer 副本的哈希等于此前审过的绑定。
- 最长 prompt：2930 / 2657 / 2653 token，都低于 envelope 3517 / 3276 / 3272。fallback 行数为 0。
- prompt 中不含 reference 文本。

### 2.4 1800 条输出

- 六个输出文件各 300 行，22 项逐条检查全部 300/300。检查内容包括：id 顺序、label、reference、exposure 标志、prompt token 数、token SHA、token 数、EOS 与 cap 标志、解码结果等于保存的文本、`batch_size` 等于调度、变体名、plan SHA、think 标记、boundary 回执与 prompt 绑定一致。
- 分批：Qwen 和 DeepSeek 是 18×16 + 1×12，Llama 是 37×8 + 1×4。base 与 adapter 的调度和 id 顺序完全相同。
- 没有任何行被过滤或裁剪；六个文件共用相同的 id 顺序、reference 和 label。
- 输出回执的 SHA 全部吻合。

| 变体 | EOS / caps | 闭合 think | peak allocated / reserved GiB |
|---|---|---|---|
| qwen base | 300 / 0 | 0 | 12.11 / 19.72 |
| qwen adapter | 300 / 0 | 0 | 12.11 / 22.61 |
| deepseek base | 297 / 3 | 299 | 17.77 / 23.03 |
| deepseek adapter | 300 / 0 | 0 | 15.37 / 21.49 |
| llama base | 164 / 136 | 0 | 13.50 / 23.00 |
| llama adapter | 300 / 0 | 0 | 12.04 / 17.47 |

### 2.5 18 组指标

我用自己转写的 course parser、自己实现的 ROUGE 和 common final-line 规则重算。18 组（3 模型 × base/adapter × all300 / 20 / 280）的每个报告值都与我的结果相等，metrics JSON 与 CSV 都核对过。比较范围包括：course valid、bin_acc、MSE、同子集 +0.5 MSE、9 个 ROUGE、common 方向的 correct / valid、balanced accuracy、多条 Prediction 行数、first-line 正确数、first/last 不一致数、always-up、EOS / caps，以及 DeepSeek base 的 extracted 单独变体及其同子集基线。

all300 的结果：

| 变体 | course valid | bin_acc | course MSE | +0.5 MSE（同子集） | 方向 correct / valid | balanced acc |
|---|---|---|---|---|---|---|
| qwen base | 208 | 0.5288 | 11.2356 | 7.1827 | 161 / 296 | 0.5352 |
| qwen adapter | 297 | 0.4747 | 8.7475 | 7.6465 | 142 / 300 | 0.5008 |
| deepseek base | 0 | – | – | – | 130 / 265 | 0.4541 |
| deepseek adapter | 297 | 0.4983 | 8.5724 | 7.6465 | 148 / 300 | 0.5134 |
| llama base | 220 | 0.4818 | 9.3727 | 8.3000 | 2 / 4 | 0.0073 |
| llama adapter | 297 | 0.5051 | 8.5152 | 7.6465 | 150 / 300 | 0.5166 |

20 行组与 280 行组（方向 correct / valid，balanced accuracy）：

| 变体 | 先前看过的 20 行 | 其余 280 行 |
|---|---|---|
| qwen base | 12 / 20，0.6061 | 149 / 276，0.5307 |
| qwen adapter | 12 / 20，0.5758 | 130 / 280，0.4964 |
| deepseek base | 10 / 17，0.4747 | 120 / 248，0.4524 |
| deepseek adapter | 8 / 20，0.3838 | 140 / 280，0.5238 |
| llama base | 0 / 0，0.0000 | 2 / 4，0.0079 |
| llama adapter | 7 / 20，0.3283 | 143 / 280，0.5299 |

always-up 在三个组上分别是 137 / 11 / 126。

## 3. 这些结果说明什么

以下是我对数字的读法。除标签分布外，这些数字都已在包内；标签分布和 always-down 是我从 test 文件直接数出来的。

- **没有任何变体超过 test 的多数类。** test 是 137 up / 163 down，恒定预测 down 得 163/300。六个变体里最高的是 Qwen base 的 161。
- **三个 adapter 超过 always-up，不代表有预测能力。** 142 / 148 / 150 高于 137，只是因为 137 是这个 split 的少数类。balanced accuracy 是 0.5008 / 0.5134 / 0.5166。
- **adapter 仍然偏向预测 up。** 三个 adapter 预测 up 的行数是 245 / 219 / 207（共 300）。训练集多数是 up（542/990），而 test 多数是 down。
- **Qwen 的 adapter 低于它的 base。** 142 对 161。DeepSeek 是 148 对 130（base 有 35 条无效方向按错计）。
- **course MSE 上每个 adapter 都不如常数。** 8.7475 / 8.5724 / 8.5152 都高于同子集的 +0.5 常数 7.6465。
- **差距在噪声范围内。** 300 行来自 10 个周，同一周内的行相关。即使按独立计算，300 行上正确数的标准差也约为 8.7 行。
- **20 行组没有显示出曝光优势，也无法据此下结论。** DeepSeek 和 Llama 的 adapter 在这 20 行上是 8 和 7，低于 always-up 的 11；样本太小。

## 4. 评审材料

### 4.1 核对过的内容

- **PDF（8 页）**：我用 pdfminer 抽取了全文，88 个数字和关键表述全部与证据一致。包内 8 张页面 PNG 我逐张看过：表格和图完整，没有裁切或重叠，页脚和页码正常。
- **README**：三张表共 18 行，全部与我的重算一致。
- **Medium 草稿**：12 条 final-test 陈述与证据一致。
- **边界表述**（PDF 和 README 都写明）：
  - 1440 条 validation 与 48 条 HITL 输出分开，HITL 是独立的 development 实验。
  - AI 的判断没有冒充 human；没有新增 human 评分；human 最终验收写为 pending。
  - 20 / 280 分开报告，280 行没有声称 pristine，没有声称干净的 benchmark 或独立重建的收益。
  - Qwen 的 test batch 16 与 validation batch 1 不同，已披露。
  - DeepSeek adapter 的未闭合 think 监督已披露为 confound。
  - 发布、提交和最终验收写为 outstanding。
- **页面上的“review pending”和 `CURRENT_SCOPE.json` 里的 false 标志**是冻结时的状态，与请求的说明一致。
- `CURRENT_SCOPE.json` 的 `prior_validation_manifest_sha256` 指向 `validation_stage_history/MANIFEST.json`，哈希吻合。
- 沿用的目录（validation 输出、selected adapters、HITL、human 评分、AI 复核、Llama 训练）与我审过的 R3 包逐字节相同；`independent_audits/` 的 16 个文件与我的原报告相同。

### 4.2 D1：Medium 草稿自相矛盾

第 3 行原文：“This is a local draft about a retrospective validation experiment. It has not been published, and **no final-test result is included**.”

同一份草稿的倒数第二段给出了六个变体的 final-test 结果和 always-up 137。这句话现在是错的，需要改掉。

### 4.3 D2：test 的基线上下文缺失

- validation 部分写了标签分布（145 up / 95 down），test 部分没有。PDF、README、Medium 三份材料都没有出现“163”，也没有提到恒定预测 down 的得分。
- PDF 第 4 节只写了“Always-up and the frozen train-majority-up baseline are 137/300”。读者看到的是 adapter 的 142 / 148 / 150 和 Qwen base 的 161 都高于 137，容易读成“在 test 上超过了基线”。
- PDF 的 all300 主表没有 balanced accuracy（只有 20 / 280 表有）。README 的表里有。
- Medium 的标题是“did not beat a simple forecasting baseline”，但它给出的唯一 test 基线是 137，而三个 adapter 都高于 137。标题与正文数字对不上。补上 163 之后，标题才成立。

说明：always-up 是冻结 plan 里预先声明的基线，报告它是对的。always-down 不是预先声明的基线，补充时应写成“标签分布的事实”，不要写成新选的基线。

### 4.4 D3：Llama base 的 test 行缺少模板回显说明

- PDF 第 3 页对 validation 有这条说明（196 条占位符，190 条 course-valid 全部解析为 up）。test 表没有对应的话。
- test 上的实际情况：234/300 条输出含 `Up/Down` 占位符；220 条 course-valid 全部解析为 up，其中 208 条 margin 为 0。
- 所以 test 表里 Llama base 的 220 条 valid、MSE 9.3727 和 bin_acc 0.4818 不是有意义的预测。它的 MSE 比 Qwen base 低，也是这个原因。

## 5. 范围限制与观察（不是缺陷）

- **授权只有一句话。** 记录里保留了原话“赶紧收尾完工了”，并注明“Interpreted as instruction to proceed and finish”；时间戳是转录时间，不是消息时间，记录里也如实写了。这句话是否就是对 final-test 的批准，只有你能确认；gate 无法区分。
- **有一次 gate 之外的 test 读取。** 19:00:31 的 audit-copy 读取发生在 Qwen 子进程运行期间，有回执。那时 prepared prompt 已经固定，这次读取不影响输出。
- **GPU 侧的事实来自 Codex。** 我验证的是记录内部一致（token 能解码回文本、哈希吻合、调度一致）。输出确实由所述权重和 adapter 在 GPU 上产生，这一点我无法独立证明。
- **我无法证明服务器上没有其他 test 访问。** 能确认的是包内的回执链完整且自洽。
- **3 条 test reference 无法被 course parser 解析**（UNH 2024-03-17、KO 2024-03-31、GS 2024-04-14；GS 这条在 20 行组里）。所以 course-valid 的上限是 297，三个 adapter 都达到了上限。
- **2 条 test 行的财务表头日期晚于预测起点**（4 天）。
- **reserved 显存余量很小。** DeepSeek base 和 Llama base 的 reserved 峰值是 23.03 和 23.00 GiB，guard 是 23.42。运行完成了，但余量只有约 0.4 GiB。
- **包内现在含有完整的 test 数据和输出。** 分享或发布这个包时请留意。

## 6. 建议的文字修正（不改数字）

以下数字都已由本轮 probe 核实，可直接使用。

**D1（Medium 第 3 行）**：删掉“no final-test result is included”，并把“retrospective validation experiment”改为同时覆盖 validation 与 final test 的表述。

**D2（PDF 第 4 节；Medium 的 final-test 段；README 可加一句）**，需要出现的事实：
- test 标签是 137 up / 163 down。
- 恒定预测 down 得 163/300，高于全部六个变体（最高是 Qwen base 的 161）。这是标签分布的事实，不是预先声明的基线。
- all300 的 balanced accuracy：qwen base 0.5352，qwen adapter 0.5008，deepseek base 0.4541，deepseek adapter 0.5134，llama base 0.0073，llama adapter 0.5166。
- 三个 adapter 预测 up 的行数：245 / 219 / 207。
- 超过 always-up 137 不构成预测能力的证据；Qwen 的 adapter（142）低于它的 base（161）。

**D3（PDF 第 4 或第 5 节）**：Llama BASE 在 test 上有 234/300 条输出含 Up/Down 占位符；220 条 course-valid 全部解析为 up，其中 208 条 margin 为 0；这是模板被 parser 接受，不是预测。

可选（不要求）：Medium 里补一句 Qwen 的 test batch 16 与 validation batch 1 不同。PDF 和 README 已经披露了。

## 7. 未做的检查

- 没有在 GPU 上重跑任何生成，没有在 Linux 上执行任何东西。
- 没有执行 Codex 的 `verify_completed_test.py` 或报告 builder；`LOCAL_VERIFICATION.json` 和 Codex 的 visual QA 我只当作 Codex 的陈述，没有当作独立证据。
- 没有自己渲染 PDF。页面外观看的是包内的 PNG；文字内容是我从 PDF 本身抽取的，两者一致。
- 没有对 1800 条 test 输出做内容层面的事实核查（grounding、归因）。
- 没有重审 1440 条 validation 输出、48 条 HITL 输出或 human 评分，只确认了它们与已审的 R3 包逐字节相同。
- PDF 第 1–4 页和第 7–8 页与此前审过的 validation 报告相比，只有页脚、标题和 final-test / audit-boundary 段落不同；这些页我没有重新逐句对证据。
- 没有把 label 或 reference 与真实市场收益对照。
- 没有逐条对照课程评分细则；Name / UNI 和 Medium 公开链接仍待你提供。
