请执行一个长程任务，完成 STAT GR5398 Fall 2026 Assignment 1 的 FinRobot 项目，交付真实运行、证据完整、可以复现和正式提交的成果。核心目标是运行系统、审计可信度、实现一个可量化改进，不是仅生成几篇自动研报。

工作区：
C:\Users\zeyug\PycharmProjects\STAT-GR5398-Columbia-University\Fall-2026

必须阅读：
- Assignment1\README.md
- Assignment1\FinRobot\README.md
- Assignment1\FinRobot\source_code 下全部文件
- Assignment1\FinRobot\submissions\README.md
- 适用的 AGENTS.md
- 上游 AI4Finance-Foundation/FinRobot 的 finrobot_equity 实际实现

提交目录：
Assignment1\FinRobot\submissions\Assignment1_ZeyuGu_zg2556\
姓名 Zeyu Gu，UNI zg2556。

一、任务组织与环境

1. 当前本地只有说明、两步运行包装和四张仅有表头的审计CSV，没有完整finrobot_equity实现。先建立逐项验收表，将课程每项要求映射到代码、运行证据和产物。旧AutoGen annual-report教程是可选热身，不能代替本作业。
2. 固定上游commit SHA并阅读数据处理、八个agent、估值、敏感性、报告渲染的实际代码。使用专用环境，记录依赖与系统。将baseline可运行所需兼容修复与正式研究改进分开，保存可应用的上游patch；不要把带.git的整个上游clone塞进课程提交。
3. 保留原始材料和已有工作，只修改本track拥有的内容，不改其他两个项目，不在共享工作目录切换分支或reset。维护可恢复进度、run manifest、费用和错误日志。
4. 一开始核查FMP数据端点权限、OpenAI或兼容服务的模型/结构化输出能力、凭证与费用上限、HTML/PDF环境。缺项一次性说明，继续无依赖工作。不要让用户在聊天里粘贴密钥，不显示/提交config.ini、.env或凭证；不自行购买服务或增加未经授权的支出。
5. 自主推进实现、调试、运行、分析与写作。过程中文，课程报告与博客默认英文。不用mock、合成数据、示例文章或空表冒充正式实验。

二、覆盖与baseline运行

6. 按课程示例用 NVDA、AMD、INTC、AAPL、GOOGL 完成上手验证；正式覆盖必须选择明确的同一行业至少5家公司，每家至少2个合理peers。示例名单不自动构成合格的同一行业。
7. 默认可调查半导体行业 NVDA、AMD、INTC、AVGO、QCOM，先核实分类与数据可用性；必要时选择数据充分的同业名单并解释。保存coverage_universe.csv，包含行业依据、选取理由、peer映射，并尽量覆盖不同价格表现的公司。重合的热身公司可复用同版本同输入结果以控制成本。
8. 对正式覆盖每家公司真实执行数据/分析/八agent文本→图表/HTML两阶段，开启课程要求的sensitivity、catalyst和enhanced-news分析。生成5份以上可打开的HTML及全部assets，保留完整output/{TICKER}/analysis文件。每家公司PDF可选，最终研究报告PDF必交。
9. 保存原始API响应、来源与抓取时间、财务期/发布日期、新闻窗口、实际送入每个agent的完整证据prompt与instructions、模型/参数、结构化响应和最终文本、token usage、耗时、重试、错误与run_id。冻结可重复使用的数据快照，不让实验比较中的市场数据随时间变化。

三、修复并追踪数据流

10. 核实当前上游存在的复现陷阱，不能假设包装脚本已覆盖：
- 预测列可能仍固定2025E/2026E/2027E，检查是否与最新actual重叠；必要时按最后actual年份生成未来三年，并保留baseline行为和修复说明。
- Step 2可能重新抓取价格、评级和目标价，也可能重新生成文字。追踪最终渲染文本，不能审计旧文本后交付另一版。
- 区分FMP首页评级/目标价、估值引擎输出和LLM解释，建立字段来源表。
- 包装脚本可能没有开启valuation analysis，需显式运行所需引擎。
- 原生sensitivity可能只给Revenue/EBITDA变化，不能当作target-price敏感性。
- 缺失数据、fallback常数和默认假设必须显式显示，不得静默当成真实观测。

四、完成全部审计

11. 数据审计：至少2家公司，每家选10项pipeline指标，逐项核对公司SEC 10-K/10-Q。
输出至少20行实质审计记录，含ticker、metric、财务期、pipeline值、filing值、币种/单位、差值、verdict、SEC accession与来源链接、页码/表名或XBRL context、解释。
派生指标展示公式。解释财年、重述、GAAP/non-GAAP、EBITDA口径等差异；市场倍数需要额外价格输入时明确日期与来源，不虚构SEC直接披露值。

12. 预测审计：记录默认三年收入增长、利润率及SG&A假设，展示其对所选公司的含义；提出有证据的调整并说明理由。区分历史actual、未来forecast、财年与自然年，不把未来已知数据用于历史预测解释。

13. 估值审计：用独立可运行notebook或spreadsheet手工复算至少一种估值，不得只调用同一引擎后称独立复现。
列出EBITDA/FCF年份、shares、net debt、target multiple、discount rate、terminal growth等全部显式和隐藏假设。检查引擎fallback，例如用估计净债务或由EBITDA推算FCF的情况。输出逐项reconciliation、差值与原因。

14. 敏感性：运行原sensitivity analyzer，再将收入/利润率每个情景传播到同一估值方法，输出真正的target-price敏感性CSV与图。明确相对收入变化和利润率百分点的区别，报告目标价范围、弹性及点估值局限。

15. Grounding审计：至少3家公司×全部8个agent sections，覆盖每一条事实性和数量性atomic claim，不能每节仅抽样几句。
输出claim级CSV，包含公司、section、run_id、原文位置、claim、Supported/Unsupported/Contradicted、生成时prompt中的精确证据位置和判定理由。
“后来上网查到为真但原prompt没有提供”的内容仍属于Unsupported，不能回填证据改变baseline分类。
报告每家公司、每section及总体的supported/total、unsupported、contradicted数量；零claim标N/A。解释哪些agent最易失去证据支持及其prompt设计原因。
审计对象必须与最终渲染文本一致；若Step 2改写，保留对应版本和审计。

16. 稳定性：对至少一个ticker，在完全相同冻结输入、模型及参数下独立运行3次，分别保存结果。
比较rating、target price、top-3 risks；报告数值方差、风险集合重合与适当排序一致性。说明提取方法和每个字段来源。固定FMP市场字段导致零方差，不能解释为LLM本身稳定。
记录每次tokens、runtime和费用。明确这是三次独立生成，不能重复复制同一缓存结果。

五、只做一个深入改进

17. 默认选择课程Option A：增加第九个verifier阶段。将确定性数字核对与基于实际输入证据的claim检查结合，输出可追溯的标注或修订，让最终报告使用验证后的文本。
18. 保存原始文字、修订文字、claim映射、来源证据、修改理由与验证状态。Unsupported不能凭“看起来合理”放行，verifier不得自造引用；Contradicted应定位冲突并纠正或删除。区分无法验证与已被证伪。
19. 在相同至少3家公司、相同冻结输入上，对完整8个sections开展before/after比较，至少报告grounding rate、contradiction rate、数字错误率、claim保留率/事实覆盖、额外tokens、费用和耗时。
不能靠删光有信息内容提高grounding rate。保存剩余错误案例和失败修复。评价方法尽量独立于verifier，明确机器辅助标注的局限；未发生真人复核就不能声称人工验证。
20. 提供相对于固定上游commit的干净diff/patch，并在干净上游副本验证能够应用和运行。不要同时铺开五种改进而漏掉必做审计。

六、交付与完成标准

21. 最终交付至少包含：
- coverage universe和peer映射；
- 可复现代码、运行脚本、依赖与脱敏配置；
- 每家公司baseline/final报告、assets和完整analysis证据；
- data、forecast/valuation、grounding、stability审计表；
- 独立估值复算notebook/表格；
- target-price敏感性结果；
- verifier实现、可应用patch、before/after scorecard；
- 数据/prompt/model版本、输入hash、运行日志、成本与runtime台账；
- 英文研究报告PDF及可编辑源；
- Medium完整分析文章；
- README、产物清单及逐项验收表。

22. 研究报告覆盖模型和环境、总API成本与每份报告耗时、行业与公司选择、全部审计、稳定性、改进设计与对照、剩余限制；明确回答哪些部分已足以供portfolio manager使用、哪些仍不可靠及原因。
Medium文章应分析自己的实验和发现，不能把系统自动生成的股票研报直接粘贴为博客。

23. 实际验收覆盖：
- 同行业≥5家公司、每家≥2 peers；
- ≥2家公司×10项数据审计；
- ≥3家公司×全部8 sections的完整claim审计；
- 同输入3次独立稳定性运行；
- 独立估值复算与目标价敏感性；
- 相同输入before/after及可应用patch；
- 原始证据可追溯、指标可从保存数据重算；
- HTML和PDF无缺图、截断、乱码或错误数字；
- 配置、日志、notebook和patch无凭证泄漏。
清楚区分离线快照复现和重新联网获取最新数据。

24. README给出从干净环境到每次实验与报告的确切命令、上游SHA、模型/脱敏配置、费用、数据时间截面及Medium真实链接。
25. Medium发布和课程master PR作为最终提交环节：先准备完整可审阅材料和diff；已有明确授权则执行，否则材料齐备后集中请求最终发布授权。课程提交必须是指定新文件夹；上游修改用patch或独立fork流程，不能用分支代替课程目录。
26. 必做项全部有真实证据才可称完成。缺API权限、真实运行、完整审计或发布环节时，完成其他独立工作，明确列出仍欠项目、原因、续跑命令和所需最少输入。禁止将工程准备完成、模板填好或报告外观完整称为作业全部达标。