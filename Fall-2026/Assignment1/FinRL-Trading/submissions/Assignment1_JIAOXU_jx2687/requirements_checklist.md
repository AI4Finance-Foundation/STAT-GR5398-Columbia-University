# 作业原文要求核对表

JIAO XU / jx2687，2026-10-11。只核对 Fall-2026 Assignment 1 的 FinRL-Trading 部分。

权威原文：[Assignment 1 总说明](https://github.com/AI4Finance-Foundation/STAT-GR5398-Columbia-University/blob/master/Fall-2026/Assignment1/README.md)、[FinRL-Trading track](https://github.com/AI4Finance-Foundation/STAT-GR5398-Columbia-University/blob/master/Fall-2026/Assignment1/FinRL-Trading/README.md)。两份在线原文保存在本包中；项目总 README 和 ML_STOCK_SELECTION.md 只作为一般方向和数据语义资料，不替代 rubric。

| 明确要求 | 状态 | 当前成果、缺口与原文依据 |
|---|---|---|
| 运行简化 FinRL 全流程 | 已完成核心阶段，原流程有效性未完成 | 已核实11行业原模型结果；保留174,571条选股；新增严格时间规则的模型、组合、回测。原 notebook 全量训练未重跑，原收益计算存在缺价和时间问题。Track §0、§1.3 |
| 历史 S&P500 PIT名单必须使用 | 已完成过滤 | 21,299→19,007行，692→674 ticker，删除2,292行及18个完全退出样本ticker；新策略决策日再查PIT成员。Track §1.1、§1.2 |
| 证明无幸存者/前视偏差 | 未完成完整证明 | 已修复明确模型标签成熟时间、保守特征可用日及隔夜收益问题；价格池覆盖2025快照，缺历史证券；vendor修订、公司行动和名单重建未独立验证。Track §0、§4，不得标“已无偏” |
| 设计选股组合 | 已完成受限策略 | 新实验RF/ET，固定top25%、等权目标，季度调仓；缺价份额留现金，股票权重不放大。Track §2–§3 |
| 真实历史完整回测、交易成本 | 已完成受限回测，完整池未完成 | 2020-06-30→2025-09-30、21次决策、1319日收益，包含隔夜持仓、首买成本、0.1% gross notional cost。缺价历史股票不可执行。Track §3 |
| 权重、毛/净收益、换手、成本、NAV、曲线 | 已完成 | results/target_weights.csv、strategy_daily.csv、comparison_nav.csv、comparison.png。Track §3 Output |
| SPY和QQQ对比 | 已完成 | 基准已独立下载补齐2020上半年，统一日期；SPY代表S&P500，QQQ代表NASDAQ100；基准为无摩擦ETF复权持有。Track §3 Output |
| 自己策略全窗口跑赢S&P500 | **未完成** | 净累计109.10% < SPY132.66%。没有伪造收益，也没有按全期收益反复调参。Track §3 Important |
| 与原始等权基线并列 | **未完成有效对比** | 保存原训练/选股、模型胜者与quoted-only诊断。原信号前视且缺价，无法把partial NAV列成有效基线收益；README参考数不当本人结果。Track §3 Reference run |
| 报告数据范围、季度/ticker、PIT删除数、保留退市公司例子 | 已完成，价格缺口披露 | PDF §2–§3，historical_missing_examples.csv；ATVI、ABMD、CTXS、KSU、SIVB、FRC保留在准备数据中，但不具备完整日价。Track §4 Data |
| 模型赢家、特征重要性、top-k理由 | 已完成新模型；原模型importance无法确认 | 原xgb88/rf87/gbm63；新ET15/RF6；新模型impurity importance及固定top25%理由。原拟合模型未保存，不能补称原模型importance。Track §4 Stock selection |
| 累计、年化、Sharpe、回撤、波动、换手 | 已完成 | metrics.csv、PDF §7；独立从最终NAV复算。风险自由利率0；CAGR按365.25天；vol/Sharpe按252。Track §4 Performance |
| 不奏效的方法及改进方向 | 已完成 | PDF §4、§8；原时点漏洞、新策略落后指数、补历史证券和公司行动数据优先。Track §4 |
| clean clone可运行Notebook/脚本与精确命令 | 已完成脚本及执行Notebook；全新下载环境未实测 | README给出独立环境、数据下载、prepare命令和新实验命令；已实测现有conda环境，Notebook逐单元执行。复制输入的独立目录重运行验证见validation_results.txt。最新上游/报价变化可导致不同结果。Track §5 |
| 研究报告PDF、小型结果文件 | 已完成供检查材料 | Research_Report.pdf、results/、cache/；原始数据库/准备数据没有打包。Track §5 |
| Medium文章和README实际链接 | 未完成发布；草稿完成 | Medium_draft.md已有结果和限制，用户需审阅发布后填链接。Track §0、§5；总说明How to submit |
| submissions/Assignment1_Name_UNI文件夹与master PR | 文件夹已准备，PR未提交 | Assignment1_JIAOXU_jx2687；用户明确不许推送或提交，最终由用户操作。总说明How to submit、Track §5 |
| 禁止API key进入GitHub、不要提交原数据 | 已检查 | 包内无原始数据、prepared或.env；密钥扫描结果见validation_results.txt。原模型源码只有实现，不含用户API凭据。Track §5与总说明Warning |

## 截止日期与时间核对

- 在线总说明：Monday, Oct 12, 2026；track：Oct 12, 2026。**都没有具体时刻或时区**。
- 已通过Canvas只读API核实课程252721的time_zone为America/New_York。assignments、modules、announcements均0项；pages接口HTTP404。不存在可读due_at供确认。
- 因此截止**日期已确认**，截止**时刻与时区无法确认**。不能擅自写23:59 EDT。旧Notebook内Spring日期2026-02-20、2018–2025建议区间不覆盖当前Fall原文。
- 本次工作在纽约时间10月11日凌晨进行；采用10月12日开始前完成检查材料作为保守目标，不称其为老师规定截止。在线核查日期2026-10-11。

## 原文自身的区间差异

Track §1.2写2020-06-30起，§3参考实验2020-05-01起；db价格终点2025-10-16，示例准备数据终点2025-09-30。实际新实验使用2020-06-30初始NAV、2020-07-01首次执行、2025-09-30末日估值，不套用参考实验区间或绩效。

## 最短最后步骤

1. 阅读PDF和此表，核对结果和AI辅助披露，尤其未达成项；按课程政策调整你的表述。
2. 审阅/发布Medium草稿，把真实URL写入README。
3. 将命名文件夹放入课程repo的FinRL-Trading/submissions，检查无数据或密钥，亲自发起对master的PR；不要把ZIP直接替代要求的目录。

未进行真实交易、paper trading、下单、GitHub推送、PR、课程提交、发邮件或Medium发布。
