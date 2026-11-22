# 选股模型加分实验：特征消融

正式评分继续使用老师的 `monthly_scores.parquet`。本实验保存在独立目录，不替换现有提交包和正式估值模块。训练前后校验课程分数文件的 SHA-256。

本轮从上一轮改进版删除五个历史估值倍数：`pe`、`ps`、`pb`、`peg`、`ev_multiple`，因为其精确可用日期尚未核实。这是一个对照实验，不是断言这五项有前视问题。保留其他特征、六个基模型与集成、配置参数、组合权重和估值过滤规则。配置文件仅改显示名，实际训练变更在 `selector.patch`；完整 `selector.py` 也在本目录。上一轮的同环境原版重训已完成 119 个月，沿用其分数与回测，另用上一轮改进版看这一次删特征本身的影响。

先试跑 2019 年 5–6 月。每月验证股票、行业组、训练截止日与原版重训一致，分数有限且无重复；通过后继续总计 119 个月，并生成开发期和验证期的 A/B 回测。开发期和验证期结果生成后，先写入 `FROZEN_BEFORE_TEST.md` 冻结候选，再各执行一次 test 和很短的 forward 总对比。之前已看过课程测试结果，且已有多次测试调用；因此不把本次 test 称为严格盲测，也不再根据其结果修改候选。

在任务根目录启动或恢复（相同输入下会复用已完成且通过校验的月份）：

```bash
export DYLD_LIBRARY_PATH="$PWD/work/openmp/lib"
export PYTHONPATH="$PWD/work/a2-retrain-deps:$PWD/work/a2-deps"
dest="$PWD/outputs/retraining/feature_ablation_v2"
python \
  "$dest/run_v2.py" --out "$dest/scores_v2.parquet" --workers 4 \
  > "$dest/run.log" 2>&1
```

此命令使用上一轮相同的按月训练驱动（调用课程按月训练函数），必须指定独立的 `--out`。运行使用 `work/retraining_v2/model` 中隔离的代码副本。查看 `run.log` 可跟踪进度，`logs/` 记录每月训练及各回测日志。成功后 `对比报告.md`、`comparison_metrics.json` 和 `raw_portfolio_results/` 保存对照结果。
