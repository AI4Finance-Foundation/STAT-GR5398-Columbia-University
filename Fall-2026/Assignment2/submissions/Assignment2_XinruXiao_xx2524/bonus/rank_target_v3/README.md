# v3横截面排名目标重训

本实验是独立bonus，不替换课程分数、正式估值模块或v2结果。完整设计和测试准入标准已在 `PRE_REGISTERED_PROTOCOL.md` 中预先固定。

唯一主要训练变化：把同一日期、同一bucket内已成熟的未来绝对收益转换为0至1的百分位排名，再训练相同的六个基础模型与stacking。沿用v2删除五个历史估值倍数、训练分区预处理和标签成熟度门槛。

从任务根目录运行或恢复：

```sh
export DYLD_LIBRARY_PATH="$PWD/work/openmp/lib"
export PYTHONPATH="$PWD/work/a2-retrain-deps:$PWD/work/a2-deps"
dest="$PWD/outputs/retraining/rank_target_v3"
python \
  "$dest/run_v3.py" --out "$dest/scores_v3.parquet" --workers 4 \
  > "$dest/run.log" 2>&1
```

脚本先试训两个月，再训练全部119个月，然后只运行dev/val A/B并依据预登记条件写出是否进入一次test的决定。课程分数SHA-256在训练前后检查。每月检查点允许同一输入身份下恢复；输入发生变化时必须使用新的实验目录。
