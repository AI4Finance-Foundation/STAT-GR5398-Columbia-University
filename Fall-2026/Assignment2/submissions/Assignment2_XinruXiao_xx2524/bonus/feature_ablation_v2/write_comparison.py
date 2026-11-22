"""Read only deterministic portfolio results and write the bonus comparison."""
import json
from pathlib import Path
import shutil

here = Path(__file__).resolve().parent
root = here.parents[2]
new = root / 'work/retraining_v2/model/output'
old = root / 'work/retraining/evaluation'
metrics = {'total_return_pct':'累计收益 %','cagr_pct':'年化收益 %',
           'max_drawdown_pct':'最大回撤 %','sharpe':'Sharpe',
           'turnover_per_year':'年换手','QQQ_total_return_pct':'QQQ 累计收益 %'}
raw = here / 'raw_portfolio_results'
raw.mkdir(exist_ok=True)
results = {}
lines = ['# 加分实验：移除五个历史估值倍数', '',
         '课程分数仍用于正式评分。重训结果单独保存。新版本沿用上一轮改进版，唯一新增的训练变化是删除 pe、ps、pb、peg、ev_multiple 五个输入列。原版重训使用未更改的 growth.json，复用同一运行环境中的已完成结果。', '',
         'A 为无估值过滤；B 为使用相同的作业估值过滤。三种重训分数与课程分数共用回测设置。', '',
         '| 时期 | 组合 | 分数来源 | ' + ' | '.join(metrics.values()) + ' |',
         '|'+'---|'*(3+len(metrics))]
for period in ('dev','val','test','forward'):
    for leg, tag in [('A','selection'),('B','valuation')]:
        for label, key, base, stem in [
            ('课程分数','course',old/'course/output','monthly_scores'),
            ('原版同环境重训','baseline',old/'retrained_baseline/output','baseline_scores'),
            ('上一轮改进重训','improved',old/'retrained_improved/output','improved_scores'),
            ('本轮特征消融','v2',new,'scores_v2')]:
            leg_tag = tag if key == 'v2' and period in ('dev','val') else leg+'_'+tag
            src = base / f'{period}_{stem}_{leg_tag}'
            stat = json.loads((src/'stats.json').read_text())
            results.setdefault(period,{}).setdefault(leg,{})[key] = stat
            lines.append(f'| {period} | {leg} | {label} | '+' | '.join(str(stat[k]) for k in metrics)+' |')
            if key == 'v2':
                target = raw / f'{period}_{leg}'
                shutil.copytree(src,target,dirs_exist_ok=True)
lines.extend(['','## 怎么读结果',''])
for period, legs in results.items():
    a = legs['A']
    lines.append(f"- {period} 的无估值组合：本轮相对原版重训的累计收益差为 {a['v2']['total_return_pct']-a['baseline']['total_return_pct']:+.2f} 个百分点；相对上一轮改进版的差为 {a['v2']['total_return_pct']-a['improved']['total_return_pct']:+.2f} 个百分点。本轮加入估值过滤后的收益变化为 {legs['B']['v2']['total_return_pct']-a['v2']['total_return_pct']:+.2f} 个百分点。")
lines.extend(['','本轮先只用开发期和验证期决定是否让候选进入最终评价，并在 `FROZEN_BEFORE_TEST.md` 写下决定和冻结版本；之后各运行一次 test 和很短的 forward 总对比，不再根据结果修改候选。课程 test 在此前研究中已经看过，因此本轮 test 不是严格未见样本。forward 仅覆盖 2026-10-12 至 2026-11-06，不能把年化指标当作长期预测。验证期和测试期改善仍不能证明稳定 Alpha。五个特征的历史信息可用时点未经核实，实验不能证明原特征存在未来信息。'])
(here/'对比报告.md').write_text('\n'.join(lines)+'\n')
(here/'comparison_metrics.json').write_text(json.dumps(results,indent=2))
print('COMPARISON COMPLETE',flush=True)
