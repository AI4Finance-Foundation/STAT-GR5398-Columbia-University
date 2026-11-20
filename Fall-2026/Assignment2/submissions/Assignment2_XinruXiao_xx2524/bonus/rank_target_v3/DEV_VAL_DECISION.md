# v3开发期与验证期决定

此文件由保存的确定性回测结果生成，未读取v3的test或forward结果。

| 时期 | 版本 | A累计收益% | A Sharpe | A最大回撤% | A年换手 | B累计收益% |
|---|---|---:|---:|---:|---:|---:|
| dev | same_environment_unchanged | 195.49 | 0.909 | -33.65 | 10.26 | 未在本表重算 |
| dev | v2 | 187.92 | 0.932 | -34.57 | 10.03 | 165.32 |
| dev | v3 | 153.2 | 0.85 | -32.5 | 9.06 | 150.34 |
| val | same_environment_unchanged | -3.09 | 0.087 | -40.94 | 9.93 | 未在本表重算 |
| val | v2 | 11.85 | 0.342 | -37.8 | 10.05 | 10.28 |
| val | v3 | 16.37 | 0.401 | -41.18 | 10.11 | 19.26 |

## 预登记准入检查

- PASS: val_A_return_at_least_v2_11_85
- PASS: val_A_sharpe_at_least_v2_0_342
- FAIL: dev_A_return_at_least_177_92
- PASS: dev_A_turnover_no_more_than_125pct_v2
- PASS: val_A_turnover_no_more_than_125pct_v2

最终决定：不进入新的test；保留v2。
