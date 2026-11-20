import json,hashlib
from pathlib import Path
import pandas as pd
r=Path(__file__).resolve().parents[2];o=r/'outputs/retraining'
a=pd.read_parquet(o/'baseline_parts/2019-06.parquet');b=pd.read_parquet(r/'work/fast_equivalence_parts/2019-06.parquet');pd.testing.assert_frame_equal(a,b)
record={'check_month':'2019-06','rows':len(a),'assert_frame_equal':'PASS (all columns, values, order and dtypes)','change':'Only when every unified blending alpha is zero, skip fitting the unused unified model; retain all bucket models and original bucket percentile score. Same driver applied to baseline and improved. Completed checkpoints from original driver remain valid. No financial assumptions or model hyperparameters changed.','limitation':'One-month empirical equality plus inspection of zero-weight algebra; this does not prove identical auxiliary unused-model diagnostics. Training labels and feature vintages retain separately disclosed limitations.'}
for name in ['train_one_month_original_driver.py','train_one_month.py']:record[name+'_sha256']=hashlib.sha256((o/name).read_bytes()).hexdigest()
(o/'runtime_optimization_equivalence.json').write_text(json.dumps(record,indent=2))
with (o/'重训研究协议.md').open('a') as f:f.write('\n## 运行效率补充（非收益调参）\n统一模型混合权重全部为0时，略过其不参与分数的训练；保留四个bucket的六模型与stacking。2019-06原驱动与快速驱动504行、所有列及dtype逐项完全相同，验证记录见runtime_optimization_equivalence.json。前期已完成的原驱动检查点保留；此后两版共用同一快速驱动。只保证最终分数等价，不声称保留未参与输出模型的诊断。\n')
print('Saved exact-equivalence record.')
