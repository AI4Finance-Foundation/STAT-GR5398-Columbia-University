"""Print every changed source unit and proposed atomic decision for explicit review."""
import argparse,json
from pathlib import Path
from grounding_audit import load_inputs,source_units
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('ticker');p.add_argument('--version',default='v7_1');p.add_argument('--section');p.add_argument('--blind-warnings',action='store_true');args=p.parse_args()
work=ROOT/'audits/grounding'/f'{args.ticker}_after_{args.version}_working'
proposals=json.loads((work/('changed_unit_proposals'+('_blind' if args.blind_warnings else '')+'.json')).read_text(encoding='utf-8'))
_,inputs=load_inputs(ROOT/'runs'/f'{args.ticker}_final_{args.version}',ROOT/'runs'/f'{args.ticker}_baseline_1')
for section,data in proposals.items():
    if args.section and section!=args.section:continue
    units={u['unit_id']:u for u in source_units(inputs[section]['text'])}
    print('\nSECTION',section)
    for unit in data['units']:
        print('UNIT',unit['unit_id'],units[unit['unit_id']]['text'])
        for i,c in enumerate(unit['claims']):
            print(i,c['classification'],c['numeric_status'],c['claim'],
                'EVIDENCE',[(r['line_start'],r['line_end']) for r in c['evidence']], 'REASON',c['reason'])
        if not unit['claims']:print('ZERO CLAIMS',unit['nonfactual_reason'])
