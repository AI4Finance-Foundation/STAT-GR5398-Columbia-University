"""Report exact uncovered source fragments in saved real verifier responses."""
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path.home()/'.cache/finrobot-zg2556/upstream/finrobot_equity/core/src'))
from modules.equity_agents.verifier_agent import source_units

records=[]
for run in sorted(Path('runs').glob('*_final_v3')):
    baseline=Path('runs')/(run.name.replace('_final_v3','_baseline_1'))
    for response in sorted((run/'verifier').glob('*/response.json')):
        section=response.parent.name
        text=(baseline/'output'/run.name.split('_')[0]/'analysis'/(section+'.txt'))
        if not text.exists():
            text=next(baseline.glob('**/analysis/'+section+'.txt'))
        units={u['source_id']:u for u in source_units(text.read_text(encoding='utf-8'))}
        review=json.loads(response.read_text(encoding='utf-8'))['structured_output']
        for unit in review['units']:
            source=units[unit['source_id']]['text']; covered=[False]*len(source)
            for claim in unit['claims']:
                if claim['original_quote']:
                    for match in re.finditer(re.escape(claim['original_quote']),source):
                        covered[match.start():match.end()]=[True]*(match.end()-match.start())
            gaps=[]
            for match in re.finditer(r'0+', ''.join('1' if x else '0' for x in covered)):
                gap=source[match.start():match.end()]
                if any(c.isalnum() for c in gap):gaps.append(gap)
            if unit['claims'] and gaps:
                records.append(dict(run=run.name,section=section,unit=unit['source_id'],source=source,gaps=gaps))
print(json.dumps(records,ensure_ascii=False,indent=2))
