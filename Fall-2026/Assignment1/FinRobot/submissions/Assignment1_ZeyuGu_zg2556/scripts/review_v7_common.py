"""Helpers for explicit task-agent review, never automatic factual labeling."""
import copy,json,re
from pathlib import Path
from audit_io import write_json,sha256
ROOT=Path(__file__).resolve().parents[1]
def A(text,label,refs=(),reason='',num='not_numeric'):
    return {'claim':text,'classification':label,'evidence':[{'line_start':a,'line_end':b} for a,b in refs],
        'reason':reason,'numeric_status':num,'claim_type':'quantitative' if num!='not_numeric' else 'factual'}
def U(*texts):
    return [A(t,'Unsupported',reason='The original writer tables and supplied short news excerpt do not establish this specific business, comparative or causal assertion.',num='unverifiable' if re.search(r'\d',t) else 'not_numeric') for t in texts]
def S(text,refs,reason,num='not_numeric'):return A(text,'Supported',refs,reason,num)
def C(text,refs,reason,num='not_numeric'):return A(text,'Contradicted',refs,reason,num)
class Review:
    def __init__(self,ticker,version='v7_1'):
        self.ticker=ticker;self.work=ROOT/'audits/grounding'/f'{ticker}_after_{version}_working'
        self.spec=json.loads((self.work/'partial_specification.json').read_text(encoding='utf-8'))
        self.todo={(u['section'],u['unit_id']):u for u in json.loads((self.work/'requires_explicit_review.json').read_text(encoding='utf-8'))}
        self.done=[]
    def put(self,section,unit,atoms,nonfactual='',reason='Explicit task-agent review of all underlying premises, independent of verifier labels.'):
        if (section,unit) not in self.todo or any(x['section']==section and x['unit_id']==unit for x in self.done):raise ValueError('Unexpected/duplicate reviewed source unit')
        self.spec['sections'][section]['units'].append({'unit_id':unit,'atoms':atoms,'nonfactual_reason':nonfactual})
        self.done.append({'section':section,'unit_id':unit,'source_text':self.todo[section,unit]['text'],'review_reason':reason})
    def save(self):
        if {(x['section'],x['unit_id']) for x in self.done}!=set(self.todo):raise ValueError('Changed units remain unreviewed')
        for section in self.spec['sections'].values():section['units'].sort(key=lambda u:u['unit_id'])
        write_json(self.work/'explicit_changed_unit_review.json',self.done)
        write_json(self.work/'reviewed_specification.json',self.spec)
        print(self.ticker,'explicitly reviewed changed units',len(self.done))
