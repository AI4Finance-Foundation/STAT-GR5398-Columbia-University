"""Offline annotation metrics against independent atomic labels, never verifier verdicts.

Warning status propagates over the whole released paragraph. Lexical fact identity
and source-span retention are explicitly proxies, not semantic recall estimates.
"""
import argparse,csv,json,re
from pathlib import Path
from grounding_audit import load_audit,paragraphs,csv_write
from audit_io import write_json,utc_now,sha256
ROOT=Path(__file__).resolve().parents[1]

def ratio(a,b):return a/b if b else 'N/A'
def key(r):return r['ticker']+'|'+r['fact_key'].strip().lower()
def summarize(rows):
    flagged=[r for r in rows if r['warning_applies']]
    bad=[r for r in rows if r['classification']!='Supported']
    c=[r for r in rows if r['classification']=='Contradicted']
    return {'claims':len(rows),'warned_claims':len(flagged),
      'unsupported_or_contradicted':len(bad),'warned_unsupported_or_contradicted':sum(r['warning_applies'] for r in bad),
      'warning_recall':ratio(sum(r['warning_applies'] for r in bad),len(bad)),
      'warning_precision':ratio(sum(r['classification']!='Supported' for r in flagged),len(flagged)),
      'unflagged_unsupported_or_contradicted':sum(not r['warning_applies'] for r in bad),
      'contradicted':len(c),'warned_contradicted':sum(r['warning_applies'] for r in c),
      'unflagged_contradicted':sum(not r['warning_applies'] for r in c),
      'unique_lexical_fact_keys':len({key(r) for r in rows}),
      'duplicate_lexical_claim_instances':len(rows)-len({key(r) for r in rows})}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pairs',type=Path,default=Path('config/grounding_v6_pairs.json'))
    p.add_argument('--output',type=Path,default=Path('audits/grounding/annotation_v6'));args=p.parse_args()
    pairs=json.loads(args.pairs.read_text());allrows=[];results=[];facts=[]
    for pair in pairs:
        bi,b=load_audit(pair['before']);ai,a=load_audit(pair['after']);run=Path(pair['final_run']);paras={}
        for section in ai['sections']:
            path=run/'output'/ai['ticker']/'analysis'/(section+'.txt')
            text=path.read_text(encoding='utf-8')
            if sha256(text)!=ai['source_hashes'][section]:raise ValueError('Changed final text')
            paras[section]={r['paragraph_id']:r for r in paragraphs(text)}
        rows=[]
        for row in a:
            paragraph=paras[row['section']][int(row['paragraph_id'])]
            warning=bool(re.match(r'^\s*\[UNVERIFIED\b',paragraph['text']))
            rows.append(dict(row,warning_applies=warning,warning_basis='Whole paragraph prefix; independent underlying assertion classification unchanged'))
        for section in [*ai['sections'],'ALL']:
            subset=rows if section=='ALL' else [r for r in rows if r['section']==section]
            results.append({'ticker':ai['ticker'],'section':section,**summarize(subset)})
        bykey={}
        for r in rows:bykey.setdefault(key(r),[]).append(r)
        for r in b:
            matching=bykey.get(key(r),[])
            facts.append({'ticker':bi['ticker'],'section':r['section'],'before_claim_id':r['claim_id'],
              'before_classification':r['classification'],'claim':r['claim'],
              'after_matching_claim_ids':json.dumps([x['claim_id'] for x in matching]),
              'lexically_retained':bool(matching),'remaining_same_label_instances':sum(x['classification']==r['classification'] for x in matching),
              'interpretation':'Exact independent fact key only; nonmatch may be paraphrase, change, or removal, not confirmed correction.'})
        allrows+=rows
    for section in [*ai['sections'],'ALL']:
        results.append({'ticker':'ALL','section':section,**summarize(allrows if section=='ALL' else [r for r in allrows if r['section']==section])})
    args.output.mkdir(parents=True,exist_ok=False)
    csv_write(args.output/'annotation_scorecard.csv',results);csv_write(args.output/'annotated_claim_inventory.csv',allrows)
    csv_write(args.output/'lexical_retention_and_error_recurrence.csv',facts)
    csv_write(args.output/'unflagged_errors.csv',[r for r in allrows if not r['warning_applies'] and r['classification']!='Supported'])
    write_json(args.output/'manifest.json',{'created_utc':utc_now(),'pairs':pairs,'human_review_performed':False,
        'baseline_warning_count':0,'baseline_warning_recall':0,
        'limitations':'Warning detection is deterministic; semantic labels are independent Codex-assisted judgments with no human validation. Lexical keys do not establish unique semantic facts. Repetition changes raw denominators. No correction is inferred from annotation.'})
    print(json.dumps([r for r in results if r['section']=='ALL'],indent=2))
if __name__=='__main__':main()
