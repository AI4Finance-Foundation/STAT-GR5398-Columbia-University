"""Load pinned course prices plus explicitly accepted recovery evidence."""
from pathlib import Path
import json
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
def aliases(root=ROOT):
 p=root/'evidence/price_recovery/security_aliases.json'
 result=json.loads(p.read_text()) if p.exists() else {}
 result.update({'FB':'META'})  # official 2022 same-class ticker change, root_corporate_action_sources
 return result

def prices(cache,root=ROOT,canonical=False):
 p=pd.read_csv(cache/'prepared/daily.csv',usecols=['tic','datadate','prccd','adj_close'],parse_dates=['datadate'])
 p['price_source']='pinned_course'
 recovery=root/'evidence/price_recovery';manifest=recovery/'recovery_manifest.json'
 if manifest.exists():
  existing=set(p.tic)
  for t in json.loads(manifest.read_text())['accepted']:
   if t in existing:raise ValueError('recovery must not overwrite pinned base prices: '+t)
   q=pd.read_csv(recovery/'accepted'/f'{t}.csv',parse_dates=['datadate']).rename(columns={'close':'prccd'});q['tic']=t
   q['price_source']='accepted_recovery:'+t
   p=pd.concat([p,q[['tic','datadate','prccd','adj_close','price_source']]],ignore_index=True)
 if canonical:
  p['security_id']=p.tic.replace(aliases(root))
  # Select ONE complete source per security, before looking at strategy returns.
  # An accepted supplemental series has priority over its pinned alias. Never
  # stitch individual dates across adjustment vintages or fill its missing days.
  sources=p[['security_id','tic','price_source']].drop_duplicates().copy()
  sources['source_priority']=(sources.price_source=='pinned_course').astype(int)
  sources['alias_priority']=(sources.tic!=sources.security_id).astype(int)
  chosen=sources.sort_values(['security_id','source_priority','alias_priority','tic'],kind='stable').drop_duplicates('security_id')
  p=p.merge(chosen[['security_id','tic','price_source']],on=['security_id','tic','price_source'],how='inner',validate='many_to_one')
  p['source_ticker']=p.tic;p['tic']=p.security_id
  p=p.drop(columns=['security_id']).sort_values(['tic','datadate'],kind='stable')
  assert not p.duplicated(['tic','datadate']).any()
 return p

def canonical_predictions(pred,root=ROOT):
 pred=pred.copy();pred['ticker']=pred.ticker.replace(aliases(root))
 if pred.duplicated(['ticker','decision_date']).any():
  pred=pred.sort_values('available' if 'available' in pred else 'decision_date').drop_duplicates(['ticker','decision_date'],keep='last')
 return pred
