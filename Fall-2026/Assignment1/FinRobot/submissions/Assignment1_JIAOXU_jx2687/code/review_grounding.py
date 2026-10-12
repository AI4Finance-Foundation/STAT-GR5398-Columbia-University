"""Apply documented human/agent review corrections to provisional grounding labels."""
from pathlib import Path
import pandas as pd
R=Path(__file__).resolve().parents[1];p=R/'audit/grounding_claims.csv';f=pd.read_csv(p);changes=[]
manual={
"NVIDIA's forward guidance suggests revenue may exceed $130 billion in 2025.":'Historical 2025A revenue is not forward guidance; no guidance source was supplied.',
"NVIDIA's EBITDA and margins are forecast to continue expanding.":'Reviewer cited actual A columns, not forecast evidence; flat baseline margins do not demonstrate expansion.',
"Peers AMD and Intel lag considerably in EBITDA performance, especially as AI adoption and data center demand accelerate.":'Peer EBITDA supports relative comparison but does not establish the compounded AI-demand assertion.',
"NVIDIA’s jump from $26.9B in 2023A to $60.9B in 2024A (+125.9%), then nearly doubling again to $130.5B in 2025A, reflects rapid adoption of GPU and data center solutions by hyperscale and enterprise customers.":'Numeric component is supported but asserted customer-adoption cause is not. Compound claim conservatively unsupported; separate extracted numeric claims retained.'}
for i,x in f.iterrows():
 reason=manual.get(x.claim)
 if x.classification=='Supported' and any(s in str(x.reason).lower() for s in ['not directly supported','not supported','no evidence','not supply','does not supply','not provided','not established']):reason=reason or 'Classification contradicted its own reasoning: missing evidence explicitly acknowledged.'
 if reason:
  changes.append(dict(ticker=x.ticker,section=x.section,claim=x.claim,old_label=x.classification,new_label='Unsupported',review_reason=reason))
  f.loc[i,'classification']='Unsupported';f.loc[i,'reason']=reason;f.loc[i,'review_status']='Reviewed correction; conservative compound-claim handling'
f.to_csv(p,index=False);pd.DataFrame(changes).to_csv(R/'audit/grounding_review_corrections.csv',index=False)
g=f.groupby(['ticker','section']).classification.agg(total='size',supported=lambda x:(x=='Supported').sum(),unsupported=lambda x:(x=='Unsupported').sum(),contradicted=lambda x:(x=='Contradicted').sum()).reset_index();g['grounding_rate']=g.supported/g.total;g.to_csv(R/'audit/grounding_rates.csv',index=False)
print('Documented grounding corrections',len(changes))
