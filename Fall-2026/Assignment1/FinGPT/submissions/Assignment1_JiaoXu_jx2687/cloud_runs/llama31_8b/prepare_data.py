from pathlib import Path
import json,re,hashlib,platform,sys
import pandas as pd
from datasets import Dataset,DatasetDict
from utils import parse_answer
ROOT=Path(__file__).resolve().parent

def main():
    raw={s:pd.read_parquet(ROOT/'data/raw'/f'{s}.parquet') for s in ['train','test']}
    audit={}
    for s,d in raw.items():
        d['start']=d.period.str[:10];d['end']=d.period.str[-10:]
        audit[s]={'count':len(d),'symbols':sorted(d.symbol.unique()),'start_min':d.start.min(),'start_max':d.start.max(),'end_max':d.end.max(),'missing':d[['prompt','answer','label','symbol','period']].isna().sum().to_dict(),'duplicate_prompt':int(d.prompt.duplicated().sum()),'duplicate_symbol_period':int(d.duplicated(['symbol','period']).sum()),'label_counts':d.label.value_counts().to_dict(),'direction_counts':d.label.str.split().str[0].value_counts().to_dict(),'reference_parse_valid':sum(parse_answer(a) is not None for a in d.answer),'crypto_prompt_count':int(d.prompt.str.contains('cryptocurrency price movement').sum()),'assumed_target_in_prompt':int(d.prompt.str.contains("assume your prediction").sum())}
        parsed=[parse_answer(a) for a in d.answer]
        audit[s]['parsed_reference_label_mismatch']=sum(p is not None and p['prediction_binary']!=(1 if l.startswith('up') else -1) for p,l in zip(parsed,d.label))
        cut=[];future=[];profile=set()
        for row in d.itertuples():
            cut+=re.findall(r'From (\d{4}-\d{2}-\d{2}) to (\d{4}-\d{2}-\d{2})',row.prompt)
            m=re.search(r'reported at (\d{4}-\d{2}-\d{2})',row.prompt)
            if m and m[1]>row.start:future.append(row.symbol+' '+row.period)
        audit[s]['financial_period_after_forecast_count']=len(future)
        audit[s]['past_price_window_end_after_forecast_count']=sum(b>row.start for row in d.itertuples() for a,b in re.findall(r'From (\d{4}-\d{2}-\d{2}) to (\d{4}-\d{2}-\d{2})',row.prompt))
    audit['cross_split_duplicate_prompts']=len(set(raw['train'].prompt)&set(raw['test'].prompt))
    audit['cross_split_symbol_period_overlap']=len(set(zip(raw['train'].symbol,raw['train'].period))&set(zip(raw['test'].symbol,raw['test'].period)))
    audit['global_train_end_le_test_start']=raw['train'].end.max()<=raw['test'].start.min()
    # Time-based validation, purged at the target-window boundary; never optimize on test.
    train=raw['train'][raw['train'].end<='2024-01-28'].copy()
    val=raw['train'][raw['train'].start>='2024-01-28'].copy()
    val=val.drop_duplicates(['symbol','period'])
    train=train.drop_duplicates(['symbol','period'])
    symbols=sorted(set(train.symbol)&set(val.symbol)&set(raw['test'].symbol))
    # Fixed design chosen BEFORE evaluation: two training rows per ticker, one val/test row.
    train=pd.concat([g.sort_values('start').sample(n=min(2,len(g)),random_state=5398) for _,g in train.groupby('symbol')],ignore_index=True)
    train=train[['prompt','answer','label','symbol','period','start','end']]
    val=val.sort_values(['symbol','start']).groupby('symbol').head(1)
    test=raw['test'].sort_values(['symbol','start'])
    out=DatasetDict({s:Dataset.from_pandas(d[['prompt','answer','label','symbol','period']],preserve_index=False) for s,d in [('train',train),('validation',val),('test',test)]})
    out.save_to_disk(str(ROOT/'data/local_subset'))
    for s in out:
        with open(ROOT/'data'/f'{s}_subset.jsonl','w') as f:
            for row in out[s]:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    audit['experiment']={'seed':5398,'counts':{s:len(out[s]) for s in out},'selection':'2 fixed random historical train rows / ticker; earliest validation row per ticker; full chronological 300-row test split','train_target_end_max':train.end.max(),'validation_start_min':val.start.min(),'test_start_min':test.start.min(),'validation_not_used_for_hyperparameter_selection':True, 'train_unique_target_weeks':int(train.period.nunique()), 'validation_unique_target_weeks':int(val.period.nunique()), 'test_unique_target_weeks':int(test.period.nunique()), 'direction_counts':{s:d.label.str.split().str[0].value_counts().to_dict() for s,d in [('train',train),('validation',val),('test',test)]}}
    audit['limitations']=['No news publication timestamps or filing/availability dates in the released rows. Quarter end is not filing date.','Current-profile market capitalization and shares outstanding can leak future snapshots into historical prompts.','Official upstream teacher code conditions analysis on realized future movement. The released dataset card does not confirm its exact generation recipe; hindsight contamination is a material risk. ROUGE does not establish independent forecasting ability.','Dataset card does not state a license or provide raw prices/news provenance. Raw data here is a local review copy; confirm redistribution permission before GitHub commit.','Base model pretraining cutoff may overlap test dates; no claim of clean prospective market performance.','No trading backtest requested or executed.']
    (ROOT/'results/data_audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
