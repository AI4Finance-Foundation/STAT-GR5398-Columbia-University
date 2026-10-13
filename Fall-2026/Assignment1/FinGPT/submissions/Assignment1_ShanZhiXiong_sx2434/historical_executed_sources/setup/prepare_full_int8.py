"""Prepare the same 32 training examples with intact answers and richer news."""
import hashlib,json,re,sys
from pathlib import Path
from transformers import AutoTokenizer
ROOT=Path('/root/shared-nvme/AI4FinRuntime')
PROJECT=ROOT/'import_20261009/assignment1_20261008'
sys.path.insert(0,str(PROJECT))
from data_processing import parts,tokenize_prompt
OUT=ROOT/'setup/full_int8_4096_20261010'
assert not OUT.exists(), 'Prepared experiment exists; preserved'
OUT.mkdir()
tok=AutoTokenizer.from_pretrained(str(ROOT/'huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28'),local_files_only=True)
def encode(row,length):
    answer=tok.encode(row['answer'].strip(),add_special_tokens=False)+[tok.eos_token_id]
    budget=length-len(answer)
    raw=row['prompt'].split('<</SYS>>',1)[-1].strip()
    raw=re.sub(r'\[/INST\]\s*$','',raw)
    raw=raw.replace('cryptocurrency price movement','stock price movement')
    complete=tokenize_prompt(tok,raw)
    core,prices,news,fields,sector=parts(row)
    full_news=partial_news=0
    if len(complete)<=budget:
        prompt=complete;retained=len(news);full_news=len(news);raw_retained=True
    else:
        raw_retained=False;text=core
        assert len(tokenize_prompt(tok,text))<=budget,'Essential prompt plus full answer exceeds context'
        def add(line):
            nonlocal text
            candidate=text+line
            if len(tokenize_prompt(tok,candidate))<=budget:text=candidate;return True
            return False
        if sector:add(f'Sector: {sector}.\n')
        for p in reversed(prices[:-1]):add(f'History {p[0]} to {p[1]}: {p[3]}.\n')
        for key in ('eps','peTTM','grossMargin','operatingMargin','roeTTM','totalDebtToEquity','currentRatio','bookValue'):
            if key in fields:add(f'{key}: {fields[key]}\n')
        ordered=sorted(enumerate(news),key=lambda x:(x[1]['date'],row['symbol'].lower() in x[1]['headline'].lower()),reverse=True)
        kept=[]
        for i,item in ordered:
            if add(f"News {item['date']}: {item['headline']}\n"):kept.append(i)
        retained=len(kept)
        for i in kept:
            item=news[i];prefix=f"Detail ({item['date']}, {item['headline']}): "
            if add(prefix+item['summary']+'\n'):full_news+=1;continue
            tokens=tok.encode(item['summary'],add_special_tokens=False);lo=0;hi=len(tokens)
            while lo<hi:
                mid=(lo+hi+1)//2
                if len(tokenize_prompt(tok,text+prefix+tok.decode(tokens[:mid])+'\n'))<=budget:lo=mid
                else:hi=mid-1
            if lo>=8 and add(prefix+tok.decode(tokens[:lo])+'\n'):partial_news+=1
        prompt=tokenize_prompt(tok,text)
    inputs=prompt+answer;labels=[-100]*len(prompt)+answer
    assert len(inputs)<=length and [v for v in labels if v!=-100]==answer
    assert tok.decode(answer[:-1])==row['answer'].strip(),'Answer changed during preparation'
    return {'sample_id':row['sample_id'],'input_ids':inputs,'attention_mask':[1]*len(inputs),'labels':labels}, {
      'sample_id':row['sample_id'],'sequence_tokens':len(inputs),'prompt_tokens':len(prompt),'answer_tokens':len(answer)-1,
      'answer_truncated':False,'raw_prompt_retained':raw_retained,'original_news':len(news),'retained_headlines':retained,
      'full_news_summaries':full_news,'partial_news_summaries':partial_news,'original_prompt_tokens':len(complete)}

summaries={};all_ids=set();train_data=None;train_audits=None
for split,count in [('train',990),('validation',240)]:
    source=PROJECT/f'data/{split}_raw.json'
    rows=json.loads(source.read_text())
    assert len(rows)==count and all(r['split']==split for r in rows)
    ids={r['sample_id'] for r in rows};assert len(ids)==count and not(ids&all_ids);all_ids|=ids
    data=[];audits=[]
    for row in rows:
        encoded,audit=encode(row,4096);data.append(encoded);audits.append(audit)
    dest=OUT/f'qwen_{split}_4096.json';dest.write_text(json.dumps(data))
    (OUT/f'qwen_{split}_4096_audit.json').write_text(json.dumps(audits,indent=2))
    summaries[split]={'samples':count,'min_sequence_tokens':min(a['sequence_tokens'] for a in audits),
      'max_sequence_tokens':max(a['sequence_tokens'] for a in audits),'mean_sequence_tokens':sum(a['sequence_tokens'] for a in audits)/count,
      'max_answer_tokens':max(a['answer_tokens'] for a in audits),'answers_truncated':0,'dropped_samples':0,
      'raw_prompts_fully_retained':sum(a['raw_prompt_retained'] for a in audits),
      'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}
    if split=='train':train_data=data;train_audits=audits
# Stress inputs come only from training: union of longest total sequences and longest answers.
ranking=sorted(range(990),key=lambda i:train_audits[i]['sequence_tokens'],reverse=True)[:16]
ranking+=sorted(range(990),key=lambda i:train_audits[i]['answer_tokens'],reverse=True)[:16]
ranking=list(dict.fromkeys(ranking))
ranking+=sorted((i for i in range(990) if i not in ranking),key=lambda i:train_audits[i]['sequence_tokens'],reverse=True)
stress=[train_data[i] for i in ranking[:32]]
(OUT/'qwen_stress_4096.json').write_text(json.dumps(stress))
(OUT/'preprocessing_summary.json').write_text(json.dumps(summaries,indent=2))
proto=json.loads((PROJECT/'protocol.json').read_text())
proto['training'].update(context_tokens=4096,quantization='LLM.int8 8-bit base; higher-precision compute and LoRA adapters',lr_scheduler='cosine',warmup_ratio=0.03)
proto['parent_protocol_sha256']=hashlib.sha256((PROJECT/'protocol.json').read_bytes()).hexdigest()
proto['version']='server4090_int8_4096_20261010'
proto['preprocessing']='Retain full reference answer plus EOS; retain original prompt when it fits; otherwise compress news input with no fixed headline/summary token caps.'
(OUT/'protocol.json').write_text(json.dumps(proto,indent=2))
print(json.dumps(summaries,indent=2),flush=True)
