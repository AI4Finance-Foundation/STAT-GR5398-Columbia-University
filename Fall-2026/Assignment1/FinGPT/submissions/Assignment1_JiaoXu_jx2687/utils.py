"""Course metric compatibility, plus explicit coverage and safe supervision.
Original: sources/utils.py, course commit in provenance.json.
"""
import re,math
from collections import defaultdict
from rouge_score import rouge_scorer
lora_module_dict={k:['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'] for k in ['llama2','llama3','deepseek','qwen']}

def parse_answer(answer):
    m=re.match(r'^\s*\[Positive Developments\]:\s*(.*)\s*\[Potential Concerns\]:\s*(.*)\s*\[Prediction (&|and) Analysis\]:\s*(.*)\s*$',answer,flags=re.DOTALL)
    if not m:return None
    pros,cons,pna=m.group(1),m.group(2),m.group(4)
    m=re.match(r'^Prediction:\s*(.*)\s*Analysis:\s*(.*)\s*$',pna,flags=re.DOTALL)
    if not m:return None
    pred,anal=m.group(1),m.group(2)
    direction=1 if re.search(r'up|increase',pred.lower()) else -1 if re.search(r'down|decrease|decline',pred.lower()) else 0
    m=re.search(r'(\d)-(\d)%',pred)
    if not m:m=re.search(r'(?:more than )?(\d)+?%',pred)
    return {'positive developments':pros,'potential concerns':cons,'prediction':direction*(int(m.group(1))+.5) if m else 0.,'prediction_binary':direction,'analysis':anal}

def calc_metrics(answers,gts):
    if len(answers)!=len(gts):raise ValueError('Mismatched lengths')
    pairs=[(parse_answer(a),parse_answer(g)) for a,g in zip(answers,gts)]
    valid=[(a,g) for a,g in pairs if a and g]
    out={'total_count':len(pairs),'valid_count':len(valid),'coverage':len(valid)/len(pairs) if pairs else 0.,'prediction_parse_failures':sum(a is None for a,g in pairs),'reference_parse_failures':sum(g is None for a,g in pairs),'bin_acc':None,'mse':None,'unconditional_direction_accuracy':sum(a is not None and g is not None and a['prediction_binary']==g['prediction_binary'] for a,g in pairs)/len(pairs) if pairs else None}
    scorer=rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'],use_stemmer=True)
    if valid:
        out['bin_acc']=sum(a['prediction_binary']==g['prediction_binary'] for a,g in valid)/len(valid)
        out['mse']=sum((a['prediction']-g['prediction'])**2 for a,g in valid)/len(valid)
    for key,field in [('pros_rouge_scores','positive developments'),('cons_rouge_scores','potential concerns'),('anal_rouge_scores','analysis')]:
        scores=[scorer.score(g[field],a[field]) for a,g in valid]
        out[key]={n:sum(s[n].fmeasure for s in scores)/len(scores) if scores else None for n in ['rouge1','rouge2','rougeL']}
    return out

SYSTEM='You are a financial analyst. Use only supplied historical information. Return exactly these blocks: [Positive Developments]:, [Potential Concerns]:, [Prediction & Analysis]:. Inside the last block use Prediction: Up or Down by 0-1%, 1-2%, 2-3%, 3-4%, 4-5%, or more than 5%, then Analysis:. Give two short factors per list and a brief analysis. Keep the whole response under 180 words. Do not claim certainty.'

def make_prompt(raw,tokenizer,budget=896):
    # Convert the original Llama instruction wrapper to this model's chat format.
    body=raw.split('<</SYS>>',1)[-1].strip().removesuffix('[/INST]')
    body=body.replace('cryptocurrency price movement','stock price movement')
    forecast=re.search(r'all the information before (\d{4}-\d{2}-\d{2})',body)
    financial=re.search(r'Some recent basic financials .*?reported at (\d{4}-\d{2}-\d{2})',body)
    future_financial=bool(forecast and financial and financial[1]>forecast[1])
    if future_financial:
        body=re.sub(r'Some recent basic financials .*?(?=Based on all the information before)', '[Financial block removed: reporting period is after forecast origin]\n\n',body,flags=re.S)
    # Preserve both the company/context prefix and final financials/task suffix.
    ids=tokenizer.encode(body,add_special_tokens=False)
    original=len(ids)
    if original>budget:
        body=tokenizer.decode(ids[:budget//2],skip_special_tokens=True)+'\n[Middle context omitted for memory-limited experiment]\n'+tokenizer.decode(ids[-budget//2:],skip_special_tokens=True)
    text=tokenizer.apply_chat_template([{'role':'system','content':SYSTEM},{'role':'user','content':body}],tokenize=False,add_generation_prompt=True)
    return text,{'original_body_tokens':original,'body_budget':budget,'context_shortened':original>budget,'future_financial_block_removed':future_financial}

def compact_answer(raw,tokenizer):
    # For training only; evaluate against the FULL original reference.
    p=parse_answer(raw)
    if p is None:raise ValueError('Unparsable training reference')
    pred=re.search(r'Prediction:\s*(.*?)\s*Analysis:',raw,re.S).group(1).strip()
    def clip(text,n):return tokenizer.decode(tokenizer.encode(text.strip(),add_special_tokens=False)[:n],skip_special_tokens=True)
    return '[Positive Developments]:\n'+clip(p['positive developments'],100)+'\n\n[Potential Concerns]:\n'+clip(p['potential concerns'],100)+'\n\n[Prediction & Analysis]:\nPrediction: '+pred+'\nAnalysis: '+clip(p['analysis'],150)
