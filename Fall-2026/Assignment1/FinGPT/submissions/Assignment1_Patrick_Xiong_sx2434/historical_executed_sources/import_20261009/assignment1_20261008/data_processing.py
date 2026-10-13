"""Deterministic input compression, without answer truncation or target leakage."""
import re

SYSTEM = ('Analyze the stock using only the supplied historical evidence. Give 2-4 concise positive developments and concerns. '
          'Use exactly: [Positive Developments]: ... [Potential Concerns]: ... '
          '[Prediction & Analysis]: Prediction: Up/Down by a percentage range. Analysis: supporting reasoning and risks.')

def tokenize_prompt(tok, text):
    return tok.apply_chat_template([{'role':'system','content':SYSTEM},{'role':'user','content':text}],
                                  tokenize=True,add_generation_prompt=True)

def parts(row):
    raw=row['prompt']
    intro=re.search(r'\[Company Introduction\]:\s*(.*?)\n\nFrom ',raw,re.S)
    name=re.match(r'(.+?) is ',intro.group(1)).group(1) if intro and re.match(r'(.+?) is ',intro.group(1)) else row['symbol']
    sector=re.search(r'in the (.+?) sector',intro.group(1)) if intro else None
    prices=re.findall(r"From (\d{4}-\d{2}-\d{2}) to (\d{4}-\d{2}-\d{2}), (.+?)'s stock price (.+?)\. News during this period",raw)
    forecast=re.search(r'next week \((\d{4}-\d{2}-\d{2}) to (\d{4}-\d{2}-\d{2})\)',raw)
    if not forecast:raise ValueError('Forecast dates missing')
    # The core is never clipped: identity, forecast horizon, and latest price move.
    core=f"Stock: {row['symbol']} ({name}). Forecast: {forecast.group(1)} to {forecast.group(2)}.\n"
    if prices:
        p=prices[-1];core+=f"Latest history {p[0]} to {p[1]}: {p[3]}.\n"
    else:raise ValueError('Historical price movement missing')
    news=[]
    current_period=''
    for block in re.split(r'(?=From \d{4}-\d{2}-\d{2} to )',raw):
        period=re.match(r'From (\d{4}-\d{2}-\d{2}) to (\d{4}-\d{2}-\d{2})',block)
        if period:current_period=period.group(2)
        for item in re.finditer(r'\[Headline\]:\s*(.*?)\n\[Summary\]:\s*(.*?)(?=\n\n|$)',block,re.S):
            news.append({'date':current_period,'headline':item.group(1).strip(),'summary':item.group(2).strip()})
    financial=re.search(r'\[Basic Financials\]:\s*(.*?)(?:\n\nBased on|$)',raw,re.S)
    fields={}
    if financial:
        for line in financial.group(1).splitlines():
            if ':' in line:
                key,value=line.split(':',1);fields[key.strip()]=value.strip()
    return core,prices,news,fields,sector.group(1) if sector else None

def compress(tok,row,budget):
    core,prices,news,fields,sector=parts(row)
    text=core
    if len(tokenize_prompt(tok,text))>budget:
        return None,{'reason':'essential_prompt_plus_full_answer_exceeds_context','required_prompt_tokens':len(tokenize_prompt(tok,text))}
    retained=[];financial_retained=[];summary_count=0
    def add(line):
        nonlocal text
        candidate=text+line
        if len(tokenize_prompt(tok,candidate))<=budget:text=candidate;return True
        return False
    # Add at least one short news headline first, then earlier prices and basic financials.
    # The input-only ordering never consults the answer or direction label.
    ordered=sorted(enumerate(news),key=lambda x:(x[1]['date'],row['symbol'].lower() in x[1]['headline'].lower()),reverse=True)
    if ordered:
        i,item=ordered[0]
        title=tok.decode(tok.encode(item['headline'],add_special_tokens=False)[:24])
        if add(f"News {item['date']}: {title}\n"):retained.append(i)
    if sector:add(f"Sector: {sector}.\n")
    for p in reversed(prices[:-1]):add(f"History {p[0]} to {p[1]}: {p[3]}.\n")
    for key in ('eps','peTTM','grossMargin','operatingMargin','roeTTM','totalDebtToEquity','currentRatio','bookValue'):
        if key in fields and add(f"{key}: {fields[key]}\n"):financial_retained.append(key)
    for i,item in ordered:
        if i in retained:continue
        title=tok.decode(tok.encode(item['headline'],add_special_tokens=False)[:24])
        if add(f"News {item['date']}: {title}\n"):retained.append(i)
    # Summaries receive remaining space after headlines, rather than displacing task or targets.
    for i in retained:
        summary=tok.decode(tok.encode(news[i]['summary'],add_special_tokens=False)[:32])
        if add(f"Detail: {summary}\n"):summary_count+=1
    ids=tokenize_prompt(tok,text)
    assert len(ids)<=budget
    return ids,dict(prompt_tokens=len(ids),original_news=len(news),retained_news=len(retained),
                    retained_summaries=summary_count,retained_financial_fields=financial_retained,
                    original_prompt_tokens=len(tok.encode(row['prompt'],add_special_tokens=False)),compressed_text=text)

def prepare(tok,rows,max_length):
    encoded=[];audit=[];rejected=[]
    for i,row in enumerate(rows):
        answer=tok.encode(row['answer'].strip(),add_special_tokens=False)
        target=answer+[tok.eos_token_id]
        prompt,info=compress(tok,row,max_length-len(target))
        info.update(index=i,symbol=row['symbol'],answer_tokens=len(answer),answer_truncated=False)
        if prompt is None:rejected.append(info);continue
        labels=[-100]*len(prompt)+target
        full=prompt+target
        assert len(full)<=max_length and full[-len(target):]==target
        info['sequence_tokens']=len(full)
        encoded.append(dict(input_ids=full,attention_mask=[1]*len(full),labels=labels,source_index=i))
        audit.append(info)
    return encoded,audit,rejected
