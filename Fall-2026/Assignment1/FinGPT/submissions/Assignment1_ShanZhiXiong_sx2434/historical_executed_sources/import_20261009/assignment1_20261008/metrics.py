"""Course metrics are unmodified; supplemental metrics clearly use different denominators."""
import importlib.util,json,math,re,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('course_utils',ROOT/'source_snapshot/course_utils.py')
course=importlib.util.module_from_spec(spec);spec.loader.exec_module(course)
def direction(text):
    words=str(text).lower()
    up=bool(re.search(r'\b(up|increase|rise|rising)\b',words))
    down=bool(re.search(r'\b(down|decrease|decline|fall|falling)\b',words))
    if up==down:return 0 if not up and re.search(r'\b(flat|unchanged|neutral)\b',words) else None
    return 1 if up else -1
def parse(text):
    pat=r'\[Positive Developments\]\s*:?\s*(.*?)\[Potential Concerns\]\s*:?\s*(.*?)\[Prediction\s*(?:&|and)\s*Analysis\]\s*:?\s*Prediction:\s*(.*?)\s*Analysis:\s*(.*)'
    m=re.search(pat,text,re.S|re.I)
    if not m:return {'format_complete':False,'direction':None}
    pros,cons,pred,anal=[x.strip() for x in m.groups()]
    return {'format_complete':bool(pros and cons and pred and anal),'direction':direction(pred),
       'prediction_text':pred,'positive_developments':pros,'potential_concerns':cons,'analysis':anal}

def forecast_direction(text):
    """Read explicit direction words in the final prediction field only.

    Broad words such as growth are deliberately excluded: they may describe
    business fundamentals rather than the forecast stock-price direction.
    """
    words=str(text).lower()
    if re.search(r'\b(?:mixed|unclear|uncertain)\s+(?:outlook|direction)\b',words):return None
    up=bool(re.search(r'\b(?:up|upwards?|upside|increase|increases|increasing|rise|rises|rising|gain|gains|gaining)\b',words))
    down=bool(re.search(r'\b(?:down|downwards?|downside|decrease|decreases|decreasing|decline|declines|declining|fall|falls|falling|drop|drops|dropping)\b',words))
    if up==down:return 0 if not up and re.search(r'\b(flat|unchanged|neutral)\b',words) else None
    return 1 if up else -1

def parse_final_answer(text):
    """Supplemental parser: ignore an explicit reasoning block and normalize headings.

    Never edits prediction content or replaces the unmodified course scores.
    An unclosed explicit think block has no final answer to parse.
    """
    if '</think>' in text:text=text.rsplit('</think>',1)[1]
    elif '<think>' in text:return {'format_complete':False,'direction':None}
    text=text.replace('**','').replace('__','')
    text=re.sub(r'(?m)^\s*(?:[-*]\s+|#{1,6}\s+)','',text)
    for label in ('Positive Developments','Potential Concerns','Prediction & Analysis'):
        pattern=re.escape(label).replace(r'\&',r'(?:&|and)')
        text=re.sub(r'(?im)^\s*\[?'+pattern+r'\]?\s*:?\s*$', '['+label+']:',text)
    parsed=parse(text)
    if 'prediction_text' in parsed:parsed['direction']=forecast_direction(parsed['prediction_text'])
    return parsed

def output_diagnostics(records):
    n=len(records);parsed=[parse_final_answer(r['generated']) for r in records]
    valid=[(r,p) for r,p in zip(records,parsed) if p['direction'] is not None]
    correct=sum(p['direction']==direction(r['label']) for r,p in valid)
    structural=sum(p['format_complete'] for p in parsed)
    return {'nonempty_count':sum(bool(r['generated'].strip()) for r in records),
      'natural_eos_count':sum(r['ended_eos'] for r in records),
      'length_limit_count':sum(r['hit_generation_limit'] for r in records),
      'normalized_structure_count':structural,
      'normalized_complete_count':sum(p['format_complete'] and r['ended_eos'] and not r['hit_generation_limit'] for r,p in zip(records,parsed)),
      'normalized_valid_direction_count':len(valid),'normalized_correct_count':correct,
      'normalized_accuracy_all_invalid_as_wrong':correct/n if n else None,
      'normalized_accuracy_valid_only':correct/len(valid) if valid else None,
      'note':'Supplemental heading normalization and final-prediction direction extraction; original course scores remain unchanged. Parsing success is not factual quality.'}
def summarize(records):
    generated=[r['generated'] for r in records];references=[r['reference'] for r in records]
    # Original parser behavior, range encoding, skipped pairs, and denominators preserved.
    strict=course.calc_metrics(generated,references)
    strict.setdefault('valid_count',0)
    n=len(records);correct=valid=complete=0;confusion={}
    for r in records:
        p=parse(r['generated']);truth=direction(r['label']);assert truth is not None
        valid+=p['direction'] is not None;correct+=p['direction']==truth
        complete+=bool(p['format_complete'] and r['ended_eos'] and not r['hit_generation_limit'])
        k=f'{truth}->{p["direction"]}';confusion[k]=confusion.get(k,0)+1
    times=[r['seconds'] for r in records]
    recalls=[]
    for truth in (-1,0,1):
        group=[r for r in records if direction(r['label'])==truth]
        if group:recalls.append(sum(parse(r['generated'])['direction']==truth for r in group)/len(group))
    return {'samples':n,'course_reference_metrics':strict,
      'course_valid_fraction':strict['valid_count']/n if n else None,
      'dataset_label_metrics':{'valid_direction_count':valid,'correct_count':correct,
         'accuracy_all_invalid_as_wrong':correct/n if n else None,'accuracy_valid_only':correct/valid if valid else None,
         'balanced_accuracy_all_invalid_as_wrong':statistics.mean(recalls) if recalls else None,'confusion':confusion},
      'output_metrics':{'complete_count':complete,'complete_fraction':complete/n if n else None,
         'length_limit_count':sum(r['hit_generation_limit'] for r in records)},
      'output_diagnostics':output_diagnostics(records),
      'inference':{'mean_seconds':statistics.mean(times) if times else None,'median_seconds':statistics.median(times) if times else None,
         'total_seconds':sum(times),'total_generated_tokens':sum(r['generated_tokens'] for r in records),
         'tokens_per_second_total':sum(r['generated_tokens'] for r in records)/sum(times) if times and sum(times) else None},
      'limitation':'Course scores compare parsable reference-answer pairs, including its original range parser; label scores compare dataset labels, not independently fetched returns. ROUGE does not establish factual quality.'}
def baselines(train,rows):
    counts={d:sum(direction(r['label'])==d for r in train) for d in (-1,0,1)}
    majority=max(counts,key=counts.get)
    return {'training_direction_counts':counts,'train_majority_direction':majority,
      'always_up_accuracy':sum(direction(r['label'])==1 for r in rows)/len(rows),
      'train_majority_accuracy':sum(direction(r['label'])==majority for r in rows)/len(rows)}
