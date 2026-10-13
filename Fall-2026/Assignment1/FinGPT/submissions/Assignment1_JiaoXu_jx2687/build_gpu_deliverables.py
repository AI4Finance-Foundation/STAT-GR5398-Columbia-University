"""Build review artifacts ONLY after the selected Llama 8B run pass full validation."""
from pathlib import Path
import csv,html,json,subprocess,sys,statistics
import nbformat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.platypus import SimpleDocTemplate,Paragraph,Table,Spacer,PageBreak,Image
from utils import parse_answer
ROOT=Path(__file__).resolve().parent
NAMES=['llama31_8b']
LABELS={'deepseek8b':'DeepSeek-R1-Distill-Llama-8B','llama31_8b':'Llama-3.1-8B'}

def fmt(x):return 'N/A' if x is None else f'{x:.4f}'

def main():
    runs={};combined=[]
    for name in NAMES:
        r=ROOT/'cloud_runs'/name
        assert json.loads((r/'results/job_status.json').read_text())['status']=='complete'
        subprocess.run([sys.executable,str(ROOT/'cloud/validate_cloud_run.py'),str(r)],check=True)
        d={'training':json.loads((r/'results/training.json').read_text()),'metrics':json.loads((r/'results/metrics.json').read_text()),'validation':json.loads((r/'results/full_validation.json').read_text()),'provenance':json.loads((r/'provenance.json').read_text()),'environment':json.loads((r/'results/cloud_environment.json').read_text())}
        d['single_request_benchmark']=json.loads((r/'results/single_request_benchmark.json').read_text())
        assert {x['variant'] for x in d['single_request_benchmark']}=={'base','lora'}
        for x in d['single_request_benchmark']:
            assert x['n']==6 and x['sample_indices']==[0,9,70,149,230,299]
            assert x['mean_single_request_s']>0 and x['median_single_request_s']>0
        d['answers']={v:[json.loads(l) for l in (r/'results'/f'{v}_answers.jsonl').read_text().splitlines()] for v in ['base','lora']}
        for m in d['metrics']:
            rows=d['answers'][m['model']]
            label_correct=sum((p:=parse_answer(x['generated_answer'])) is not None and p['prediction_binary']==(1 if x['label'].startswith('up') else -1) for x in rows)
            combined.append({'architecture':LABELS[name],**m,'label_direction_accuracy_all_rows':label_correct/300})
        runs[name]=d
    (ROOT/'results/gpu_metrics.json').write_text(json.dumps(combined,indent=2)+'\n')
    flat=[]
    for m in combined:
        row={k:v for k,v in m.items() if not isinstance(v,dict)}
        for section in ['pros_rouge_scores','cons_rouge_scores','anal_rouge_scores']:
            for metric,v in m[section].items():row[section+'_'+metric]=v
        flat.append(row)
    keys=list(dict.fromkeys(k for row in flat for k in row))
    with open(ROOT/'results/gpu_metrics.csv','w') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(flat)
    # Deterministic cross-company/week examples, never selected for favorable metrics.
    selected=[0,9,70,149,230,299]
    worksheet=ROOT/'HITL_review_Llama.csv'
    if not worksheet.exists():
        fields=['architecture','sample_index','symbol','period','label','base_parse_valid','lora_parse_valid','input_prompt','base_answer','lora_answer','reference_answer','human_grounding_0_2','human_reasoning_0_2','human_clarity_0_2','human_notes']
        with open(worksheet,'w') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
            for name,d in runs.items():
                for i in selected:
                    b=d['answers']['base'][i];l=d['answers']['lora'][i]
                    w.writerow({'architecture':LABELS[name],'sample_index':i,'symbol':b['symbol'],'period':b['period'],'label':b['label'],'base_parse_valid':b['parse_valid'],'lora_parse_valid':l['parse_valid'],'input_prompt':b['input_prompt'],'base_answer':b['generated_answer'],'lora_answer':l['generated_answer'],'reference_answer':b['reference_answer']})
    figdir=ROOT/'figures';figdir.mkdir(exist_ok=True)
    x=range(len(combined));labels=[m['architecture'].replace('DeepSeek-R1-Distill-Llama-8B','DeepSeek8B').replace('Llama-3.1-8B','Llama3.1-8B')+'\n'+m['model'] for m in combined]
    fig,axes=plt.subplots(1,2,figsize=(10,3.1),layout='constrained')
    axes[0].bar(x,[100*m['coverage'] for m in combined],color=['#9ba8b8','#326b8d']*2)
    axes[0].set_title('Strict paired parsing coverage');axes[0].set_ylabel('% of 300 test cases');axes[0].set_ylim(0,105)
    axes[1].bar(x,[100*m['label_direction_accuracy_all_rows'] for m in combined],color=['#9ba8b8','#326b8d']*2)
    benchmark=sum(r['label'].startswith('up') for r in runs[NAMES[0]]['answers']['base'])/300
    axes[1].axhline(benchmark*100,color='#c67846',linestyle='--',label='Fixed train-majority always-up')
    axes[1].set_title('Label direction accuracy, all cases');axes[1].set_ylim(0,105);axes[1].legend(fontsize=7)
    for ax in axes:ax.set_xticks(list(x),labels,fontsize=8);ax.spines[['top','right']].set_visible(False)
    fig.savefig(figdir/'gpu_coverage_accuracy.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,2.8),layout='constrained')
    for name,d in runs.items():
        log=[x for x in d['training']['train_log'] if 'loss' in x]
        ax.plot([x['epoch'] for x in log],[x['loss'] for x in log],marker='.',label=LABELS[name])
    ax.set(xlabel='Epoch',ylabel='Logged training loss',title='One epoch on the fixed 60-row subset');ax.legend(fontsize=8);ax.spines[['top','right']].set_visible(False)
    fig.savefig(figdir/'gpu_training_loss.png',dpi=180);plt.close(fig)
    styles=getSampleStyleSheet();styles.add(ParagraphStyle(name='BodySmall',fontSize=9.5,leading=13,spaceAfter=7));styles.add(ParagraphStyle(name='Small',fontSize=8,leading=10,spaceAfter=5))
    story=[]
    def para(s,style='BodySmall'):story.append(Paragraph(s,styles[style]))
    def table(rows,widths):
        t=Table([[Paragraph(html.escape(str(v)),styles['Small']) for v in row] for row in rows],colWidths=widths,repeatRows=1)
        t.setStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8eff4')),('LINEBELOW',(0,0),(-1,-1),.25,colors.HexColor('#ccd5de')),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)])
        story.extend([t,Spacer(1,8)])
    para('FinGPT Forecaster: Llama-3.1-8B LoRA experiment','Title')
    para('STAT GR5398 Fall 2026 - Assignment 1 | Actual executed results','Heading2')
    para('This report covers Llama-3.1-8B only, by the student’s explicit scope decision on October12. The second-model requirement (DeepSeek-R1-Distill-Llama-8B training and full comparison) is INCOMPLETE. DeepSeek training was completed, but its evaluation was stopped with160base/0adapter outputs saved; its follow-on queue was cancelled. No waiver or instructor acceptance is assumed. Qwen7B was also omitted. The student has reviewed the work and published the Medium article; numeric HITL ratings remain blank. Formal submission is being prepared for JIAO XU (jx2687).')
    para('1. Results and their meaning','Heading2')
    table([['Architecture / variant','Valid / 300','Bin acc*','MSE*','Label acc**']]+[[m['architecture']+' / '+m['model'],m['valid_count'],fmt(m['bin_acc']),fmt(m['mse']),fmt(m['label_direction_accuracy_all_rows'])] for m in combined],[200,70,75,85,82])
    para('* Course-compatible direction accuracy and signed-bin midpoint MSE, conditional on both reference and answer parsing. MSE units are percentage-points squared, not raw return error. Empty valid sets are N/A. **Direction against provided labels, using all300 cases and penalizing invalid answers; this extra metric does not require the reference to parse. Baseline always-up accuracy: '+fmt(benchmark)+'.','Small')
    story.extend([Image(str(figdir/'gpu_coverage_accuracy.png'),width=510,height=158),Spacer(1,8)])
    para('These are retrospective imitation/formatting and teacher-overlap results. They do not establish unbiased forecasting, financial sentiment classification, trading returns or profitability. Teacher generation and pretraining contamination cannot be excluded. A single seed, small training subset and correlated ticker-week outcomes limit generalization.')
    story.append(PageBreak())
    para('2. Hardware, data and training','Heading2')
    para('User-provided Google Colab Tesla T4, 15,360MiB GPU memory, about12.67GB system RAM and112.64GB disk. Python3.13.15; torch2.11.0+cu130. GPU/CUDA execution verified. The student purchased100Pay-As-You-Go compute units for$9.99; no purchase was made by the assistant. Persistent results and model cache were saved to an independent Google Drive folder. Each job had its own working directory; jobs ran serially and did not touch the FinRL project or Positron kernels. Environment manifests: cloud_runs/&lt;model&gt;/results/cloud_environment.json.')
    para('Pinned dataset: FinGPT/fingpt-forecaster-dow30-202305-202405, revision4b234310115a6f2cb5806dcd95f801b60d177882. Original1230train/300test across30 companies. Actual experiment:60training rows (2fixed-seed historical rows per ticker),30time-separated validation rows, all300test rows. Training subset spans only2target weeks; test spans10weekly periods from2024-02-25 through2024-05-05. Validation is from original training data; no test-based hyperparameter selection.')
    para('No missing required fields, duplicate prompt rows or cross-split ticker-period overlap were found. Original directions: train687up/543down, test137up/163down. One training and3test teacher references fail the parser. Obvious future-dated financial blocks in64train/2test source rows were removed from model inputs. Filing availability, news timestamps and historical company-profile snapshots are missing. Teacher explanations may condition on realized future labels; exact released-data generation provenance is not fully documented. Labels/returns were not independently reconstructed from raw prices.')
    para('Actual resource adaptation:4-bit NF4 double quantization with fp16 compute and rank8 LoRA (alpha16, dropout0.1) on q/k/v/o and gate/up/down projections. Batch1, accumulation4, learning rate5e-5, constant scheduler, warmup0.03, one epoch, seed5398, gradient checkpointing. Prompt body896tokens; sequence cap2048; teacher training targets shortened while preserving prediction and all3blocks. Evaluation references remain unmodified. The first8-bit loading attempt failed; its precise error was lost after a runtime reset. DeepSpeed was not used; TensorBoard was used. These deviations must not be represented as exact compliance with the original8-bit/DeepSpeed recipe.')
    table([['Architecture','Train loss','Val loss','Train wall s','Changed tensors']]+[[LABELS[n],fmt(d['training']['training_metrics']['train_loss']),fmt(d['training']['validation_metrics']['eval_loss']),f"{d['training']['train_runtime_wall_s']:.1f}",d['training']['changed_adapter_tensors']] for n,d in runs.items()],[210,65,65,85,87])
    para('All saved adapter tensors are finite; updates were verified and requires_grad was restricted to adapters. Frozen-weight hashing covers only the first4096elements of one frozen parameter, not every base weight. Quantized tensor numel counts packed storage and must not be treated as the original dense8B parameter count. Exact hashes and recomputation: results/full_validation.json in each run.','Small')
    story.append(PageBreak())
    para('3. Evaluation, ROUGE and timing','Heading2')
    para('Greedy decoding,512new-token limit, identical saved compressed inputs, base versus its own adapter. Responses are scored exactly as generated; failed formatting is never repaired. Batched decoding uses left padding and SDPA; prior DeepSeek base cases used single-example eager attention. OOM fallback reduces batch size if needed. Batch wall time is GPU-synchronized; elapsed_s for a batched row is amortized batch wall time / batch size, not single-request latency. Loading and warmup are excluded. Mixed overall means cannot establish a speed improvement; comparable_latency matches identical batch groups and attention implementation.')
    table([['Architecture / section','ROUGE1 F1*','ROUGE2 F1*','ROUGEL F1*']]+[[m['architecture']+' / '+m['model']+' / '+label,*[fmt(m[key][k]) for k in ['rouge1','rouge2','rougeL']]] for m in combined for key,label in [('pros_rouge_scores','pros'),('cons_rouge_scores','cons'),('anal_rouge_scores','analysis')]],[267,81,82,82])
    para('* Only paired parse-valid answers/references; ROUGE measures lexical F1 overlap and is not grounding, factuality or reasoning quality. Token-cap hits, coverage and latency strata are saved for every variant.','Small')
    table([['Architecture','Matched cases','Base amortized s','LoRA amortized s']]+[[LABELS[n],d['validation']['comparable_latency']['matched_case_count'],fmt(d['validation']['comparable_latency'].get('base')),fmt(d['validation']['comparable_latency'].get('lora'))] for n,d in runs.items()],[225,80,103,104])
    para('Separate single-request timing: six fixed cases per variant, batch1 SDPA, synchronized prefill plus decoding, model loading and warmup excluded. These timings are observational and do not establish statistically significant speed differences.','Small')
    table([['Architecture / variant','Cases','Mean request s','Median request s']]+[[LABELS[n]+' / '+x['variant'],x['n'],fmt(x['mean_single_request_s']),fmt(x['median_single_request_s'])] for n,d in runs.items() for x in d['single_request_benchmark']],[250,50,105,107])
    story.append(PageBreak())
    para('4. Review, reproducibility and submission boundaries','Heading2')
    story.extend([Image(str(figdir/'gpu_training_loss.png'),width=510,height=177),Spacer(1,8)])
    para('HITL_review_Llama.csv contains6deterministic paired examples for Llama, including raw base/adapter answers, inputs and references. Human grounding, reasoning and clarity fields remain blank. AI-assisted summaries cannot be counted as the student’s human review. Training loss measures imitation of shortened targets, not market forecasting accuracy.')
    para('The original Assignment1 explicitly permits multiple tracks with separate submission folders. It does not establish that FinGPT work counts toward this student’s registered or graduation credits. The semester-level research platform, new labeling strategy, social-data integration and near-real-time market validation are not automatically mandatory Assignment1 deliverables. Self-collecting NASDAQ100 data is optional in the FinGPT assignment.')
    para('Reproduction: scripts, exact commands and model revisions are listed in README.md, results/COMMANDS.md and cloud_runs/&lt;model&gt;/provenance.json. All300raw responses per variant are stored under each run. The local Qwen0.5B experiment is auxiliary and is not presented as a specified8B model. Student: JIAO XU (jx2687). Medium publication is complete (URL in README). Numeric HITL ratings remain blank; additional course AI policy is unverified.')
    for title,url in [('Assignment1 original','https://github.com/AI4Finance-Foundation/STAT-GR5398-Columbia-University/tree/master/Fall-2026/Assignment1'),('FinGPT track original','https://github.com/AI4Finance-Foundation/STAT-GR5398-Columbia-University/tree/master/Fall-2026/Assignment1/FinGPT'),('Official Forecaster','https://github.com/AI4Finance-Foundation/FinGPT/tree/master/fingpt/FinGPT_Forecaster'),('Dataset','https://huggingface.co/datasets/FinGPT/fingpt-forecaster-dow30-202305-202405'),('DeepSeek model','https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B'),('Llama model','https://huggingface.co/meta-llama/Llama-3.1-8B')]:para(f'<link href="{url}">{title}</link>','Small')
    report=ROOT/'Research_Report.pdf';old=ROOT/'Local_Qwen_Research_Report.pdf'
    if report.exists() and not old.exists():old.write_bytes(report.read_bytes())
    def footer(c,d):c.setFont('Helvetica',8);c.drawString(48,24,'GR5398 Assignment1 | JIAO XU (jx2687) | Actual8B results');c.drawRightString(564,24,str(d.page))
    SimpleDocTemplate(str(report),pagesize=(612,792),leftMargin=48,rightMargin=48,topMargin=42,bottomMargin=42).build(story,onFirstPage=footer,onLaterPages=footer)
    nb=nbformat.v4.new_notebook();cells=[]
    cells.append(nbformat.v4.new_markdown_cell('# FinGPT Assignment1: executed8B review\n\nLlama only,300test rows per variant. The required second-model experiment remains incomplete by student decision. Medium publication is complete. Student: JIAO XU (jx2687). Numeric HITL ratings remain blank. Model-count and resource adaptations: Research_Report.pdf and REQUIREMENTS_CHECKLIST.md.'))
    cells.append(nbformat.v4.new_code_cell("from pathlib import Path\nimport json,pandas as pd\nfrom IPython.display import display,Image\nROOT=Path.cwd()\nmetrics=json.loads((ROOT/'results/gpu_metrics.json').read_text())\ndisplay(pd.read_csv(ROOT/'results/gpu_metrics.csv'))"))
    cells.append(nbformat.v4.new_code_cell("display(Image(filename=str(ROOT/'figures/gpu_coverage_accuracy.png')))\ndisplay(Image(filename=str(ROOT/'figures/gpu_training_loss.png')))"))
    cells.append(nbformat.v4.new_code_cell("import subprocess,sys\nfor name in ['llama31_8b']:\n    subprocess.run([sys.executable,str(ROOT/'cloud/validate_cloud_run.py'),str(ROOT/'cloud_runs'/name)],check=True)\n    print(name,json.loads((ROOT/'cloud_runs'/name/'results/full_validation.json').read_text()))"))
    cells.append(nbformat.v4.new_markdown_cell('ROUGE is paired-reference F1 overlap. Missing timestamps, possible hindsight-conditioned teachers and pretraining overlap prevent prospective forecasting claims. Batch timings are amortized, not single-request latency.'))
    cells.append(nbformat.v4.new_code_cell("review=pd.read_csv(ROOT/'HITL_review_Llama.csv')\ndisplay(review[['architecture','symbol','period','label','base_parse_valid','lora_parse_valid']])\nprint('Human review ratings must be provided by the student; no AI score is presented as human review.')"))
    nb.cells=cells;nb.metadata={'kernelspec':{'name':'fingpt-isolated','display_name':'FinGPT isolated experiment','language':'python'},'language_info':{'name':'python','version':'3.12.14'}}
    nbformat.validate(nb);nbformat.write(nb,ROOT/'FinGPT_Assignment1_GPU_Review.ipynb')
    print('Built actual GPU report, charts, metric table, HITL worksheet and review notebook.')
if __name__=='__main__':main()
