"""Build final report only after every prescribed evaluation has completed."""
from pathlib import Path
import csv,json,html,os
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.shapes import Drawing,String
r=Path(__file__).resolve().parents[2];o=r/'outputs/retraining';pkg=o/'submissions/Assignment2_XinruXiao_xx2524';pkg.mkdir(parents=True,exist_ok=True)
assert json.loads((o/'training_status.json').read_text())['status']=='complete'
for v in ['course','retrained_baseline','retrained_improved']:assert (r/'work/retraining/evaluation'/v/'output/evaluation_complete.json').exists(),v

def read(name):
 with (o/'analysis'/name).open() as f:return list(csv.DictReader(f))
metrics=read('portfolio_metrics.csv');coverage=read('valuation_coverage.csv');signals=read('signal_diagnostics.csv');paired=read('paired_valuation_effect.csv')
log=json.loads((pkg/'test_run_ledger_summary.json').read_text())
v2_dir=o/'feature_ablation_v2'
v2_results=json.loads((v2_dir/'comparison_metrics.json').read_text())
v2_integrity=json.loads((v2_dir/'final_integrity.json').read_text())
v3_dir=o/'rank_target_v3'
v3_decision=json.loads((v3_dir/'dev_val_decision.json').read_text())
v3_integrity=json.loads((v3_dir/'final_integrity.json').read_text())
pdfmetrics.registerFont(TTFont('CJK',os.environ.get('REPORT_CJK_FONT','/System/Library/Fonts/Supplemental/Arial Unicode.ttf')))
s=getSampleStyleSheet();s.add(ParagraphStyle(name='CN',fontName='CJK',fontSize=10,leading=15,spaceAfter=8,wordWrap='CJK'));s.add(ParagraphStyle(name='HeadCN',parent=s['CN'],fontSize=15,leading=21,textColor=colors.HexColor('#173A56'),spaceAfter=12));s.add(ParagraphStyle(name='TitleCN',parent=s['CN'],fontSize=23,leading=30,spaceAfter=18));s.add(ParagraphStyle(name='SmallCN',parent=s['CN'],fontSize=7.5,leading=10))
story=[];md=[]
def para(text,style='CN'):
 story.append(Paragraph(html.escape(str(text)).replace('\n','<br/>'),s[style]));md.append(str(text)+'\n')
def heading(text):para(text,'HeadCN')
def table(data,widths):
 t=Table([[Paragraph(html.escape(str(c)),s['SmallCN']) for c in row] for row in data],colWidths=widths,repeatRows=1,hAlign='LEFT');t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E8F0F5')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.5,colors.HexColor('#6B879B')),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F6F8FA')])]));story.append(t);story.append(Spacer(1,9));md.append('\n'.join(' | '.join(map(str,row)) for row in data)+'\n')
def page():story.append(PageBreak())
def n(x,d=2):
 try:return f'{float(x):.{d}f}' if x!='' else '未成熟/无值'
 except (ValueError,TypeError):return str(x)
def pick(v,p,group):return next(x for x in metrics if x['variant']==v and x['period']==p and '_'+group+'_' in x['run'])
def v2pick(p,group):return v2_results[p][group]['v2']
def nav_chart(period):
 series=[];dates=None
 for group in ['A','B']:
  row=pick('course',period,group)
  with (r/'work/retraining/evaluation/course/output'/row['run']/'equity.csv').open() as f:data=list(csv.DictReader(f))
  current_dates=[x[''][:10] for x in data]
  if dates is None:dates=current_dates
  assert dates==current_dates
  first=float(data[0]['equity']);series.append([float(x['equity'])/first*100 for x in data])
 curves=read('benchmark_curves.csv')
 for benchmark in ['SPY','QQQ']:
  values={x['date']:float(x['normalized_value']) for x in curves if x['period']==period and x['benchmark']==benchmark}
  series.append([values[d] for d in dates])
 drawing=Drawing(508,255);plot=LinePlot();plot.x=42;plot.y=40;plot.width=440;plot.height=185
 plot.data=[[(i,value) for i,value in enumerate(line)] for line in series]
 palette=['#195D9D','#D16D22','#7C8995','#8257A5']
 for i,c in enumerate(palette):plot.lines[i].strokeColor=colors.HexColor(c);plot.lines[i].strokeWidth=1.0
 plot.lines[2].strokeDashArray=[4,2];plot.lines[3].strokeDashArray=[1,2]
 plot.xValueAxis.valueMin=0;plot.xValueAxis.valueMax=len(dates)-1
 plot.xValueAxis.valueSteps=sorted(set(round(i*(len(dates)-1)/4) for i in range(5)))
 plot.xValueAxis.labelTextFormat=lambda v:dates[min(len(dates)-1,max(0,int(round(v))))]
 plot.xValueAxis.labels.fontName='CJK';plot.xValueAxis.labels.fontSize=7
 plot.yValueAxis.valueMin=0;plot.yValueAxis.labels.fontName='CJK';plot.yValueAxis.labels.fontSize=7
 drawing.add(plot)
 for i,label in enumerate(['A选股','B加估值','SPY','QQQ']):drawing.add(String(55+i*115,10,label,fontName='CJK',fontSize=8,fillColor=colors.HexColor(palette[i])))
 drawing.add(String(42,240,period.upper()+' | NAV: first saved close = 100',fontName='CJK',fontSize=9,fillColor=colors.HexColor('#173A56')))
 story.append(drawing);md.append('图：'+period+'净值曲线；源为保存的equity.csv与benchmark_curves.csv。\n')
variants=[('course','课程分数：主实验'),('retrained_baseline','同环境原版重训：对照'),('retrained_improved','训练流程改进：bonus')]
para('成长科技股的估值过滤是否有效？','TitleCN');heading('全候选估值、四时期评价与选股重训')
para('Xinru Xiao | UNI xx2524 | 单人项目\nSTAT GR5398 · Fall 2026 · Assignment 2 · Valuation Track')
para('课程固定版本：f5df2feab002b4d463fdd3bf30c34f6d91c0e49a\n扩展研究冻结版本：1ba67615924078d9680702870352496583672341','SmallCN')
para('主实验使用课程原始选股分数；估值检查全部当周growth_tech候选，择时full，不开发情绪，不启用止损。重训是单独bonus，不替代课程评分基准。所有历史数字来自保存的确定性结果，不调用LLM生成财务输入。')
para('核心问题是估值模块的边际贡献，而非寻找最好看的收益。课程分数下，开发期过滤后表现变差，验证期改善很小，测试期改善；前向短样本仍有很大损失。不同阶段不一致，不能据此证明稳定Alpha。')
para('重训中的“改进版”指训练流程修订，不代表投资表现更好：它在验证期改善，但测试期累计收益和Sharpe低于同环境原版。两层研究均保留负面证据，不按测试表现重新选择配置。')
para('追加的特征消融先只看开发期和验证期并冻结版本，再各运行一次测试期和短forward。它删除五个可用时点尚未核实的历史估值倍数；验证期和测试期A组改善，但开发期退步、短forward仍大亏，不能称为稳定Alpha，也不替代课程分数。')
para('进一步的v3把绝对收益标签改为同日期、同bucket内的横截面排名。它完整训练119个月，但因开发期A只有153.20%，未达到测试前预登记的177.92%底线，因此没有运行v3的新test或forward；最终保留v2。')
t=[['阶段','A累计%','B累计%','B-A百分点','A回撤%','B回撤%']]
for p in ['dev','val','test','forward']:
 a,b=pick('course',p,'A'),pick('course',p,'B');t.append([p,n(a['total_return_pct']),n(b['total_return_pct']),n(float(b['total_return_pct'])-float(a['total_return_pct'])),n(a['max_drawdown_pct']),n(b['max_drawdown_pct'])])
table(t,[58,85,85,100,90,90])
para('原top20实验与结论被完整保留。新实验发生在原测试期已被查看之后，因此新增test和forward历史回放属于扩展回顾性证据，不能重新标称严格未见样本。')
page();heading('1. 研究设计与固定规则')
para('顺序：课程选股模型产生排序候选 → 对所有当时growth_tech候选计算估值并过滤 → 课程分配器递补并配置10只股票 → full择时 → 下周首个交易日开盘执行。估值本身不是先从所有市场股票中任选10只。股票池仍受课程指数成员、行业类别和模型分数覆盖约束。')
para('每周股票组合用全部候选；每月官方信号评价仍给top20，另外给all_growth诊断。全覆盖指检查全部可得候选，不指所有股票均有有效估值。缺数据、方法不足或分歧较大时保留，绝不填入猜测数字。')
para('综合至少两方法有效；中值取有效方法中值的中位数，区间取有效方法区间并集。只有价格高于综合上界且method_spread≤3才剔除。分歧为最高/最低方法中值。并集区间不是校准后的未来价格置信区间。')
para('A=选股，B=选股+估值，C=选股+full，D=选股+估值+full；本研究A=C且B=D是预期基准结果。课程规则：约95%目标资金投入、每周调仓、整股、单边6bp成本、无做空无杠杆。未为提高收益增加止损或情绪策略。')
para('期间：dev 2017-01-03至2021-12-31；val 2022-01-03至2023-12-29；test 2024-01-02至2026-10-09；forward可用2026-10-12至2026-11-06。119个选股月份覆盖2017-01至2026-11。信号表跨期间可重复使用2026-10月，不能简单把各期月份数相加当独立样本数。')
page();heading('2. 三种估值方法、假设与来源')
table([['项','来源/计算','固定假设或边界'],['DCF收入','连续四季每股收入，历史八季同比','初始增速裁剪-10%至30%；十年衰减至2.5%'],['利润/再投资','四季EBITDA及D&A比例中位数；CapEx绝对值/收入','增量NWC=0；税率0至35%；不改课程DCF引擎'],['WACC','DGS10/100；对SPY周收益Beta再Blume调整','ERP5%；债务利差2%；账面debt_to_capital近似'],['DCF范围','引擎敏感性网格','WACC上下1个百分点；g=2%、2.5%、3%'],['历史EV/EBITDA','过去36个完整月末的价格与当时可知财报','至少12个有效快照；P25/P50/P75'],['同行P/E','当时同板块指数成员，剔除自身，正TTM EPS','至少5家；P/E在0至200；P25/P50/P75']],[85,235,188])
para('流量每股数字按四季求和，余额取最新一期，比例不求和。shares_outstanding=1，净债务=每股有息债务−现金。历史企业价值按当时价格加净债务构造，EBITDA由各季收入×利润率求和。不得使用数据库未经核验的pe/ev_multiple、当前共识或未来数据回填。')
para('Beta最多五年周频数据，至少52周；早期没有完整五年价格，不能说所有估计都用了五年。ERP、信用利差、终值增长、零NWC及3倍分歧阈值均为披露的研究选择，不是假装观测事实。')
para('完整常数、边界、来源和质量标签另列于assumptions_and_quality.csv。标签区分观测输入、估计、账面代理、固定研究假设和范围裁剪，不代表正确概率或已经通过经济验证。')
para('DCF引擎终值会重算CapEx/D&A，可能造成不经济的FCFF跳升；遵守不改引擎要求并披露。账面债务权重不是市场价值权重。广义同行板块不保证业务可比，亏损公司无法用正P/E。')
para('引擎的具体终值规则：anchor=min(历史D&A比例,历史CapEx比例)，终值D&A=末年收入×anchor，终值CapEx=D&A×(1+g)，终值EBIT也扣除同一anchor。这是维护投资近似，不是完整资产滚动或ROIC约束。引擎还要求WACC−g至少1.5个百分点、终值FCF和股权价值为正；不满足则该方法失效，敏感性无效格点留空，不强行给出价格。')
page();heading('3. 数据与前视审计')
para('学生函数仅通过被冻结日期的DataView取数据；守卫禁止函数内直接读文件、联网或调用LLM。财报可知日期优先accepted_date，再filing_date，再保守滞后；四季必须连续。历史快照不得用晚于快照的修订版回填，缺旧版本会损失覆盖。')
para('已知限制：分红复权价格与财务每股数据的口径未完全核验；真实财季、修订历史、行业成员历史及盘中可知时刻未完全还原。DataView日期截断本身不能证明所有上游数据没有修订或幸存者偏差。')
para('KEYS教学审计：课程2018-12-31标签行的负EPS与公司真实截至2018-10-31财季亏损吻合，不能把课程标签直接当真实财政截止日。官方2018Q4公告与10-K有7.09亿美元商誉减值。中位数利润率正常化与全年加权利润率差异大，说明DCF依赖盈利假设；外部资料只用于解释，没有写进历史信号。')
para('执行限制：DISCK课程价格止于2022-04-08；回测缺开盘价时不成交，收盘价前填。原验证组合2022-05-02仍有477股残余持仓、总持股达11只。未私改课程引擎，不能把框架收益称为无摩擦可执行收益。')
para('改进重训使填补/截尾参数只在训练分区拟合，推断样本不参与拟合；最终重拟合只用train+val。按课程季度交易日映射加3个月门槛屏蔽未成熟y_return。该门槛是粗粒度约定，不证明底层每条标签的实际成交日和版本完全可知。')
para('仍保留的重训局限：课程stacking cv=3不是专门的时间序列purged CV；特征工程与模型选择在内部验证上仍可能有偏差。改进版是针对两类可靠性问题的预设修订，不宣称已解决所有机器学习前视风险。')
para('范围区分：本次估值模块从不使用数据库pe/ev_multiple，倍数均重新计算；前两版bonus选股为保持预先固定对照，仍沿用课程FEATURE_COLS中的pe/ev_multiple等特征。追加特征消融删除pe、ps、pb、peg、ev_multiple，用于检查模型对这些输入的依赖；这不证明原特征存在未来信息。不能把估值模块的审计结论扩大成所有选股输入已可靠。')
page();heading('4. 覆盖率：检查过与估值成功是两回事')
t=[['阶段','范围','候选-月','有效','覆盖%','股票数']]
for p in ['dev','val','test','forward']:
 for x in coverage:
  if x['variant']=='course' and x['period']==p:t.append([p,x['scope'],x['candidate_months'],x['valued'],n(float(x['coverage'])*100),x['distinct_tickers']])
table(t,[55,95,90,80,80,108])
para('以上为课程分数的月度信号诊断，不是每周过滤的调用总数。全部每周输入及失效原因保存在weekly_all_candidates文件；各方法有效数量在valuation_coverage.csv，缺失理由在missing_valuation_reasons.csv。')
para('早期覆盖不足主要与历史八季增长、至少12个历史倍数快照、正盈利和同行数量有关。扩大检查范围提高研究完整性，但不能保证增加收益。剔除未进入最终前10的股票，可能根本不改变持仓。')
page();heading('5. 两层证据之一：估值排序与区间')
t=[['阶段','范围','63日月均IC','126日','252日','一年样本数','落入率%']]
for x in signals:
 if x['variant']!='course':continue
 t.append([x['period'],x['scope'],n(x.get('mean_monthly_spearman_63'),3),n(x.get('mean_monthly_spearman_126'),3),n(x.get('mean_monthly_spearman_252'),3),n(x.get('interval_hit_n_252'),0),n(float(x['interval_hit_rate_252'])*100) if x.get('interval_hit_rate_252') else '未成熟'])
table(t,[53,78,78,65,65,83,86])
para('IC为每月横截面mispricing与未来收益的Spearman相关再取月均；pooled相关另存但不替代官方口径。负IC表示看起来更低估的股票后来相对表现更差。开发、验证与测试方向不稳定；重叠收益、共同市场暴露与行业效应使普通t值不能证明独立Alpha。')
para('官方一年落入率针对全局价格序列已延伸252交易日的记录，并取该窗口内最后可得个股价格；如果个股终点缺报价，这个价格可能提前，不能视为严格252日终点。它不是不可观测内在价值的“估准率”。大并集区间容易覆盖未来价格，必须同时看宽度。forward没有成熟63/126/252日标签，不能报告预测准确性。')
for x in signals:
 if x['variant']=='course' and x['scope']=='all_growth':para(x['period']+' 全候选区间相对宽度中位数：'+n(float(x['median_relative_interval_width'])*100)+'%。','SmallCN')
page();heading('价格终点质量：不能把缺失报价当成熟标签')
t=[['时期','估值记录','当日缺报价','严格252日样本','缺终点仍有官方收益','严格252日月均IC']]
for x in read('price_endpoint_summary.csv'):
 if x['variant']=='course' and x['scope']=='all_growth':t.append([x['period'],x['valued_records'],x['stale_asof_records'],x['strict_endpoint_n_252'],x['missing_endpoint_with_official_return_n_252'],n(x['strict_mean_monthly_ic_252'],3)])
table(t,[55,80,80,95,100,98])
para('严格终点诊断要求决策日和第252个交易日都有原始报价。它只补充审计，不改变交易、估值参数或官方原始报告。原始输出仍全部保留，不能静默替换为更有利的统计。63/126日同口径结果见price_endpoint_summary.csv。')
para('发现forward的WBD在2026-10-09没有当日原始收盘价，DataView在14个自然日内取最近价格30.95，故估值mispricing与决策价格一致，但官方评价CSV的price为缺失。不能据此算收益。开发/验证/测试也有个股252日终点缺报价而官方用窗口内较早报价的记录，均已逐条标注。')
page();heading('6. 两层证据之二：主实验完整消融')
for p in ['dev','val','test','forward']:
 heading(p.upper())
 t=[['组','累计%','CAGR%','Sharpe','回撤%','波动%','换手/年']]
 for group in 'ABCD':
  x=pick('course',p,group);t.append([group,n(x['total_return_pct']),n(x['cagr_pct']),n(x['sharpe'],3),n(x['max_drawdown_pct']),n(x['volatility_pct']),n(x['turnover_per_year'])])
 table(t,[30,77,77,76,82,82,84])
 if p=='val':page();heading('6. 主实验消融（续）')
para('课程Sharpe采用rf=0；换手为双向成交名义金额/平均净值/年数。短forward的年化收益和Sharpe极不稳定，只保留官方值，不将其外推一年。该期约30%的损失已显示风险，过滤的微小改善不足以消除风险。')
page();heading('净值路径：开发期与验证期')
nav_chart('dev');nav_chart('val')
para('按交易日顺序排列，分别以各期第一个保存净值归一为100。策略含课程执行成本；SPY/QQQ是被动复权价格参照，并非同成本、同95%目标仓位的模拟策略。')
page();heading('净值路径：测试期与短前向期')
nav_chart('test');nav_chart('forward')
para('每期组合分别初始化，不能把四张图拼接成一个连续实盘账户。前向曲线很短且损失明显，不能用测试期历史上涨掩盖这段风险。')
page();heading('7. 为什么验证改善小，哪些结论稳定？')
para('课程全候选验证期A累计-5.84%，B为-5.49%，只改善0.35个百分点；最大回撤从-42.52%到-41.99%，仍很深。平均每周剔除1.9名候选，比原top20约0.4名更多，但这不等于替换1.9只实际持仓。宽区间、缺失保留及候选递补共同限制过滤强度。')
para('扩大覆盖后开发与测试的最终组合收益在报告精度下与原top20结果相同，验证差异很小，提示新增低排名过滤多数没有改变最终组合。这是观察，不是严格因果归因。提高过滤频率或收窄区间并不自动提高Alpha，也可能剔除高速成长赢家并增加交易成本。')
para('利润率敏感：FCFF≈收入×经营利润率×(1−税率)+折旧−CapEx−ΔNWC，利润率的持续小变化会累积多年。终值敏感：TV=FCFF下一年/(WACC−g)，两者接近时分母很小。DCF估值变化是模型判断变化，不能直接理解为真实企业价值每次都同步剧变。')
para('原开发期预定80条记录的规则敏感性（两方法以上、分歧2x/3x/4x）仅标记2/4/5条；三方法均有效时0/1/2条。该审计保留为局部信号证据，不冒充全市场组合参数搜索。新的按年结果完整保存在annual_results.csv，不只选有利年份。')
para('目前更稳定的是程序契约、固定规则下可复现的计算与已披露的覆盖事实；尚未证明稳定的是收益改善、跨年度IC及经济意义上的真实内在价值。应先核实数据口径、再投资经济一致性及独立新样本，而不是为了扩大改善追逐本次测试收益。')
para('年度风险表另外保存历史日VaR95和CVaR95：分别为日收益5%分位对应的损失和该尾部平均损失，按非负损失报告。它们是样本描述，不是损失上限，不自动构成止损规则；短样本尤其不稳定。保持择时baseline时不按这些事后统计新增卖出规则，以免把估值与风控贡献混在一起。')
page();heading('补充稳健性：全部开发期候选与实际持仓')
t=[['最少方法数','分歧阈值','候选-月','有效估值','剔除标记','标记占比%']]
for x in read('development_all_candidate_rule_sensitivity.csv'):
 if x['year']=='all_dev':t.append([x['minimum_methods'],x['max_spread'],x['candidate_months'],x['valid_aggregate'],x['flagged_for_removal'],n(float(x['flagged_share'])*100)])
table(t,[85,80,85,85,85,88])
para('该表覆盖60个月全部5708条开发期候选记录，直接重用保存的各方法有效性和估值结果，仅改变规则判断。不是六次完整组合回测，不包含测试收益，不用于重选配置。阈值变宽会增加剔除，但并不能据此证明收益改善。')
t=[['阶段/组','平均持股数','最多持股数','最大单股占比%','费用估计$','名单不同天数%']]
for x in read('holdings_and_costs.csv'):
 if x['variant']=='course':t.append([x['period']+'/'+x['group'][0],n(x['mean_holdings']),x['max_holdings'],n(x['max_realized_stock_weight_pct']),n(x['nominal_cost_estimate_dollars']),n(x['held_names_differ_days_pct'])])
table(t,[80,80,80,90,90,88])
para('实际持股由保存成交数量累计重建；最大权重为日终价格漂移后的实际值，不是目标上限。费用=已保存成交名义金额绝对值×0.0006的近似重建，不是无成本策略对照。名单差异不包含同名股票权重改变；残余持仓会使实际持股超过10只。两版重训同口径诊断完整保存在CSV。')
hold=read('holdings_and_costs.csv')
short_cases=sum(float(x['minimum_reconstructed_shares'])<-.000001 for x in hold);negative_cash=sum(float(x['min_implied_cash_pct'])<-.000001 for x in hold)
para('全部版本保存成交的重建中，负股数的时期/组别数为'+str(short_cases)+'，隐含现金比例显著为负的时期/组别数为'+str(negative_cash)+'。部分阶段股票总占比接近100%，说明残余持仓和价格漂移会使名义5%缓冲消失；这仍需结合陈旧报价限制解释，不能把目标缓冲误写成始终保持5%现金。')
page();heading('逐年稳健性：不只看完整区间的赢家')
annual=read('annual_results.csv');t=[['时期/年份','A收益%','B收益%','差值百分点','A回撤%','B回撤%']];annual_deltas=[]
for period in ['dev','val','test','forward']:
 years=sorted({x['year'] for x in annual if x['variant']=='course' and x['run'].startswith(period+'_')})
 for year in years:
  aa=next(x for x in annual if x['variant']=='course' and x['run'].startswith(period+'_') and x['run'].endswith('A_selection') and x['year']==year)
  bb=next(x for x in annual if x['variant']=='course' and x['run'].startswith(period+'_') and x['run'].endswith('B_valuation') and x['year']==year)
  delta=float(bb['total_return_pct'])-float(aa['total_return_pct']);annual_deltas.append(delta)
  t.append([period+'/'+year,n(aa['total_return_pct']),n(bb['total_return_pct']),n(delta),n(aa['max_drawdown_pct']),n(bb['max_drawdown_pct'])])
table(t,[90,80,80,98,80,80])
para('按时期拆分的年度区段中，过滤收益差为正的有'+str(sum(x>0 for x in annual_deltas))+'段，为负的有'+str(sum(x<0 for x in annual_deltas))+'段。此计数只是描述，不是独立样本显著性检验；2026的test与forward为不同起止和分别初始化的组合，不能拼接成一个全年策略。')
para('每个年度重设峰值，年度最大回撤不等于完整策略最大回撤；完整期间风险以官方消融表为准。所有版本的年度波动、Sharpe、历史日VaR95/CVaR95及样本数保存在annual_results.csv，短forward尤其不能据少量尾部样本推断稳定风险。')
page();heading('8. 重训设计、环境与可靠性')
para('同一机器、同一版本依赖分别重训原growth.json与改进版，119个月各自独立输出，不覆盖课程scores。四个bucket均保留六种基模型RF、HistGBM、ExtraTrees、Ridge、XGBoost、LightGBM及stacking，随机种子与超参数不变。改进限于分区预处理与标签成熟门槛。')
para('实际覆盖并非每个bucket都119个月：课程分数、原版和改进版的growth_tech/cyclical/defensive均为119个月，real_assets均为110个月，缺2017-01至2017-09。代码沿用课程的数据门槛与跳过逻辑，不为缺失分组补造分数；本次未逐月判定具体缺失原因。正式组合使用的growth_tech完整覆盖119个月，每月原始分数股票数82至117。详见score_coverage_by_bucket.csv。')
para('两类修订一起比较，未做各自独立的模型消融，因此版本差异只能解释为这组训练修订的共同影响，不能分别给每一项修改归因多少收益。')
para('实际依赖：numpy2.4.6、pandas3.0.1、sklearn1.9.1、xgboost3.2.0、lightgbm4.7.0。与课程构建版本不完全一致，所以必须用同环境原版重训做对照。不能把“重训原版−课程分数”的差异归因于算法创新。')
para('资源控制：每个进程数值线程1，月份间并行。混合权重alpha全为0时略过不参与输出的统一模型拟合；原驱动与快速驱动在2019-06共504行的最终分数全部列、值、顺序、dtype完全一致。四bucket模型不省略；记录和两个驱动均附。只验证最终分数等价，不声称未用模型的诊断完全相同。')
para('初期布尔行业dummy参与quantile导致失败，修复为浮点后重新训练对应缺失月份；失败日志保留，不把失败算完成。预处理测试验证推断极端值不改变训练表示、缺失值有限、原输入不变、布尔列兼容。官方估值检查0 FAIL，测试调用WARN如实保留。')
comparisons=read('monthly_score_comparison.csv');t=[['对照','growth_tech月数','月均秩相关','平均Top10重合']]
for left,right in [('course','retrained_baseline'),('retrained_baseline','retrained_improved')]:
 xs=[x for x in comparisons if x['left']==left and x['right']==right and x['bucket']=='growth_tech'];t.append([left+' vs '+right,len(xs),n(sum(float(x['spearman']) for x in xs)/len(xs),3),n(sum(float(x['top10_overlap']) for x in xs)/len(xs))])
table(t,[215,90,100,103])
para('Top10重合是同月同bucket原始rank_score最高10名的描述性比较，不包含实际组合的成员递补、持仓延续加权或估值过滤，因此不等于实际持仓重合率。实际名单差异另由成交记录重建。')
for variant,label in variants[1:]:
 page();heading('9. '+label)
 for p in ['dev','val','test','forward']:
  heading(p.upper());t=[['组','累计%','CAGR%','Sharpe','回撤%','波动%','换手/年']]
  for group in 'ABCD':
   x=pick(variant,p,group);t.append([group,n(x['total_return_pct']),n(x['cagr_pct']),n(x['sharpe'],3),n(x['max_drawdown_pct']),n(x['volatility_pct']),n(x['turnover_per_year'])])
  table(t,[30,77,77,76,82,82,84])
  if p=='val':page();heading('9. '+label+'（续）')
 para('这是已接触测试后的扩展研究。对比原版重训与改进重训才能讨论训练流程变化；A/B差异才对应同一选股分数下估值模块贡献。不得挑出收益更高的一版替换正式课程评分结果。')
page();heading('9. 追加bonus：移除五个历史估值倍数')
para('在上一轮流程修订版上删除pe、ps、pb、peg、ev_multiple，其他特征、六个基模型、stacking、随机种子、组合参数和估值模块保持不变。原因是这五个历史字段的精确可用日期尚未核实；本实验只检验移除后的表现差异，不断言原字段存在前视。')
para('119个月训练产生'+str(v2_integrity['rows'])+'条分数，无重复或非有限值，股票/月份/bucket键与同环境未改原版完全一致。课程分数hash保持不变。growth_tech、cyclical和defensive各119个月，real_assets为110个月，继承课程早期缺口。')
t=[['阶段','A累计%','B累计%','B-A百分点','A Sharpe','A回撤%']]
for p in ['dev','val','test','forward']:
 a,b=v2pick(p,'A'),v2pick(p,'B');t.append([p,n(a['total_return_pct']),n(b['total_return_pct']),n(float(b['total_return_pct'])-float(a['total_return_pct'])),n(a['sharpe'],3),n(a['max_drawdown_pct'])])
table(t,[58,85,85,100,90,90])
para('测试期前冻结记录显示：开发期A为187.92%，低于上一轮194.14%；验证期A为11.85%，高于上一轮7.73%，因此只把它列为可进入一次最终测试的候选。冻结后测试期A为209.36%，同环境未改原版为188.87%，上一轮流程修订版为158.79%。但测试期A最大回撤为-34.66%，Sharpe 1.337仍低于同环境原版1.363；收益提高没有同时带来全面风险改善。')
para('估值过滤在本轮开发、验证、测试分别改变累计收益-22.60、-1.57、-9.03个百分点，短forward仅+0.13个百分点。说明估值模块的边际作用依赖上游候选；在这个新排序上，它多数时候剔除了随后表现更好的股票。短forward的A仍亏20.64%，区间只有2026-10-12至2026-11-06，不能用年化数字外推。')
para('课程test此前已被查看，本轮test不是严格盲测。版本在测试前冻结，测试和forward后不再修改；新增隔离实验只登记一次test调用。该结果作为加分附录，不替代正式课程分数、估值模块或主消融结论。')
page();heading('9. 追加bonus：v3横截面排名目标')
para('v3在v2基础上只改变训练目标：把每个已成熟未来收益转换为同一财务日期、同一bucket内0至1的百分位排名。特征、六个基础模型、stacking、预处理、标签成熟门槛、随机种子、组合参数、估值模块和交易规则保持不变。协议和准入门槛在训练前写入并冻结。')
para('v3完整训练'+str(v3_integrity['month_count'])+'个月、'+str(v3_integrity['rows'])+'条分数，无重复或非有限值，课程分数hash未变。growth_tech月度排序与v2的平均Spearman相关为'+n(v3_integrity['growth_tech_mean_monthly_spearman_v3_vs_v2'],3)+'，Top10平均重合'+n(v3_integrity['growth_tech_mean_top10_overlap_v3_vs_v2'])+'只，说明目标变化实质改变了排序。')
t=[['时期','版本','A累计%','A Sharpe','A回撤%','A换手/年','B累计%']]
for p in ['dev','val']:
 for key,label in [('same_environment_unchanged','同环境原版'),('v2','v2'),('v3','v3排名目标')]:
  a=v3_decision['results'][p][key+'_A'];b=v3_decision['results'][p].get(key+'_B')
  t.append([p,label,n(a['total_return_pct']),n(a['sharpe'],3),n(a['max_drawdown_pct']),n(a['turnover_per_year']),n(b['total_return_pct']) if b else '未重算'])
table(t,[48,95,75,70,75,78,75])
para('v3验证期A从v2的11.85%提高到16.37%，Sharpe从0.342提高到0.401；验证期B进一步达到19.26%。但是开发期A从v2的187.92%降到153.20%，低于预登记的177.92%底线，且开发期Sharpe从0.932降到0.850。五项准入检查中只有开发期收益底线失败，协议因此拒绝v3进入新的test。')
para('最终选择v2，不是因为v2所有指标都最好，而是v3没有满足在查看测试答案前写下的跨时期最低稳定性要求。v3的test和forward目录不存在，新增test调用为0；没有为了让v3通过而修改门槛。该负面实验完整保留，证明排名目标在验证期有效并不保证较长开发期也改善。')
page();heading('10. 跨版本估值边际贡献与基准')
t=[['分数','阶段','Δ累计百分点','ΔCAGR百分点','ΔSharpe','Δ回撤百分点']]
for v,label in variants:
 for p in ['dev','val','test','forward']:
  x=next(x for x in paired if x['variant']==v and x['period']==p);t.append([v.replace('retrained_',''),p,n(x['delta_total_return_pct']),n(x['delta_cagr_pct']),n(x['delta_sharpe'],3),n(x['delta_max_drawdown_pct'])])
table(t,[95,55,95,95,80,88])
para('正的回撤差值表示回撤较浅；收益差为同一期累计收益之差，不是独立复利策略回报。真实Alpha还需要风险暴露控制和新的独立样本，不能直接把B−A称为风险调整后的Alpha。')
aa=pick('retrained_baseline','test','A');bb=pick('retrained_improved','test','A')
para('同环境重训的测试期A组对照：原版累计'+n(aa['total_return_pct'])+'%，流程修订版'+n(bb['total_return_pct'])+'%；Sharpe由'+n(aa['sharpe'],3)+'变为'+n(bb['sharpe'],3)+'。较浅的最大回撤不能抵消所有收益/波动指标的退步。验证期更好并未稳定延续；本研究不据此追加测试调参或把更好看的分数替代正式课程基准。')
para('SPY与QQQ的累计收益、Sharpe均保留在每张官方消融表。原课程trailing-P/E例子开发期覆盖900/1200、63日月均IC约-0.129、一年区间落入约35%；与本方法覆盖集合和区间宽度不同，不能只按落入率判优。择时保持full，不应把sma200/engine的研究混入估值模块贡献。')
page();heading('市场参照：SPY与QQQ完整风险收益口径')
t=[['阶段','基准','累计%','CAGR%','Sharpe','回撤%','波动%']]
for x in read('benchmark_metrics.csv'):t.append([x['period'],x['benchmark'],n(x['total_return_pct']),n(x['cagr_pct']),n(x['sharpe'],3),n(x['max_drawdown_pct']),n(x['volatility_pct'])])
table(t,[60,50,80,80,78,80,80])
para('使用与官方相同起止日期和performance函数，累计收益与Sharpe已逐项核对官方消融表。它们是被动复权价格参照，不是同样6bp执行成本下模拟的交易策略；课程没有为其定义策略换手，故记为不适用，不能编造为零交易成本的同规则对照。')
para('年度收益从保存净值的日变动计算，首个净值之前的变动与课程指标一样不计入；跨年首日变动归入新年度。年度回撤在该年重新设峰值，不能替代完整期间最大回撤。VaR/CVaR也仅描述该年度样本。')
page();heading('11. 个人判断、测试次数与复现')
para('我的判断：我目前不愿意拿真钱使用这套估值过滤。最担心的是估值对利润率和终值高度敏感，因此结果不够稳定；市场环境或关键假设稍有变化，模型估值就可能明显改变。')
para('验证期改善很小，说明过滤器只在少数边界候选上产生了有限影响，而且这种影响尚未跨时期稳定延续。模型参数变动影响的是估计值，并不代表真实内在价值或市场价格一定同幅变化。只有在新的独立样本、不同年份与危机阶段、合理参数邻域以及交易成本下仍保持一致方向，我才会改变目前不投资的判断。')
para('正式提交的output/test_runs.jsonl记录3次课程分数相关测试期脚本调用。把原研究、同环境对照和特征消融v2全部日志合并后，总计'+str(log['total_logged_invocations'])+'次，其中特征消融v2有1次成功test调用。v3因未通过预登记开发期底线，新增test调用为0。全部原始行及来源路径随各实验保存。调用次数是脚本调用次数，不等于独立模型数量或独立样本数；不删除重复、失败或已接触记录以伪装一次测试。')
para('最终代码和数据hash、119个月完整性检查、环境依赖、两版配置与代码差异、训练日志、所有原始四组消融及估值CSV均附。README说明标准模块单独复制和bonus完整复现两条路径。数据包不随提交分发，无需API key。')
para('Medium文章已由Xinru Xiao发布：https://medium.com/@xx2524/does-valuation-improve-a-growth-stock-portfolio-b41963156218')
heading('来源与文件索引')
para('课程要求与固定源代码：https://github.com/AI4Finance-Foundation/STAT-GR5398-Columbia-University/tree/f5df2feab002b4d463fdd3bf30c34f6d91c0e49a/Fall-2026/Assignment2','SmallCN')
para('课程DATA_MANIFEST.json：FMP财报/价格、FRED宏观；SEC KEYS 2018 10-K：https://www.sec.gov/Archives/edgar/data/1601046/000160104618000030/keys-10312018x10k.htm；Keysight官方2018-11-20业绩公告。','SmallCN')
para('正式课程分数实验位于output/；追加实验位于bonus/feature_ablation_v2/和bonus/rank_target_v3/；汇总表位于analysis/；协议、代码hash与日志位于reproduction/。','SmallCN')
def footer(c,d):
 c.setFont('CJK',8);c.setFillColor(colors.HexColor('#607080'));c.drawString(43,26,'Xinru Xiao · xx2524 | Valuation and retraining');c.drawRightString(550,26,str(d.page))
doc=SimpleDocTemplate(str(pkg/'research_report.pdf'),pagesize=(595,842),rightMargin=43,leftMargin=43,topMargin=43,bottomMargin=43,title='GR5398 A2 - Expanded Valuation and Retraining',author='Xinru Xiao');doc.build(story,onFirstPage=footer,onLaterPages=footer)
(pkg/'research_report_zh.md').write_text('\n'.join(md));print(pkg/'research_report.pdf')
