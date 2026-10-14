"""Explicit independent review of V8.2 visible text against the original writer prompt.

Indices below are deliberately enumerated after reading every changed source unit.
No verifier labels/number bindings are imported. Warnings are stripped only to
record their underlying assertions; they never determine an audit classification.
"""
import re
from review_v7_common import Review,U,S,C
from prepare_after_adjudication import normalized
r=Review('NVDA','reasoning_v8_2');p=r.put
F=((4,4),(6,6));CM=((4,4),(10,10));EM=((4,4),(12,12));SG=((4,4),(13,13));EPS=((4,4),(15,15));PE=((4,4),(16,16));N=((42,46),)
def text(s,i):return normalized(r.todo[s,i]['text'])
def unsupported(s,ids):
    for i in ids:p(s,i,U(text(s,i)))
def supported(s,ids,refs,reason,num='not_numeric'):
    for i in ids:p(s,i,[S(text(s,i),refs,reason,num)])
def values(s,i,metric,pairs,refs):
    p(s,i,[S(f'NVIDIA {metric} is {v} in {y}.',refs,'Original table row and A/E header directly provide this value and period; E is a forecast, not an observed outcome.','correct') for y,v in pairs])
growth=S('NVIDIA revenue has grown rapidly.',F,'Actual revenue rises from26.914bn in2022A to215.938bn in2026A; this does not claim monotonic annual growth rates.')
margin=S('NVIDIA EBITDA margin has expanded.',EM,'Actual margin grows from42.2% in2022A to66.9% in2026A; not a claim of monotonic growth.')
cmgrowth=S('NVIDIA contribution margin has rebounded from2023A.',CM,'Contribution margin rises from56.9% in2023A to75% in2025A, then71.1% in2026A.')
def trend(s,i,metric,pairs,refs):
    atoms=[S(f'NVIDIA {metric} is {v} in {y}.',refs,'Direct original row/header value with actual/forecast distinction.','correct') for y,v in pairs]
    atoms.append(S(text(s,i),refs,'Direction follows the quoted endpoints; E endpoint remains a scenario forecast, not realized history.','correct'))
    p(s,i,atoms)
def bad_cagr(s,i):
    p(s,i,[S('The revenue table displays a generic CAGR of68.3%.',F,'The generic CAGR cell is literally68.3%, but it has no right to redefine the endpoint interval.','correct'),C('NVIDIA revenue CAGR from2022A to2029E is68.3%.',F,'(249952553760/26914000000)^(1/7)-1 =37.4892%, not68.3%; the latter matches the historical2022A–2026A interval.','incorrect')])

s='tagline'
unsupported(s,[0,6,7,8,9,10,11]);p(s,4,[growth])
values(s,1,'contribution margin',[('2024A','72.7%'),('2025A','75.0%'),('2026A','71.1%'),('2027E','72.1%'),('2028E','73.1%'),('2029E','74.1%')],CM)
values(s,2,'EBITDA margin',[('2027E','70.5%'),('2028E','72.0%'),('2029E','73.5%')],EM)
values(s,3,'EBITDA margin',[('2029E','73.5%')],EM)
supported(s,[5],F,'The sentence reports only the literal generic table CAGR cell and assigns no incorrect interval.','correct')

s='company_overview'
unsupported(s,[0,2,3,5,7,9,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,47,48,52,55,56,57,58,62,63,64,65,66,67,68,69,70,71,72,73,74,75,77,78,79,80,81,83,89,90,91,93,94,95,96,97,98,99,100,101,102,103,104,105])
p(s,1,U('NVIDIA specializes in designing GPUs.','NVIDIA specializes in manufacturing GPUs.'))
supported(s,[6],((45,45),),'Explicit Data Center segment sales establish service to that market, but not dominance or a broader product catalog.')
for i,y,v in [(10,'2023A','26974000000.0'),(11,'2025A','130497000000.0'),(41,'2024A','60922000000.0')]:values(s,i,'revenue',[(y,v)],F)
for i,pairs in [(12,[('2023A','26974000000.0'),('2025A','130497000000.0')]),(42,[('2023A','26974000000.0'),('2024A','60922000000.0')]),(43,[('2024A','60922000000.0'),('2025A','130497000000.0')])]:trend(s,i,'revenue',pairs,F)
supported(s,[44],F,'(130497000000/26974000000)^(1/2)-1 =119.9518%, which does exceed68%; a warning cannot turn this true inequality into unsupported.','correct')
trend(s,45,'contribution margin',[('2023A','56.9%'),('2024A','72.7%'),('2025A','75.0%')],CM)
values(s,46,'contribution margin',[('2024A','72.7%'),('2025A','75.0%'),('2026A','71.1%')],CM)
values(s,49,'EBITDA margin',[('2023A','22.2%')],EM);values(s,50,'EBITDA margin',[('2025A','66.0%')],EM)
trend(s,51,'EBITDA margin',[('2023A','22.2%'),('2025A','66.0%')],EM)
supported(s,[53],((45,45),),'The supplied quarterly Data Center growth is117% year-on-year, sufficient for the narrow descriptive growth statement.')
supported(s,[54],N,'Original supplied news explicitly reports89.0bn Data Center revenue.','correct')
p(s,60,U('New GPU architectures promise to sustain NVIDIA competitive edge.','New AI platforms promise to sustain NVIDIA competitive edge.'))
p(s,76,U('Huang vision positioned NVIDIA as an AI hardware-integration pioneer.','Huang vision positioned NVIDIA as an AI software-integration pioneer.'))
values(s,84,'EPS',[('2023A','0.18')],EPS);values(s,85,'EPS',[('2025A','2.97')],EPS)
trend(s,86,'EPS',[('2023A','0.18'),('2025A','2.97')],EPS)
values(s,87,'EPS',[('2027E','5.23'),('2028E','5.6'),('2029E','5.88')],EPS)
supported(s,[92],((19,27),),'NVIDIA exceeds Intel in2024 and2025 actuals and2027 forecast but trails in2022/2023; the qualified often statement is supported, not always.')

s='investment_overview'
unsupported(s,[3,7,10,13,14,17,18,19,20,22,23,25,26,28,30,31,34,35,41,42,43,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60])
p(s,2,[],nonfactual='Truncated reporting-verb fragment contains no independent factual predicate; following units preserve the reported result.')
supported(s,[4,5,6,8,11,15,16,24,32,33],N,'Read in the immediately preceding quarterly-result context, the amount, growth rate or GAAP/non-GAAP label is explicitly in the supplied announcement.','correct')
for i in [12,29]:p(s,i,[S('NVIDIA reported gross margin of75% in Q2 fiscal2027.',N,'Both GAAP and non-GAAP75% are explicit.','correct')]+U('NVIDIA gross margin matched management guidance.' if i==12 else 'The reported gross margin establishes operational efficiency.'))
p(s,27,[S('NVIDIA gross margin was75% in Q2 fiscal2027.',N,'Explicit current-quarter gross margin.','correct')]+U('NVIDIA gross margin held steady over time.'))
for i in [36,38]:p(s,i,[growth]+U(text(s,i)))
for i in [37,39]:p(s,i,[margin]+U(text(s,i)))
supported(s,[44],((45,45),),'The excerpt reports117% Data Center year-on-year growth; no explanation of causes follows from it.')
p(s,45,U('NVIDIA is a critical enabler in the AI ecosystem.','The AI ecosystem is expanding.'))

s='valuation_overview'
unsupported(s,[3,9,15,18,23,27,28,29,30,31,33,34,35,36,37,38,40,41,42,43,44,45,47])
p(s,0,U('NVIDIA trailing PE is elevated relative to AMD.','NVIDIA forward PE is elevated relative to AMD.','NVIDIA trailing PE is elevated relative to Intel.','NVIDIA forward PE is elevated relative to Intel.'))
values(s,1,'PE ratio',[('2023A','106.44444444444444'),('2027E','35.9')],PE);values(s,2,'PE ratio',[('2026A','37.82352941176471')],PE)
supported(s,[4],F,'Generic CAGR matches the displayed row, with no explicit false interval.','correct')
p(s,5,[growth]+U('AI demand primarily drives NVIDIA revenue growth.','Data Center demand primarily drives NVIDIA revenue growth.','Gaming demand primarily drives NVIDIA revenue growth.'))
values(s,6,'EBITDA margin',[('2023A','22.2%'),('2029E','73.5%')],EM)
values(s,7,'contribution margin',[('2023A','56.9%'),('2029E','74.1%')],CM)
p(s,8,[margin,cmgrowth]+U('Margin expansion establishes improving operational efficiency.','Margin expansion establishes improving profitability quality.'))
p(s,11,U('NVIDIA dominates AI technologies.','NVIDIA dominates GPU technologies.'))
p(s,12,U('AI technologies are growth engines commanding higher multiples.','GPU technologies are growth engines commanding higher multiples.'))
supported(s,[13],((37,37),),'Direct original EV/EBITDA row.','correct')
p(s,14,[S('NVIDIA EV/EBITDA was81.0476 in2023.',((34,34),),'Direct peer row.','correct'),S('NVIDIA EV/EBITDA was31.4283 in2026.',((37,37),),'Direct peer row.','correct')])
p(s,19,U('NVIDIA competitive positioning supports a valuation premium.','NVIDIA innovation pipeline supports a valuation premium.','NVIDIA leadership supports a valuation premium.'))
p(s,24,U('NVIDIA valuation peaks coincided with accelerated data-center GPU adoption.','NVIDIA valuation peaks coincided with accelerated autonomous-application GPU adoption.'))
values(s,25,'PE ratio',[('2023A','106.44444444444444'),('2029E','32.4')],PE)
p(s,26,[S('NVIDIA PE has moderated from2023A.',PE,'2023A PE106.4444 is above2026A37.8235.'),S('NVIDIA EV/EBITDA has moderated from2023.',((34,37),),'81.0476 in2023 falls to31.4283 in2026.')]+U('Multiple moderation coincides with stabilization in growth expectations.'))
p(s,32,U('Comparable companies with similar growth prospects are scarce.','Comparable companies with similar technological moats are scarce.'))

s='risks'
unsupported(s,[1,2,4,5,6,7,8,9,11,12,13,14,17,18,19,20,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,50,52,53,56,57,58,59,60,61,62,63,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104])
p(s,0,U('NVIDIA faces market risks.','NVIDIA faces competitive risks.','NVIDIA faces operational risks.','NVIDIA faces financial risks.','NVIDIA faces regulatory risks.'))
bad_cagr(s,3)
p(s,15,U('NVIDIA geopolitical tensions especially involve China.','NVIDIA trade restrictions especially involve China.'))
p(s,16,U('Geopolitical tensions complicate NVIDIA supply chains.','Trade restrictions complicate NVIDIA supply chains.'))
supported(s,[54],((1,48),),'Inspection of the entire original prompt: financial table, peer comparisons and news contain no debt balance. This is a statement of an input omission, not an assumed zero debt.')

s='competitor_analysis'
unsupported(s,[1,2,3,9,10,11,13,14,16,17,18,20,21,22,23,27,28,29,30,31,32,33,34,38,39,40,41,43,44,45,46,47,49,50,51,52,53,54,57,58,59,60,61,63,67,69,72,76,77,78,79,80,81,82,83,84,85])
values(s,4,'revenue growth',[('2024A','125.9%'),('2025A','114.2%'),('2026A','65.5%')],((4,4),(14,14)))
values(s,5,'EBITDA margin',[('2023A','22.2%'),('2024A','58.4%'),('2025A','66.0%'),('2026A','66.9%'),('2027E','70.5%'),('2028E','72.0%'),('2029E','73.5%')],EM)
for i in [6,7]:p(s,i,[growth,margin]+U(text(s,i)))
p(s,26,U('Intel increases GPU investment to challenge NVIDIA.','Intel increases AI-hardware investment to challenge NVIDIA.'))
p(s,48,U('NVIDIA has extensive AI frameworks.','NVIDIA has extensive deep-learning frameworks.'))
bad_cagr(s,55);values(s,56,'EBITDA margin',[('2029E','73.5%')],EM)
p(s,62,U('NVIDIA integration with AI research providers amplifies its competitive position.','NVIDIA integration with cloud providers amplifies its competitive position.'))
p(s,65,U('NVIDIA R&D investments create high switching costs.','NVIDIA ecosystem development creates high switching costs.'))
p(s,66,U('NVIDIA customers rely on its hardware stack.','NVIDIA customers rely on its software stack.'))
p(s,68,[S('NVIDIA Data Center quarterly revenue was89.0bn.',N,'Explicit excerpt amount.','correct'),S('NVIDIA Data Center quarterly revenue grew117% year-on-year.',N,'Explicit excerpt growth.','correct')])
p(s,75,U('NVIDIA financial performance indicates investor confidence in durable competitive advantages.','NVIDIA market valuation indicates investor confidence in durable competitive advantages.'))

s='major_takeaways'
unsupported(s,[6,9,10,11,12,16,17,18,19,20,22,23,24,25,28,29,32,33,35,36,37,38,39,40,45,46,47,48,49,50,52])
p(s,0,[S(text(s,0),F,'The mixed actual/forecast series expands about9.29x from2022A to2029E; this is a description of the displayed trajectory, not an assertion that2029 has occurred.')])
bad_cagr(s,1)
supported(s,[2],F,'2023A–2026A revenue grows8.01x, accounting for almost all historical2022A–2026A CAGR; little growth occurs2022A–2023A. No unsupported business cause is added.')
trend(s,3,'revenue',[('2023A','26974000000.0'),('2026A','215938000000.0')],F)
p(s,4,[growth]+U('NVIDIA AI leadership fuels its growth.','NVIDIA Data Center leadership fuels its growth.'))
p(s,5,[S('NVIDIA Q2 fiscal2027 revenue grew106% year-on-year.',N,'Explicit announcement title and excerpt.','correct'),S('NVIDIA Q2 fiscal2027 Data Center revenue grew117% year-on-year.',N,'Explicit announcement title and excerpt.','correct')])
p(s,7,U('NVIDIA capitalizes on structural AI shifts.','NVIDIA capitalizes on structural cloud shifts.'))
p(s,8,U('Strong GPU demand sustains NVIDIA growth.','Strong AI-platform demand sustains NVIDIA growth.'))
values(s,13,'contribution margin',[('2023A','56.9%')],CM)
values(s,14,'contribution margin',[('2024A','72.7%'),('2025A','75.0%'),('2026A','71.1%'),('2027E','72.1%'),('2028E','73.1%'),('2029E','74.1%')],CM)
p(s,15,[cmgrowth,growth,S('Contribution-margin recovery coincides with revenue expansion after2023A.',F+CM,'Both changes are visible; coincidence does not establish a causal cost-efficiency mechanism.')])
for i in [26,34]:trend(s,i,'SG&A margin',[('2023A','9.0%'),('2029E','0.6%')],SG)
p(s,27,[S('Forecast2029E SG&A margin is0.6%.',SG,'Explicit forecast row.','correct')]+U('The0.6% forecast margin establishes ultra-efficient operations.'))
trend(s,30,'revenue',[('2023A','26974000000.0'),('2029E','249952553760.0')],F)
p(s,31,[growth]+U('SG&A costs are fixed.','Fixed SG&A costs explain the spreading of overhead across revenue.'))
p(s,41,[margin]);p(s,53,[growth])
for i in [42,44]:trend(s,i,'EBITDA margin',[('2023A','22.2%'),('2029E','73.5%')],EM)

s='news_summary'
unsupported(s,[1,4,5,9,14,15,17,22,24,25,27,29,30,31,32,33,34,40,41,43,44,45])
supported(s,[0],N,'The supplied release reports106% overall and117% Data Center growth; descriptive extraordinary growth requires no unprovided business mechanism.')
trend(s,6,'EBITDA margin',[('2027E','70.5%'),('2029E','73.5%')],EM)
supported(s,[7],((42,46),),'Explicit release title and quarter in text.','correct')
supported(s,[10],((45,45),),'Explicit Data Center year-on-year percentage.','correct')
p(s,16,[S('NVIDIA reported strong revenue growth.',N,'106% year-on-year growth is explicit.')]+U('NVIDIA reported strong earnings growth.','Strong earnings cause an immediate stock-price increase.','Revenue growth causes an immediate stock-price increase.'))
bad_cagr(s,19);values(s,20,'EBITDA margin',[('2029E','73.5%')],EM)
trend(s,21,'EPS',[('2027E','5.23'),('2028E','5.6'),('2029E','5.88')],EPS)
p(s,23,[S('NVIDIA forecast EPS increases.',EPS,'5.23/5.6/5.88 are increasing model forecasts, not management guidance.')]+U('Forecast EPS growth establishes sustained profitability.','PE ratios establish sustained profitability.'))
p(s,26,[S('NVIDIA exceeds AMD in latest common annual EBITDA growth.',((24,25),),'2024→2025 NVIDIA142.07% vsAMD38.36%.'),C('NVIDIA exceeds Intel in latest common annual EBITDA growth.',((24,25),),'Intel grows1093.18% from1.203bn to14.354bn, aboveNVIDIA142.07%; absolute levels do not establish superior growth.')])
p(s,28,[S(f'{t} EV/EBITDA was{v} in2023.',((30,34),),'Direct original peer table entity/year cell.','correct') for t,v in [('NVIDIA','81.0476'),('AMD','57.1197'),('Intel','22.4823')]])
p(s,39,U('NVIDIA AI leadership drives positive market sentiment.','NVIDIA Data Center leadership drives positive market sentiment.'))
r.save()
