"""Inline LaTeX vector equivalents of the three existing research PDF figures."""
import csv
import hashlib
from pathlib import Path


def build_figures(root):
    sources = {
        'grounding': 'audits/grounding/comparison_v6/before_after_scorecard.csv',
        'nvda': 'audits/target_price_sensitivity/NVDA/target_price_sensitivity.csv',
        'avgo': 'audits/sensitivity/AVGO/target_price_sensitivity.csv',
    }
    records = {}; inputs = []
    for key, name in sources.items():
        path = root / name
        records[key] = list(csv.DictReader(path.open(encoding='utf-8-sig', newline='')))
        inputs.append({'path':name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    rows = {r['ticker']:r for r in records['grounding'] if r['section']=='ALL' and r['ticker']!='ALL'}
    assert set(rows)=={'NVDA','AMD','INTC'}
    out = [r'\begin{figure}[H]\centering',r'\begin{tikzpicture}',
           r'\begin{groupplot}[group style={group size=2 by 1,horizontal sep=1.45cm},',
           r'width=0.43\textwidth,height=5.0cm,ybar,/pgf/bar width=9pt,',
           r'symbolic x coords={NVDA,AMD,INTC},xtick=data,ymin=0,ylabel={Percent},',
           r'enlarge x limits=0.24,ymajorgrids=true,grid style={gray!15},',
           r'tick label style={font=\small},title style={font=\small\bfseries},',
           r'legend style={at={(0.5,-0.23)},anchor=north,font=\scriptsize,draw=none},legend columns=2]']
    for field, title, ymax in [('grounding','Supported / all assertions',35),('contradiction','Contradicted / all assertions',5.2)]:
        out.append(r'\nextgroupplot[title={'+title+'},ymax='+str(ymax)+']')
        for treatment, color in [('before','beforeblue'),('after','afterteal')]:
            coordinates=' '.join('('+ticker+','+format(float(rows[ticker][treatment+'_'+field+'_rate'])*100,'.15g')+')' for ticker in ['NVDA','AMD','INTC'])
            out.append(r'\addplot[fill='+color+',draw='+color+'] coordinates {'+coordinates+'};')
        out.append(r'\legend{Baseline,Verifier V6}')
    out += [r'\end{groupplot}',r'\end{tikzpicture}',r'\caption{Measured grounding and contradiction rates on all 24 sections; repeated claims affect denominators. Historical V6 evaluation.}',r'\end{figure}']
    figures={'grounding':'\n'.join(out)}
    for key, ticker, period in [('nvda','NVDA','2029E'),('avgo','AVGO','2028E')]:
        data=records[key]; assert len(data)==25
        values=[float(r['target_price']) for r in data]
        out=[r'\begin{figure}[H]\centering',r'\begin{tikzpicture}',
             r'\begin{axis}[width=0.87\textwidth,height=9.2cm,',
             r'title={'+ticker+': '+period+r' EV/EBITDA target-price scenarios},title style={font=\small\bfseries},',
             r'xmin=-0.5,xmax=4.5,ymin=-0.5,ymax=4.5,y dir=reverse,',
             r'xtick={0,1,2,3,4},xticklabels={$-2$,$-1$,$0$,$+1$,$+2$},',
             r'ytick={0,1,2,3,4},yticklabels={$-5$,$-2.5$,$0$,$+2.5$,$+5$},',
             r'xlabel={EBITDA margin change (percentage points)},',
             r'ylabel={Forecast revenue level change (relative \%)},',
             r'tick label style={font=\small},label style={font=\small},',
             r'colormap/viridis,colorbar,point meta min='+str(min(values))+',point meta max='+str(max(values))+',',
             r'colorbar style={ylabel={Target price (USD per share)},ylabel style={font=\small},tick label style={font=\small}}]',
             r'\addplot[matrix plot*,mesh/cols=5,point meta=explicit] table[x=x,y=y,meta=z] {', 'x y z']
        for index,value in enumerate(values): out.append(f'{index%5} {index//5} {value:.16g}')
        out.append('};')
        middle=(min(values)+max(values))/2
        for index,value in enumerate(values):
            color='white' if value<middle else 'black'
            out.append(r'\node[font=\small\sffamily,text='+color+'] at (axis cs:'+str(index%5)+','+str(index//5)+'){'+format(value,'.2f')+'};')
        out += [r'\end{axis}',r'\end{tikzpicture}',
                r'\caption{'+ticker+r' target-price sensitivity from the saved '+period+r' EV/EBITDA scenarios. Revenue shocks change the forecast level; margin shocks are percentage points. The multiple and shares are fixed; net debt remains 10\% of each scenario\textquotesingle s enterprise value. These are conditional scenarios, not confidence intervals.}',r'\end{figure}']
        figures[key]='\n'.join(out)
    return figures,inputs
