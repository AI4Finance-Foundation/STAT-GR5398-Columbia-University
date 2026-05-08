from __future__ import annotations
from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw

COLORS={"original":(31,119,180),"modified":(214,39,40),"qqq":(255,127,14),"spy":(44,160,44)}

def _canvas(title):
    img=Image.new("RGB",(1200,700),"white"); d=ImageDraw.Draw(img)
    d.text((50,25),title,fill=(0,0,0)); d.rectangle((70,80,1130,620),outline=(180,180,180))
    return img,d

def _plot_lines(df,path,title,cols):
    img,d=_canvas(title); vals=df[cols].astype(float)
    mn=float(vals.min().min()); mx=float(vals.max().max())
    if mx==mn: mx=mn+1
    n=len(vals); left,top,right,bottom=70,80,1130,620
    for j,c in enumerate(cols):
        s=vals[c].ffill().fillna(0).to_list(); pts=[]
        for i,v in enumerate(s):
            x=left+(right-left)*(i/max(n-1,1)); y=bottom-(bottom-top)*((float(v)-mn)/(mx-mn)); pts.append((x,y))
        if len(pts)>1: d.line(pts,fill=COLORS.get(c,(0,0,0)),width=3)
        d.text((80+180*j,640),c,fill=COLORS.get(c,(0,0,0)))
    Path(path).parent.mkdir(parents=True,exist_ok=True); img.save(path)

def plot_equity(eq,path): _plot_lines(eq,path,"Equity Curve: Original vs Modified vs QQQ vs SPY",["original","modified","qqq","spy"])
def plot_drawdown(dd,path): _plot_lines(dd,path,"Drawdown: Original vs Modified vs QQQ vs SPY",["original","modified","qqq","spy"])
def plot_weights(w,path,title):
    cols=[c for c in w.columns if w[c].abs().sum()>0][:8]
    if not cols: cols=list(w.columns[:1])
    _plot_lines(w[cols],path,title,cols)
def plot_active_group(o,m,path):
    mp={"DEFAULT":0,"DEFENSIVE":1,"REAL_ASSETS":2,"GROWTH_TECH":3}
    df=pd.DataFrame({"original":o.selected_group.map(mp).values,"modified":m.selected_group.map(mp).values},index=pd.to_datetime(o.rebalance_date))
    _plot_lines(df,path,"Active Group History",["original","modified"])

