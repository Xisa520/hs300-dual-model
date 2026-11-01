import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import torch
from torch import nn

LOOKBACK=20
FEATURES=['dma','ama','dma_hist','dma_slope','dma_cross','dma_gap','return_1','return_5','volatility_10','volume_z']

def clean(df):
    d=df.copy(); d.columns=[str(c).lower() for c in d.columns]
    for c in ['date','open','high','low','close','volume']:
        if c not in d: raise ValueError('缺少字段 '+c)
    d['date']=pd.to_datetime(d['date'],errors='coerce')
    for c in ['open','high','low','close','volume']:
        d[c]=pd.to_numeric(d[c],errors='coerce')
    d=d.dropna(subset=['date','open','high','low','close','volume']).drop_duplicates('date',keep='last').sort_values('date').reset_index(drop=True)
    if (d[['open','high','low','close']]<=0).any().any(): raise ValueError('存在非正价格')
    d['pct_change_recomputed']=d.close.pct_change()*100
    return d

def indicators(raw):
    d=clean(raw); c=d.close
    ema12=c.ewm(span=12,adjust=False).mean(); ema26=c.ewm(span=26,adjust=False).mean()
    d['dif']=ema12-ema26; d['dea']=d.dif.ewm(span=9,adjust=False).mean(); d['macd_hist']=d.dif-d.dea
    d['dma']=c.rolling(10).mean()-c.rolling(50).mean();d['ama']=d.dma.rolling(10).mean()
    d['dma_hist']=d.dma-d.ama; d['dma_slope']=d.dma.diff();d['dma_cross']=(d.dma_hist>0).astype(int)
    d['dma_gap']=d.dma/c
    d['return_1']=c.pct_change(); d['return_5']=c.pct_change(5)
    d['volatility_10']=d.return_1.rolling(10).std()
    v=d.volume; d['volume_z']=(v-v.rolling(60).mean())/(v.rolling(60).std()+1e-9)
    d['y_next']=(c.shift(-1)>=c).astype(float); d.loc[d.index[-1],'y_next']=np.nan
    return d

def draw_series(vals, out, lower=None, upper=None):
    x=np.asarray(vals,dtype=float); x=np.nan_to_num(x,nan=0,posinf=0,neginf=0)
    lo=x.min() if lower is None else lower;hi=x.max() if upper is None else upper
    if hi-lo<1e-9:hi=lo+1
    yy=np.clip(np.rint((1-(x-lo)/(hi-lo))*29+1).astype(int),0,31)
    xcoords=np.rint(np.linspace(1,30,len(x))).astype(int)
    for i in range(len(x)-1):
        n=max(abs(xcoords[i+1]-xcoords[i]),abs(yy[i+1]-yy[i]),1)+1
        xx=np.rint(np.linspace(xcoords[i],xcoords[i+1],n)).astype(int)
        yys=np.rint(np.linspace(yy[i],yy[i+1],n)).astype(int)
        out[yys,xx]=1

def image_for_window(w):
    a=np.zeros((3,32,32),dtype=np.float32)
    # 通道1：OHLC的高低范围与开收位置；每一窗口独立归一化
    lows=w.low.to_numpy(); highs=w.high.to_numpy(); opens=w.open.to_numpy();closes=w.close.to_numpy()
    lo=lows.min();hi=highs.max(); scale=max(hi-lo,1e-9)
    for j in range(len(w)):
        xx=int(round(1+j*29/(len(w)-1)))
        yl=int(np.clip(round(30-29*(lows[j]-lo)/scale),0,31));yh=int(np.clip(round(30-29*(highs[j]-lo)/scale),0,31))
        a[0,min(yl,yh):max(yl,yh)+1,xx]=1
        yo=int(np.clip(round(30-29*(opens[j]-lo)/scale),0,31));yc=int(np.clip(round(30-29*(closes[j]-lo)/scale),0,31))
        a[0,yo,max(0,xx-1):xx+1]=1;a[0,yc,xx:min(32,xx+2)]=1
    draw_series(w.dif.to_numpy(),a[1]);draw_series(w.dea.to_numpy(),a[1]*0 if False else a[1])
    draw_series(w.volume_z.to_numpy(),a[2])
    return a

def make_images(d):
    xs=[];ix=[]
    for i in range(LOOKBACK-1,len(d)):
        w=d.iloc[i-LOOKBACK+1:i+1]
        if w[['dif','dea','volume_z']].isna().any().any():continue
        xs.append(image_for_window(w));ix.append(i)
    return np.asarray(xs,dtype=np.float32),np.asarray(ix)

class CNNStock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Sequential(nn.Conv2d(3,8,3,padding=1),nn.LeakyReLU(),nn.MaxPool2d(2),nn.Conv2d(8,16,3,padding=1),nn.LeakyReLU(),nn.MaxPool2d(2))
        self.fc=nn.Sequential(nn.Flatten(),nn.Linear(16*8*8,64),nn.LeakyReLU(),nn.Dropout(.25),nn.Linear(64,16),nn.LeakyReLU(),nn.Linear(16,1))
    def forward(self,x):return self.fc(self.conv(x)).squeeze(-1)
