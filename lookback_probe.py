"""Offline, predeclared lookback probe for the conditional factor; not a submission."""

import csv
import numpy as np
import pandas as pd

from data_loader import list_periods, load_period
from evaluation import backtest


def ranks(features):
    tickers = sorted(features["Feature.1"].columns, key=lambda x: int(x.split(".")[-1]))
    r = []
    for i in range(1, 7):
        v = np.nan_to_num(features[f"Feature.{i}"][tickers].to_numpy(float))
        o = np.argsort(np.argsort(v, axis=1, kind="mergesort"), axis=1, kind="mergesort")
        r.append((o.astype(float)-9.5)/9.5)
    return r, tickers


def smooth(v, span=120, state=None):
    a=2/(span+1); s=np.zeros(v.shape[1]) if state is None else state.copy(); out=np.empty_like(v)
    for i in range(len(v)):
        s=a*v[i]+(1-a)*s; out[i]=s
    return out-out.mean(axis=1,keepdims=True), s-s.mean()


def evaluate(lookback):
    nets=[]; ics=[]; folds=[]
    for period in list_periods():
        d=load_period(period); nval=min(800,int(.2*len(d["features"]))); nt=len(d["features"])-nval
        r,tickers=ranks(d["features"]); core=sum(r[:4])/4; factor=-core**3+core*r[5]; factor-=factor.mean(axis=1,keepdims=True)
        state_blocks=[]
        for x in r:
            state_blocks += [x.mean(axis=1), x.std(axis=1), np.mean(np.abs(x-np.median(x,axis=1,keepdims=True)),axis=1)]
        state=np.column_stack(state_blocks); start=max(0,nt-lookback); c=np.median(state[start:nt],axis=0); s=np.median(np.abs(state[start:nt]-c),axis=0)*1.4826; s=np.where(s>1e-8,s,1); xs=np.clip((state-c)/s,-6,6)
        y=np.mean(factor[start:nt]*np.nan_to_num(d["target"][tickers].to_numpy(float)[start:nt]),axis=1); x=xs[start:nt]; beta=np.linalg.solve(x.T@x/len(x)+.01*np.eye(x.shape[1]),x.T@(y-y.mean())/len(x)); fit=x@beta; reg=(xs[nt:]@beta-fit.mean())/max(fit.std(),1e-12); w=1+.5*np.clip(reg,-2,2)
        _,st=smooth(factor[:nt]); predv,_=smooth(factor[nt:]*w[:,None],120,st); pred=pd.DataFrame(predv,index=d["returns"].index[nt:],columns=tickers); ret=d["returns"].iloc[nt:][tickers]; pnl=backtest(pred,ret); nets.append(pnl); ics.append(float(pred.corrwith(d["target"].iloc[nt:][tickers],axis=1).mean())); folds.append(float(pnl.mean()/pnl.std()))
    allp=pd.concat(nets,ignore_index=True); return {"lookback":lookback,"full":float(allp.mean()/allp.std()),"early":float(pd.concat(nets[:19],ignore_index=True).mean()/pd.concat(nets[:19],ignore_index=True).std()),"late":float(pd.concat(nets[19:],ignore_index=True).mean()/pd.concat(nets[19:],ignore_index=True).std()),"ic":float(np.mean(ics)),"positive":int(np.sum(np.array(folds)>0)),"fold_std":float(np.std(folds))}


def main():
    rows=[evaluate(x) for x in (500,1000,2500,5000,10000,20000)]
    for row in rows: print(row, flush=True)
    with open("lookback_probe.csv","w",newline="") as h: w=csv.DictWriter(h,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__": main()
