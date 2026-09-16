"""Summarize every prescribed E3 run, including unsuccessful reference checks."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/e3_mpl')
import json
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/e3_slab'
rows=[json.loads(p.read_text()) for p in sorted(OUT.glob('seed*.json'))]
assert len(rows)==18
summary=[]
for eps in sorted({r['epsilon'] for r in rows}):
    group=[r for r in rows if r['epsilon']==eps]
    entry={'epsilon':eps,'seeds':len(group)}
    for key in ('E_diff','E_q_diff','fick_defect'):
        values=[r['evaluation']['128'][key] for r in group]
        entry[key]=float(np.median(values));entry[key+'_min']=min(values);entry[key+'_max']=max(values)
    summary.append(entry)
with (OUT/'summary.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=summary[0]);w.writeheader();w.writerows(summary)
fig,ax=plt.subplots(figsize=(6,4))
for key in ('E_diff','E_q_diff','fick_defect'):
    x=[r['epsilon'] for r in summary]
    ax.loglog(x,[r[key] for r in summary],'o-',label=key)
    ax.fill_between(x,[r[key+'_min'] for r in summary],[r[key+'_max'] for r in summary],alpha=.15)
ax.set(xlabel='epsilon',ylabel='Relative difference / defect');ax.legend();ax.grid(alpha=.3);fig.tight_layout()
fig.savefig(OUT/'diffusion_limit.pdf');fig.savefig(OUT/'diffusion_limit.png',dpi=200)
print(json.dumps(summary,indent=2))
from run_e3_slab import fs, symmetric, rf_evaluator, gauss
fs.uniform=symmetric
ev=rf_evaluator(64,11)
x=np.linspace(0,1,257);v,w=gauss(128)
xx,vv=np.meshgrid(x,v,indexing='ij');r,j,_,_=ev(xx,vv)
fig,axes=plt.subplots(1,2,figsize=(9,3.5))
for eps in (1.,.1,.001,1e-6):
    c=np.load(OUT/f'seed11_eps{eps:.0e}.npz')['coefficients']
    axes[0].plot(x,(r@c)@w,label=f'eps={eps:g}')
    axes[1].plot(x,(j@c)@(v*w),label=f'eps={eps:g}')
axes[0].plot(x,1-x,'k--',label='diffusion')
axes[1].axhline(1/3,color='k',linestyle='--',label='diffusion')
for ax,label in zip(axes,['rho','q=<v j>']):
    ax.set(xlabel='x',ylabel=label);ax.legend(fontsize=8);ax.grid(alpha=.3)
fig.tight_layout();fig.savefig(OUT/'profiles.pdf');fig.savefig(OUT/'profiles.png',dpi=200)
