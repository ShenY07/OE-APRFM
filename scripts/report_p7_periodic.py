"""One P7 density/amplitude figure and one error/time refinement table."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/p7_periodic_mpl')
import json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'results/p7_periodic'
records=[json.loads(p.read_text()) for p in sorted(out.glob('eps*.json'))]
rows=[]
fig,axes=plt.subplots(1,2,figsize=(10,4))
for r in records:
    for s in r['snapshots']:
        rows.append(dict(epsilon=r['epsilon'],dt=r['dt'],seed=r['seed'],rank=r['rank'],**s))
    if r['dt']!=.002:continue
    stem=f"eps{r['epsilon']:.0e}_dt{r['dt']:.0e}_seed{r['seed']}"
    d=np.load(out/f'{stem}.npz');h=d['history'];x=d['x']
    axes[0].plot(x,d['rho_100'],label=f"eps={r['epsilon']:g}, t=.2")
    axes[1].plot(h[:,0],h[:,1],label=f"eps={r['epsilon']:g}")
xx=np.linspace(0,1,257);tt=np.linspace(0,.2,101)
axes[0].plot(xx,1+.2*np.exp(-4*np.pi**2*.2/3)*np.cos(2*np.pi*xx),'k--',label='continuous diffusion')
axes[1].plot(tt,.2*np.exp(-4*np.pi**2*tt/3),'k--',label='continuous diffusion')
axes[1].plot(tt,.2*(1+4*np.pi**2*.002/3)**(-np.arange(101)),':',label='BE diffusion')
for ax,ylabel in zip(axes,['rho at t=.2','cosine amplitude']):
    ax.set_ylabel(ylabel);ax.grid(alpha=.2);ax.legend(fontsize=7)
axes[0].set_xlabel('x');axes[1].set_xlabel('t');fig.tight_layout()
for ext in ['pdf','png']:fig.savefig(out/f'density_amplitude.{ext}',dpi=180)
if rows:
    with (out/'summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(f'{len(records)} configurations, {len(rows)} snapshots; not an acceptance claim')
