"""Report independent quadrature refinements for E1."""
import csv
import json
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/oe_minimum_gain_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/minimum_gain'
data = json.loads((OUT/'p1_J16_seed11_refined128.json').read_text())
rows = data['rows']
final = [r for r in rows if (r['nx'],r['nv'],r['nq'])==(128,128,128)]
table=[]
for r in final:
    e=r['epsilon']
    base=next(x for x in rows if x['epsilon']==e and (x['nx'],x['nv'],x['nq'])==(64,64,64))
    inner=next(x for x in rows if x['epsilon']==e and (x['nx'],x['nv'],x['nq'])==(64,64,128))
    table.append(dict(epsilon=e,beta=r['beta'],rank=r['numerical_rank_x'],
        inner_relative_change=abs(inner['beta']/base['beta']-1),
        outer_relative_change=abs(r['beta']/inner['beta']-1)))
with (OUT/'quadrature_check.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=table[0]);w.writeheader();w.writerows(table)
fig,ax=plt.subplots(figsize=(6,4))
ordered=sorted(table,key=lambda r:r['epsilon'])
ax.plot([r['epsilon'] for r in ordered],[r['beta'] for r in ordered],'o-')
ax.set_xscale('symlog',linthresh=1e-6)
ax.set(xlabel='epsilon (zero included)',ylabel='Minimum gain beta',title='Fixed OE RF space: J=16 per field, seed 11')
ax.grid(alpha=.3);fig.tight_layout()
fig.savefig(OUT/'beta_epsilon.png',dpi=200)
fig.savefig(OUT/'beta_epsilon.pdf')
print('maximum inner change',max(r['inner_relative_change'] for r in table))
print('maximum outer change',max(r['outer_relative_change'] for r in table))
