"""Native-field quadrature, inflow and conservation audit; preserve main runs."""
import csv,json
import numpy as np
from run_e3_slab import ROOT,fs,symmetric,rf_evaluator,gauss,relative
OUT=ROOT/'results/e3_slab'
def save(name,rows):
    with (OUT/name).open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
transport=[]
for p in sorted(OUT.glob('seed*.json')):
    m=json.loads(p.read_text())
    if 'transport_reference' not in m:continue
    r=m['transport_reference']
    transport.append(dict(seed=m['seed'],epsilon=m['epsilon'],**r,
                         ratio_f=r['refinement_f']/r['E_f'],ratio_rho=r['refinement_rho']/r['E_rho']))
save('transport_check.csv',transport)
fs.uniform=symmetric;ev=rf_evaluator(64,11)
x,wx=gauss(128);v,wv=gauss(128);xx,vv=np.meshgrid(x,v,indexing='ij')
r,j,rx,jx=ev(xx,vv)
rl,jl,_,_=ev(np.zeros_like(v),v);rr,jr,_,_=ev(np.ones_like(v),v)
xi=.1+.8*x;xxi,vvi=np.meshgrid(xi,v,indexing='ij');_,ji,rxi,_=ev(xxi,vvi)
rows=[]
for eps in (1.,1e-6):
    base=np.load(OUT/f'seed11_eps{eps:.0e}.npz')['coefficients']
    brho=(r@base)@wv;bq=(j@base)@(wv*v)
    for n in (8,16,32):
        path=OUT if n==8 else OUT/f'quadrature_audit/q{n}'
        c=np.load(path/f'seed11_eps{eps:.0e}.npz')['coefficients']
        rho=(r@c)@wv;q=(j@c)@(wv*v);qx=(jx@c)@(wv*v)
        defect=(ji@c)@(wv*v)+(rxi@c)@wv/3
        inflow=np.sqrt(np.sum(wv*((rl@c+eps*(jl@c)-1)**2+(rr@c-eps*(jr@c))**2))/2)
        rows.append(dict(epsilon=eps,seed=11,operator_quadrature=n,
            E_diff=relative(rho,1-x,wx),E_q_diff=relative(q,np.full_like(q,1/3),wx),
            fick_defect=float(np.sqrt(np.sum(wx*defect**2)*9)),
            rho_change_from_q8=relative(rho,brho,wx),q_change_from_q8=relative(q,bq,wx),
            inflow_absolute_L2=float(inflow),q_min=float(q.min()),q_max=float(q.max()),
            q_derivative_L2=float(np.sqrt(np.sum(wx*qx*qx)))))
save('quadrature_inflow_balance.csv',rows)
print(json.dumps(rows,indent=2))
