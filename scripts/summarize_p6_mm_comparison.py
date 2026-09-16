"""Compare archived OE P6 and newly run MM P6 on identical evaluation grids."""
import csv
import json
from pathlib import Path
import numpy as np
from build_transport_diagnostics import compute, trap_weights

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/p6_mm_comparison'


def main():
    rows=[]
    for method,folder in [('OE-APRFM',ROOT/'results/requirement_2026_08_20/p6_raw'),('MM-APRFM',OUT/'mm')]:
        for path in sorted(folder.glob('*.json')):
            r=json.loads(path.read_text())
            if r.get('problem')!='p6':continue
            with np.load(path.with_suffix('.npz')) as z:
                x=z['x'];v=z['v'] if 'v' in z else z['velocity']
                assert np.allclose(x,np.linspace(0,1,257))
                assert np.allclose(v,np.linspace(-1,1,128))
                eps=r['epsilon']
                truth=1+.25*np.sin(2*np.pi*x[:,None])+eps*v[None,:]*.25*np.cos(np.pi*x[:,None])
                archived=z['exact'] if 'exact' in z else z['reference_f']
                np.testing.assert_allclose(truth,archived,atol=1e-14)
                row=dict(method=method,epsilon=eps,seed=r['seed'],N_coef=r['num_columns'],N_row=r['num_rows'],E_f=r['relative_l2_f'],E_rho=r['relative_l2_rho'])
                row.update(compute(z['f'],truth,trap_weights(v),v[:,None],trap_weights(x),eps))
                row['source']=str(path.relative_to(ROOT));rows.append(row)
    assert len(rows)==18,len(rows)
    def save(name,data):
        with (OUT/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    save('seedwise.csv',rows)
    summary=[]
    metrics=['E_f','E_rho','flux_relative_l2','g_relative_l2','g_even_error_l2','g_odd_error_l2']
    for eps in [1.,.001,.000001]:
        for method in ['OE-APRFM','MM-APRFM']:
            group=[r for r in rows if r['epsilon']==eps and r['method']==method]
            assert {r['seed'] for r in group}=={11,23,37}
            row=dict(epsilon=eps,method=method,N_coef=group[0]['N_coef'],N_row=group[0]['N_row'])
            for key in metrics:
                values=[r[key] for r in group]
                row[key]=float(np.median(values));row[key+'_min']=min(values);row[key+'_max']=max(values)
            summary.append(row)
    save('summary.csv',summary)
    lines=['# P6 angular manufactured: OE vs MM','',
           'f=1+0.25 sin(2 pi x)+epsilon v 0.25 cos(pi x). Both methods use 128 total coefficients, one spatial/angular patch, tanh scale 1, rcond=1e-12, seeds 11/23/37, CPU float64. MM allocates 64 features to rho and 64 to g; OE allocates 64 to r and 64 to j.',
           '', 'This is a coefficient-budget comparison, not identical sampling: OE uses 2944 residual rows, MM 3008. Both evaluate the same 257x128 grid. f/rho errors follow the existing unweighted convention; flux/g use physical spatial trapezoidal and normalized angular trapezoidal weights. Different training angular quadratures and feature spaces are retained. No matched-time or speedup claim is made: OE is archived, MM newly computed.',
           '', 'Source conversion was derived from epsilon*v*f_x=rho-f+epsilon^2*q. With a=.25 cos(pi x), MM RHS is [a_prime/3, v*(rho_prime+a)+epsilon*(v^2-1/3)*a_prime, 0]. Both inflow boundaries use exact f at the physical signed velocities. P1/P2 source paths remain unchanged.',
           '', '| epsilon | method | E_f | E_rho | relative scaled flux | relative g |', '|---|---|---|---|---|---|']
    for r in summary:
        lines.append('| '+' | '.join([f"{r['epsilon']:g}",r['method']]+[f'{r[k]:.5e}' for k in metrics[:4]])+' |')
    lines+=['','All seedwise errors and ranges are retained. This P6 solution is angular-dependent, but its even part is exactly isotropic and its microscopic field is linear in v; it is not a general angular-anisotropy benchmark. Conclusions are restricted to this case and frozen equal-allocation budget.',
            '', 'Reproduction: run scripts/run_mm_aprfm_1d.py with --problem p6 --features 64 --epsilon {1,1e-3,1e-6} --seed {11,23,37} --output-dir results/p6_mm_comparison/mm, then python3 scripts/summarize_p6_mm_comparison.py. Original frozen tables have not been overwritten.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    for r in summary:print({k:r[k] for k in ['epsilon','method']+metrics[:4]})


if __name__=='__main__':main()
