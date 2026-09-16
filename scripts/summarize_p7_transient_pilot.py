"""Summarize all tested methods, including failed regimes; render pilot figures."""
import csv
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/mpl_p7_transient')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/p7_transient_pilot/three_seed_J128'


def main():
    runs=json.loads((OUT/'runs.json').read_text())
    audits=json.loads((OUT/'reference_audit.json').read_text())
    rows=[]
    for eps in sorted({r['epsilon'] for r in runs},reverse=True):
        audit=next(a for a in audits if a['epsilon']==eps)
        for method in ('oe_ap','oe_unprojected','direct'):
            group=[r for r in runs if r['epsilon']==eps and r['method']==method]
            for k,t in enumerate((.05,.1)):
                row=dict(epsilon=eps,method=method,t=t,seeds=len(group),N_coef=group[0]['N_coef'],N_row=group[0]['N_row'])
                for metric in ('E_f','E_rho'):
                    values=[r['snapshots'][k][metric] for r in group]
                    row[metric]=float(np.median(values))
                    row[metric+'_min']=min(values); row[metric+'_max']=max(values)
                row['density_reference_discrepancy']=audit['density_refinement'][k]
                row['kinetic_reference_discrepancy']=audit['kinetic_refinement'][k]
                row['reference_check_pass']=bool(row['density_reference_discrepancy']<.1*row['E_rho'] and row['kinetic_reference_discrepancy']<.1*row['E_f'])
                row['compute_seconds']=float(np.median([r['compute_seconds'] for r in group]))
                rows.append(row)
    with (OUT/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lines=['# P7 transient pilot: measured results','',
           'Exploratory NumPy space-time tanh random-feature implementation of the projected/rescaled odd-even equations. This is an extension prototype, not the frozen steady solver or a reproduction of the authors\' APNN implementation.',
           '', 'Case: zero initial data, left inflow 1, right inflow 0, x in [0,1], t in [0,0.1]. Source: https://arxiv.org/html/2306.15381v4 (Problem 1, Case I); epsilon=1 and 0.01 are extensions.',
           '', 'J=128 per OE field; 256 coefficients for every method; seeds 11,23,37. Same physical interior/boundary/initial samples. OE AP has an additional 256 macro rows (5376 vs 5120); this is a coefficient-budget-controlled structural comparison, not identical residual cost. Direct RFM uses an unconstrained full-angular space and therefore also changes approximation structure.',
           '', 'Reference: backward Euler and staggered parity finite volumes. Coarse 128 cells/8 positive angles/400 steps; fine 256/16/800. Angular nodes are Gauss–Legendre on [0,1]. Refinement compares both density and full kinetic fields on the coarse grid; it is an empirical discrepancy check, not a rigorous error bound.',
           '', '| epsilon | method | t | E_f median | E_rho median [min,max] | reference check |', '|---|---|---|---|---|---|']
    for r in rows:
        lines.append(f"| {r['epsilon']:g} | {r['method']} | {r['t']} | {r['E_f']:.4g} | {r['E_rho']:.4g} [{r['E_rho_min']:.4g},{r['E_rho_max']:.4g}] | {r['reference_check_pass']} |")
    lines+=['', 'Interpretation: projected/rescaled OE retains percent-level accuracy in the two small-epsilon regimes; the two controls deteriorate substantially. The kinetic regime epsilon=1 fails accuracy requirements for all three global spaces, and its kinetic reference refinement is insufficient for an OE error report under the 10% rule. It must not be advertised as a successful all-regime benchmark.',
            '', 'Timing is exploratory: one solve per seed, BLAS threads fixed to one, a small OE library warm-up only, no three-repeat same-seed timing protocol. No speedup claim against the frozen S3 measurements, a deterministic solver, or published APNN is supported.',
            '', 'Validation: analytical feature derivatives checked by finite differences; exact parity checked; small-epsilon reference checked against an independent heat-equation Fourier series; discrete mass balance checked. Initial and inflow conditions are soft RF constraints and the incompatible space-time corner is excluded.',
            '', 'Next publication gate: improve/resolve kinetic-front and initial-corner approximation, independently refine space/time/angle, validate residuals on held-out points, and reproduce external APNN and deterministic baseline under one timing protocol. No current advantage over MM-APNN is established.',
            '', 'Reproduce: `python3 test/test_p7_transient_pilot.py`; `python3 scripts/run_p7_transient_pilot.py --features 128 --nx 128 --nt 400 --output results/p7_transient_pilot/three_seed_J128`; `python3 scripts/summarize_p7_transient_pilot.py`.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    fig,ax=plt.subplots(figsize=(6.6,4.1))
    labels={'oe_ap':'Projected/rescaled OE (pilot)','oe_unprojected':'Unprojected OE','direct':'Direct RFM'}
    for method in labels:
        group=sorted([r for r in rows if r['method']==method and r['t']==.1],key=lambda r:r['epsilon'])
        ax.errorbar([r['epsilon'] for r in group],[r['E_rho'] for r in group],yerr=np.array([[r['E_rho']-r['E_rho_min'] for r in group],[r['E_rho_max']-r['E_rho'] for r in group]]),marker='o',capsize=3,label=labels[method])
    ax.set(xscale='log',yscale='log',xlabel=r'$\varepsilon$',ylabel=r'Relative density error at $t=0.1$',title='Transient pilot: 256 coefficients, three seeds')
    ax.axhline(.01,color='gray',linestyle=':',linewidth=1)
    ax.legend(fontsize=8); ax.grid(alpha=.2); fig.tight_layout()
    fig.savefig(OUT/'density_error.png',dpi=180); fig.savefig(OUT/'density_error.pdf'); plt.close(fig)


if __name__=='__main__': main()
