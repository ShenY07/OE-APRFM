"""Derive balanced target-accuracy costs from the existing S3 aggregate CSV.

This is a discrete observed-budget summary, not an interpolated time-to-solution
or a seedwise success guarantee. Original measurements are never modified.
"""
import csv
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/requirement_2026_08_20/tables_frozen'


def render():
    with (OUT/'table_S3_accuracy_cost_sweep.csv').open() as f:
        rows=list(csv.DictReader(f))
    methods=list(dict.fromkeys(r['method'] for r in rows))
    body=[]; records=[]
    for problem in ('p1','p3'):
        for tol in (1e-2,1e-3,1e-4):
            for method in methods:
                qualifying=[r for r in rows if r['problem']==problem and r['method']==method and max(float(r['E_f']),float(r['E_rho']))<=tol]
                best=min(qualifying,key=lambda r:float(r['solve_time'])) if qualifying else None
                smallest=min(qualifying,key=lambda r:(int(r['model_size']),float(r['solve_time']))) if qualifying else None
                n_tau=smallest['model_size'] if smallest else ''
                records.append(dict(problem=problem,tolerance=tol,method=method,budget=best['budget'] if best else '',compute_seconds=best['solve_time'] if best else '',N_tau=n_tau,N_tau_budget=smallest['budget'] if smallest else '',status='attained' if best else 'not attained in tested budgets'))
                name='1D' if problem=='p1' else '2D'
                power=int(round(__import__('math').log10(tol)))
                budget=best['budget'] if best else '---'
                seconds=f"{float(best['solve_time']):.2f}" if best else '---'
                body.append(f"{name} & $10^{{{power}}}$ & {method} & {budget} & {seconds} & {n_tau or '---'}"+r"\\")
            body.append(r'\addlinespace')
    caption=(r'Observed cost at a common accuracy target for the budgets in Table~\ref{tab:supp-efficiency}, at $\varepsilon=10^{-3}$. '
             r'A budget qualifies when both median errors satisfy $E_f,E_\rho\leq\tau$; deterministic entries use their single-run errors. '
             r'The least reported computation time among qualifying tested budgets is shown. A dash denotes no qualifying tested budget, not impossibility. '
             r'$N_\tau$ is the smallest model size among qualifying tested budgets, selected independently of minimum time. Model size counts coefficients, trainable network parameters, or discrete unknowns as in Table~\ref{tab:supp-efficiency}; it is not memory usage. '
             r'No interpolation or extrapolation is used; these are aggregate-budget comparisons, not seedwise success probabilities or measured stopping times. '
             r'Timing inherits the accounting and hardware limitations of Table~\ref{tab:supp-efficiency}.')
    tex='% BEGIN GENERATED S3 TARGET ACCURACY\n'+r'\begin{table}[t]'+'\n'+r'\centering'+'\n'+r'\caption{'+caption+'}\n'+r'\label{tab:supp-target-accuracy}'+'\n'+r'\begin{tabular}{lcllrr}'+'\n'+r'\toprule'+'\n'+r'Problem & $\tau$ & Method & Fastest budget & $T_{\rm comp}$ (s) & $N_\tau$\\'+'\n'+r'\midrule'+'\n'+'\n'.join(body)+'\n'+r'\bottomrule'+'\n'+r'\end{tabular}'+'\n'+r'\end{table}'+'\n% END GENERATED S3 TARGET ACCURACY\n'
    (OUT/'table_S3a.tex').write_text(tex)
    with (OUT/'table_S3a_target_accuracy.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    # Separate f/density targets are diagnostic alternatives; retain all methods.
    sizes=[]
    for problem in ('p1','p3'):
        for metric in ('E_f','E_rho'):
            for tol in (1e-2,1e-3,1e-4):
                for method in methods:
                    qualifying=[r for r in rows if r['problem']==problem and r['method']==method and float(r[metric])<=tol]
                    best=min(qualifying,key=lambda r:(int(r['model_size']),float(r['solve_time']))) if qualifying else None
                    sizes.append(dict(problem=problem,metric=metric,tolerance=tol,method=method,N_tau=best['model_size'] if best else '',budget=best['budget'] if best else '',time_at_N_tau=best['solve_time'] if best else ''))
    with (OUT/'target_model_size_by_metric.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(sizes[0]));writer.writeheader();writer.writerows(sizes)
    return tex


def update():
    tex=render()
    for name in ('all_available_tables.tex','supplement_tables.tex'):
        path=OUT/name
        text=path.read_text()
        text=re.sub(r'% BEGIN GENERATED S3 TARGET ACCURACY\n.*?% END GENERATED S3 TARGET ACCURACY\n*','',text,flags=re.S)
        marker='% Table S4:'
        index=text.find(marker)
        text=text[:index]+tex+'\n'+text[index:] if index>=0 else text.rstrip()+'\n\n'+tex
        path.write_text(text)


if __name__=='__main__': update()
