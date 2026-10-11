"""Validate paired systems/fields and update the manuscript angular-domain table."""
import json,re
from pathlib import Path
from statistics import median
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/angular_domain_comparison'
records=[];checks=[]
for problem in ('p1','p3'):
 for seed in (11,23,37):
  dirs=[OUT/problem/k/str(seed) for k in ('reduced','expanded')]
  rr=[json.loads((p/'result.json').read_text()) for p in dirs]
  aa=[np.load(p/'system.npz') for p in dirs]
  for r in rr:records.append(dict(r))
  a,b=aa
  gram_delta=np.linalg.norm(a['A'].T@a['A']-b['A'].T@b['A'])/np.linalg.norm(a['A'].T@a['A'])
  rhs_delta=np.linalg.norm(a['A'].T@a['b']-b['A'].T@b['b'])/np.linalg.norm(a['A'].T@a['b'])
  np.testing.assert_allclose(a['b']@a['b'],b['b']@b['b'],rtol=1e-12)
  assert gram_delta<1e-12 and rhs_delta<1e-12
  assert rr[0]['rank']==rr[1]['rank'] and rr[0]['num_columns']==rr[1]['num_columns']==128
  fields=[np.load(next(p.glob('*oe_aprfm*.npz'))) for p in dirs]
  fa,fb=fields
  assert fa['f'].shape==fb['f'].shape
  for key in ('x','v','y','theta'):
   if key in fa:np.testing.assert_array_equal(fa[key],fb[key])
  field_delta=np.linalg.norm(fa['f']-fb['f'])/np.linalg.norm(fa['f'])
  assert field_delta<1e-8
  checks.append(dict(problem=problem,seed=seed,gram_relative_difference=float(gram_delta),rhs_relative_difference=float(rhs_delta),relative_field_difference=float(field_delta),ranks=[r['rank'] for r in rr],evaluation_shape=list(fa['f'].shape)))
summary=[]
for problem in ('p1','p3'):
 for domain in ('expanded','reduced'):
  rr=[r for r in records if r['problem']==problem and r['angular_domain']==domain]
  row=dict(problem=problem,domain=domain)
  for key in ('angular_locations','num_rows','num_columns','relative_l2_f','relative_l2_rho','assembly_seconds','solve_seconds'):
   row[key]=median(r[key] for r in rr)
  summary.append(row)
(OUT/'summary.json').write_text(json.dumps(dict(summary=summary,checks=checks),indent=2)+'\n')
def sci(x):
 s,e=f'{x:.3e}'.split('e');return '$'+s+r'\times10^{'+str(int(e))+'}$'
lines=[]
for r in summary:
 label='Full orbit' if r['domain']=='expanded' else ('Half range' if r['problem']=='p1' else 'First quadrant')
 lines.append(f"{'1D' if r['problem']=='p1' else '2D'} & {label} & {r['angular_locations']} & {r['num_rows']} & {sci(r['relative_l2_f'])} & {sci(r['relative_l2_rho'])} & {r['assembly_seconds']:.2f} & {r['solve_seconds']:.4f}"+r'\\')
caption=r'''Controlled angular-domain reduction for the manufactured problems at $\varepsilon=10^{-3}$, with 128 coefficients and seeds 11, 23 and 37 (medians). The full-orbit control evaluates the same parity residual blocks at all symmetry-related angular locations; it is a redundant expansion of the OE discretization. Its interior rows receive a factor $m^{-1/2}$ after row normalization ($m=2$ in 1D and $m=4$ in 2D), while physical inflow rows are unchanged. Thus the trial space, boundary data and weighted least-squares objective are matched. $N_\theta$ counts angular collocation locations per interior spatial point, not quadrature nodes. All solves use dense SVD with column equilibration and $\mathtt{rcond}=10^{-8}$; The one-dimensional problem uses $J^r=J^j=64$; the two-dimensional problem uses 32 features per component in the four-component representation. Assembly times include first-call execution/compilation; solve times cover column equilibration and SVD. Runs use serial fresh CPU processes with one BLAS thread. This test measures removal of redundant angular evaluations, not an accuracy advantage over a different full-angle method.'''
table='\n'.join([r'\begin{table}[!htb]',r'\centering\small',r'\setlength{\tabcolsep}{3pt}',r'\caption{'+caption+'}',r'\label{tab:angular-domain-reduction}',r'\resizebox{\linewidth}{!}{%',r'\begin{tabular}{llrrrrrr}',r'\toprule',r'Problem & Angular sampling & $N_\theta$ & $N_{\rm row}$ & $E_f$ & $E_\rho$ & $T_{\rm asm}$ (s) & $T_{\rm solve}$ (s)\\',r'\midrule',*lines,r'\bottomrule',r'\end{tabular}}',r'\end{table}'])
p=ROOT/'results/table.md';text=p.read_text();pattern=r'\\begin\{table\}.*?\\end\{table\}'
blocks=list(re.finditer(pattern,text,re.S));existing=next((b for b in blocks if r'\label{tab:angular-domain-reduction}' in b[0]),None)
if existing:text=text[:existing.start()]+table+text[existing.end():]
else:
 b=next(b for b in blocks if r'\label{tab:parity' in b[0]);text=text[:b.end()]+'\n\n'+table+text[b.end():]
old='The two-dimensional variable-scattering cutoff study uses physical-direction inflow traces and 58096 residual rows, as detailed in Table \\ref{tab:p5-cutoff}.'
addition=r' Angular-domain reduction is tested separately in Table \ref{tab:angular-domain-reduction}; its controlled configurations do not replace the main-study budgets listed here.'
if addition not in text:text=text.replace(old,old+addition)
p.write_text(text)
(OUT/'table.tex').write_text(table+'\n')
print(json.dumps(dict(summary=summary,checks=checks),indent=2))
