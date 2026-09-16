"""Publication table export and component reference checks; no new solves."""
import csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/e6_mm_comparison'
def read(p):
    with p.open() as f:return list(csv.DictReader(f))
summary=read(OUT/'summary.csv');seeds=read(OUT/'seedwise.csv');refs=read(ROOT/'results/e6_periodic/reference_comparison.csv')
accept=[]
for r in seeds:
    if r['method']!='MM':continue
    ref=next(a for a in refs if a['run'].startswith('factorized_check/') and float(a['epsilon'])==float(r['epsilon']) and a['reference']==r['reference'] and float(a['t'])==float(r['t']))
    for k,delta in [('Ef','reference_delta_f'),('Erho','reference_delta_rho'),('Eq','reference_delta_q_absolute')]:
        # Reference q norm recovered from absolute/relative error of the OE row.
        d=float(ref[delta])
        if k=='Eq':d/=float(ref['Eq_absolute'])/float(ref['Eq_relative'])
        ratio=d/float(r[k]);accept.append(dict(seed=r['seed'],epsilon=r['epsilon'],t=r['t'],reference=r['reference'],metric=k,ratio=ratio,passed=ratio<.1))
with (OUT/'reference_acceptance.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(accept[0]));w.writeheader();w.writerows(accept)
tex=[r'\begin{table}[t]',r'\centering',r'\caption{E6 evolution errors at $t=0.2$, $\Delta t=0.002$. Medians over seeds 11, 23, and 37; 256 coefficients for both methods. CT and BE denote continuous-time and same-step backward-Euler transport references.}',r'\label{tab:e6-mm}',r'\begin{tabular}{llrrrr}',r'\toprule',r'$\varepsilon$ & Method & $E_f^{\rm CT}$ & $E_f^{\rm BE}$ & $E_\rho^{\rm BE}$ & $E_q^{\rm BE}$ \\',r'\midrule']
for i,r in enumerate(summary):
    if i==2:tex.append(r'\midrule')
    tex.append(' & '.join([r'$1$' if float(r['epsilon'])==1 else r'$10^{-3}$',r['method']+'-APRFM']+[f'{float(r[k]):.3e}' for k in ['Ef_CT','Ef_BE','Erho_BE','Eq_BE']])+r' \\')
tex.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}'])
(OUT/'main_table.tex').write_text('\n'.join(tex)+'\n')
doc=ROOT/'docs/e6_mm_comparison.md';lines=[doc.read_text().split('\n## 附录：末时刻逐seed诊断')[0],'\n## 附录：末时刻逐seed诊断\n','| epsilon | 方法 | seed | rank | Nrow | 最大质量漂移 | D0 | 独立宏观密度 BE 误差 |','|---|---|---|---|---|---|---|---|']
for r in seeds:
    if float(r['t'])!=.2 or r['reference']!='be':continue
    lines.append('| '+' | '.join(r[k] if k in ['method','seed','rank','Nrow'] or not r[k] else f'{float(r[k]):.6g}' for k in ['epsilon','method','seed','rank','Nrow','max_mass_drift','D0','Erho_macro'])+' |')
lines+=['\n主表完整[min,max]见summary.csv；t=.02/.10/.20的逐seed误差见seedwise.csv。OE的D0列不适用，不填0。\n',
         f"MM分量级参考检查：{sum(a['passed'] for a in accept)}/{len(accept)} 满足参考差异小于方法误差10%；详见reference_acceptance.csv。\n",
         '本次CT误差两法接近；同dt BE误差下本预算的OE更小。仅适用于这个预设问题与宏微特征分配，不推广为普遍优势。D0非零，因此物理密度与MM宏观场并非严格相同。\n',
         '不增加重复解剖面图或误差图；现有两张E6图保持不变。论文排版提供results/e6_mm_comparison/main_table.tex（booktabs三线表、统一科学计数法、无竖线），需加载booktabs宏包。未进行MM独立测试加密，不将OE的测试加密自动视为MM验收。\n']
grid_path=ROOT/'results/comparison_repairs_2026_09_10/mm_test_refinement.csv'
if grid_path.exists():
    grid=read(grid_path)
    lines.append(f"最新MM独立测试加密：{sum(r['passed']=='True' for r in grid)}/{len(grid)}行符合1%标准（含基准行），最大相对变化{max(float(r['relative_change']) for r in grid):.6g}。以上未加密的历史说明作废。全实验见experiments_current_summary_2026_09_10.md。")
doc.write_text('\n'.join(lines))
print('Table and appendix exported;',len(accept),'reference checks; failed',sum(not a['passed'] for a in accept))
