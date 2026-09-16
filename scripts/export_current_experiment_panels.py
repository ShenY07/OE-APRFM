"""Reproducible standalone E1--E6 panels; no new solves or reference substitution."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/oe_current_panels_mpl')
import csv,json,hashlib,zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/figures/current_e1_e6_2026_09_10'
OUT.mkdir(parents=True,exist_ok=True)
BLUE='#0072B2';ORANGE='#D55E00';GREEN='#009E73';PURPLE='#CC79A7'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':10,
 'legend.fontsize':8,'lines.linewidth':1.5,'lines.markersize':4,
 'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,
 'savefig.dpi':220})
manifest=[];sources=set()
def read(path):
    sources.add(path)
    with (ROOT/path).open() as f:return list(csv.DictReader(f))
def val(rows,key):return np.array([float(r[key]) for r in rows])
def axes(xlabel,ylabel):
    fig,ax=plt.subplots(figsize=(3.8,3.1),layout='constrained')
    ax.set(xlabel=xlabel,ylabel=ylabel);ax.tick_params(direction='out')
    ax.grid(which='major',alpha=.16,linewidth=.5)
    return fig,ax
def save(fig,ax,name,note):
    # Validate rendered labels, independently of ASCII data keys/file names.
    labels=[ax.get_xlabel(),ax.get_ylabel(),*ax.get_legend_handles_labels()[1]]
    assert not ax.get_title(), 'Standalone panels must not have titles'
    for label in labels:
        for word in ('rho','epsilon','beta','Delta'):
            assert word not in label.replace('\\varepsilon','').replace('\\'+word,''), f'Unformatted mathematical symbol: {label}'
    old_name=name
    # File names must not depend on case alone to distinguish f from flux F.
    replacements={
        'E2_parity_budget_f':'E2_parity_budget_distribution_error',
        'E2_parity_budget_rho':'E2_parity_budget_density_error',
        'E3_slab_E_diff':'E3_slab_diffusion_density_error',
        'E3_slab_E_q_diff':'E3_slab_diffusion_scaled_current_error',
        'relative_l2_f':'distribution_relative_L2_error',
        'relative_l2_rho':'density_relative_L2_error',
        'residual_half':'normalized_training_residual_RMS',
        'budget_dependence_Ef_':'budget_dependence_distribution_error_',
        'budget_dependence_Erho_':'budget_dependence_density_error_',
        'budget_dependence_EF_':'budget_dependence_physical_flux_error_',
        'fixed_collocation_feature_error':'fixed_collocation_distribution_error_vs_features',
        '_transport_error_three_times':'_distribution_error_three_times',
    }
    for before,after in replacements.items():name=name.replace(before,after)
    if old_name!=name:
        for ext in ('pdf','png'):
            old=OUT/f'{old_name}.{ext}';new=OUT/f'{name}.{ext}'
            if old.exists() and not new.exists():old.rename(new)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False,loc='lower center',bbox_to_anchor=(.5,1.02),ncol=2,handlelength=1.7,columnspacing=1)
    for ext in ('pdf','png'):fig.savefig(OUT/f'{name}.{ext}',bbox_inches='tight')
    plt.close(fig);manifest.append((name,note))
def curve(ax,rows,x,y,label,color,lo=None,hi=None):
    rows=sorted(rows,key=lambda r:float(r[x]));xx=val(rows,x);yy=val(rows,y)
    ax.plot(xx,yy,'o-',label=label,color=color)
    if lo and hi:ax.fill_between(xx,val(rows,lo),val(rows,hi),color=color,alpha=.14,linewidth=0)

e1=read('results/minimum_gain/quadrature_check.csv')
fig,ax=axes(r'$\varepsilon$',r'$\beta_{\varepsilon,m}$')
curve(ax,[r for r in e1 if float(r['epsilon'])>0],'epsilon','beta','OE',BLUE)
ax.set_xscale('log');ax.axhline(float(e1[-1]['beta']),color=ORANGE,ls='--',label=r'$\varepsilon=0$')
save(fig,ax,'E1_fixed_space_minimum_gain','32-dimensional fixed space, no truncation; epsilon=0 drawn as horizontal limit, not on log axis.')
e2=read('results/requirement_2026_08_20/tables_frozen/table_S4_parity_feature_budget.csv')
for metric in ('f','rho'):
    fig,ax=axes(r'$N_{\mathrm{coef}}$',{'f':r'$E_f$','rho':r'$E_{\rho}$'}[metric])
    for label,color in [('Parity-constrained',BLUE),('Unconstrained',ORANGE)]:
        curve(ax,[r for r in e2 if r['representation']==label],'Ncoef','median_E_'+metric,label,color,'min_E_'+metric,'max_E_'+metric)
    ax.set_xscale('log',base=2);ax.set_yscale('log')
    save(fig,ax,f'E2_parity_budget_{metric}','Same J matched rows/coefficients; rows vary across J. Three-seed median and range; old initialization.')
from feature_resolution_data import feature_records
for problem in ('p1','p3'):
    fig,ax=axes(r'$N_{\mathrm{coef}}$',r'$E_f$')
    rr=[r for r in feature_records() if r['problem']==problem]
    for r in rr:sources.add(r['source'])
    grouped=[]
    for n in sorted({r['num_columns'] for r in rr}):
        a=[r['relative_l2_f'] for r in rr if r['num_columns']==n]
        grouped.append(dict(n=n,e=np.median(a),lo=min(a),hi=max(a)))
    curve(ax,grouped,'n','e','OE',BLUE,'lo','hi');ax.set_xscale('log',base=2);ax.set_yscale('log')
    save(fig,ax,f'E2_{problem}_fixed_collocation_feature_error','Audited fixed-collocation records; P3 four-component implementation.')
e3=read('results/e3_slab/summary.csv')
for key,label in [('E_diff',r'$E_{\rho,\mathrm{diff}}$'),('E_q_diff',r'$E_{q,\mathrm{diff}}$'),('fick_defect',r'$D_{\mathrm{Fick}}$')]:
    fig,ax=axes(r'$\varepsilon$',label);curve(ax,e3,'epsilon',key,'OE',BLUE,key+'_min',key+'_max')
    ax.set_xscale('log');ax.set_yscale('log');save(fig,ax,'E3_slab_'+key,'Fixed budget, three-seed median/range. Nonmanufactured slab.')
p5=[]
for p in sorted((ROOT/'results/comparison_repairs_2026_09_10').glob('p5_*/*.json')):
    sources.add(str(p.relative_to(ROOT)));p5.append(json.loads(p.read_text()))
for key,label in [('relative_l2_f',r'$E_f$'),('relative_l2_rho',r'$E_\rho$'),('rank',r'$r_{\mathrm{eff}}$'),('coefficient_norm',r'$\|c\|_2$'),('residual_half','Training residual RMS')]:
    fig,ax=axes('SVD relative cutoff',label);curve(ax,p5,'rcond',key,'OE',BLUE);ax.set_xscale('log')
    if key!='rank':ax.set_yscale('log')
    save(fig,ax,'E4_P5_seed11_same_version_'+key,'epsilon=1e-3; same-version cutoff sensitivity, not three-seed main results. Residual is complete normalized training residual, not independent physical residual; reference adequacy remains to be checked.')
e5=read('results/e5_mixed/summary.csv')
for key,label in [('Ef',r'$E_f$'),('Erho',r'$E_\rho$'),('EF',r'$E_F$')]:
    fig,ax=axes(r'$N_{\mathrm{row}}$',label)
    for method,color in [('oe',BLUE),('mm',ORANGE)]:
        curve(ax,[r for r in e5 if r['method']==method],'Nrow',key,method.upper(),color,key+'_min',key+'_max')
    ax.set_xscale('log',base=2);ax.set_yscale('log')
    save(fig,ax,'E5_budget_dependence_'+key+'_old_reference_pending_reassessment','All eight presets retained. Original reference; new references not yet postprocessed. Actual rows include repeated equations and changing block weights, not independent information. Positive budgets OE/MM each 8,16,32,64 in order.')
timing=[]
for method in ('oe','mm'):
    for n in (32,64):
        a=[]
        for p in (ROOT/'results/comparison_repairs_2026_09_10').glob(f'timing_{method}_{n}_*/*/metadata.json'):
            sources.add(str(p.relative_to(ROOT)));r=json.loads(p.read_text());a.append(sum(r[k] for k in ('feature_seconds','assembly_seconds','solve_seconds')))
        assert len(a)==3
        timing.append(dict(method=method,n=n,t=np.median(a),lo=min(a),hi=max(a)))
fig,ax=axes('Positive angular nodes','Recorded wall time (s)')
for method,color in [('oe',BLUE),('mm',ORANGE)]:curve(ax,[r for r in timing if r['method']==method],'n','t',method.upper(),color,'lo','hi')
ax.set_xticks([32,64]);save(fig,ax,'E5_seed11_repeated_wall_time_not_speedup','Seed11 only, three repetitions, feature+assembly+solve. Serial queue does not guarantee isolated host. Not paired with three-seed median errors.')
e6=read('results/e6_mm_comparison/seedwise.csv')
for eps in (1.,.001):
    for kind in ('ct','be'):
        fig,ax=axes(r'$t$',r'$E_f^{\mathrm{'+kind.upper()+'}}$')
        for method,color in [('OE',BLUE),('MM',ORANGE)]:
            grouped=[]
            for t in (.02,.1,.2):
                a=[float(r['Ef']) for r in e6 if r['method']==method and float(r['epsilon'])==eps and r['reference']==kind and float(r['t'])==t]
                assert len(a)==3;grouped.append(dict(t=t,e=np.median(a),lo=min(a),hi=max(a)))
            curve(ax,grouped,'t','e',method,color,'lo','hi')
        ax.set_yscale('log');ax.set_xticks([.02,.1,.2])
        save(fig,ax,f'E6_eps{eps:.0e}_{kind}_transport_error_three_times','Three actually evaluated output times only; seed median/range. CT includes time error; BE compares same time step. CT curves may overlap.')
acc=read('results/e6_periodic/accuracy_summary.csv')
fig,ax=axes(r'$\varepsilon$',r'$E_{\rho,\mathrm{diff}}^{\mathrm{BE}}$')
for t,color in [(.02,BLUE),(.1,ORANGE),(.2,GREEN)]:
    curve(ax,[r for r in acc if float(r['t'])==t and r['reference']=='be'],'epsilon','Ediff_BE',f'$t={t:g}$',color,'Ediff_BE_min','Ediff_BE_max')
ax.set_xscale('log');ax.set_yscale('log');save(fig,ax,'E6_discrete_diffusion_fixed_dt','Difference from BE diffusion, not RF error or CT total error.')
tr=read('results/e6_periodic/time_refinement.csv')
fig,ax=axes(r'$\Delta t$',r'$E_f^{\mathrm{CT}}$')
for eps,color in [(1.,BLUE),(.001,ORANGE)]:curve(ax,[r for r in tr if float(r['epsilon'])==eps and r['reference']=='ct'],'dt','Ef',r'$\varepsilon='+str(eps)+'$',color)
ax.set_xscale('log',base=2);ax.set_yscale('log');save(fig,ax,'E6_time_refinement_seed11_CT','T=0.2 fixed, 50/100/200 BE steps; seed11 only.')
for eps in (1.,.001):
    path=next((ROOT/'results/e6_periodic/factorized_check').glob(f's11_eps{eps:.0e}_dt2e-03_J128_q64/fields.npz'))
    sources.add(str(path.relative_to(ROOT)));d=np.load(path);x=d['x']
    for key,label in [('rho',r'$\rho$'),('q',r'$q$')]:
        fig,ax=axes(r'$x$',label)
        for step,color in [(0,'#666666'),(10,BLUE),(50,ORANGE),(100,GREEN)]:
            ax.plot(x,d[f'{key}_{step}'],color=color,label=f'$t={step*.002:g}$')
        save(fig,ax,f'E6_eps{eps:.0e}_seed11_{key}_evolution','OE seed11 actual saved fields; q(0)=0 retained. Time colors consistent across field panels; not a multi-method comparison.')
    fig,ax=axes(r'$t$','Cosine amplitude')
    refpath=ROOT/f'results/e6_periodic/references/eps{eps:.0e}_dt2e-03_n256.npz';sources.add(str(refpath.relative_to(ROOT)));ref=np.load(refpath)
    _,w=np.polynomial.legendre.leggauss(256);w=w/2
    ts=[0,.02,.1,.2];am=[2*np.mean((d[f'rho_{int(round(t/.002))}']-1)*np.cos(2*np.pi*x)) for t in ts]
    ax.plot(ts,am,'o-',color=BLUE,label='OE')
    for kind,color,style in [('ct',ORANGE,'--'),('be',GREEN,':')]:
        aa=[.2]+[.2*float(np.real(ref[f'{kind}_{t:.2f}'][:256]@w)) for t in ts[1:]]
        ax.plot(ts,aa,style,color=color,label=kind.upper(),marker='x')
    save(fig,ax,f'E6_eps{eps:.0e}_seed11_cosine_amplitude_CT_BE','Saved output times only; native reference angular moments. OE and BE may overlap; amplitude exposes background-masked time error.')
notes=['# Current standalone experiment panels','', 'Blue: OE/parity; orange: MM/unconstrained. Bands show min/max, not confidence intervals. No panel titles; labels identify axes only.','', 'Only PDFs produced by this script enter the ZIP; old incompatible P7 and archived plots are excluded. E4 is a sensitivity appendix, E5 uses the original reference. E6 error-evolution panels are supplementary, not a replacement for the compact four-row comparison table.','']
notes += ['Naming: distribution_error = Ef; density_error = Erho; physical_flux_error = EF; scaled_current_error = Eq. File names do not distinguish physical quantities by letter case alone.', '']
notes += [f'- `{name}.pdf`: {note}' for name,note in manifest]
notes += ['', '## Sources']+[f'- `{p}`' for p in sorted(sources)]
(OUT/'MANIFEST.md').write_text('\n'.join(notes)+'\n')
archive=OUT.parent/'current_e1_e6_2026_09_10_pdfs.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for name,_ in manifest:z.write(OUT/f'{name}.pdf',arcname=f'{name}.pdf')
    z.write(OUT/'MANIFEST.md',arcname='MANIFEST.md')
with zipfile.ZipFile(archive) as z:assert z.testzip() is None
print(json.dumps(dict(panels=len(manifest),zip=str(archive),bytes=archive.stat().st_size),indent=2))
