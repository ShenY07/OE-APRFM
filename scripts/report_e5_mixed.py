"""Common physical-quadrature evaluation; incomplete sweeps remain labelled partial."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/e5_mpl')
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/e5_mixed'

def relative(a,b,w=1):
    return float(np.sqrt(np.sum(w*(a-b)**2)/np.sum(w*b*b)))

def main():
    coarse=np.load(OUT/'reference/p2_parity_ref_eps_1e+00_level_A.npz')
    fine=np.load(OUT/'reference/p2_parity_ref_eps_1e+00_level_B.npz')
    x,v,w=fine['x'],fine['velocity'],fine['weights']/2
    fref=fine['f'];rho=fref@w;flux=fref@(w*v)
    spatial=np.stack([np.interp(x,coarse['x'],col) for col in coarse['f'].T],axis=1)
    interpolated=np.stack([np.interp(v,coarse['velocity'],row) for row in spatial])
    # Moments use each reference's native quadrature; compare spatial fields
    # on the coarse grid to avoid spatial endpoint extrapolation.
    cw=coarse['weights']/2
    crho=coarse['f']@cw
    cflux=coarse['f']@(cw*coarse['velocity'])
    # Acceptance uses ONE common grid and fine-reference denominator for
    # both reference differences and method errors. Moments remain native.
    def common_f(values):
        sx=np.stack([np.interp(coarse['x'],x,col) for col in values.T],axis=1)
        return np.stack([np.interp(coarse['velocity'],v,row) for row in sx])
    common_ref=common_f(fref)
    common_rho=np.interp(coarse['x'],x,rho)
    common_flux=np.interp(coarse['x'],x,flux)
    acceptance_delta={'Ef':relative(coarse['f'],common_ref,cw),
                      'Erho':relative(crho,common_rho),
                      'EF':relative(cflux,common_flux)}
    refinement={'Ef':relative(interpolated,fref,w),
                'Erho':relative(np.interp(coarse['x'],x,rho),crho),
                'EF':relative(np.interp(coarse['x'],x,flux),cflux)}
    rows=[]
    for path in sorted(OUT.glob('*/metadata.json')):
        meta=json.loads(path.read_text())
        files=list(path.parent.glob('p2_*.npz'))
        if len(files)!=1:raise ValueError(f'Missing or ambiguous predictions: {path}')
        data=np.load(files[0]);f=data['f']
        np.testing.assert_allclose(data['x'],x)
        np.testing.assert_allclose(data['velocity'],v)
        errors={'Ef':relative(f,fref,w),'Erho':relative(f@w,rho),'EF':relative(f@(w*v),flux)}
        row=dict(method=meta['method_label'],seed=meta['seed'],Nang_positive=meta['positive_angular_budget'],
                 Ncoef=meta['num_columns'],Nrow=meta['num_rows'],**errors,
                 cost_seconds=sum(meta[k] for k in ('feature_seconds','assembly_seconds','solve_seconds')))
        for key,value in refinement.items():
            row['reference_delta_'+key]=value
        acceptance_errors={'Ef':relative(common_f(f),common_ref,cw),
                           'Erho':relative(np.interp(coarse['x'],x,f@w),common_rho),
                           'EF':relative(np.interp(coarse['x'],x,f@(w*v)),common_flux)}
        for key,value in acceptance_delta.items():
            row['acceptance_delta_'+key]=value
            row['acceptance_method_error_'+key]=acceptance_errors[key]
            row['reference_ratio_'+key]=value/acceptance_errors[key]
            row['reference_pass_'+key]=value<.1*acceptance_errors[key]
        # A coarse-grid check cannot certify the fine-grid main-table error.
        # Retain the original columns for compatibility, but make their scope
        # explicit and report the main-grid f diagnostic separately.
        row['reference_acceptance_scope']='common_coarse_grid_only'
        row['main_grid_reference_ratio_Ef']=refinement['Ef']/errors['Ef']
        row['main_grid_reference_pass_Ef']=refinement['Ef'] < .1*errors['Ef']
        row['main_grid_reference_status']='interpolated refinement diagnostic, not rigorous certification'
        row['density_definition']='physical: angular mean of reconstructed f'
        row['D0_status']='unavailable: separate macro/g and coefficients not archived'
        row['Rtr_status']='unavailable: exact field derivative not archived'
        rows.append(row)
    if not rows:return
    with (OUT/'physical_metrics.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=[]
    for method in ('oe','mm'):
        for n in (8,16,32,64):
            group=[r for r in rows if r['method']==method and r['Nang_positive']==n]
            if not group:continue
            r=dict(method=method,Nang_positive=n,seeds=len(group))
            for key in ('Nrow','Ef','Erho','EF','cost_seconds'):
                values=[a[key] for a in group]
                r[key]=float(np.median(values));r[key+'_min']=min(values);r[key+'_max']=max(values)
            summary.append(r)
    with (OUT/'summary.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    fig,axes=plt.subplots(1,2,figsize=(9,3.6))
    for method in ('oe','mm'):
        group=[r for r in summary if r['method']==method]
        if not group:continue
        for ax,key in zip(axes,('Ef','Erho')):
            xx=[r['Nrow'] for r in group];yy=[r[key] for r in group]
            ax.loglog(xx,yy,'o-',label=method.upper())
            ax.fill_between(xx,[r[key+'_min'] for r in group],[r[key+'_max'] for r in group],alpha=.15)
            ax.set(xlabel='Actual residual rows',ylabel=key);ax.grid(alpha=.2);ax.legend()
    fig.suptitle(f'E5: {len(rows)}/24 runs; quadrature acceptance pending')
    fig.tight_layout()
    for ext in ('pdf','png'):fig.savefig(OUT/f'angular_budget.{ext}',dpi=180)
    print(json.dumps({'completed':len(rows),'expected':24,'reference_refinement':refinement},indent=2))

if __name__=='__main__':main()
