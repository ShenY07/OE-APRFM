"""Isolated OE/MM workers for the prescribed fixed physical mixed slab."""
import os
os.environ.setdefault('JAX_ENABLE_X64','true')
os.environ.setdefault('JAX_PLATFORMS','cpu')
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
OUT=ROOT/'results/e5_mixed'


def config(epsilon=1.):
    from configuration.p2_heterogeneous_1d import get_config
    import jax.numpy as jnp
    c=get_config(1.)
    c.model.coeff.scattering=lambda x:1/(.01+.5*(jnp.tanh(6.5-11*x)+jnp.tanh(11*x-4.5)))
    c.model.coeff.absorption=lambda x:0*x
    c.model.source=lambda x,v:0*v
    c.model.num_quads=64
    return c


def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['oe','mm','references','all'],default='all')
    p.add_argument('--seed',type=int,default=11);p.add_argument('--angles',type=int,default=8)
    a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    refdir=OUT/'reference';refdir.mkdir(exist_ok=True)
    if a.method=='all':
        subprocess.run([sys.executable,__file__,'--method','references'],check=True)
        for method in ('oe','mm'):
            for seed in (11,23,37):
                for angles in (8,16,32,64):
                    result=subprocess.run([sys.executable,__file__,'--method',method,'--seed',str(seed),'--angles',str(angles)])
                    if result.returncode:
                        failed=OUT/f'{method}_s{seed}_a{angles}'
                        failed.mkdir(parents=True,exist_ok=True)
                        (failed/'failure.json').write_text(json.dumps(dict(returncode=result.returncode,
                            note='Worker failed; other preset runs continue. See execution log.'))+'\n')
        return
    if a.method=='references':
        from numerical.parity_reference import solve_parity_gmres_1d
        for level,grid in [('A',(256,64)),('B',(512,128))]:
            path=refdir/f'p2_parity_ref_eps_1e+00_level_{level}.npz'
            if path.exists():continue
            r=solve_parity_gmres_1d(config(),grid=grid,tol=1e-11)
            if not r['converged']:raise RuntimeError('E5 reference failed to converge')
            np.savez_compressed(path,**r)
        return
    directory=OUT/f'{a.method}_s{a.seed}_a{a.angles}'
    diagnostic=os.environ.get('E5_FIELD_DIAGNOSTICS')=='1'
    old_directory=directory
    if diagnostic:
        directory=OUT/'diagnostic_replay'/directory.name
        from e5_field_diagnostics import install,evaluate
    if (directory/'metadata.json').exists():return
    if a.method=='oe':
        import run_p2_oe_aprfm as runner
        from run_e3_slab import symmetric
        runner.function_space.uniform=symmetric
        runner.get_config=config
        if diagnostic: state=install(runner,'oe')
        r=runner.run(1.,a.seed,directory,refdir,partitions=(2,4),features=72,
                     rcond=1e-6,collocation=(128,a.angles),reference_level='B')
    else:
        os.environ.setdefault('MM_ASSEMBLY_BATCH_SIZE','16')
        # Import vendor before any current modules.* imports (isolated worker).
        import run_mm_aprfm_1d as runner
        import jax.numpy as jnp
        from scipy.linalg import lstsq
        def symmetric(scale):
            return lambda key,shape,dtype=jnp.float64:runner.random.uniform(key,shape,dtype,minval=-scale,maxval=scale)
        runner.mm_function_space.uniform=symmetric
        runner.get_p2_config=config
        # Match the actual OE spatial points and mirrored midpoint velocities,
        # including the same 128 inflow points on each side.
        class MatchedSample:
            def __init__(self,domain,sizes,mode):
                xx=jnp.linspace(0.,1.,128)[1:-1]
                positive=(jnp.arange(a.angles)+.5)/a.angles
                vv=jnp.concatenate((-positive[::-1],positive))
                X,V=jnp.meshgrid(xx,vv,indexing='ij')
                self.pts_int=(X.reshape(-1,1),V.reshape(-1,1))
                boundary=jnp.linspace(0.,1.,129)[1:,None]
                self.pts_left=(jnp.zeros_like(boundary),boundary)
                self.pts_right=(jnp.ones_like(boundary),-boundary)
        runner.Sample1D=MatchedSample
        # Existing MM runner uses ROOT/results/references for the p2 branch.
        # Redirect its read-only reference lookup into this experiment.
        runner.ROOT=OUT/'mm_reference_root'
        target=runner.ROOT/'results/references';target.mkdir(parents=True,exist_ok=True)
        link=target/'p2_parity_ref_eps_1e+00_level_B.npz'
        if not link.exists():link.symlink_to(refdir/'p2_parity_ref_eps_1e+00_level_B.npz')
        # Match OE column equilibration as well as row normalization/cutoff.
        def equilibrated(A,b,**kw):
            scale=np.linalg.norm(A,axis=0);scale=np.where(scale>1e-14,scale,1.)
            c,res,rank,s=lstsq(A/scale,b,**kw)
            return c/scale,res,rank,s
        runner.lstsq=equilibrated
        original=runner.MicroMacroPointwiseInteriorConstraint1D
        def equation(**kw):
            kw['num_quads']=64
            return original(**kw)
        runner.MicroMacroPointwiseInteriorConstraint1D=equation
        if diagnostic: state=install(runner,'mm')
        r=runner.run('p2',1.,a.seed,directory,partitions=(2,4),rho_features=64,g_features=128,
                     rcond=1e-6,collocation=(128,2*a.angles+2))
    assert r['num_columns']==1152
    metadata=dict(r,experiment='E5',initialization='U(-1,1)',
        physical_scattering='1/(.01+.5*(tanh(6.5-11*x)+tanh(11*x-4.5)))',
        positive_angular_budget=a.angles,operator_quadrature=64,
        assembly_batch_size=int(os.environ.get('MM_ASSEMBLY_BATCH_SIZE','0')) if a.method=='mm' else 0,
        angular_sampling='midpoints; MM uses exactly mirrored OE directions',
        spatial_interior_count=126,boundary_count_per_side=128,
        reference_refinement_status='pending',method_label=a.method,
        note='p2 filename is reused runner plumbing; this is E5 physics, not P2')
    if diagnostic:evaluate(state,a.method,directory,old_directory)
    (directory/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(a.method,a.seed,a.angles,r['relative_l2_f'],r['relative_l2_rho'],flush=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/report_e5_mixed.py')],check=True)


if __name__=='__main__':main()
