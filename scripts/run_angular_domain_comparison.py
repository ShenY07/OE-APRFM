"""Controlled symmetry-expanded versus representative-angle OE experiment.

The full-domain control evaluates the same local parity residual blocks at
all orbit members. It is intentionally redundant, not an independent direct-f
method. Inverse-multiplicity weights preserve the reduced least-squares metric.
Run each case in a fresh process with single-thread CPU float64.
"""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.linalg import lstsq
import jax.numpy as jnp


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--problem', choices=['p1','p3'], required=True)
    p.add_argument('--domain', choices=['reduced','expanded'], required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a=p.parse_args(); out=a.output_dir; out.mkdir(parents=True,exist_ok=True)
    import run_p1_oe_aprfm as one
    import run_p3_oe_aprfm as two
    runner=one if a.problem=='p1' else two
    orbit=2 if a.problem=='p1' else 4
    multiplicity=orbit if a.domain=='expanded' else 1
    state={}; matrices=[]; rhs=[]
    original_sample=runner.Sample1D if a.problem=='p1' else runner.Sample2D
    def sample(*args,**kwargs):
        s=original_sample(*args,**kwargs)
        sides=['pts_left','pts_right'] if a.problem=='p1' else ['pts_left','pts_right','pts_lower','pts_upper']
        state['boundary_rows']=sum(len(getattr(s,k)[0]) for k in sides)
        pts=s.pts_int
        state['reduced_interior_points']=len(pts[0])
        state['reduced_angular_locations']=len(np.unique(np.asarray(pts[-1])))
        if multiplicity>1:
            angle=pts[-1]
            reflected=[angle,-angle] if orbit==2 else [angle,jnp.pi-angle,jnp.pi+angle,2*jnp.pi-angle]
            s.pts_int=tuple([jnp.tile(x,(orbit,1)) for x in pts[:-1]]+[jnp.concatenate(reflected)])
        return s
    if a.problem=='p1': runner.Sample1D=sample
    else: runner.Sample2D=sample
    # Evaluate orbit representatives in their local parity coordinates. This
    # keeps all equation blocks, feature parameters and RHS definitions fixed.
    def fold(t):
        if orbit==2:return jnp.abs(t)
        t=jnp.mod(t,jnp.pi)
        return jnp.minimum(t,jnp.pi-t)
    if a.problem=='p3':
        original=two.V2InteriorConstraint2D
        class Interior(original):
            def __call__(self,x,y,t):return super().__call__(x,y,fold(t))
        two.V2InteriorConstraint2D=Interior
        original_source=two.make_source_evaluator
        def source(*args,**kwargs):
            f=original_source(*args,**kwargs)
            return lambda x,y,t:f(x,y,fold(t))
        two.make_source_evaluator=source
    def normalize(A,b,factors=1):
        A=np.asarray(A,dtype=float).copy();b=np.asarray(b,dtype=float).reshape(-1).copy()
        n=np.linalg.norm(A,axis=1);n=np.where(n>1e-30,n,1.)
        return A/n[:,None]*np.asarray(factors)[...,None], b/n*np.asarray(factors)
    def solve_arrays(A,b):
        t=perf_counter(); scale=np.linalg.norm(A,axis=0);scale=np.where(scale>1e-14,scale,1.)
        c,_,rank,s=lstsq(A/scale,b,cond=1e-8,lapack_driver='gelsd');c=c/scale
        state.update(rank=int(rank),solve_seconds=perf_counter()-t,num_rows=len(b),num_columns=A.shape[1])
        np.savez_compressed(out/'system.npz',A=A,b=b,coefficients=c)
        return c[:,None],dict(rank=int(rank),condition_number=float(s[0]/s[rank-1]),coefficient_norm=float(np.linalg.norm(c)), normalized_residual_rms=float(np.linalg.norm(A@c-b)/np.sqrt(len(b))),largest_singular_value=float(s[0]),smallest_effective_singular_value=float(s[rank-1]),singular_value_threshold=float(1e-8*s[0]),num_rows=len(b),num_columns=A.shape[1])
    if a.problem=='p1':
        def solve(A,b,**kwargs):
            factors=np.ones(len(b));factors[state['boundary_rows']:]/=np.sqrt(multiplicity)
            A,b=normalize(A,b,factors);return solve_arrays(A,b)
        one.solve=solve
        result=one.run(.001,a.seed,out,features=64,collocation=(32,16),rcond=1e-8)
    else:
        class Dense:
            def __init__(self,*args,**kwargs):self.num_rows=0
            def add(self,A,b,row_factors=1):
                boundary=self.num_rows<state['boundary_rows']
                f=np.asarray(row_factors)/(1 if boundary else np.sqrt(multiplicity))
                A,b=normalize(A,b,f);matrices.append(A);rhs.append(b);self.num_rows+=len(b)
            def solve(self):return solve_arrays(np.vstack(matrices),np.concatenate(rhs))
        two.StreamingQRLeastSquares=Dense
        result=two.run(.001,a.seed,out,'p3',features=32,collocation=(8,8,8),rcond=1e-8)
    result.update(state,angular_domain=a.domain,orbit_multiplicity=multiplicity,
        angular_locations=state['reduced_angular_locations']*multiplicity,
        protocol='same_basis_same_weighted_objective_symmetry_expanded_control',
        solver='dense gelsd; row L2 then inverse-orbit sqrt weights then column L2',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
