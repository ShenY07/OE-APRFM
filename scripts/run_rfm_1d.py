"""Run a matched original-equation 1-D RFM baseline."""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
from time import perf_counter
os.environ.setdefault("JAX_ENABLE_X64", "true")
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from configuration.p1_manufactured_1d import get_config as p1_config
from configuration.p2_heterogeneous_1d import get_config as p2_config
sys.path.insert(0,str(ROOT/"baselines/mm_oerfm/original/src"))
import jax, jax.numpy as jnp, numpy as np
from jax import jit,random,vmap
from scipy.linalg import lstsq
from constraints.continuous1d import PointwiseBoundaryConstraint1D,PointwiseInteriorConstraint1D
from modules.generator import Sample1D
from modules.solution import Constructor1D
jax.config.update("jax_enable_x64",True)

def run(problem,epsilon,seed,output_dir,partitions,features,rcond):
 c=(p1_config if problem=="p1" else p2_config)(epsilon); m=c.model; px,pv=partitions
 domain={"x":tuple(c.mesh.domain.x),"v":(-1.,1.)}; strides={"x":1./px,"v":2./pv}; unknowns=px*pv*features
 common=dict(domain=domain,strides=strides,Jn={"f":features},scale=1.,init_rng=random.key(seed),kn=epsilon,activation=jnp.tanh)
 t=perf_counter(); eq=PointwiseInteriorConstraint1D(**common,num_quads=8,coeff_fns=dict(m.coeff)); bc=PointwiseBoundaryConstraint1D(**common); params={}
 eqf=jit(vmap(lambda x,v:eq.apply(params,x,v))); bcf=jit(vmap(lambda x,v:bc.apply(params,x,v))); feature=perf_counter()-t
 sample=Sample1D(domain,{"interior":(32,34),"boundary":64},"uniform"); t=perf_counter()
 left,right=np.asarray(bcf(*sample.pts_left)),np.asarray(bcf(*sample.pts_right)); interior=np.asarray(eqf(*sample.pts_int)).reshape(-1,unknowns)
 q=np.asarray(vmap(m.source)(*sample.pts_int)).reshape(-1); A=np.vstack((left,right,interior)); b=np.concatenate((np.full(left.shape[0],1. if problem=="p1" else .5),np.zeros(right.shape[0]),epsilon**2*q))
 norms=np.linalg.norm(A,axis=1); keep=norms>np.finfo(float).eps; A,b=A[keep]/norms[keep,None],b[keep]/norms[keep]; assembly=perf_counter()-t
 t=perf_counter(); coef,_,rank,s=lstsq(A,b,cond=rcond,lapack_driver="gelsd"); solve=perf_counter()-t; retained=s[s>rcond*s[0]]; cond=float(retained[0]/retained[-1])
 constructor=Constructor1D(**common,coefficients=jnp.asarray(coef)); approx=jit(vmap(lambda x,v:constructor.apply(params,x,v))); t=perf_counter()
 if problem=="p1": x=np.linspace(0,1,257); velocity=np.linspace(-1,1,128); ref=1-x[:,None]+np.zeros((1,128)); weights=None
 else:
  with np.load(ROOT/f"results/references/p2_parity_ref_eps_{epsilon:.0e}_level_B.npz") as d: x,velocity,ref,weights=d["x"],d["velocity"],d["f"],d["weights"]
 f=np.empty_like(ref)
 for begin in range(0,len(x),64):
  xx,vv=np.meshgrid(x[begin:begin+64],velocity,indexing="ij"); f[begin:begin+64]=np.asarray(approx(jnp.asarray(xx.reshape(-1,1)),jnp.asarray(vv.reshape(-1,1)))).reshape(xx.shape)
 rho=np.trapezoid(f,velocity,axis=1)/2; rr=np.trapezoid(ref,velocity,axis=1)/2
 ef=float(np.linalg.norm(f-ref)/np.linalg.norm(ref)) if weights is None else float(np.sqrt(np.sum(weights[None,:]*(f-ref)**2)/np.sum(weights[None,:]*ref**2))); er=float(np.linalg.norm(rho-rr)/np.linalg.norm(rr)); evaluation=perf_counter()-t
 rec=dict(problem=problem,method="rfm",epsilon=epsilon,seed=seed,relative_l2_f=ef,relative_l2_rho=er,condition_number=cond,rank=int(rank),num_rows=int(A.shape[0]),num_columns=unknowns,oversampling_ratio=float(A.shape[0]/unknowns),partitions=list(partitions),features_per_patch=features,rcond=rcond,feature_seconds=feature,assembly_seconds=assembly,solve_seconds=solve,evaluation_seconds=evaluation,total_seconds=feature+assembly+solve+evaluation)
 output_dir.mkdir(parents=True,exist_ok=True); stem=f"{problem}_rfm_eps_{epsilon:.0e}_seed_{seed}"; (output_dir/f"{stem}.json").write_text(json.dumps(rec,indent=2)+"\n"); np.savez_compressed(output_dir/f"{stem}.npz",x=x,velocity=velocity,f=f,rho=rho,reference_f=ref,reference_rho=rr,error_f=np.abs(f-ref),error_rho=np.abs(rho-rr)); return rec

if __name__=="__main__":
 p=argparse.ArgumentParser(); p.add_argument("--problem",choices=("p1","p2"),required=True); p.add_argument("--epsilon",type=float,required=True); p.add_argument("--seed",type=int,required=True); p.add_argument("--partitions",type=int,nargs=2,required=True); p.add_argument("--features",type=int,required=True); p.add_argument("--rcond",type=float,required=True); p.add_argument("--output-dir",type=Path,default=Path("results/baselines/rfm")); a=p.parse_args(); print(json.dumps(run(a.problem,a.epsilon,a.seed,a.output_dir,tuple(a.partitions),a.features,a.rcond),indent=2))
