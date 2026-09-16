"""E6 frozen periodic BE protocol; independent of the historical P7 pilot."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
from scipy.linalg import svd
from e6_protocol import PROTOCOL,IMPLEMENTATION,digest,validate
ROOT=Path(__file__).resolve().parents[1]

def gauss(n,half=False):
    v,w=np.polynomial.legendre.leggauss(n)
    return ((v+1)/2,w/2) if half else (v,w/2)

class Space:
    def __init__(self,J,seed):
        self.J=J
        self.p=np.random.default_rng(seed).uniform(-1,1,(J,3))
    def __call__(self,x,v):
        x,v=np.broadcast_arrays(x,v);p=self.p
        plus=np.tanh((2*x[...,None]-1)*p[:,0]+v[...,None]*p[:,1]+p[:,2])
        minus=np.tanh((2*x[...,None]-1)*p[:,0]-v[...,None]*p[:,1]+p[:,2])
        dxp=2*p[:,0]*(1-plus**2);dxm=2*p[:,0]*(1-minus**2)
        z=np.zeros_like(plus)
        return (np.concatenate((z,(plus+minus)/2),axis=-1),
                np.concatenate(((plus-minus)/2,z),axis=-1),
                np.concatenate((z,(dxp+dxm)/2),axis=-1),
                np.concatenate(((dxp-dxm)/2,z),axis=-1))

def run(eps,dt,seed,J,nq,batch):
    out=ROOT/'results/e6_periodic'/batch/f's{seed}_eps{eps:.0e}_dt{dt:.0e}_J{J}_q{nq}'
    if (out/'result.json').exists():
        if validate(out/'result.json'):return
        raise RuntimeError('Existing record failed protocol/completeness validation; preserve and audit before rerun')
    out.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    space=Space(J,seed);x=(np.arange(128)+.5)/128;v,w=gauss(16,True)
    xx,vv=np.meshgrid(x,v,indexing='ij');r,j,rx,jx=space(xx,vv)
    q,wq=gauss(nq);xq,vq=np.meshgrid(x,q,indexing='ij');rq,_,_,jqx=space(xq,vq)
    P=np.einsum('xvk,v->xk',rq,wq);V=np.einsum('xvk,v->xk',jqx,wq*q)
    vb,wb=gauss(32,True);rl,jl,_,_=space(0,vb);rr,jr,_,_=space(1,vb)
    A=np.concatenate((P/dt+V,
        ((1+eps**2/dt)*(r-P[:,None,:])+eps**2*(vv[...,None]*jx-V[:,None,:])).reshape(-1,2*J),
        (vv[...,None]*rx+(1+eps**2/dt)*j).reshape(-1,2*J),rl-rr,jl-jr))
    H=np.concatenate((P/dt,(eps**2/dt*(r-P[:,None,:])).reshape(-1,2*J),
                      (eps**2/dt*j).reshape(-1,2*J),np.zeros((64,2*J))))
    assert A.shape==(4288,2*J)
    blockw=np.concatenate((np.full(128,1/128),np.tile(w/128,128),np.tile(w/128,128),wb,wb))
    sizes=[128,2048,2048,32,32];offset=0
    for size in sizes:
        np.testing.assert_allclose(blockw[offset:offset+size].sum(),1);offset+=size
    norms=np.linalg.norm(A,axis=1);norms=np.where(norms>1e-30,norms,1.)
    factor=np.sqrt(blockw)/norms;Aw=A*factor[:,None]
    cs=np.linalg.norm(Aw,axis=0);cs=np.where(cs>1e-14,cs,1.)
    u,s,vt=svd(Aw/cs,full_matrices=False);keep=s>1e-12*s[0]
    def solve_rhs(b):
        return (vt[keep].T@((u[:,keep].T@(factor*b))/s[keep]))/cs
    assembly=time.perf_counter()-start
    xe=(np.arange(512)+.5)/512;ve,we=gauss(128)
    xx,vv=np.meshgrid(xe,ve,indexing='ij');re,je,_,_=space(xx,vv)
    be,bw=gauss(128,True);r0,j0,_,_=space(0,be);r1,j1,_,_=space(1,be)
    analytic=1+.2*np.cos(2*np.pi*x)
    rhs=np.concatenate((analytic/dt,np.zeros(4160)))
    const_rhs=np.concatenate((np.ones(128)/dt,np.zeros(4160)))
    cc=solve_rhs(const_rhs)
    regression=dict(constant_r_error=float(np.sqrt(np.mean(((re@cc)-1)**2))),
                    constant_j_error=float(np.sqrt(np.mean((je@cc)**2))))
    c=None;history=[];snapshots={'x':xe,'v':ve,'w':we,'parameters':space.p}
    nt=round(.2/dt);assert abs(nt*dt-.2)<1e-12
    advance_seconds=0.
    for step in range(nt+1):
        t=step*dt
        if step==0:
            rho=1+.2*np.cos(2*np.pi*xe);current=np.zeros_like(xe);f=np.broadcast_to(rho[:,None],(512,128))
            br=bj=0.
        else:
            tic=time.perf_counter();c=solve_rhs(rhs if step==1 else H@c)
            advance_seconds+=time.perf_counter()-tic
            rv=re@c;jv=je@c;f=rv+eps*jv;rho=rv@we;current=jv@(ve*we)
            br=float(np.sqrt(bw@(((r0-r1)@c)**2)));bj=float(np.sqrt(bw@(((j0-j1)@c)**2)))
        amplitude=float(2*np.mean((rho-1)*np.cos(2*np.pi*xe)))
        target=1+.2*(1+4*np.pi**2*dt/3)**(-step)*np.cos(2*np.pi*xe)
        history.append(dict(t=t,mass_drift=float(abs(rho.mean()-1)),amplitude=amplitude,
                            Ediff_BE=float(np.linalg.norm(rho-target)/np.linalg.norm(target)),
                            periodic_r=br,periodic_j=bj,min_f=float(f.min())))
        if step==0 or any(abs(t-ts)<1e-12 for ts in [.02,.1,.2]):
            snapshots[f'rho_{step}']=rho;snapshots[f'q_{step}']=current;snapshots[f'f_{step}']=f
            if c is not None:snapshots[f'coefficients_{step}']=c.copy()
    np.savez_compressed(out/'fields.npz',**snapshots)
    record=dict(epsilon=eps,dt=dt,seed=seed,J=J,Ncoef=2*J,Nrow=4288,nq=nq,
        initialization='numpy default_rng(seed), U(-1,1), Jx3; shared raw basis for r/j',
        feature_sha256=hashlib.sha256(space.p.tobytes()).hexdigest(),rcond=1e-12,rank=int(keep.sum()),
        block_weight_sums=[1]*5,constant_step_regression=regression,
        assembly_factorization_seconds=assembly,advance_seconds=advance_seconds,
        total_seconds=time.perf_counter()-start,history=history,
        status='trajectory computed; reference/test refinement pending; regression errors are diagnostics, not automatic pass')
    record.update(protocol_id=PROTOCOL,implementation_id=IMPLEMENTATION,config_hash=digest(record),
                  feature_hash=record['feature_sha256'],valid_for_e6=True)
    (out/'result.json').write_text(json.dumps(record,indent=2)+'\n')
    print(out,regression,history[-1],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--epsilon',type=float,nargs='+',default=[1,.001]);p.add_argument('--seed',type=int,default=11)
    p.add_argument('--dt',type=float,default=.002);p.add_argument('--features',type=int,default=128)
    p.add_argument('--quadrature',type=int,default=64);p.add_argument('--batch',default='main')
    a=p.parse_args()
    for eps in a.epsilon:run(eps,a.dt,a.seed,a.features,a.quadrature,a.batch)
