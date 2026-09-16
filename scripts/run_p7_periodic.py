"""Periodic source-free P7: fixed RF space, backward Euler, Fourier reference.

New experiment; never overwrites the earlier inflow space-time pilot.
"""
import os
os.environ.setdefault('JAX_ENABLE_X64','true')
os.environ.setdefault('JAX_PLATFORMS','cpu')
import argparse,json,csv
from pathlib import Path
import numpy as np
from scipy.linalg import svd,expm
from run_e3_slab import ROOT,fs,symmetric,rf_evaluator,gauss,relative

def diffusion(t,dt,x):
    rate=4*np.pi**2/3
    return 1+.2*np.exp(-rate*t)*np.cos(2*np.pi*x),1+.2*(1+rate*dt)**(-round(t/dt))*np.cos(2*np.pi*x)

def reference(eps,t,x,n):
    v,w=np.polynomial.legendre.leggauss(n);w=w/2
    L=(np.ones((n,1))*w[None,:]-np.eye(n))/eps**2-2j*np.pi*np.diag(v)/eps
    a=expm(t*L)@np.full(n,.2,dtype=complex)
    f=1+np.real(np.exp(2j*np.pi*x[:,None])*a[None,:])
    return v,w,f,f@w,np.real(np.exp(2j*np.pi*x)*(a@(v*w)))/eps

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=11)
    p.add_argument('--epsilon',type=float,nargs='+',default=[1,.1,.01,.001])
    p.add_argument('--dt',type=float,default=.002);p.add_argument('--features',type=int,default=64)
    p.add_argument('--rcond',type=float,default=1e-12)
    args=p.parse_args();out=ROOT/'results/p7_periodic';out.mkdir(parents=True,exist_ok=True)
    fs.uniform=symmetric;ev=rf_evaluator(args.features,args.seed)
    # Features are drawn once, then reused for every step and epsilon.
    x=(np.arange(32)+.5)/32;v=(np.arange(32)+.5)/32
    xx,vv=np.meshgrid(x,v,indexing='ij');r,j,rx,jx=ev(xx,vv)
    q,w=gauss(32);xq,vq=np.meshgrid(x,q,indexing='ij');rq,_,_,jxq=ev(xq,vq)
    rho=np.einsum('xvk,v->xk',rq,w);fluxx=np.einsum('xvk,v->xk',jxq,w*q)
    R=r.reshape(-1,2*args.features);J=j.reshape(R.shape)
    P=np.broadcast_to(rho[:,None,:],r.shape).reshape(R.shape)
    V=np.broadcast_to(fluxx[:,None,:],r.shape).reshape(R.shape)
    vb=(np.arange(64)+.5)/64
    rl,jl,_,_=ev(np.zeros_like(vb),vb);rr,jr,_,_=ev(np.ones_like(vb),vb)
    B=np.concatenate((rl-rr,jl-jr))
    xe,wx=gauss(128);ve,we=gauss(128);xxe,vve=np.meshgrid(xe,ve,indexing='ij')
    re,je,_,_=ev(xxe,vve)
    initial=np.concatenate((1+.2*np.cos(2*np.pi*xx.ravel()),np.zeros(xx.size),np.zeros(len(B))))
    init_A=np.concatenate((R,J,B))
    c0=np.linalg.lstsq(init_A,initial,rcond=args.rcond)[0]
    mass0=float(wx@((re@c0)@we))
    for eps in args.epsilon:
        stem=f'eps{eps:.0e}_dt{args.dt:.0e}_seed{args.seed}'
        if (out/f'{stem}.json').exists():continue
        dt=args.dt;nt=round(.2/dt);assert abs(nt*dt-.2)<1e-12
        a1=P/dt+V
        a2=eps**2*((R-P)/dt+vv.ravel()[:,None]*jx.reshape(R.shape)-V)+R-P
        a3=(1+eps**2/dt)*J+vv.ravel()[:,None]*rx.reshape(R.shape)
        A=np.concatenate((a1,a2,a3,B))
        H=np.concatenate((P/dt,eps**2*(R-P)/dt,eps**2*J/dt,np.zeros_like(B)))
        scale=np.linalg.norm(A,axis=1);scale=np.where(scale>1e-30,scale,1.)
        As=A/scale[:,None];cs=np.linalg.norm(As,axis=0);cs=np.where(cs>1e-14,cs,1.)
        U,s,W=svd(As/cs,full_matrices=False);keep=s>args.rcond*s[0]
        inverse=(W[keep].T/s[keep])@U[:,keep].T
        advance=(inverse@(H/scale[:,None]))/cs[:,None]
        c=c0.copy();records=[];history=[];snap={}
        for step in range(nt+1):
            t=step*dt;dens=(re@c)@we;current=(je@c)@(we*ve)
            amp=float(2*np.sum(wx*(dens-1)*np.cos(2*np.pi*xe)))
            history.append([t,amp,float(abs(wx@dens-mass0))])
            if step==0 or any(abs(t-target)<1e-10 for target in (.02,.10,.20)):
                vf,wf,fr,dr,qr=reference(eps,t,xe,128)
                _,_,_,dr2,qr2=reference(eps,t,xe,256)
                xt,vt=np.meshgrid(xe,vf,indexing='ij');rf,jf,_,_=ev(xt,vt)
                fh=rf@c+eps*(jf@c)
                dc,db=diffusion(t,dt,xe)
                qabs=float(np.sqrt(np.sum(wx*(current-qr)**2)))
                qnorm=float(np.sqrt(np.sum(wx*qr**2)))
                records.append(dict(t=t,Ef=relative(fh,fr,wx[:,None]*wf),Erho=relative(dens,dr,wx),
                    Eq_absolute=qabs,Eq_relative=qabs/qnorm if qnorm>1e-12 else None,
                    Ediff_continuous=relative(dens,dc,wx),Ediff_BE=relative(dens,db,wx),
                    amplitude=amp,amplitude_reference=float(2*np.sum(wx*(dr-1)*np.cos(2*np.pi*xe))),
                    mass_drift=history[-1][2],reference_rho_change=relative(dr2,dr,wx),
                    reference_q_absolute_change=float(np.sqrt(np.sum(wx*(qr2-qr)**2)))))
                snap[f'c_{step}']=c.copy();snap[f'rho_{step}']=dens;snap[f'q_{step}']=current
            if step<nt:c=advance@c
        metadata=dict(epsilon=eps,dt=dt,seed=args.seed,features_per_field=args.features,
            Ncoef=2*args.features,Nrow=len(A),rcond=args.rcond,rank=int(keep.sum()),
            initialization='U(-1,1)',operator_quadrature=32,periodic_rows=len(B),
            objective='unit-row normalized sum of squares, column equilibration, fixed SVD cutoff',
            initial_projection_error=float(np.linalg.norm(init_A@c0-initial)/np.linalg.norm(initial)),
            initial_mass=mass0,snapshots=records,
            note='q(0)=0 physically; Fick relation not imposed at t=0. Reference is continuous-time angular-discrete Fourier evolution.')
        (out/f'{stem}.json').write_text(json.dumps(metadata,indent=2)+'\n')
        np.savez_compressed(out/f'{stem}.npz',x=xe,wx=wx,history=np.array(history),**snap)
        print(stem,records[-1],flush=True)

if __name__=='__main__':main()
