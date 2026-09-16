"""Exploratory space-time RF experiment; not frozen manuscript results.

Jin--Ma--Wu (CiCP 2024), Problem 1 Case I, extended epsilon sweep.
eps**2 f_t + eps*v*f_x = <f>-f, x in (0,1), t in (0,.1).
Zero initial data; left incoming f=1, right incoming f=0.
Independent staggered parity finite-volume / backward-Euler reference.
"""
import argparse
import os
import json
import time
from contextlib import nullcontext
from pathlib import Path

for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = os.environ.get('P7_BLAS_THREADS', '1')
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import sparse
from scipy.linalg import lstsq
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse.linalg import splu
from scipy.stats import qmc


def quadrature(n):
    v, w = leggauss(n)
    return (v + 1) / 2, w / 2


def reference(eps, nx, nv, nt):
    start = time.perf_counter()
    v, w = quadrature(nv)
    h, dt = 1 / nx, .1 / nt
    alpha = eps**2 / (eps**2 + dt)
    beta = dt / (eps**2 + dt)
    # j at faces and r at cell centers; boundary traces obey inflow exactly.
    den = eps + h / (2 * v * beta)
    boundary = 1 / den
    old_boundary = h / (2 * v) * eps**2 / dt / den
    rows, cols, data = [], [], []
    for i in range(nx):
        for k in range(nv):
            p = i * nv + k
            diag = 1 / dt + 1 / eps**2
            for sign in (-1, 1):
                if 0 <= i + sign < nx:
                    d = beta * v[k]**2 / h**2
                    rows.append(p); cols.append((i + sign) * nv + k); data.append(-d)
                    diag += d
                else:
                    diag += v[k] * boundary[k] / h
            rows.append(p); cols.append(p); data.append(diag)
            for l in range(nv):
                rows.append(p); cols.append(i * nv + l); data.append(-w[l] / eps**2)
    matrix = sparse.coo_matrix((data, (rows, cols)), shape=(nx*nv, nx*nv)).tocsc()
    lu = splu(matrix)
    r = np.zeros((nx, nv)); j = np.zeros((nx + 1, nv))
    snapshots = []
    balances = []
    for step in range(1, nt + 1):
        old_flux = alpha * j
        old_flux[0] = boundary + old_boundary * j[0]
        old_flux[-1] = old_boundary * j[-1]
        rhs = r / dt - v * np.diff(old_flux, axis=0) / h
        new_r = lu.solve(rhs.ravel()).reshape(nx, nv)
        new_j = old_flux.copy()
        new_j[1:-1] -= beta * v * np.diff(new_r, axis=0) / h
        new_j[0] -= boundary * new_r[0]
        new_j[-1] += boundary * new_r[-1]
        balance = h * np.sum((new_r-r) @ w) / dt + (new_j[-1]-new_j[0]) @ (v*w)
        balances.append(float(abs(balance)))
        r, j = new_r, new_j
        if step in (nt//2, nt):
            snapshots.append((r.copy(), (j[:-1]+j[1:])/2))
    return dict(x=(np.arange(nx)+.5)*h, v=v, w=w,
                r=np.stack([s[0] for s in snapshots]), j=np.stack([s[1] for s in snapshots]),
                seconds=time.perf_counter()-start, balance_max=max(balances))


class Features:
    def __init__(self, n, seed):
        rng = np.random.default_rng(seed)
        self.a = rng.uniform(-2, 2, (3, n))
        self.b = rng.uniform(-2, 2, n)

    def raw(self, t, x, v):
        z = np.column_stack((20*t-1, 2*x-1, v)) @ self.a + self.b
        f = np.tanh(z)
        df = 1-f*f
        return f, df*self.a[0]*20, df*self.a[1]*2

    def parity(self, t, x, v, odd):
        pos, neg = self.raw(t,x,v), self.raw(t,x,-v)
        sign = -1 if odd else 1
        return tuple((a+sign*b)/2 for a,b in zip(pos,neg))


def build(eps, n, seed, method, samples=256, nv=8, *,
          initial_weight=1.0, boundary_weight=1.0, block_average=True,
          feature_factory=None, boundary_data=None, initial_data=None,
          repeat_macro=False, boundary_samples=256, initial_samples=512):
    if initial_weight <= 0 or boundary_weight <= 0:
        raise ValueError('Initial and boundary loss weights must be positive')
    start = time.perf_counter()
    v,w = quadrature(nv)
    points = qmc.Sobol(2, scramble=True, seed=seed).random_base2(int(np.log2(samples)))
    t,x = points[:,0]*.1, points[:,1]
    tt,xx,vv = np.repeat(t,nv),np.repeat(x,nv),np.tile(v,len(t))
    factory = Features if feature_factory is None else feature_factory
    rfeat, jfeat = factory(n,seed), factory(n,seed+1000)
    blocks=[]; targets=[]; raw_blocks=[]
    def add(a,b,weight=1.0):
        a=np.asarray(a).reshape(-1,2*n); b=np.broadcast_to(b,(len(a),)).copy()
        norm=np.linalg.norm(a,axis=1)
        scale=1/np.maximum(norm,1e-14)
        if block_average:
            scale=scale/np.sqrt(len(a))
        scale=scale*np.sqrt(weight)
        raw_blocks.append((a,b,scale))
        blocks.append(a*scale[:,None]); targets.append(b*scale)
    if method == 'direct':
        feat=factory(2*n,seed)
        # Full symmetric angular quadrature, same physical phase samples as OE.
        vv=np.concatenate((vv,-vv)); tt=np.tile(tt,2); xx=np.tile(xx,2)
        f,ft,fx=feat.raw(tt,xx,vv)
        half=len(f)//2
        avg=((f[:half]+f[half:])/2).reshape(-1,nv,2*n)
        avg=np.einsum('q,mqj->mj',w,avg)
        avg=np.tile(np.repeat(avg,nv,axis=0),(2,1))
        add(eps**2*ft+eps*vv[:,None]*fx+f-avg,0)
    else:
        r,rt,rx=rfeat.parity(tt,xx,vv,False)
        j,jt,jx=jfeat.parity(tt,xx,vv,True)
        def avg(a):
            return np.repeat(np.einsum('q,mqj->mj',w,a.reshape(-1,nv,n)),nv,axis=0)
        ar,art,ajx=avg(r),avg(rt),avg(vv[:,None]*jx)
        if method == 'oe_ap':
            macro=np.column_stack((art,ajx))
            add(macro if repeat_macro else macro[::nv],0)
            add(np.column_stack((eps**2*(rt-art)+r-ar,eps**2*(vv[:,None]*jx-ajx))),0)
        else:
            add(np.column_stack((eps**2*rt+r-ar,eps**2*vv[:,None]*jx)),0)
        add(np.column_stack((vv[:,None]*rx,eps**2*jt+j)),0)
    rng=np.random.default_rng(seed+2000)
    # Exclude the incompatible t=x=0 corner; use identical samples for methods.
    tb=rng.uniform(0,.1,boundary_samples); vb=rng.uniform(0,1,boundary_samples)
    for side in (0,1):
        xb=np.full_like(tb,side); signed=vb if side==0 else -vb
        if method=='direct': a=feat.raw(tb,xb,signed)[0]
        else: a=np.column_stack((rfeat.parity(tb,xb,signed,False)[0],eps*jfeat.parity(tb,xb,signed,True)[0]))
        target=1-side if boundary_data is None else boundary_data(tb,xb,signed)
        add(a,target,boundary_weight)
    xi=rng.uniform(0,1,initial_samples); vi=rng.uniform(-1,1,initial_samples); ti=np.zeros_like(xi)
    if method=='direct': a=feat.raw(ti,xi,vi)[0]
    else: a=np.column_stack((rfeat.parity(ti,xi,vi,False)[0],eps*jfeat.parity(ti,xi,vi,True)[0]))
    add(a,0 if initial_data is None else initial_data(xi,vi),initial_weight)
    a,b=np.vstack(blocks),np.concatenate(targets)
    assembly=time.perf_counter()-start
    start=time.perf_counter()
    col=np.maximum(np.linalg.norm(a,axis=0),1e-14)
    c,_,rank,s=lstsq(a/col,b,cond=1e-12,lapack_driver='gelsd')
    c=c/col
    solve=time.perf_counter()-start
    def evaluate(t,x,v):
        if method=='direct': return feat.raw(t,x,v)[0]@c
        return rfeat.parity(t,x,v,False)[0]@c[:n]+eps*jfeat.parity(t,x,v,True)[0]@c[n:]
    if method != 'direct':
        def parity_fields(t,x,v):
            return tuple(a@c[:n] for a in rfeat.parity(t,x,v,False)) + tuple(a@c[n:] for a in jfeat.parity(t,x,v,True))
        evaluate.parity_fields=parity_fields
    names=(['transport'] if method=='direct' else
           ['macro','micro','odd'] if method=='oe_ap' else ['even','odd'])+['left','right','initial']
    block_diagnostics={name:dict(rows=len(mat),physical_rms=float(np.sqrt(np.mean((mat@c-rhs)**2))),
                                weighted_loss=float(np.sum(((mat@c-rhs)*scale)**2)))
                       for name,(mat,rhs,scale) in zip(names,raw_blocks)}
    return evaluate,dict(method=method,epsilon=eps,seed=seed,J=n,N_coef=2*n,N_row=len(a),
                        initial_weight=initial_weight,boundary_weight=boundary_weight,
                        block_average=block_average,block_diagnostics=block_diagnostics,
                        repeat_macro=repeat_macro,
                        assembly_seconds=assembly,linear_seconds=solve,compute_seconds=assembly+solve,
                        rank=int(rank),condition=float(s[0]/s[-1]) if s[-1]>0 else float('inf'),
                        training_rms=float(np.sqrt(np.mean((a@c-b)**2))))


def errors(evaluate,ref,eps):
    x,v=np.meshgrid(ref['x'],ref['v'],indexing='ij'); out=[]
    for k,t in enumerate((.05,.1)):
        args=(np.full(x.size,t),x.ravel(),v.ravel())
        fp=evaluate(*args).reshape(x.shape)
        fm=evaluate(args[0],args[1],-args[2]).reshape(x.shape)
        rp=ref['r'][k]+eps*ref['j'][k]; rm=ref['r'][k]-eps*ref['j'][k]
        ef=np.sqrt(np.sum(((fp-rp)**2+(fm-rm)**2)*ref['w'])/np.sum((rp**2+rm**2)*ref['w']))
        rho=(fp+fm)/2@ref['w']; true=ref['r'][k]@ref['w']
        er=np.linalg.norm(rho-true)/np.linalg.norm(true)
        r_pred=(fp+fm)/2
        # Evaluate j directly when available to avoid small-epsilon cancellation.
        j_pred=(evaluate.parity_fields(*args)[3].reshape(x.shape)
                if hasattr(evaluate,'parity_fields') else (fp-fm)/(2*eps))
        def relative(a,b):
            return float(np.sqrt(np.sum((a-b)**2*ref['w'])/np.sum(b*b*ref['w'])))
        out.append(dict(t=t,E_f=float(ef),E_rho=float(er),
                        E_r=relative(r_pred,ref['r'][k]),E_j=relative(j_pred,ref['j'][k])))
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='results/p7_transient_pilot')
    p.add_argument('--features',type=int,default=64)
    p.add_argument('--seeds',type=int,nargs='+',default=[11,23,37])
    p.add_argument('--epsilons',type=float,nargs='+',default=[1,.01,.001])
    p.add_argument('--nx',type=int,default=128)
    p.add_argument('--nt',type=int,default=400)
    p.add_argument('--initial-weight',type=float,default=1.0)
    p.add_argument('--boundary-weight',type=float,default=1.0)
    p.add_argument('--loss-aggregation',choices=('block_mean','row_sum'),default='block_mean')
    args=p.parse_args(); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    (out/'config.json').write_text(json.dumps(vars(args),indent=2))
    results=[]; audits=[]
    with nullcontext():
        build(.01,16,0,'oe_ap',samples=32,nv=4)  # library warm-up, excluded
        for eps in args.epsilons:
            coarse=reference(eps,args.nx,8,args.nt)
            fine=reference(eps,args.nx*2,16,args.nt*2)
            np.savez_compressed(out/f'reference_eps_{eps:g}.npz',**fine)
            dr=[]; df=[]
            for k in (0,1):
                fc=fine['r'][k]@fine['w']; cc=coarse['r'][k]@coarse['w']
                interp=np.interp(coarse['x'],fine['x'],fc)
                dr.append(float(np.linalg.norm(interp-cc)/np.linalg.norm(interp)))
                xc,vc=np.meshgrid(coarse['x'],coarse['v'],indexing='ij')
                query=np.column_stack((xc.ravel(),vc.ravel()))
                ri=RegularGridInterpolator((fine['x'],fine['v']),fine['r'][k])(query).reshape(xc.shape)
                ji=RegularGridInterpolator((fine['x'],fine['v']),fine['j'][k])(query).reshape(xc.shape)
                numerator=np.sum(((ri-coarse['r'][k])**2+eps**2*(ji-coarse['j'][k])**2)*coarse['w'])
                denominator=np.sum((ri**2+eps**2*ji**2)*coarse['w'])
                df.append(float(np.sqrt(numerator/denominator)))
            audits.append(dict(epsilon=eps,density_refinement=dr,kinetic_refinement=df,coarse_seconds=coarse['seconds'],fine_seconds=fine['seconds'],balance_max=fine['balance_max']))
            print('REFERENCE',audits[-1],flush=True)
            for seed in args.seeds:
                for method in ('oe_ap','oe_unprojected','direct'):
                    evaluate,record=build(eps,args.features,seed,method,
                                          initial_weight=args.initial_weight,
                                          boundary_weight=args.boundary_weight,
                                          block_average=args.loss_aggregation=='block_mean')
                    record['snapshots']=errors(evaluate,fine,eps)
                    results.append(record)
                    (out/'runs.json').write_text(json.dumps(results,indent=2))
                    print('RUN',record,flush=True)
            (out/'reference_audit.json').write_text(json.dumps(audits,indent=2))


if __name__=='__main__': main()
