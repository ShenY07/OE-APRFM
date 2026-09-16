"""Quadrature minimum gain in the supplied H1 parity norm; no training cutoff."""
import os
os.environ.setdefault('JAX_ENABLE_X64', 'true')
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from scipy.linalg import qr, solve_triangular, svdvals

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def gauss(n):
    z, w = np.polynomial.legendre.leggauss(n)
    return (z + 1) / 2, w / 2


def matrices(evaluate, nx, nv, nq, epsilon):
    """evaluate(x,v) -> (r,j,rx,jx), each with final coefficient axis."""
    x, wx = gauss(nx)
    v, wv = gauss(nv)
    q, wq = gauss(nq)
    xx, vv = np.meshgrid(x, v, indexing='ij')
    r, j, rx, jx = evaluate(xx, vv)
    xq, vq = np.meshgrid(x, q, indexing='ij')
    rq, _, _, jxq = evaluate(xq, vq)
    rho = np.einsum('xvk,v->xk', rq, wq)
    mean = np.einsum('xvk,v->xk', jxq, wq * q)
    weight = np.sqrt(wx[:, None] * wv[None, :])[..., None]
    columns = r.shape[-1]
    cx = np.concatenate([(a * weight).reshape(-1, columns) for a in (r,j,rx,jx)])
    # Fixed P1 medium: sigma_s=1, sigma_a=0. No manufactured source.
    r2 = epsilon**2 * (vv[..., None] * jx - mean[:, None, :]) + r - rho[:, None, :]
    r3 = vv[..., None] * rx + j
    rl, jl, _, _ = evaluate(np.zeros_like(v), v)
    rr, jr, _, _ = evaluate(np.ones_like(v), v)
    cy = np.concatenate([
        mean * np.sqrt(wx)[:, None],
        (r2 * weight).reshape(-1, columns),
        (r3 * weight).reshape(-1, columns),
        (rl + epsilon * jl) * np.sqrt(wv / 2)[:, None],
        (rr - epsilon * jr) * np.sqrt(wv / 2)[:, None],
    ])
    return cx, cy


def minimum_gain(cx, cy):
    sx = svdvals(cx)
    tolerance = np.finfo(float).eps * max(cx.shape) * sx[0]
    rank = int(np.sum(sx > tolerance))
    result = dict(columns=cx.shape[1], numerical_rank_x=rank,
                  rank_tolerance=tolerance, sigma_x_max=sx[0], sigma_x_min=sx[-1],
                  condition_x=sx[0]/sx[-1], truncated=False)
    if rank != cx.shape[1]:
        return dict(result, beta=None, status='unresolved full-space rank; no directions removed')
    _, rx, pivot = qr(cx, mode='economic', pivoting=True)
    whitened = solve_triangular(rx.T, cy[:, pivot].T, lower=True).T
    beta = float(svdvals(whitened)[-1])
    return dict(result, beta=beta, lambda_min=beta**2, status='full-space QR')


def analytic_checks():
    def one(x,v):
        z = np.zeros(x.shape + (1,))
        return x[...,None], z, np.ones_like(z), z
    def two(x,v):
        # Independently derived span{(1,0),(0,v)}: G=diag(1,1/3),
        # K=diag(1,(1+eps^2)/3); opposite boundary cross terms cancel.
        z = np.zeros_like(x)
        return np.stack([z+1,z],-1), np.stack([z,v],-1), np.stack([z,z],-1), np.stack([z,z],-1)
    for eps in (0., .001, 1.):
        cx, cy = matrices(one,4,4,4,eps)
        np.testing.assert_allclose(cx.T@cx, [[4/3]], atol=1e-14)
        np.testing.assert_allclose(cy.T@cy, [[5/6]], atol=1e-14)
        np.testing.assert_allclose(minimum_gain(cx,cy)['beta'], np.sqrt(5/8), atol=1e-14)
        cx, cy = matrices(two,4,4,4,eps)
        np.testing.assert_allclose(cy.T@cy, np.diag([1,(1+eps**2)/3]), atol=1e-14)
        np.testing.assert_allclose(minimum_gain(cx,cy)['beta'], 1., atol=1e-14)


def rf_evaluator(features, seed):
    import jax
    import jax.numpy as jnp
    from jax import random
    import modules.function_space as fs
    jax.config.update('jax_enable_x64', True)
    fs.seedXV = seed
    model = fs.RandomFeatureSpaceXV(domain={'x':(0.,1.),'v':(0.,1.)},
        strides={'x':1.,'v':1.}, Jn=features, scale=1., activation=jnp.tanh)
    params = model.init(random.key(seed), jnp.zeros(1), jnp.zeros(1))
    def fields(x,v):
        a = model.apply(params, jnp.array([x]), jnp.array([v])).reshape(-1)
        b = model.apply(params, jnp.array([x]), jnp.array([-v])).reshape(-1)
        z = jnp.zeros_like(a)
        return jnp.stack([jnp.concatenate([z,(a+b)/2]), jnp.concatenate([(a-b)/2,z])])
    values = jax.jit(jax.vmap(fields))
    derivatives = jax.jit(jax.vmap(jax.jacfwd(fields, argnums=0)))
    def evaluate(x,v):
        shape = x.shape + (2*features,)
        f = np.asarray(values(x.ravel(),v.ravel()))
        d = np.asarray(derivatives(x.ravel(),v.ravel()))
        return tuple(a.reshape(shape) for a in (f[:,0],f[:,1],d[:,0],d[:,1]))
    return evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', type=int, default=16)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--output', type=Path, default=ROOT/'results/minimum_gain/p1_J16_seed11.json')
    args = parser.parse_args()
    analytic_checks()
    evaluate = rf_evaluator(args.features,args.seed)
    rows = []
    # Refine projection quadrature separately before refining outer norms.
    for nx,nv,nq in [(16,16,16),(16,16,32),(32,32,32),(32,32,64),(64,64,64),(64,64,128),(128,128,128)]:
        for eps in (1.,.1,.01,.001,1e-6,0.):
            cx,cy = matrices(evaluate,nx,nv,nq,eps)
            row = dict(nx=nx,nv=nv,nq=nq,epsilon=eps,**minimum_gain(cx,cy))
            rows.append(row)
            print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(features=args.features,seed=args.seed,
        medium='sigma_s=1,sigma_a=0', domain=[0,1], analytic_checks='passed',
        space='actual RandomFeatureSpaceXV, one patch, scale=1, j/r coefficient order',
        norms='user supplied H1 x H1 and unweighted inflow L2', rows=rows),indent=2)+'\n')


if __name__ == '__main__':
    main()
