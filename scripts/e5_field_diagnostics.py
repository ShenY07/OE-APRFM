"""Capture the solved RF constructor and evaluate physical diagnostics by AD."""
import json
import numpy as np
import jax
import jax.numpy as jnp

def install(runner,method):
    name='OddEvenDecompositionConstructor1D' if method=='oe' else 'MicroMacroConstructor1D'
    original=getattr(runner,name);state={}
    def constructor(**kw):
        state['kwargs']=kw
        state['field']=original(**kw)
        return state['field']
    setattr(runner,name,constructor)
    state['original']=original
    return state

def evaluate(state,method,directory,old_directory):
    kw=state['kwargs'];model=state['field'];c=np.asarray(kw['coefficients']).ravel()
    np.savez_compressed(directory/'coefficients.npz',coefficients=c)
    params={} if method=='mm' else model.init(kw['init_rng'],jnp.zeros(1),jnp.ones(1))
    def value(x,v):return model.apply(params,jnp.atleast_1d(x),jnp.atleast_1d(v)).squeeze()
    fn=jax.jit(jax.vmap(lambda x,v:jnp.stack((value(x,v),jax.grad(value,0)(x,v)))))
    macro=None
    if method=='mm':
        mc=c.copy();mc[128:]=0
        m=state['original'](**dict(kw,coefficients=jnp.asarray(mc)))
        macro=jax.jit(jax.vmap(lambda x:m.apply({},jnp.atleast_1d(x),jnp.ones(1)).squeeze()))
    result=[]
    for n in (64,128):
        z,w=np.polynomial.legendre.leggauss(n);x=(z+1)/2;wx=w/2;v=z;wv=w/2
        xx,vv=np.meshgrid(x,v,indexing='ij');xf=xx.ravel();vf=vv.ravel()
        values=np.concatenate([np.asarray(fn(xf[i:i+16],vf[i:i+16])) for i in range(0,len(xf),16)])
        f=values[:,0].reshape(n,n);fx=values[:,1].reshape(n,n)
        rho=f@wv;flux=f@(wv*v)
        kappa=.01+.5*(np.tanh(6.5-11*x)+np.tanh(11*x-4.5))
        residual=v[None,:]*fx+(f-rho[:,None])/kappa[:,None]
        row=dict(quadrature=n,Rtr_absolute_L2=float(np.sqrt(np.sum(wx[:,None]*wv*residual**2))))
        fields=dict(x=x,velocity=v,wx=wx,wv=wv,f=f,fx=fx,rho_physical=rho,flux=flux,Rtr=residual)
        if macro is not None:
            mr=np.asarray(macro(x));g=f-mr[:,None];mean_g=g@wv
            row['D0_absolute_L2']=float(np.sqrt(np.sum(wx*mean_g**2)))
            fields.update(rho_macro=mr,g=g,mean_g=mean_g)
            np.testing.assert_allclose(rho,mr+mean_g,atol=1e-12,rtol=1e-10)
        np.savez_compressed(directory/f'physical_fields_q{n}.npz',**fields)
        result.append(row)
    old=np.load(next(old_directory.glob('p2_*.npz')))
    new=np.load(next(directory.glob('p2_*.npz')))
    discrepancy=float(np.linalg.norm(new['f']-old['f'])/np.linalg.norm(old['f']))
    report=dict(method=method,replay_relative_field_change=discrepancy,
                replay_matches=discrepancy<1e-8,metrics=result,
                note='Physical normalized angular measure; independent Gauss x/v quadrature; AD x derivative. No theoretical certification.')
    (directory/'physical_diagnostics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
