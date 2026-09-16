"""Independent consistency checks for the exploratory transient implementation."""
import importlib.util
from pathlib import Path
import numpy as np

spec=importlib.util.spec_from_file_location('p7',Path(__file__).parents[1]/'scripts/run_p7_transient_pilot.py')
p7=importlib.util.module_from_spec(spec)
spec.loader.exec_module(p7)


def test_parity_and_derivatives():
    feat=p7.Features(8,11)
    t=np.array([.02,.07]); x=np.array([.2,.8]); v=np.array([.3,.9])
    for odd in (False,True):
        f,ft,fx=feat.parity(t,x,v,odd)
        np.testing.assert_allclose(feat.parity(t,x,-v,odd)[0],(-1 if odd else 1)*f)
        h=1e-6
        np.testing.assert_allclose(ft,(feat.parity(t+h,x,v,odd)[0]-feat.parity(t-h,x,v,odd)[0])/(2*h),atol=1e-7)
        np.testing.assert_allclose(fx,(feat.parity(t,x+h,v,odd)[0]-feat.parity(t,x-h,v,odd)[0])/(2*h),atol=1e-7)


def test_reference_diffusion_limit_and_conservation():
    ref=p7.reference(1e-4,128,8,800)
    x=ref['x']; rho=ref['r'][-1]@ref['w']
    # Exact heat solution: rho_t=rho_xx/3, left=1,right=0, initial=0.
    m=np.arange(1,101)[:,None]
    exact=1-x- np.sum(2/(m*np.pi)*np.sin(m*np.pi*x)*np.exp(-(m*np.pi)**2*.1/3),axis=0)
    assert np.linalg.norm(rho-exact)/np.linalg.norm(exact)<.003
    assert ref['balance_max']<1e-6


def test_parity_error_norms():
    v,w=p7.quadrature(8)
    ref=dict(x=np.array([.25,.75]),v=v,w=w,
             r=np.ones((2,2,8)),j=np.broadcast_to(v,(2,2,8)))
    eps=.01
    def evaluate(t,x,v):return 1.1+eps*1.2*v
    for s in p7.errors(evaluate,ref,eps):
        np.testing.assert_allclose([s['E_r'],s['E_j'],s['E_rho']],[.1,.2,.1],atol=1e-12)
        expected=np.sqrt((.1**2+eps**2*.2**2*np.sum(w*v*v))/(1+eps**2*np.sum(w*v*v)))
        np.testing.assert_allclose(s['E_f'],expected,atol=1e-12)


if __name__=='__main__':
    test_parity_and_derivatives()
    test_reference_diffusion_limit_and_conservation()
    print('P7 derivative, parity, diffusion-limit and conservation checks passed')
