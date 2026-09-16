"""Independent manufactured transport and local feature consistency checks."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_p7_transient_pilot import Features, build
from p7_local_features import LocalFeatures
import numpy as np


class PolynomialFeatures(Features):
    def __init__(self,n,seed):
        assert n==9
    def raw(self,t,x,v):
        zero=np.zeros_like(t); one=np.ones_like(t)
        return (np.column_stack((one,x**3,t*x,x*v*v,x,x*x*v,t*v,v,v**3)),
                np.column_stack((zero,zero,x,zero,zero,zero,v,zero,zero)),
                np.column_stack((zero,3*x*x,t,v*v,one,2*x*v,zero,zero,zero)))


def test_manufactured_transport():
    rng=np.random.default_rng(702)
    t=rng.uniform(0,.1,257); x=rng.uniform(0,1,257); v=rng.uniform(-1,1,257)
    for eps in (1,.01,.001):
        # Exact homogeneous transport solution; nonzero compatible initial/inflow data.
        def exact(t,x,v):
            r=1+x**3+2*t*x+6*eps**2*x*(v*v-1/3)
            j=-3*v*x*x-2*t*v+eps**2*(4*v-6*v**3)
            return r+eps*j
        for method in ('oe_ap','oe_unprojected'):
            f,_=build(eps,9,11,method,feature_factory=PolynomialFeatures,
                      boundary_data=exact,initial_data=lambda x,v:exact(0*x,x,v))
            np.testing.assert_allclose(f(t,x,v),exact(t,x,v),rtol=1e-9,atol=1e-9)
            r,rt,rx,j,jt,jx=f.parity_fields(t,x,v)
            np.testing.assert_allclose(rt,2*x,atol=1e-8)
            np.testing.assert_allclose(jt,-2*v,atol=1e-8)
            np.testing.assert_allclose(eps**2*jt+v*rx+j,0,atol=1e-8)


def test_local_derivatives_and_partition():
    rng=np.random.default_rng(792)
    t=rng.uniform(.001,.099,200); x=rng.uniform(.001,.999,200); v=rng.uniform(-1,1,200)
    f=LocalFeatures(128,11,t_edges=(0,.02,.1),x_edges=(0,.15,1))
    psi,pt,px=f.weights(t,x)
    np.testing.assert_allclose(psi.sum(axis=1),1,atol=1e-14)
    np.testing.assert_allclose(pt.sum(axis=1),0,atol=1e-11)
    np.testing.assert_allclose(px.sum(axis=1),0,atol=1e-11)
    for odd in (False,True):
        val,dt,dx=f.parity(t,x,v,odd)
        h=1e-7
        np.testing.assert_allclose(dt,(f.parity(t+h,x,v,odd)[0]-f.parity(t-h,x,v,odd)[0])/(2*h),rtol=1e-5,atol=2e-6)
        np.testing.assert_allclose(dx,(f.parity(t,x+h,v,odd)[0]-f.parity(t,x-h,v,odd)[0])/(2*h),rtol=1e-5,atol=2e-6)
        np.testing.assert_allclose(f.parity(t,x,-v,odd)[0],(-1 if odd else 1)*val,atol=1e-14)


if __name__=='__main__':
    test_manufactured_transport()
    test_local_derivatives_and_partition()
    print('Manufactured PDE solve, local PoU, parity and derivative checks passed')
