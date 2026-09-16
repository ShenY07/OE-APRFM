"""Check the original PoU convention and derivatives in the transient port."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from p7_consistent_features import ConsistentFeatures


def test_derivatives_parity_and_shared_pool():
    f=ConsistentFeatures(128,11,partitions=(2,2,2))
    other=ConsistentFeatures(128,1011,partitions=(2,2,2))
    for a,b in zip(f.params,other.params):np.testing.assert_array_equal(a,b)
    rng=np.random.default_rng(7)
    t=rng.uniform(.001,.099,100);x=rng.uniform(.001,.999,100);v=rng.uniform(-1,1,100)
    for odd in (False,True):
        a,dt,dx=f.parity(t,x,v,odd);h=1e-7
        np.testing.assert_allclose(f.parity(t,x,-v,odd)[0],(-1 if odd else 1)*a,atol=1e-14)
        np.testing.assert_allclose(dt,(f.parity(t+h,x,v,odd)[0]-f.parity(t-h,x,v,odd)[0])/(2*h),atol=1e-6,rtol=1e-5)
        np.testing.assert_allclose(dx,(f.parity(t,x+h,v,odd)[0]-f.parity(t,x-h,v,odd)[0])/(2*h),atol=1e-6,rtol=1e-5)


def test_single_patch_matches_aligned_baseline():
    from tune_p7_original_parameters import OriginalSinglePatch
    t=np.array([0.,.02,.1]);x=np.array([0.,.4,1.]);v=np.array([-.9,.3,1.])
    f=ConsistentFeatures(32,11,partitions=(1,1,1),scale=.5)
    g=OriginalSinglePatch(32,11,scale=.5)
    for a,b in zip(f.raw(t,x,v),g.raw(t,x,v)):np.testing.assert_allclose(a,b,atol=1e-12)
