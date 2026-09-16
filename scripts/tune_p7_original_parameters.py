"""Single-patch parameter sweep aligned to the original 1D RF conventions.

NumPy random stream (not bitwise Flax initialization). Same [0,scale) law,
center/halfwidth coordinates, shared r/j feature pool, unit-L2 row sum.
Single patch makes normalized psi_a and psi_b identical on the domain.
"""
import json
from pathlib import Path
from functools import partial
from run_p7_transient_pilot import Features, build, reference, errors
from audit_p7_local_space import held_out
import numpy as np


class OriginalSinglePatch(Features):
    def __init__(self,n,seed,scale=1):
        # Original equal-budget r/j fields use the same inner RF initialization.
        seed=seed-1000 if seed>=1000 else seed
        rng=np.random.default_rng(seed)
        draws=rng.uniform(0,1,(n,4))*scale
        self.a=draws[:,:3].T
        self.b=draws[:,3]
    def raw(self,t,x,v):
        z=np.column_stack((20*t-1,2*x-1,2*v-1))@self.a+self.b
        phi=np.tanh(z); d=1-phi*phi
        return phi,d*self.a[0]*20,d*self.a[1]*2


CONFIGS=[
    (128,1.,1024), (256,.5,1024), (256,1.,1024),
    (256,2.,1024), (512,1.,2048), (512,2.,2048),
]


def main():
    out=Path('results/p7_transient_pilot/original_parameter_sweep.json')
    rows=[]
    for eps in (1,.001):
        ref=reference(eps,256,16,800)
        for n,scale,samples in CONFIGS:
            f,r=build(eps,n,11,'oe_ap',samples=samples,nv=16,
                      block_average=False,repeat_macro=True,feature_factory=partial(OriginalSinglePatch,scale=scale))
            r.update(feature_scale=scale,samples=samples,training_positive_angles=16,
                     partitions=[1,1,1],initial_points=512,boundary_points_per_side=256,
                     held_out=held_out(f,eps),snapshots=errors(f,ref,eps))
            rows.append(r)
            out.write_text(json.dumps(dict(protocol=__doc__,runs=rows),indent=2))
            print(json.dumps(r),flush=True)


if __name__=='__main__':
    main()
