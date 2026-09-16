"""Fixed-budget local-feature screen with independent physical residuals."""
import argparse
import json
from functools import partial
from pathlib import Path
from run_p7_transient_pilot import Features, build, reference, errors, quadrature
from p7_local_features import LocalFeatures
import numpy as np

FACTORIES={
    'global':Features,
    'local_x':partial(LocalFeatures,x_edges=(0,.05,.15,.4,1)),
    'local_tx':partial(LocalFeatures,t_edges=(0,.02,.1),x_edges=(0,.15,1)),
}


def held_out(f,eps):
    rng=np.random.default_rng(1907)
    v,w=quadrature(16)
    t=rng.uniform(0,.1,512); x=rng.uniform(0,1,512)
    vv=np.tile(v,len(t))
    r,rt,rx,j,jt,jx=(a.reshape(-1,len(v)) for a in f.parity_fields(np.repeat(t,len(v)),np.repeat(x,len(v)),vv))
    art=rt@w; ajx=(v*jx)@w; ar=r@w
    macro=art+ajx
    micro=eps**2*(rt-art[:,None]+v*jx-ajx[:,None])+r-ar[:,None]
    odd=eps**2*jt+v*rx+j
    record=dict(macro_rms=float(np.sqrt(np.mean(macro**2))),
                micro_rms=float(np.sqrt(np.mean(micro**2@w))),
                odd_rms=float(np.sqrt(np.mean(odd**2@w))))
    x=rng.uniform(0,1,4096); v=rng.uniform(-1,1,4096); t=rng.uniform(0,.1,4096)
    record.update(initial_rms=float(np.sqrt(np.mean(f(0*x,x,v)**2))),
                  left_rms=float(np.sqrt(np.mean((f(t,0*x,abs(v))-1)**2))),
                  right_rms=float(np.sqrt(np.mean(f(t,0*x+1,-abs(v))**2))))
    return record


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cases',nargs='+',choices=FACTORIES,default=list(FACTORIES))
    parser.add_argument('--epsilons',nargs='+',type=float,default=[1,.001])
    parser.add_argument('--seeds',nargs='+',type=int,default=[11])
    parser.add_argument('--features',type=int,default=128)
    parser.add_argument('--samples',type=int,default=256)
    parser.add_argument('--output',default='results/p7_transient_pilot/local_space_screen.json')
    args=parser.parse_args()
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    records=[]
    for eps in args.epsilons:
        ref=reference(eps,256,16,800)
        for case in args.cases:
            for seed in args.seeds:
                f,r=build(eps,args.features,seed,'oe_ap',samples=args.samples,feature_factory=FACTORIES[case])
                r.update(case=case,samples=args.samples,held_out=held_out(f,eps),snapshots=errors(f,ref,eps))
                records.append(r)
                out.write_text(json.dumps(dict(config=vars(args),runs=records),indent=2))
                print(json.dumps(r),flush=True)


if __name__=='__main__':
    main()
