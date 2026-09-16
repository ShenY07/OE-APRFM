"""Single-seed diagnostics of the RF space, collocation and linear solve.

Snapshot fitting uses reference data for diagnosis only, never a PDE benchmark.
"""
import json
from pathlib import Path
import run_p7_transient_pilot as p
import numpy as np
from scipy.linalg import lstsq


def snapshot_fit(ref, eps):
    rf, jf = p.Features(128, 11), p.Features(128, 1011)
    mats, rhs = [], []
    for k, t in enumerate((.05, .1)):
        x, v = np.meshgrid(ref['x'][::2], ref['v'], indexing='ij')
        tt = np.full(x.size, t)
        weight = np.tile(np.sqrt(ref['w']), len(x))
        for sign in (1, -1):
            mat = np.column_stack((rf.parity(tt,x.ravel(),sign*v.ravel(),False)[0],
                                   eps*jf.parity(tt,x.ravel(),sign*v.ravel(),True)[0]))
            target = (ref['r'][k,::2]+sign*eps*ref['j'][k,::2]).ravel()
            mats.append(mat*weight[:,None]); rhs.append(target*weight)
    a, b = np.vstack(mats), np.concatenate(rhs)
    scale = np.maximum(np.linalg.norm(a,axis=0),1e-14)
    c, _, rank, s = lstsq(a/scale,b,cond=1e-12,lapack_driver='gelsd')
    c /= scale
    def evaluate(t,x,v):
        return rf.parity(t,x,v,False)[0]@c[:128]+eps*jf.parity(t,x,v,True)[0]@c[128:]
    test = dict(ref)
    test.update(x=ref['x'][1::2],r=ref['r'][:,1::2],j=ref['j'][:,1::2])
    return dict(rank=int(rank),held_out_snapshots=p.errors(evaluate,test,eps))


def main():
    rows=[]
    original_features=p.Features
    class WiderX(original_features):
        def __init__(self,n,seed):
            super().__init__(n,seed)
            self.a[1]*=4
    for eps in (1,.001):
        ref=p.reference(eps,256,16,800)
        for name,samples,feature in (('baseline',256,original_features),
                                     ('more_points',1024,original_features),
                                     ('wider_x',256,WiderX)):
            p.Features=feature
            captured={}
            def solve(a,b,**kwargs):
                answer=lstsq(a,b,**kwargs)
                c=answer[0]; residual=a@c-b
                q=lstsq(a,b,cond=1e-12,lapack_driver='gelsy')[0]
                captured.update(stationarity=float(np.linalg.norm(a.T@residual)/(np.linalg.norm(a)*np.linalg.norm(residual))),
                                qr_vs_svd_prediction=float(np.linalg.norm(a@(q-c))/np.linalg.norm(b)),
                                relative_matrix_residual=float(np.linalg.norm(residual)/np.linalg.norm(b)))
                return answer
            p.lstsq=solve
            try:
                f,record=p.build(eps,128,11,'oe_ap',samples=samples)
                record.update(case=name,linear_audit=captured,snapshots=p.errors(f,ref,eps))
                if name!='more_points': record['supervised_snapshot_fit']=snapshot_fit(ref,eps)
                rows.append(record)
                print(json.dumps(record),flush=True)
            finally:
                p.Features=original_features
                p.lstsq=lstsq
    out=Path(__file__).resolve().parents[1]/'results/p7_transient_pilot/rfm_space_audit_seed11.json'
    out.write_text(json.dumps(rows,indent=2))


if __name__=='__main__':
    main()
