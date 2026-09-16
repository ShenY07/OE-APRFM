"""Small fixed-seed P1 cutoff diagnostic, without altering archived experiments."""
import os
os.environ.setdefault('JAX_ENABLE_X64','true')
os.environ.setdefault('JAX_PLATFORMS','cpu')
import sys
from pathlib import Path
import argparse,json,hashlib,subprocess,csv
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
OUT=ROOT/'results/p1_cutoff_check'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--method',choices=['oe','mm','all'],default='all')
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if args.method=='all':
        for method in ('oe','mm'):
            subprocess.run([sys.executable,__file__,'--method',method],check=True)
        rows=[json.loads(p.read_text()) for p in sorted(OUT.glob('*/diagnostic.json'))]
        with (OUT/'summary.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        print(json.dumps(rows,indent=2),flush=True)
        return
    captured={}
    if args.method=='oe':
        import run_p1_oe_aprfm as runner
        original=runner.solve
        def solve(A,b,**kw):
            captured['matrix_sha256']=hashlib.sha256(np.ascontiguousarray(A).tobytes()+np.ascontiguousarray(b).tobytes()).hexdigest()
            c,d=original(A,b,**kw)
            captured['coefficient_norm']=float(np.linalg.norm(c))
            captured['raw_residual_norm']=float(np.linalg.norm(A@c.reshape(-1)-b.reshape(-1)))
            return c,d
        runner.solve=solve
    else:
        import run_mm_aprfm_1d as runner
        original=runner.lstsq
        def solve(A,b,**kw):
            captured['matrix_sha256']=hashlib.sha256(np.ascontiguousarray(A).tobytes()+np.ascontiguousarray(b).tobytes()).hexdigest()
            scale=np.linalg.norm(A,axis=0);scale=np.where(scale>1e-14,scale,1.)
            c,res,rank,s=original(A/scale,b,**kw);c=c/scale
            captured['coefficient_norm']=float(np.linalg.norm(c))
            captured['raw_residual_norm']=float(np.linalg.norm(A@c-b))
            return c,res,rank,s
        runner.lstsq=solve
    hashes=[]
    for cutoff in (1e-6,1e-8,1e-10,1e-12):
        directory=OUT/f'{args.method}_{cutoff:.0e}'
        if args.method=='oe':
            r=runner.run(.001,11,directory,features=64,partitions=(1,1),collocation=(32,32),rcond=cutoff)
        else:
            r=runner.run('p1',.001,11,directory,partitions=(1,1),rho_features=64,g_features=64,collocation=(32,32),rcond=cutoff)
        row=dict(method=args.method,epsilon=.001,seed=11,rcond=cutoff,
                 Ef=r['relative_l2_f'],Erho=r['relative_l2_rho'],rank=r['rank'],Ncoef=r['num_columns'],Nrow=r['num_rows'],
                 initialization='legacy U(0,1)',residual_units='raw OE; row-normalized MM',**captured)
        hashes.append(row['matrix_sha256']);assert len(set(hashes))==1,'Matrix/RHS changed during cutoff sweep'
        (directory/'diagnostic.json').write_text(json.dumps(row,indent=2)+'\n')
        print(json.dumps(row),flush=True)

if __name__=='__main__':main()
