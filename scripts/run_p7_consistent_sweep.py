"""Fixed-configuration epsilon sweep; only original RF parameters vary."""
import json
import os
from pathlib import Path
from functools import partial
from run_p7_transient_pilot import build,reference,errors
from p7_consistent_features import ConsistentFeatures
from audit_p7_local_space import held_out


def main():
 out=Path('results/p7_transient_pilot/consistent_pou_sweep.json')
 rows=json.loads(out.read_text()) if out.exists() else []
 configs=[((1,1,1),256,.5),((1,2,2),256,.5),((2,2,2),512,.5)]
 for parts,n,scale in configs:
  for eps in (1,.01,.001):
   if any(tuple(r['partitions'])==parts and r['J']==n and r['scale']==scale and r['epsilon']==eps for r in rows):continue
   f,r=build(eps,n,11,'oe_ap',samples=1024,nv=16,repeat_macro=True,block_average=False,
      boundary_samples=1024,initial_samples=2048,feature_factory=partial(ConsistentFeatures,partitions=parts,scale=scale))
   ref=reference(eps,256,16,800)
   r.update(partitions=parts,scale=scale,samples=1024,nv=16,boundary_samples=1024,initial_samples=2048,blas_threads=int(os.environ.get('P7_BLAS_THREADS','1')),
            held_out=held_out(f,eps),snapshots=errors(f,ref,eps))
   rows.append(r);out.write_text(json.dumps(rows,indent=2));print(parts,eps,r['snapshots'],flush=True)

if __name__=='__main__':main()
