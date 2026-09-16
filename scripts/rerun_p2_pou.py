import os,subprocess,json
from pathlib import Path
root=Path(__file__).resolve().parents[1];os.chdir(root)
out=root/'results/pou_fix_2026_09_08/p2';out.mkdir(parents=True,exist_ok=True)
env=dict(os.environ,PYTHONPATH=str(root/'src'),JAX_PLATFORMS='cpu',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
for eps in (1,.001):
 for seed in (11,23,37):
  result=out/f'p2_oe_aprfm_eps_{eps:.0e}_seed_{seed}_pou_b.json'
  if result.exists() and result.with_suffix('.npz').exists():
   print('EXISTS',eps,seed,flush=True)
   continue
  log=out/f'eps_{eps:g}_seed_{seed}.log'
  cmd=['python3','scripts/run_p2_oe_aprfm.py','--epsilon',str(eps),'--seed',str(seed),'--output-dir',str(out),'--reference-dir','results/pou_fix_2026_09_08/references','--partitions','2','4','--features','64','--rcond','1e-6','--tag','pou_b']
  with log.open('w') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
  print('DONE',eps,seed,flush=True)
