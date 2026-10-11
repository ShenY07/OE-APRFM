"""Run the twelve angular-domain controls serially in fresh CPU processes."""
import os,subprocess,json,sys,hashlib
from pathlib import Path
root=Path(__file__).resolve().parents[1];os.chdir(root);out=root/'results/angular_domain_comparison';out.mkdir(exist_ok=True)
env=os.environ.copy();env.update(PYTHONPATH='src:scripts',JAX_ENABLE_X64='true',JAX_PLATFORMS='cpu',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
records=[]
for problem in ['p1','p3']:
 for seed in [11,23,37]:
  for domain in (['reduced','expanded'] if seed!=23 else ['expanded','reduced']):
   dest=out/problem/domain/str(seed);dest.mkdir(parents=True,exist_ok=True)
   cmd=[sys.executable,'scripts/run_angular_domain_comparison.py','--problem',problem,'--domain',domain,'--seed',str(seed),'--output-dir',str(dest.relative_to(root))]
   print('START',problem,seed,domain,flush=True)
   with (dest/'process.log').open('w') as f:r=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
   if r.returncode:print((dest/'process.log').read_text(),flush=True);sys.exit(r.returncode)
   records.append(dict(command=cmd,result=str((dest/'result.json').relative_to(root))))
   (out/'manifest.json').write_text(json.dumps(dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),runs=records,environment={k:env[k] for k in ['JAX_ENABLE_X64','JAX_PLATFORMS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']},sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path('scripts/run_angular_domain_comparison.py'),Path('scripts/run_p1_oe_aprfm.py'),Path('scripts/run_p3_oe_aprfm.py'),*Path('src').rglob('*.py')]}),indent=2)+'\n')
   print('DONE',problem,seed,domain,flush=True)
