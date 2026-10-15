"""Build the P1--P5 model collection matched to the OE-APNN baseline cases."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def solve_case(problem, epsilon, seed):
    out = ROOT / 'results/runs/oe_aprfm_baseline' / problem / f'eps_{epsilon:.0e}' / f'seed_{seed}'
    out.mkdir(parents=True, exist_ok=True)
    target = ROOT / 'models/oe_aprfm/baseline' / problem / f'eps_{epsilon:.0e}' / f'seed_{seed}'
    if (target/'model.npz').exists() and (out/'reload_check.json').exists():
        return str(target.relative_to(ROOT))
    command = [sys.executable, '-m', 'experiments.retrain_oe_aprfm', '--case', problem, str(epsilon), str(seed)]
    env = dict(os.environ, PYTHONPATH=str(ROOT/'src')+os.pathsep+str(ROOT),
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', JAX_ENABLE_X64='true')
    with (out/'solve.log').open('w') as log:
        subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    stem=f'{problem}_oe_aprfm_eps_{epsilon:.0e}_seed_{seed}'
    shutil.move(str(out/f'{stem}_model'),target)
    print('Completed',problem,epsilon,seed,flush=True)
    return str(target.relative_to(ROOT))


def case(problem, epsilon, seed):
    import numpy as np
    from oeaprfm.checkpoints import load_model
    out = ROOT/'results/runs/oe_aprfm_baseline'/problem/f'eps_{epsilon:.0e}'/f'seed_{seed}'
    if problem == 'p1':
        from experiments.run_p1_oe_aprfm import run
        record=run(epsilon,seed,out,features=64)
    elif problem == 'p2':
        from experiments.run_p2_oe_aprfm import run
        record=run(epsilon,seed,out,ROOT/'data/references/p2',partitions=(2,4),features=64,rcond=1e-6)
    else:
        from experiments.run_p3_oe_aprfm import run
        record=run(epsilon,seed,out,problem=problem,features=128,partitions=(1,1,1),collocation=(16,16,16),p5_definition='square_hole')
    stem=f'{problem}_oe_aprfm_eps_{epsilon:.0e}_seed_{seed}'
    predictor=load_model(out/f'{stem}_model')
    path=out/f'{problem}_oe_aprfm_eps_{epsilon:.0e}_seed_{seed}.npz'
    with np.load(path) as d:
        axes = (d['x'], d['v'] if 'v' in d else d['velocity']) if problem in ('p1','p2') else (d['x'],d['y'],d['theta'])
        indices = [np.unique(np.linspace(0,len(axis)-1,5,dtype=int)) for axis in axes]
        points=np.meshgrid(*(axis[index] for axis,index in zip(axes,indices)),indexing='ij')
        actual=predictor(*points);expected=d['f'][np.ix_(*indices)]
        np.testing.assert_allclose(actual,expected,rtol=2e-9,atol=2e-9)
        check=dict(max_absolute_difference=float(np.max(np.abs(actual-expected))),points=int(actual.size),passed=True)
    (out/'reload_check.json').write_text(json.dumps(check,indent=2)+'\n')
    print(json.dumps(record),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',nargs=3,metavar=('PROBLEM','EPSILON','SEED'))
    p.add_argument('--workers',type=int,default=2)
    a=p.parse_args()
    if a.case:
        case(a.case[0],float(a.case[1]),int(a.case[2]));return
    cases=[(f'p{i}',eps,seed) for i in range(1,6) for eps in (1.,.001) for seed in (7,11,17)]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        paths=list(pool.map(lambda args:solve_case(*args),cases))
    (ROOT/'models/oe_aprfm/baseline/index.json').write_text(json.dumps(paths,indent=2)+'\n')


if __name__=='__main__':main()
