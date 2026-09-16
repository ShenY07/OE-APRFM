"""Serial, restartable repair queue. Old experimental records stay untouched."""
import os, sys, json, time, subprocess, csv, fcntl
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/comparison_repairs_2026_09_10'

def worker(task):
    import numpy as np
    if task=='mm_grid':
        from run_e6_mm import Space, gauss
        rows=[]
        for p in sorted((ROOT/'results/e6_mm_comparison').glob('s*/result.json')):
            r=json.loads(p.read_text());d=np.load(p.parent/'fields.npz');sp=Space(r['seed'])
            np.testing.assert_array_equal(sp.m,d['macro_parameters']);np.testing.assert_array_equal(sp.g,d['micro_parameters'])
            eps=r['epsilon'];ref=np.load(ROOT/f'results/e6_periodic/references/eps{eps:.0e}_dt2e-03_n256.npz')
            vn,wn=gauss(256);base={}
            for nx,nv in ((512,128),(1024,128),(512,256)):
                x=(np.arange(nx)+.5)/nx;v,w=gauss(nv);M,_=sp.macro(x)
                for step in (10,50,100):
                    c=d[f'coefficients_{step}'];g=np.concatenate([sp.micro(x[i:i+32,None],v[None,:])[0]@c[128:] for i in range(0,nx,32)])
                    f=(M@c[:128])[:,None]+eps*g;rho=f@w;q=g@(w*v);phase=np.exp(2j*np.pi*x)
                    for kind in ('ct','be'):
                        a=ref[f'{kind}_{step*.002:.2f}'];av=a[:256] if nv==256 else a[256:]
                        targets=(1+.2*np.real(phase[:,None]*av),1+.2*np.real(phase*(a[:256]@wn)),.2*np.real(phase*(a[:256]@(vn*wn)))/eps)
                        for key,z,b,weight in zip(('Ef','Erho','Eq'),(f,rho,q),targets,(w,1,1)):
                            error=float(np.sqrt(np.sum(weight*(z-b)**2)/np.sum(weight*b*b)));index=(step,kind,key)
                            if (nx,nv)==(512,128):base[index]=error
                            change=abs(error-base[index])/max(base[index],1e-300)
                            rows.append(dict(run=p.parent.name,nx=nx,nv=nv,step=step,reference=kind,metric=key,error=error,relative_change=change,passed=change<.01))
        with (OUT/'mm_test_refinement.csv').open('w') as f:
            wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    elif task.startswith('reference_'):
        from run_e5_mixed import config
        from numerical.parity_reference import solve_parity_gmres_1d
        nx,nv=map(int,task.split('_')[1:]);r=solve_parity_gmres_1d(config(),grid=(nx,nv),tol=1e-11)
        if not r['converged']:raise RuntimeError('reference not converged')
        np.savez_compressed(OUT/f'{task}.npz',**r)
    elif task.startswith('timing_'):
        import run_e5_mixed as m
        _,method,angles,repeat=task.split('_');m.OUT=OUT/task;m.OUT.mkdir(exist_ok=True)
        if not (m.OUT/'reference').exists():
            (m.OUT/'reference').symlink_to(ROOT/'results/e5_mixed/reference',target_is_directory=True)
        # Do not rewrite the official reports from a timing replay.
        m.subprocess.run=lambda *a,**kw: None
        sys.argv=[m.__file__,'--method',method,'--angles',angles,'--seed','11'];m.main()

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'queue.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=dict(os.environ,PYTHONPATH=str(ROOT/'src')+':'+str(ROOT/'scripts'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',JAX_ENABLE_X64='true')
    jobs=[('mm_grid',[sys.executable,__file__,'mm_grid'])]
    for cutoff in ('1e-5','1e-6','1e-7'):
        jobs.append(('p5_'+cutoff,[sys.executable,'scripts/run_p3_oe_aprfm.py','--problem','p5','--epsilon','0.001','--seed','11','--partitions','1','1','2','--features','128','--scale','.25','--rcond',cutoff,'--block-weights','10','1','1','1','--collocation','32','32','16','--output-dir',str(OUT/('p5_'+cutoff))]))
    for grid in ('1024_128','512_256','1024_256'):
        jobs.append(('reference_'+grid,[sys.executable,__file__,'reference_'+grid]))
    for method in ('oe','mm'):
        for angles in (32,64):
            for repeat in range(3):
                name=f'timing_{method}_{angles}_{repeat}';jobs.append((name,[sys.executable,__file__,name]))
    status_path=OUT/'status.json';status=json.loads(status_path.read_text()) if status_path.exists() else {}
    for name,cmd in jobs:
        if status.get(name,{}).get('state')=='completed':continue
        status[name]=dict(state='running',started=time.time(),command=cmd);status_path.write_text(json.dumps(status,indent=2))
        with (OUT/(name+'.log')).open('a') as log:
            result=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        status[name].update(state='completed' if result.returncode==0 else 'failed',returncode=result.returncode,finished=time.time())
        status_path.write_text(json.dumps(status,indent=2))
    print('Queue finished; inspect status.json. Completion is not scientific acceptance.',flush=True)

if __name__=='__main__':
    if len(sys.argv)>1:
        OUT.mkdir(parents=True,exist_ok=True);worker(sys.argv[1])
    else:main()
