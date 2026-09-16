"""Continuous and BE Fourier references on common probe velocities, no interpolation."""
import json,csv
from pathlib import Path
import numpy as np
from scipy.linalg import expm,lu_factor,lu_solve
from e6_protocol import validate
ROOT=Path(__file__).resolve().parents[1]/'results/e6_periodic'

def amplitude(eps,dt,n,probe):
    v,w=np.polynomial.legendre.leggauss(n);w=w/2
    # Native quadrature evolves independently. Probe ODEs are forced by its
    # density and do not feed back: same angular points for both refinements.
    L=(np.ones((n,1))*w-np.eye(n))/eps**2-2j*np.pi*np.diag(v)/eps
    m=len(probe);K=np.zeros((n+m,n+m),complex);K[:n,:n]=L
    K[n:,:n]=np.ones((m,1))*w/eps**2
    K[n:,n:]=np.diag(-1/eps**2-2j*np.pi*probe/eps)
    initial=np.ones(n+m,complex);a=initial.copy();lu=lu_factor(np.eye(n+m)-dt*K)
    result={0.:(initial,initial)}
    for step in range(1,round(.2/dt)+1):
        a=lu_solve(lu,a);t=round(step*dt,10)
        if t in (.02,.1,.2):result[t]=(expm(t*K)@initial,a.copy())
    return result,w

def rel(a,b,w):return float(np.sqrt(np.sum(w*abs(a-b)**2)/np.sum(w*abs(b)**2)))

def main():
    cache=ROOT/'references';cache.mkdir(exist_ok=True)
    rows=[]
    for path in sorted(ROOT.glob('*/*/result.json')):
        if not validate(path):continue
        r=json.loads(path.read_text());d=np.load(path.parent/'fields.npz');x=d['x'];v=d['v'];wv=d['w']
        eps,dt=r['epsilon'],r['dt'];refs={}
        for n in (128,256):
            target=cache/f'eps{eps:.0e}_dt{dt:.0e}_n{n}.npz'
            if not target.exists():
                amps,w=amplitude(eps,dt,n,v);saved={'native_weights':w}
                for t,(ct,be) in amps.items():
                    saved[f'ct_{t:.2f}']=ct;saved[f'be_{t:.2f}']=be
                np.savez_compressed(target,**saved)
            refs[n]=np.load(target)
        for t in (0.,.02,.1,.2):
            step=round(t/dt);rho=d[f'rho_{step}'];q=d[f'q_{step}'];f=d[f'f_{step}']
            phase=np.exp(2j*np.pi*x)
            for kind in ('ct','be'):
                fields={}
                for n in (128,256):
                    a=refs[n][f'{kind}_{t:.2f}'];wn=refs[n]['native_weights'];vn=np.polynomial.legendre.leggauss(n)[0]
                    fields[n]=(1+.2*np.real(phase[:,None]*a[n:]),
                               1+.2*np.real(phase*(a[:n]@wn)),
                               .2*np.real(phase*(a[:n]@(wn*vn)))/eps)
                ff,rr,qq=fields[256];fc,rc,qc=fields[128]
                qnorm=float(np.sqrt(np.mean(qq**2)));qerr=float(np.sqrt(np.mean((q-qq)**2)))
                row=dict(run=str(path.parent.relative_to(ROOT)),epsilon=eps,dt=dt,seed=r['seed'],J=r['J'],t=t,reference=kind,
                    Ef=rel(f,ff,wv),Erho=rel(rho,rr,1),Eq_absolute=qerr,Eq_relative=qerr/qnorm if qnorm>1e-12 else None,
                    reference_delta_f=rel(fc,ff,wv),reference_delta_rho=rel(rc,rr,1),
                    reference_delta_q_absolute=float(np.sqrt(np.mean((qc-qq)**2))),
                    amplitude_reference=float(2*np.mean((rr-1)*np.cos(2*np.pi*x))),
                    Ediff_BE=r['history'][step]['Ediff_BE'])
                rows.append(row)
        with (ROOT/'reference_comparison.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        print(path.parent.name,'references compared',flush=True)
    print('Done; reference comparison is not RF test-grid refinement.')
if __name__=='__main__':main()
