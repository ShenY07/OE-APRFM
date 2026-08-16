"""Plot first-quadrant OE-RFM 2-D fields and 3-D solution/error surfaces."""

import json, os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-oerfm")
import matplotlib
import mpl_toolkits
local = str(Path(matplotlib.__file__).parent.parent / "mpl_toolkits")
mpl_toolkits.__path__[:] = [local] + [p for p in mpl_toolkits.__path__ if p != local]
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np

ROOT=Path("results/quadrant/final"); OUT=Path("results/figures/quadrant")

def record(problem, eps):
    files=list(ROOT.glob(f"{problem}_oe_aprfm_eps_{eps:.0e}_seed_11_quadrant*.json"))
    path=min(files,key=lambda p:json.loads(p.read_text())["relative_l2_f"])
    return path,json.loads(path.read_text())

def plot(problem):
    fig2,ax2=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    fig3=plt.figure(figsize=(15,8),constrained_layout=True)
    for row,eps in enumerate((1.,1e-3)):
        path,rec=record(problem,eps)
        with np.load(path.with_suffix(".npz")) as d:
            x,y,t=d["x"],d["y"],d["theta"]; f=d["f"]; ref=d["reference_f"]
            rho=d["rho"]; rr=d["reference_rho"]; mask=d["mask"].astype(bool)
        er=np.abs(rho-rr); fields=(rr,rho,er)
        for ax,z,title in zip(ax2[row],fields,("reference density","OE-RFM density","absolute density error")):
            z=np.where(mask,z,np.nan); im=ax.imshow(z.T,origin="lower",extent=(x[0],x[-1],y[0],y[-1]),cmap="magma" if "error" in title else "viridis"); ax.set(xlabel="x",ylabel="y",title=title+fr" ($\varepsilon={eps:.0e}$)"); fig2.colorbar(im,ax=ax)
        k=np.argmin(abs(t-np.pi/4)); xx,yy=np.meshgrid(x,y,indexing="ij")
        for col,(z,title) in enumerate(zip((ref[:,:,k],f[:,:,k],np.abs(f[:,:,k]-ref[:,:,k])),("reference $f$","OE-RFM $f$","absolute $f$ error"))):
            z=np.where(mask,z,np.nan); ax=fig3.add_subplot(2,3,row*3+col+1,projection="3d"); surf=ax.plot_surface(xx,yy,z,cmap="magma" if col==2 else "viridis",linewidth=0); ax.set(xlabel="x",ylabel="y",title=title+fr", $\theta=\pi/4$, $\varepsilon={eps:.0e}$"); fig3.colorbar(surf,ax=ax,shrink=.55,pad=.08)
        ax.text2D(.02,.92,fr"$E_f={rec['relative_l2_f']:.2e}$\n$E_\rho={rec['relative_l2_rho']:.2e}$",transform=ax.transAxes)
    OUT.mkdir(parents=True,exist_ok=True); fig2.savefig(OUT/f"{problem}_density_2d.png",dpi=260); fig3.savefig(OUT/f"{problem}_f_theta_pi4_3d.png",dpi=260); plt.close(fig2); plt.close(fig3)

if __name__=="__main__":
    for p in ("p3","p4","p5"): plot(p)
