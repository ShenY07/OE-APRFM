"""Generate publication figures as separate, title-free image files."""
import json, os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR","/tmp/matplotlib-oerfm")
import matplotlib
import mpl_toolkits
local=str(Path(matplotlib.__file__).parent.parent/"mpl_toolkits")
mpl_toolkits.__path__[:]=[local]+[p for p in mpl_toolkits.__path__ if p!=local]
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa:F401
import numpy as np
from scipy.interpolate import RectBivariateSpline

OUT=Path("results/figures/latest"); OUT.mkdir(parents=True,exist_ok=True)
EP=((1.0,"e0"),(1e-3,"e3"))

def save2(x,y,z,path,xlabel="x",ylabel="y"):
    fig,ax=plt.subplots(figsize=(6,5),constrained_layout=True)
    im=ax.imshow(z.T,origin="lower",extent=(x[0],x[-1],y[0],y[-1]),aspect="auto",cmap="magma" if "err" in path.name else "viridis")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); fig.colorbar(im,ax=ax)
    fig.savefig(path,dpi=320,bbox_inches="tight"); plt.close(fig)

def save3(x,y,z,path,ylabel="y"):
    xx,yy=np.meshgrid(x,y,indexing="ij"); fig=plt.figure(figsize=(6.4,5.2),constrained_layout=True); ax=fig.add_subplot(111,projection="3d")
    s=ax.plot_surface(xx,yy,z,cmap="magma" if "err" in path.name else "viridis",linewidth=0,antialiased=True)
    ax.set_xlabel("x"); ax.set_ylabel(ylabel); fig.colorbar(s,ax=ax,shrink=.62,pad=.09)
    fig.savefig(path,dpi=320,bbox_inches="tight"); plt.close(fig)

def refine_field(x, y, z, n):
    """Interpolate a stored field onto a denser grid for publication plots."""
    xd=np.linspace(float(x[0]),float(x[-1]),n)
    yd=np.linspace(float(y[0]),float(y[-1]),n)
    zd=RectBivariateSpline(x,y,z,kx=3,ky=3)(xd,yd)
    return xd,yd,zd

def p2_record(eps):
    root=Path("results/consistency/p2"); items=[(p,json.loads(p.read_text())) for p in root.glob(f"p2_oe_aprfm_eps_{eps:.0e}_seed_*_fixed.json")]
    med=np.median([r["relative_l2_f"] for _,r in items]); return min(items,key=lambda z:abs(z[1]["relative_l2_f"]-med))[0]

def plot_p2():
    for eps,etag in EP:
        p=p2_record(eps)
        with np.load(p.with_suffix(".npz")) as d: x,v,f,ref=d["x"],d["velocity"],d["f"],d["reference_f"]
        for key,z in (("ref",ref),("sol",f),("err",np.abs(f-ref))):
            save2(x,v,z,OUT/f"p2_{etag}_{key}_f2d.png",ylabel="v")
            save3(x,v,z,OUT/f"p2_{etag}_{key}_f3d.png",ylabel="v")

def q_record(problem,eps):
    files=list(Path("results/quadrant/final").glob(f"{problem}_oe_aprfm_eps_{eps:.0e}_seed_11_*.json")); return min(files,key=lambda p:json.loads(p.read_text())["relative_l2_f"])

def plot_quadrant():
    for problem in ("p3","p4","p5"):
      for eps,etag in EP:
        p=q_record(problem,eps)
        with np.load(p.with_suffix(".npz")) as d:
            x,y,t=d["x"],d["y"],d["theta"]; f=d["f"]; ref=d["reference_f"]; rho=d["rho"]; rr=d["reference_rho"]; mask=d["mask"].astype(bool)
        k=int(np.argmin(abs(t-np.pi/4)))
        plot_n=321 if problem=="p4" else 193
        xd,yd,_=refine_field(x,y,rr,plot_n)
        if problem=="p4":
            xx,yy=np.meshgrid(xd,yd,indexing="ij")
            plot_mask=xx**2+yy**2>=.25
        else:
            plot_mask=np.ones((plot_n,plot_n),dtype=bool)
        for key,z in (("ref",rr),("sol",rho),("err",np.abs(rho-rr))):
            _,_,zd=refine_field(x,y,z,plot_n)
            save2(xd,yd,np.where(plot_mask,zd,np.nan),OUT/f"{problem}_{etag}_{key}_rho2d.png")
        for key,z in (("ref",ref[:,:,k]),("sol",f[:,:,k]),("err",np.abs(f[:,:,k]-ref[:,:,k]))):
            _,_,zd=refine_field(x,y,z,plot_n)
            save3(xd,yd,np.where(plot_mask,zd,np.nan),OUT/f"{problem}_{etag}_{key}_f3d_th45.png")

def dd_records():
    import csv
    return list(csv.DictReader(open("results/tables/domain_decomposition_fixed_local_1d2d.csv")))

def plot_dd():
    rows=dd_records()
    for dim in ("1D","2D"):
        fig,ax=plt.subplots(figsize=(6,4.5),constrained_layout=True)
        for eps,etag in EP:
            rr=sorted((r for r in rows if r["dimension"]==dim and float(r["epsilon"])==eps),key=lambda r:int(r["spatial_patches"]))
            ax.semilogy([int(r["spatial_patches"]) for r in rr],[float(r["relative_l2_f"]) for r in rr],linewidth=2.2,label=fr"$\varepsilon={eps:.0e}$")
        ax.set_xlabel("number of spatial patches"); ax.set_ylabel(r"relative $L^2$ error of $f$"); ax.set_xticks((1,2,4)); ax.grid(alpha=.3); ax.legend(frameon=False)
        fig.savefig(OUT/f"dd_{dim.lower()}_ef.png",dpi=320,bbox_inches="tight"); plt.close(fig)
    for dim,problem in (("1D","p1"),("2D","p3")):
      for eps,etag in EP:
        rr=[r for r in rows if r["dimension"]==dim and float(r["epsilon"])==eps]; best=min(rr,key=lambda r:float(r["relative_l2_f"])); p=Path(best["file"])
        with np.load(p.with_suffix(".npz")) as d:
          if dim=="1D": x,y,f,ref=d["x"],d["v"],d["f"],d["exact"]; ylabel="v"
          else:
            x,y,t,f,ref=d["x"],d["y"],d["theta"],d["f"],d["reference_f"]; k=int(np.argmin(abs(t-np.pi/4))); f,ref=f[:,:,k],ref[:,:,k]; ylabel="y"
        for key,z in (("ref",ref),("sol",f),("err",np.abs(f-ref))): save3(x,y,z,OUT/f"dd_{problem}_{etag}_{key}_f3d.png",ylabel=ylabel)

if __name__=="__main__": plot_p2(); plot_quadrant(); plot_dd()
