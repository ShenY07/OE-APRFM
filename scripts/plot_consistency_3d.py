"""Diverse 2-D/3-D visualizations for fixed-parameter P2/P3 experiments."""

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-oerfm")
import matplotlib
import mpl_toolkits
# Debian's system mpl_toolkits may precede the user-installed Matplotlib and
# make mplot3d ABI-incompatible. Prefer the toolkit shipped beside Matplotlib.
_local_toolkit = str(Path(matplotlib.__file__).parent.parent / "mpl_toolkits")
mpl_toolkits.__path__[:] = [_local_toolkit] + [p for p in mpl_toolkits.__path__ if p != _local_toolkit]
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401; registers projection='3d'
import numpy as np


FIG = Path("results/figures/consistency")


def representative(root, problem, epsilon):
    items = [(p, json.loads(p.read_text())) for p in root.glob(f"{problem}_oe_aprfm_eps_{epsilon:.0e}_seed_*_fixed.json")]
    target = np.median([r["relative_l2_f"] for _, r in items])
    return min(items, key=lambda item: abs(item[1]["relative_l2_f"] - target))


def surface(axis, xx, yy, zz, title, ylabel, cmap="viridis"):
    artist = axis.plot_surface(xx, yy, zz, cmap=cmap, linewidth=0, antialiased=True)
    axis.set_title(title); axis.set_xlabel("x"); axis.set_ylabel(ylabel)
    return artist


def plot_p2():
    root = Path("results/consistency/p2")
    fig = plt.figure(figsize=(15, 8), constrained_layout=True)
    for row, eps in enumerate((1.0, 1e-3)):
        path, rec = representative(root, "p2", eps)
        with np.load(path.with_suffix(".npz")) as d:
            x, v, f, ref = d["x"], d["velocity"], d["f"], d["reference_f"]
        xx, vv = np.meshgrid(x, v, indexing="ij"); fields = (ref, f, np.abs(f-ref))
        for col, (z, title) in enumerate(zip(fields, ("reference $f$", "OE-RFM $f$", "absolute error"))):
            ax = fig.add_subplot(2, 3, row*3+col+1, projection="3d")
            art = surface(ax, xx, vv, z, title+fr", $\varepsilon={eps:.0e}$", "v", "magma" if col==2 else "viridis")
            fig.colorbar(art, ax=ax, shrink=.55, pad=.08)
        ax.text2D(.02,.92,fr"$E_f={rec['relative_l2_f']:.2e}$",transform=ax.transAxes)
    FIG.mkdir(parents=True, exist_ok=True); fig.savefig(FIG/"p2_f_xv_3d.png", dpi=260); plt.close(fig)

    fig, axes = plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    for row, eps in enumerate((1.0,1e-3)):
        path,_=representative(root,"p2",eps)
        with np.load(path.with_suffix(".npz")) as d: x,v,f,ref=d["x"],d["velocity"],d["f"],d["reference_f"]
        for ax,z,title in zip(axes[row],(ref,f,np.abs(f-ref)),("reference","OE-RFM","absolute error")):
            im=ax.contourf(x,v,z.T,40,cmap="magma" if "error" in title else "viridis"); ax.set(xlabel="x",ylabel="v",title=title+fr" ($\varepsilon={eps:.0e}$)"); fig.colorbar(im,ax=ax)
    fig.savefig(FIG/"p2_f_xv_contours.png",dpi=260); plt.close(fig)


def plot_p3():
    root=Path("results/consistency/p3"); theta_target=np.pi/4
    fig=plt.figure(figsize=(15,8),constrained_layout=True)
    for row,eps in enumerate((1.0,1e-3)):
        path,rec=representative(root,"p3",eps)
        with np.load(path.with_suffix(".npz")) as d:
            x,y,t,f,ref=d["x"],d["y"],d["theta"],d["f"],d["reference_f"]
        k=int(np.argmin(abs(t-theta_target))); xx,yy=np.meshgrid(x,y,indexing="ij"); fields=(ref[:,:,k],f[:,:,k],np.abs(f[:,:,k]-ref[:,:,k]))
        for col,(z,title) in enumerate(zip(fields,("reference $f$","OE-RFM $f$","absolute error"))):
            ax=fig.add_subplot(2,3,row*3+col+1,projection="3d"); art=surface(ax,xx,yy,z,title+fr", $\theta=\pi/4$, $\varepsilon={eps:.0e}$","y","magma" if col==2 else "viridis"); fig.colorbar(art,ax=ax,shrink=.55,pad=.08)
        ax.text2D(.02,.92,fr"$E_f={rec['relative_l2_f']:.2e}$",transform=ax.transAxes)
    FIG.mkdir(parents=True,exist_ok=True); fig.savefig(FIG/"p3_f_xy_theta_pi4_3d.png",dpi=260); plt.close(fig)

    fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
    points=((0,0),(.5,-.5))
    for row,eps in enumerate((1.0,1e-3)):
        path,_=representative(root,"p3",eps)
        with np.load(path.with_suffix(".npz")) as d: x,y,t,f,ref=d["x"],d["y"],d["theta"],d["f"],d["reference_f"]
        for col,(xp,yp) in enumerate(points):
            i,j=np.argmin(abs(x-xp)),np.argmin(abs(y-yp)); ax=axes[row,col]
            ax.plot(t,ref[i,j],label="reference"); ax.plot(t,f[i,j],"--",label="OE-RFM"); ax.set(xlabel=r"$\theta$",ylabel="$f$",title=fr"$(x,y)=({xp:g},{yp:g}),\ \varepsilon={eps:.0e}$"); ax.grid(alpha=.3)
    axes[0,0].legend(frameon=False); fig.savefig(FIG/"p3_angular_linecuts.png",dpi=260); plt.close(fig)


if __name__ == "__main__":
    plot_p2(); plot_p3()
