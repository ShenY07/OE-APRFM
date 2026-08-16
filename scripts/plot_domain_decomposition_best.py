"""Plot decomposition trends and only the best P3 fields (no point plots)."""

import csv, json, os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR","/tmp/matplotlib-oerfm")
import matplotlib
import mpl_toolkits
local=str(Path(matplotlib.__file__).parent.parent/"mpl_toolkits"); mpl_toolkits.__path__[:]=[local]+[p for p in mpl_toolkits.__path__ if p!=local]
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa:F401
import numpy as np

root=Path("results/quadrant/domain_decomposition"); final=Path("results/quadrant/final"); out=Path("results/figures/quadrant/best"); out.mkdir(parents=True,exist_ok=True)

def candidates(eps):
    paths=list(root.glob(f"p3_oe_aprfm_eps_{eps:.0e}_seed_11_*.json"))+list(final.glob(f"p3_oe_aprfm_eps_{eps:.0e}_seed_11_quadrant_f256.json"))
    return [(p,json.loads(p.read_text())) for p in paths]

rows=list(csv.DictReader(open("results/tables/p3_domain_decomposition.csv")))
fig,axes=plt.subplots(1,2,figsize=(9,3.7),constrained_layout=True)
for eps in (1.,1e-3):
    rr=sorted((r for r in rows if float(r["epsilon"])==eps),key=lambda r:int(r["spatial_patches"]))
    n=[int(r["spatial_patches"]) for r in rr]
    axes[0].semilogy(n,[float(r["relative_l2_f"]) for r in rr],linewidth=2.2,label=fr"$\varepsilon={eps:.0e}$")
    axes[1].semilogy(n,[float(r["relative_l2_rho"]) for r in rr],linewidth=2.2,label=fr"$\varepsilon={eps:.0e}$")
for ax,title in zip(axes,(r"relative $L^2$ error of $f$",r"relative $L^2$ error of $\rho$")):
    ax.set(xlabel="number of spatial patches",ylabel="relative error",title=title,xticks=(1,2,4)); ax.grid(alpha=.3); ax.legend(frameon=False)
fig.savefig(out/"p3_domain_decomposition_accuracy.png",dpi=300); plt.close(fig)

for eps in (1.,1e-3):
    path,rec=min(candidates(eps),key=lambda z:z[1]["relative_l2_f"])
    with np.load(path.with_suffix(".npz")) as d: x,y,t=d["x"],d["y"],d["theta"]; f=d["f"]; ref=d["reference_f"]; mask=d["mask"].astype(bool)
    k=int(np.argmin(abs(t-np.pi/4))); xx,yy=np.meshgrid(x,y,indexing="ij")
    fig=plt.figure(figsize=(14,4.5),constrained_layout=True)
    for col,(z,title) in enumerate(zip((ref[:,:,k],f[:,:,k],np.abs(f[:,:,k]-ref[:,:,k])),("reference $f$","best OE-RFM $f$","absolute error"))):
        ax=fig.add_subplot(1,3,col+1,projection="3d"); z=np.where(mask,z,np.nan); s=ax.plot_surface(xx,yy,z,cmap="magma" if col==2 else "viridis",linewidth=0); ax.set(xlabel="x",ylabel="y",title=title+fr", $\varepsilon={eps:.0e}$"); fig.colorbar(s,ax=ax,shrink=.6,pad=.08)
    fig.savefig(out/f"p3_best_eps_{eps:.0e}_3d.png",dpi=300); plt.close(fig)
print(out)
