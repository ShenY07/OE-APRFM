"""Create fixed-local-capacity 1-D/2-D domain-decomposition tables and figures."""
import csv,json,os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR","/tmp/matplotlib-oerfm")
import matplotlib
import mpl_toolkits
local=str(Path(matplotlib.__file__).parent.parent/"mpl_toolkits"); mpl_toolkits.__path__[:]=[local]+[p for p in mpl_toolkits.__path__ if p!=local]
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa:F401
import numpy as np

roots={"1D":Path("results/domain_decomposition/1d_fixed_local"),"2D":Path("results/quadrant/domain_decomposition_fixed_local")}
records=[]
for dim,root in roots.items():
  for path in root.glob("*.json"):
    r=json.loads(path.read_text()); px=int(r["partitions"][0]); py=1 if dim=="1D" else int(r["partitions"][1]); patches=px*py
    records.append({"dimension":dim,"epsilon":r["epsilon"],"partition":f"{px}" if dim=="1D" else f"{px}x{py}","spatial_patches":patches,"features_per_patch":r["features_per_patch"],"total_features":patches*r["features_per_patch"],"num_coefficients":r.get("num_columns",2*patches*r["features_per_patch"]),"relative_l2_f":r["relative_l2_f"],"relative_l2_rho":r.get("relative_l2_rho",""),"condition_number":r["condition_number"],"total_seconds":r["total_seconds"],"file":str(path)})
records.sort(key=lambda r:(r["dimension"],-float(r["epsilon"]),r["spatial_patches"]))
out=Path("results/tables/domain_decomposition_fixed_local_1d2d.csv"); fields=list(records[0]);
with out.open("w",newline="") as h: w=csv.DictWriter(h,fieldnames=fields); w.writeheader(); w.writerows(records)

fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
for ax,dim in zip(axes,("1D","2D")):
  for eps in (1.,1e-3):
    rr=sorted((r for r in records if r["dimension"]==dim and float(r["epsilon"])==eps),key=lambda r:r["spatial_patches"])
    ax.semilogy([r["spatial_patches"] for r in rr],[r["relative_l2_f"] for r in rr],linewidth=2.2,label=fr"$\varepsilon={eps:.0e}$")
  ax.set(xlabel="number of spatial patches",ylabel=r"relative $L^2$ error of $f$",title=dim+": 64 features per patch",xticks=(1,2,4)); ax.grid(alpha=.3); ax.legend(frameon=False)
figdir=Path("results/figures/domain_decomposition"); figdir.mkdir(parents=True,exist_ok=True); fig.savefig(figdir/"fixed_features_per_patch_1d2d.png",dpi=300); plt.close(fig)

# Best 1-D phase-space solution for each epsilon: reference, approximation, error.
fig=plt.figure(figsize=(15,8),constrained_layout=True)
for row,eps in enumerate((1.,1e-3)):
  rr=[r for r in records if r["dimension"]=="1D" and float(r["epsilon"])==eps]; best=min(rr,key=lambda r:r["relative_l2_f"]); path=Path(best["file"])
  with np.load(path.with_suffix(".npz")) as d: x,v,f,exact=d["x"],d["v"],d["f"],d["exact"]
  xx,vv=np.meshgrid(x,v,indexing="ij")
  for col,(z,title) in enumerate(zip((exact,f,np.abs(f-exact)),("reference $f$","best OE-RFM $f$","absolute error"))):
    ax=fig.add_subplot(2,3,row*3+col+1,projection="3d"); s=ax.plot_surface(xx,vv,z,cmap="magma" if col==2 else "viridis",linewidth=0); ax.set(xlabel="x",ylabel="v",title=title+fr", $\varepsilon={eps:.0e}$"); fig.colorbar(s,ax=ax,shrink=.55,pad=.08)
fig.savefig(figdir/"p1_best_f_xv_3d.png",dpi=300); plt.close(fig)

# Best 2-D fixed-local-capacity solution at theta=pi/4.
fig=plt.figure(figsize=(15,8),constrained_layout=True)
for row,eps in enumerate((1.,1e-3)):
  rr=[r for r in records if r["dimension"]=="2D" and float(r["epsilon"])==eps]; best=min(rr,key=lambda r:r["relative_l2_f"]); path=Path(best["file"])
  with np.load(path.with_suffix(".npz")) as d: x,y,t=d["x"],d["y"],d["theta"]; f=d["f"]; ref=d["reference_f"]
  k=int(np.argmin(abs(t-np.pi/4))); xx,yy=np.meshgrid(x,y,indexing="ij")
  for col,(z,title) in enumerate(zip((ref[:,:,k],f[:,:,k],np.abs(f[:,:,k]-ref[:,:,k])),("reference $f$","best OE-RFM $f$","absolute error"))):
    ax=fig.add_subplot(2,3,row*3+col+1,projection="3d"); s=ax.plot_surface(xx,yy,z,cmap="magma" if col==2 else "viridis",linewidth=0); ax.set(xlabel="x",ylabel="y",title=title+fr", $\varepsilon={eps:.0e}$, {best['partition']} patches"); fig.colorbar(s,ax=ax,shrink=.55,pad=.08)
fig.savefig(figdir/"p3_best_fixed_local_3d.png",dpi=300); plt.close(fig)
print(out)
