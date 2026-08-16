#!/usr/bin/env python3
"""Create separate title-free reference/solution/error figures for OE-APNN."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

root = Path("results/baselines/oe_apnn_imported")
out = Path("results/figures/oe_apnn")
out.mkdir(parents=True, exist_ok=True)

def save2d(x, y, z, path, label):
    fig, ax = plt.subplots(figsize=(6.8, 5.4), constrained_layout=True)
    obj=ax.contourf(x,y,z.T,80,cmap="viridis" if "err" not in path.name else "magma")
    ax.set_xlabel("x"); ax.set_ylabel(label); fig.colorbar(obj,ax=ax)
    fig.savefig(path,dpi=320); plt.close(fig)

def save3d(x, y, z, path, ylabel):
    xx,yy=np.meshgrid(x,y,indexing="ij"); fig=plt.figure(figsize=(7.2,5.8),constrained_layout=True)
    ax=fig.add_subplot(111,projection="3d"); obj=ax.plot_surface(xx,yy,z,cmap="viridis" if "err" not in path.name else "magma",linewidth=0,antialiased=True)
    ax.set_xlabel("x"); ax.set_ylabel(ylabel); ax.set_zlabel("f" if "err" not in path.name else "|e|"); fig.colorbar(obj,ax=ax,shrink=.65,pad=.1)
    fig.savefig(path,dpi=320); plt.close(fig)

for pi in range(1, 6):
  for label,short in (("eps_1e0","e0"),("eps_1e-3","e3")):
    dirs=list((root/f"P{pi}"/label).glob("seed_*"))
    ready=[]
    for d in dirs:
      m=json.loads((d/"metrics.json").read_text())
      if m.get("E_f") is not None and (d/"solution.npz").exists(): ready.append((m["E_f"],d))
    if not ready: continue
    _,d=sorted(ready)[len(ready)//2]; a=np.load(d/"solution.npz")
    if pi<=2:
      x,v=a["x"],a["velocity"]
      for key,abbr in (("reference_f","ref"),("f","sol"),("error_f","err")):
        z=a[key]; save2d(x,v,z,out/f"apnn_p{pi}_{short}_{abbr}_f2d.png","v"); save3d(x,v,z,out/f"apnn_p{pi}_{short}_{abbr}_f3d.png","v")
    else:
      x,y,t=a["x"],a["y"],a["theta"]; k=int(np.argmin(np.abs(t-np.pi/4)))
      for key,abbr in (("reference_f","ref"),("f","sol"),("error_f","err")):
        z=a[key][:,:,k]
        if "mask" in a.files:
          z=np.where(a["mask"],z,np.nan)
        save2d(x,y,z,out/f"apnn_p{pi}_{short}_{abbr}_f2d.png","y"); save3d(x,y,z,out/f"apnn_p{pi}_{short}_{abbr}_f3d.png","y")
print(f"figures written to {out}")
