"""Smooth normalized compact PoU features for the exploratory P7 solver."""
import numpy as np
from run_p7_transient_pilot import Features


class LocalFeatures(Features):
    def __init__(self, n, seed, *, t_edges=(0,.1), x_edges=(0,1)):
        cells=[(ta,tb,xa,xb) for ta,tb in zip(t_edges[:-1],t_edges[1:])
               for xa,xb in zip(x_edges[:-1],x_edges[1:])]
        self.centers=np.array([((ta+tb)/2,(xa+xb)/2) for ta,tb,xa,xb in cells])
        self.radii=np.array([((tb-ta)/2,(xb-xa)/2) for ta,tb,xa,xb in cells])
        self.parts=[]
        for k in range(len(cells)):
            count=n//len(cells)+(k<n%len(cells))
            if count==0: raise ValueError('Each local patch requires at least one feature')
            self.parts.append(Features(count,seed+7919*k))

    @staticmethod
    def bump(z):
        inside=abs(z)<1
        d=np.where(inside,1-z*z,1)
        value=np.where(inside,np.exp(-1/d),0)
        derivative=np.where(inside,value*(-2*z/d**2),0)
        return value,derivative

    def weights(self,t,x):
        q=(np.column_stack((t,x))[:,None,:]-self.centers)/(1.75*self.radii)
        val,der=self.bump(q)
        g=val[:,:,0]*val[:,:,1]
        gt=der[:,:,0]*val[:,:,1]/(1.75*self.radii[:,0])
        gx=val[:,:,0]*der[:,:,1]/(1.75*self.radii[:,1])
        total=g.sum(axis=1,keepdims=True)
        if np.any(total<=0): raise ValueError('Evaluation point outside covered domain')
        psi=g/total
        return psi,(gt-psi*gt.sum(axis=1,keepdims=True))/total,(gx-psi*gx.sum(axis=1,keepdims=True))/total

    def raw(self,t,x,v):
        psi,pt,px=self.weights(t,x)
        values=[]; dts=[]; dxs=[]
        for k,f in enumerate(self.parts):
            z=np.column_stack(((t-self.centers[k,0])/self.radii[k,0],
                               (x-self.centers[k,1])/self.radii[k,1],v))@f.a+f.b
            phi=np.tanh(z); dphi=1-phi*phi
            values.append(psi[:,k,None]*phi)
            dts.append(pt[:,k,None]*phi+psi[:,k,None]*dphi*f.a[0]/self.radii[k,0])
            dxs.append(px[:,k,None]*phi+psi[:,k,None]*dphi*f.a[1]/self.radii[k,1])
        return tuple(np.column_stack(a) for a in (values,dts,dxs))
