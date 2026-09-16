"""Original psi_b RF convention extended to time; identical trial/test evaluation."""
import numpy as np
from run_p7_transient_pilot import Features


class ConsistentFeatures(Features):
    def __init__(self,n,seed,partitions=(1,2,2),scale=1.):
        nt,nx,nv=partitions
        self.centers=np.array([( (i+.5)*.1/nt,(j+.5)/nx,(k+.5)/nv)
                              for i in range(nt) for j in range(nx) for k in range(nv)])
        self.radius=np.array([.05/nt,.5/nx,.5/nv])
        if n%len(self.centers):raise ValueError('Feature count must divide evenly over patches')
        seed=seed-1000 if seed>=1000 else seed
        self.params=[np.random.default_rng(seed+7919*k).uniform(0,scale,(n//len(self.centers),4)) for k in range(len(self.centers))]
    @staticmethod
    def bump(q):
        a=abs(q)
        return (np.where(a<.75,1,np.where(a>1.25,0,.5*(1-np.sin(2*np.pi*a)))),
                np.where((a>=.75)&(a<=1.25),-np.pi*np.cos(2*np.pi*q)*np.sign(q),0))
    def raw(self,t,x,v):
        coords=np.column_stack((t,x,v))
        q=(coords[:,None,:]-self.centers[None,:,:])/self.radius
        qp=q.copy();qp[:,:,2]=(abs(v[:,None])-self.centers[:,2])/self.radius[2]
        b,db=self.bump(qp)
        g=b.prod(axis=2)
        gt=db[:,:,0]*b[:,:,1]*b[:,:,2]/self.radius[0]
        gx=b[:,:,0]*db[:,:,1]*b[:,:,2]/self.radius[1]
        total=g.sum(axis=1,keepdims=True)
        if np.any(total<=0):raise ValueError('Uncovered point')
        w=g/total;wt=(gt-w*gt.sum(axis=1,keepdims=True))/total;wx=(gx-w*gx.sum(axis=1,keepdims=True))/total
        result=[[],[],[]]
        for k,p in enumerate(self.params):
            f=np.tanh(q[:,k]@p[:,:3].T+p[:,3]);d=1-f*f
            result[0].append(w[:,k,None]*f)
            result[1].append(wt[:,k,None]*f+w[:,k,None]*d*p[:,0]/self.radius[0])
            result[2].append(wx[:,k,None]*f+w[:,k,None]*d*p[:,1]/self.radius[1])
        return tuple(np.column_stack(a) for a in result)
