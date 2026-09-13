"""Reusable image-derived sparse-track model; loading does not execute fitting.

The return value local_polynomial_gradient is the local fitted polynomial slope.
It is not the full derivative of the moving weighted-regression estimator.
"""
import numpy as np
from scipy.spatial import cKDTree

class PTVModel:
    def __init__(self,path='work/ptv_tracks.npz'):
        z=np.load(path);m=z['accepted']
        self.points=z['points'][m];self.disp=z['disp'][m]
        self.tree=cKDTree(self.points)
        self.quality=np.clip((z['ncc'][m]-.5)/.4,.1,1)*np.clip(z['ambiguity_gap'][m]/.08,.1,1)*np.exp(-z['fb'][m]**2/.5)
    def evaluate(self,query,scale=10,order=2):
        pred=[];grad=[];rms=[];n=[];reach=[]
        for pt in np.atleast_2d(query):
            dd,ii=self.tree.query(pt,k=min(45,len(self.points)))
            near=dd<max(scale*2,dd[min(11,len(dd)-1)])
            dd=dd[near];ii=ii[near];q=(self.points[ii]-pt)/scale
            X=np.c_[np.ones(len(q)),q[:,0],q[:,1]]
            if order==2:X=np.c_[X,.5*q[:,0]**2,q[:,0]*q[:,1],.5*q[:,1]**2]
            base=np.exp(-.5*(dd/scale)**2)*self.quality[ii];w=base.copy();Y=self.disp[ii]
            ridge=np.diag([1e-8,1e-5,1e-5]+([.04,.04,.04] if order==2 else []))
            for _ in range(7):
                coef=np.linalg.solve(X.T@(w[:,None]*X)+ridge,X.T@(w[:,None]*Y))
                residual=np.linalg.norm(Y-X@coef,axis=1)
                sc=max(.45,1.4826*np.median(residual))
                w=base/(1+(residual/(2*sc))**4)
            pred.append(coef[0]);grad.append(coef[1:3].T/scale);rms.append(np.sqrt(np.sum(w*residual**2)/sum(w)));n.append(np.sum(w>.1));reach.append(dd.min())
        return {'disp':np.array(pred),'local_polynomial_gradient':np.array(grad),'rms':np.array(rms),'count':np.array(n),'nearest_feature':np.array(reach)}
    def gradient(self,query,scale=10,h=1.):
        query=np.atleast_2d(query)
        dx=(self.evaluate(query+[h,0],scale)['disp']-self.evaluate(query-[h,0],scale)['disp'])/(2*h)
        dy=(self.evaluate(query+[0,h],scale)['disp']-self.evaluate(query-[0,h],scale)['disp'])/(2*h)
        return np.stack([dx,dy],axis=-1)
