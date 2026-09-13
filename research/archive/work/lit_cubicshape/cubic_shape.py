"""Local cubic image deformation and consistent compact-polynomial blending.
Extends work/extended_flow.py and field_model.py, retaining their objective.
"""
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

def basis(z,order=3):
    z=np.atleast_2d(z);x,y=z[:,0],z[:,1];o=np.ones(len(z));n=np.zeros(len(z))
    q=[o,x,y,.5*x*x,x*y,.5*y*y]
    dx=[n,o,n,x,y,n];dy=[n,n,o,n,x,y]
    if order==3:
        q += [x*x*x/6,.5*x*x*y,.5*x*y*y,y*y*y/6]
        dx += [.5*x*x,x*y,.5*y*y,n]
        dy += [n,.5*x*x,x*y,.5*y*y]
    return np.array(q).T,np.array(dx).T,np.array(dy).T

def fit_local(A,B,va,vb,center,seed,r=13,order=3,reg=5e-5,cubic_multiplier=16,maxiter=35):
    x,y=center;h,w=A.shape
    xi=np.arange(max(0,int(np.floor(x-r))),min(w,int(np.ceil(x+r+1))))
    yi=np.arange(max(0,int(np.floor(y-r))),min(h,int(np.ceil(y+r+1))))
    xx,yy=np.meshgrid(xi,yi);xx=xx.ravel();yy=yy.ravel();ok=va[yy,xx];xx,yy=xx[ok],yy[ok]
    z=(np.column_stack([xx,yy])-[x,y])/r;Q,Dx,Dy=basis(z,order);k=Q.shape[1]
    p=np.zeros((2,k));p[:,:min(k,seed.shape[1])]=seed[:,:min(k,seed.shape[1])]
    a=A[yy,xx];weights=np.exp(-np.sum(z*z,axis=1)/(2*.65**2));by,bx=np.gradient(B)
    lam=np.array([0,reg,reg]+[4*reg]*3+([cubic_multiplier*reg]*4 if order==3 else []))*weights.sum()
    def sample(pp):
        dest=np.column_stack([xx,yy])+Q@pp.T;coords=[dest[:,1],dest[:,0]]
        b=map_coordinates(B,coords,order=1,mode='nearest')
        valid=map_coordinates(vb.astype(float),coords,order=1,mode='constant',cval=0)>.99
        valid&=(dest[:,0]>=1)&(dest[:,0]<w-2)&(dest[:,1]>=1)&(dest[:,1]<h-2)
        return b,valid,coords
    it=-1
    for it in range(maxiter):
        b,valid,coords=sample(p);basew=weights*valid
        if basew.sum()<25:return p,dict(ncc=-1.,rms=100.,nvalid=0,cond=1e20,iterations=it,mindet=-1.,support_fraction=0.)
        M=np.column_stack([b,np.ones(len(b))]);gb=np.linalg.lstsq(M*np.sqrt(basew[:,None]),a*np.sqrt(basew),rcond=None)[0]
        gb[0]=np.clip(gb[0],.3,3);gb[1]=np.sum(basew*(a-gb[0]*b))/basew.sum()
        resid=gb[0]*b+gb[1]-a;wt=basew/np.sqrt(1+(resid/.8)**2)
        jx=map_coordinates(bx,coords,order=1,mode='nearest')*gb[0];jy=map_coordinates(by,coords,order=1,mode='nearest')*gb[0]
        J=np.column_stack([Q*jx[:,None],Q*jy[:,None]])
        H=J.T@(J*wt[:,None])+np.diag(np.tile(lam,2)+1e-4)
        grad=J.T@(wt*resid)+np.tile(lam,2)*p.ravel()
        try:step=np.linalg.solve(H,-grad).reshape(2,k)
        except np.linalg.LinAlgError:break
        movement=np.max(np.linalg.norm(Q@step.T,axis=1))
        if movement>2.5:step*=2.5/movement
        accepted=False
        for fac in [1,.5,.25,.1]:
            pp=p+fac*step;bb,vv,_=sample(pp);common=valid&vv;rr=gb[0]*bb+gb[1]-a
            obj=np.sum(weights[common]*.8**2*(np.sqrt(1+(rr[common]/.8)**2)-1))+.5*np.sum(lam*pp*pp)
            old=np.sum(weights[common]*.8**2*(np.sqrt(1+(resid[common]/.8)**2)-1))+.5*np.sum(lam*p*p)
            if obj<=old+1e-8 and common.sum()>=.95*valid.sum():p=pp;accepted=True;break
        if not accepted or np.max(np.abs(fac*step))<.007:break
    b,valid,_=sample(p);wt=weights*valid;M=np.column_stack([b,np.ones(len(b))]);gb=np.linalg.lstsq(M*np.sqrt(wt[:,None]),a*np.sqrt(wt),rcond=None)[0]
    gb[0]=np.clip(gb[0],.3,3);gb[1]=np.sum(wt*(a-gb[0]*b))/max(wt.sum(),1e-10)
    am=np.sum(wt*a)/max(wt.sum(),1e-10);bm=np.sum(wt*b)/max(wt.sum(),1e-10)
    ncc=np.sum(wt*(a-am)*(b-bm))/np.sqrt(np.sum(wt*(a-am)**2)*np.sum(wt*(b-bm)**2)+1e-12)
    rms=np.sqrt(np.sum(wt*(gb[0]*b+gb[1]-a)**2)/max(wt.sum(),1e-10))
    jac=np.stack([Dx@p.T,Dy@p.T],axis=2)/r;det=np.linalg.det(np.eye(2)[None]+jac)
    return p,dict(ncc=float(ncc),rms=float(rms),nvalid=int(valid.sum()),cond=float(np.linalg.cond(H)),iterations=it+1,mindet=float(det.min()),support_fraction=float(wt.sum()/weights.sum()))

class CubicField:
    def __init__(self,path,blend_radius=16):
        data=np.load(str(path));self.data={k:data[k] for k in data.files};self.points=data['points'];self.params=data['params'];self.radius=float(data['radius']);self.R=float(blend_radius);self.tree=cKDTree(self.points);self.order=3 if self.params.shape[2]==10 else 2
    def evaluate(self,query):
        out=[];gr=[]
        for point in np.atleast_2d(query):
            ii=self.tree.query_ball_point(point,self.R)
            if not ii:out.append([np.nan,np.nan]);gr.append(np.full((2,2),np.nan));continue
            ii=np.array(ii);distvec=point-self.points[ii];t=np.linalg.norm(distvec,axis=1)/self.R
            w=(1-t)**4*(1+4*t);dw=-20/self.R**2*(1-t[:,None])**3*distvec
            q,dx,dy=basis(distvec/self.radius,self.order)
            value=np.einsum('nij,nj->ni',self.params[ii],q)
            vx=np.einsum('nij,nj->ni',self.params[ii],dx)/self.radius;vy=np.einsum('nij,nj->ni',self.params[ii],dy)/self.radius
            avg=np.sum(w[:,None]*value,axis=0)/w.sum()
            gx=np.sum(w[:,None]*vx+dw[:,0,None]*(value-avg),axis=0)/w.sum();gy=np.sum(w[:,None]*vy+dw[:,1,None]*(value-avg),axis=0)/w.sum()
            out.append(avg);gr.append(np.column_stack([gx,gy]))
        return np.array(out),np.array(gr)
