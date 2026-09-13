"""Original fit_local with exact reusable image-gradient and mask arrays.

Changes are limited to optional arguments, identical target-gradient lookup,
and caching the unchanged floating-point target validity mask. Passing caches
avoids allocating full-image arrays repeatedly for each local fit/iteration.
Original function SHA256: 2e6df2118811c8de1c41919c5bfba7f22dd0002d565cde71c3e9f28b5bc1d206
"""
import numpy as np
from scipy.ndimage import map_coordinates

def fit_local(A,B,va,vb,center,seed,r=19,order=1,reg=.0001,maxiter=35,seed_affine=None,stride=1,target_gradient=None,target_mask_float=None):
    target_mask_float=vb.astype(float) if target_mask_float is None else target_mask_float
    x,y=center;h,w=A.shape
    xi=np.arange(max(0,int(np.floor(x-r))),min(w,int(np.ceil(x+r+1))),stride)
    yi=np.arange(max(0,int(np.floor(y-r))),min(h,int(np.ceil(y+r+1))),stride)
    xx,yy=np.meshgrid(xi,yi);xx=xx.ravel();yy=yy.ravel()
    ok=va[yy,xx];xx=xx[ok];yy=yy[ok]
    qx=(xx-x)/r;qy=(yy-y)/r
    Q=np.array([np.ones(len(xx)),qx,qy]+([.5*qx*qx,qx*qy,.5*qy*qy] if order==2 else [])).T
    k=Q.shape[1];p=np.zeros((2,k));p[:,0]=seed
    if seed_affine is not None:p[:,:3]=seed_affine[:,:3]
    a=A[yy,xx];weights=np.exp(-(qx*qx+qy*qy)/(2*.65**2))
    by,bx=np.gradient(B) if target_gradient is None else target_gradient
    lam=np.array([0]+[reg]*2+([reg*4]*3 if order==2 else []))*np.sum(weights)
    def sample(p):
        dest=np.column_stack([xx,yy])+Q.dot(p.T)
        coordinates=[dest[:,1],dest[:,0]]
        b=map_coordinates(B,coordinates,order=1,mode='nearest')
        valid=(map_coordinates(target_mask_float,coordinates,order=1,mode='constant',cval=0)>.99)
        valid&=(dest[:,0]>=1)&(dest[:,0]<w-2)&(dest[:,1]>=1)&(dest[:,1]<h-2)
        return b,valid,coordinates
    old=1e20;iterations=0
    for it in range(maxiter):
        b,valid,coords=sample(p);basew=weights*valid
        if basew.sum()<25:return np.full((2,k),np.nan),{'ncc':-1,'rms':100,'nvalid':0,'cond':1e20}
        M=np.column_stack([b,np.ones(len(b))])
        gb=np.linalg.lstsq(M*basew[:,None]**.5,a*basew**.5,rcond=None)[0]
        gb[0]=np.clip(gb[0],.3,3)
        gb[1]=np.sum(basew*(a-gb[0]*b))/basew.sum()
        resid=gb[0]*b+gb[1]-a
        # Charbonnier/Huber reweighting, moderate robustness to unmatched particles
        robust=1/np.sqrt(1+(resid/.8)**2)
        wt=basew*robust
        jx=map_coordinates(bx,coords,order=1,mode='nearest')*gb[0]
        jy=map_coordinates(by,coords,order=1,mode='nearest')*gb[0]
        J=np.column_stack([Q*jx[:,None],Q*jy[:,None]])
        # Intensity gain/offset eliminated approximately at each Gauss-Newton step
        WJ=J*wt[:,None];H=J.T.dot(WJ)+np.diag(np.tile(lam,2)+1e-4)
        grad=J.T.dot(wt*resid)+np.tile(lam,2)*p.ravel()
        try: step=np.linalg.solve(H,-grad).reshape(2,k)
        except np.linalg.LinAlgError:break
        movement=np.max(np.linalg.norm(Q.dot(step.T),axis=1))
        if movement>2.5:step*=2.5/movement
        objective=np.sum(basew*.8**2*(np.sqrt(1+(resid/.8)**2)-1))+.5*np.sum(lam*p*p)
        accepted=False
        for fac in [1,.5,.25,.1]:
            pp=p+fac*step
            bb,vv,_=sample(pp)
            # no objective reward for losing valid pixels
            common=valid&vv
            rr=gb[0]*bb+gb[1]-a
            o=np.sum(weights[common]*.8**2*(np.sqrt(1+(rr[common]/.8)**2)-1))+.5*np.sum(lam*pp*pp)
            oo=np.sum(weights[common]*.8**2*(np.sqrt(1+(resid[common]/.8)**2)-1))+.5*np.sum(lam*p*p)
            if o<=oo+1e-8 and common.sum()>=.95*valid.sum():p=pp;accepted=True;break
        iterations=it+1
        if not accepted or np.max(np.abs(fac*step))<.007:break
    b,valid,_=sample(p);wt=weights*valid
    M=np.column_stack([b,np.ones(len(b))])
    gb=np.linalg.lstsq(M*wt[:,None]**.5,a*wt**.5,rcond=None)[0]
    gb[0]=np.clip(gb[0],.3,3);gb[1]=np.sum(wt*(a-gb[0]*b))/wt.sum()
    am=np.sum(wt*a)/wt.sum();bm=np.sum(wt*b)/wt.sum()
    ncc=np.sum(wt*(a-am)*(b-bm))/np.sqrt(np.sum(wt*(a-am)**2)*np.sum(wt*(b-bm)**2)+1e-12)
    rms=np.sqrt(np.sum(wt*(gb[0]*b+gb[1]-a)**2)/wt.sum())
    cond=float(np.linalg.cond(H)) if 'H' in locals() else 1e20
    ux=np.full(len(qx),p[0,1]/r);uy=np.full(len(qx),p[0,2]/r);vx=np.full(len(qx),p[1,1]/r);vy=np.full(len(qx),p[1,2]/r)
    if order==2:
        ux+=(p[0,3]*qx+p[0,4]*qy)/r;uy+=(p[0,4]*qx+p[0,5]*qy)/r
        vx+=(p[1,3]*qx+p[1,4]*qy)/r;vy+=(p[1,4]*qx+p[1,5]*qy)/r
    mindet=float(np.min((1+ux)*(1+vy)-uy*vx))
    return p,dict(ncc=float(ncc),rms=float(rms),nvalid=int(valid.sum()),cond=cond,iterations=iterations,mindet=mindet,support_fraction=float(wt.sum()/weights.sum()))
