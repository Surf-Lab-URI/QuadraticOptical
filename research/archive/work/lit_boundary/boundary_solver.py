"""Boundary-adaptive local quadratic registration.
Images and geometric surface traces only. Parameter derivatives stay Cartesian.
A surface-following support is an implementation inspired by interface-adaptive
PIV, not an implementation of the published Jia et al algorithm (no synthetic
particles or wall velocity/tangential-flow boundary condition).
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np
from scipy.ndimage import gaussian_filter,map_coordinates

def normalize(raw,valid,sigma=.65):
    v=valid.astype(float);w=gaussian_filter(v,sigma)
    im=gaussian_filter(raw*v,sigma)/np.maximum(w,1e-5)
    mean=gaussian_filter(im*v,7)/np.maximum(gaussian_filter(v,7),1e-5)
    hp=im-mean;rms=np.sqrt(gaussian_filter(hp*hp*v,9)/np.maximum(gaussian_filter(v,9),1e-5)+25)
    return np.clip(hp/rms,-3,4)

def image_inputs(rawA,rawB,sa,sb,margin=10,glare=False):
    yy,xx=np.indices(rawA.shape)
    va=yy>=sa[None]+margin;vb=yy>=sb[None]+margin
    if glare:
        va&=(rawA<250)&(gaussian_filter(rawA,3)<180)
        vb&=(rawB<250)&(gaussian_filter(rawB,3)<180)
    return normalize(rawA,va),normalize(rawB,vb),va,vb

def fit(A,B,va,vb,surf,center,seed,strip=False,projected=False,r=13,reg=.00005,maxiter=35,curvature_prior=False):
    h,w=A.shape;x,y=center;p=np.array(seed,dtype=float).copy()
    diag=dict(ncc=np.nan,rms=np.nan,nvalid=0,cond=np.inf,iterations=0,mindet=np.nan,support_fraction=0.,effective_n=0.,normal_spread=0.,tangent_spread=0.,gain=np.nan,offset=np.nan)
    ext=30 if strip else r
    xi=np.arange(max(0,int(np.floor(x-ext))),min(w,int(np.ceil(x+ext+1))))
    yi=np.arange(max(0,int(np.floor(y-ext))),min(h,int(np.ceil(y+ext+1))))
    xx,yy=np.meshgrid(xi,yi);xx=xx.ravel();yy=yy.ravel();dx=xx-x;dy=yy-y
    if strip:
        slope=float(np.interp(x,np.arange(w),np.gradient(surf)));root=np.sqrt(1+slope*slope)
        tangent=(dx+slope*dy)/root
        normal=((yy-surf[xx])-(y-np.interp(x,np.arange(w),surf)))/root
        inside=(np.abs(tangent)<=19)&(np.abs(normal)<=9)
        ww=np.exp(-.5*((tangent/(19*.65))**2+(normal/(9*.65))**2))
    else:
        tangent=dx;normal=dy;inside=np.ones(len(xx),bool)
        ww=np.exp(-(dx*dx+dy*dy)/(2*(r*.65)**2))
    use=inside&va[yy,xx];xx=xx[use];yy=yy[use];weights=ww[use];tangent=tangent[use];normal=normal[use]
    if len(xx)<25 or weights.sum()<25:return np.full((2,6),np.nan),diag
    qx=(xx-x)/r;qy=(yy-y)/r
    Q=np.c_[np.ones(len(xx)),qx,qy,.5*qx*qx,qx*qy,.5*qy*qy]
    a=A[yy,xx];by,bx=np.gradient(B);lam=np.array([0,reg,reg,4*reg,4*reg,4*reg])*weights.sum()
    prior=np.zeros((12,12));anchor=p.ravel().copy()
    if curvature_prior:
        # Compensate for information lost by shrinking normal support13->9.
        # Gaussian fourth moments determine normal quadratic information;
        # second moments enter the mixed tangent/normal curvature direction.
        slope=float(np.interp(x,np.arange(w),np.gradient(surf)));root=np.sqrt(1+slope*slope)
        nx,ny=-slope/root,1/root;tx,ty=1/root,slope/root
        nn=np.array([nx*nx,2*nx*ny,ny*ny]);tn=np.array([tx*nx,tx*ny+ty*nx,ty*ny])
        curv=4*reg*weights.sum()*(((13/9)**4-1)*np.outer(nn,nn)+((13/9)**2-1)*np.outer(tn,tn))
        prior[3:6,3:6]=curv;prior[9:12,9:12]=curv
    def sample(par):
        dest=np.c_[xx,yy]+Q@par.T;coords=[dest[:,1],dest[:,0]]
        b=map_coordinates(B,coords,order=1,mode='nearest')
        valid=map_coordinates(vb.astype(float),coords,order=1,mode='constant',cval=0)>.99
        valid&=(dest[:,0]>=1)&(dest[:,0]<w-2)&(dest[:,1]>=1)&(dest[:,1]<h-2)
        return b,valid,coords
    def photometry(b,wt,robust=False):
        M=np.c_[b,np.ones(len(b))];usewt=wt.copy();gb=np.array([1.,0.])
        for _ in range(3 if robust else 1):
            gb=np.linalg.lstsq(M*np.sqrt(usewt[:,None]),a*np.sqrt(usewt),rcond=None)[0]
            gb[0]=np.clip(gb[0],.3,3);gb[1]=np.sum(usewt*(a-gb[0]*b))/max(usewt.sum(),1e-12)
            res=M@gb-a;usewt=wt/np.sqrt(1+(res/.8)**2)
        return gb,M@gb-a,M
    H=np.eye(12);it=0
    for it in range(maxiter):
        b,valid,coords=sample(p);basew=weights*valid
        if basew.sum()<25:return np.full((2,6),np.nan),diag
        gb,resid,M=photometry(b,basew,projected);wt=basew/np.sqrt(1+(resid/.8)**2)
        jx=map_coordinates(bx,coords,order=1,mode='nearest')*gb[0];jy=map_coordinates(by,coords,order=1,mode='nearest')*gb[0]
        J=np.c_[Q*jx[:,None],Q*jy[:,None]]
        if projected:
            nuisance=np.linalg.solve(M.T@(wt[:,None]*M)+np.eye(2)*1e-10,M.T@(wt[:,None]*J))
            J=J-M@nuisance
        H=J.T@(wt[:,None]*J)+np.diag(np.tile(lam,2)+1e-4)+prior
        grad=J.T@(wt*resid)+np.tile(lam,2)*p.ravel()+prior@(p.ravel()-anchor)
        try:step=np.linalg.solve(H,-grad).reshape(2,6)
        except np.linalg.LinAlgError:break
        movement=np.max(np.linalg.norm(Q@step.T,axis=1))
        if movement>2.5:step*=2.5/movement
        took=False
        for fac in [1.,.5,.25,.1]:
            pp=p+fac*step;bb,vv,_=sample(pp);common=valid&vv
            if common.sum()<.95*valid.sum():continue
            if projected:
                _,rr,_=photometry(bb,weights*common,True);_,r0,_=photometry(b,weights*common,True)
            else:rr=gb[0]*bb+gb[1]-a;r0=resid
            def loss(res,par):return np.sum(weights[common]*.8**2*(np.sqrt(1+(res[common]/.8)**2)-1))+.5*np.sum(lam*par*par)+.5*(par.ravel()-anchor)@prior@(par.ravel()-anchor)
            if loss(rr,pp)<=loss(r0,p)+1e-8:p=pp;took=True;break
        if not took or np.max(np.abs(fac*step))<.007:break
    b,valid,_=sample(p);wt=weights*valid
    if wt.sum()<25:return np.full((2,6),np.nan),diag
    gb,resid,_=photometry(b,wt,projected);am=np.sum(wt*a)/wt.sum();bm=np.sum(wt*b)/wt.sum()
    ncc=np.sum(wt*(a-am)*(b-bm))/np.sqrt(np.sum(wt*(a-am)**2)*np.sum(wt*(b-bm)**2)+1e-12)
    ux=(p[0,1]+p[0,3]*qx+p[0,4]*qy)/r;uy=(p[0,2]+p[0,4]*qx+p[0,5]*qy)/r
    vx=(p[1,1]+p[1,3]*qx+p[1,4]*qy)/r;vy=(p[1,2]+p[1,4]*qx+p[1,5]*qy)/r
    mean=lambda v:np.dot(wt,v)/wt.sum()
    diag=dict(ncc=float(ncc),rms=float(np.sqrt(mean(resid**2))),nvalid=int(valid.sum()),cond=float(np.linalg.cond(H)),iterations=it+1,mindet=float(np.min((1+ux)*(1+vy)-uy*vx)),support_fraction=float(wt.sum()/weights.sum()),effective_n=float(wt.sum()**2/np.sum(wt**2)),normal_spread=float(np.sqrt(mean((normal-mean(normal))**2))),tangent_spread=float(np.sqrt(mean((tangent-mean(tangent))**2))),gain=float(gb[0]),offset=float(gb[1]))
    return p,diag
