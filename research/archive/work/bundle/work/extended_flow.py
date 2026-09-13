"""Masked local affine/quadratic image registration, images only.
Manual vectors are used only by the separate evaluation routine.
Coordinates are zero-based pixel centers, y positive downward.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates, median_filter
from scipy.optimize import least_squares
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree
from concurrent.futures import ThreadPoolExecutor
import json,time
BASE='inputs/ExpLCL_1_03-123'

def load_images(margin=10, sigma=1.0):
    m=np.load('work/metadata.npz')
    A=np.asarray(Image.open(BASE+'_imgA.tif'),dtype=float)
    B=np.asarray(Image.open(BASE+'_imgB.tif'),dtype=float)
    # surface arrays align directly with zero-based crop, per p_orig-p and crop bounds
    sa=m['surfa'].ravel()-1;sb=m['surfb'].ravel()-1
    y,x=np.indices(A.shape)
    va=y>=sa[None,:]+margin;vb=y>=sb[None,:]+margin
    def norm(im):
        # Global intensity normalization; blur removes one-pixel sensor noise.
        smooth=gaussian_filter(im,sigma)
        high=smooth-gaussian_filter(im,7)
        rms=np.sqrt(gaussian_filter(high**2,9)+25)
        return np.clip(high/rms,-3,4)
    return norm(A),norm(B),va,vb,sa,sb,A,B

def translation_seed(A,B,va,vb,center,r=18,search=(55,20)):
    x,y=np.round(center).astype(int);h,w=A.shape
    x0=max(0,x-r);x1=min(w,x+r+1);y0=max(0,y-r);y1=min(h,y+r+1)
    T=A[y0:y1,x0:x1];V=va[y0:y1,x0:x1].astype(float)
    # dX [-15,+55], dY [-20,+20] established broadly, not from manual velocities
    dx0=max(-15,-x0);dx1=min(search[0],w-x1)
    dy0=max(-search[1],-y0);dy1=min(search[1],h-y1)
    S=B[y0+dy0:y1+dy1,x0+dx0:x1+dx1];W=vb[y0+dy0:y1+dy1,x0+dx0:x1+dx1].astype(float)
    def corr(a,b):return fftconvolve(a,b[::-1,::-1],mode='valid')
    n=corr(W,V); at=corr(W,V*T); bs=corr(W*S,V)
    aa=corr(W,V*T*T)-at*at/np.maximum(n,1)
    bb=corr(W*S*S,V)-bs*bs/np.maximum(n,1)
    cc=corr(W*S,V*T)-at*bs/np.maximum(n,1)
    C=cc/np.sqrt(np.maximum(aa*bb,1e-12));C[n<max(40,.65*V.sum())]=-2
    peaks=[];C1=C.copy()
    for _ in range(3):
        iy,ix=np.unravel_index(np.argmax(C1),C.shape)
        peaks.append((np.array([dx0+ix,dy0+iy],float),float(C[iy,ix])))
        C1[max(0,iy-3):iy+4,max(0,ix-3):ix+4]=-2
    return peaks

def fit_local(A,B,va,vb,center,seed,r=19,order=1,reg=.0001,maxiter=35,seed_affine=None,stride=1):
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
    by,bx=np.gradient(B)
    lam=np.array([0]+[reg]*2+([reg*4]*3 if order==2 else []))*np.sum(weights)
    def sample(p):
        dest=np.column_stack([xx,yy])+Q.dot(p.T)
        coordinates=[dest[:,1],dest[:,0]]
        b=map_coordinates(B,coordinates,order=1,mode='nearest')
        valid=(map_coordinates(vb.astype(float),coordinates,order=1,mode='constant',cval=0)>.99)
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

def evaluate(points,disp):
    m=np.load('work/metadata.npz');p=m['p'];a=p[:,0].T-1;b=p[:,1].T-1
    # Evaluate predictions via linear interpolation; manual not used to form fields.
    from scipy.interpolate import LinearNDInterpolator
    pred=LinearNDInterpolator(points,disp)(a)
    err=np.linalg.norm(pred-(b-a),axis=1)
    depth=a[:,1]-np.interp(a[:,0],np.arange(501),m['surfa'].ravel()-1)
    out={}
    for name,sel in [('all',np.ones(200,bool)),('depth<40',depth<40),('depth40-80',(depth>=40)&(depth<80)),('depth80+',depth>=80)]:
        e=err[sel];e=e[np.isfinite(e)]
        out[name]=dict(n=len(e),median=float(np.median(e)),mean=float(np.mean(e)),rmse=float(np.sqrt(np.mean(e**2))),p90=float(np.percentile(e,90)))
    return out,pred,err

def run(name='affine',radius=19,order=1,reg=.0001,seedfile=None,margin=10,step=8):
    A,B,va,vb,sa,sb,rawA,rawB=load_images(margin)
    points=np.array([(x,y) for y in range(24,177,step) for x in range(60,385,step) if y>=sa[x]+12],float)
    sf=np.load(seedfile) if seedfile else None
    t=time.time()
    def process(pair):
        i,point=pair
        if sf is not None:
            if 'flow' in sf:
                x,y=np.round(point).astype(int);seed=sf['flow'][y,x];sa0=None
            else:
                idx=np.argmin(np.sum((sf['points']-point)**2,axis=1));seed=sf['params'][idx,:,0];sa0=sf['params'][idx,:,:3].copy();sa0[:,1:]*=radius/float(sf['radius'])
            candidates=[(seed,0)]
        else:candidates=translation_seed(A,B,va,vb,point,r=radius);sa0=None
        candidates=candidates[:1] if candidates[0][1]>.55 else candidates[:3]
        fits=[fit_local(A,B,va,vb,point,z[0],r=radius,order=order,reg=reg,seed_affine=sa0) for z in candidates]
        scores=[q[1]['ncc'] for q in fits];best=fits[np.argmax(scores)]
        if i%100==0:print(name,i,len(points),'elapsed',round(time.time()-t,1),flush=True)
        return best
    with ThreadPoolExecutor(max_workers=4) as ex:res=list(ex.map(process,enumerate(points)))
    params=np.array([r[0] for r in res]);stats={key:np.array([r[1].get(key,np.nan) for r in res]) for key in res[0][1]}
    np.savez('work/'+name+'.npz',points=points,params=params,radius=radius,margin=margin,order=order,reg=reg,**stats)
    metrics,pred,err=evaluate(points,params[:,:,0]);print(name,json.dumps(metrics),flush=True)
    with open('work/'+name+'_metrics.json','w') as f:json.dump(metrics,f,indent=2)
    return metrics
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--name',default='affine');parser.add_argument('--radius',type=int,default=19);parser.add_argument('--order',type=int,default=1);parser.add_argument('--reg',type=float,default=.0001);parser.add_argument('--seedfile');parser.add_argument('--margin',type=int,default=10);parser.add_argument('--step',type=int,default=8)
    args=parser.parse_args();run(**vars(args))
