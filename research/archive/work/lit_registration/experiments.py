"""Image-only geometric/interpolation ablations; original models remain untouched.

Symmetric variants use a quadratic displacement in midpoint coordinates, sample
A at m-d(m)/2 and B at m+d(m)/2, then convert the fitted local 2-jet back to the
same source-A centers before the established compact blending/evaluation.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import sys,json,time
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter,map_coordinates
from scipy.interpolate import RectBivariateSpline
from concurrent.futures import ThreadPoolExecutor
HERE=Path(__file__).resolve().parent;WORK=HERE.parent;sys.path.insert(0,str(WORK))
from field_model import LocalField

class Sampler:
 def __init__(self,im,cubic):
  self.im=im;self.cubic=cubic
  if cubic:self.spline=RectBivariateSpline(np.arange(im.shape[0]),np.arange(im.shape[1]),im,kx=3,ky=3,s=0)
  else:self.gy,self.gx=np.gradient(im)
 def value(self,xy):
  if self.cubic:return self.spline.ev(xy[:,1],xy[:,0])
  return map_coordinates(self.im,[xy[:,1],xy[:,0]],order=1,mode='nearest')
 def gradient(self,xy):
  if self.cubic:return np.c_[self.spline.ev(xy[:,1],xy[:,0],dx=0,dy=1),self.spline.ev(xy[:,1],xy[:,0],dx=1,dy=0)]
  return np.c_[map_coordinates(self.gx,[xy[:,1],xy[:,0]],order=1,mode='nearest'),map_coordinates(self.gy,[xy[:,1],xy[:,0]],order=1,mode='nearest')]

def hessian(p,r):
 h=np.empty((2,2,2));h[:,0,0]=p[:,3]/r**2;h[:,0,1]=h[:,1,0]=p[:,4]/r**2;h[:,1,1]=p[:,5]/r**2
 return h

def package_jet(t,G,H,r):
 return np.c_[t,G*r,np.stack([H[:,0,0],H[:,0,1],H[:,1,1]],axis=1)*r*r]

def forward_to_mid(p,r):
 G=p[:,1:3]/r;H=hessian(p,r);C=np.linalg.inv(np.eye(2)+.5*G)
 return package_jet(p[:,0],G@C,np.einsum('ia,ajk,jb,kc->ibc',C,H,C,C),r)

def basis(z,r):
 x=z[:,0]/r;y=z[:,1]/r
 return np.c_[np.ones(len(z)),x,y,.5*x*x,x*y,.5*y*y]

def local_jac(p,z,r):
 H=hessian(p,r)
 return p[:,1:3][None]/r+np.einsum('ijk,nk->nij',H,z)

def mid_to_forward(p,r,source_center,mid_center):
 m=source_center+p[:,0]/2
 for _ in range(20):
  z=m-mid_center;Q=basis(z[None],r);v=(Q@p.T)[0];G=local_jac(p,z[None],r)[0]
  delta=np.linalg.solve(np.eye(2)-.5*G,m-.5*v-source_center);m-=delta
  if np.max(abs(delta))<1e-10:break
 z=m-mid_center;v=(basis(z[None],r)@p.T)[0];G=local_jac(p,z[None],r)[0];C=np.linalg.inv(np.eye(2)-.5*G)
 pp=package_jet(v,G@C,np.einsum('ia,ajk,jb,kc->ibc',C,hessian(p,r),C,C),r)
 return pp,float(np.linalg.norm(m-.5*v-source_center))

def solve(A,B,va,vb,center,p0,r=13,symmetric=False,cubic=False,reg=.00005,maxiter=40,samplers=None,project_photometric=False):
 h,w=A.shape;S,T=samplers if samplers else (Sampler(A,cubic),Sampler(B,cubic))
 center=np.asarray(center,dtype=float);mc=center+p0[:,0]/2 if symmetric else center
 try:p=forward_to_mid(p0,r) if symmetric else p0.copy()
 except np.linalg.LinAlgError:return p0.copy(),dict(ncc=-1,failed=True)
 gy,gx=np.mgrid[-r:r+1,-r:r+1];off=np.c_[gx.ravel(),gy.ravel()];coords=mc+off;Q=basis(off,r)
 weights=np.exp(-np.sum((off/r)**2,axis=1)/(2*.65**2))
 if not symmetric:
  valid0=map_coordinates(va.astype(float),[coords[:,1],coords[:,0]],order=1,mode='constant')>.99
  coords=coords[valid0];Q=Q[valid0];weights=weights[valid0];off=off[valid0]
 lam=np.array([0,reg,reg,4*reg,4*reg,4*reg])*weights.sum()
 def mask(v,xy):
  return (map_coordinates(v.astype(float),[xy[:,1],xy[:,0]],order=1,mode='constant',cval=0)>.99)&(xy[:,0]>=1)&(xy[:,0]<w-2)&(xy[:,1]>=1)&(xy[:,1]<h-2)
 def sample(p):
  flow=Q@p.T;aa=coords-.5*flow if symmetric else coords;bb=coords+.5*flow if symmetric else coords+flow
  return S.value(aa),T.value(bb),mask(va,aa)&mask(vb,bb),aa,bb
 def photometric(a,b,wt):
  M=np.c_[b,np.ones(len(b))];coef=np.linalg.lstsq(M*np.sqrt(wt)[:,None],a*np.sqrt(wt),rcond=None)[0]
  coef[0]=np.clip(coef[0],.3,3);coef[1]=np.sum(wt*(a-coef[0]*b))/wt.sum()
  if project_photometric:
   for _ in range(6):
    residual=coef[0]*b+coef[1]-a;rw=wt/np.sqrt(1+(residual/.8)**2)
    newcoef=np.linalg.lstsq(M*np.sqrt(rw)[:,None],a*np.sqrt(rw),rcond=None)[0]
    newcoef[0]=np.clip(newcoef[0],.3,3);newcoef[1]=np.sum(rw*(a-newcoef[0]*b))/rw.sum()
    if np.max(abs(newcoef-coef))<1e-7:coef=newcoef;break
    coef=newcoef
  return coef
 def rho(e):return .8**2*(np.sqrt(1+(e/.8)**2)-1)
 failed=False;accepted_steps=0
 for it in range(maxiter):
  a,b,valid,ac,bc=sample(p);wt=weights*valid
  if wt.sum()<25:failed=True;break
  gb=photometric(a,b,wt);res=gb[0]*b+gb[1]-a;rw=wt/np.sqrt(1+(res/.8)**2)
  grad=T.gradient(bc)*gb[0]
  if symmetric:grad=.5*(grad+S.gradient(ac))
  J=np.c_[Q*grad[:,0,None],Q*grad[:,1,None]]
  if project_photometric:
   photo=np.c_[b,np.ones(len(b))];Hphoto=photo.T@(rw[:,None]*photo)+np.eye(2)*1e-9
   J=J-photo@np.linalg.solve(Hphoto,photo.T@(rw[:,None]*J))
  H=J.T@(rw[:,None]*J)+np.diag(np.tile(lam,2)+1e-4)
  rhs=J.T@(rw*res)+np.tile(lam,2)*p.ravel()
  try:step=np.linalg.solve(H,-rhs).reshape(2,6)
  except np.linalg.LinAlgError:failed=True;break
  move=np.max(np.linalg.norm(Q@step.T,axis=1))
  if move>2.5:step*=2.5/move
  accepted=False
  for fac in [1,.5,.25,.1]:
   cand=p+fac*step;aa,bb,vv,_,_=sample(cand);common=valid&vv
   if common.sum()<.95*valid.sum():continue
   newgb=photometric(aa,bb,weights*common) if project_photometric else gb
   newres=newgb[0]*bb+newgb[1]-aa
   new=np.sum(weights[common]*rho(newres[common]))+.5*np.sum(lam*cand*cand)
   old=np.sum(weights[common]*rho(res[common]))+.5*np.sum(lam*p*p)
   if new<=old+1e-8:p=cand;accepted=True;accepted_steps+=1;break
  if not accepted or np.max(abs(fac*step))<.007:break
 a,b,valid,_,_=sample(p);wt=weights*valid
 if wt.sum()<25:return p0.copy(),dict(ncc=-1,failed=True,accepted_steps=accepted_steps)
 gb=photometric(a,b,wt);am=np.sum(wt*a)/wt.sum();bm=np.sum(wt*b)/wt.sum()
 ncc=np.sum(wt*(a-am)*(b-bm))/np.sqrt(np.sum(wt*(a-am)**2)*np.sum(wt*(b-bm)**2)+1e-12)
 try:
  pnew,inverr=mid_to_forward(p,r,center,mc) if symmetric else (p,0.)
  failed|=not np.isfinite(pnew).all() or inverr>1e-6
 except np.linalg.LinAlgError:failed=True;pnew=p0.copy();inverr=np.inf
 if failed:pnew=p0.copy()
 return pnew,dict(ncc=float(ncc),failed=bool(failed),rms=float(np.sqrt(np.sum(wt*(gb[0]*b+gb[1]-a)**2)/wt.sum())),nvalid=int(valid.sum()),support_fraction=float(wt.sum()/weights.sum()),accepted_steps=accepted_steps,iterations=it+1,inverse_residual=inverr)

def inputs():
 m=np.load(WORK/'metadata.npz');sa=m['surfa'].ravel()-1;sb=m['surfb'].ravel()-1
 base='data/ExpLCL_1_03-123'
 A=np.asarray(Image.open(base+'_imgA.tif'),float);B=np.asarray(Image.open(base+'_imgB.tif'),float);y,x=np.indices(A.shape)
 va=y>=sa[None]+10;vb=y>=sb[None]+10
 def norm(raw,v):
  vv=v.astype(float);im=gaussian_filter(raw*vv,.65)/np.maximum(gaussian_filter(vv,.65),1e-5)
  mean=gaussian_filter(im*vv,7)/np.maximum(gaussian_filter(vv,7),1e-5);hp=im-mean
  rms=np.sqrt(gaussian_filter(hp*hp*vv,9)/np.maximum(gaussian_filter(vv,9),1e-5)+25)
  return np.clip(hp/rms,-3,4)
 return norm(A,va),norm(B,vb),va,vb

def evaluate(path):
 m=np.load(WORK/'metadata.npz');manual=m['p'][:,0].T-1;truth=m['p'][:,1].T-m['p'][:,0].T
 d,g=LocalField(path).evaluate(manual);e=np.linalg.norm(d-truth,axis=1);dep=manual[:,1]-np.interp(manual[:,0],np.arange(501),m['surfa'].ravel()-1)
 out={}
 for name,sel in [('all',np.ones(200,bool)),('near40',dep<40),('depth40_80',(dep>=40)&(dep<80)),('depth80plus',dep>=80)]:
  ee=e[sel];out[name]=dict(n=int(sel.sum()),nfinite=int(np.isfinite(ee).sum()),median=float(np.nanmedian(ee)),mean=float(np.nanmean(ee)),rms=float(np.sqrt(np.nanmean(ee**2))),p90=float(np.nanpercentile(ee,90)))
 np.savez(HERE/(Path(path).stem+'_evaluation.npz'),points=manual,disp=d,gradient=g,error=e,depth=dep)
 return out

def run():
 f=np.load(WORK/'final_ptv13.npz');pts=f['points'];p0=f['params'];A,B,va,vb=inputs();metrics={'baseline':evaluate(WORK/'final_ptv13.npz')}
 for name,symmetric,cubic in [('forward_refit',False,False),('forward_cubic',False,True),('symmetric_bilinear',True,False),('symmetric_cubic',True,True)]:
  t=time.time();samplers=(Sampler(A,cubic),Sampler(B,cubic))
  def task(i):return solve(A,B,va,vb,pts[i],p0[i],r=13,symmetric=symmetric,cubic=cubic,samplers=samplers)
  with ThreadPoolExecutor(max_workers=4) as ex:results=list(ex.map(task,range(len(pts))))
  p=np.array([v[0] for v in results]);stats={k:np.array([v[1].get(k,np.nan) for v in results]) for k in set().union(*(v[1] for v in results))}
  path=HERE/(name+'.npz');np.savez(path,points=pts,params=p,radius=13,**stats)
  metrics[name]=evaluate(path);metrics[name]['diagnostics']=dict(seconds=time.time()-t,failed_fallbacks=int(stats['failed'].sum()),median_ncc=float(np.median(stats['ncc'])),max_parameter_change=float(np.max(abs(p-p0))))
  print(name,json.dumps(metrics[name]),flush=True)
 (HERE/'real_metrics.json').write_text(json.dumps(metrics,indent=2))

if __name__=='__main__':run()
