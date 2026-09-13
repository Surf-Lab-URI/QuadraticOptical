import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from extended_flow import *
import numpy as np,time
from scipy.ndimage import gaussian_filter
f=np.load('work/final_ptv13.npz');r=float(f['radius']);pts=f['points'];par=f['params'];A,B,va,vb,sa,sb,rawA,rawB=load_images(10,.65)
def norm(raw,v):
 w=gaussian_filter(v.astype(float),.65);im=gaussian_filter(raw*v,.65)/np.maximum(w,1e-5)
 vv=v.astype(float);mean=gaussian_filter(im*vv,7)/np.maximum(gaussian_filter(vv,7),1e-5)
 hp=im-mean;rms=np.sqrt(gaussian_filter(hp*hp*vv,9)/np.maximum(gaussian_filter(vv,9),1e-5)+25)
 return np.clip(hp/rms,-3,4)
A=norm(rawA,va);B=norm(rawB,vb)
def task(i):
 p=par[i];dest=pts[i]+p[:,0];q=np.zeros((2,3));q[:,0]=-p[:,0]
 try:q[:,1:]=(np.linalg.inv(np.eye(2)+p[:,1:3]/r)-np.eye(2))*r
 except np.linalg.LinAlgError:return np.full((2,6),np.nan),dict(ncc=-1)
 return fit_local(B,A,vb,va,dest,q[:,0],r=int(r),order=2,reg=.00005,seed_affine=q)
t=time.time()
with ThreadPoolExecutor(max_workers=4) as ex:res=list(ex.map(task,range(len(pts))))
p=np.array([z[0] for z in res]);st={k:np.array([z[1].get(k,np.nan) for z in res]) for k in res[0][1]}
fb=np.linalg.norm(p[:,:,0]+par[:,:,0],axis=1)
np.savez('work/final_ptv13_reverse.npz',points=pts,params=p,fb_error=fb,**st)
print('reverse seconds',time.time()-t,'fb percentiles',np.percentile(fb,[0,50,90,95,100]),flush=True)
