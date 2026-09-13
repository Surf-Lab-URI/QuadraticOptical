"""Fixed translation-only normalized cross-correlation baseline.
A->B displacement, zero-based pixel centers; surface and p converted by minus1.
No manual vectors are used to seed, tune or estimate the field.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import numpy as np,json,time
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve
from scipy.interpolate import LinearNDInterpolator
from concurrent.futures import ThreadPoolExecutor
W=Path('work');D=Path('historical-user-files/Downloads');z=np.load(W/'metadata.npz')
a=np.asarray(Image.open(D/'ExpLCL_1_03-123_imgA.tif'),dtype=float);b=np.asarray(Image.open(D/'ExpLCL_1_03-123_imgB.tif'),dtype=float)
sa=z['surfa'].ravel()-1;sb=z['surfb'].ravel()-1;y,x=np.indices(a.shape);va=y>=sa[None,:]+10;vb=y>=sb[None,:]+10
# Same modest particle denoising and local contrast normalization as affine solver.
def norm(I):
 smooth=gaussian_filter(I,1);high=smooth-gaussian_filter(I,7);rms=np.sqrt(gaussian_filter(high**2,9)+25);return np.clip(high/rms,-3,4)
A=norm(a);B=norm(b);points=np.array([(x,y) for y in range(24,177,8) for x in range(60,385,8) if y>=sa[x]+12],float)
def get_translation(center,r=19):
 x,y=np.round(center).astype(int);h,w=A.shape;x0=max(0,x-r);x1=min(w,x+r+1);y0=max(0,y-r);y1=min(h,y+r+1)
 T=A[y0:y1,x0:x1];V=va[y0:y1,x0:x1].astype(float);dx0=max(-15,-x0);dx1=min(55,w-x1);dy0=max(-20,-y0);dy1=min(20,h-y1)
 S=B[y0+dy0:y1+dy1,x0+dx0:x1+dx1];M=vb[y0+dy0:y1+dy1,x0+dx0:x1+dx1].astype(float)
 corr=lambda q,t:fftconvolve(q,t[::-1,::-1],mode='valid')
 n=corr(M,V);at=corr(M,V*T);bs=corr(M*S,V);aa=corr(M,V*T*T)-at*at/np.maximum(n,1);bb=corr(M*S*S,V)-bs*bs/np.maximum(n,1);cc=corr(M*S,V*T)-at*bs/np.maximum(n,1)
 C=cc/np.sqrt(np.maximum(aa*bb,1e-12));C[n<max(40,.65*V.sum())]=-2;iy,ix=np.unravel_index(np.argmax(C),C.shape);peak=float(C[iy,ix]);integer=np.array([ix+dx0,iy+dy0],float);disp=integer.copy()
 # Independent parabolic subpixel interpolation in x and y around largest peak.
 for axis in [0,1]:
  k=ix if axis==0 else iy;bound=C.shape[1] if axis==0 else C.shape[0]
  if k<=0 or k>=bound-1:continue
  left=C[iy,ix-1] if axis==0 else C[iy-1,ix];right=C[iy,ix+1] if axis==0 else C[iy+1,ix]
  denom=left-2*peak+right
  if denom< -1e-8:disp[axis]+=np.clip(.5*(left-right)/denom,-.75,.75)
 C[max(0,iy-3):iy+4,max(0,ix-3):ix+4]=-2;second=float(np.max(C));return disp,integer,peak,second,float(n[iy,ix])
t=time.time()
with ThreadPoolExecutor(max_workers=4) as pool:res=list(pool.map(get_translation,points))
disp=np.array([v[0] for v in res]);integer=np.array([v[1] for v in res]);ncc=np.array([v[2] for v in res]);second=np.array([v[3] for v in res]);nvalid=np.array([v[4] for v in res]);xy=z['p'][:,0,:].T-1;truth=(z['p'][:,1,:]-z['p'][:,0,:]).T;depth=xy[:,1]-np.interp(xy[:,0],np.arange(501),sa)
def evaluate(flow):
 pred=LinearNDInterpolator(points,flow)(xy);e=np.linalg.norm(pred-truth,axis=1);metrics={}
 for name,mask in [('all',np.ones(200,bool)),('depth<40',depth<40),('depth40-80',(depth>=40)&(depth<80)),('depth80+',depth>=80)]:
  ee=e[mask&np.isfinite(e)];metrics[name]={'n':len(ee),'median':float(np.median(ee)),'mean':float(np.mean(ee)),'rmse':float(np.sqrt(np.mean(ee*ee))),'p90':float(np.percentile(ee,90)),'under1':float(np.mean(ee<1)),'under2':float(np.mean(ee<2))}
 return metrics,pred,e
metrics,pred,e=evaluate(disp);intmetrics,_,_=evaluate(integer)
np.savez_compressed(W/'baseline_ncc.npz',points=points,disp=disp,params=disp[:,:,None],integer=integer,ncc=ncc,second_peak=second,nvalid=nvalid,radius=19,margin=10,order=0,pred=pred,truth=truth,error=e,xy=xy,depth=depth)
(W/'baseline_ncc_metrics.json').write_text(json.dumps({'subpixel':metrics,'integer':intmetrics,'elapsed_s':time.time()-t,'grid_points':len(points)},indent=2));print('NCC done',len(points),time.time()-t,json.dumps(metrics),flush=True)
