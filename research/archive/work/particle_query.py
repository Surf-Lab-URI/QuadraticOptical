import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np,json,time
from extended_flow import *
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
from scipy.optimize import minimize
m=np.load('work/metadata.npz');pts=m['p'][:,0].T-1;f=np.load('work/propagate0.npz');flow=np.load('work/seed_dis.npz')['flow'];conf=np.load('work/baseline_confidence_matlab.npz');print(conf.files,flush=True)
A,B,va,vb,sa,sb,rawA,rawB=load_images(10,.5)
# Tiny patches use lightly smoothed raw intensities with local gain/bias normalization.
A=gaussian_filter(rawA,.5);B=gaussian_filter(rawB,.5)
good=(f['ncc']>.7);ip=f['points'][good];pp=f['params'][good];tree=cKDTree(ip)

def prior(pt):
 dist,idx=tree.query(pt,k=12);par=pp[idx];pred=par[:,:,0]+np.einsum('nij,nj->ni',par[:,:,1:]/19,pt-ip[idx])
 # Robust vector median of image-only affine extrapolations.
 dd=np.linalg.norm(pred[:,None]-pred[None,:],axis=2);weights=np.exp(-dist**2/(2*25**2));cost=dd.dot(weights);k=np.argmin(cost)
 sel=np.linalg.norm(pred-pred[k],axis=1)<4
 return np.average(pred[sel],axis=0,weights=weights[sel]),np.std(pred[sel],axis=0)

def ncc(pt,disp,r=4):
 y,x=np.mgrid[-r:r+1,-r:r+1];x=x.ravel()+pt[0];y=y.ravel()+pt[1]
 aa=map_coordinates(A,[y,x],order=1);bb=map_coordinates(B,[y+disp[1],x+disp[0]],order=1)
 valid=map_coordinates(va.astype(float),[y,x],order=1)>.99;valid&=map_coordinates(vb.astype(float),[y+disp[1],x+disp[0]],order=1)>.99
 if valid.sum()<.85*len(valid):return -1
 aa=aa[valid];bb=bb[valid];aa-=aa.mean();bb-=bb.mean()
 return float(aa.dot(bb)/np.sqrt(aa.dot(aa)*bb.dot(bb)+1e-12))

def task(pt):
 pr,spread=prior(pt)
 # Search all integer displacements within a 9px radius of image-only prior.
 cand=[]
 for dy in range(-8,9):
  for dx in range(-8,9):
   d=pr+np.array([dx,dy]);c=ncc(pt,d);score=c-.005*(dx*dx+dy*dy)
   cand.append((score,d,c))
 cand.sort(key=lambda z:-z[0]);seeds=[]
 for c in cand:
  if all(np.linalg.norm(c[1]-s[1])>2 for s in seeds):seeds.append(c)
  if len(seeds)==3:break
 fits=[]
 for score,d,c in seeds:
  opt=minimize(lambda z:1-ncc(pt,z)+.005*np.sum((z-pr)**2),d,method='Nelder-Mead',options={'maxiter':100,'xatol':.01})
  fits.append((opt.fun,opt.x,ncc(pt,opt.x)))
 fits.sort(key=lambda z:z[0]);best=fits[0]
 return best[1],pr,spread,best[2],float(fits[1][0]-best[0])
t=time.time()
with ThreadPoolExecutor(max_workers=4) as ex:res=list(ex.map(task,pts))
a=np.array([r[0] for r in res]);d=m['p'][:,1].T-m['p'][:,0].T;e=np.linalg.norm(a-d,axis=1);dep=pts[:,1]-np.interp(pts[:,0],np.arange(501),sa)
np.savez('work/particle_query.npz',points=pts,disp=a,prior=np.array([r[1] for r in res]),spread=np.array([r[2] for r in res]),ncc=np.array([r[3] for r in res]),ambiguity_gap=np.array([r[4] for r in res]))
for name,sel in [('all',np.ones(200,bool)),('near40',dep<40),('deep80',dep>=80)]:print(name,int(sel.sum()),np.median(e[sel]),np.mean(e[sel]),np.percentile(e[sel],90),flush=True)
print('time',time.time()-t,flush=True)
for i in [83,84,93,94,95,141,142]:print(i+1,'prior',res[i][1],'pred',a[i],'true',d[i],'err',e[i],flush=True)
