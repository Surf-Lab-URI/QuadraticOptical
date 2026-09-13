import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np,json,time,sys
from extended_flow import *
from scipy.spatial import cKDTree
import argparse
pa=argparse.ArgumentParser();pa.add_argument('--name');pa.add_argument('--radius',type=int,default=13);pa.add_argument('--order',type=int,default=1);pa.add_argument('--seedfile',default='work/propagate2.npz');pa.add_argument('--margin',type=int,default=10);pa.add_argument('--sigma',type=float,default=.65);pa.add_argument('--reg',type=float,default=.00005);pa.add_argument('--maskednorm',action='store_true');args=pa.parse_args()
A,B,va,vb,sa,sb,rawA,rawB=load_images(args.margin,args.sigma)
if args.maskednorm:
 def norm(raw,v):
  w=gaussian_filter(v.astype(float),args.sigma);im=gaussian_filter(raw*v,args.sigma)/np.maximum(w,1e-5)
  vv=v.astype(float);mean=gaussian_filter(im*vv,7)/np.maximum(gaussian_filter(vv,7),1e-5)
  hp=im-mean;rms=np.sqrt(gaussian_filter(hp*hp*vv,9)/np.maximum(gaussian_filter(vv,9),1e-5)+25)
  return np.clip(hp/rms,-3,4)
 A=norm(rawA,va);B=norm(rawB,vb)
f=np.load(args.seedfile);pts=f['points'];par=f['params'];radius=float(f['radius']);t=time.time()
def task(i):
 pp=par[i].copy();pp[:,1:3]*=args.radius/radius
 return fit_local(A,B,va,vb,pts[i],pp[:,0],r=args.radius,order=args.order,reg=args.reg,seed_affine=pp)
with ThreadPoolExecutor(max_workers=4) as ex:res=list(ex.map(task,range(len(pts))))
p=np.array([r[0] for r in res]);st={k:np.array([r[1].get(k,np.nan) for r in res]) for k in res[0][1]}
np.savez('work/'+args.name+'.npz',points=pts,params=p,radius=args.radius,order=args.order,reg=args.reg,margin=args.margin,sigma=args.sigma,maskednorm=args.maskednorm,**st)
met,_,_=evaluate(pts,p[:,:,0]);print(args.name,'time',round(time.time()-t,1),json.dumps(met),flush=True)
with open('work/'+args.name+'_metrics.json','w') as out:json.dump(met,out,indent=2)
