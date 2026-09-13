import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np,json,time
from extended_flow import *
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
import argparse
pp=argparse.ArgumentParser();pp.add_argument('--name',default='direct11');pp.add_argument('--radius',type=int,default=11);pp.add_argument('--order',type=int,default=1);args=pp.parse_args()
A,B,va,vb,sa,sb,_,_=load_images(10,.65);m=np.load('work/metadata.npz');pts=m['p'][:,0].T-1
flow=np.load('work/seed_dis.npz')['flow']
ref=np.load('work/propagate0.npz');tree=cKDTree(ref['points']);dist,inds=tree.query(pts,k=5)
t=time.time()
def task(i):
 pt=pts[i];cands=[]
 # Grid and manual locations only query the same image estimator; endpoint never read here.
 seed=map_coordinates(flow,[np.repeat(pt[1],2),np.repeat(pt[0],2),np.arange(2)],order=1)
 cands.append((seed,None))
 for d,j in zip(dist[i],inds[i]):
  if ref['ncc'][j]<.6:continue
  par=ref['params'][j].copy();par[:,0]+=par[:,1:].dot((pt-ref['points'][j])/19);par[:,1:]*=args.radius/19
  cands.append((par[:,0],par))
 cands.extend((p[0],None) for p in translation_seed(A,B,va,vb,pt,r=args.radius)[:3])
 fits=[]
 for seed,aff in cands:
  p,st=fit_local(A,B,va,vb,pt,seed,r=args.radius,reg=.00005,seed_affine=aff)
  if args.order==2:p,st=fit_local(A,B,va,vb,pt,p[:,0],r=args.radius,order=2,reg=.00005,seed_affine=p)
  if st.get('mindet',0)>.15 and st.get('support_fraction',0)>.7:fits.append((p,st))
 return max(fits,key=lambda z:z[1]['ncc']) if fits else (np.full((2,6 if args.order==2 else 3),np.nan),dict(ncc=0))
with ThreadPoolExecutor(max_workers=4) as ex:res=list(ex.map(task,range(len(pts))))
par=np.array([r[0] for r in res]);st={k:np.array([r[1].get(k,np.nan) for r in res]) for k in res[0][1]}
np.savez('work/'+args.name+'.npz',points=pts,params=par,radius=args.radius,**st)
d=m['p'][:,1].T-m['p'][:,0].T;e=np.linalg.norm(par[:,:,0]-d,axis=1);dep=pts[:,1]-np.interp(pts[:,0],np.arange(501),sa)
met={}
for name,sel in [('all',np.ones(200,bool)),('near40',dep<40),('deep80',dep>=80)]:
 v=e[sel];met[name]=dict(n=int(sel.sum()),median=float(np.nanmedian(v)),mean=float(np.nanmean(v)),p90=float(np.nanpercentile(v,90)))
print(args.name,'time',time.time()-t,json.dumps(met),flush=True)
with open('work/'+args.name+'_metrics.json','w') as f:json.dump(met,f,indent=2)
