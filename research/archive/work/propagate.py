import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np,time,json
from extended_flow import load_images,fit_local,evaluate
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial import cKDTree
A,B,va,vb,sa,sb,_,_=load_images()
a=np.load('work/affine19.npz');b=np.load('work/affine19_dis.npz')
pts=a['points'];par=a['params'].copy();score=a['ncc'].copy()
choose=b['ncc']>score;par[choose]=b['params'][choose];score[choose]=b['ncc'][choose]
# Neighbor propagation uses image evidence only; preserves whole affine warp.
for passnum in range(3):
 t=time.time();old=par.copy();sc=score.copy();good=(sc>.60)&(np.linalg.det(np.eye(2)[None]+old[:,:,1:]/19)>.3)
 inds=np.where(good)[0];tree=cKDTree(pts[good]);dist,nbr=tree.query(pts,k=8)
 def run(i):
  candidates=[old[i]]
  if sc[i]<.86:
   for dd,jj in zip(np.atleast_1d(dist[i]),np.atleast_1d(nbr[i])):
    if dd>65 or dd<1:continue
    j=inds[jj];pp=old[j].copy();pp[:,0]+=pp[:,1:].dot((pts[i]-pts[j])/19)
    if any(np.linalg.norm(pp[:,0]-c[:,0])<.8 and np.linalg.norm(pp[:,1:]-c[:,1:])<2 for c in candidates):continue
    candidates.append(pp)
  out=[]
  for pp in candidates:
   fit,st=fit_local(A,B,va,vb,pts[i],pp[:,0],r=19,order=1,reg=.00005,seed_affine=pp)
   if st.get('mindet',0)>.15 and st.get('support_fraction',0)>.7:out.append((fit,st))
  if not out:return old[i],dict(ncc=float(sc[i]),mindet=0,support_fraction=0,rms=9,nvalid=0,cond=1e20,iterations=0)
  return max(out,key=lambda z:z[1]['ncc'])
 with ThreadPoolExecutor(max_workers=4) as ex:res=list(ex.map(run,range(len(pts))))
 par=np.array([r[0] for r in res]);stats={k:np.array([r[1].get(k,np.nan) for r in res]) for k in res[0][1]};score=stats['ncc']
 name='propagate'+str(passnum)
 np.savez('work/'+name+'.npz',points=pts,params=par,radius=19,**stats)
 met,pred,err=evaluate(pts,par[:,:,0]);print(name,'time',time.time()-t,json.dumps(met),flush=True)
 with open('work/'+name+'_metrics.json','w') as f:json.dump(met,f,indent=2)
