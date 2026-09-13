"""Smooth compact blending of independently image-fitted local polynomials.
Both translations and analytic derivatives refer to zero-based source coordinates.
"""
import numpy as np
from scipy.spatial import cKDTree
class LocalField:
 def __init__(self,path,blend_radius=16.):
  f=np.load(path);self.points=f['points'];self.params=f['params'];self.radius=float(f['radius']);self.tree=cKDTree(self.points);self.R=float(blend_radius);self.data={k:f[k] for k in f.files}
 def evaluate(self,query,diagnostics=False):
  query=np.atleast_2d(query);out=[];gr=[];quality=[]
  for q in query:
   if not np.all(np.isfinite(q)):
    out.append([np.nan]*2);gr.append(np.full((2,2),np.nan));quality.append(np.nan);continue
   ii=self.tree.query_ball_point(q,self.R)
   if not ii:out.append([np.nan]*2);gr.append(np.full((2,2),np.nan));quality.append(np.nan);continue
   ii=np.array(ii);d=q-self.points[ii];dist=np.linalg.norm(d,axis=1);t=dist/self.R
   w=(1-t)**4*(1+4*t)
   if w.sum()<1e-12:
    out.append([np.nan]*2);gr.append(np.full((2,2),np.nan));quality.append(np.nan);continue
   dw=-20/self.R**2*(1-t[:,None])**3*d
   z=d/self.radius;zx=z[:,0];zy=z[:,1]
   Q=np.column_stack([np.ones(len(ii)),zx,zy]);qx=np.column_stack([np.zeros(len(ii)),np.ones(len(ii)),np.zeros(len(ii))])/self.radius;qy=qx[:,[0,2,1]]
   if self.params.shape[2]==6:
    Q=np.column_stack([Q,.5*zx*zx,zx*zy,.5*zy*zy]);qx=np.column_stack([qx,zx/self.radius,zy/self.radius,np.zeros(len(ii))]);qy=np.column_stack([qy,np.zeros(len(ii)),zx/self.radius,zy/self.radius])
   value=np.einsum('nij,nj->ni',self.params[ii],Q);dx=np.einsum('nij,nj->ni',self.params[ii],qx);dy=np.einsum('nij,nj->ni',self.params[ii],qy)
   avg=np.sum(w[:,None]*value,axis=0)/w.sum();gx=np.sum(w[:,None]*dx+dw[:,0,None]*(value-avg),axis=0)/w.sum();gy=np.sum(w[:,None]*dy+dw[:,1,None]*(value-avg),axis=0)/w.sum()
   out.append(avg);gr.append(np.column_stack([gx,gy]));quality.append(float(np.dot(w,self.data.get('ncc',np.ones(len(self.points)))[ii])/w.sum()))
  result=(np.array(out),np.array(gr))
  return result+(np.array(quality),) if diagnostics else result
if __name__=='__main__':
 import json
 m=np.load('work/metadata.npz');p=m['p'][:,0].T-1;truth=m['p'][:,1].T-m['p'][:,0].T;dep=p[:,1]-np.interp(p[:,0],np.arange(501),m['surfa'].ravel()-1)
 for name in ['final_ptv13','final_ptv19','final_affine13','final_ptv13_margin14','final_spline13']:
  model=LocalField('work/'+name+'.npz');d,g=model.evaluate(p);e=np.linalg.norm(d-truth,axis=1)
  met={}
  for label,sel in [('all',np.ones(len(p),bool)),('near40',dep<40),('depth40-80',(dep>=40)&(dep<80)),('depth80+',dep>=80)]:
   v=e[sel];met[label]=dict(n=int(sel.sum()),median=float(np.median(v)),mean=float(np.mean(v)),rmse=float(np.sqrt(np.mean(v*v))),p90=float(np.percentile(v,90)),under1=float(np.mean(v<1)),under2=float(np.mean(v<2)))
  eps=1e-3;fd=(model.evaluate(p+np.array([eps,0]))[0]-model.evaluate(p-np.array([eps,0]))[0])/(2*eps)
  print(name,json.dumps(met),'gradient_fd_max',np.max(np.abs(fd-g[:,:,0])),flush=True)
  np.savez('work/'+name+'_blended_validation.npz',points=p,disp=d,gradient=g,error=e,depth=dep)
  with open('work/'+name+'_blended_metrics.json','w') as f:json.dump(met,f,indent=2)
