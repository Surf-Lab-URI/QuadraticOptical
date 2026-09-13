"""Fresh cubic reverse registration and matched-support diagnostics.
Uses the previously fixed screen thresholds; no manual errors set thresholds.
Alternates are the three frozen earlier-method variants, explicitly not newly
refit cubic window/mask ablations. This is a diagnostic comparison, not promotion.
"""
from experiments import *
from scipy.spatial import cKDTree
A,B,va,vb=inputs();f=np.load(HERE/'forward_cubic.npz');pts=f['points'];par=f['params'];r=13;target=pts+par[:,:,0];t=time.time();samplers=(Sampler(B,True),Sampler(A,True))
def min_det(p,center,mask):
 yy,xx=np.mgrid[-r:r+1,-r:r+1];off=np.c_[xx.ravel(),yy.ravel()];pos=center+off
 valid=(map_coordinates(mask.astype(float),[pos[:,1],pos[:,0]],order=1,mode='constant')>.99)
 G=local_jac(p,off[valid],r)
 return float(np.min(np.linalg.det(np.eye(2)+G))) if len(G) else np.nan

def task(i):
 G=par[i,:,1:3]/r;q=np.zeros((2,6));q[:,0]=-par[i,:,0]
 try:q[:,1:3]=(np.linalg.inv(np.eye(2)+G)-np.eye(2))*r
 except np.linalg.LinAlgError:return q,dict(ncc=-1,failed=True,support_fraction=0,mindet=-1)
 p,stats=solve(B,A,vb,va,target[i],q,r=r,cubic=True,samplers=samplers)
 stats['mindet']=min_det(p,target[i],vb);return p,stats
with ThreadPoolExecutor(max_workers=4) as ex:results=list(ex.map(task,range(len(pts))))
p=np.array([v[0] for v in results]);stats={k:np.array([v[1].get(k,np.nan) for v in results]) for k in set().union(*(v[1] for v in results))}
np.savez(HERE/'forward_cubic_reverse.npz',points=target,params=p,radius=r,**stats)
print('reverse finished seconds',time.time()-t,flush=True)
orig=np.load(WORK/'final_results.npz');q=orig['query'][:927];model=LocalField(HERE/'forward_cubic.npz');reverse=LocalField(HERE/'forward_cubic_reverse.npz')
u,g,ncc=model.evaluate(q,True);bu,bg=reverse.evaluate(q+u);fb=np.linalg.norm(u+bu,axis=1)
fmindet=np.array([min_det(pp,pt,va) for pp,pt in zip(par,pts)])
def share(model,queries,valid):
 out=[]
 for pt in queries:
  ids=model.tree.query_ball_point(pt,16)
  if not ids:out.append(0);continue
  dd=np.linalg.norm(model.points[ids]-pt,axis=1)/16;ww=(1-dd)**4*(1+4*dd)
  out.append(float(np.sum(ww*valid[ids])/sum(ww)))
 return np.array(out)
localshare=share(model,q,(fmindet>.05)&(f['support_fraction']>.85));backshare=share(reverse,q+u,(stats['mindet']>.05)&(stats['support_fraction']>.85))
spread=np.max(np.linalg.norm(orig['alternative_displacements'][:,:927]-u[None],axis=2),axis=0)
gspread=np.max(abs(orig['alternative_gradients'][:,:927,0,0]-g[None,:,0,0]),axis=0)
dep=orig['depth'][:927];dest=q+u;sb=orig['surface_b'];targetdep=dest[:,1]-np.interp(dest[:,0],np.arange(501),sb)
det=np.linalg.det(np.eye(2)+g)
accepted=orig['alternatives_available'][:927]&(localshare>=.95)&(backshare>=.95)&(dep>=12)&(dep<=150)&(targetdep>=10)&orig['support'][:927]&(ncc>=.6)&(fb<=1)&(spread<=1.5)&(det>.2)&np.all(np.isfinite(u),axis=1)
goodgrad=accepted&(dep>=20)&(gspread<=.08)&(orig['nearest_feature'][:927]<=10)
e=np.linalg.norm(u[:200]-orig['manual_truth'],axis=1);old=orig['manual_error'];nears=dep[:200]<40;frozen=orig['accepted'][:200];joint=frozen&accepted[:200]
def metric(v,mask):
 vv=v[mask];return dict(n=len(vv),mean=float(vv.mean()) if len(vv) else None,median=float(np.median(vv)) if len(vv) else None,rms=float(np.sqrt(np.mean(vv**2))) if len(vv) else None)
result=dict(description=__doc__,reverse_seconds=time.time()-t,reverse_failed_fallbacks=int(stats['failed'].sum()),local_nonpositive_determinants=int(sum(fmindet<=0)),reverse_nonpositive_determinants=int(sum(stats['mindet']<=0)),forward_backward_manual_percentiles=np.percentile(fb[:200],[0,50,90,95,100]).tolist(),forward_backward_grid_percentiles=np.percentile(fb[200:],[0,50,90,95,100]).tolist(),previous_forward_backward_grid_percentiles=np.percentile(orig['fb'][200:927],[0,50,90,95,100]).tolist(),new_accepted_manual=int(accepted[:200].sum()),new_accepted_near_manual=int(sum(accepted[:200]&nears)),new_accepted_grid=int(accepted[200:].sum()),new_gradient_grid=int(goodgrad[200:].sum()),minimum_accepted_grid_depth=float(dep[200:][accepted[200:]].min()),metrics={})
for name,sel in [('all',np.ones(200,bool)),('near40',nears),('frozen_screen',frozen),('frozen_near40',frozen&nears),('new_screen',accepted[:200]),('new_near40',accepted[:200]&nears),('joint_screen',joint),('joint_near40',joint&nears)]:
 result['metrics'][name]={'old':metric(old,sel),'cubic':metric(e,sel)}
np.savez(HERE/'forward_cubic_quality.npz',query=q,disp=u,gradient=g,ncc=ncc,fb=fb,method_spread=spread,gradient_spread=gspread,accepted=accepted,gradient_accepted=goodgrad,local_valid_share=localshare,reverse_valid_share=backshare,local_mindet=fmindet,manual_error=e)
(HERE/'quality_metrics.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
