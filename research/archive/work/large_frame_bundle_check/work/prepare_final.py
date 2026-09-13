import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np,json,time
from field_model import LocalField
from scipy.spatial import cKDTree,Delaunay
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import map_coordinates
m=np.load('work/metadata.npz');manual=m['p'][:,0].T-1;truth=m['p'][:,1].T-m['p'][:,0].T;sa=m['surfa'].ravel()-1;sb=m['surfb'].ravel()-1
model=LocalField('work/final_ptv13.npz');alternates=[LocalField('work/'+name+'.npz') for name in ['final_affine13','final_ptv19','final_ptv13_margin14']]
rev=np.load('work/final_ptv13_reverse.npz');fbinterp=LinearNDInterpolator(model.points,rev['fb_error'],fill_value=np.nan)
tr=np.load('work/ptv_tracks.npz');features=tr['points'][tr['accepted']];tree=cKDTree(features);hull=Delaunay(features)
xx,yy=np.meshgrid(np.arange(60,391,2.),np.arange(12,181,2.));dense=np.column_stack([xx.ravel(),yy.ravel()]);q=np.concatenate([manual,model.points,dense]);t=time.time()
u,g,ncc=model.evaluate(q,True);ua=[];ga=[]
for alt in alternates:
 a,b=alt.evaluate(q);ua.append(a);ga.append(b)
def finite_max(a):
 v=np.max(np.where(np.isfinite(a),a,-np.inf),axis=0);v[~np.isfinite(v)]=np.nan;return v
spread=finite_max(np.linalg.norm(np.array(ua)-u[None],axis=2));gspread=finite_max(np.abs(np.array(ga)[:,:,0,0]-g[None,:,0,0]));alternatives_available=np.all(np.isfinite(np.array(ua)),axis=(0,2))&np.all(np.isfinite(np.array(ga)),axis=(0,2,3))
back=LocalField('work/final_reverse_field.npz');bu,_=back.evaluate(q+u);fb=np.linalg.norm(u+bu,axis=1);near=tree.query(q)[0];nh=np.array([len(j) for j in tree.query_ball_point(q,25)])
depth=q[:,1]-np.interp(q[:,0],np.arange(501),sa);dest=q+u;targetdep=dest[:,1]-np.interp(dest[:,0],np.arange(501),sb)
support=(hull.find_simplex(q)>=0)&(near<=12)&(nh>=6)
def valid_share(mod,query):
 good=(mod.data['mindet']>.05)&(mod.data['support_fraction']>.85);shares=[]
 for pt in query:
  if not np.all(np.isfinite(pt)):shares.append(0.);continue
  ii=mod.tree.query_ball_point(pt,mod.R)
  if not ii:shares.append(0.);continue
  t=np.linalg.norm(pt-mod.points[ii],axis=1)/mod.R;w=(1-t)**4*(1+4*t)
  shares.append(np.dot(w,good[ii])/w.sum() if w.sum()>1e-12 else 0.)
 return np.array(shares)
local_valid_share=valid_share(model,q);reverse_valid_share=valid_share(back,q+u)
determinant=np.full(len(g),np.nan);finite=np.all(np.isfinite(g),axis=(1,2));determinant[finite]=np.linalg.det(np.eye(2)[None]+g[finite]);np.seterr(invalid='ignore')
accepted=alternatives_available&(local_valid_share>=.95)&(reverse_valid_share>=.95)&(depth>=12)&(depth<=150)&(targetdep>=10)&support&(ncc>=.6)&(fb<=1)&(spread<=1.5)&(determinant>.2)&np.all(np.isfinite(u),axis=1)
goodgrad=accepted&(depth>=20)&(gspread<=.08)&(near<=10)
# No metric below is used for the image-only confidence screen above.
e=np.linalg.norm(u[:200]-truth,axis=1)
def metrics(mask):
 v=e[mask];return dict(n=int(mask.sum()),median=float(np.median(v)) if len(v) else None,mean=float(np.mean(v)) if len(v) else None,rmse=float(np.sqrt(np.mean(v*v))) if len(v) else None,p90=float(np.percentile(v,90)) if len(v) else None,within1=float(np.mean(v<1)) if len(v) else None,within2=float(np.mean(v<2)) if len(v) else None)
metrics_out={}
for key,sel in [('all',np.ones(200,bool)),('near40',depth[:200]<40),('depth40_80',(depth[:200]>=40)&(depth[:200]<80)),('depth80_plus',depth[:200]>=80)]:
 metrics_out[key]=metrics(sel);metrics_out[key+'_screened']=metrics(sel&accepted[:200])
N=len(model.points);dg=slice(200+N,None);gg=slice(200,200+N)
meta=dict(metrics=metrics_out,grid_nodes=N,accepted_grid=int(accepted[gg].sum()),gradient_grid=int(goodgrad[gg].sum()),min_accepted_grid_depth=float(np.min(depth[gg][accepted[gg]])),min_gradient_grid_depth=float(np.min(depth[gg][goodgrad[gg]])),accepted_features=int(tr['accepted'].sum()),min_feature_depth=float(np.min(features[:,1]-np.interp(features[:,0],np.arange(501),sa))),manual_min_depth=float(depth[:200].min()),manual_closest=[dict(pick=int(i+1),depth=float(depth[i]),manual=truth[i].tolist(),predicted=u[i].tolist(),epe=float(e[i]),accepted=bool(accepted[i]),ncc=float(ncc[i]),fb=float(fb[i]),spread=float(spread[i])) for i in np.argsort(depth[:200])[:7]],gradient_range=np.percentile(g[gg,0,0][goodgrad[gg]],[1,50,99]).tolist(),gradient_spread_median=float(np.median(gspread[gg][goodgrad[gg]])),thresholds=dict(min_depth=12,max_depth=150,min_target_depth=10,nearest_feature_max=12,feature_count_radius25_min=6,ncc_min=.6,fb_max=1,method_spread_max=1.5,min_determinant=.2,gradient_depth_min=20,gradient_method_spread_max=.08,local_valid_share_min=.95,reverse_valid_share_min=.95,all_three_alternatives_required=True))
np.savez_compressed('work/final_results.npz',query=q,disp=u,gradient=g,ncc=ncc,fb=fb,method_spread=spread,gradient_spread=gspread,depth=depth,target_depth=targetdep,support=support,alternatives_available=alternatives_available,local_valid_share=local_valid_share,reverse_valid_share=reverse_valid_share,accepted=accepted,gradient_accepted=goodgrad,nearest_feature=near,feature_count25=nh,determinant=determinant,manual_truth=truth,manual_error=e,grid_count=N,dense_x=xx,dense_y=yy,surface_a=sa,surface_b=sb,alternative_displacements=np.array(ua),alternative_gradients=np.array(ga))
with open('work/final_summary.json','w') as f:json.dump(meta,f,indent=2)
print(json.dumps(meta,indent=2));print('seconds',time.time()-t,flush=True)
