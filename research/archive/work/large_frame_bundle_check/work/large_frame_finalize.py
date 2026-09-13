"""Apply original conservative screening to the expanded, original-order model."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,json
import numpy as np
from scipy.spatial import cKDTree,Delaunay
from scipy.ndimage import map_coordinates
sys.path.insert(0,str(Path(__file__).resolve().parent))
from field_model import LocalField

O=Path('work/large_frame');inp=np.load(O/'inputs.npz');origin=inp['origin0']
main=LocalField(O/'main.npz');back=LocalField(O/'reverse.npz')
alts=[LocalField(O/(k+'.npz')) for k in ['affine','large_window','margin14']]
grid=inp['points'];manual=inp['manual_xy'];N=len(grid)
xx,yy=np.meshgrid(np.arange(319,720,4.),np.arange(350,797,4.))
dense=np.c_[xx.ravel(),yy.ravel()]-origin
q=np.r_[manual,grid,dense];u,g,ncc=main.evaluate(q,True)
ua=[];ga=[]
for m in alts:
 a,b=m.evaluate(q);ua.append(a);ga.append(b)
ua=np.array(ua);ga=np.array(ga)
available=np.all(np.isfinite(ua),axis=(0,2))&np.all(np.isfinite(ga),axis=(0,2,3))
spread=np.max(np.linalg.norm(ua-u[None],axis=2),axis=0)
gspread=np.max(np.abs(ga[:,:,0,0]-g[None,:,0,0]),axis=0)
bu,_=back.evaluate(q+u);fb=np.linalg.norm(u+bu,axis=1)
tr=np.load(O/'ptv_tracks.npz');features=tr['points'][tr['accepted']]
tree=cKDTree(features);hull=Delaunay(features)
nearest=tree.query(q)[0];number=np.array([len(v) for v in tree.query_ball_point(q,25)])
support=(hull.find_simplex(q)>=0)&(nearest<=12)&(number>=6)
sa,sb=inp['surface_a'],inp['surface_b'];sx=np.arange(len(sa))
depth=q[:,1]-np.interp(q[:,0],sx,sa);target=q+u
targetdepth=target[:,1]-np.interp(target[:,0],sx,sb)
def share(mod,query):
 good=(mod.data['mindet']>.05)&(mod.data['support_fraction']>.85)
 out=[]
 for pt in query:
  if not np.all(np.isfinite(pt)):out.append(0);continue
  ii=mod.tree.query_ball_point(pt,mod.R)
  if not ii:out.append(0);continue
  t=np.linalg.norm(pt-mod.points[ii],axis=1)/mod.R;w=(1-t)**4*(1+4*t)
  out.append(np.dot(w,good[ii])/w.sum() if w.sum()>1e-12 else 0.)
 return np.array(out)
forward_share=share(main,q);reverse_share=share(back,q+u)
det=np.full(len(g),np.nan);finite=np.all(np.isfinite(g),axis=(1,2))
det[finite]=np.linalg.det(np.eye(2)[None]+g[finite])
accepted=available&(forward_share>=.95)&(reverse_share>=.95)&(depth>=12)&(depth<=float(inp['max_depth']))&(targetdepth>=10)&support&(ncc>=.6)&(fb<=1)&(spread<=1.5)&(det>.2)&np.all(np.isfinite(u),axis=1)
gradient_accepted=accepted&(depth>=20)&(gspread<=.08)&(nearest<=10)
truth=inp['manual_truth'];error=np.linalg.norm(u[:200]-truth,axis=1)
old=np.load('work/final_results.npz');oldchange=np.linalg.norm(u[:200]-old['disp'][:200],axis=1)
piv=np.c_[map_coordinates(inp['supplied_dx'],[manual[:,1],manual[:,0]],order=1),
           map_coordinates(inp['supplied_dy'],[manual[:,1],manual[:,0]],order=1)]
piverror=np.linalg.norm(piv-truth,axis=1)
def metrics(v,sel):
 t=v[sel];t=t[np.isfinite(t)]
 return dict(n=int(len(t)),mean=float(np.mean(t)) if len(t) else None,median=float(np.median(t)) if len(t) else None,
             rmse=float(np.sqrt(np.mean(t*t))) if len(t) else None,p90=float(np.percentile(t,90)) if len(t) else None)
ms={}
for label,s in [('all',np.ones(200,bool)),('near40',depth[:200]<40),('depth40_80',(depth[:200]>=40)&(depth[:200]<80)),('depth80plus',depth[:200]>=80)]:
 ms[label]=metrics(error,s);ms[label+'_screened']=metrics(error,s&accepted[:200]);ms[label+'_supplied_PIV']=metrics(piverror,s)
gs=slice(200,200+N);ng=int(accepted[gs].sum());gg=int(gradient_accepted[gs].sum())
summary=dict(method='Original automatic particle tracking plus masked local quadratic registration; bilinear image sampling, frozen original local fits, C2 blend radius16.',
             grid_nodes=N,accepted_grid=ng,gradient_grid=gg,manual_metrics=ms,
             min_accepted_grid_depth=float(np.min(depth[gs][accepted[gs]])),min_gradient_grid_depth=float(np.min(depth[gs][gradient_accepted[gs]])),
             accepted_features=int(tr['accepted'].sum()),original_features=int(np.sum(tr['accepted']&tr['original_track'])) if 'original_track' in tr.files else 675,
             max_change_at_original_manual_points=float(np.max(oldchange)),changed_manual_picks=(np.where(oldchange>1e-9)[0]+1).tolist(),
             threshold_note='All original evidence thresholds retained. Maximum depth expanded from150 to354pixels as an analysis-extent change.',
             thresholds=dict(min_depth=12,max_depth=354,target_depth=10,ncc=.6,fb=1,spread=1.5,det=.2,nearest=12,count25=6,valid_share=.95,gradient_depth=20,gradient_spread=.08,gradient_nearest=10),
             origin_zero_based=origin.tolist(),DX=float(inp['DX']),DT=float(inp['DT']),physical_units_confirmed=False,
             independence_note='Old manual displacements did not fit either model; this same pair was previously used for exploratory method comparison. No additional manual picks exist in the new MAT file.')
np.savez_compressed(O/'results.npz',query=q,query_full=q+origin,disp=u,gradient=g,ncc=ncc,fb=fb,
                    depth=depth,target_depth=targetdepth,method_spread=spread,gradient_spread=gspread,
                    alternatives_available=available,local_valid_share=forward_share,reverse_valid_share=reverse_share,
                    support=support,nearest_feature=nearest,feature_count25=number,determinant=det,
                    accepted=accepted,gradient_accepted=gradient_accepted,grid_count=N,dense_x_full=xx,dense_y_full=yy,
                    manual_truth=truth,manual_error=error,old_manual_prediction=old['disp'][:200],manual_change=oldchange,
                    supplied_PIV_at_manual=piv,supplied_PIV_error=piverror,alternative_displacements=ua,alternative_gradients=ga,
                    original_grid_flag=main.data['frozen_original'],origin0=origin,full_surface_a=inp['full_surface_a'],full_surface_b=inp['full_surface_b'],DX=inp['DX'],DT=inp['DT'])
(O/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
