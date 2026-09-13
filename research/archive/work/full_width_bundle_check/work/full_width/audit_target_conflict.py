"""Inspect one shared-target candidate pair without modifying any fit or screen."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,json
import numpy as np
from scipy.spatial import cKDTree,Delaunay
from scipy.ndimage import map_coordinates
sys.path.insert(0,'work')
from field_model import LocalField
from ptv_model import PTVModel
O=Path('work/full_width');i=np.load(O/'inputs.npz');t=np.load(O/'ptv_tracks.npz');rows=np.array([5647,5710])
points=t['points'][rows];gx,gy=np.meshgrid(np.arange(247,288,8),np.arange(411,452,8));grid=np.c_[gx.ravel(),gy.ravel()]-i['origin0'];q=np.r_[points,grid]
primary=LocalField(O/'main.npz');back=LocalField(O/'reverse.npz');alts=[LocalField(O/(v+'.npz')) for v in ['affine','large_window','margin14']]
d,g,ncc=primary.evaluate(q,True);ad=[];ag=[]
for model in alts:
 v,j=model.evaluate(q);ad.append(v);ag.append(j)
ad=np.array(ad);ag=np.array(ag);bd,_=back.evaluate(q+d);fb=np.linalg.norm(d+bd,axis=1);spread=np.max(np.linalg.norm(ad-d[None],axis=2),axis=0);gspread=np.max(abs(ag[:,:,0,0]-g[None,:,0,0]),axis=0)
features=t['points'][t['accepted']];tree=cKDTree(features);hull=Delaunay(features);near=tree.query(q)[0];count=np.array([len(v) for v in tree.query_ball_point(q,25)]);support=(hull.find_simplex(q)>=0)&(near<=12)&(count>=6)
def share(model,p):
 good=(model.data['mindet']>.05)&(model.data['support_fraction']>.85);out=[]
 for point in p:
  ids=model.tree.query_ball_point(point,model.R);dist=np.linalg.norm(point-model.points[ids],axis=1)/model.R;w=(1-dist)**4*(1+4*dist);out.append(np.dot(w,good[ids])/w.sum())
 return np.array(out)
fs=share(primary,q);bs=share(back,q+d);depth=q[:,1]-np.interp(q[:,0],np.arange(len(i['surface_a'])),i['surface_a']);td=q[:,1]+d[:,1]-np.interp(q[:,0]+d[:,0],np.arange(len(i['surface_b'])),i['surface_b']);det=np.linalg.det(np.eye(2)[None]+g)
def visible(mask,p):return map_coordinates(mask.astype(float),[p[:,1],p[:,0]],order=1,mode='constant',cval=0)>.99
tests=dict(alternatives=np.isfinite(ad).all(axis=(0,2))&np.isfinite(ag).all(axis=(0,2,3)),forward_share=fs>=.95,reverse_share=bs>=.95,source_depth=(depth>=12)&(depth<=354),target_depth=td>=10,support=support,ncc=ncc>=.6,fb=fb<=1,spread=spread<=1.5,det=det>.2,finite=np.isfinite(d).all(axis=1),source_visible=visible(i['va'],q),target_visible=visible(i['vb'],q+d))
accepted=np.logical_and.reduce(list(tests.values()));gradaccepted=accepted&(depth>=20)&(gspread<=.08)&(near<=10)
records=[]
for k in range(len(q)):
 records.append(dict(source_full=(q[k]+i['origin0']).tolist(),particle_row=int(rows[k]) if k<2 else None,displacement=d[k].tolist(),gradient=g[k].tolist(),ncc=float(ncc[k]),fb=float(fb[k]),spread=float(spread[k]),gradient_spread=float(gspread[k]),depth=float(depth[k]),forward_share=float(fs[k]),reverse_share=float(bs[k]),accepted=bool(accepted[k]),gradient_accepted=bool(gradaccepted[k]),failures=[name for name in tests if not tests[name][k]]))
p=PTVModel(O/'ptv_tracks.npz').evaluate(points);tracks=[]
for k,row in enumerate(rows):
 tracks.append(dict(row=int(row),source_full=(points[k]+i['origin0']).tolist(),track_displacement=t['disp'][row].tolist(),coarse_prior=t['prior'][row].tolist(),robust_particle_field=p['disp'][k].tolist(),final_continuous_field=d[k].tolist(),track_vs_coarse_residual=float(np.linalg.norm(t['disp'][row]-t['prior'][row])),track_vs_robust_field_residual=float(np.linalg.norm(t['disp'][row]-p['disp'][k])),final_field_ncc=float(ncc[k]),final_field_accepted=bool(accepted[k])))
out=dict(target_distance=float(np.linalg.norm(points[0]+t['disp'][rows[0]]-points[1]-t['disp'][rows[1]])),source_distance=float(np.linalg.norm(points[0]-points[1])),tracks=tracks,query_records=records,manual_used=False,predictions_or_thresholds_modified=False)
(O/'target_conflict_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
