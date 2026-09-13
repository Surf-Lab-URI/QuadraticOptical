"""Apply original conservative evidence tests over the full image width."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,json,time
import numpy as np
from scipy.spatial import cKDTree,Delaunay
from scipy.ndimage import map_coordinates
sys.path.insert(0,str(Path(__file__).resolve().parent))
from field_model import LocalField
W=Path('work/full_width');inp=np.load(W/'inputs.npz');origin=inp['origin0'];t0=time.time()
main=LocalField(W/'main.npz');reverse=LocalField(W/'reverse.npz')
alternatives=[LocalField(W/(k+'.npz')) for k in ['affine','large_window','margin14']]
grid=inp['points'];manual=inp['manual_xy'];N=len(grid)
xx,yy=np.meshgrid(np.arange(0,2048,4.),np.arange(344,749,4.))
dense=np.c_[xx.ravel(),yy.ravel()]-origin;q=np.r_[manual,grid,dense]
u,g,ncc=main.evaluate(q,True);print('primaryevaluated',len(q),'seconds',round(time.time()-t0,1),flush=True)
ua=[];ga=[]
for model in alternatives:
    a,b=model.evaluate(q);ua.append(a);ga.append(b)
ua=np.array(ua);ga=np.array(ga)
available=np.all(np.isfinite(ua),axis=(0,2))&np.all(np.isfinite(ga),axis=(0,2,3))
spread=np.max(np.linalg.norm(ua-u[None],axis=2),axis=0)
gspread=np.max(np.abs(ga[:,:,0,0]-g[None,:,0,0]),axis=0)
bu,_=reverse.evaluate(q+u);fb=np.linalg.norm(u+bu,axis=1)
print('variants/reverseevaluated','seconds',round(time.time()-t0,1),flush=True)
tr=np.load(W/'ptv_tracks.npz');features=tr['points'][tr['accepted']];tree=cKDTree(features);hull=Delaunay(features)
nearest=tree.query(q)[0];count=np.array([len(v) for v in tree.query_ball_point(q,25)])
support=(hull.find_simplex(q)>=0)&(nearest<=12)&(count>=6)
sa,sb=inp['surface_a'],inp['surface_b'];sx=np.arange(len(sa));target=q+u
depth=q[:,1]-np.interp(q[:,0],sx,sa);targetdepth=target[:,1]-np.interp(target[:,0],sx,sb)
def share(mod,query):
    good=(mod.data['mindet']>.05)&(mod.data['support_fraction']>.85);out=[]
    for pt in query:
        if not np.all(np.isfinite(pt)):out.append(0.);continue
        ii=mod.tree.query_ball_point(pt,mod.R)
        if not ii:out.append(0.);continue
        t=np.linalg.norm(pt-mod.points[ii],axis=1)/mod.R;w=(1-t)**4*(1+4*t)
        out.append(np.dot(w,good[ii])/w.sum() if w.sum()>1e-12 else 0.)
    return np.array(out)
forward_share=share(main,q);reverse_share=share(reverse,target)
det=np.full(len(g),np.nan);finiteg=np.all(np.isfinite(g),axis=(1,2));det[finiteg]=np.linalg.det(np.eye(2)[None]+g[finiteg])
def visible(mask,pts,strict_target=False):
    inside=np.isfinite(pts).all(axis=1)&(pts[:,0]>=0)&(pts[:,0]<=mask.shape[1]-1)&(pts[:,1]>=0)&(pts[:,1]<=mask.shape[0]-1)
    if strict_target:
        inside&=(pts[:,0]>=1)&(pts[:,0]<mask.shape[1]-2)&(pts[:,1]>=1)&(pts[:,1]<mask.shape[0]-2)
    valid=np.zeros(len(pts),bool)
    valid[inside]=map_coordinates(mask.astype(float),[pts[inside,1],pts[inside,0]],order=1,mode='constant',cval=0)>.99
    return valid
source_visible=visible(inp['va'],q);target_visible=visible(inp['vb'],target,strict_target=True)
with np.errstate(invalid='ignore'):
    evidence=available&(forward_share>=.95)&(reverse_share>=.95)&(depth>=12)&(depth<=354)&(targetdepth>=10)&support&(ncc>=.6)&(fb<=1)&(spread<=1.5)&(det>.2)&np.all(np.isfinite(u),axis=1)
    accepted=evidence&source_visible&target_visible
    gradient_accepted=accepted&(depth>=20)&(gspread<=.08)&(nearest<=10)
truth=inp['manual_truth'];error=np.linalg.norm(u[:200]-truth,axis=1)
previous=np.load('work/large_frame/results.npz');old=np.load('work/final_results.npz')
previouschange=np.linalg.norm(u[:200]-previous['disp'][:200],axis=1)
originalchange=np.linalg.norm(u[:200]-old['disp'][:200],axis=1)
def metrics(sel):
    a=error[sel];a=a[np.isfinite(a)]
    return dict(n=len(a),mean=float(a.mean()) if len(a) else None,median=float(np.median(a)) if len(a) else None,
                rmse=float(np.sqrt(np.mean(a*a))) if len(a) else None,p90=float(np.percentile(a,90)) if len(a) else None)
ms={}
for name,sel in [('all',np.ones(200,bool)),('near40',depth[:200]<40),('depth40_80',(depth[:200]>=40)&(depth[:200]<80)),('depth80plus',depth[:200]>=80)]:
    ms[name]=metrics(sel);ms[name+'_screened']=metrics(sel&accepted[:200])
gs=slice(200,200+N);ok=accepted[gs];dep=depth[gs];gok=gradient_accepted[gs]
depthbins=[]
for lo,hi in zip([12,20,40,80,150,250],[20,40,80,150,250,354.001]):
    a=(dep>=lo)&(dep<hi);depthbins.append(dict(low=lo,high=min(hi,354),evaluated=int(a.sum()),accepted=int((a&ok).sum()),gradient=int((a&gok).sum())))
widthbins=[]
for lo,hi in [(0,512),(512,1024),(1024,1536),(1536,2048)]:
    a=(grid[:,0]>=lo)&(grid[:,0]<hi);shallow=a&(dep<40)
    widthbins.append(dict(x_low=lo,x_high=hi,evaluated=int(a.sum()),accepted=int((a&ok).sum()),gradient=int((a&gok).sum()),
                          min_accepted_depth=float(dep[a&ok].min()) if np.any(a&ok) else None,near40_accepted=int((shallow&ok).sum())))
summary=dict(method='Original bilinear maskedquadratic opticalflow and automaticPTV initialization, unchanged evidence thresholds, entire image width.',
    grid_nodes=N,accepted_grid=int(ok.sum()),gradient_grid=int(gok.sum()),manual_metrics=ms,
    accepted_features=int(tr['accepted'].sum()),min_accepted_grid_depth=float(dep[ok].min()),min_gradient_grid_depth=float(dep[gok].min()),
    accepted_grid_x_min=float(grid[ok,0].min()),accepted_grid_x_max=float(grid[ok,0].max()),depth_bins=depthbins,horizontal_bins=widthbins,
    previous_manual_max_change_px=float(previouschange.max()),previous_manual_changed_picks=(np.where(previouschange>1e-9)[0]+1).tolist(),
    original_manual_max_change_px=float(originalchange.max()),previous_manual_acceptance_changes=int(np.sum(accepted[:200]!=previous['accepted'][:200])),
    mask_visibility_exclusions_grid=int(np.sum(evidence[gs]&~(source_visible[gs]&target_visible[gs]))),
    thresholds=dict(min_depth=12,max_depth=354,target_depth=10,ncc=.6,fb=1,spread=1.5,det=.2,nearest=12,count25=6,valid_share=.95,
                    gradient_depth=20,gradient_spread=.08,gradient_nearest=10),
    origin_zero_based=origin.tolist(),DX=float(inp['DX']),DT=float(inp['DT']),physical_units_confirmed=False,
    full_width_cm_if_SI=float(inp['DX'])*2048*100,depth_cm_if_SI=float(inp['DX'])*354*100,
    field_domain_note='Explicit source/target visiblepixel check prevents showing estimates in missingmaskdata or outsideimage; original evidence thresholds unchanged.',
    validation_note='Only the existing200manualmatches, locatedaroundthefirstdepression, are available. No newmanualvalidationacrossotherdepressions or addeddepth.',
    elapsed_seconds=time.time()-t0)
np.savez_compressed(W/'results.npz',query=q,query_full=q+origin,disp=u,gradient=g,ncc=ncc,fb=fb,depth=depth,target_depth=targetdepth,
    method_spread=spread,gradient_spread=gspread,alternatives_available=available,local_valid_share=forward_share,reverse_valid_share=reverse_share,
    support=support,nearest_feature=nearest,feature_count25=count,determinant=det,source_visible=source_visible,target_visible=target_visible,
    evidence_pass=evidence,accepted=accepted,gradient_accepted=gradient_accepted,grid_count=N,dense_x_full=xx,dense_y_full=yy,
    manual_truth=truth,manual_error=error,previous_manual_prediction=previous['disp'][:200],previous_manual_change=previouschange,
    original_manual_prediction=old['disp'][:200],original_manual_change=originalchange,alternative_displacements=ua,alternative_gradients=ga,
    previous_grid_flag=main.data['frozen_previous'],original_grid_flag=main.data['frozen_original'],origin0=origin,
    full_surface_a=inp['full_surface_a'],full_surface_b=inp['full_surface_b'],DX=inp['DX'],DT=inp['DT'])
(W/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
