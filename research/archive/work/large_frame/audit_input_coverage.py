"""Read-only independent masks, stitching, tracking, and screen audit."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json
import numpy as np
from scipy.ndimage import gaussian_filter,map_coordinates
from scipy.spatial import cKDTree

O=Path('work/large_frame');i=np.load(O/'inputs.npz');r=np.load(O/'results.npz');t=np.load(O/'ptv_tracks.npz');old=np.load('work/final_results.npz');co=np.load(O/'coarse_affine.npz')
origin=i['origin0'];oldorigin=i['old_origin0'];N=int(r['grid_count']);sl=slice(200,200+N);q=r['query'][sl];u=r['disp'][sl];ac=r['accepted'][sl];depth=r['depth'][sl];orig=r['original_grid_flag'];globalq=q+origin;globaltarget=globalq+u

def stats(x):
 x=np.asarray(x);x=x[np.isfinite(x)]
 return dict(n=len(x),minimum=float(x.min()),median=float(np.median(x)),p90=float(np.percentile(x,90)),maximum=float(x.max())) if len(x) else dict(n=0)
def sample(a,xy):
 return map_coordinates(a.astype(float),[xy[:,1],xy[:,0]],order=1,mode='constant',cval=0)
def in_old_roi(xy):
 return np.all((xy>=oldorigin)&(xy<=oldorigin+500),axis=1)

source_visible=sample(i['availability_a'],q)>.99;target_visible=sample(i['availability_b'],q+u)>.99
source_valid=sample(i['va'],q)>.99;target_valid=sample(i['vb'],q+u)>.99
seam_distance_source=np.min(np.c_[globalq-oldorigin,oldorigin+500-globalq],axis=1)
seam_distance_target=np.min(np.c_[globaltarget-oldorigin,oldorigin+500-globaltarget],axis=1)
crop_distance_target=np.min(np.c_[q+u,np.array([619,539])-(q+u)],axis=1)
shallow=[]
for j in np.where(ac&(depth<19.018))[0]:
 shallow.append(dict(grid_index=int(j),global_xy=globalq[j].tolist(),displacement=u[j].tolist(),depth=float(depth[j]),target_depth=float(r['target_depth'][sl][j]),source_valid=bool(source_valid[j]),target_valid=bool(target_valid[j]),inside_original_raw_A=bool(in_old_roi(globalq)[j]),inside_original_raw_B=bool(in_old_roi(globaltarget)[j]),ncc=float(r['ncc'][sl][j]),fb=float(r['fb'][sl][j]),spread=float(r['method_spread'][sl][j]),nearest_feature=float(r['nearest_feature'][sl][j]),feature_count25=int(r['feature_count25'][sl][j]),forward_valid_share=float(r['local_valid_share'][sl][j]),reverse_valid_share=float(r['reverse_valid_share'][sl][j]),frozen_original=bool(orig[j])))

# Original-grid correspondence and exact changed-screen causes.
oldq=old['query'][200:927]+oldorigin-origin
dist,newidx=cKDTree(q).query(oldq);assert np.all(dist==0)
def screen_parts(data,ix,maxdepth):
 return dict(alternatives=data['alternatives_available'][ix],forward_share=data['local_valid_share'][ix]>=.95,reverse_share=data['reverse_valid_share'][ix]>=.95,source_depth_min=data['depth'][ix]>=12,source_depth_max=data['depth'][ix]<=maxdepth,target_depth=data['target_depth'][ix]>=10,support=data['support'][ix],ncc=data['ncc'][ix]>=.6,fb=data['fb'][ix]<=1,spread=data['method_spread'][ix]<=1.5,determinant=data['determinant'][ix]>.2,finite=np.isfinite(data['disp'][ix]).all(axis=1))
oldparts=screen_parts(old,np.arange(200,927),150);newparts=screen_parts(r,newidx+200,354)
changed=[]
for j in np.where(old['accepted'][200:927]!=ac[newidx])[0]:
 k=newidx[j]
 changed.append(dict(old_index=int(j),new_grid_index=int(k),global_xy=globalq[k].tolist(),depth=float(depth[k]),old_accepted=bool(old['accepted'][200+j]),new_accepted=bool(ac[k]),old_failures=[z for z in oldparts if not oldparts[z][j]],new_failures=[z for z in newparts if not newparts[z][j]],displacement_change=float(np.linalg.norm(u[k]-old['disp'][200+j]))))

# Recompute accepted new-particle raw photometry without invoking track optimizer.
ta=gaussian_filter(i['rawA'],.5);tb=gaussian_filter(i['rawB'],.5)
Y,X=np.mgrid[-4:5,-4:5];off=np.c_[X.ravel(),Y.ravel()]
def ncc_check(pt,disp,A,B,va,vb):
 c=pt+off;d=c+disp;v=(sample(va,c)>.99)&(sample(vb,d)>.99)
 a=sample(A,c)[v];b=sample(B,d)[v];a=a-a.mean();b=b-b.mean()
 return np.dot(a,b)/np.sqrt(np.dot(a,a)*np.dot(b,b)+1e-10),int(v.sum())
ids=np.flatnonzero(t['accepted']&~t['original_candidate']);fn=[];bn=[];nv=[];rv=[]
for j in ids:
 n,v=ncc_check(t['points'][j],t['disp'][j],ta,tb,i['va'],i['vb']);fn.append(n);nv.append(v)
 n,v=ncc_check(t['points'][j]+t['disp'][j],t['back_disp'][j],tb,ta,i['vb'],i['va']);bn.append(n);rv.append(v)
fn=np.array(fn);bn=np.array(bn)
dest=t['points'][t['accepted']]+t['disp'][t['accepted']];pairs=sorted(cKDTree(dest).query_pairs(1.0));accepted_ids=np.flatnonzero(t['accepted'])
duplicate_records=[]
for a,b in pairs:
 aa,bb=accepted_ids[[a,b]]
 duplicate_records.append(dict(rows=[int(aa),int(bb)],target_distance=float(np.linalg.norm(dest[a]-dest[b])),source_distance=float(np.linalg.norm(t['points'][aa]-t['points'][bb])),either_new=bool(not(t['original_candidate'][aa]&t['original_candidate'][bb]))))

newco=~co['original_coarse'];coq=co['points'][newco];sup=np.c_[sample(i['supplied_dx'],coq),sample(i['supplied_dy'],coq)]
oldchange=np.linalg.norm(u[newidx]-old['disp'][200:927],axis=1)
summary=dict(grid_total=N,grid_accepted=int(ac.sum()),source_availability_fail_accepted=int(np.sum(ac&~source_visible)),target_availability_fail_accepted=int(np.sum(ac&~target_visible)),source_margin10_mask_fail_accepted=int(np.sum(ac&~source_valid)),target_margin10_mask_fail_accepted=int(np.sum(ac&~target_valid)),accepted_source_inside_original_raw=int(np.sum(ac&in_old_roi(globalq))),accepted_target_inside_original_raw=int(np.sum(ac&in_old_roi(globaltarget))),accepted_source_distance_to_original_raw_edge=stats(seam_distance_source[ac]),accepted_target_distance_to_original_raw_edge=stats(seam_distance_target[ac]),accepted_target_distance_to_analysis_crop_edge=stats(crop_distance_target[ac]),shallower_than_original_min=shallow,original_grid_acceptance_changes=changed,original_grid_accepted_before=int(old['accepted'][200:927].sum()),original_grid_accepted_after=int(ac[newidx].sum()),original_grid_displacement_change=stats(oldchange),new_accepted_track_NCC_recompute_max_difference=float(np.max(abs(fn-t['ncc'][ids]))),new_accepted_reverse_NCC_recompute_max_difference=float(np.max(abs(bn-t['back_ncc'][ids]))),new_accepted_forward_common_pixel_count=stats(nv),new_accepted_reverse_common_pixel_count=stats(rv),accepted_target_pairs_within1px=duplicate_records,new_coarse_translation_movement_from_supplied_initializer=stats(np.linalg.norm(co['params'][newco,:,0]-sup,axis=1)),new_coarse_fallback_count=int(np.sum(co['coarse_fallback'][newco])),manual_used_for_this_audit=False)
(O/'input_coverage_audit.json').write_text(json.dumps(summary,indent=2))
small={k:v for k,v in summary.items() if k not in ['shallower_than_original_min','original_grid_acceptance_changes','accepted_target_pairs_within1px']};small['shallow_count']=len(shallow);small['changed_original_grid_count']=len(changed);small['duplicate_target_pair_count']=len(pairs)
print(json.dumps(small,indent=2));print('SHALLOW',json.dumps(shallow,indent=2))
