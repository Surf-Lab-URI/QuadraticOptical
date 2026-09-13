"""Independent read-only field endpoint and previous-domain preservation audit."""
from pathlib import Path
import json
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
O=Path('work/full_width');r=np.load(O/'results.npz');i=np.load(O/'inputs.npz');b=np.load('work/large_frame/results.npz');bi=np.load('work/large_frame/inputs.npz')
N=int(r['grid_count']);ix=np.arange(200,200+N);q=r['query'][ix];d=r['disp'][ix];ac=r['accepted'][ix];ga=r['gradient_accepted'][ix];full=q+i['origin0'];target=q+d
def sample(a,p):
 out=np.zeros(len(p));valid=np.isfinite(p).all(axis=1)
 out[valid]=map_coordinates(a.astype(float),[p[valid,1],p[valid,0]],order=1,mode='constant',cval=0)
 return out
def stats(x):
 x=np.asarray(x);x=x[np.isfinite(x)]
 return dict(n=len(x),min=float(x.min()),median=float(np.median(x)),p90=float(np.percentile(x,90)),max=float(x.max())) if len(x) else dict(n=0)
source_available=sample(i['availability_a'],q)>.99;target_available=sample(i['availability_b'],target)>.99
source_valid=sample(i['va'],q)>.99;target_valid=sample(i['vb'],target)>.99
# Match the original fit_local target interpolation/gradient border guard.
target_sampling_inside=(target[:,0]>=1)&(target[:,0]<i['vb'].shape[1]-2)&(target[:,1]>=1)&(target[:,1]<i['vb'].shape[0]-2)
target_valid &= target_sampling_inside
inside=(q[:,0]>=0)&(q[:,0]<=2047)&(target[:,0]>=0)&(target[:,0]<=2047)&(q[:,1]>=0)&(q[:,1]<=579)&(target[:,1]>=0)&(target[:,1]<=579)
assert np.all(source_available[ac]&target_available[ac]&source_valid[ac]&target_valid[ac]&inside[ac])
assert np.array_equal(source_valid,r['source_visible'][ix]);assert np.array_equal(target_valid,r['target_visible'][ix])
assert np.array_equal(ac,r['evidence_pass'][ix]&source_valid&target_valid)
height,width=i['rawA'].shape;edge_source=np.minimum(full[:,0],2047-full[:,0]);edge_target=np.minimum(target[:,0],2047-target[:,0]);vertical_target=np.minimum(target[:,1],height-1-target[:,1]);bins={}
for lo,hi in [(0,32),(32,128),(128,512),(512,1024),(1024,1536),(1536,1920),(1920,2016),(2016,2048)]:
 sel=(full[:,0]>=lo)&(full[:,0]<hi)
 bins[f'x{lo}_{hi}']=dict(grid=int(sel.sum()),accepted=int(np.sum(sel&ac)),gradient=int(np.sum(sel&ga)),accepted_FB=stats(r['fb'][ix][sel&ac]),accepted_NCC=stats(r['ncc'][ix][sel&ac]),source_edge_distance=stats(edge_source[sel&ac]),target_edge_distance=stats(edge_target[sel&ac]))
oldix=np.arange(200,200+int(b['grid_count']));oldq=b['query'][oldix]+bi['origin0']-i['origin0'];distance,mapping=cKDTree(q).query(oldq);assert np.all(distance==0)
change=np.linalg.norm(d[mapping]-b['disp'][oldix],axis=1);common=ac[mapping]&b['accepted'][oldix]
changed=[]
for j in np.flatnonzero(ac[mapping]!=b['accepted'][oldix]):
 k=mapping[j];changed.append(dict(previous_index=int(j),new_index=int(k),global_xy=full[k].tolist(),previous_accepted=bool(b['accepted'][oldix[j]]),new_accepted=bool(ac[k]),displacement_change=float(change[j])))
edge_records=[]
for j in np.flatnonzero(ac&(edge_source<16)):
 edge_records.append(dict(global_xy=full[j].tolist(),target_global=(target[j]+i['origin0']).tolist(),depth=float(r['depth'][ix][j]),ncc=float(r['ncc'][ix][j]),fb=float(r['fb'][ix][j]),spread=float(r['method_spread'][ix][j]),nearest_feature=float(r['nearest_feature'][ix][j]),feature_count25=int(r['feature_count25'][ix][j]),source_edge_distance=float(edge_source[j]),target_edge_distance=float(edge_target[j])))
excluded=np.flatnonzero(r['evidence_pass'][ix]&~(source_valid&target_valid));excluded_records=[dict(global_xy=full[j].tolist(),target_global=(target[j]+i['origin0']).tolist(),source_valid=bool(source_valid[j]),target_valid=bool(target_valid[j])) for j in excluded]
summary=dict(total_grid=N,accepted=int(ac.sum()),gradient_accepted=int(ga.sum()),accepted_sources_and_endpoints_observed=True,explicit_visibility_screen_recomputed_exact=True,visibility_excluded_grid=excluded_records,source_horizontal_edge_distance=stats(edge_source[ac]),target_horizontal_edge_distance=stats(edge_target[ac]),target_vertical_analysis_edge_distance=stats(vertical_target[ac]),horizontal_bins=bins,accepted_within16px_of_horizontal_edge=edge_records,previous_grid_count=int(len(oldix)),previous_grid_accepted=int(b['accepted'][oldix].sum()),previous_grid_now_accepted=int(ac[mapping].sum()),previous_grid_flag_changes=changed,previous_grid_displacement_change=stats(change),previous_grid_displacement_change_common_accepted=stats(change[common]),manual_displacement_max_change=float(np.max(np.linalg.norm(r['disp'][:200]-b['disp'][:200],axis=1))),manual_acceptance_changes=int(np.sum(r['accepted'][:200]!=b['accepted'][:200])),manual_used_to_select_vectors=False)
main=np.load(O/'main.npz')
assert np.array_equal(main['points'],q)
summary['accepted_grid_with_source_window_truncated_at_horizontal_edge']=int(np.sum(ac&(edge_source<float(main['radius']))))
summary['source_window_truncated_patch_condition']=stats(main['cond'][ac&(edge_source<float(main['radius']))])
summary['interior_patch_condition']=stats(main['cond'][ac&(edge_source>=float(main['radius']))])
(O/'field_edges_audit.json').write_text(json.dumps(summary,indent=2))
print(json.dumps({k:v for k,v in summary.items() if k not in ['accepted_within16px_of_horizontal_edge','previous_grid_flag_changes','visibility_excluded_grid']},indent=2));print('edge_count',len(edge_records),'previous_changed_flags',len(changed),'visibility_excluded',len(excluded))
