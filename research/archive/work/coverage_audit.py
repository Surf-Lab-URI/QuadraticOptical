import numpy as np,json
from scipy.spatial import Delaunay,cKDTree
from pathlib import Path
z=np.load('work/final_results.npz');q=z['query'];tr=np.load('work/ptv_tracks.npz');features=tr['points'][tr['accepted']];hull=Delaunay(features)
crit={
'finite_displacement':np.all(np.isfinite(z['disp']),axis=1),
'source_depth_min12':z['depth']>=12,'source_depth_max150':z['depth']<=150,
'mapped_depth_min10':z['target_depth']>=10,
'inside_particle_hull':hull.find_simplex(q)>=0,
'nearest_particle_max12':z['nearest_feature']<=12,
'particle_count25_min6':z['feature_count25']>=6,
'ncc_min0.60':z['ncc']>=.6,
'forward_backward_max1':z['fb']<=1,
'method_spread_max1.5':z['method_spread']<=1.5,
'blended_determinant_min0.2':z['determinant']>.2,
'all_three_alternatives':z['alternatives_available'],
'forward_local_valid_share_min0.95':z['local_valid_share']>=.95,
'reverse_local_valid_share_min0.95':z['reverse_valid_share']>=.95,
}
assert np.array_equal(np.logical_and.reduce(list(crit.values())),z['accepted'])
ng=int(z['grid_count']);groups={'manual':np.arange(200),'grid':np.arange(200,200+ng)}
bands={'lt20':lambda d:d<20,'20to40':lambda d:(d>=20)&(d<40),'40plus':lambda d:d>=40,'all':lambda d:np.ones(len(d),bool)}
geom=['finite_displacement','source_depth_min12','source_depth_max150','mapped_depth_min10','inside_particle_hull','nearest_particle_max12','particle_count25_min6']
emp=['ncc_min0.60','forward_backward_max1','method_spread_max1.5','blended_determinant_min0.2']
local=['forward_local_valid_share_min0.95','reverse_local_valid_share_min0.95']
def conjunction(keys):return np.logical_and.reduce([crit[k] for k in keys])
geompass=conjunction(geom); emppass=conjunction(emp);localpass=conjunction(local);altpass=crit['all_three_alternatives']
base=geompass&emppass
# These are diagnostic provisional displays, not new data or confidence-calibrated estimates.
# Strict image-screen provisional retains every previous criterion except local whole-window validity shares.
provisional_window_only=base&altpass
# Requiring velocity alternatives only removes the derivative-specific sensitivity prerequisite for velocity.
velocityalts=np.all(np.isfinite(z['alternative_displacements']),axis=(0,2))
provisional_velocityonly=base&velocityalts
# Two finite alternatives is useful only as a separately labelled weaker diagnostic tier.
number_velocityalts=np.sum(np.all(np.isfinite(z['alternative_displacements']),axis=2),axis=0)
provisional_twoalts=base&(number_velocityalts>=2)
out={'definitions':{'depth_bands':'source vertical distance below provided surface; exact depths have ~1 pixel bookkeeping ambiguity','failure_counts':'overlapping criterion/group failures; exclusive groups prioritize geometry then image tests then alternative availability then local-window','provisional_window_only':'Original acceptance except waive 95% forward/reverse good whole-window contribution; all image evidence, all alternatives and support geometry unchanged. Explicitly provisional; no manual discrepancy used.','provisional_velocity_only':'As above but require finite displacement alternatives, omit gradient-alternative requirement for velocity.','provisional_twoalts':'As above with at least two finite displacement alternatives; weaker sensitivity evidence.'},'groups':{}}
for name,ids in groups.items():
 out['groups'][name]={}
 for band,sel in bands.items():
  ii=ids[sel(z['depth'][ids])]
  o={'total':len(ii),'accepted':int(z['accepted'][ii].sum()),'excluded':int((~z['accepted'][ii]).sum()),'criteria_failed':{k:int((~v[ii]).sum()) for k,v in crit.items()},'overlapping_group_failed':{'geometry':int((~geompass[ii]).sum()),'image_tests':int((~emppass[ii]).sum()),'alternatives':int((~altpass[ii]).sum()),'local_windows':int((~localpass[ii]).sum())},'exclusive_failure_groups':{'geometry':int((~geompass[ii]).sum()),'image_tests_given_geometry':int((geompass&~emppass)[ii].sum()),'alternatives_given_prior_pass':int((base&~altpass)[ii].sum()),'local_windows_given_all_else_pass':int((base&altpass&~localpass)[ii].sum())}}
  for tier,mask in [('provisional_window_only',provisional_window_only),('provisional_velocity_only',provisional_velocityonly),('provisional_twoalts',provisional_twoalts)]:
   new=mask&~z['accepted'];v=ii[mask[ii]];n=ii[new[ii]]
   o[tier]={'total_including_original':len(v),'additional':len(n),'min_depth':float(np.min(z['depth'][v])) if len(v) else None}
   if name=='manual':
    o[tier]['additional_picks_one_based']=(n+1).tolist();o[tier]['posthoc_mean_epe']=float(np.mean(z['manual_error'][v])) if len(v) else None;o[tier]['posthoc_additional_mean_epe']=float(np.mean(z['manual_error'][n])) if len(n) else None
  if name=='manual':
   o['posthoc_excluded_mean_epe']=float(np.mean(z['manual_error'][ii[~z['accepted'][ii]]])) if np.any(~z['accepted'][ii]) else None
  out['groups'][name][band]=o
out['closest_manual']=[]
for i in np.argsort(z['depth'][:200])[:12]:
 o={'pick':int(i+1),'point':q[i].tolist(),'disp':z['disp'][i].tolist(),'truth':z['manual_truth'][i].tolist(),'manual_epe':float(z['manual_error'][i]),'failed':[k for k,v in crit.items() if not v[i]]}
 for key in ['depth','target_depth','ncc','fb','method_spread','determinant','nearest_feature','feature_count25','local_valid_share','reverse_valid_share','alternatives_available']:o[key]=z[key][i].item()
 o['local_contributions']={}
 for label,path,pt in [('forward','work/final_ptv13.npz',q[i]),('reverse','work/final_reverse_field.npz',q[i]+z['disp'][i])]:
  f=np.load(path);pp=f['points'];tt=cKDTree(pp);ii=np.asarray(tt.query_ball_point(pt,16));dist=np.linalg.norm(pt-pp[ii],axis=1)/16;ww=(1-dist)**4*(1+4*dist);ww/=sum(ww)
  bad_jac=f['mindet'][ii]<=.05;bad_support=f['support_fraction'][ii]<=.85
  o['local_contributions'][label]={'bad_jacobian_weight':float(ww[bad_jac].sum()),'bad_support_weight':float(ww[bad_support].sum()),'both_bad_weight':float(ww[bad_jac&bad_support].sum()),'contributors':[dict(grid_index=int(j),point=pp[j].tolist(),weight=float(w),min_jacobian=float(f['mindet'][j]),support_fraction=float(f['support_fraction'][j]),ncc=float(f['ncc'][j])) for j,w in zip(ii,ww) if w>.005]}
 out['closest_manual'].append(o)
# Track diagnostic counts by the same source bands.
td=tr['points'][:,1]-np.interp(tr['points'][:,0],np.arange(501),z['surface_a']);out['particle_tracks']={}
for band,sel in bands.items():
 mask=sel(td);acc=mask&tr['accepted'];out['particle_tracks'][band]={'candidates':int(mask.sum()),'accepted':int(acc.sum()),'min_accepted_depth':float(np.min(td[acc])) if np.any(acc) else None}
# Additional-only nodes serialized for root to re-use.
out['provisional_indices']={'window_only_additional':np.flatnonzero(provisional_window_only&~z['accepted']).tolist(),'velocity_only_additional':np.flatnonzero(provisional_velocityonly&~z['accepted']).tolist(),'two_alternatives_additional':np.flatnonzero(provisional_twoalts&~z['accepted']).tolist()}
Path('work/coverage_audit.json').write_text(json.dumps(out,indent=2))
print(json.dumps({'groups':out['groups'],'closest_two':out['closest_manual'][:2],'tracks':out['particle_tracks']},indent=2))
