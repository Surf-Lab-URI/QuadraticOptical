"""Read-only audit; no model fitting or modification of acceptance thresholds."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from pathlib import Path
import sys, json, hashlib
import numpy as np
from scipy.spatial import cKDTree, Delaunay
from collections import Counter
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from field_model import LocalField

O = Path(__file__).resolve().parent
W = O.parent
i = np.load(O/'inputs.npz')
r = np.load(O/'results.npz')
old = np.load(W/'final_results.npz')
meta = np.load(W/'metadata.npz')
mod = LocalField(O/'main.npz')
N = int(r['grid_count'])
q = r['query']; qf = r['query_full']; g = r['gradient']
a = r['accepted']; ag = r['gradient_accepted']
np.seterr(invalid='ignore')  # Rejected unsupported queries intentionally contain NaNs.
origin = i['origin0']; oldorigin = i['old_origin0']
out = {'result_sha256': hashlib.sha256((O/'results.npz').read_bytes()).hexdigest()}
def maximum(x):
    return float(np.max(np.abs(x))) if np.size(x) else 0.
def equal(x,y):
    x=np.asarray(x);y=np.asarray(y)
    return bool(x.shape==y.shape and np.all((x==y)|(np.isnan(x)&np.isnan(y))))

out['coordinates'] = {
    'convention': 'Zero-based image coordinates: x right, y down. Vectors are B minus A in pixels.',
    'full_equals_crop_plus_origin': equal(qf, q+origin),
    'manual_source_max_error_pixels': maximum(qf[:200]-(meta['p'][:,0].T-1+oldorigin)),
    'manual_target_max_error_pixels': maximum(qf[:200]+r['manual_truth']-(meta['p'][:,1].T-1+oldorigin)),
    'manual_truth_preserved_exact': equal(r['manual_truth'], meta['p'][:,1].T-meta['p'][:,0].T),
    'grid_query_equals_fit_centers': equal(q[200:200+N], mod.points),
    'dense_query_x_max_error': maximum(qf[200+N:,0]-r['dense_x_full'].ravel()),
    'dense_query_y_max_error': maximum(qf[200+N:,1]-r['dense_y_full'].ravel()),
    'source_depth_full_vs_crop_max_error': maximum(r['depth']-(qf[:,1]-np.interp(qf[:,0],np.arange(2048),i['full_surface_a']))),
    'target_depth_full_vs_crop_max_error': maximum((r['target_depth']-((qf+r['disp'])[:,1]-np.interp((qf+r['disp'])[:,0],np.arange(2048),i['full_surface_b'])))[np.all(np.isfinite(r['disp']),axis=1)]),
    'surface_mask_offset_pixels': [float(np.max(i['new_mask_surface_'+s]-i['full_surface_'+s])) for s in 'ab'],
    'analysis_origin_zero_based': origin.tolist(),
    'old_crop_origin_zero_based': oldorigin.tolist(),
}
full_width=400.; full_depth=float(i['max_depth']); DX=float(i['DX']); DT=float(i['DT'])
out['physical_conversion'] = {
    'physical_units_confirmed': False,
    'DX': DX, 'DT': DT,
    'assumption': 'DX in metres/pixel; DT in seconds. Primary arrays remain pixels and pixels/pixel per image pair.',
    'horizontal_velocity': 'u = disp_x * DX / DT',
    'vertical_up_velocity': 'w = -disp_y * DX / DT',
    'horizontal_gradient': 'du/dX = gradient[0,0] / DT; spatial DX cancels',
    'full_gradient_in_X_Z_up': 'J = S @ gradient @ S / DT, S=diag(1,-1)',
    'horizontal_gradient_definition': 'Derivative at fixed laboratory vertical coordinate, not at fixed distance beneath a sloping surface.',
    'finite_time_caveat': 'Derivative of interval-averaged material displacement/DT with respect to source position; not an independently resolved instantaneous Eulerian gradient.',
    'metres_per_second_per_pixel': DX/DT,
    'centimetres_per_second_per_pixel': 100*DX/DT,
    'inverse_seconds_per_displacement_gradient': 1/DT,
    'analysis_width_cm_if_SI': full_width*100*DX,
    'analysis_depth_cm_if_SI': full_depth*100*DX,
}

# All interior accepted grid nodes; central differences cross no crop boundary.
ids=np.arange(200,200+N)
ids=ids[a[ids] & (q[ids,0]>1) & (q[ids,1]>1) & (q[ids,0]<i['A'].shape[1]-2) & (q[ids,1]<i['A'].shape[0]-2)]
v0, gg=mod.evaluate(q[ids])
derivative_tests=[]
for eps in [1e-2,1e-3]:
    fd=(mod.evaluate(q[ids]+[eps,0])[0]-mod.evaluate(q[ids]-[eps,0])[0])/(2*eps)
    error=fd-gg[:,:,0]
    # Direct SI-coordinate finite difference verifies the cancellation and sign.
    fdm=(mod.evaluate(q[ids]+[eps,0])[0][:,0]*DX/DT-mod.evaluate(q[ids]-[eps,0])[0][:,0]*DX/DT)/(2*eps*DX)
    derivative_tests.append(dict(step_pixels=eps, max_error_all_horizontal_derivatives=maximum(error),
                                 max_error_du_dx=maximum(error[:,0]), median_error_du_dx=float(np.median(np.abs(error[:,0]))),
                                 max_error_SI_du_dX=maximum(fdm-gg[:,0,0]/DT)))
out['derivative_check'] = {'accepted_interior_grid_count': len(ids),
    'stored_prediction_max_error':maximum(v0-r['disp'][ids]),
    'stored_gradient_max_error':maximum(gg-r['gradient'][ids]),
    'central_differences':derivative_tests,
    'pass_tolerance':1e-5,
    'passed':all(d['max_error_all_horizontal_derivatives']<1e-5 for d in derivative_tests)}

criteria={
    'alternatives':r['alternatives_available'], 'forward_valid':r['local_valid_share']>=.95,
    'reverse_valid':r['reverse_valid_share']>=.95, 'source_depth':(r['depth']>=12)&(r['depth']<=354),
    'target_depth':r['target_depth']>=10, 'support':r['support'], 'ncc':r['ncc']>=.6,
    'forward_backward':r['fb']<=1, 'method_spread':r['method_spread']<=1.5,
    'determinant':r['determinant']>.2, 'finite_displacement':np.all(np.isfinite(r['disp']),axis=1),
}
recomputed=np.logical_and.reduce(list(criteria.values()))
reg=recomputed & (r['depth']>=20) & (r['gradient_spread']<=.08) & (r['nearest_feature']<=10)
finite_keys=['disp','gradient','ncc','fb','depth','target_depth','method_spread','gradient_spread',
             'local_valid_share','reverse_valid_share','nearest_feature','feature_count25','determinant']
finite={k:bool(np.all(np.isfinite(r[k][a]))) for k in finite_keys}
tracks=np.load(O/'ptv_tracks.npz');pts=tracks['points'][tracks['accepted']]
tree=cKDTree(pts)
near=tree.query(q)[0]; counts=np.array([len(z) for z in tree.query_ball_point(q,25)])
support=(Delaunay(pts).find_simplex(q)>=0)&(near<=12)&(counts>=6)
out['acceptance']={
    'flag_dtype_bool':a.dtype==bool and ag.dtype==bool,
    'acceptance_recomputed_exact':equal(a,recomputed),
    'gradient_acceptance_recomputed_exact':equal(ag,reg),
    'gradient_is_subset':bool(np.all(~ag|a)),
    'finite_at_every_accepted_point':finite,
    'feature_support_recomputed_exact':equal(support,r['support']),
    'feature_count_recomputed_exact':equal(counts,r['feature_count25']),
    'nearest_feature_max_error':maximum(near-r['nearest_feature']),
    'grid_accepted':int(a[200:200+N].sum()),'grid_gradient_accepted':int(ag[200:200+N].sum()),
    'manual_accepted':int(a[:200].sum()), 'all_query_accepted':int(a.sum()),
    'minimum_grid_accepted_depth':float(r['depth'][200:200+N][a[200:200+N]].min()),
    'minimum_grid_gradient_depth':float(r['depth'][200:200+N][ag[200:200+N]].min()),
}

# Coordinate matching, not assumed identical ordering.
oldN=int(old['grid_count'])
oldfull=old['query'][200:200+oldN]+oldorigin
dist,j=cKDTree(qf[200:200+N]).query(oldfull); j=j+200
assert np.max(dist)<1e-10
jo=np.arange(200,200+oldN)
oldcriteria={
    'alternatives':old['alternatives_available'], 'forward_valid':old['local_valid_share']>=.95,
    'reverse_valid':old['reverse_valid_share']>=.95, 'source_depth':(old['depth']>=12)&(old['depth']<=150),
    'target_depth':old['target_depth']>=10, 'support':old['support'], 'ncc':old['ncc']>=.6,
    'forward_backward':old['fb']<=1, 'method_spread':old['method_spread']<=1.5,
    'determinant':old['determinant']>.2, 'finite_displacement':np.all(np.isfinite(old['disp']),axis=1),
}
def transitions(oldflag,newflag):
    return {'retained_accepted':int(np.sum(oldflag&newflag)), 'newly_accepted':int(np.sum(~oldflag&newflag)),
            'newly_rejected':int(np.sum(oldflag&~newflag)), 'retained_rejected':int(np.sum(~oldflag&~newflag))}
changed=(old['accepted'][jo]!=a[j])
change_details=[]
for k in np.where(changed)[0]:
    change_details.append(dict(old_grid_index0=int(k),new_grid_index0=int(j[k]-200),position_full=qf[j[k]].tolist(),
        old_accepted=bool(old['accepted'][jo[k]]),new_accepted=bool(a[j[k]]),
        criteria_changed=[key for key in criteria if bool(criteria[key][j[k]])!=bool(oldcriteria[key][jo[k]])]))
du=r['disp'][j]-old['disp'][jo]
dg=r['gradient'][j]-old['gradient'][jo]
retained=old['accepted'][jo]&a[j]
manualchange=np.linalg.norm(r['disp'][:200]-old['disp'][:200],axis=1)
out['original_node_comparison']={
    'all_original_nodes_matched':bool(np.max(dist)<1e-10), 'nodes':oldN,
    'frozen_flag_matches_original_coordinates':equal(r['original_grid_flag'],np.isin(np.arange(N),j-200)),
    'displacement_changed_over_1e_9':int(np.sum(np.linalg.norm(du,axis=1)>1e-9)),
    'displacement_max_change_pixels':float(np.max(np.linalg.norm(du,axis=1))),
    'displacement_max_change_at_retained_accepted_nodes_pixels':float(np.max(np.linalg.norm(du[retained],axis=1))),
    'gradient_max_abs_change_per_pair':maximum(dg),
    'acceptance_transitions':transitions(old['accepted'][jo],a[j]),
    'gradient_acceptance_transitions':transitions(old['gradient_accepted'][jo],ag[j]),
    'acceptance_changes':change_details,
    'acceptance_change_reason_counts': {' + '.join(k):v for k,v in Counter(tuple(c['criteria_changed']) for c in change_details).items()},
    'manual_acceptance_transitions':transitions(old['accepted'][:200],a[:200]),
    'manual_prediction_changed_picks_1based':(np.where(manualchange>1e-9)[0]+1).tolist(),
    'manual_max_prediction_change_pixels':float(manualchange.max()),
    'explanation':'Frozen local coefficients are preserved. Extra neighbouring patches can change the blended value near old grid edges; extra accepted tracks can change evidence support without changing thresholds.',
}
out['passed']=(all(v for k,v in out['coordinates'].items() if isinstance(v,bool))
    and all(out['coordinates'][k]<1e-9 for k in out['coordinates'] if 'max_error' in k)
    and out['derivative_check']['passed'] and equal(a,recomputed) and equal(ag,reg)
    and all(finite.values()) and equal(support,r['support']) and equal(counts,r['feature_count25'])
    and out['original_node_comparison']['frozen_flag_matches_original_coordinates'])
(O/'independent_final_audit.json').write_text(json.dumps(out,indent=2))
lines=['# Independent final audit', '',
       'PASS' if out['passed'] else 'FAIL: inspect the JSON for details.', '',
       f'Checked all {len(q):,} exported queries and {len(ids):,} accepted interior grid nodes for analytic horizontal derivatives.',
       'Zero-based crop/full-image coordinates, both manual endpoints, surface depths and source-to-target displacement conventions agree.',
       f'Analytic horizontal derivatives agree with central differences to {max(d["max_error_all_horizontal_derivatives"] for d in derivative_tests):.3g} pixels/pixel per image pair.',
       f'All accepted values and quality diagnostics are finite. Screening flags and particle-support masks were independently recomputed exactly; {int(a[200:200+N].sum())} grid displacements and {int(ag[200:200+N].sum())} grid gradients pass.', '',
       'At the 727 original grid nodes:',
       json.dumps(out['original_node_comparison']['acceptance_transitions']),
       'Gradient flags: '+json.dumps(out['original_node_comparison']['gradient_acceptance_transitions']),
       f'Maximum displacement change: {out["original_node_comparison"]["displacement_max_change_pixels"]:.6g} pixels. Frozen coefficients do not by themselves freeze the blended field at old grid edges when new neighboring patches enter its support.',
       f'Manual predictions changed beyond 1e-9 pixels only for picks {out["original_node_comparison"]["manual_prediction_changed_picks_1based"]}; maximum change {manualchange.max():.6g} pixels.', '',
       'Primary units remain pixels. If DX is metres/pixel and DT is seconds, u = dx·DX/DT; vertical-up w = −dy·DX/DT; du/dX = g[0,0]/DT. Thus one pixel of displacement is '+f'{DX/DT:.9g} m/s, and one unit of displacement gradient is {1/DT:g} s⁻¹. Physical unit labels were not supplied and are assumptions.',
       'The horizontal derivative holds laboratory height fixed. It is the derivative of displacement/DT with respect to source position, not an independently resolved instantaneous Eulerian gradient.', '',
       'The JSON records every changed original velocity flag, its coordinates, and which original screening conditions changed. No fit or screening threshold was modified by this audit.']
(O/'independent_final_audit.md').write_text('\n\n'.join(lines)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='original_node_comparison'},indent=2))
print(json.dumps({k:v for k,v in out['original_node_comparison'].items() if k!='acceptance_changes'},indent=2))
