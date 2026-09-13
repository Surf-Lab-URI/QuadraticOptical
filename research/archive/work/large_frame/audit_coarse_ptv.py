import numpy as np,json
from pathlib import Path
O=Path('work/large_frame');i=np.load(O/'inputs.npz');t=np.load(O/'ptv_tracks.npz');b=np.load('work/ptv_tracks.npz');c=np.load(O/'coarse_affine.npz');bc=np.load('work/propagate0.npz');shift=i['old_origin0']-i['origin0'];n=len(b['points']);rows=[]
for k in b.files:
 ref=b[k]+shift if k=='points' else b[k];actual=t[k][:n]
 assert np.array_equal(actual,ref),k
 rows.append(k)
new=np.arange(n,len(t['points']));acc=(t['ncc'][new]>.68)&(t['back_ncc'][new]>.68)&(t['fb'][new]<1)&(t['ambiguity_gap'][new]>.015)&(t['back_gap'][new]>.01)
assert np.array_equal(acc,t['accepted'][new]);assert np.allclose(t['fb'][new],np.linalg.norm(t['disp'][new]+t['back_disp'][new],axis=1))
assert np.all(t['original_candidate'][:n]) and not np.any(t['original_candidate'][n:]);assert np.array_equal(t['original_accepted_track'],t['original_candidate']&t['accepted'])
for j in np.flatnonzero(c['original_coarse']):
 oldindex=int(c['original_coarse_index'][j]);assert np.array_equal(c['points'][j],bc['points'][oldindex]+shift);assert np.array_equal(c['params'][j],bc['params'][oldindex])
res=dict(original_rows=n,original_fields_exact=rows,original_accepted=int(sum(t['accepted'][:n])),new_rows=len(new),new_accepted=int(sum(acc)),coarse_old_exact=int(sum(c['original_coarse'])),new_acceptance_recomputed_exact=True,coordinate_origin0=i['origin0'].tolist(),bins={})
for lo,hi in [(14,20),(20,40),(40,80),(80,150),(150,250),(250,379.001)]:
 d=t['source_depth'][new];s=(d>=lo)&(d<hi);keep=s&acc;res['bins'][str(lo)+'to'+str(hi)]=dict(new_candidates=int(sum(s)),new_accepted=int(sum(keep)),median_fb=float(np.median(t['fb'][new][keep])) if keep.any() else None)
(O/'coarse_ptv_audit.json').write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))
