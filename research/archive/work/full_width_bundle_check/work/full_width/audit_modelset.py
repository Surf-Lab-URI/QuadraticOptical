"""Independent numerical checks of the full-width frozen patch models."""
import json
from pathlib import Path
import numpy as np
W=Path(__file__).resolve().parent
P=W.parent/'large_frame'
inp=np.load(W/'inputs.npz')
report={}
for name in ['main','affine','large_window','margin14']:
    f=np.load(W/(name+'.npz'));p=np.load(P/(name+'.npz'))
    idx=f['previous_index'];shift=p['origin0']-inp['origin0']
    assert len(idx)==2184
    assert np.array_equal(f['points'][idx],p['points']+shift)
    assert np.allclose(f['params'][idx],p['params'],rtol=0,atol=0,equal_nan=True)
    assert np.array_equal(f['frozen_original'][idx],p['frozen_original'])
    assert int(f['frozen_previous'].sum())==2184
    assert int(f['frozen_original'].sum())==727
    for key in ['ncc','rms','nvalid','cond','iterations','mindet','support_fraction']:
        assert np.allclose(f[key][idx],p[key],rtol=0,atol=0,equal_nan=True),key
    report[name]=dict(previous_points_and_coefficients_exact=True,
        previous_statistics_exact=True,frozen_previous=2184,frozen_original=727,
        total_nodes=len(f['points']),new_finite=int(np.isfinite(f['params'][~f['frozen_previous']]).all(axis=(1,2)).sum()),
        failed_total=int(np.sum(~np.isfinite(f['params']).all(axis=(1,2)))),
        radius=float(f['radius']),order=int(f['order']),reg=float(f['reg']),margin=int(f['margin']))
f=np.load(W/'main.npz');r=np.load(W/'reverse.npz');old=np.load(P/'reverse.npz')
idx=np.searchsorted(r['forward_index'],f['previous_index'])
assert np.array_equal(r['forward_index'][idx],f['previous_index'])
assert np.allclose(r['params'][idx],old['params'],rtol=0,atol=0,equal_nan=True)
assert np.allclose(r['points'][idx],old['points']+old['origin0']-inp['origin0'],rtol=0,atol=1e-12)
assert np.isfinite(r['points']).all()
assert np.array_equal(r['source_points'],f['points'][r['forward_index']])
endpoint=f['points'][r['forward_index']]+f['params'][r['forward_index'], :, 0]
assert np.array_equal(r['points'],endpoint)
for key in ['ncc','rms','nvalid','cond','iterations','mindet','support_fraction']:
    assert np.allclose(r[key][idx],old[key],rtol=0,atol=0,equal_nan=True),key
assert np.array_equal(np.sort(np.r_[r['forward_index'],r['missing_forward_indices']]),np.arange(len(f['points'])))
report['reverse']=dict(previous_coefficients_and_statistics_exact=True,
    B_centered_endpoints_exact=True,total_finite_centers=len(r['points']),
    failed_params=int(np.sum(~np.isfinite(r['params']).all(axis=(1,2)))),
    explicitly_missing_centers=len(r['missing_forward_indices']),
    frozen_previous=int(r['frozen_previous'].sum()),frozen_original=int(r['frozen_original'].sum()))
assert report['reverse']['frozen_previous']==2184 and report['reverse']['frozen_original']==727
report['passed']=True
(W/'model_independent_audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
