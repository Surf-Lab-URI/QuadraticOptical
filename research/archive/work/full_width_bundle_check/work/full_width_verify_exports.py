"""Check delivered MAT, NumPy and CSV values against the completed model."""
import csv,json
from pathlib import Path
import numpy as np
from scipy.io import loadmat
with np.load('work/full_width/results.npz') as stored:
    z={k:stored[k] for k in stored.files}
p=Path('outputs/full_width_conservative_');n=int(z['grid_count']);sl=slice(200,200+n)
m=loadmat(str(p)+'field.mat');a=np.load(str(p)+'field.npz');checks=[]
for key in a.files:
    v=a[key]
    if v.dtype.kind not in 'fibu':continue
    vv=np.asarray(m[key]);assert vv.size==v.size,key
    assert np.allclose(vv.reshape(v.shape),v,rtol=0,atol=0,equal_nan=True),key;checks.append(key)
u=a['grid_displacement_conservative_px_per_pair'];g=a['grid_d_dx_dx_conservative'];ok=z['accepted'][sl];gok=z['gradient_accepted'][sl]
assert np.all(np.isnan(u[~ok])) and np.all(np.isnan(g[~gok]))
assert np.allclose(u[ok],z['disp'][sl][ok],rtol=0,atol=0)
assert np.all(z['source_visible'][sl][ok]) and np.all(z['target_visible'][sl][ok])
rows=list(csv.DictReader(open(str(p)+'grid.csv')));assert len(rows)==n
for i,r in enumerate(rows):
    assert int(r['accepted'])==ok[i] and int(r['gradient_accepted'])==gok[i]
    for key,expected in [('x_full_zero_based_px',z['query_full'][sl][i,0]),('dx_raw_px_per_pair',z['disp'][sl][i,0]),
                         ('dx_conservative_px_per_pair',u[i,0]),('V_up_conservative_assumed_mps',-u[i,1]*float(z['DX'])/float(z['DT'])),
                         ('dU_dX_conservative_assumed_per_s',g[i]/float(z['DT']))]:
        assert np.allclose(float(r[key]),expected,rtol=0,atol=0,equal_nan=True),(i,key)
manual=list(csv.DictReader(open(str(p)+'manual.csv')));assert len(manual)==200
for i,r in enumerate(manual):
    assert float(r['endpoint_error_px'])==z['manual_error'][i]
    assert int(r['accepted'])==z['accepted'][i]
result=dict(mat_npz_numeric_arrays_identical=len(checks),grid_csv_rows=len(rows),manual_csv_rows=len(manual),accepted_vectors=int(ok.sum()),
    accepted_horizontal_gradients=int(gok.sum()),conservative_arrays_reject_with_nan=True,accepted_centers_visible=True,
    mat_npz_csv_units_signs_and_coordinates_verified=True,physical_units_confirmed=False,passed=True)
Path(str(p)+'export_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
