"""Read-only audit of fresh symmetric bootstrap candidates and sensitivity."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import argparse,hashlib,json
from pathlib import Path
import numpy as np

BASE=Path(__file__).resolve().parent


def read(path,keys=None):
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k] for k in (z.files if keys is None else keys)}


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def run(pair):
    folder=BASE/str(pair);path=folder/'coarse_search.npz'
    if not path.exists():return dict(pair=pair,status='pending')
    inp=read(folder/'inputs.npz',['points','surface_a','origin0','requested_max_depth_px',
         'fitting_max_depth_px','detector_max_depth_px','image_only','supplied_velocity_used','A'])
    z=read(path);p=z['points'];n=len(p)
    assert np.array_equal(p,inp['points'])
    assert bool(inp['image_only']) and not bool(inp['supplied_velocity_used'])
    assert bool(z['image_only']) and not bool(z['supplied_velocity_used'])
    assert not bool(z['previous_models_or_tracks_used'])
    assert str(z['input_sha256'])==sha(folder/'inputs.npz')
    h,w=inp['A'].shape
    # Independently reconstruct the symmetric window limits, including the
    # separate image-boundary restriction on the source rectangle.
    center=np.round(p).astype(int)
    low=np.maximum(center-19,0)
    high=np.minimum(center+20,[w,h])
    expected=np.c_[np.maximum(-64,-low[:,0]),np.minimum(64,w-high[:,0]),
                   np.maximum(-64,-low[:,1]),np.minimum(64,h-high[:,1])]
    assert np.array_equal(expected,z['search_bounds'])
    seeds=z['candidate_seeds'];seed_valid=np.isfinite(seeds).all(axis=2)
    assert np.array_equal(seed_valid,z['candidate_seed_valid'])
    assert np.array_equal(seeds[:,6],np.zeros((n,2)))
    assert np.all(np.abs(seeds[:,:3][seed_valid[:,:3]])<=64)
    assert np.all(np.abs(seeds[:,3:6][seed_valid[:,3:6]])<=48)
    assert np.all(z['candidate_seed_common_count'][:,:6][seed_valid[:,:6]]>=
                  np.broadcast_to(z['search_required_common_pixels'][:,None],(n,6))[seed_valid[:,:6]]-1e-8)
    assert np.all(np.isnan(z['candidate_params'][~seed_valid]))
    assert np.all(z['candidate_ncc'][~seed_valid]==-1)
    with np.errstate(invalid='ignore'):
        admissible=(np.isfinite(z['candidate_params']).all(axis=(2,3))&
            np.isfinite(z['candidate_ncc'])&(z['candidate_mindet']>.15)&
            (z['candidate_support_fraction']>.7))
    assert np.array_equal(admissible,z['candidate_admissible'])
    selected=[];selected48=[]
    for row in range(n):
        unique=len(set(map(tuple,seeds[row,seed_valid[row]])))
        assert unique==z['unique_seed_fit_count'][row]
        for group,prefix,chosen in [([0,1,2,6],'',selected),([3,4,5,6],'bootstrap48_',selected48)]:
            eligible=[i for i in group if admissible[row,i]]
            success=bool(eligible)
            if not eligible:
                eligible=[i for i in group if np.isfinite(z['candidate_params'][row,i]).all() and
                          np.isfinite(z['candidate_ncc'][row,i])]
            pick=max(eligible,key=lambda i:z['candidate_ncc'][row,i]) if eligible else group[-1]
            chosen.append(pick)
            assert pick==z[prefix+'selected_candidate'][row]
            assert np.allclose(z[prefix+'params'][row],z['candidate_params'][row,pick],rtol=0,atol=0,equal_nan=True)
            expected_ncc=z['candidate_ncc'][row,pick] if success else -1.
            assert z[prefix+'ncc'][row]==expected_ncc
    depth=p[:,1]-np.interp(p[:,0],np.arange(len(inp['surface_a'])),inp['surface_a'])
    good=(z['ncc']>.7)&np.isfinite(z['params']).all(axis=(1,2))
    good48=(z['bootstrap48_ncc']>.7)&np.isfinite(z['bootstrap48_params']).all(axis=(1,2))
    shared=good&good48
    delta=np.linalg.norm(z['params'][:,:,0]-z['bootstrap48_params'][:,:,0],axis=1)
    dg=np.max(np.abs(z['params'][:,:,1:]-z['bootstrap48_params'][:,:,1:]),axis=(1,2))/19
    assert np.allclose(delta,z['bootstrap_displacement_disagreement'],rtol=0,atol=1e-12,equal_nan=True)
    assert np.allclose(dg,z['bootstrap_gradient_disagreement'],rtol=0,atol=0,equal_nan=True)
    report_region=depth<=float(inp['requested_max_depth_px'])
    material=shared&(delta>1.)
    near=(depth<40)&good
    out=dict(pair=pair,status='passed',all_candidate_selection_checks_pass=True,
        input_hash_matches=True,no_velocity_inputs_or_old_fields=True,
        coarse_nodes=n,usable_primary=int(good.sum()),usable_bootstrap48=int(good48.sum()),
        common_usable=int(shared.sum()),wide_only_usable=int((good&~good48).sum()),
        narrow_only_usable=int((good48&~good).sum()),
        mean_common_difference_px=float(np.mean(delta[shared])) if shared.any() else None,
        max_common_difference_px=float(np.max(delta[shared])) if shared.any() else None,
        common_difference_over_point1_px=int(np.sum(shared&(delta>.1))),
        common_difference_over1_px=int(material.sum()),
        within_requested_cm_difference_over1_px=int(np.sum(material&report_region)),
        usable_shallower40=int(near.sum()),shallow_difference_over1_px=int(np.sum(material&(depth<40))),
        primary_seed_on_artificial64_limit=int(np.sum(good&z['selected_seed_on_search_limit'])),
        primary_center_translation_beyond64=int(np.sum(good&z['selected_translation_beyond_search'])),
        competing_high_ncc_basins=int(np.sum(good&z['bootstrap_ambiguous'])),
        candidate_fit_count=int(z['unique_seed_fit_count'].sum()),
        invalid_peak_slots_never_fitted=int(np.sum(~seed_valid[:,:6])),
        all_source_bounds_and_candidate_support_verified=True,
        bootstrap_note='Coarse alternative only; independently tracking/fitting bootstrap48 would test final-field sensitivity.',
        material_disagreements=[dict(index0=int(i),x=float(p[i,0]),y=float(p[i,1]),
            depth=float(depth[i]),difference=float(delta[i]),wide=z['params'][i,:,0].tolist(),
            narrow=z['bootstrap48_params'][i,:,0].tolist(),ncc64=float(z['ncc'][i]),
            ncc48=float(z['bootstrap48_ncc'][i])) for i in np.flatnonzero(material)])
    (folder/'coarse_bootstrap_audit.json').write_text(json.dumps(out,indent=2)+'\n')
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pair',type=int,choices=[80,100],required=True)
    a=p.parse_args();r=run(a.pair);print(json.dumps({k:v for k,v in r.items() if k!='material_disagreements'},indent=2))
