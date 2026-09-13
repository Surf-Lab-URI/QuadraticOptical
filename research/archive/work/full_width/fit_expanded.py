"""Extend the ORIGINAL estimator across image width, freezing prior models.

Run from project root. All fitting uses images and automatic particle tracks;
manual targets are read only in the final preservation/validation audit.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import json
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'work'))
from fit_local_cached import fit_local
from ptv_model import PTVModel
from field_model import LocalField

OUT = ROOT / 'work' / 'full_width'
PREVIOUS = ROOT / 'work' / 'large_frame'
STATS = ['ncc', 'rms', 'nvalid', 'cond', 'iterations', 'mindet', 'support_fraction']
VARIANTS = [('final_ptv13', 13, 2, 10), ('final_affine13', 13, 1, 10),
            ('final_ptv19', 19, 2, 10), ('final_ptv13_margin14', 13, 2, 14)]
OUTPUT_NAMES = dict(final_ptv13='main', final_affine13='affine',
                    final_ptv19='large_window', final_ptv13_margin14='margin14')


def normalize(raw, valid):
    v = valid.astype(float)
    smooth = gaussian_filter(raw*v, .65)/np.maximum(gaussian_filter(v, .65), 1e-5)
    mean = gaussian_filter(smooth*v, 7)/np.maximum(gaussian_filter(v, 7), 1e-5)
    hp = smooth-mean
    rms = np.sqrt(gaussian_filter(hp*hp*v, 9)/np.maximum(gaussian_filter(v, 9), 1e-5)+25)
    return np.clip(hp/rms, -3, 4)


def images(inputs, margin):
    if margin == 10:
        return tuple(inputs[k] for k in ['A', 'B', 'va', 'vb'])
    y = np.indices(inputs['A'].shape)[0]
    va = inputs['availability_a'] & (y >= inputs['surface_a'][None]+margin)
    vb = inputs['availability_b'] & (y >= inputs['surface_b'][None]+margin)
    return normalize(inputs['rawA'], va), normalize(inputs['rawB'], vb), va, vb


def previous_indices(inputs, previous):
    shift = previous['origin0']-inputs['origin0']
    dist, idx = cKDTree(inputs['points']).query(previous['points']+shift)
    assert np.all(dist == 0) and len(np.unique(idx)) == len(idx)
    return idx


def prepare_seeds(inputs, tracks):
    old = np.load(PREVIOUS/'main.npz')
    old_idx = previous_indices(inputs, old)
    new = np.ones(len(inputs['points']), bool)
    new[old_idx] = False
    query = inputs['points'][new]
    # The automatic-track model uses original scale/order/ridge settings.
    prior = PTVModel(str(tracks)).evaluate(query, scale=10, order=2)
    assert np.isfinite(prior['disp']).all()
    assert np.isfinite(prior['local_polynomial_gradient']).all()
    seeds = dict(points=query, new_indices=np.flatnonzero(new), old_indices=old_idx,
                 scale=np.array(10), source=str(tracks), **prior)
    np.savez_compressed(OUT/'seeds.npz', **seeds)
    return seeds


def forward(inputs, seeds, name, radius, order, margin, workers=4):
    old = np.load(PREVIOUS/(OUTPUT_NAMES[name]+'.npz'))
    points = inputs['points']
    old_idx = previous_indices(inputs, old)
    new_idx = np.asarray(seeds['new_indices'])
    assert np.array_equal(points[new_idx], seeds['points'])
    params = np.full((len(points), 2, 6 if order == 2 else 3), np.nan)
    params[old_idx] = old['params']
    stats = {k: np.full(len(points), np.nan) for k in STATS}
    for k in STATS:
        if k in old.files:
            stats[k][old_idx] = old[k]
    A, B, va, vb = images(inputs, margin)
    target_gradient = np.gradient(B)
    target_mask_float = vb.astype(float)
    start = time.time()
    def task(j):
        p = np.zeros((2, 3))
        p[:, 0] = seeds['disp'][j]
        p[:, 1:3] = seeds['local_polynomial_gradient'][j]*radius
        result = fit_local(A, B, va, vb, points[new_idx[j]], p[:, 0],
                           r=radius, order=order, reg=.00005,
                           maxiter=35, seed_affine=p, target_gradient=target_gradient,
                           target_mask_float=target_mask_float)
        if j % 250 == 0:
            print(name, j, '/', len(new_idx), 'seconds', round(time.time()-start, 1), flush=True)
        return result
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(task, range(len(new_idx))))
    for idx, (p, st) in zip(new_idx, results):
        params[idx] = p
        for k in STATS:
            stats[k][idx] = st.get(k, np.nan)
    frozen = np.zeros(len(points), bool)
    frozen[old_idx] = True
    original = np.zeros(len(points), bool)
    original[old_idx] = old['frozen_original']
    assert np.array_equal(np.isnan(params[old_idx]), np.isnan(old['params']))
    assert np.allclose(params[old_idx], old['params'], rtol=0, atol=0, equal_nan=True)
    np.savez_compressed(OUT/(OUTPUT_NAMES[name]+'.npz'), points=points, params=params,
        radius=radius, order=order, reg=.00005, margin=margin, sigma=.65,
        maskednorm=True, frozen_previous=frozen, frozen_original=original,
        previous_index=old_idx, original_index=old_idx[old['frozen_original']],
        origin0=inputs['origin0'], interpolation='bilinear; original fit_local',
        initialization='PTVModel scale10 order2; existing patches frozen', **stats)
    summary = dict(seconds=time.time()-start, frozen=int(frozen.sum()), fitted=len(new_idx),
                   finite=int(np.isfinite(params).all(axis=(1, 2)).sum()),
                   new_finite=int(np.isfinite(params[new_idx]).all(axis=(1, 2)).sum()))
    print(name, json.dumps(summary), flush=True)
    return summary


def reverse(inputs, workers=4):
    f = np.load(OUT/'main.npz')
    old = np.load(PREVIOUS/'reverse.npz')
    old_forward = np.load(PREVIOUS/'main.npz')
    old_idx = previous_indices(inputs, old_forward)
    shift = old_forward['origin0']-inputs['origin0']
    source_points = f['points']
    forward_params = f['params']
    destination = source_points+forward_params[:, :, 0]
    assert np.allclose(destination[old_idx], old['points']+shift, rtol=0, atol=1e-12)
    # Reverse polynomial centers are B destinations, never A source locations.
    params = np.full_like(forward_params, np.nan)
    params[old_idx] = old['params']
    stats = {k: np.full(len(params), np.nan) for k in STATS}
    for k in STATS:
        if k in old.files:
            stats[k][old_idx] = old[k]
    new = ~f['frozen_previous']
    new_idx = np.flatnonzero(new)
    A, B, va, vb = images(inputs, 10)
    target_gradient = np.gradient(A)
    target_mask_float = va.astype(float)
    start = time.time()
    def task(i):
        pp = forward_params[i]
        if not np.isfinite(pp).all():
            return np.full((2, 6), np.nan), {'ncc': -1}
        p = np.zeros((2, 3)); p[:, 0] = -pp[:, 0]
        try:
            p[:, 1:3] = (np.linalg.inv(np.eye(2)+pp[:, 1:3]/13)-np.eye(2))*13
        except np.linalg.LinAlgError:
            return np.full((2, 6), np.nan), {'ncc': -1}
        return fit_local(B, A, vb, va, destination[i], p[:, 0], r=13,
                         order=2, reg=.00005, maxiter=35, seed_affine=p,
                         target_gradient=target_gradient,target_mask_float=target_mask_float)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(task, new_idx))
    for idx, (p, st) in zip(new_idx, results):
        params[idx] = p
        for k in STATS:
            stats[k][idx] = st.get(k, np.nan)
    assert np.allclose(params[old_idx], old['params'], rtol=0, atol=0, equal_nan=True)
    fb = np.linalg.norm(params[:, :, 0]+forward_params[:, :, 0], axis=1)
    # Edge forward fits can fail because B leaves the image. Such fits have no
    # finite B center and cannot enter a spatial tree. Retain a record of their
    # absence, rather than invent a reverse center or a valid displacement.
    center_finite = np.isfinite(destination).all(axis=1)
    assert np.all(center_finite[old_idx])
    stats = {k: v[center_finite] for k, v in stats.items()}
    np.savez_compressed(OUT/'reverse.npz', points=destination[center_finite],
        source_points=source_points[center_finite], params=params[center_finite], radius=13, order=2,
        reg=.00005, margin=10, sigma=.65, maskednorm=True,
        frozen_previous=f['frozen_previous'][center_finite],
        frozen_original=f['frozen_original'][center_finite], fb_error=fb[center_finite],
        forward_index=np.flatnonzero(center_finite),
        missing_forward_indices=np.flatnonzero(~center_finite),
        origin0=inputs['origin0'], direction='B to A; polynomial centers in B', **stats)
    summary = dict(seconds=time.time()-start, frozen=len(old_idx), fitted=len(new_idx),
                   finite=int(np.isfinite(params).all(axis=(1, 2)).sum()),
                   omitted_nonfinite_B_centers=int(np.sum(~center_finite)),
                   patch_fb_percentiles=np.nanpercentile(fb, [0, 50, 90, 95, 100]).tolist())
    print('reverse', json.dumps(summary), flush=True)
    return summary


def audit(inputs):
    q = inputs['manual_xy']
    previous_main=np.load(PREVIOUS/'main.npz')
    shift=previous_main['origin0']-inputs['origin0']
    oldq = q-shift
    truth = inputs['manual_truth']
    depth = q[:, 1]-np.interp(q[:, 0], np.arange(len(inputs['surface_a'])), inputs['surface_a'])
    report = {}
    for name, _, _, _ in VARIANTS:
        new = LocalField(str(OUT/(OUTPUT_NAMES[name]+'.npz')))
        old = LocalField(str(PREVIOUS/(OUTPUT_NAMES[name]+'.npz')))
        d, g = new.evaluate(q); od, og = old.evaluate(oldq)
        change = np.linalg.norm(d-od, axis=1)
        ge = np.max(np.abs(g-og), axis=(1, 2))
        e = np.linalg.norm(d-truth, axis=1)
        rec = dict(max_displacement_change=float(np.nanmax(change)),
                   max_gradient_change=float(np.nanmax(ge)),
                   changed_manual_indices_one_based=(np.flatnonzero(change>1e-10)+1).tolist())
        for label, mask in [('all200', np.ones(len(q), bool)), ('depth_lt40', depth<40)]:
            ee=e[mask]
            rec[label] = dict(n=int(mask.sum()), finite=int(np.isfinite(ee).sum()),
                              mean=float(np.nanmean(ee)), median=float(np.nanmedian(ee)),
                              rms=float(np.sqrt(np.nanmean(ee*ee))))
        report[name]=rec
        np.savez_compressed(OUT/(name+'_manual_audit.npz'), points=q, displacement=d,
            gradient=g, previous_displacement=od, previous_gradient=og, error=e,
            displacement_change=change, gradient_change=ge, depth=depth)
    (OUT/'manual_preservation_audit.json').write_text(json.dumps(report, indent=2))
    print('manual audit', json.dumps(report), flush=True)
    # Blending can change near the old grid perimeter, even with coefficients
    # frozen. Evaluate all old grid nodes and an independent four-pixel grid.
    pg=previous_main['points']
    xx,yy=np.meshgrid(np.arange(pg[:,0].min(),pg[:,0].max()+1,4.),
                      np.arange(pg[:,1].min(),pg[:,1].max()+1,4.))
    probes=np.r_[pg,np.c_[xx.ravel(),yy.ravel()]]
    model=LocalField(str(OUT/'main.npz'))
    old_model=LocalField(str(PREVIOUS/'main.npz'))
    d,g=model.evaluate(probes+shift);od,og=old_model.evaluate(probes)
    added=model.points[~model.data['frozen_previous']]
    distance=cKDTree(added).query(probes+shift)[0]
    interior=distance>16+1e-9
    finite=np.isfinite(d).all(axis=1)&np.isfinite(od).all(axis=1)
    change=np.linalg.norm(d-od,axis=1)
    gchange=np.max(np.abs(g-og),axis=(1,2))
    assert not np.any(interior & np.isfinite(od).all(axis=1) & ~np.isfinite(d).all(axis=1))
    assert np.nanmax(change[interior&finite])<1e-9
    assert np.nanmax(gchange[interior&finite])<1e-9
    pr=dict(probes=len(probes),previous_grid_nodes=len(pg),
        common_finite=int(finite.sum()),
        finite_old_became_nonfinite=int(np.sum(np.isfinite(od).all(axis=1)&~np.isfinite(d).all(axis=1))),
        interior_count=int(np.sum(interior&finite)),
        max_interior_displacement_change=float(np.nanmax(change[interior&finite])),
        max_interior_gradient_change=float(np.nanmax(gchange[interior&finite])),
        changed_finite_probes=int(np.sum(change[finite]>1e-10)),
        max_finite_displacement_change=float(np.nanmax(change[finite])),
        max_finite_gradient_change=float(np.nanmax(gchange[finite])))
    np.savez_compressed(OUT/'previous_field_audit.npz',previous_query=probes,
        query=probes+shift,displacement=d,gradient=g,previous_displacement=od,
        previous_gradient=og,interior=interior,distance_to_added_node=distance,
        displacement_change=change,gradient_change=gchange)
    (OUT/'previous_field_audit.json').write_text(json.dumps(pr,indent=2))
    print('previous field audit',json.dumps(pr),flush=True)
    return report


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--tracks', type=Path, default=OUT/'ptv_tracks.npz')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--variants', nargs='*', default=[v[0] for v in VARIANTS])
    ap.add_argument('--reverse-only', action='store_true')
    ap.add_argument('--skip-reverse', action='store_true')
    ap.add_argument('--audit-only', action='store_true')
    args=ap.parse_args()
    inputs=np.load(OUT/'inputs.npz')
    if args.audit_only:
        audit(inputs); return
    report={}
    if not args.reverse_only:
        if not args.tracks.is_file():
            raise FileNotFoundError('Automatic tracks are not ready: '+str(args.tracks))
        seeds=prepare_seeds(inputs, args.tracks)
        for name, r, order, margin in VARIANTS:
            if name in args.variants:
                report[name]=forward(inputs, seeds, name, r, order, margin, args.workers)
    if not args.skip_reverse:
        report['reverse']=reverse(inputs, args.workers)
    (OUT/'fit_runtime.json').write_text(json.dumps(report, indent=2))
    if all((OUT/(OUTPUT_NAMES[v[0]]+'.npz')).is_file() for v in VARIANTS):
        audit(inputs)


if __name__ == '__main__':
    main()
