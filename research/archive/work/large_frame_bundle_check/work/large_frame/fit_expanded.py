"""Extend the ORIGINAL bilinear quadratic estimator without refitting old nodes.

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
from extended_flow import fit_local
from ptv_model import PTVModel
from field_model import LocalField

OUT = ROOT / 'work' / 'large_frame'
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


def original_indices(inputs, original):
    shift = inputs['old_origin0']-inputs['origin0']
    dist, idx = cKDTree(inputs['points']).query(original['points']+shift)
    assert np.all(dist == 0) and len(np.unique(idx)) == len(idx)
    return idx


def prepare_seeds(inputs, tracks):
    old = np.load(ROOT/'work/final_ptv13.npz')
    old_idx = original_indices(inputs, old)
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
    old = np.load(ROOT/'work'/ (name+'.npz'))
    points = inputs['points']
    old_idx = original_indices(inputs, old)
    new_idx = np.asarray(seeds['new_indices'])
    assert np.array_equal(points[new_idx], seeds['points'])
    params = np.full((len(points), 2, 6 if order == 2 else 3), np.nan)
    params[old_idx] = old['params']
    stats = {k: np.full(len(points), np.nan) for k in STATS}
    for k in STATS:
        if k in old.files:
            stats[k][old_idx] = old[k]
    A, B, va, vb = images(inputs, margin)
    start = time.time()
    def task(j):
        p = np.zeros((2, 3))
        p[:, 0] = seeds['disp'][j]
        p[:, 1:3] = seeds['local_polynomial_gradient'][j]*radius
        result = fit_local(A, B, va, vb, points[new_idx[j]], p[:, 0],
                           r=radius, order=order, reg=.00005,
                           maxiter=35, seed_affine=p)
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
    assert np.array_equal(np.isnan(params[old_idx]), np.isnan(old['params']))
    assert np.allclose(params[old_idx], old['params'], rtol=0, atol=0, equal_nan=True)
    np.savez_compressed(OUT/(OUTPUT_NAMES[name]+'.npz'), points=points, params=params,
        radius=radius, order=order, reg=.00005, margin=margin, sigma=.65,
        maskednorm=True, frozen_original=frozen, original_index=old_idx,
        origin0=inputs['origin0'], interpolation='bilinear; original fit_local',
        initialization='PTVModel scale10 order2; existing patches frozen', **stats)
    summary = dict(seconds=time.time()-start, frozen=int(frozen.sum()), fitted=len(new_idx),
                   finite=int(np.isfinite(params).all(axis=(1, 2)).sum()),
                   new_finite=int(np.isfinite(params[new_idx]).all(axis=(1, 2)).sum()))
    print(name, json.dumps(summary), flush=True)
    return summary


def reverse(inputs, workers=4):
    f = np.load(OUT/'main.npz')
    old = np.load(ROOT/'work/final_reverse_field.npz')
    old_forward = np.load(ROOT/'work/final_ptv13.npz')
    old_idx = original_indices(inputs, old_forward)
    shift = inputs['old_origin0']-inputs['origin0']
    source_points = f['points']
    destination = source_points+f['params'][:, :, 0]
    assert np.allclose(destination[old_idx], old['points']+shift, rtol=0, atol=1e-12)
    # Reverse polynomial centers are B destinations, never A source locations.
    params = np.full_like(f['params'], np.nan)
    params[old_idx] = old['params']
    stats = {k: np.full(len(params), np.nan) for k in STATS}
    for k in STATS:
        if k in old.files:
            stats[k][old_idx] = old[k]
    new = ~f['frozen_original']
    new_idx = np.flatnonzero(new)
    A, B, va, vb = images(inputs, 10)
    start = time.time()
    def task(i):
        pp = f['params'][i]
        if not np.isfinite(pp).all():
            return np.full((2, 6), np.nan), {'ncc': -1}
        p = np.zeros((2, 3)); p[:, 0] = -pp[:, 0]
        try:
            p[:, 1:3] = (np.linalg.inv(np.eye(2)+pp[:, 1:3]/13)-np.eye(2))*13
        except np.linalg.LinAlgError:
            return np.full((2, 6), np.nan), {'ncc': -1}
        return fit_local(B, A, vb, va, destination[i], p[:, 0], r=13,
                         order=2, reg=.00005, maxiter=35, seed_affine=p)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(task, new_idx))
    for idx, (p, st) in zip(new_idx, results):
        params[idx] = p
        for k in STATS:
            stats[k][idx] = st.get(k, np.nan)
    assert np.allclose(params[old_idx], old['params'], rtol=0, atol=0, equal_nan=True)
    fb = np.linalg.norm(params[:, :, 0]+f['params'][:, :, 0], axis=1)
    np.savez_compressed(OUT/'reverse.npz', points=destination,
        source_points=source_points, params=params, radius=13, order=2,
        reg=.00005, margin=10, sigma=.65, maskednorm=True,
        frozen_original=f['frozen_original'], fb_error=fb,
        origin0=inputs['origin0'], direction='B to A; polynomial centers in B', **stats)
    summary = dict(seconds=time.time()-start, frozen=len(old_idx), fitted=len(new_idx),
                   finite=int(np.isfinite(params).all(axis=(1, 2)).sum()),
                   patch_fb_percentiles=np.nanpercentile(fb, [0, 50, 90, 95, 100]).tolist())
    print('reverse', json.dumps(summary), flush=True)
    return summary


def audit(inputs):
    q = inputs['manual_xy']
    oldq = q-(inputs['old_origin0']-inputs['origin0'])
    truth = inputs['manual_truth']
    depth = q[:, 1]-np.interp(q[:, 0], np.arange(len(inputs['surface_a'])), inputs['surface_a'])
    report = {}
    for name, _, _, _ in VARIANTS:
        new = LocalField(str(OUT/(OUTPUT_NAMES[name]+'.npz')))
        old = LocalField(str(ROOT/'work'/(name+'.npz')))
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
            gradient=g, original_displacement=od, original_gradient=og, error=e,
            displacement_change=change, gradient_change=ge, depth=depth)
    (OUT/'manual_preservation_audit.json').write_text(json.dumps(report, indent=2))
    print('manual audit', json.dumps(report), flush=True)
    return report


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--tracks', type=Path, default=OUT/'combined_tracks.npz')
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
