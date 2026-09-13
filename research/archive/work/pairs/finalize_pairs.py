"""Conservative, full-depth fresh-pair fields and separate diagonal gradients.

The local polynomial and compact C2 blend are unchanged. Vectorized evaluation
only shares spatial searches and batches algebra; actual failed neighbors retain
their original NaN propagation. Native classical PIV is compared independently
of hybrid acceptance, disagreement, and its supplied correlation coefficient.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import argparse
from pathlib import Path
import sys
import time

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree, Delaunay

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_pairs import (WORK, read_npz, file_hash, scalar_text, signature,
                       atomic_npz, atomic_json)

ALTERNATIVES = ['affine', 'large_window', 'margin14']
THRESHOLDS = dict(min_depth=12., max_depth=None, target_depth=10., ncc=.6,
                  fb=1., spread=1.5, det=.2, nearest=12., count25=6,
                  valid_share=.95, local_mindet=.05, local_support_fraction=.85,
                  gradient_depth=20., gradient_spread=.08, gradient_nearest=10.,
                  visible_mask_interpolated_min=.99, blend_radius=16.)


def neighborhoods(tree, query, radius=16.):
    """One spatial query per geometry; keep original single-point tree order."""
    query = np.atleast_2d(query)
    finite = np.isfinite(query).all(axis=1)
    result = [[] for _ in range(len(query))]
    if finite.any():
        found = tree.query_ball_point(query[finite], radius, return_sorted=False)
        for j, indices in zip(np.flatnonzero(finite), found):
            result[j] = indices
    return result


class FastLocalField:
    """Batched equivalent of field_model.LocalField, including its derivative."""
    def __init__(self, path, blend_radius=16.):
        self.data = read_npz(path) if not isinstance(path, dict) else path
        self.points = self.data['points']
        self.params = self.data['params']
        self.radius = float(self.data['radius'])
        self.R = float(blend_radius)
        self.tree = cKDTree(self.points)
        if self.params.shape != (len(self.points), 2, self.params.shape[2]):
            raise ValueError('Unexpected local-polynomial dimensions.')
        if self.params.shape[2] not in (3, 6):
            raise ValueError('Only the original affine and quadratic bases are allowed.')

    def evaluate(self, query, diagnostics=False, neighbors=None, batch_size=4096,
                 include_share=False):
        query = np.atleast_2d(query).astype(float, copy=False)
        if neighbors is None:
            neighbors = neighborhoods(self.tree, query, self.R)
        if len(neighbors) != len(query):
            raise ValueError('Neighborhood count must match queries.')
        n = len(query)
        result = np.full((n, 2), np.nan)
        gradient = np.full((n, 2, 2), np.nan)
        ncc = np.full(n, np.nan)
        share = np.zeros(n)
        model_ncc = self.data.get('ncc', np.ones(len(self.points)))
        with np.errstate(invalid='ignore'):
            good = ((self.data.get('mindet', np.full(len(self.points), np.nan)) > .05) &
                    (self.data.get('support_fraction', np.full(len(self.points), np.nan)) > .85))
        for start in range(0, n, batch_size):
            stop = min(start+batch_size, n)
            local = neighbors[start:stop]
            lengths = np.array([len(v) for v in local])
            maximum = int(lengths.max()) if len(lengths) else 0
            if not maximum:
                continue
            indices = np.zeros((stop-start, maximum), dtype=np.int64)
            exists = np.arange(maximum)[None, :] < lengths[:, None]
            for j, ii in enumerate(local):
                indices[j, :len(ii)] = ii
            d = query[start:stop, None, :] - self.points[indices]
            # Artificial padding is zeroed before arithmetic. Actual neighbors,
            # including failed fits at zero kernel weight, retain their NaNs.
            d[~exists] = 0.
            t = np.linalg.norm(d, axis=2)/self.R
            weight = (1-t)**4*(1+4*t)
            weight[~exists] = 0.
            total = weight.sum(axis=1)
            defined = (total >= 1e-12) & np.isfinite(query[start:stop]).all(axis=1)
            denominator = np.where(defined, total, 1.)
            dw = -20/self.R**2*(1-t[:, :, None])**3*d
            dw[~exists] = 0.
            zx = d[:, :, 0]/self.radius
            zy = d[:, :, 1]/self.radius
            one = np.ones_like(zx); zero = np.zeros_like(zx)
            Q = np.stack([one, zx, zy], axis=2)
            qx = np.stack([zero, one, zero], axis=2)/self.radius
            qy = np.stack([zero, zero, one], axis=2)/self.radius
            if self.params.shape[2] == 6:
                Q = np.concatenate([Q, np.stack([.5*zx*zx, zx*zy, .5*zy*zy], axis=2)], axis=2)
                qx = np.concatenate([qx, np.stack([zx, zy, zero], axis=2)/self.radius], axis=2)
                qy = np.concatenate([qy, np.stack([zero, zx, zy], axis=2)/self.radius], axis=2)
            coefficients = self.params[indices].copy()
            coefficients[~exists] = 0.
            value = np.einsum('nmij,nmj->nmi', coefficients, Q)
            dx = np.einsum('nmij,nmj->nmi', coefficients, qx)
            dy = np.einsum('nmij,nmj->nmi', coefficients, qy)
            average = np.sum(weight[:, :, None]*value, axis=1)/denominator[:, None]
            deviation = value-average[:, None, :]
            # A NaN average should invalidate actual neighbors, but must not
            # poison a result through artificial padding alone.
            deviation[~exists] = 0.
            gx = np.sum(weight[:, :, None]*dx + dw[:, :, 0, None]*deviation,
                        axis=1)/denominator[:, None]
            gy = np.sum(weight[:, :, None]*dy + dw[:, :, 1, None]*deviation,
                        axis=1)/denominator[:, None]
            quality = model_ncc[indices].copy()
            quality[~exists] = 0.
            quality = np.sum(weight*quality, axis=1)/denominator
            valid_share = np.sum(weight*good[indices], axis=1)/denominator
            use = start+np.flatnonzero(defined)
            result[use] = average[defined]
            gradient[use, :, 0] = gx[defined]
            gradient[use, :, 1] = gy[defined]
            ncc[use] = quality[defined]
            # Original share() uses > rather than >= at this boundary.
            valid_share[total <= 1e-12] = 0.
            share[start:stop] = valid_share
        out = (result, gradient)
        if diagnostics:
            out += (ncc,)
        if include_share:
            out += (share,)
        return out


def visible(mask, points, strict_target=False):
    points = np.asarray(points)
    inside = (np.isfinite(points).all(axis=1) &
              (points[:, 0] >= 0) & (points[:, 0] <= mask.shape[1]-1) &
              (points[:, 1] >= 0) & (points[:, 1] <= mask.shape[0]-1))
    if strict_target:
        inside &= ((points[:, 0] >= 1) & (points[:, 0] < mask.shape[1]-2) &
                   (points[:, 1] >= 1) & (points[:, 1] < mask.shape[0]-2))
    valid = np.zeros(len(points), bool)
    valid[inside] = map_coordinates(mask.astype(float),
                                   [points[inside, 1], points[inside, 0]],
                                   order=1, mode='constant', cval=0.) > .99
    return valid


def native_classical(inputs, query):
    """Exact native-node lookup: preserve native NaNs and exclude extrapolation."""
    x = inputs['classical_x']; y = inputs['classical_y']
    dx = inputs['classical_dx']; dy = inputs['classical_dy']
    dcor = inputs['classical_dcor']
    if (dx.shape != (len(y), len(x)) or dy.shape != dx.shape or
            dcor.shape != dx.shape or np.any(np.diff(x) <= 0) or np.any(np.diff(y) <= 0)):
        raise ValueError('Native PIV axes/arrays are not a regular increasing rectilinear grid.')
    ix = np.searchsorted(x, query[:, 0]); iy = np.searchsorted(y, query[:, 1])
    inside = (ix < len(x)) & (iy < len(y)) & np.isfinite(query).all(axis=1)
    exact = np.zeros(len(query), bool)
    jj = np.flatnonzero(inside)
    exact[jj] = (x[ix[jj]] == query[jj, 0]) & (y[iy[jj]] == query[jj, 1])
    jj = np.flatnonzero(exact)
    displacement = np.full((len(query), 2), np.nan)
    quality = np.full(len(query), np.nan)
    displacement[jj, 0] = dx[iy[jj], ix[jj]]
    displacement[jj, 1] = dy[iy[jj], ix[jj]]
    quality[jj] = dcor[iy[jj], ix[jj]]
    if 'classical_at_points' in inputs:
        if not np.allclose(displacement, inputs['classical_at_points'], rtol=0, atol=0, equal_nan=True):
            raise ValueError('Native exact lookup disagrees with the independent input mapping.')
    return displacement, quality, exact


def evaluate_cached(directory, name, model, query, neighbors, provenance, batch_size):
    cache = directory/('evaluated_'+name+'.npz')
    run_signature = signature(dict(provenance, model=name,
                                    model_sha256=file_hash(directory/(name+'.npz'))))
    if cache.exists():
        saved = read_npz(cache)
        if scalar_text(saved.get('run_signature', '')) == run_signature:
            if np.allclose(saved['query'], query, rtol=0, atol=0, equal_nan=True):
                return tuple(saved[k] for k in ['disp', 'gradient', 'ncc', 'share'])
    result = model.evaluate(query, diagnostics=True, neighbors=neighbors,
                            batch_size=batch_size, include_share=True)
    atomic_npz(cache, query=query, disp=result[0], gradient=result[1], ncc=result[2],
               share=result[3], run_signature=run_signature, complete=True)
    return result


def finite_metrics(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not len(values):
        return dict(n=0, mean=None, median=None, rmse=None, p90=None)
    return dict(n=len(values), mean=float(values.mean()), median=float(np.median(values)),
                rmse=float(np.sqrt(np.mean(values*values))), p90=float(np.percentile(values, 90)))


def optional_range(values, selected):
    return [float(values[selected].min()), float(values[selected].max())] if selected.any() else None


def run_pair(pair, args):
    started = time.time()
    directory = args.base_dir/str(pair)
    inputs_path = directory/'inputs.npz'
    tracks_path = directory/'ptv_tracks.npz'
    inputs = read_npz(inputs_path)
    q = inputs['points']; origin = inputs['origin0']
    if not np.array_equal(origin, [0, 0]):
        raise ValueError('Fresh full-frame coordinates require origin0=[0,0].')
    n = len(q)
    provenance = dict(input_sha256=file_hash(inputs_path), tracks_sha256=file_hash(tracks_path),
                      evaluator_sha256=file_hash(Path(__file__)), thresholds=THRESHOLDS)
    models = {}
    for name in ['main']+ALTERNATIVES+['reverse']:
        model = FastLocalField(directory/(name+'.npz'))
        if not bool(model.data.get('complete', False)):
            raise ValueError(name+' is not a completed fit stage.')
        for key in ['input_sha256', 'tracks_sha256']:
            if scalar_text(model.data.get(key, '')) != provenance[key]:
                raise ValueError(name+' provenance does not match current '+key+'.')
        if name != 'reverse' and not np.array_equal(model.points, q):
            raise ValueError('All forward fit centers must equal the fresh source grid.')
        models[name] = model
    neighbors = neighborhoods(models['main'].tree, q)
    u, g, ncc, forward_share = evaluate_cached(directory, 'main', models['main'], q,
                                               neighbors, provenance, args.batch_size)
    alternative_u = []; alternative_g = []
    for name in ALTERNATIVES:
        a, b, _, _ = evaluate_cached(directory, name, models[name], q, neighbors,
                                      provenance, args.batch_size)
        alternative_u.append(a); alternative_g.append(b)
        print('pair', pair, name, 'evaluated', n, 'seconds', round(time.time()-started, 1), flush=True)
    ua = np.array(alternative_u); ga = np.array(alternative_g)
    target = q+u
    reverse_neighbors = neighborhoods(models['reverse'].tree, target)
    bu, _, _, reverse_share = evaluate_cached(directory, 'reverse', models['reverse'], target,
                                                reverse_neighbors, provenance, args.batch_size)
    fb = np.linalg.norm(u+bu, axis=1)
    available = np.all(np.isfinite(ua), axis=(0, 2)) & np.all(np.isfinite(ga), axis=(0, 2, 3))
    spread = np.max(np.linalg.norm(ua-u[None], axis=2), axis=0)
    spread_xx = np.max(np.abs(ga[:, :, 0, 0]-g[None, :, 0, 0]), axis=0)
    spread_yy = np.max(np.abs(ga[:, :, 1, 1]-g[None, :, 1, 1]), axis=0)
    tracks = read_npz(tracks_path)
    features = tracks['points'][tracks['accepted']]
    tree = cKDTree(features); hull = Delaunay(features)
    nearest = tree.query(q)[0]
    count = np.asarray(tree.query_ball_point(q, 25., return_length=True), dtype=int)
    support = (hull.find_simplex(q) >= 0) & (nearest <= 12) & (count >= 6)
    sx = np.arange(len(inputs['surface_a']))
    depth = q[:, 1]-np.interp(q[:, 0], sx, inputs['surface_a'])
    target_depth = target[:, 1]-np.interp(target[:, 0], sx, inputs['surface_b'])
    determinant = np.full(n, np.nan)
    finite_g = np.all(np.isfinite(g), axis=(1, 2))
    determinant[finite_g] = np.linalg.det(np.eye(2)[None]+g[finite_g])
    source_visible = visible(inputs['va'], q)
    target_visible = visible(inputs['vb'], target, strict_target=True)
    with np.errstate(invalid='ignore'):
        evidence = (available & (forward_share >= .95) & (reverse_share >= .95) &
                    (depth >= 12) & (target_depth >= 10) & support & (ncc >= .6) &
                    (fb <= 1) & (spread <= 1.5) & (determinant > .2) &
                    np.all(np.isfinite(u), axis=1))
        accepted = evidence & source_visible & target_visible
        gradient_xx = accepted & (depth >= 20) & (spread_xx <= .08) & (nearest <= 10)
        gradient_yy = accepted & (depth >= 20) & (spread_yy <= .08) & (nearest <= 10)
    classical, classical_quality, classical_exact = native_classical(inputs, q)
    classical_source = visible(inputs['va'], q)
    classical_target = visible(inputs['vb'], q+classical)
    classical_available = (classical_exact & np.isfinite(classical).all(axis=1) &
                           classical_source & classical_target)
    # Retain raw agreement wherever numerically defined; exported comparisons
    # must use the explicitly stored availability and hybrid acceptance flags.
    comparison_error = np.linalg.norm(u-classical, axis=1)
    common = accepted & classical_available
    metadata = dict(pair=str(pair), grid_count=n, origin0=origin, DX=inputs['DX'], DT=inputs['DT'],
                     physical_units_confirmed=False, manual_data_used=False,
                     old_fits_used=False, full_depth=True, maximum_depth_applied=False,
                     gradient_definition='G[i,j]=d displacement_component_i / d image_coordinate_j; x-right, y-down',
                     conditional_si_conversion='u=DX/DT*disp_x; w_up=-DX/DT*disp_y; du/dx=G00/DT; dw/dz_up=G11/DT',
                     classical_comparison_note='Agreement with supplied native PIV, not independent ground truth; dense supplied PIV initializes tracking search.',
                     input_sha256=provenance['input_sha256'], tracks_sha256=provenance['tracks_sha256'])
    output = dict(query=q, query_full=q+origin, disp=u, gradient=g, ncc=ncc, fb=fb,
                   depth=depth, target_depth=target_depth, method_spread=spread,
                   gradient_spread_xx=spread_xx, gradient_spread_yy=spread_yy,
                   accepted=accepted, gradient_accepted_xx=gradient_xx, gradient_accepted_yy=gradient_yy,
                   alternative_displacements=ua, alternative_gradients=ga,
                   nearest_feature=nearest, feature_count25=count, determinant=determinant,
                   source_visible=source_visible, target_visible=target_visible,
                   evidence_pass=evidence, local_valid_share=forward_share, reverse_valid_share=reverse_share,
                   support=support, alternatives_available=available,
                   classical_displacement_native_at_grid=classical,
                   classical_available_at_grid=classical_available,
                   classical_ncc_at_grid=classical_quality, classical_comparison_error=comparison_error,
                   classical_exact_node_at_grid=classical_exact,
                   classical_source_visible_at_grid=classical_source,
                   classical_target_visible_at_grid=classical_target,
                   classical_finite_ncc_at_grid=np.isfinite(classical_quality),
                   surface_a=inputs['surface_a'], surface_b=inputs['surface_b'], **metadata)
    atomic_npz(directory/'results.npz', **output)
    summary = dict(pair=str(pair), method='Original conservative bilinear masked quadratic optical flow initialized from fresh automatic particle tracks.',
                    grid_nodes=n, accepted_grid=int(accepted.sum()),
                    gradient_grid_xx=int(gradient_xx.sum()), gradient_grid_yy=int(gradient_yy.sum()),
                    gradient_grid_both=int((gradient_xx & gradient_yy).sum()),
                    accepted_features=len(features), min_accepted_grid_depth=float(depth[accepted].min()) if accepted.any() else None,
                    source_grid_depth_range_px=optional_range(depth, np.ones(n, bool)),
                    accepted_grid_depth_range_px=optional_range(depth, accepted),
                    accepted_grid_x_range_px=optional_range(q[:, 0], accepted),
                    accepted_grid_y_range_px=optional_range(q[:, 1], accepted),
                    mask_visibility_exclusions_grid=int((evidence & ~(source_visible & target_visible)).sum()),
                    classical_native_exact_grid=int(classical_exact.sum()),
                    classical_native_finite_grid=int((classical_exact & np.isfinite(classical).all(axis=1)).sum()),
                    classical_available_grid=int(classical_available.sum()),
                    classical_available_without_finite_dcor=int((classical_available & ~np.isfinite(classical_quality)).sum()),
                    comparison_common_grid=int(common.sum()),
                    hybrid_classical_agreement_pixels_per_pair=finite_metrics(comparison_error[common]),
                    thresholds=THRESHOLDS, origin_zero_based=origin.tolist(), DX=float(inputs['DX']), DT=float(inputs['DT']),
                    physical_units_confirmed=False, full_depth=True, maximum_depth_applied=False,
                    manual_validation_available=False, manual_data_used=False, previous_pair_fits_used=False,
                    gradient_note='Separate screens for G00 and G11; these are not a divergence-free constraint or a full-tensor confidence statement. Both diagonal physical derivatives divide by DT when SI calibration is assumed.',
                    classical_note=metadata['classical_comparison_note'],
                    surface_note='The geometric trace offset is inferred from the prior exporter convention; actual retained-image availability is applied independently.',
                    source_hashes=provenance, elapsed_seconds=time.time()-started)
    atomic_json(directory/'summary.json', summary)
    print('pair', pair, 'finished', n, 'nodes;', int(accepted.sum()), 'velocities;',
          int(gradient_xx.sum()), 'xx gradients;', int(gradient_yy.sum()), 'yy gradients;',
          round(time.time()-started, 1), 'seconds', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs', nargs='+', required=True, choices=['80', '100'])
    parser.add_argument('--base-dir', type=Path, default=WORK/'pairs')
    parser.add_argument('--batch-size', type=int, default=4096)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('batch-size must be positive.')
    for pair in args.pairs:
        run_pair(pair, args)


if __name__ == '__main__':
    main()
