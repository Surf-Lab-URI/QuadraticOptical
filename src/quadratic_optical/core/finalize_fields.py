"""Image-only conservative fields in the upper centimetre and arbitrary queries.

All models are freshly fitted from the supplied images and image-derived tracks.
This module never reads supplied velocity fields or previous prediction files.
The C2 local-polynomial blend and evidence thresholds are unchanged; the domain
is restricted to 0.01/DX pixels below the local inferred surface. A fitting halo
supplies stencil support without being reported as additional requested depth.
"""
import os
import argparse
from pathlib import Path
import time
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree, Delaunay
try:
    from scipy.spatial import QhullError
except ImportError:  # SciPy versions predating the public top-level export.
    from scipy.spatial.qhull import QhullError
from ._provenance import forbidden_input_keys
from .fit_fields import ( read_npz, file_hash, scalar_text, atomic_npz,
                        atomic_json, verify_image_only, verify_tracks)
ALTERNATIVES = ['affine', 'large_window', 'margin14']
THRESHOLDS = dict(min_depth=12., requested_depth_m=.01, target_depth=10., ncc=.6,
                  fb=1., spread=1.5, det=.2, nearest=12., count25=6,
                  valid_share=.95, local_mindet=.05, local_support_fraction=.85,
                  gradient_depth=20., gradient_spread=.08, gradient_nearest=10.,
                  visible_mask_interpolated_min=.99, blend_radius=16.,
                  upper_domain_roundoff_tolerance_px=1e-9)


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


def load_geometry(path):
    """Read a strict geometry/mask whitelist, never a velocity-array payload."""
    keys = ['va', 'vb', 'surface_a', 'surface_b', 'points', 'origin0', 'DX', 'DT']
    with np.load(path, allow_pickle=False) as z:
        marker = {k: z[k] for k in ['image_only', 'supplied_velocity_used'] if k in z.files}
        marker.update({k: None for k in forbidden_input_keys(z.files)})
        verify_image_only(marker, 'Image inputs')
        missing = [k for k in keys if k not in z.files]
        if missing:
            raise ValueError('Missing geometry fields: ' + ', '.join(missing))
        data = {k: z[k] for k in keys}
        for k in ('physical_units_confirmed', 'surface_geometry_inferred', 'surface_trace_offset_px',
                  'requested_max_depth_px', 'fitting_max_depth_px', 'detector_max_depth_px',
                  'min_depth_px', 'min_target_depth_px', 'detector_min_depth_px',
                  'gradient_min_depth_px', 'surface_exclusion_px'):
            if k in z.files:
                data[k] = z[k]
    return data


class ConservativeEvaluator:
    """Fresh image-only fields with conservative tests at arbitrary positions.

    query is an N by 2 array of zero-based frame-A coordinates, x right and
    y down. Returned displacements use the same axes in pixels per image pair.
    alternative_displacements has shape (3,N,2) in the fixed ALTERNATIVES order.
    No result cache is shared between calls; integration paths may differ from
    the regular grid, contain missing/invalid positions, and be evaluated in
    independently sized batches without reusing another query's field values.
    """
    def __init__(self, directory, requested_depth_m=.01, batch_size=4096):
        self.directory = Path(directory)
        self.inputs = load_geometry(self.directory/'inputs.npz')
        # Depth floors travel with the inputs; older files predate them and keep
        # the original literals, so existing results stay reproducible.
        def _floor(name, fallback):
            value = self.inputs.get(name)
            return fallback if value is None else float(np.asarray(value).ravel()[0])
        self.min_depth = _floor('min_depth_px', 12.)
        self.min_target_depth = _floor('min_target_depth_px', 10.)
        self.gradient_min_depth = _floor('gradient_min_depth_px', 20.)
        self.origin0 = self.inputs['origin0']
        if not np.array_equal(self.origin0, [0, 0]):
            raise ValueError('These full-image source coordinates require origin0=[0,0].')
        self.DX = float(self.inputs['DX']); self.DT = float(self.inputs['DT'])
        if not np.isfinite([self.DX, self.DT]).all() or min(self.DX, self.DT) <= 0:
            raise ValueError('DX and DT must be positive finite calibration numbers.')
        self.requested_depth_m = float(requested_depth_m)
        if not np.isfinite(self.requested_depth_m) or self.requested_depth_m <= 0:
            raise ValueError('The requested depth must be positive and finite.')
        self.max_depth_px = self.requested_depth_m/self.DX
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError('batch_size must be positive.')
        self.provenance = dict(input_sha256=file_hash(self.directory/'inputs.npz'),
                               tracks_sha256=file_hash(self.directory/'ptv_tracks.npz'),
                               evaluator_sha256=file_hash(Path(__file__)),
                               image_only=True, supplied_velocity_used=False)
        tracks = verify_tracks(self.directory/'ptv_tracks.npz', self.provenance['input_sha256'])
        self.features = tracks['points'][tracks['accepted']]
        self.feature_tree = cKDTree(self.features)
        if len(self.features) < 3:
            raise ValueError('At least three noncollinear accepted image tracks are required for the support hull.')
        try:
            self.feature_hull = Delaunay(self.features)
        except QhullError as exc:
            raise ValueError('Accepted image tracks do not span a two-dimensional support hull.') from exc
        self.models = {}
        for name in ['main']+ALTERNATIVES+['reverse']:
            model = FastLocalField(self.directory/(name+'.npz'))
            verify_image_only(model.data, 'Local model '+name)
            if not bool(model.data.get('complete', False)):
                raise ValueError(name+' is not a completed fit stage.')
            for key in ['input_sha256', 'tracks_sha256']:
                if scalar_text(model.data.get(key, '')) != self.provenance[key]:
                    raise ValueError(name+' does not identify current '+key+'.')
            if name != 'reverse' and not np.array_equal(model.points, self.inputs['points']):
                raise ValueError('Every forward model must use the current fitting grid including its halo.')
            self.models[name] = model
        self.surface_x = np.arange(len(self.inputs['surface_a']))

    def evaluate_conservative(self, query):
        query = np.asarray(query, dtype=float)
        if query.ndim == 1 and query.shape == (2,):
            query = query[None, :]
        if query.ndim != 2 or query.shape[1] != 2:
            raise ValueError('query must have shape (N,2).')
        q = query; n = len(q)
        neighbors = neighborhoods(self.models['main'].tree, q)
        u, g, ncc, forward_share = self.models['main'].evaluate(
            q, diagnostics=True, neighbors=neighbors, batch_size=self.batch_size, include_share=True)
        alternative_u = []; alternative_g = []
        for name in ALTERNATIVES:
            a, b = self.models[name].evaluate(q, neighbors=neighbors, batch_size=self.batch_size)
            alternative_u.append(a); alternative_g.append(b)
        ua = np.array(alternative_u); ga = np.array(alternative_g)
        target = q+u
        reverse_neighbors = neighborhoods(self.models['reverse'].tree, target)
        bu, _, _, reverse_share = self.models['reverse'].evaluate(
            target, diagnostics=True, neighbors=reverse_neighbors,
            batch_size=self.batch_size, include_share=True)
        fb = np.linalg.norm(u+bu, axis=1)
        available = np.all(np.isfinite(ua), axis=(0, 2)) & np.all(np.isfinite(ga), axis=(0, 2, 3))
        spread = np.max(np.linalg.norm(ua-u[None], axis=2), axis=0)
        spread_xx = np.max(np.abs(ga[:, :, 0, 0]-g[None, :, 0, 0]), axis=0)
        spread_yy = np.max(np.abs(ga[:, :, 1, 1]-g[None, :, 1, 1]), axis=0)
        finite_q = np.isfinite(q).all(axis=1)
        nearest = np.full(n, np.inf); count = np.zeros(n, dtype=int); in_hull = np.zeros(n, bool)
        if finite_q.any():
            nearest[finite_q] = self.feature_tree.query(q[finite_q])[0]
            count[finite_q] = self.feature_tree.query_ball_point(q[finite_q], 25., return_length=True)
            in_hull[finite_q] = self.feature_hull.find_simplex(q[finite_q]) >= 0
        support = in_hull & (nearest <= 12) & (count >= 6)
        depth = q[:, 1]-np.interp(q[:, 0], self.surface_x, self.inputs['surface_a'])
        target_depth = target[:, 1]-np.interp(target[:, 0], self.surface_x, self.inputs['surface_b'])
        determinant = np.full(n, np.nan)
        finite_g = np.all(np.isfinite(g), axis=(1, 2))
        determinant[finite_g] = np.linalg.det(np.eye(2)[None]+g[finite_g])
        source_visible = visible(self.inputs['va'], q)
        target_visible = visible(self.inputs['vb'], target, strict_target=True)
        with np.errstate(invalid='ignore'):
            # Only the requested-domain upper edge gets floating-point slack;
            # no original image-consistency or lower-depth threshold changes.
            requested_domain = finite_q & (depth <= self.max_depth_px+1e-9)
            evidence = (available & (forward_share >= .95) & (reverse_share >= .95) &
                        (depth >= self.min_depth) & requested_domain &
                        (target_depth >= self.min_target_depth) & support &
                        (ncc >= .6) & (fb <= 1) & (spread <= 1.5) & (determinant > .2) &
                        np.all(np.isfinite(u), axis=1))
            accepted = evidence & source_visible & target_visible
            gradient_xx = accepted & (depth >= self.gradient_min_depth) & (spread_xx <= .08) & (nearest <= 10)
            gradient_yy = accepted & (depth >= self.gradient_min_depth) & (spread_yy <= .08) & (nearest <= 10)
        return dict(query=q.copy(), query_full=q+self.origin0, disp=u, gradient=g, ncc=ncc,
                    fb=fb, depth=depth, target_depth=target_depth, method_spread=spread,
                    gradient_spread_xx=spread_xx, gradient_spread_yy=spread_yy,
                    accepted=accepted, gradient_accepted_xx=gradient_xx, gradient_accepted_yy=gradient_yy,
                    alternative_displacements=ua, alternative_gradients=ga,
                    nearest_feature=nearest, feature_count25=count, determinant=determinant,
                    source_visible=source_visible, target_visible=target_visible,
                    evidence_pass=evidence, local_valid_share=forward_share,
                    reverse_valid_share=reverse_share, support=support,
                    alternatives_available=available, requested_depth_domain=requested_domain)


def optional_range(values, selected):
    return [float(values[selected].min()), float(values[selected].max())] if selected.any() else None


def run(directory, requested_depth_m=.01, batch_size=4096):
    """Evaluate and save the requested conservative reporting grid."""
    started = time.time()
    directory = Path(directory).resolve()
    pair = directory.name
    evaluator = ConservativeEvaluator(directory, requested_depth_m=requested_depth_m, batch_size=batch_size)
    inp = evaluator.inputs
    depth = inp['points'][:, 1]-np.interp(inp['points'][:, 0], evaluator.surface_x, inp['surface_a'])
    selected = (depth >= 12) & (depth <= evaluator.max_depth_px+1e-9)
    q = inp['points'][selected]
    result = evaluator.evaluate_conservative(q)
    n = len(q); accepted = result['accepted']; gx = result['gradient_accepted_xx']; gy = result['gradient_accepted_yy']
    metadata = dict(pair=str(pair), grid_count=n, fitting_grid_count=len(inp['points']),
                     grid_indices_in_fitting_grid=np.flatnonzero(selected),
                     origin0=inp['origin0'], DX=inp['DX'], DT=inp['DT'],
                     requested_depth_m=evaluator.requested_depth_m, requested_depth_px=evaluator.max_depth_px,
                     physical_units_confirmed=bool(inp.get('physical_units_confirmed', False)),
                     manual_data_used=False, old_fits_used=False,
                     gradient_definition='G[i,j]=d displacement_component_i / d image_coordinate_j; x-right,y-down',
                     conditional_si_conversion='u=DX/DT*disp_x; w_up=-DX/DT*disp_y; du/dx=G00/DT; dw/dz_up=G11/DT',
                     domain_note='Only source depths <=requested_depth_m/DX are reported; extra fitted halo supplies support.',
                     surface_a=inp['surface_a'], surface_b=inp['surface_b'],
                     alternative_names=np.array(ALTERNATIVES), **evaluator.provenance)
    atomic_npz(directory/'results.npz', **dict(result, **metadata))
    # THRESHOLDS records the defaults; a run with different floors must not have
    # its summary claim the defaults were used.
    thresholds = dict(THRESHOLDS, max_depth=evaluator.max_depth_px,
                      min_depth=evaluator.min_depth, target_depth=evaluator.min_target_depth,
                      gradient_depth=evaluator.gradient_min_depth)
    summary = dict(pair=str(pair), method='Fresh image-only masked robust quadratic registration initialized by image-only automatic particle tracks.',
                    grid_nodes=n, fitting_grid_nodes=len(inp['points']), accepted_grid=int(accepted.sum()),
                    gradient_grid_xx=int(gx.sum()), gradient_grid_yy=int(gy.sum()),
                    gradient_grid_both=int((gx & gy).sum()), accepted_features=len(evaluator.features),
                    accepted_grid_depth_range_px=optional_range(result['depth'], accepted),
                    accepted_grid_x_range_px=optional_range(q[:, 0], accepted),
                    accepted_grid_y_range_px=optional_range(q[:, 1], accepted),
                    mask_visibility_exclusions_grid=int((result['evidence_pass'] &
                                                        ~(result['source_visible'] & result['target_visible'])).sum()),
                    requested_depth_m=evaluator.requested_depth_m, requested_depth_px=evaluator.max_depth_px,
                    thresholds=thresholds, origin_zero_based=inp['origin0'].tolist(),
                    DX=evaluator.DX, DT=evaluator.DT, physical_units_confirmed=bool(inp.get('physical_units_confirmed', False)),
                    image_only=True, supplied_velocity_used=False,
                    manual_validation_available=False, previous_pair_fits_used=False,
                    gradient_note='Separate G00 and G11 sensitivity screens; both divide by DT for conditional physical derivatives. No incompressibility constraint.',
                    surface_note='Surface geometry and its offset come from input preparation; actual retained-image availability is applied independently.',
                    integration_api='ConservativeEvaluator(directory).evaluate_conservative(query), zero-based (x,y-down), no query-result cache.',
                    source_hashes=evaluator.provenance, elapsed_seconds=time.time()-started)
    atomic_json(directory/'summary.json', summary)
    print('image-only pair', pair, 'finished', n, 'requested grid nodes;', int(accepted.sum()),
          'velocities;', int(gx.sum()), 'xx gradients;', int(gy.sum()), 'yy gradients;',
          round(time.time()-started, 1), 'seconds', flush=True)


    return result, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--requested-depth-m', type=float, default=.01)
    parser.add_argument('--batch-size', type=int, default=4096)
    run(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
