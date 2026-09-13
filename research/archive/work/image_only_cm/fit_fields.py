"""Image-only conservative local registration, with atomic resumable stages.

No supplied velocity arrays, earlier predictions, or manual references are used. The
imported fit_local implementation is unchanged; exact image-gradient/mask
arrays are shared across the independent fits. Run only after inputs.npz and
the complete atomic ptv_tracks.npz have been prepared for the requested pair.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / 'work'
SOLVER_PATH = WORK / 'full_width' / 'fit_local_cached.py'
sys.path.insert(0, str(WORK))
sys.path.insert(0, str(SOLVER_PATH.parent))
from fit_local_cached import fit_local
from ptv_model import PTVModel

SCHEMA_VERSION = 1
STATS = ['ncc', 'rms', 'nvalid', 'cond', 'iterations', 'mindet', 'support_fraction']
VARIANTS = {
    'main': (13, 2, 10),
    'affine': (13, 1, 10),
    'large_window': (19, 2, 10),
    'margin14': (13, 2, 14),
}
SETTINGS = dict(schema_version=SCHEMA_VERSION, fresh_pair=True, image_only=True,
                regularization=5e-5, maxiter=35, sigma=.65,
                ptv_scale=10., ptv_order=2, interpolation='bilinear',
                variants=VARIANTS, reverse_radius=13, reverse_order=2,
                reverse_margin=10, old_fits_used=False, manual_data_used=False,
                supplied_velocity_used=False)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def scalar_text(value):
    return str(np.asarray(value).item())


def read_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def atomic_npz(path, **data):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **data)
    os.replace(str(temporary), str(path))


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    os.replace(str(temporary), str(path))


def signature(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def verify_image_only(data, label):
    """Prevent accidentally mixing these fits with prior comparison artifacts."""
    if not bool(np.asarray(data.get('image_only', False)).item()):
        raise ValueError(label + ' must explicitly declare image_only=True.')
    if 'supplied_velocity_used' not in data or bool(np.asarray(data['supplied_velocity_used']).item()):
        raise ValueError(label + ' must explicitly declare supplied_velocity_used=False.')
    forbidden = [k for k in data if (k.startswith('classical') or
                 k.startswith('supplied_d') or k in ('delta_x', 'delta_z', 'delta_x1', 'delta_z1'))]
    if forbidden:
        raise ValueError(label + ' contains forbidden supplied-velocity fields: ' + ', '.join(forbidden))


def verify_tracks(path, expected_input_hash):
    data = read_npz(path)
    verify_image_only(data, 'Automatic tracks')
    if scalar_text(data.get('input_sha256', '')) != expected_input_hash:
        raise ValueError('Automatic tracks do not identify the current image-only inputs.')
    return data


def inputs_from(path):
    # Select only fields required for image fitting. No manual data is read.
    keys = ['A', 'B', 'rawA', 'rawB', 'va', 'vb', 'availability_a',
            'availability_b', 'surface_a', 'surface_b', 'points', 'origin0']
    with np.load(path, allow_pickle=False) as z:
        missing = [k for k in keys if k not in z.files]
        if missing:
            raise ValueError('Missing fitting input keys: ' + ', '.join(missing))
        marker = {k: z[k] for k in ['image_only', 'supplied_velocity_used'] if k in z.files}
        # Field-name inspection does not read excluded arrays.
        marker.update({k: None for k in z.files if k.startswith('classical') or
                       k.startswith('supplied_d') or k in ('delta_x', 'delta_z', 'delta_x1', 'delta_z1')})
        verify_image_only(marker, 'Image inputs')
        out = {k: z[k] for k in keys}
    shape = out['A'].shape
    if len(shape) != 2 or any(out[k].shape != shape for k in
           ['B', 'rawA', 'rawB', 'va', 'vb', 'availability_a', 'availability_b']):
        raise ValueError('Image and mask shapes must agree.')
    if not np.all(np.isfinite(out['A'])) or not np.all(np.isfinite(out['B'])):
        raise ValueError('Normalized fitting images must be finite.')
    for key in ['va', 'vb', 'availability_a', 'availability_b']:
        if out[key].dtype != bool:
            raise ValueError(key + ' must be a boolean mask.')
    if out['points'].ndim != 2 or out['points'].shape[1] != 2:
        raise ValueError('points must be an N by 2 source-coordinate array.')
    if not np.isfinite(out['points']).all():
        raise ValueError('Source-grid coordinates must be finite.')
    if any(out[k].shape != (shape[1],) for k in ['surface_a', 'surface_b']):
        raise ValueError('Surface vectors must cover every image column.')
    if not np.array_equal(out['origin0'], [0, 0]):
        raise ValueError('Fresh complete-image pairs require origin0=[0,0].')
    return out


def normalize(raw, valid):
    """The same masked normalization as the original conservative estimator."""
    v = valid.astype(float)
    smooth = gaussian_filter(raw*v, .65) / np.maximum(gaussian_filter(v, .65), 1e-5)
    mean = gaussian_filter(smooth*v, 7) / np.maximum(gaussian_filter(v, 7), 1e-5)
    high = smooth-mean
    rms = np.sqrt(gaussian_filter(high*high*v, 9) /
                  np.maximum(gaussian_filter(v, 9), 1e-5) + 25)
    return np.clip(high/rms, -3, 4)


def images(inputs, margin):
    if margin == 10:
        return tuple(inputs[k] for k in ['A', 'B', 'va', 'vb'])
    y = np.arange(inputs['A'].shape[0])[:, None]
    va = inputs['availability_a'] & (y >= inputs['surface_a'][None, :] + margin)
    vb = inputs['availability_b'] & (y >= inputs['surface_b'][None, :] + margin)
    return normalize(inputs['rawA'], va), normalize(inputs['rawB'], vb), va, vb


def verify_saved(data, run_signature, points, stage):
    if scalar_text(data.get('run_signature', '')) != run_signature:
        raise ValueError(stage + ': existing checkpoint belongs to different inputs/settings; '
                         'archive it explicitly before starting a different run.')
    if not np.array_equal(data['points'], points):
        raise ValueError(stage + ': source coordinates differ from its checkpoint.')


def seed_stage(directory, inputs, common, checkpoint_every):
    target = directory/'seeds.npz'
    checkpoint = directory/'seeds_checkpoint.npz'
    points = inputs['points']; n = len(points)
    run_signature = signature(dict(common, stage='seeds', scale=10., order=2))
    if target.exists():
        data = read_npz(target)
        verify_saved(data, run_signature, points, 'seeds')
        if not bool(data.get('complete', False)):
            raise ValueError('seeds.npz is not a complete stage.')
        return data
    if checkpoint.exists():
        data = read_npz(checkpoint)
        verify_saved(data, run_signature, points, 'seeds')
    else:
        data = dict(points=points, disp=np.full((n, 2), np.nan),
                    local_polynomial_gradient=np.full((n, 2, 2), np.nan),
                    rms=np.full(n, np.nan), count=np.full(n, np.nan),
                    nearest_feature=np.full(n, np.nan), processed=np.zeros(n, bool),
                    run_signature=run_signature, complete=False, elapsed_seconds=0.,
                    scale=10., order=2, origin0=inputs['origin0'],
                    image_only=True, supplied_velocity_used=False,
                    input_sha256=common['input_sha256'], tracks_sha256=common['tracks_sha256'])
    model = PTVModel(directory/'ptv_tracks.npz')
    if len(model.points) < 2:
        raise ValueError('At least two accepted automatic tracks are required for the existing PTV prior.')
    pending = np.flatnonzero(~data['processed'])
    elapsed = float(data.get('elapsed_seconds', 0.)); started = time.time()
    for first in range(0, len(pending), checkpoint_every):
        idx = pending[first:first+checkpoint_every]
        result = model.evaluate(points[idx], scale=10, order=2)
        if not np.isfinite(result['disp']).all() or not np.isfinite(result['local_polynomial_gradient']).all():
            raise ValueError('Nonfinite particle-prior initialization; no substitute prior was inserted.')
        for key, value in result.items():
            data[key][idx] = value
        data['processed'][idx] = True
        data['elapsed_seconds'] = elapsed+time.time()-started
        data['complete'] = bool(data['processed'].all())
        atomic_npz(checkpoint, **data)
        print('pair', directory.name, 'seeds', int(data['processed'].sum()), '/', n,
              'seconds', round(float(data['elapsed_seconds']), 1), flush=True)
    data['complete'] = bool(data['processed'].all())
    atomic_npz(target, **data)
    return data


def empty_model(points, radius, order, margin, inputs, common, run_signature, stage):
    n = len(points); k = 6 if order == 2 else 3
    data = dict(points=points, params=np.full((n, 2, k), np.nan),
                radius=radius, order=order, margin=margin, reg=5e-5,
                maxiter=35, sigma=.65, maskednorm=True,
                processed=np.zeros(n, bool), complete=False, elapsed_seconds=0.,
                origin0=inputs['origin0'], run_signature=run_signature, stage=stage,
                input_sha256=common['input_sha256'], tracks_sha256=common['tracks_sha256'],
                solver_sha256=common['solver_sha256'], schema_version=SCHEMA_VERSION,
                interpolation='bilinear; unchanged fit_local_cached',
                initialization='fresh automatic PTVModel scale10 order2; no earlier-pair fits',
                fresh_pair=True, manual_data_used=False, image_only=True,
                supplied_velocity_used=False)
    data.update({key: np.full(n, np.nan) for key in STATS})
    return data


def complete_summary(data, stage, path):
    return dict(stage=stage, path=str(path), complete=bool(data['complete']),
                saved_rows=len(data['points']),
                finite_parameters=int(np.isfinite(data['params']).all(axis=(1, 2)).sum()),
                elapsed_seconds=float(data['elapsed_seconds']))


def forward_stage(directory, inputs, seeds, common, stage, workers, checkpoint_every):
    radius, order, margin = VARIANTS[stage]
    points = inputs['points']; n = len(points)
    run_signature = signature(dict(common, stage=stage, radius=radius, order=order,
                                   margin=margin, seeds_signature=scalar_text(seeds['run_signature'])))
    target = directory/(stage+'.npz'); checkpoint = directory/(stage+'_checkpoint.npz')
    if target.exists():
        data = read_npz(target)
        verify_saved(data, run_signature, points, stage)
        if not bool(data.get('complete', False)):
            raise ValueError(str(target) + ' is incomplete.')
        return complete_summary(data, stage, target)
    if checkpoint.exists():
        data = read_npz(checkpoint)
        verify_saved(data, run_signature, points, stage)
    else:
        data = empty_model(points, radius, order, margin, inputs, common, run_signature, stage)
    A, B, va, vb = images(inputs, margin)
    # Reused verbatim arrays: no approximation or downsampling is introduced.
    gradient = np.gradient(B); target_mask = vb.astype(float)
    pending = np.flatnonzero(~data['processed'])
    elapsed = float(data.get('elapsed_seconds', 0.)); started = time.time()
    def task(j):
        affine = np.zeros((2, 3))
        affine[:, 0] = seeds['disp'][j]
        affine[:, 1:3] = seeds['local_polynomial_gradient'][j]*radius
        return fit_local(A, B, va, vb, points[j], affine[:, 0], r=radius,
                         order=order, reg=5e-5, maxiter=35, seed_affine=affine,
                         target_gradient=gradient, target_mask_float=target_mask)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for first in range(0, len(pending), checkpoint_every):
            idx = pending[first:first+checkpoint_every]
            fits = list(pool.map(task, idx))
            for j, (params, stats) in zip(idx, fits):
                data['params'][j] = params
                for key in STATS:
                    data[key][j] = stats.get(key, np.nan)
            data['processed'][idx] = True
            data['elapsed_seconds'] = elapsed+time.time()-started
            data['complete'] = bool(data['processed'].all())
            atomic_npz(checkpoint, **data)
            print('pair', directory.name, stage, int(data['processed'].sum()), '/', n,
                  'seconds', round(float(data['elapsed_seconds']), 1), flush=True)
    data['complete'] = bool(data['processed'].all())
    atomic_npz(target, **data)
    return complete_summary(data, stage, target)


def reverse_stage(directory, inputs, common, workers, checkpoint_every):
    main_path = directory/'main.npz'
    if not main_path.exists():
        raise FileNotFoundError('A completed main.npz is required before reverse fitting.')
    forward = read_npz(main_path)
    if not bool(forward.get('complete', False)):
        raise ValueError('main.npz is incomplete.')
    for key in ['input_sha256', 'tracks_sha256', 'solver_sha256']:
        if scalar_text(forward.get(key, '')) != common[key]:
            raise ValueError('main.npz does not match the current ' + key + '.')
    source = forward['points']; params_f = forward['params']; n = len(source)
    destination = source+params_f[:, :, 0]
    center_finite = np.isfinite(destination).all(axis=1)
    run_signature = signature(dict(common, stage='reverse', parent_main_sha256=file_hash(main_path)))
    target = directory/'reverse.npz'; checkpoint = directory/'reverse_checkpoint.npz'
    if target.exists():
        data = read_npz(target)
        verify_saved(data, run_signature, destination[center_finite], 'reverse')
        if not bool(data.get('complete', False)):
            raise ValueError('reverse.npz is incomplete.')
        return complete_summary(data, 'reverse', target)
    if checkpoint.exists():
        data = read_npz(checkpoint)
        # Invalid forward endpoints are represented by NaNs in this checkpoint.
        if scalar_text(data.get('run_signature', '')) != run_signature:
            raise ValueError('Reverse checkpoint does not match the current forward field.')
        if not np.allclose(data['points'], destination, rtol=0, atol=0, equal_nan=True):
            raise ValueError('Reverse endpoint centers differ from the checkpoint.')
    else:
        data = empty_model(destination, 13, 2, 10, inputs, common, run_signature, 'reverse')
        data['processed'][~center_finite] = True
        data['ncc'][~center_finite] = -1.
        data['source_points'] = source
        data['initialization'] = 'negative local forward translation; inverse center affine Jacobian'
        data['direction'] = 'B to A; polynomial centers in frame B'
    A, B, va, vb = images(inputs, 10)
    gradient = np.gradient(A); target_mask = va.astype(float)
    pending = np.flatnonzero(~data['processed'])
    elapsed = float(data.get('elapsed_seconds', 0.)); started = time.time()
    def task(j):
        pp = params_f[j]
        if not np.isfinite(pp).all():
            return np.full((2, 6), np.nan), {'ncc': -1.}
        affine = np.zeros((2, 3)); affine[:, 0] = -pp[:, 0]
        try:
            affine[:, 1:3] = (np.linalg.inv(np.eye(2)+pp[:, 1:3]/13)-np.eye(2))*13
        except np.linalg.LinAlgError:
            return np.full((2, 6), np.nan), {'ncc': -1.}
        return fit_local(B, A, vb, va, destination[j], affine[:, 0], r=13,
                         order=2, reg=5e-5, maxiter=35, seed_affine=affine,
                         target_gradient=gradient, target_mask_float=target_mask)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for first in range(0, len(pending), checkpoint_every):
            idx = pending[first:first+checkpoint_every]
            fits = list(pool.map(task, idx))
            for j, (params, stats) in zip(idx, fits):
                data['params'][j] = params
                for key in STATS:
                    data[key][j] = stats.get(key, np.nan)
            data['processed'][idx] = True
            data['elapsed_seconds'] = elapsed+time.time()-started
            data['complete'] = bool(data['processed'].all())
            atomic_npz(checkpoint, **data)
            print('pair', directory.name, 'reverse', int(data['processed'].sum()), '/', n,
                  'seconds', round(float(data['elapsed_seconds']), 1), flush=True)
    data['complete'] = bool(data['processed'].all())
    # A cKDTree cannot contain NaN centers. Remove only such undefined centers;
    # finite-center failed fits stay in the blend and preserve NaN propagation.
    final = dict(data)
    for key in ['points', 'params', 'processed', 'source_points'] + STATS:
        final[key] = data[key][center_finite]
    final['source_indices'] = np.flatnonzero(center_finite)
    final['source_grid_count'] = n
    final['undefined_forward_centers'] = int((~center_finite).sum())
    final['fb_error'] = np.linalg.norm(final['params'][:, :, 0] + params_f[center_finite, :, 0], axis=1)
    atomic_npz(target, **final)
    return complete_summary(final, 'reverse', target)


def run_pair(pair, args):
    directory = args.base_dir/str(pair)
    input_path = directory/'inputs.npz'; tracks_path = directory/'ptv_tracks.npz'
    if not input_path.exists() or not tracks_path.exists():
        raise FileNotFoundError('Both complete inputs.npz and ptv_tracks.npz must exist in '+str(directory))
    inputs = inputs_from(input_path)
    verify_tracks(tracks_path, file_hash(input_path))
    common = dict(settings=SETTINGS, input_sha256=file_hash(input_path),
                  tracks_sha256=file_hash(tracks_path), solver_sha256=file_hash(SOLVER_PATH),
                  ptv_model_sha256=file_hash(WORK/'ptv_model.py'))
    report_path = directory/'fit_summary.json'
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    if report and report.get('run_signature') != signature(common):
        raise ValueError('Existing fit summary belongs to different inputs/settings.')
    report.update(pair=str(pair), settings=SETTINGS, run_signature=signature(common),
                  input_sha256=common['input_sha256'], tracks_sha256=common['tracks_sha256'],
                  grid_nodes=len(inputs['points']), no_previous_pair_fits=True,
                  image_only=True, supplied_velocity_used=False)
    atomic_json(report_path, report)
    seeds = None
    if 'seeds' in args.stages or any(stage in VARIANTS for stage in args.stages):
        seeds = seed_stage(directory, inputs, common, args.checkpoint_every)
        report['seeds'] = dict(complete=bool(seeds['complete']), rows=len(seeds['points']),
                               elapsed_seconds=float(seeds['elapsed_seconds']))
        atomic_json(report_path, report)
    for stage in args.stages:
        if stage in VARIANTS:
            report[stage] = forward_stage(directory, inputs, seeds, common, stage,
                                          args.workers, args.checkpoint_every)
            atomic_json(report_path, report)
        elif stage == 'reverse':
            report[stage] = reverse_stage(directory, inputs, common, args.workers,
                                          args.checkpoint_every)
            atomic_json(report_path, report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs', nargs='+', required=True, choices=['80', '100'])
    parser.add_argument('--base-dir', type=Path, default=WORK/'image_only_cm')
    parser.add_argument('--stages', nargs='+', choices=['seeds']+list(VARIANTS)+['reverse'],
                        default=['seeds']+list(VARIANTS)+['reverse'])
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--checkpoint-every', type=int, default=512)
    args = parser.parse_args()
    if args.workers < 1 or args.checkpoint_every < 1:
        parser.error('workers and checkpoint-every must be positive.')
    for pair in args.pairs:
        run_pair(pair, args)


if __name__ == '__main__':
    main()
