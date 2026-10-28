"""Refit one pair with every fitted polynomial forced to a chosen order.

Reuses the pair's prepared inputs and particle tracks unchanged -- the rough
survey is already linear and particle matching fits no polynomial -- then redoes
the whole fitting stage with the polynomial order as a parameter:

  starting field from particle tracks   (package: order 2)
  four forward fits                     (package: orders 2, 1, 2, 2)
  backward fit, image B to image A      (package: order 2)

The mathematics is the package's own: this calls its solver (fit_local) and its
particle-surface model (PTVModel) directly. Only the stage plumbing is restated
here, because the package hard-codes order 2 inside fit_fields.py, and editing
that file would change its fingerprint and stop `run` from resuming every
existing output directory.

Restating plumbing risks a silent divergence, so --verify N first runs the
stages at the package's own orders on N grid points and requires bit-identical
agreement with the pair's existing seeds/main/affine/large_window/margin14/
reverse files. Only then is the requested order trusted to be the sole
difference.

Usage:
  refit_order.py <source pair dir> <destination pair dir> --order 1 [--verify 400] [--workers 24]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
import json
import multiprocessing as mp
import shutil
import time
from pathlib import Path

import numpy as np

from quadratic_optical.core import fit_fields as ff
from quadratic_optical.core.fit_local_cached import fit_local
from quadratic_optical.core.ptv_model import PTVModel

# Everything steps 1-4 produced, plus preparation. Nothing from the fitting
# stage onward is copied, so no quadratic result can leak into the refit.
KEEP = ['inputs.npz', 'input_manifest.json', 'ptv_tracks.npz', 'tracking_settings.json',
        'tracking_summary.json', 'particle_candidates.npz', 'tracks_forward.npz',
        'tracks_reverse.npz', 'coarse_affine.npz', 'coarse_search.npz',
        'coarse_bootstrap48.npz', 'bootstrap_sensitivity.json', 'source_access_audit.json']

# The backward fit's margin is a literal 10 px in the package, independent of
# the configured surface exclusion. Mirrored deliberately so this refit differs
# from the package in polynomial order ONLY; see the note in the README.
REVERSE_MARGIN = 10.

_G = {}


def _seed_chunk(idx):
    return idx, _G['ptv'].evaluate(_G['points'][idx], scale=10, order=_G['order'])


def _forward(j):
    G = _G
    affine = np.zeros((2, 3))
    affine[:, 0] = G['seed_disp'][j]
    affine[:, 1:3] = G['seed_grad'][j] * G['radius']
    return fit_local(G['A'], G['B'], G['va'], G['vb'], G['points'][j], affine[:, 0],
                     r=G['radius'], order=G['order'], reg=5e-5, maxiter=35, seed_affine=affine,
                     target_gradient=G['gradient'], target_mask_float=G['mask'])


def _reverse(j):
    G = _G
    pp = G['params_f'][j]; k = G['k']
    if not np.isfinite(pp).all():
        return np.full((2, k), np.nan), {'ncc': -1.}
    affine = np.zeros((2, 3)); affine[:, 0] = -pp[:, 0]
    try:
        affine[:, 1:3] = (np.linalg.inv(np.eye(2) + pp[:, 1:3] / 13) - np.eye(2)) * 13
    except np.linalg.LinAlgError:
        return np.full((2, k), np.nan), {'ncc': -1.}
    # Source is B and target is A here, exactly as the package's backward fit.
    return fit_local(G['B'], G['A'], G['vb'], G['va'], G['dest'][j], affine[:, 0], r=13,
                     order=G['order'], reg=5e-5, maxiter=35, seed_affine=affine,
                     target_gradient=G['gradient'], target_mask_float=G['mask'])


def pmap(func, items, workers):
    """Fork-based map: children inherit _G copy-on-write, no image pickling."""
    if workers <= 1:
        return [func(i) for i in items]
    with mp.get_context('fork').Pool(workers) as pool:
        return pool.map(func, items, chunksize=max(1, len(items) // (workers * 8)))


def common_for(directory, inputs, order):
    settings = dict(ff.SETTINGS)
    settings.update(
        ptv_order=order, reverse_order=order,
        variants={s: (r, order, m) for s, (r, _o, m) in ff.VARIANTS.items()},
        research_refit_order=order,
        research_refit_script='research/fit_order/refit_order.py',
        note='Every fitted polynomial forced to order %d; steps 1-4 reused unchanged.' % order)
    return dict(settings=settings, input_sha256=ff.file_hash(directory / 'inputs.npz'),
                tracks_sha256=ff.file_hash(directory / 'ptv_tracks.npz'),
                solver_sha256=ff.file_hash(ff.SOLVER_PATH),
                ptv_model_sha256=ff.file_hash(Path(ff.__file__).resolve().with_name('ptv_model.py')),
                source_sha256=ff.source_hashes(ff.FITTING_SOURCES))


def compute_seeds(inputs, tracks_path, points, order, workers):
    _G.clear(); _G.update(ptv=PTVModel(tracks_path), points=points, order=order)
    chunks = np.array_split(np.arange(len(points)), max(1, workers * 8))
    out = {}
    for idx, res in pmap(_seed_chunk, [c for c in chunks if len(c)], workers):
        for key, value in res.items():
            out.setdefault(key, np.empty((len(points),) + np.asarray(value).shape[1:]))
            out[key][idx] = value
    return out


def compute_forward(inputs, points, seed_disp, seed_grad, stage, order, workers):
    radius, _pkg_order, nominal = ff.VARIANTS[stage]
    margin = ff.base_margin(inputs) + (nominal - ff.DEFAULT_EXCLUSION)
    A, B, va, vb = ff.images(inputs, margin)
    _G.clear(); _G.update(A=A, B=B, va=va, vb=vb, points=points, seed_disp=seed_disp,
                          seed_grad=seed_grad, radius=radius, order=order,
                          gradient=np.gradient(B), mask=vb.astype(float))
    return pmap(_forward, range(len(points)), workers), radius, margin


def compute_reverse(inputs, dest, params_f, order, workers):
    A, B, va, vb = ff.images(inputs, REVERSE_MARGIN)
    k = 6 if order == 2 else 3
    _G.clear(); _G.update(A=A, B=B, va=va, vb=vb, dest=dest, params_f=params_f, order=order,
                          k=k, gradient=np.gradient(A), mask=va.astype(float))
    return pmap(_reverse, range(len(dest)), workers)


def pack(fits, data):
    for j, (params, stats) in enumerate(fits):
        data['params'][j] = params
        for key in ff.STATS:
            data[key][j] = stats.get(key, np.nan)
    data['processed'][:] = True
    data['complete'] = True
    return data


def verify(src, inputs, n, workers):
    """Reproduce the package's own fits on n points; any difference is fatal."""
    points = inputs['points']
    rng = np.random.default_rng(20261028)
    sample = np.sort(rng.choice(len(points), size=min(n, len(points)), replace=False))
    report = {}
    seeds = ff.read_npz(src / 'seeds.npz')
    got = compute_seeds(inputs, src / 'ptv_tracks.npz', points[sample], 2, workers)
    for key in ('disp', 'local_polynomial_gradient'):
        report['seeds.' + key] = float(np.nanmax(np.abs(got[key] - seeds[key][sample])))
    for stage, (_r, pkg_order, _m) in ff.VARIANTS.items():
        ref = ff.read_npz(src / (stage + '.npz'))
        fits, _, _ = compute_forward(inputs, points[sample], seeds['disp'][sample],
                                     seeds['local_polynomial_gradient'][sample], stage,
                                     pkg_order, workers)
        mine = np.array([f[0] for f in fits])
        report[stage] = float(np.nanmax(np.abs(mine - ref['params'][sample])))
        report[stage + '.nan_pattern_equal'] = bool(np.array_equal(np.isnan(mine),
                                                                  np.isnan(ref['params'][sample])))
    main = ff.read_npz(src / 'main.npz'); rev = ff.read_npz(src / 'reverse.npz')
    keep = np.isin(rev['source_indices'], sample)
    rows = rev['source_indices'][keep]
    pf = main['params'][rows]
    fits = compute_reverse(inputs, main['points'][rows] + pf[:, :, 0], pf, 2, workers)
    mine = np.array([f[0] for f in fits])
    report['reverse'] = float(np.nanmax(np.abs(mine - rev['params'][keep])))
    exact = all(v == 0.0 for k, v in report.items() if not k.endswith('nan_pattern_equal')) and \
        all(v for k, v in report.items() if k.endswith('nan_pattern_equal'))
    return exact, report, len(sample)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('source'); ap.add_argument('destination')
    ap.add_argument('--order', type=int, choices=[1, 2], required=True)
    ap.add_argument('--verify', type=int, default=400)
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 4))
    args = ap.parse_args()
    src = Path(args.source).resolve(); dst = Path(args.destination).resolve()
    inputs = ff.inputs_from(src / 'inputs.npz')
    ff.verify_tracks(src / 'ptv_tracks.npz', ff.file_hash(src / 'inputs.npz'))

    if args.verify:
        t = time.time()
        exact, report, n = verify(src, inputs, args.verify, args.workers)
        print('fidelity check on %d grid points (%.0f s): %s' % (n, time.time() - t,
              'EXACT -- every stage reproduced bit-for-bit' if exact else 'MISMATCH'))
        for k, v in report.items():
            print('   %-28s %s' % (k, v))
        if not exact:
            raise SystemExit('Refusing to refit: this script does not reproduce the package.')

    if dst.exists():
        raise SystemExit('Destination exists; refusing to overwrite: ' + str(dst))
    dst.mkdir(parents=True)
    for name in KEEP:
        if (src / name).exists():
            shutil.copy2(src / name, dst / name)
    inputs = ff.inputs_from(dst / 'inputs.npz')
    common = common_for(dst, inputs, args.order)
    points = inputs['points']; started = time.time()

    t = time.time()
    s = compute_seeds(inputs, dst / 'ptv_tracks.npz', points, args.order, args.workers)
    seed_sig = ff.signature(dict(common, stage='seeds', scale=10., order=args.order))
    ff.atomic_npz(dst / 'seeds.npz', points=points, processed=np.ones(len(points), bool),
                  complete=True, run_signature=seed_sig, scale=10., order=args.order,
                  origin0=inputs['origin0'], image_only=True, supplied_velocity_used=False,
                  input_sha256=common['input_sha256'], tracks_sha256=common['tracks_sha256'],
                  elapsed_seconds=time.time() - t, **s)
    print('seeds  order %d  %d points  %.0f s' % (args.order, len(points), time.time() - t), flush=True)

    for stage in ff.VARIANTS:
        t = time.time()
        fits, radius, margin = compute_forward(inputs, points, s['disp'],
                                               s['local_polynomial_gradient'], stage,
                                               args.order, args.workers)
        sig = ff.signature(dict(common, stage=stage, radius=radius, order=args.order,
                                margin=margin, seeds_signature=seed_sig))
        data = ff.empty_model(points, radius, args.order, margin, inputs, common, sig, stage)
        data['initialization'] = 'PTVModel scale10 order%d; research refit' % args.order
        data['elapsed_seconds'] = time.time() - t
        ff.atomic_npz(dst / (stage + '.npz'), **pack(fits, data))
        print('%-13s order %d  window %d  margin %.0f  %.0f s'
              % (stage, args.order, 2 * radius + 1, margin, time.time() - t), flush=True)

    t = time.time()
    main = ff.read_npz(dst / 'main.npz'); pf = main['params']; source = main['points']
    dest = source + pf[:, :, 0]; finite = np.isfinite(dest).all(axis=1)
    fits = compute_reverse(inputs, dest, pf, args.order, args.workers)
    sig = ff.signature(dict(common, stage='reverse', parent_main_sha256=ff.file_hash(dst / 'main.npz')))
    data = pack(fits, ff.empty_model(dest, 13, args.order, REVERSE_MARGIN, inputs, common, sig, 'reverse'))
    data['source_points'] = source
    data['initialization'] = 'negative local forward translation; inverse center affine Jacobian'
    data['direction'] = 'B to A; polynomial centers in frame B'
    final = dict(data)
    for key in ['points', 'params', 'processed', 'source_points'] + ff.STATS:
        final[key] = data[key][finite]
    final['source_indices'] = np.flatnonzero(finite)
    final['source_grid_count'] = len(dest)
    final['undefined_forward_centers'] = int((~finite).sum())
    final['fb_error'] = np.linalg.norm(final['params'][:, :, 0] + pf[finite, :, 0], axis=1)
    ff.atomic_npz(dst / 'reverse.npz', **final)
    print('reverse        order %d  window 27  margin %.0f  %.0f s'
          % (args.order, REVERSE_MARGIN, time.time() - t), flush=True)

    summary = dict(pair=dst.name, refit_order=args.order, settings=common['settings'],
                   input_sha256=common['input_sha256'], tracks_sha256=common['tracks_sha256'],
                   source_pair_directory=str(src), elapsed_seconds=time.time() - started,
                   image_only=True, supplied_velocity_used=False)
    (dst / 'fit_summary.json').write_text(json.dumps(summary, indent=2, default=str) + '\n')
    print('REFIT_DONE %s order %d in %.0f s' % (dst.name, args.order, time.time() - started))


if __name__ == '__main__':
    main()
