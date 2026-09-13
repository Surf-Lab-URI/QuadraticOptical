"""Fresh-pair conservative affine prior and reciprocal particle tracking.

Examples, from any working directory:
  python work/pairs/track_pairs.py --pair 80 --workers 4
  python work/pairs/track_pairs.py --pair 100 --stop-after coarse

Each completed chunk is committed atomically. Repeating a command resumes the
same input/code/settings run; no previous-pair coefficients or tracks are used.
The classical displacement arrays provide initialization only. Manual data are
not loaded. The full image depth is used, subject to the unchanged surface and
image-boundary visibility tests.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter, map_coordinates
from scipy.optimize import minimize
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'work'))
sys.path.insert(0, str(ROOT / 'work' / 'full_width'))
from extended_flow import translation_seed
from fit_local_cached import fit_local

STATS = ['ncc', 'rms', 'nvalid', 'cond', 'iterations', 'mindet', 'support_fraction']
INPUT_KEYS = ['A', 'B', 'rawA', 'rawB', 'va', 'vb', 'surface_a',
              'surface_b', 'points', 'origin0', 'supplied_dx', 'supplied_dy']
SETTINGS = dict(
    coarse_radius=19, coarse_order=1, coarse_reg=.00005,
    coarse_max_iterations=35, poor_coarse_ncc=.7,
    coarse_valid_mindet=.15, coarse_valid_support=.7,
    coarse_fallback_search_dx=[-15, 55], coarse_fallback_search_dy=[-20, 20],
    coarse_fallback_candidates=3, coarse_fallback_min_score=.05,
    detector_dog_sigmas=[.6, 2], detector_local_max_width=5,
    detector_min_peak=8, detector_background_sigma=5,
    detector_background_max=180, detector_min_source_depth=14,
    detector_max_source_depth=None, detector_patch_within_image=True,
    particle_image_smoothing_sigma=.5, track_patch_radius=4,
    track_displacement_search_radius=8, track_min_common_fraction=.85,
    track_prior_penalty=.005, track_alternative_separation=2.5,
    track_optimizer='Nelder-Mead', track_max_iterations=55,
    track_xatol=.035, track_fatol=.0002,
    track_ncc_min=.68, track_reverse_ncc_min=.68, track_fb_max=1.,
    track_gap_min=.015, track_reverse_gap_min=.01,
    prior_neighbor_count=12, prior_spatial_sigma=25,
    prior_displacement_consensus_radius=4, manual_data_used=False,
    classical_displacements_used_only_for_coarse_initialization=True,
    frozen_previous_coefficients=0, frozen_previous_tracks=0,
    target_gradient_cache='identical np.gradient(B) and vb.astype(float)',
)


def load_npz(path, keys=None):
    """Eager loading avoids sharing a lazy ZIP reader across worker threads."""
    with np.load(str(path), allow_pickle=False) as z:
        return {k: z[k] for k in (z.files if keys is None else keys)}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_npz(path, **arrays):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    with open(tmp, 'wb') as f:
        np.savez_compressed(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(tmp), str(path))


def atomic_json(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    os.replace(str(tmp), str(path))


class PairTracker:
    """Original full-width coarse/particle arithmetic without frozen rows."""
    def __init__(self, inputs):
        self.inputs = inputs
        for k in INPUT_KEYS:
            setattr(self, k, inputs[k])
        shape = self.A.shape
        assert self.A.ndim == 2 and self.B.shape == shape
        for k in ['rawA', 'rawB', 'va', 'vb', 'supplied_dx', 'supplied_dy']:
            assert getattr(self, k).shape == shape, (k, getattr(self, k).shape)
        assert self.va.dtype == bool and self.vb.dtype == bool
        assert self.surface_a.shape == self.surface_b.shape == (shape[1],)
        assert self.points.ndim == 2 and self.points.shape[1] == 2
        assert np.isfinite(self.points).all()
        self.target_gradient = np.gradient(self.B)
        self.target_mask_float = self.vb.astype(float)

    def coarse_task(self, i):
        pt = self.points[i]
        ini = np.array([map_coordinates(s, [pt[1:2], pt[0:1]], order=1)[0]
                        for s in [self.supplied_dx, self.supplied_dy]])
        if not np.isfinite(ini).all():
            ini = np.zeros(2)
        def fit(seed):
            return fit_local(self.A, self.B, self.va, self.vb, pt, seed,
                r=19, order=1, reg=.00005, maxiter=35,
                target_gradient=self.target_gradient,
                target_mask_float=self.target_mask_float)
        p, stats = fit(ini)
        fits = [(p, stats)]
        fallback = int(stats.get('ncc', -1) < .7 or
            stats.get('mindet', 0) <= .15 or
            stats.get('support_fraction', 0) <= .7 or
            not np.isfinite(p).all())
        if fallback:
            for seed, score in translation_seed(self.A, self.B, self.va,
                                                self.vb, pt, r=19, search=(55, 20)):
                if score < .05:
                    continue
                fits.append(fit(seed))
        valid = [v for v in fits if np.isfinite(v[0]).all() and
                 v[1].get('mindet', 0) > .15 and
                 v[1].get('support_fraction', 0) > .7]
        if valid:
            p, stats = max(valid, key=lambda v: v[1].get('ncc', -1))
        else:
            finite = [v for v in fits if np.isfinite(v[0]).all()]
            p, stats = max(finite, key=lambda v: v[1].get('ncc', -1)) if finite else fits[0]
            stats = dict(stats)
            # Invalid coarse maps never contribute to the tracking prior.
            stats['ncc'] = -1.
        return p, stats, fallback

    def initialize_tracking(self, coarse):
        good = (coarse['ncc'] > .7) & np.isfinite(coarse['params']).all(axis=(1, 2))
        if good.sum() < 12:
            raise RuntimeError('Fewer than 12 usable image-derived coarse maps; '
                               'the unchanged tracking prior cannot be formed.')
        self.coarse_good_count = int(good.sum())
        self.ip = coarse['points'][good]
        self.pp = coarse['params'][good]
        self.tree = cKDTree(self.ip)
        J = np.eye(2)[None] + self.pp[:, :, 1:] / 19
        self.invJ = np.linalg.inv(J)
        self.bp = self.ip + self.pp[:, :, 0]
        self.btree = cKDTree(self.bp)
        self.TA = gaussian_filter(self.rawA, .5)
        self.TB = gaussian_filter(self.rawB, .5)
        self.track_va = self.va.astype(float)
        self.track_vb = self.vb.astype(float)
        dy, dx = np.mgrid[-4:5, -4:5]
        self.off = np.c_[dx.ravel(), dy.ravel()]
        dy, dx = np.mgrid[-8:9, -8:9]
        self.shifts = np.c_[dx.ravel(), dy.ravel()]

    def prior(self, pt, back=False):
        tree = self.btree if back else self.tree
        dist, idx = tree.query(pt, k=12)
        if back:
            pred = -self.pp[idx, :, 0] + np.einsum('nij,nj->ni',
                self.invJ[idx] - np.eye(2)[None], pt - self.bp[idx])
        else:
            pred = self.pp[idx, :, 0] + np.einsum('nij,nj->ni',
                self.pp[idx, :, 1:] / 19, pt - self.ip[idx])
        dd = np.linalg.norm(pred[:, None] - pred[None, :], axis=2)
        weights = np.exp(-dist**2 / (2 * 25**2))
        med = np.argmin(dd.dot(weights))
        sel = np.linalg.norm(pred - pred[med], axis=1) < 4
        return np.average(pred[sel], axis=0, weights=weights[sel])

    def track(self, pt, back=False):
        src, dst, vs, vd = ((self.TB, self.TA, self.track_vb, self.track_va)
                           if back else (self.TA, self.TB, self.track_va, self.track_vb))
        pr = self.prior(pt, back)
        coords = pt + self.off
        aa = map_coordinates(src, [coords[:, 1], coords[:, 0]], order=1)
        sm = map_coordinates(vs, [coords[:, 1], coords[:, 0]], order=1) > .99
        def cc(ds):
            pos = coords[None] + ds[:, None]
            bb = map_coordinates(dst, [pos[:, :, 1], pos[:, :, 0]], order=1)
            vm = map_coordinates(vd, [pos[:, :, 1], pos[:, :, 0]], order=1) > .99
            vm &= sm[None]
            n = vm.sum(axis=1)
            den = np.maximum(n, 1)
            am = (aa[None] * vm).sum(axis=1) / den
            bm = (bb * vm).sum(axis=1) / den
            az = (aa[None] - am[:, None]) * vm
            bz = (bb - bm[:, None]) * vm
            cor = (az * bz).sum(axis=1) / np.sqrt(
                (az * az).sum(axis=1) * (bz * bz).sum(axis=1) + 1e-10)
            cor[n < .85 * len(aa)] = -1
            return cor
        candidates = pr[None] + self.shifts
        cors = cc(candidates)
        scores = cors - .005 * np.sum(self.shifts**2, axis=1)
        ii = np.argsort(-scores)
        sel = [ii[0]]
        for idx in ii[1:]:
            if np.linalg.norm(candidates[idx] - candidates[sel[0]]) > 2.5:
                sel.append(idx)
                break
        fits = []
        for idx in sel:
            opt = minimize(lambda d: 1 - cc(d[None])[0] + .005 * np.sum((d - pr)**2),
                candidates[idx], method='Nelder-Mead',
                options={'maxiter': 55, 'xatol': .035, 'fatol': .0002})
            fits.append((float(opt.fun), opt.x, float(cc(opt.x[None])[0])))
        fits.sort(key=lambda v: v[0])
        best = fits[0]
        outside = np.linalg.norm(candidates - best[1][None], axis=1) > 2.5
        alternatives = [float(np.min(1 - scores[outside]))]
        alternatives += [v[0] for v in fits[1:]
                         if np.linalg.norm(v[1] - best[1]) > 2.5]
        gap = max(0., min(alternatives) - best[0])
        return best[1], best[2], gap, pr

    def detect(self):
        hp = gaussian_filter(self.rawA, .6) - gaussian_filter(self.rawA, 2)
        background = gaussian_filter(self.rawA, 5)
        yy, xx = np.indices(self.rawA.shape)
        depth = yy - self.surface_a[None]
        det = ((hp == maximum_filter(hp, size=5)) & (hp > 8) &
               (xx >= 4) & (xx < self.rawA.shape[1] - 4) &
               (yy >= 4) & (yy < self.rawA.shape[0] - 4) &
               (depth >= 14) & self.va & (background < 180))
        # No maximum depth: this pair is analyzed throughout the full image.
        return dict(points=np.c_[xx[det], yy[det]].astype(float), strength=hp[det])


class Runner:
    def __init__(self, args):
        self.args = args
        self.out = ROOT / 'work' / 'pairs' / str(args.pair)
        input_path = self.out / 'inputs.npz'
        if not input_path.is_file():
            raise FileNotFoundError('Inputs are not ready: ' + str(input_path))
        self.out.mkdir(parents=True, exist_ok=True)
        files = [Path(__file__), ROOT/'work'/'extended_flow.py',
                 ROOT/'work'/'full_width'/'fit_local_cached.py']
        identity = dict(pair=args.pair, input_sha256=sha256(input_path),
                        source_sha256={p.name: sha256(p) for p in files},
                        settings=SETTINGS, chunk_size=args.chunk_size)
        self.signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.chunks = self.out / 'tracking_checkpoints' / self.signature[:16]
        self.chunks.mkdir(parents=True, exist_ok=True)
        atomic_json(self.chunks/'manifest.json', identity)
        self.started = time.time()
        self.tracker = PairTracker(load_npz(input_path, INPUT_KEYS))
        self.stages = {}
        self.new_chunks = 0
        atomic_json(self.out/'coarse_ptv_settings.json', dict(SETTINGS,
            pair=args.pair, run_signature=self.signature,
            input_sha256=identity['input_sha256'], origin0=self.tracker.origin0.tolist(),
            image_shape=list(self.tracker.A.shape), fit_grid_nodes=len(self.tracker.points),
            detector_image_x=[4, self.tracker.A.shape[1]-5],
            detector_image_y=[4, self.tracker.A.shape[0]-5],
            chunk_size=args.chunk_size, workers=args.workers))

    def completed(self, name):
        path = self.out / (name + '.npz')
        if not path.is_file():
            return None
        z = load_npz(path)
        if str(z.get('run_signature', '')) != self.signature:
            print('Ignoring incompatible completed file:', path.name, flush=True)
            return None
        print('Reusing completed', name, 'with', len(z['points']), 'rows', flush=True)
        return z

    def write(self, name, arrays):
        atomic_npz(self.out/(name+'.npz'), **dict(arrays, run_signature=self.signature))

    def stage(self, name, points, task, pack):
        """A chunk is either fully committed or recomputed after interruption."""
        count = len(points)
        pieces = []
        started = time.time()
        resumed = 0
        with ThreadPoolExecutor(max_workers=self.args.workers) as pool:
            for start in range(0, count, self.args.chunk_size):
                stop = min(count, start+self.args.chunk_size)
                path = self.chunks / ('%s_%07d_%07d.npz' % (name, start, stop))
                if path.is_file():
                    piece = load_npz(path)
                    if not np.array_equal(piece['points'], points[start:stop]):
                        raise RuntimeError('Checkpoint point-order mismatch: '+str(path))
                    resumed += stop-start
                else:
                    if self.args.max_chunks and self.new_chunks >= self.args.max_chunks:
                        print('Requested chunk limit reached; committed progress can be resumed.', flush=True)
                        return None
                    results = list(pool.map(task, range(start, stop)))
                    piece = pack(results)
                    piece['points'] = points[start:stop]
                    atomic_npz(path, **piece)
                    self.new_chunks += 1
                pieces.append(piece)
                print('Pair', self.args.pair, name, stop, '/', count,
                      'rows; resumed', resumed, '; elapsed', round(time.time()-started, 1),
                      'seconds', flush=True)
        if not pieces:
            raise RuntimeError('No particle candidates or fitting points for '+name)
        data = {k: np.concatenate([piece[k] for piece in pieces], axis=0)
                for k in pieces[0]}
        self.stages[name] = dict(rows=count, resumed_rows=resumed, seconds=time.time()-started)
        return data

    def run(self):
        tr = self.tracker
        coarse = self.completed('coarse_affine')
        if coarse is None:
            def pack_coarse(result):
                return dict(params=np.array([v[0] for v in result]),
                    coarse_fallback=np.array([v[2] for v in result]),
                    **{k: np.array([v[1].get(k, np.nan) for v in result]) for k in STATS})
            coarse = self.stage('coarse', tr.points, tr.coarse_task, pack_coarse)
            if coarse is None:
                return False
            coarse.update(radius=np.array(19), order=np.array(1), reg=np.array(.00005),
                          origin0=tr.origin0)
            self.write('coarse_affine', coarse)
        print('Coarse maps complete:', len(coarse['points']), '; usable prior maps:',
              int(np.sum((coarse['ncc']>.7)&np.isfinite(coarse['params']).all(axis=(1,2)))), flush=True)
        if self.args.stop_after == 'coarse':
            return True
        tracks = self.completed('ptv_tracks')
        if tracks is not None:
            return True
        tr.initialize_tracking(coarse)
        candidates = self.completed('particle_candidates')
        if candidates is None:
            candidates = tr.detect()
            self.write('particle_candidates', candidates)
        pts = candidates['points']
        strength = candidates['strength']
        print('Particle candidates:', len(pts), '; full image depth; no frozen rows', flush=True)
        def pack_track(result):
            return dict(disp=np.array([v[0] for v in result]),
                        ncc=np.array([v[1] for v in result]),
                        gap=np.array([v[2] for v in result]),
                        prior=np.array([v[3] for v in result]))
        fw = self.completed('tracks_forward')
        if fw is None:
            fw = self.stage('forward', pts, lambda i: tr.track(pts[i]), pack_track)
            if fw is None:
                return False
            fw['strength'] = strength
            self.write('tracks_forward', fw)
        if self.args.stop_after == 'forward':
            return True
        dest = pts + fw['disp']
        bw = self.completed('tracks_reverse')
        if bw is None:
            bw = self.stage('reverse', dest, lambda i: tr.track(dest[i], True), pack_track)
            if bw is None:
                return False
            self.write('tracks_reverse', bw)
        fb = np.linalg.norm(fw['disp'] + bw['disp'], axis=1)
        accepted = ((fw['ncc'] > .68) & (bw['ncc'] > .68) & (fb < 1) &
                    (fw['gap'] > .015) & (bw['gap'] > .01))
        depth = pts[:, 1] - np.interp(pts[:, 0], np.arange(len(tr.surface_a)), tr.surface_a)
        tracks = dict(points=pts, disp=fw['disp'], prior=fw['prior'],
            ncc=fw['ncc'], ambiguity_gap=fw['gap'], back_disp=bw['disp'],
            back_ncc=bw['ncc'], back_gap=bw['gap'], fb=fb, accepted=accepted,
            strength=strength, source_depth=depth, origin0=tr.origin0)
        self.write('ptv_tracks', tracks)
        summary = dict(pair=self.args.pair, total_candidate_rows=len(pts),
            total_accepted_tracks=int(accepted.sum()), coarse_count=len(tr.points),
            coarse_good_prior_fits=tr.coarse_good_count,
            coarse_fallback_count=int(coarse['coarse_fallback'].sum()),
            accepted_source_depth_range=([float(depth[accepted].min()),
                float(depth[accepted].max())] if accepted.any() else None),
            full_image_depth=True, manual_data_used=False, frozen_previous_rows=0,
            run_signature=self.signature, stages_this_invocation=self.stages,
            invocation_seconds=time.time()-self.started)
        atomic_json(self.out/'coarse_ptv_summary.json', summary)
        print(json.dumps(summary, indent=2), flush=True)
        return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pair', type=int, choices=[80, 100], required=True)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--chunk-size', type=int, default=512)
    p.add_argument('--stop-after', choices=['coarse', 'forward', 'reverse'], default='reverse')
    p.add_argument('--max-chunks', type=int, default=0,
                   help='Optional bounded run: maximum new chunks before returning (0 means unlimited).')
    args = p.parse_args()
    if args.workers < 1 or args.chunk_size < 1 or args.max_chunks < 0:
        p.error('workers/chunk-size must be positive; max-chunks must be nonnegative')
    completed = Runner(args).run()
    return 0 if completed else 2


if __name__ == '__main__':
    sys.exit(main())
