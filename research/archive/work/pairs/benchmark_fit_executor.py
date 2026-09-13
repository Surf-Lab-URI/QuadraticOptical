"""Bounded mixed-variant solver benchmark without changing production files."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.spatial import cKDTree

import fit_pairs
from process_executor import ForkProcessExecutor

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'work'/'pairs'


def exact(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def main():
    started = time.time()
    inp = fit_pairs.inputs_from(OUT/'80'/'inputs.npz')
    coarse = fit_pairs.read_npz(OUT/'80'/'coarse_affine.npz')
    # These initial values only benchmark executor equivalence. They are not
    # the production PTV prior and are never written into fitting checkpoints.
    grid = inp['points']; tree = cKDTree(grid)
    queries = []
    for depth in [14., 180., 800., 1630.]:
        for x in [15., 675., 1365., 2047.]:
            surface = np.interp(x, np.arange(len(inp['surface_a'])), inp['surface_a'])
            queries.append([x, surface+depth])
    idx = tree.query(np.array(queries))[1]
    assert len(np.unique(idx)) == 16
    points = grid[idx]
    source_p = coarse['params'][idx]
    seeds = np.nan_to_num(source_p, nan=0., posinf=0., neginf=0.)
    caches = {}
    for margin in [10, 14]:
        A, B, va, vb = fit_pairs.images(inp, margin)
        caches[margin] = (A, B, va, vb, np.gradient(B), vb.astype(float))
    A, B, va, vb, _, _ = caches[10]
    reverse_cache = (B, A, vb, va, np.gradient(A), va.astype(float))
    def forward(stage, j):
        r, order, margin = fit_pairs.VARIANTS[stage]
        A, B, va, vb, grad, mask = caches[margin]
        p = seeds[j].copy(); p[:, 1:3] *= r/19
        return fit_pairs.fit_local(A, B, va, vb, points[j], p[:, 0], r=r,
            order=order, reg=5e-5, maxiter=35, seed_affine=p,
            target_gradient=grad, target_mask_float=mask)
    # Reverse initialization is exactly the production driver formula applied
    # to these bounded representative primary fits.
    initial_main = [forward('main', j)[0] for j in range(len(points))]
    cases = [(stage, j) for stage in fit_pairs.VARIANTS for j in range(len(points))]
    cases += [('reverse', j) for j in range(len(points))]
    def task(i):
        stage, j = cases[i]
        if stage != 'reverse':
            return forward(stage, j)
        pp = initial_main[j]
        if not np.isfinite(pp).all():
            return np.full((2, 6), np.nan), {'ncc': -1.}
        p = np.zeros((2, 3)); p[:, 0] = -pp[:, 0]
        try:
            p[:, 1:3] = (np.linalg.inv(np.eye(2)+pp[:, 1:3]/13)-np.eye(2))*13
        except np.linalg.LinAlgError:
            return np.full((2, 6), np.nan), {'ncc': -1.}
        A, B, va, vb, grad, mask = reverse_cache
        return fit_pairs.fit_local(A, B, va, vb, points[j]+pp[:, 0], p[:, 0],
            r=13, order=2, reg=5e-5, maxiter=35, seed_affine=p,
            target_gradient=grad, target_mask_float=mask)
    report = dict(pair=80, unique_points=16, tasks=len(cases), workers=4,
        stages=list(fit_pairs.VARIANTS)+['reverse'],
        initialization='Pair-80 coarse affine maps, only for executor benchmark; production PTV initialization unchanged',
        production_jobs_not_interrupted=True,
        fit_driver_sha256=fit_pairs.file_hash(OUT/'fit_pairs.py'),
        solver_sha256=fit_pairs.file_hash(fit_pairs.SOLVER_PATH),
        preparation_seconds=time.time()-started, timings=[],
        load_average_start=list(os.getloadavg()), cpu_count=os.cpu_count())
    reference = None
    for backend in ['threads', 'fork', 'fork', 'threads']:
        cls = ThreadPoolExecutor if backend == 'threads' else ForkProcessExecutor
        t = time.perf_counter()
        with cls(max_workers=4) as pool:
            result = list(pool.map(task, range(len(cases))))
        elapsed = time.perf_counter()-t
        if reference is None:
            reference = result
        for i, ((p0, s0), (p1, s1)) in enumerate(zip(reference, result)):
            assert exact(p0, p1), ('parameters', cases[i])
            assert s0.keys() == s1.keys(), ('diagnostic keys', cases[i])
            for key in s0:
                assert exact(s0[key], s1[key]), (key, cases[i], s0[key], s1[key])
        rec=dict(backend=backend, seconds=elapsed, fits_per_second=len(cases)/elapsed,
                 all_parameters_and_diagnostics_bitwise_equal=True)
        report['timings'].append(rec);print(json.dumps(rec),flush=True)
    ts = [v['seconds'] for v in report['timings'] if v['backend']=='threads']
    fs = [v['seconds'] for v in report['timings'] if v['backend']=='fork']
    report.update(thread_seconds_mean=float(np.mean(ts)),fork_seconds_mean=float(np.mean(fs)),
        mean_speedup=float(np.mean(ts)/np.mean(fs)),elapsed_seconds=time.time()-started,
        all_results_bitwise_equal=True,
        nonfinite_parameter_cases=int(sum(not np.isfinite(v[0]).all() for v in reference)),
        load_average_end=list(os.getloadavg()))
    data=dict(points=points,source_grid_indices=idx)
    for stage in list(fit_pairs.VARIANTS)+['reverse']:
        rows=[i for i,c in enumerate(cases) if c[0]==stage]
        data[stage+'_params']=np.array([reference[i][0] for i in rows])
        for key in fit_pairs.STATS:
            data[stage+'_'+key]=np.array([reference[i][1].get(key,np.nan) for i in rows])
    np.savez_compressed(OUT/'fit_executor_benchmark.npz',**data)
    (OUT/'fit_executor_benchmark.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
