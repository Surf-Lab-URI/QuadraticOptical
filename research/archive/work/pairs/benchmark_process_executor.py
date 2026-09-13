"""Bounded exact-output benchmark, leaving all production stages untouched."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from track_pairs import PairTracker, load_npz, INPUT_KEYS, sha256
from process_executor import ForkProcessExecutor

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'work'/'pairs'


def packed(results):
    return np.array([np.r_[v[0], v[1], v[2], v[3]] for v in results], dtype=np.float64)


def main():
    started = time.time()
    tr = PairTracker(load_npz(OUT/'80'/'inputs.npz', INPUT_KEYS))
    tr.initialize_tracking(load_npz(OUT/'80'/'coarse_affine.npz'))
    candidates = load_npz(OUT/'80'/'particle_candidates.npz')
    points = candidates['points']
    depth = points[:, 1]-np.interp(points[:, 0], np.arange(len(tr.surface_a)), tr.surface_a)
    # One representative per 8 by 8 horizontal/depth cell, selected from image
    # coordinates only. Do not select by NCC, manual errors or optimizer cost.
    xb = np.linspace(0, tr.A.shape[1], 9)
    db = np.linspace(depth.min(), depth.max()+1e-8, 9)
    indices = []
    for iy in range(8):
        for ix in range(8):
            ii = np.flatnonzero((points[:, 0]>=xb[ix]) & (points[:, 0]<xb[ix+1]) &
                               (depth>=db[iy]) & (depth<db[iy+1]))
            if not len(ii):
                continue
            dx = (points[ii, 0]-(xb[ix]+xb[ix+1])/2)/(xb[ix+1]-xb[ix])
            dy = (depth[ii]-(db[iy]+db[iy+1])/2)/(db[iy+1]-db[iy])
            indices.append(int(ii[np.argmin(dx*dx+dy*dy)]))
    indices = np.array(indices)
    assert 0 < len(indices) <= 64 and len(np.unique(indices)) == len(indices)
    query = points[indices]
    task = lambda i: tr.track(query[i])
    report = dict(pair=80, sample_count=len(indices), indices=indices.tolist(),
                  workers=4, production_jobs_not_interrupted=True,
                  tracking_source_sha256=sha256(OUT/'track_pairs.py'),
                  cpu_count=os.cpu_count(), load_average_start=list(os.getloadavg()),
                  preparation_seconds=time.time()-started, timings=[])
    reference = None
    saved = {}
    # Reverse the order in the second round to limit ordering/warm-cache bias.
    for run, backend in enumerate(['threads', 'fork', 'fork', 'threads']):
        cls = ThreadPoolExecutor if backend == 'threads' else ForkProcessExecutor
        t = time.perf_counter()
        with cls(max_workers=4) as pool:
            result = packed(list(pool.map(task, range(len(query)))))
        elapsed = time.perf_counter()-t
        if reference is None:
            reference = result.copy()
        exact = np.array_equal(reference.view(np.uint64), result.view(np.uint64))
        assert exact, 'The executor changed the returned numerical values.'
        key=backend+'_'+str(run)
        saved[key] = result
        rec=dict(backend=backend, seconds=elapsed,
                 candidates_per_second=len(query)/elapsed, bitwise_equal=bool(exact))
        report['timings'].append(rec)
        print(json.dumps(rec), flush=True)
    thread_times=[r['seconds'] for r in report['timings'] if r['backend']=='threads']
    fork_times=[r['seconds'] for r in report['timings'] if r['backend']=='fork']
    report.update(thread_seconds_mean=float(np.mean(thread_times)),
                  fork_seconds_mean=float(np.mean(fork_times)),
                  mean_speedup=float(np.mean(thread_times)/np.mean(fork_times)),
                  all_results_bitwise_equal=True,
                  load_average_end=list(os.getloadavg()),
                  elapsed_seconds=time.time()-started)
    np.savez_compressed(OUT/'process_executor_benchmark.npz',indices=indices,points=query,**saved)
    (OUT/'process_executor_benchmark.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
