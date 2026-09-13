"""Compare both fresh image-only coarse bootstraps at every particle candidate.

Uses the exact production prior implementation. No particle fit is rerun, and
no primary field/checkpoint or source code is changed. This measures how the
coarse bootstrap choice changes forward tracking initialization, not calibrated
final-field uncertainty.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import argparse,json,time
from pathlib import Path
import numpy as np
import tracking as m


def run(pair):
    started=time.time();folder=Path(__file__).resolve().parent/str(pair)
    path=folder/'coarse_search.npz'
    if not path.is_file():return dict(pair=pair,status='pending')
    inp=m.load_inputs(folder/'inputs.npz');coarse=m.read(path)
    assert bool(coarse['image_only']) and not bool(coarse['supplied_velocity_used'])
    assert str(coarse['input_sha256'])==m.sha(folder/'inputs.npz')
    tr=m.ImageOnlyTracker(inp)
    if (folder/'particle_candidates.npz').is_file():
        candidates=m.read(folder/'particle_candidates.npz')
        assert bool(candidates['image_only']) and not bool(candidates['supplied_velocity_used'])
        assert str(candidates['input_sha256'])==str(coarse['input_sha256'])
    else:
        candidates=tr.detect()
    pts=candidates['points'];depth=pts[:,1]-np.interp(pts[:,0],np.arange(len(tr.surface_a)),tr.surface_a)
    outputs={}
    for radius in [64,48]:
        tr.initialize_tracking(m.coarse_view(coarse,radius))
        values=np.array([tr.prior(pt,False) for pt in pts])
        assert np.isfinite(values).all()
        outputs[radius]=values
        print('Image-only prior audit pair',pair,'radius',radius,'candidates',len(pts),flush=True)
    delta=np.linalg.norm(outputs[64]-outputs[48],axis=1)
    ids=np.flatnonzero(delta>.1)
    order=np.argsort(delta)[::-1]
    report=dict(pair=pair,status='passed',candidate_count=len(pts),all_priors_finite=True,
        exact_production_prior_used=True,wide_radius=64,narrow_radius=48,
        mean_difference_px=float(delta.mean()),median_difference_px=float(np.median(delta)),
        p95_difference_px=float(np.percentile(delta,95)),p99_difference_px=float(np.percentile(delta,99)),
        max_difference_px=float(delta.max()),count_over_point1_px=int(np.sum(delta>.1)),
        count_over1_px=int(np.sum(delta>1)),count_over2point5_px=int(np.sum(delta>2.5)),
        near_surface_below40_count_over_point1_px=int(np.sum((depth<40)&(delta>.1))),
        within_requested_cm_count_over_point1_px=int(np.sum((depth<=float(tr.requested_max_depth_px))&(delta>.1))),
        bootstrap_note='Actual forward tracking seed sensitivity only; no tracking/final-field uncertainty claim.',
        input_sha256=str(coarse['input_sha256']),coarse_sha256=m.sha(path),
        seconds=time.time()-started,
        largest_differences=[dict(index0=int(i),x=float(pts[i,0]),y=float(pts[i,1]),
            depth=float(depth[i]),difference=float(delta[i]),prior64=outputs[64][i].tolist(),
            prior48=outputs[48][i].tolist()) for i in order[:20]])
    np.savez_compressed(folder/'tracking_prior_bootstrap_audit.npz',points=pts,depth=depth,
        prior64=outputs[64],prior48=outputs[48],difference_px=delta,
        indices_over_point1_px=ids,coordinates_over_point1_px=pts[ids],
        indices_over1_px=np.flatnonzero(delta>1),coordinates_over1_px=pts[delta>1])
    (folder/'tracking_prior_bootstrap_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pair',type=int,choices=[80,100],required=True)
    args=p.parse_args();r=run(args.pair)
    print(json.dumps({k:v for k,v in r.items() if k!='largest_differences'},indent=2))
