"""Compare complete fresh-image tracking branches; no external references."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

BASE=Path(__file__).resolve().parent


def read(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def metrics(values):
    a=np.asarray(values);a=a[np.isfinite(a)]
    if not len(a):return dict(n=0)
    return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),
        p95=float(np.percentile(a,95)),p99=float(np.percentile(a,99)),maximum=float(a.max()),
        above_point01=int(np.sum(a>.01)),above_point1=int(np.sum(a>.1)),
        above_point5=int(np.sum(a>.5)),above1=int(np.sum(a>1)))


def gates(z):
    return dict(forward_ncc=z['ncc']>.68,reverse_ncc=z['back_ncc']>.68,
                reciprocal_error=z['fb']<1.,forward_gap=z['ambiguity_gap']>.015,
                reverse_gap=z['back_gap']>.01)


def run(pair):
    folder=BASE/str(pair);paths=[folder/'ptv_tracks.npz',folder/'bootstrap48'/'ptv_tracks.npz']
    if not all(p.is_file() for p in paths):return dict(pair=pair,status='pending')
    initial_hash=[sha(p) for p in paths];a,b=[read(p) for p in paths]
    for z in [a,b]:
        assert bool(z['image_only']) and not bool(z['supplied_velocity_used'])
        assert not bool(z['previous_models_or_tracks_used'])
        assert np.array_equal(z['accepted'],np.logical_and.reduce(list(gates(z).values())))
        assert np.allclose(z['fb'],np.linalg.norm(z['disp']+z['back_disp'],axis=1),rtol=0,atol=0,equal_nan=True)
    assert str(a['input_sha256'])==str(b['input_sha256'])
    for key in ['points','strength','source_depth','origin0']:
        assert np.array_equal(a[key],b[key]),key
    assert int(a['bootstrap_radius'])==64 and int(b['bootstrap_radius'])==48
    x=a['accepted'];y=b['accepted'];both=x&y;either=x|y;changed=x!=y
    delta=np.linalg.norm(a['disp']-b['disp'],axis=1)
    backdelta=np.linalg.norm(a['back_disp']-b['back_disp'],axis=1)
    near=a['source_depth']<40;report_depth=a['source_depth']<=float(a['requested_max_depth_px'])
    prior=read(folder/'tracking_prior_bootstrap_audit.npz')
    assert np.array_equal(prior['points'],a['points'])
    affected=prior['difference_px']>.1
    ag=gates(a);bg=gates(b)
    def details(i):
        return dict(index0=int(i),x=float(a['points'][i,0]),y=float(a['points'][i,1]),
            depth=float(a['source_depth'][i]),prior_difference=float(prior['difference_px'][i]),
            displacement_difference=float(delta[i]),accepted64=bool(x[i]),accepted48=bool(y[i]),
            displacement64=a['disp'][i].tolist(),displacement48=b['disp'][i].tolist(),
            ncc64=float(a['ncc'][i]),ncc48=float(b['ncc'][i]),
            fb64=float(a['fb'][i]),fb48=float(b['fb'][i]),
            failed64=[k for k,v in ag.items() if not v[i]],failed48=[k for k,v in bg.items() if not v[i]])
    greatest=np.flatnonzero(both)[np.argsort(delta[both])[::-1]][:20]
    report=dict(pair=pair,status='passed',same_fresh_image_input=True,
        no_supplied_velocities_or_old_fields=True,all_candidate_points_strengths_and_depths_identical=True,
        candidate_count=len(x),accepted64=int(x.sum()),accepted48=int(y.sum()),
        accepted_intersection=int(both.sum()),primary_only=int((x&~y).sum()),
        bootstrap48_only=int((y&~x).sum()),both_rejected=int((~x&~y).sum()),
        acceptance_state_changes=int(changed.sum()),
        acceptance_state_changes_below40=int(np.sum(changed&near)),
        acceptance_state_changes_within_requested_cm=int(np.sum(changed&report_depth)),
        accepted_intersection_displacement_difference=metrics(delta[both]),
        accepted_intersection_near_surface_difference=metrics(delta[both&near]),
        accepted_intersection_within_requested_cm_difference=metrics(delta[both&report_depth]),
        accepted_union_displacement_difference=metrics(delta[either]),
        accepted_intersection_reverse_displacement_difference=metrics(backdelta[both]),
        prior_affected_count=int(affected.sum()),prior_affected_state_changes=int(np.sum(affected&changed)),
        prior_affected_common_accepted_difference=metrics(delta[affected&both]),
        affected_prior_track_details=[details(i) for i in np.flatnonzero(affected)],
        changed_acceptance_details=[details(i) for i in np.flatnonzero(changed)],
        largest_common_accepted_displacement_differences=[details(i) for i in greatest],
        source_sha256=dict(primary=initial_hash[0],bootstrap48=initial_hash[1]),
        bootstrap_note='Fresh image-only initialization sensitivity; agreement is not independent truth validation.')
    assert initial_hash==[sha(p) for p in paths]
    report['source_files_unchanged']=True
    np.savez_compressed(folder/'bootstrap_track_comparison.npz',points=a['points'],depth=a['source_depth'],
        accepted64=x,accepted48=y,disp64=a['disp'],disp48=b['disp'],
        displacement_difference=delta,reverse_displacement_difference=backdelta,
        affected_priors=affected,prior_difference=prior['difference_px'],
        accepted_intersection=both,acceptance_changed=changed,
        primary_only=x&~y,bootstrap48_only=y&~x)
    (folder/'bootstrap_track_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pair',type=int,choices=[80,100],required=True)
    r=run(p.parse_args().pair)
    print(json.dumps({k:v for k,v in r.items() if not k.endswith('_details') and k!='largest_common_accepted_displacement_differences'},indent=2))
