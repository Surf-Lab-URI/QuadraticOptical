"""Signed translation, mask, image-boundary, provenance and cache tests."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import json,time,tempfile
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
import tracking as m


def translated(a,dx,dy,fill=0):
    b=np.full_like(a,fill);h,w=a.shape
    sx0=max(0,-dx);sx1=min(w,w-dx);sy0=max(0,-dy);sy1=min(h,h-dy)
    b[sy0+dy:sy1+dy,sx0+dx:sx1+dx]=a[sy0:sy1,sx0:sx1]
    return b


def main():
    started=time.time();rng=np.random.RandomState(20260911);shape=(320,320)
    spikes=np.zeros(shape);iy=rng.randint(0,320,2400);ix=rng.randint(0,320,2400)
    np.add.at(spikes,(iy,ix),rng.uniform(40,200,len(ix)))
    A=gaussian_filter(spikes,.9)+rng.uniform(0,.2,shape)
    mask=np.ones(shape,bool);mask[:20]=False
    cases=[]
    for dx,dy,center in [(23,-17,[160,160]),(-29,19,[160,160]),
             (-57,-45,[160,160]),(61,55,[160,160]),(0,0,[160,160]),
             (-64,0,[160,160]),(64,0,[160,160]),(0,-64,[160,160]),(0,64,[160,160]),
             (15,6,[7,90]),(-23,-8,[312,160]),(9,-12,[160,312])]:
        B=translated(A,dx,dy);vb=translated(mask,dx,dy,False)
        score,count,bounds,source_n,required=m.translation_ncc(A,B,mask,vb,center)
        seeds,scores,counts=m.peaks(score,count,bounds,64)
        assert np.array_equal(seeds[0],[dx,dy]),(dx,dy,center,seeds,scores)
        assert abs(scores[0]-1)<1e-10
        assert counts[0]>=required-1e-8
        # Compare a few NCC values to an independent direct common-pixel sum.
        cx,cy=np.round(center).astype(int);x0=max(0,cx-19);x1=min(320,cx+20)
        y0=max(0,cy-19);y1=min(320,cy+20)
        errors=[]
        for sx,sy in [(dx,dy),(0,0),(int(bounds[0]),int(bounds[2]))]:
            valid=mask[y0:y1,x0:x1]&vb[y0+sy:y1+sy,x0+sx:x1+sx]
            if valid.sum()+1e-8<required:continue
            aa=A[y0:y1,x0:x1][valid];bb=B[y0+sy:y1+sy,x0+sx:x1+sx][valid]
            aa=aa-aa.mean();bb=bb-bb.mean()
            direct=np.dot(aa,bb)/np.sqrt(max(np.dot(aa,aa)*np.dot(bb,bb),1e-12))
            errors.append(abs(direct-score[sy-bounds[2],sx-bounds[0]]))
        assert max(errors)<1e-10
        cases.append(dict(shift=[dx,dy],center=center,recovered=seeds[0].tolist(),
                          bounds=bounds.tolist(),ncc=float(scores[0]),max_direct_ncc_error=max(errors)))
    # Artificial search limits are truly symmetric; image clipping is separate.
    assert cases[0]['bounds']==[-64,64,-64,64]
    assert cases[9]['bounds'][0]==0 and cases[10]['bounds'][1]==0 and cases[11]['bounds'][3]==0
    # A perfect-intensity match with only 25 common pixels is not admissible.
    tiny=np.zeros(shape,bool);tiny[158:163,158:163]=True
    score,count,bounds,source_n,required=m.translation_ncc(A,A,tiny,tiny,[160,160])
    assert source_n==25 and required==40 and np.all(score==-2)
    invalid_seeds,_,_=m.peaks(score,count,bounds,64)
    assert np.isnan(invalid_seeds).all()
    corrupted=score.copy();corrupted[0,0]=np.nan;corrupted[1,1]=np.inf
    assert np.isnan(m.peaks(corrupted,count,bounds,64)[0]).all()
    # A broadly available source paired with a narrow visible target strip
    # also cannot win from a high correlation on a tiny overlap.
    strip=np.zeros(shape,bool);strip[:,158:163]=True
    score,count,bounds,source_n,required=m.translation_ncc(A,A,mask,strip,[160,160])
    assert np.all(score==-2) and np.max(count)<required
    # Full coarse solution and cached wide/narrow duplication check.
    dx,dy=23,-17;B=translated(A,dx,dy);vb=translated(mask,dx,dy,False)
    inp=dict(A=A,B=B,rawA=A,rawB=B,va=mask,vb=vb,surface_a=np.full(320,10.),
             surface_b=np.full(320,10.+dy),points=np.array([[160.,160.]]),origin0=np.array([0,0]),
             DX=np.array(5.65e-5),DT=np.array(.01),requested_max_depth_px=np.array(177.),
             fitting_max_depth_px=np.array(241.),detector_max_depth_px=np.array(266.),
             image_only=np.array(True),supplied_velocity_used=np.array(False))
    tr=m.ImageOnlyTracker(inp);t=time.time();coarse=tr.coarse_task(0);seconds=time.time()-t
    assert np.linalg.norm(coarse['params'][:,0]-[dx,dy])<1e-8
    assert coarse['selected_admissible'] and coarse['ncc']>.999999
    assert coarse['unique_seed_fit_count']==len(set(map(tuple,coarse['candidate_seeds'])))
    assert np.linalg.norm(coarse['bootstrap48_params'][:,0]-[dx,dy])<1e-8
    invalid_stats=dict(ncc=np.nan,mindet=1.,support_fraction=1.)
    valid_stats=dict(ncc=.9,mindet=1.,support_fraction=1.)
    assert not m.admissible(np.zeros((2,3)),invalid_stats)
    assert m.best_candidate([(np.zeros((2,3)),invalid_stats),
                             (np.zeros((2,3)),valid_stats)],[0,1])==(1,True)
    tiny_tracker=m.ImageOnlyTracker(dict(inp,va=tiny,vb=tiny,A=A,B=A))
    tiny_fit=tiny_tracker.coarse_task(0)
    assert tiny_fit['unique_seed_fit_count']==1
    assert np.array_equal(tiny_fit['candidate_seed_valid'],[False]*6+[True])
    assert not tiny_fit['selected_admissible'] and tiny_fit['selected_candidate']==6
    # Whitelisted images load, metadata marker is permitted only when false,
    # and introducing any velocity array is a hard error.
    with tempfile.TemporaryDirectory(prefix='image_only_guard_') as td:
        path=Path(td)/'inputs.npz';np.savez(path,**inp);m.load_inputs(path)
        bad=dict(inp,supplied_dx=np.zeros(shape));np.savez(path,**bad)
        try:m.load_inputs(path)
        except ValueError:pass
        else:raise AssertionError('Forbidden velocity array was accepted')
        bad=dict(inp,supplied_velocity_used=np.array(True));np.savez(path,**bad)
        try:m.load_inputs(path)
        except ValueError:pass
        else:raise AssertionError('Velocity-used marker was accepted')
    report=dict(passed=True,signed_translation_cases=cases,
        symmetric_artificial_bounds=True,image_clipping_bounds_correct=True,
        tiny_common_overlap_rejected=True,velocity_input_guard_passed=True,
        invalid_and_nonfinite_peaks_never_fitted=True,nonfinite_final_ncc_rejected=True,
        full_affine_coarse_translation_error=float(np.linalg.norm(coarse['params'][:,0]-[dx,dy])),
        candidate_cache_unique_fits=int(coarse['unique_seed_fit_count']),
        candidate_slots=len(coarse['candidate_seeds']),synthetic_coarse_seconds=seconds,
        no_real_pair_velocity_or_track_data_read=True,total_seconds=time.time()-started)
    out=Path(__file__).resolve().parent
    (out/'tracking_synthetic_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
