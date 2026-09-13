"""Independent detector, reciprocal closure, acceptance, and photometry audit.

This reads completed tracking files and recomputes diagnostics without calling
the tracking optimizer or altering any model/acceptance values. Raw-patch NCC
is reconstructed independently at the saved forward and reverse displacements.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import argparse
import json
import time
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter, map_coordinates

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'work'/'pairs'


def read(path, keys=None):
    with np.load(path, allow_pickle=False) as z:
        return {k:z[k] for k in (z.files if keys is None else keys)}


def fixed_sample(points, depth, tracks):
    """Coordinate strata plus predeclared diagnostic/boundary cases."""
    chosen = set()
    xedges = np.linspace(0, 2048, 9)
    dedges = np.linspace(depth.min(), depth.max()+1e-8, 9)
    for y in range(8):
        for x in range(8):
            inside = ((points[:,0]>=xedges[x])&(points[:,0]<xedges[x+1])&
                      (depth>=dedges[y])&(depth<dedges[y+1]))
            for accepted in [False, True]:
                rows = np.flatnonzero(inside & (tracks['accepted']==accepted))
                if len(rows):
                    chosen.add(int(rows[len(rows)//3]))
                    chosen.add(int(rows[(2*len(rows))//3]))
    for values in [depth,points[:,0],points[:,1],tracks['ncc'],tracks['back_ncc'],
                   tracks['fb'],np.abs(tracks['ncc']-.68),
                   np.abs(tracks['ambiguity_gap']-.015),np.abs(tracks['back_gap']-.01)]:
        order=np.argsort(values)
        chosen.update(int(i) for i in order[:8])
        chosen.update(int(i) for i in order[-8:])
    return np.array(sorted(chosen),dtype=int)


def independent_ncc(src, dst, smask, dmask, point, displacement, offsets):
    source = point+offsets
    target = source+displacement
    a = map_coordinates(src,[source[:,1],source[:,0]],order=1,mode='constant',cval=0)
    b = map_coordinates(dst,[target[:,1],target[:,0]],order=1,mode='constant',cval=0)
    visible = ((map_coordinates(smask,[source[:,1],source[:,0]],order=1,
                                mode='constant',cval=0)>.99)&
               (map_coordinates(dmask,[target[:,1],target[:,0]],order=1,
                                mode='constant',cval=0)>.99))
    n = int(visible.sum())
    if n < .85*len(offsets):
        return -1., n
    a = a[visible]; b = b[visible]
    ac = a-a.mean(); bc = b-b.mean()
    ncc = np.dot(ac,bc)/np.sqrt(np.dot(ac,ac)*np.dot(bc,bc)+1e-10)
    return float(ncc), n


def run(pair):
    started=time.time(); directory=BASE/str(pair)
    path=directory/'ptv_tracks.npz'
    if not path.is_file():
        return dict(pair=pair,status='pending',reason='Complete atomic ptv_tracks.npz not present')
    inp=read(directory/'inputs.npz',['rawA','rawB','va','vb','surface_a','origin0'])
    tr=read(path)
    points=tr['points'];n=len(points);rawA=inp['rawA'];rawB=inp['rawB']
    va=inp['va'];vb=inp['vb'];sa=inp['surface_a']
    assert np.array_equal(inp['origin0'],[0,0])
    assert np.array_equal(tr['origin0'],inp['origin0'])
    for key in ['disp','prior','ncc','ambiguity_gap','back_disp','back_ncc','back_gap',
                'fb','accepted','strength','source_depth']:
        assert len(tr[key])==n,(key,len(tr[key]),n)
    assert points.shape==(n,2) and np.isfinite(points).all()
    assert np.unique(points,axis=0).shape[0]==n
    assert np.array_equal(points,points.astype(int))
    assert tr['accepted'].dtype==bool
    accepted=tr['accepted'];dest=points+tr['disp']
    assert np.isfinite(tr['disp'][accepted]).all()
    assert np.isfinite(tr['back_disp'][accepted]).all()
    # Detector is recreated independently; no old candidate list is reused.
    hp=gaussian_filter(rawA,.6)-gaussian_filter(rawA,2)
    bg=gaussian_filter(rawA,5)
    yy,xx=np.indices(rawA.shape);depthmap=yy-sa[None,:]
    detector=((hp==maximum_filter(hp,size=5))&(hp>8)&(bg<180)&va&
              (xx>=4)&(xx<rawA.shape[1]-4)&(yy>=4)&(yy<rawA.shape[0]-4)&
              (depthmap>=14))
    recomputed_points=np.column_stack(np.nonzero(detector)[::-1]).astype(float)
    assert np.array_equal(points,recomputed_points),'Detector positions/order differ'
    assert np.array_equal(tr['strength'],hp[detector]),'Detector peak strengths differ'
    depth=points[:,1]-np.interp(points[:,0],np.arange(len(sa)),sa)
    assert np.allclose(depth,tr['source_depth'],rtol=0,atol=0,equal_nan=True)
    fb=np.linalg.norm(tr['disp']+tr['back_disp'],axis=1)
    assert np.allclose(fb,tr['fb'],rtol=0,atol=0,equal_nan=True),'Reciprocity changed'
    recovered=((tr['ncc']>.68)&(tr['back_ncc']>.68)&(fb<1.)&
               (tr['ambiguity_gap']>.015)&(tr['back_gap']>.01))
    assert np.array_equal(recovered,accepted),'Acceptance thresholds differ'
    # All source-center masks must hold by detector construction. Target-center
    # availability is reported, not added as a new particle acceptance rule.
    assert va[points[:,1].astype(int),points[:,0].astype(int)].all()
    finite_dest=np.isfinite(dest).all(axis=1)
    target_center_visible=np.zeros(n,bool)
    target_center_visible[finite_dest]=map_coordinates(vb.astype(float),
        [dest[finite_dest,1],dest[finite_dest,0]],order=1,mode='constant',cval=0)>.99
    sample=fixed_sample(points,depth,tr)
    TA=gaussian_filter(rawA,.5);TB=gaussian_filter(rawB,.5)
    av=va.astype(float);bv=vb.astype(float)
    dy,dx=np.mgrid[-4:5,-4:5];off=np.column_stack([dx.ravel(),dy.ravel()])
    fncc=[];bncc=[];fn=[];bn=[]
    for i in sample:
        a,na=independent_ncc(TA,TB,av,bv,points[i],tr['disp'][i],off)
        b,nb=independent_ncc(TB,TA,bv,av,dest[i],tr['back_disp'][i],off)
        fncc.append(a);bncc.append(b);fn.append(na);bn.append(nb)
    fncc=np.array(fncc);bncc=np.array(bncc);fn=np.array(fn);bn=np.array(bn)
    fe=np.abs(fncc-tr['ncc'][sample]);be=np.abs(bncc-tr['back_ncc'][sample])
    assert np.nanmax(fe)<1e-12,('forward NCC error',float(np.nanmax(fe)))
    assert np.nanmax(be)<1e-12,('reverse NCC error',float(np.nanmax(be)))
    assert np.isfinite(fncc).all() and np.isfinite(bncc).all()
    sample_accepted=accepted[sample]
    assert np.all(fn[sample_accepted]>=69) and np.all(bn[sample_accepted]>=69)
    summary=dict(pair=pair,status='passed',all_checks_pass=True,
        candidate_count=n,accepted_count=int(accepted.sum()),
        detector_points_and_strengths_exact=True,
        all_reciprocal_errors_exact=True,all_acceptance_flags_exact=True,
        source_depth_range=[float(depth.min()),float(depth.max())],
        accepted_depth_range=[float(depth[accepted].min()),float(depth[accepted].max())],
        candidates_beyond_old_depth354=int(np.sum(depth>354)),
        accepted_beyond_old_depth354=int(np.sum(accepted&(depth>354))),
        all_source_centers_available=True,
        accepted_unavailable_target_centers=int(np.sum(accepted&~target_center_visible)),
        sampled_count=len(sample),sampled_accepted=int(sample_accepted.sum()),
        sampled_rejected=int((~sample_accepted).sum()),
        max_forward_ncc_reconstruction_error=float(np.nanmax(fe)),
        max_reverse_ncc_reconstruction_error=float(np.nanmax(be)),
        minimum_sampled_accepted_forward_common_pixels=int(fn[sample_accepted].min()),
        minimum_sampled_accepted_reverse_common_pixels=int(bn[sample_accepted].min()),
        minimum_required_common_pixels=69,patch_pixels=81,
        no_optimization_rerun=True,no_acceptance_or_model_changes=True,
        seconds=time.time()-started)
    np.savez_compressed(directory/'track_independent_audit_samples.npz',indices=sample,
        points=points[sample],depth=depth[sample],accepted=sample_accepted,
        forward_ncc=fncc,reverse_ncc=bncc,forward_common_count=fn,reverse_common_count=bn,
        forward_saved_ncc=tr['ncc'][sample],reverse_saved_ncc=tr['back_ncc'][sample],
        forward_ncc_error=fe,reverse_ncc_error=be,
        target_center_visible=target_center_visible[sample])
    (directory/'track_independent_audit.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs',nargs='+',choices=['80','100'],default=['80','100'])
    args=parser.parse_args()
    for pair in args.pairs:
        print(json.dumps(run(int(pair)),indent=2),flush=True)


if __name__=='__main__':
    main()
