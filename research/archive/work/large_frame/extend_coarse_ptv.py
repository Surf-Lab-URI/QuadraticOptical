"""Extend original image-derived affine coarse prior and reciprocal particle tracks.
Original coarse fits and all original candidate/acceptance rows are frozen.
No manual coordinates/endpoints are read by this stage.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import sys,time,json
from pathlib import Path
sys.path.insert(0,'work')
import numpy as np
from scipy.ndimage import gaussian_filter,maximum_filter,map_coordinates
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from concurrent.futures import ThreadPoolExecutor
from extended_flow import fit_local,translation_seed
O=Path('work/large_frame');inp=np.load(O/'inputs.npz')
# Explicit image-only key selection prevents accidental use of validation fields.
A=inp['A'];B=inp['B'];rawA=inp['rawA'];rawB=inp['rawB'];va=inp['va'];vb=inp['vb'];sa=inp['surface_a'];sb=inp['surface_b'];points=inp['points'];origin=inp['origin0'];oldorigin=inp['old_origin0'];shift=oldorigin-origin;maxdepth=float(inp['max_depth']);supx=inp['supplied_dx'];supy=inp['supplied_dy']
base=np.load('work/propagate0.npz');base={k:base[k] for k in base.files};oldpts=base['points']+shift;lookup={tuple(p):i for i,p in enumerate(oldpts)};original=np.array([tuple(p) in lookup for p in points]);original_index=np.array([lookup.get(tuple(p),-1) for p in points])
assert original.sum()==len(oldpts)==727
SETTINGS=dict(original_coarse_path='work/propagate0.npz',original_tracks_path='work/ptv_tracks.npz',frozen_original_coarse=727,origin0=origin.tolist(),original_to_local_shift=shift.tolist(),coarse_radius=19,coarse_order=1,coarse_reg=.00005,poor_coarse_ncc=.7,coarse_valid_mindet=.15,coarse_valid_support=.7,coarse_fallback_search_dx=[-15,55],coarse_fallback_search_dy=[-20,20],coarse_fallback_candidates=3,detector_dog_sigmas=[.6,2],detector_local_max_width=5,detector_min_peak=8,detector_background_sigma=5,detector_background_max=180,detector_min_source_depth=14,detector_max_source_depth=maxdepth+25,detector_global_x=[294,744],near_original_candidate_exclusion_radius=2.5,track_patch_radius=4,track_displacement_search_radius=8,track_min_common_fraction=.85,track_prior_penalty=.005,track_ncc_min=.68,track_reverse_ncc_min=.68,track_fb_max=1.,track_gap_min=.015,track_reverse_gap_min=.01,prior_neighbor_count=12,prior_spatial_sigma=25,prior_displacement_consensus_radius=4,manual_data_used=False)
(O/'coarse_ptv_settings.json').write_text(json.dumps(SETTINGS,indent=2))
t0=time.time()
if (O/'coarse_affine.npz').exists():
    co=dict(np.load(O/'coarse_affine.npz'));print('reusingcompletedcoarse',len(co['points']),flush=True)
else:
    count=[0]
    def coarse_task(i):
        if original[i]:
            j=original_index[i]
            return base['params'][j],{k:base[k][j].item() for k in ['ncc','rms','nvalid','cond','iterations','mindet','support_fraction']},0
        pt=points[i];ini=np.array([map_coordinates(s,[pt[1:2],pt[0:1]],order=1)[0] for s in [supx,supy]])
        if not np.isfinite(ini).all():ini=np.zeros(2)
        p,stats=fit_local(A,B,va,vb,pt,ini,r=19,order=1,reg=.00005)
        fits=[(p,stats)];fallback=int(stats.get('ncc',-1)<.7 or stats.get('mindet',0)<=.15 or stats.get('support_fraction',0)<=.7 or not np.isfinite(p).all())
        if fallback:
            for seed,score in translation_seed(A,B,va,vb,pt,r=19,search=(55,20)):
                if score<.05:continue
                pp,st=fit_local(A,B,va,vb,pt,seed,r=19,order=1,reg=.00005);fits.append((pp,st))
        valid=[v for v in fits if np.isfinite(v[0]).all() and v[1].get('mindet',0)>.15 and v[1].get('support_fraction',0)>.7]
        if valid:p,stats=max(valid,key=lambda v:v[1].get('ncc',-1))
        else:
            finite=[v for v in fits if np.isfinite(v[0]).all()]
            if finite:p,stats=max(finite,key=lambda v:v[1].get('ncc',-1))
            else:p,stats=fits[0]
            stats=dict(stats);stats['ncc']=-1. # Exclude invalid new coarse maps from the tracking prior.
        count[0]+=1
        if count[0]%200==0:print('newcoarse',count[0],'elapsed',round(time.time()-t0,1),flush=True)
        return p,stats,fallback
    with ThreadPoolExecutor(max_workers=4) as pool:result=list(pool.map(coarse_task,range(len(points))))
    params=np.array([v[0] for v in result]);keys=sorted(set().union(*(v[1].keys() for v in result)));stats={k:np.array([v[1].get(k,np.nan) for v in result]) for k in keys}
    co=dict(points=points,params=params,radius=19,original_coarse=original,original_coarse_index=original_index,coarse_fallback=np.array([v[2] for v in result]),**stats)
    for i in np.where(original)[0]:assert np.array_equal(params[i],base['params'][original_index[i]])
    np.savez_compressed(O/'coarse_affine.npz',**co)
    print('coarsecomplete',len(points),'new',int(sum(~original)),'priorgood',int(sum(co['ncc']>.7)),'seconds',round(time.time()-t0,1),flush=True)
# Same original PTV prior: robust medoid of affine extrapolations from 12 nearby image fits.
good=(co['ncc']>.7)&np.isfinite(co['params']).all(axis=(1,2));ip=co['points'][good];pp=co['params'][good];tree=cKDTree(ip);J=np.eye(2)[None]+pp[:,:,1:]/19
invJ=np.linalg.inv(J);bp=ip+pp[:,:,0];btree=cKDTree(bp)
def prior(pt,back=False):
    tt=btree if back else tree;dist,idx=tt.query(pt,k=12)
    if back:pred=-pp[idx,:,0]+np.einsum('nij,nj->ni',invJ[idx]-np.eye(2)[None],pt-bp[idx])
    else:pred=pp[idx,:,0]+np.einsum('nij,nj->ni',pp[idx,:,1:]/19,pt-ip[idx])
    dd=np.linalg.norm(pred[:,None]-pred[None,:],axis=2);weights=np.exp(-dist**2/(2*25**2));med=np.argmin(dd.dot(weights));sel=np.linalg.norm(pred-pred[med],axis=1)<4
    return np.average(pred[sel],axis=0,weights=weights[sel])
# Original tracking uses sigma .5 raw-image smoothing, no continuous-field normalized intensities.
TA=gaussian_filter(rawA,.5);TB=gaussian_filter(rawB,.5);va=va.astype(float);vb=vb.astype(float)
DY,DX=np.mgrid[-4:5,-4:5];off=np.c_[DX.ravel(),DY.ravel()];DY,DX=np.mgrid[-8:9,-8:9];shifts=np.c_[DX.ravel(),DY.ravel()]
def track(pt,back=False):
    src,dst,vs,vd=(TB,TA,vb,va) if back else (TA,TB,va,vb);pr=prior(pt,back);coords=pt+off;aa=map_coordinates(src,[coords[:,1],coords[:,0]],order=1);sm=map_coordinates(vs,[coords[:,1],coords[:,0]],order=1)>.99
    def cc(ds):
        pos=coords[None]+ds[:,None];bb=map_coordinates(dst,[pos[:,:,1],pos[:,:,0]],order=1);vm=map_coordinates(vd,[pos[:,:,1],pos[:,:,0]],order=1)>.99;vm&=sm[None];n=vm.sum(axis=1);den=np.maximum(n,1)
        am=(aa[None]*vm).sum(axis=1)/den;bm=(bb*vm).sum(axis=1)/den;az=(aa[None]-am[:,None])*vm;bz=(bb-bm[:,None])*vm
        cor=(az*bz).sum(axis=1)/np.sqrt((az*az).sum(axis=1)*(bz*bz).sum(axis=1)+1e-10);cor[n<.85*len(aa)]=-1
        return cor
    candidates=pr[None]+shifts;cors=cc(candidates);scores=cors-.005*np.sum(shifts**2,axis=1);ii=np.argsort(-scores);sel=[ii[0]]
    for idx in ii[1:]:
        if np.linalg.norm(candidates[idx]-candidates[sel[0]])>2.5:sel.append(idx);break
    fits=[]
    for idx in sel:
        opt=minimize(lambda d:1-cc(d[None])[0]+.005*np.sum((d-pr)**2),candidates[idx],method='Nelder-Mead',options={'maxiter':55,'xatol':.035,'fatol':.0002});fits.append((float(opt.fun),opt.x,float(cc(opt.x[None])[0])))
    fits.sort(key=lambda v:v[0]);best=fits[0];outside=np.linalg.norm(candidates-best[1][None],axis=1)>2.5;alternatives=[float(np.min(1-scores[outside]))];alternatives+=[v[0] for v in fits[1:] if np.linalg.norm(v[1]-best[1])>2.5];gap=max(0.,min(alternatives)-best[0])
    return best[1],best[2],gap,pr
old=np.load('work/ptv_tracks.npz');old={k:old[k] for k in old.files};op=old['points']+shift;otree=cKDTree(op)
hp=gaussian_filter(rawA,.6)-gaussian_filter(rawA,2);background=gaussian_filter(rawA,5);yy,xx=np.indices(rawA.shape);globalx=xx+origin[0];depth=yy-sa[None]
det=(hp==maximum_filter(hp,size=5))&(hp>8)&(globalx>=294)&(globalx<=744)&(depth>=14)&(depth<=maxdepth+25)&(yy<rawA.shape[0]-5)&(background<180)
newpts=np.c_[xx[det],yy[det]].astype(float);strength=hp[det];exclude=otree.query(newpts)[0]<=2.5;newpts=newpts[~exclude];strength=strength[~exclude]
np.savez_compressed(O/'new_particle_candidates.npz',points=newpts,strength=strength)
print('newparticlecandidates',len(newpts),'excludednearoriginal',int(exclude.sum()),'seconds',round(time.time()-t0,1),flush=True)
count=[0]
def forward(i):
    v=track(newpts[i]);count[0]+=1
    if count[0]%300==0:print('newforward',count[0],'elapsed',round(time.time()-t0,1),flush=True)
    return v
with ThreadPoolExecutor(max_workers=4) as pool:fw=list(pool.map(forward,range(len(newpts))))
disp=np.array([v[0] for v in fw]);dest=newpts+disp
np.savez_compressed(O/'new_tracks_forward.npz',points=newpts,disp=disp,ncc=np.array([v[1] for v in fw]),gap=np.array([v[2] for v in fw]),prior=np.array([v[3] for v in fw]),strength=strength)
count=[0]
def backward(i):
    v=track(dest[i],True);count[0]+=1
    if count[0]%300==0:print('newreverse',count[0],'elapsed',round(time.time()-t0,1),flush=True)
    return v
with ThreadPoolExecutor(max_workers=4) as pool:bw=list(pool.map(backward,range(len(newpts))))
back=np.array([v[0] for v in bw]);fb=np.linalg.norm(disp+back,axis=1);ncc=np.array([v[1] for v in fw]);gap=np.array([v[2] for v in fw]);bncc=np.array([v[1] for v in bw]);bgap=np.array([v[2] for v in bw]);accepted=(ncc>.68)&(bncc>.68)&(fb<1)&(gap>.015)&(bgap>.01)
new=dict(points=newpts,disp=disp,prior=np.array([v[3] for v in fw]),ncc=ncc,ambiguity_gap=gap,back_disp=back,back_ncc=bncc,back_gap=bgap,fb=fb,accepted=accepted,strength=strength)
combined={k:np.concatenate([op if k=='points' else old[k],new[k]],axis=0) for k in old}
flag=np.r_[np.ones(len(op),bool),np.zeros(len(newpts),bool)];combined.update(original_candidate=flag,original_track=flag,original_accepted_track=flag&combined['accepted'],origin0=origin,source_depth=combined['points'][:,1]-np.interp(combined['points'][:,0],np.arange(len(sa)),sa))
for k in old:
    assert np.array_equal(combined[k][:len(op)],op if k=='points' else old[k]),k
assert int(sum(combined['accepted'][:len(op)]))==675
np.savez_compressed(O/'ptv_tracks.npz',**combined);np.savez_compressed(O/'new_ptv_tracks.npz',**new)
summary=dict(original_candidate_rows=len(op),original_accepted_tracks=int(sum(old['accepted'])),new_candidate_rows=len(newpts),new_accepted_tracks=int(sum(accepted)),total_candidate_rows=len(combined['points']),total_accepted_tracks=int(sum(combined['accepted'])),old_rows_exactly_preserved_except_coordinate_shift=True,coarse_original_preserved=727,coarse_new_count=int(sum(~original)),coarse_good_prior_fits=int(sum(good)),new_accepted_min_depth=float(np.min(combined['source_depth'][len(op):][accepted])) if accepted.any() else None,total_seconds=time.time()-t0)
(O/'coarse_ptv_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
