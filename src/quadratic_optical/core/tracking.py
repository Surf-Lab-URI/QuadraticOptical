"""Fresh IMAGE-ONLY symmetric bootstrap and reciprocal particle tracking.

No supplied velocity, previous vector field, or previous track data are read.
Inputs contain images, masks, free-surface geometry, grid, and calibration only.
Primary coarse initialization uses unconditional masked NCC over +/-64 pixels
in both axes. A +/-48 subset is also fitted using the same candidate cache.
Only this fresh image pair's coarse maps initialize its particle matcher.

"""
import os

import argparse
import hashlib
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter, map_coordinates
from scipy.optimize import minimize
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree

from types import SimpleNamespace
from .fit_local_cached import fit_local
from ._provenance import forbidden_input_keys, source_hashes, TRACKING_SOURCES

SOLVER = Path(__file__).resolve().with_name('fit_local_cached.py')

STATS = ['ncc', 'rms', 'nvalid', 'cond', 'iterations', 'mindet', 'support_fraction']
INPUT_KEYS = ['A', 'B', 'rawA', 'rawB', 'va', 'vb', 'surface_a', 'surface_b',
              'points', 'origin0', 'DX', 'DT', 'requested_max_depth_px',
              'fitting_max_depth_px', 'detector_max_depth_px',
              'image_only', 'supplied_velocity_used']
SETTINGS = dict(image_only=True, previous_fields_or_tracks_used=False,
    velocity_inputs_allowed=False, search_radius_primary=64,
    search_radius_sensitivity=48, search_axes='dx and dy symmetric',
    search_peak_count=3, search_peak_exclusion_half_width=3,
    search_min_common_pixels=40, search_min_common_source_fraction=.65,
    zero_seed_always_included=True, coarse_radius=19, coarse_order=1,
    coarse_reg=5e-5, coarse_maxiter=35, coarse_prior_ncc_min=.7,
    coarse_valid_mindet=.15, coarse_valid_support=.7,
    competing_solution_separation=2.5, descriptive_ambiguity_ncc_gap=.03,
    detector_dog_sigmas=[.6,2], detector_local_max_width=5, detector_min_peak=8,
    detector_background_sigma=5, detector_background_max=180,
    detector_min_source_depth=14, detector_patch_radius=4,
    particle_image_smoothing_sigma=.5, track_search_radius=8,
    track_min_common_fraction=.85, track_prior_penalty=.005,
    track_alternative_separation=2.5, track_maxiter=55,
    track_xatol=.035, track_fatol=.0002,
    track_ncc_min=.68, track_reverse_ncc_min=.68, track_fb_max=1.,
    track_gap_min=.015, track_reverse_gap_min=.01,
    prior_neighbor_count=12, prior_spatial_sigma=25,
    prior_displacement_consensus_radius=4)


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()


def read(path, keys=None):
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k] for k in (z.files if keys is None else keys)}


def atomic_npz(path, **data):
    path=Path(path); tmp=path.with_name(path.name+'.tmp-'+str(os.getpid()))
    with open(tmp,'wb') as f:
        np.savez_compressed(f,**data); f.flush(); os.fsync(f.fileno())
    os.replace(str(tmp),str(path))


def atomic_json(path, data):
    path=Path(path); tmp=path.with_name(path.name+'.tmp-'+str(os.getpid()))
    tmp.write_text(json.dumps(data,indent=2)+'\n'); os.replace(str(tmp),str(path))


def load_inputs(path):
    # Reject recognizable velocity/reference fields even though the whitelist
    # below would not load them. Geometry/calibration metadata remain allowed.
    with np.load(path,allow_pickle=False) as z:
        forbidden = forbidden_input_keys(z.files)
        if forbidden:
            raise ValueError('Velocity/reference fields forbidden in image-only inputs: '+', '.join(forbidden))
        missing=[k for k in INPUT_KEYS if k not in z.files]
        if missing: raise ValueError('Missing image-only input keys: '+', '.join(missing))
        inp={k:z[k] for k in INPUT_KEYS}
        if not bool(inp['image_only']) or bool(inp['supplied_velocity_used']):
            raise ValueError('Input provenance must affirm image_only and no supplied velocity use.')
    return inp


def translation_ncc(A, B, va, vb, center, radius=19, search_radius=64):
    """NCC(A[x], B[x+(dx,dy)]), with signed symmetric displacement axes.

    Interrogation/search rectangles are clipped to actual image bounds. NCC
    uses only jointly available pixels, with >=max(40, .65*source_count).
    This search never correlates air pixels or padded-outside-image values.
    """
    x,y=np.round(center).astype(int); h,w=A.shape
    x0=max(0,x-radius);x1=min(w,x+radius+1)
    y0=max(0,y-radius);y1=min(h,y+radius+1)
    T=A[y0:y1,x0:x1]; V=va[y0:y1,x0:x1].astype(float)
    dx0=max(-search_radius,-x0);dx1=min(search_radius,w-x1)
    dy0=max(-search_radius,-y0);dy1=min(search_radius,h-y1)
    if x0>=x1 or y0>=y1 or dx0>dx1 or dy0>dy1:
        raise ValueError('Invalid source/search rectangle.')
    S=B[y0+dy0:y1+dy1,x0+dx0:x1+dx1]
    W=vb[y0+dy0:y1+dy1,x0+dx0:x1+dx1].astype(float)
    def corr(a,b): return fftconvolve(a,b[::-1,::-1],mode='valid')
    count=corr(W,V); asum=corr(W,V*T); bsum=corr(W*S,V)
    den=np.maximum(count,1)
    aa=corr(W,V*T*T)-asum*asum/den
    bb=corr(W*S*S,V)-bsum*bsum/den
    cross=corr(W*S,V*T)-asum*bsum/den
    score=cross/np.sqrt(np.maximum(aa*bb,1e-12))
    threshold=max(40.,.65*V.sum())
    score[count<threshold]=-2.
    bounds=np.array([dx0,dx1,dy0,dy1],int)
    assert score.shape==(dy1-dy0+1,dx1-dx0+1)
    return score,count,bounds,float(V.sum()),float(threshold)


def peaks(score,count,bounds,search_radius):
    dx0,dx1,dy0,dy1=bounds
    xx=np.arange(dx0,dx1+1);yy=np.arange(dy0,dy1+1)
    inside=(np.abs(yy[:,None])<=search_radius)&(np.abs(xx[None,:])<=search_radius)
    valid=inside&np.isfinite(score)&(score>-1.5)
    working=score.copy();working[~valid]=-np.inf
    # Fixed-size diagnostic slots; unfilled slots are never used as fit seeds.
    seeds=np.full((3,2),np.nan);scores=np.full(3,-2.);support=np.zeros(3)
    for j in range(3):
        if not np.isfinite(working).any():break
        iy,ix=np.unravel_index(np.argmax(working),working.shape)
        seeds[j]=[dx0+ix,dy0+iy];scores[j]=score[iy,ix];support[j]=count[iy,ix]
        working[max(0,iy-3):iy+4,max(0,ix-3):ix+4]=-np.inf
    return seeds,scores,support


def admissible(params, stats):
    return (np.isfinite(params).all() and np.isfinite(stats.get('ncc',np.nan)) and
            stats.get('mindet',0)>.15 and
            stats.get('support_fraction',0)>.7)


def best_candidate(fits, indices):
    good=[i for i in indices if admissible(*fits[i])]
    if good: return max(good,key=lambda i:fits[i][1].get('ncc',-1)),True
    finite=[i for i in indices if np.isfinite(fits[i][0]).all() and
            np.isfinite(fits[i][1].get('ncc',np.nan))]
    return (max(finite,key=lambda i:fits[i][1].get('ncc',-1)) if finite else indices[-1]),False


class ImageOnlyTracker:
    def __init__(self, inp):
        self.inp=inp
        for k in INPUT_KEYS: setattr(self,k,inp[k])
        shape=self.A.shape
        assert len(shape)==2
        for k in ['B','rawA','rawB','va','vb']: assert getattr(self,k).shape==shape
        assert self.va.dtype==self.vb.dtype==bool
        assert self.surface_a.shape==self.surface_b.shape==(shape[1],)
        assert np.array_equal(self.origin0,[0,0])
        assert self.points.ndim==2 and self.points.shape[1]==2
        assert np.isfinite(self.points).all()
        self.target_gradient=np.gradient(self.B);self.target_mask_float=self.vb.astype(float)

    def coarse_task(self,i):
        pt=self.points[i]
        score,counts,bounds,source_count,minimum=translation_ncc(
            self.A,self.B,self.va,self.vb,pt,19,64)
        wide,ws,wc=peaks(score,counts,bounds,64)
        narrow,ns,nc=peaks(score,counts,bounds,48)
        zero_ix=-bounds[0];zero_iy=-bounds[2]
        seeds=np.concatenate([wide,narrow,np.zeros((1,2))])
        scores=np.r_[ws,ns,score[zero_iy,zero_ix]]
        common=np.r_[wc,nc,counts[zero_iy,zero_ix]]
        cache={};fits=[]
        for seed in seeds:
            if not np.isfinite(seed).all():
                fits.append((np.full((2,3),np.nan),{'ncc':-1.,'nvalid':0}))
                continue
            key=tuple(seed)
            if key not in cache:
                cache[key]=fit_local(self.A,self.B,self.va,self.vb,pt,seed,
                    r=19,order=1,reg=5e-5,maxiter=35,target_gradient=self.target_gradient,
                    target_mask_float=self.target_mask_float)
            fits.append(cache[key])
        selected,good=best_candidate(fits,[0,1,2,6])
        alternate,agood=best_candidate(fits,[3,4,5,6])
        p,s=fits[selected];pa,sa=fits[alternate]
        out=dict(params=p,candidate_seeds=seeds,candidate_seed_ncc=scores,
                 candidate_seed_valid=np.isfinite(seeds).all(axis=1),
                 candidate_seed_common_count=common,candidate_params=np.array([f[0] for f in fits]),
                 candidate_admissible=np.array([admissible(*f) for f in fits]),
                 selected_candidate=selected,selected_admissible=good,
                 bootstrap48_params=pa,bootstrap48_selected_candidate=alternate,
                 bootstrap48_admissible=agood,search_bounds=bounds,
                 search_source_valid_pixels=source_count,search_required_common_pixels=minimum,
                 unique_seed_fit_count=len(cache),
                 bootstrap_displacement_disagreement=float(np.linalg.norm(p[:,0]-pa[:,0])),
                 bootstrap_gradient_disagreement=float(np.max(np.abs(p[:,1:]-pa[:,1:]))/19),
                 selected_seed_on_search_limit=bool(np.any(np.abs(seeds[selected])==64)),
                 bootstrap48_seed_on_search_limit=bool(np.any(np.abs(seeds[alternate])==48)),
                 selected_seed_on_image_limit=bool(seeds[selected,0] in bounds[:2] or seeds[selected,1] in bounds[2:]),
                 selected_translation_beyond_search=bool(np.any(np.abs(p[:,0])>64)))
        for key in STATS:
            out[key]=s.get(key,np.nan)
            out['bootstrap48_'+key]=sa.get(key,np.nan)
            out['candidate_'+key]=np.array([f[1].get(key,np.nan) for f in fits])
        # Preserve true candidate scores above, but exclude inadmissible maps
        # from the same historical (> .7) tracking-prior gate.
        if not good:out['ncc']=-1.
        if not agood:out['bootstrap48_ncc']=-1.
        competitors=[j for j in [0,1,2,6] if j!=selected and admissible(*fits[j]) and
                     np.linalg.norm(fits[j][0][:,0]-p[:,0])>2.5]
        runner=max(competitors,key=lambda j:fits[j][1].get('ncc',-1)) if competitors else -1
        out['competing_candidate']=runner
        out['competing_ncc_gap']=(s.get('ncc',-1)-fits[runner][1].get('ncc',-1)) if runner>=0 else np.nan
        out['bootstrap_ambiguous']=bool(good and runner>=0 and out['competing_ncc_gap']<.03)
        return out

    def initialize_tracking(self,coarse):
        good=(coarse['ncc']>.7)&np.isfinite(coarse['params']).all(axis=(1,2))
        if good.sum()<12:raise RuntimeError('Fewer than 12 usable image-only coarse maps; no replacement prior inserted.')
        self.coarse_good_count=int(good.sum());self.ip=coarse['points'][good];self.pp=coarse['params'][good]
        self.tree=cKDTree(self.ip);J=np.eye(2)[None]+self.pp[:,:,1:]/19
        self.invJ=np.linalg.inv(J);self.bp=self.ip+self.pp[:,:,0];self.btree=cKDTree(self.bp)
        self.TA=gaussian_filter(self.rawA,.5);self.TB=gaussian_filter(self.rawB,.5)
        self.track_va=self.va.astype(float);self.track_vb=self.vb.astype(float)
        dy,dx=np.mgrid[-4:5,-4:5];self.off=np.c_[dx.ravel(),dy.ravel()]
        dy,dx=np.mgrid[-8:9,-8:9];self.shifts=np.c_[dx.ravel(),dy.ravel()]

    def prior(self,pt,back=False):
        tree=self.btree if back else self.tree;dist,idx=tree.query(pt,k=12)
        if back:
            pred=-self.pp[idx,:,0]+np.einsum('nij,nj->ni',self.invJ[idx]-np.eye(2)[None],pt-self.bp[idx])
        else:
            pred=self.pp[idx,:,0]+np.einsum('nij,nj->ni',self.pp[idx,:,1:]/19,pt-self.ip[idx])
        dd=np.linalg.norm(pred[:,None]-pred[None,:],axis=2)
        weights=np.exp(-dist**2/(2*25**2));med=np.argmin(dd.dot(weights))
        sel=np.linalg.norm(pred-pred[med],axis=1)<4
        return np.average(pred[sel],axis=0,weights=weights[sel])

    def track(self,pt,back=False):
        src,dst,vs,vd=((self.TB,self.TA,self.track_vb,self.track_va) if back else
                      (self.TA,self.TB,self.track_va,self.track_vb))
        pr=self.prior(pt,back);coords=pt+self.off
        aa=map_coordinates(src,[coords[:,1],coords[:,0]],order=1)
        sm=map_coordinates(vs,[coords[:,1],coords[:,0]],order=1)>.99
        def cc(ds):
            pos=coords[None]+ds[:,None]
            bb=map_coordinates(dst,[pos[:,:,1],pos[:,:,0]],order=1)
            vm=map_coordinates(vd,[pos[:,:,1],pos[:,:,0]],order=1)>.99;vm&=sm[None]
            n=vm.sum(axis=1);den=np.maximum(n,1)
            am=(aa[None]*vm).sum(axis=1)/den;bm=(bb*vm).sum(axis=1)/den
            az=(aa[None]-am[:,None])*vm;bz=(bb-bm[:,None])*vm
            cor=(az*bz).sum(axis=1)/np.sqrt((az*az).sum(axis=1)*(bz*bz).sum(axis=1)+1e-10)
            cor[n<.85*len(aa)]=-1
            return cor
        candidates=pr[None]+self.shifts;cors=cc(candidates)
        scores=cors-.005*np.sum(self.shifts**2,axis=1);ii=np.argsort(-scores);sel=[ii[0]]
        for idx in ii[1:]:
            if np.linalg.norm(candidates[idx]-candidates[sel[0]])>2.5:sel.append(idx);break
        fits=[]
        for idx in sel:
            opt=minimize(lambda d:1-cc(d[None])[0]+.005*np.sum((d-pr)**2),candidates[idx],
                         method='Nelder-Mead',options={'maxiter':55,'xatol':.035,'fatol':.0002})
            fits.append((float(opt.fun),opt.x,float(cc(opt.x[None])[0])))
        fits.sort(key=lambda v:v[0]);best=fits[0]
        outside=np.linalg.norm(candidates-best[1][None],axis=1)>2.5
        alternatives=[float(np.min(1-scores[outside]))]
        alternatives += [v[0] for v in fits[1:] if np.linalg.norm(v[1]-best[1])>2.5]
        gap=max(0.,min(alternatives)-best[0])
        return best[1],best[2],gap,pr

    def detect(self):
        hp=gaussian_filter(self.rawA,.6)-gaussian_filter(self.rawA,2)
        background=gaussian_filter(self.rawA,5);yy,xx=np.indices(self.rawA.shape)
        depth=yy-self.surface_a[None]
        det=((hp==maximum_filter(hp,size=5))&(hp>8)&(xx>=4)&(xx<self.rawA.shape[1]-4)&
             (yy>=4)&(yy<self.rawA.shape[0]-4)&(depth>=14)&
             (depth<=float(self.detector_max_depth_px))&self.va&(background<180))
        return dict(points=np.c_[xx[det],yy[det]].astype(float),strength=hp[det])


def coarse_view(search,radius):
    if radius==64:return dict(search)
    result=dict(search)
    result['params']=search['bootstrap48_params']
    for k in STATS:result[k]=search['bootstrap48_'+k]
    result['selected_candidate']=search['bootstrap48_selected_candidate']
    result['selected_admissible']=search['bootstrap48_admissible']
    return result


class Runner:
    def __init__(self,args):
        self.args=args;self.base=Path(args.directory).resolve()
        path=self.base/'inputs.npz'
        if not path.is_file():raise FileNotFoundError('Fresh image-only inputs not ready: '+str(path))
        self.out=self.base if args.bootstrap_radius==64 else self.base/'bootstrap48'
        self.out.mkdir(parents=True,exist_ok=True)
        inp=load_inputs(path);self.tr=ImageOnlyTracker(inp);self.started=time.time()
        if self.out != self.base:
            branch_input = self.out/'inputs.npz'
            if branch_input.exists():
                if sha(branch_input) != sha(path):
                    raise ValueError('Alternate bootstrap inputs differ from the primary image pair.')
            else:
                temporary = branch_input.with_name(branch_input.name+'.tmp-'+str(os.getpid()))
                shutil.copyfile(str(path), str(temporary))
                os.replace(str(temporary), str(branch_input))
        self.identity=dict(input_sha256=sha(path),settings=SETTINGS,
            source_sha256=source_hashes(TRACKING_SOURCES),
            requested_max_depth_px=float(inp['requested_max_depth_px']),
            fitting_max_depth_px=float(inp['fitting_max_depth_px']),
            detector_max_depth_px=float(inp['detector_max_depth_px']),chunk_size=args.chunk_size)
        self.coarse_signature=hashlib.sha256(json.dumps(self.identity,sort_keys=True).encode()).hexdigest()
        self.signature=hashlib.sha256((self.coarse_signature+str(args.bootstrap_radius)).encode()).hexdigest()
        self.coarse_chunks=self.base/'tracking_checkpoints'/self.coarse_signature[:16]
        self.chunks=self.out/'tracking_checkpoints'/self.signature[:16]
        for p in [self.coarse_chunks,self.chunks]:p.mkdir(parents=True,exist_ok=True)
        atomic_json(self.coarse_chunks/'manifest.json',self.identity)
        self.new_chunks=0;self.stages={}
        self.read_paths=[str(path)];self.written_paths=[]
        atomic_json(self.out/'tracking_settings.json',dict(self.identity,
            bootstrap_radius=args.bootstrap_radius,run_signature=self.signature,
            input_keys_read=INPUT_KEYS,executor=args.executor,workers=args.workers))
        self.access_audit()

    def access_audit(self):
        atomic_json(self.out/'source_access_audit.json',dict(image_only=True,
            velocity_arrays_read=False,previous_models_or_tracks_read=False,
            input_keys_read=INPUT_KEYS,input_data_file=str(self.base/'inputs.npz'),
            current_run_artifacts_read=self.read_paths[1:],current_run_artifacts_written=self.written_paths,
            computational_source_files=list(self.identity['source_sha256']),
            coarse_signature=self.coarse_signature,run_signature=self.signature))

    def completed(self,path,signature):
        if not path.is_file():return None
        z=read(path);self.read_paths.append(str(path));self.access_audit()
        if str(z.get('run_signature',''))!=signature:
            raise ValueError('Existing file has different fresh-image inputs/code/settings: '+str(path))
        return z

    def write(self,path,data,signature):
        atomic_npz(path,**dict(data,run_signature=signature,image_only=True,
            supplied_velocity_used=False,previous_models_or_tracks_used=False,
            input_sha256=self.identity['input_sha256'],
            tracking_sha256=self.identity['source_sha256']['tracking.py'],
            solver_sha256=self.identity['source_sha256']['fit_local_cached.py']))
        self.written_paths.append(str(path));self.access_audit()

    def stage(self,name,points,task,pack,coarse=False):
        folder=self.coarse_chunks if coarse else self.chunks
        pieces=[];resumed=0;started=time.time()
        with ThreadPoolExecutor(max_workers=self.args.workers) as pool:
            for first in range(0,len(points),self.args.chunk_size):
                last=min(len(points),first+self.args.chunk_size)
                path=folder/('%s_%07d_%07d.npz'%(name,first,last))
                if path.is_file():
                    piece=read(path);self.read_paths.append(str(path));resumed+=last-first
                    assert np.array_equal(piece['points'],points[first:last])
                else:
                    if self.args.max_chunks and self.new_chunks>=self.args.max_chunks:
                        self.access_audit();return None
                    piece=pack(list(pool.map(task,range(first,last))))
                    piece['points']=points[first:last];atomic_npz(path,**piece)
                    self.written_paths.append(str(path));self.new_chunks+=1
                pieces.append(piece)
                self.access_audit()
                print('Image-only pair',self.args.pair,name,last,'/',len(points),
                      'rows; resumed',resumed,'; elapsed',round(time.time()-started,1),'s',flush=True)
        if not pieces:raise ValueError('No usable source points/candidates for '+name)
        self.stages[name]=dict(rows=len(points),resumed_rows=resumed,seconds=time.time()-started)
        return {k:np.concatenate([v[k] for v in pieces],axis=0) for k in pieces[0]}

    def run(self):
        tr=self.tr
        search=self.completed(self.base/'coarse_search.npz',self.coarse_signature)
        if search is None:
            def pack(result):return {k:np.array([v[k] for v in result]) for k in result[0]}
            search=self.stage('coarse',tr.points,tr.coarse_task,pack,True)
            if search is None:return False
            search.update(radius=np.array(19),order=np.array(1),reg=np.array(5e-5),origin0=tr.origin0)
            self.write(self.base/'coarse_search.npz',search,self.coarse_signature)
        coarse=coarse_view(search,self.args.bootstrap_radius)
        self.write(self.out/'coarse_affine.npz',coarse,self.signature)
        # Standalone alternate map checkpoint makes the sensitivity reviewable
        # before deciding whether to run a full alternative tracking branch.
        if self.args.bootstrap_radius==64:
            self.write(self.base/'coarse_bootstrap48.npz',coarse_view(search,48),self.coarse_signature)
        good64=(search['ncc']>.7)&np.isfinite(search['params']).all(axis=(1,2))
        good48=(search['bootstrap48_ncc']>.7)&np.isfinite(search['bootstrap48_params']).all(axis=(1,2))
        shared=good64&good48;diff=search['bootstrap_displacement_disagreement']
        sensitivity=dict(common_usable=int(shared.sum()),wide_only_usable=int((good64&~good48).sum()),
            narrow_only_usable=int((good48&~good64).sum()),
            differing_usable_over_point1_px=int(np.sum(shared&(diff>.1))),
            differing_usable_over1_px=int(np.sum(shared&(diff>1))),
            max_common_usable_displacement_difference=float(np.max(diff[shared])) if shared.any() else None,
            mean_common_usable_displacement_difference=float(np.mean(diff[shared])) if shared.any() else None,
            primary_seed_on_artificial_search_limit=int(np.sum(good64&search['selected_seed_on_search_limit'])),
            primary_fit_beyond_search_radius=int(np.sum(good64&search['selected_translation_beyond_search'])),
            competing_high_ncc_basins=int(np.sum(good64&search['bootstrap_ambiguous'])),
            meaning='Coarse bootstrap sensitivity only; not a final-field uncertainty bound.')
        atomic_json(self.out/'bootstrap_sensitivity.json',sensitivity)
        print('Image-only coarse complete',len(tr.points),'usable64',int(good64.sum()),
              'usable48',int(good48.sum()),json.dumps(sensitivity),flush=True)
        if self.args.stop_after=='coarse':return True
        existing=self.completed(self.out/'ptv_tracks.npz',self.signature)
        if existing is not None:return True
        tr.initialize_tracking(coarse)
        candidates=self.completed(self.out/'particle_candidates.npz',self.signature)
        if candidates is None:
            candidates=tr.detect();self.write(self.out/'particle_candidates.npz',candidates,self.signature)
        pts=candidates['points'];strength=candidates['strength']
        def pack_track(result):
            return dict(disp=np.array([v[0] for v in result]),ncc=np.array([v[1] for v in result]),
                        gap=np.array([v[2] for v in result]),prior=np.array([v[3] for v in result]))
        fw=self.completed(self.out/'tracks_forward.npz',self.signature)
        if fw is None:
            fw=self.stage('forward',pts,lambda i:tr.track(pts[i]),pack_track)
            if fw is None:return False
            fw['strength']=strength;self.write(self.out/'tracks_forward.npz',fw,self.signature)
        if self.args.stop_after=='forward':return True
        dest=pts+fw['disp'];bw=self.completed(self.out/'tracks_reverse.npz',self.signature)
        if bw is None:
            bw=self.stage('reverse',dest,lambda i:tr.track(dest[i],True),pack_track)
            if bw is None:return False
            self.write(self.out/'tracks_reverse.npz',bw,self.signature)
        fb=np.linalg.norm(fw['disp']+bw['disp'],axis=1)
        accepted=(fw['ncc']>.68)&(bw['ncc']>.68)&(fb<1)&(fw['gap']>.015)&(bw['gap']>.01)
        depth=pts[:,1]-np.interp(pts[:,0],np.arange(len(tr.surface_a)),tr.surface_a)
        tracks=dict(complete=np.array(True),points=pts,disp=fw['disp'],prior=fw['prior'],ncc=fw['ncc'],
            ambiguity_gap=fw['gap'],back_disp=bw['disp'],back_ncc=bw['ncc'],back_gap=bw['gap'],
            fb=fb,accepted=accepted,strength=strength,source_depth=depth,origin0=tr.origin0,
            bootstrap_radius=np.array(self.args.bootstrap_radius),
            requested_max_depth_px=tr.requested_max_depth_px,
            fitting_max_depth_px=tr.fitting_max_depth_px,detector_max_depth_px=tr.detector_max_depth_px)
        self.write(self.out/'ptv_tracks.npz',tracks,self.signature)
        summary=dict(pair=self.args.pair,image_only=True,bootstrap_radius=self.args.bootstrap_radius,
            grid_nodes=len(tr.points),usable_coarse=tr.coarse_good_count,candidates=len(pts),
            accepted=int(accepted.sum()),requested_max_depth_px=float(tr.requested_max_depth_px),
            fitting_max_depth_px=float(tr.fitting_max_depth_px),
            detector_max_depth_px=float(tr.detector_max_depth_px),
            accepted_depth_range=[float(depth[accepted].min()),float(depth[accepted].max())] if accepted.any() else None,
            bootstrap_sensitivity=sensitivity,stages_this_invocation=self.stages,
            invocation_seconds=time.time()-self.started,input_sha256=self.identity['input_sha256'],
            run_signature=self.signature,
            output_sha256={name:sha(self.out/name) for name in ['coarse_affine.npz','ptv_tracks.npz']})
        atomic_json(self.out/'tracking_summary.json',summary);self.access_audit()
        print(json.dumps(summary,indent=2),flush=True)
        return True


def run(directory, workers=1, bootstrap_radius=64, chunk_size=256,
        max_chunks=0, stop_after='reverse'):
    """Run/resume this directory's fresh image-only bootstrap and tracking.

    Returns True when the requested stage is complete, False when max_chunks
    stops early. The optional radius-48 branch is written under bootstrap48/;
    both branches use this same image pair's fresh symmetric coarse search.
    No external velocity file or previously predicted field is an input.
    """
    if workers < 1 or chunk_size < 1 or max_chunks < 0:
        raise ValueError('Invalid worker/chunk parameters.')
    if bootstrap_radius not in (64, 48):
        raise ValueError('The validated bootstrap radii are 64 and 48 pixels.')
    if stop_after not in ('coarse', 'forward', 'reverse'):
        raise ValueError('stop_after must be coarse, forward, or reverse.')
    directory = Path(directory).resolve()
    args = SimpleNamespace(directory=directory, pair=directory.name,
                           workers=int(workers), bootstrap_radius=bootstrap_radius,
                           chunk_size=int(chunk_size), max_chunks=int(max_chunks),
                           stop_after=stop_after, executor='threads')
    return Runner(args).run()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--bootstrap-radius', type=int, choices=[64, 48], default=64)
    p.add_argument('--chunk-size', type=int, default=256)
    p.add_argument('--max-chunks', type=int, default=0)
    p.add_argument('--stop-after', choices=['coarse', 'forward', 'reverse'], default='reverse')
    return 0 if run(**vars(p.parse_args())) else 2


if __name__ == '__main__':
    raise SystemExit(main())
