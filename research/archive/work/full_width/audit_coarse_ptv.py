"""Independent, read-only audit of full-width particle/coarse extension."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json
import numpy as np
from scipy.ndimage import gaussian_filter,map_coordinates
from scipy.spatial import cKDTree

O=Path('work/full_width');i=np.load(O/'inputs.npz');t=np.load(O/'ptv_tracks.npz');c=np.load(O/'coarse_affine.npz')
b=np.load('work/large_frame/ptv_tracks.npz');bc=np.load('work/large_frame/coarse_affine.npz');bi=np.load('work/large_frame/inputs.npz');shift=bi['origin0']-i['origin0'];n=len(b['points'])
standard=['points','disp','prior','ncc','ambiguity_gap','back_disp','back_ncc','back_gap','fb','accepted','strength']
preserved={k:bool(np.array_equal(t[k][:n],b[k]+shift if k=='points' else b[k])) for k in standard}
lineage={k:bool(np.array_equal(t[k][:n],b[k])) for k in ['original_candidate','original_track','original_accepted_track']}
dist,idx=cKDTree(c['points']).query(bc['points']+shift);assert np.all(dist==0)
coarse={k:bool(np.all((c[k][idx]==bc[k])|(np.isnan(c[k][idx])&np.isnan(bc[k])))) for k in ['params','ncc','rms','nvalid','cond','iterations','mindet','support_fraction','coarse_fallback']}
expected=(t['ncc']>.68)&(t['back_ncc']>.68)&(t['fb']<1)&(t['ambiguity_gap']>.015)&(t['back_gap']>.01)
assert np.array_equal(t['accepted'][n:],expected[n:])
assert np.allclose(t['fb'],np.linalg.norm(t['disp']+t['back_disp'],axis=1),rtol=0,atol=1e-12)
assert all(preserved.values()) and all(lineage.values()) and all(coarse.values())
va=i['va'].astype(float);vb=i['vb'].astype(float);aa=gaussian_filter(i['rawA'],.5);bb=gaussian_filter(i['rawB'],.5)
Y,X=np.mgrid[-4:5,-4:5];off=np.c_[X.ravel(),Y.ravel()]
def sample(image,coords):return map_coordinates(image,[coords[...,1],coords[...,0]],order=1,mode='constant',cval=0)
def check(ids,reverse=False):
 errors=[];counts=[]
 src,dst,sm,tm=(bb,aa,vb,va) if reverse else (aa,bb,va,vb)
 for part in np.array_split(ids,max(1,int(np.ceil(len(ids)/1000)))):
  if not len(part):continue
  centers=t['points'][part]+(t['disp'][part] if reverse else 0)
  d=t['back_disp'][part] if reverse else t['disp'][part]
  coords=centers[:,None]+off;target=coords+d[:,None]
  valid=(sample(sm,coords)>.99)&(sample(tm,target)>.99);den=valid.sum(axis=1)
  av=sample(src,coords);bv=sample(dst,target)
  av=(av-(av*valid).sum(axis=1)[:,None]/den[:,None])*valid
  bv=(bv-(bv*valid).sum(axis=1)[:,None]/den[:,None])*valid
  cc=(av*bv).sum(axis=1)/np.sqrt((av*av).sum(axis=1)*(bv*bv).sum(axis=1)+1e-10)
  errors.extend(abs(cc-t['back_ncc' if reverse else 'ncc'][part]));counts.extend(den)
 return dict(n=len(ids),max_ncc_disagreement=float(max(errors)),minimum_common_pixels=int(min(counts)),median_common_pixels=float(np.median(counts)))
new=np.arange(len(t['points']))>=n;ids=np.flatnonzero(new&t['accepted'])
fw=check(ids);bw=check(ids,True)
def stats(x):
 x=np.asarray(x);x=x[np.isfinite(x)]
 return dict(n=len(x),min=float(x.min()),median=float(np.median(x)),p90=float(np.percentile(x,90)),max=float(x.max())) if len(x) else dict(n=0)
ac=t['accepted'];pts=t['points'][ac];target=pts+t['disp'][ac];globalpts=pts+i['origin0'];globaltarget=target+i['origin0'];height,width=aa.shape
centermask=dict(source_unavailable=int(np.sum(sample(i['availability_a'].astype(float),pts)<=.99)),target_unavailable=int(np.sum(sample(i['availability_b'].astype(float),target)<=.99)),source_margin_mask_invalid=int(np.sum(sample(va,pts)<=.99)),target_margin_mask_invalid=int(np.sum(sample(vb,target)<=.99)))
pairs=cKDTree(target).query_pairs(1.);at=np.flatnonzero(ac);duplicates=[]
for a,z in sorted(pairs):
 x,y=at[[a,z]];duplicates.append(dict(rows=[int(x),int(y)],source_full=globalpts[[a,z]].tolist(),target_distance=float(np.linalg.norm(target[a]-target[z])),source_distance=float(np.linalg.norm(pts[a]-pts[z])),either_new=bool(new[x]|new[y])))
bins={}
for lo,hi in [(0,256),(256,512),(512,768),(768,1024),(1024,1280),(1280,1536),(1536,1792),(1792,2048)]:
 x=t['points'][:,0]+i['origin0'][0];sel=(x>=lo)&(x<hi);bins[f'x{lo}_{hi}']=dict(candidates=int(sel.sum()),accepted=int(np.sum(sel&ac)),new_accepted=int(np.sum(sel&ac&new)),fb=stats(t['fb'][sel&ac]))
summary=dict(previous_rows=n,previous_accepted=int(b['accepted'].sum()),standard_track_arrays_preserved=preserved,lineage_preserved=lineage,coarse_previous_rows=len(bc['points']),coarse_arrays_preserved=coarse,new_candidates=int(new.sum()),new_accepted=int(len(ids)),total_accepted=int(ac.sum()),acceptance_recomputed_exact=True,forward_photometry=fw,reverse_photometry=bw,accepted_center_visibility=centermask,accepted_source_horizontal_image_edge_distance=stats(np.minimum(globalpts[:,0],2047-globalpts[:,0])),accepted_target_horizontal_image_edge_distance=stats(np.minimum(globaltarget[:,0],2047-globaltarget[:,0])),accepted_target_vertical_analysis_edge_distance=stats(np.minimum(target[:,1],height-1-target[:,1])),target_pairs_within1px=duplicates,horizontal_bins=bins,source_depth=stats(t['source_depth'][ac]),new_source_depth=stats(t['source_depth'][new&ac]),origin0=i['origin0'].tolist(),previous_origin0=bi['origin0'].tolist(),manual_used=False)
(O/'coarse_ptv_audit.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
