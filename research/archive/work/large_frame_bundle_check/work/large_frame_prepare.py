"""Prepare verified larger same-observation frames and preserve original overlap."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,hashlib
import numpy as np
import h5py
from PIL import Image
from scipy.ndimage import gaussian_filter

O=Path('work/large_frame');O.mkdir(exist_ok=True)
OLD=Path('data/ExpLCL_1_03-123')
NEW=Path('data/ExpLCL_1_03_123')
oldmeta=np.load('work/metadata.npz')
origin=np.array([240,300]);oldorigin=np.array([299,339]);end=np.array([860,840])
dx=None;dt=None
with h5py.File(str(NEW)+'_PIV.mat','r') as f:
    ns=[f['imSurf'+fr+'/surfacePIVImg'][:].ravel() for fr in 'ab']
    dx=float(f['compVel/DX'][0,0]);dt=float(f['compVel/DT'][0,0])
    supplied_dx=f['compVel/delta_x1'][:].T[origin[1]:end[1],origin[0]:end[0]]
    supplied_dy=-f['compVel/delta_z1'][:].T[origin[1]:end[1],origin[0]:end[0]]
geometric=[oldmeta['surf'+fr+'_orig'].ravel()-2 for fr in 'ab']
raws=[];avs=[];news=[];masks=[];normal=[];records=[]
for i,fr in enumerate('AB'):
    path=Path(str(NEW)+'_img'+fr+'.tif');new=np.asarray(Image.open(path),float)
    old=np.asarray(Image.open(str(OLD)+'_img'+fr+'.tif'),float)
    oldroi=(slice(oldorigin[1],oldorigin[1]+501),slice(oldorigin[0],oldorigin[0]+501))
    observed=new[oldroi]>0
    assert np.array_equal(new[oldroi][observed],old[observed])
    yy,xx=np.indices(new.shape)
    first=(new>0).argmax(axis=0)
    assert np.array_equal(first,np.round(ns[i]))
    assert np.allclose(ns[i],geometric[i]+12,rtol=0,atol=1e-10)
    availability=yy>=first[None,:]
    raw=new.copy();raw[oldroi]=old;availability[oldroi]=True
    crop=(slice(origin[1],end[1]),slice(origin[0],end[0]))
    r=raw[crop];av=availability[crop];sf=geometric[i][origin[0]:end[0]]-origin[1]
    sy,sx=np.indices(r.shape);v=av&(sy>=sf[None,:]+10)
    def norm(im,valid):
        vf=valid.astype(float)
        smooth=gaussian_filter(im*vf,.65)/np.maximum(gaussian_filter(vf,.65),1e-5)
        mean=gaussian_filter(smooth*vf,7)/np.maximum(gaussian_filter(vf,7),1e-5)
        hp=smooth-mean
        rms=np.sqrt(gaussian_filter(hp*hp*vf,9)/np.maximum(gaussian_filter(vf,9),1e-5)+25)
        return np.clip(hp/rms,-3,4)
    raws.append(r);avs.append(av);masks.append(v);normal.append(norm(r,v));news.append(new[crop])
    records.append(dict(frame=fr,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        old_overlap_retained_pixels=int(observed.sum()),old_pixels_restored=int(np.sum((~observed)&(old>0)))))
gx,gy=np.meshgrid(np.arange(319,720,8.),np.arange(355,796,8.))
fullpoints=np.c_[gx.ravel(),gy.ravel()]
depth=fullpoints[:,1]-np.interp(fullpoints[:,0],np.arange(2048),geometric[0])
fullpoints=fullpoints[(depth>=12)&(depth<=354)]
oldgrid=np.load('work/final_ptv13.npz')['points']+oldorigin
existing={tuple(p) for p in fullpoints}
assert all(tuple(p) in existing for p in oldgrid)
manual=oldmeta['p'][:,0].T-1+oldorigin
truth=oldmeta['p'][:,1].T-oldmeta['p'][:,0].T
np.savez_compressed(O/'inputs.npz',A=normal[0],B=normal[1],rawA=raws[0],rawB=raws[1],
                    newA=news[0],newB=news[1],availability_a=avs[0],availability_b=avs[1],va=masks[0],vb=masks[1],
                    surface_a=geometric[0][origin[0]:end[0]]-origin[1],
                    surface_b=geometric[1][origin[0]:end[0]]-origin[1],
                    full_surface_a=geometric[0],full_surface_b=geometric[1],
                    new_mask_surface_a=ns[0],new_mask_surface_b=ns[1],
                    origin0=origin,old_origin0=oldorigin,points=fullpoints-origin,
                    original_points=oldgrid-origin,manual_xy=manual-origin,manual_truth=truth,
                    supplied_dx=supplied_dx,supplied_dy=supplied_dy,DX=dx,DT=dt,max_depth=354)
meta=dict(input_frames=records,analysis_origin_zero_based=origin.tolist(),analysis_end_exclusive=end.tolist(),
          old_crop_origin_zero_based=oldorigin.tolist(),grid_nodes=len(fullpoints),old_grid_nodes=len(oldgrid),
          display_x_full_pixels=[319,719],maximum_depth_pixels=354,
          geometric_surface_note='Original traced surface transformed exactly to full image coordinates; equals old surfa_orig/surfb_orig minus2. New MAT surfacePIVImg equals this geometry plus12 and defines already-applied glare mask, not an additional physical surface.',
          old_manual_transform='global zero-based = (old p[:,frame,:].T -1) +[299,339]; original p_orig bookkeeping minus[2,2].',
          DX=dx,DT=dt,physical_units_confirmed=False,
          units_assumption='If DX is metres/pixel and DT seconds: field width2.26cm and depth2cm. Retain pixel quantities as primary until confirmed.',
          original_model_preservation='Original local polynomial fits will be retained in the old grid; only new locations fitted with original method. No manual target used for fitting.',
          references_note='New compVel arrays are provided computed PIV displacements, not manual validation. HDF5 arrays transposed; delta_z negated for image-down convention.')
(O/'input_manifest.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2))
