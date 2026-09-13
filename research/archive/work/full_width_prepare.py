"""Prepare the complete horizontal field, preserving the verified raw overlap."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,hashlib
import numpy as np
import h5py
from PIL import Image
from scipy.ndimage import gaussian_filter
O=Path('work/full_width');O.mkdir(exist_ok=True)
old=np.load('work/large_frame/inputs.npz');meta=np.load('work/metadata.npz')
origin=np.array([0,260]);end=np.array([2048,840]);oldorigin=np.array([299,339])
previous_origin=old['origin0'];maxdepth=354
newbase='data/ExpLCL_1_03_123'
oldbase='data/ExpLCL_1_03-123'
with h5py.File(newbase+'_PIV.mat','r') as f:
    ns=[f['imSurf'+fr+'/surfacePIVImg'][:].ravel() for fr in 'ab']
    DX=float(f['compVel/DX'][0,0]);DT=float(f['compVel/DT'][0,0])
    supplied_dx=f['compVel/delta_x1'][:].T[260:840,:]
    supplied_dy=-f['compVel/delta_z1'][:].T[260:840,:]
geom=[old['full_surface_'+fr] for fr in 'ab'];raws=[];availability=[];normal=[];masks=[];records=[]
for j,fr in enumerate('AB'):
    path=Path(newbase+'_img'+fr+'.tif');new=np.asarray(Image.open(str(path)),float)
    prior=np.asarray(Image.open(oldbase+'_img'+fr+'.tif'),float)
    overlap=(slice(339,840),slice(299,800));visible=new[overlap]>0
    assert np.array_equal(new[overlap][visible],prior[visible])
    first=(new>0).argmax(axis=0);assert np.array_equal(first,np.round(ns[j]))
    assert np.allclose(ns[j],geom[j]+12,rtol=0,atol=1e-10)
    yy,xx=np.indices(new.shape);av=yy>=first[None,:]
    raw=new.copy();raw[overlap]=prior;av[overlap]=True
    raw=raw[260:840];av=av[260:840];sf=geom[j]-origin[1]
    sy=np.indices(raw.shape)[0];v=av&(sy>=sf[None,:]+10)
    vf=v.astype(float)
    sm=gaussian_filter(raw*vf,.65)/np.maximum(gaussian_filter(vf,.65),1e-5)
    mean=gaussian_filter(sm*vf,7)/np.maximum(gaussian_filter(vf,7),1e-5)
    hp=sm-mean;rms=np.sqrt(gaussian_filter(hp*hp*vf,9)/np.maximum(gaussian_filter(vf,9),1e-5)+25)
    normal.append(np.clip(hp/rms,-3,4));raws.append(raw);availability.append(av);masks.append(v)
    # Previous preparation is identical in raw pixels, masks, and surface geometry.
    crop=(slice(40,580),slice(240,860))
    assert np.array_equal(raw[crop],old['raw'+fr])
    assert np.array_equal(av[crop],old['availability_'+fr.lower()])
    assert np.array_equal(v[crop],old['v'+fr.lower()])
    records.append(dict(frame=fr,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        full_columns_verified=2048,restored_old_pixels=int(np.sum((~visible)&(prior>0)))))
gx,gy=np.meshgrid(np.arange(7,2048,8.),np.arange(347,796,8.))
fullpoints=np.c_[gx.ravel(),gy.ravel()]
depth=fullpoints[:,1]-np.interp(fullpoints[:,0],np.arange(2048),geom[0])
fullpoints=fullpoints[(depth>=12)&(depth<=maxdepth)]
previous_points=old['points']+previous_origin
lookup={tuple(p) for p in fullpoints}
assert all(tuple(p) in lookup for p in previous_points)
manual=old['manual_xy']+previous_origin
np.savez_compressed(O/'inputs.npz',A=normal[0],B=normal[1],rawA=raws[0],rawB=raws[1],
 availability_a=availability[0],availability_b=availability[1],va=masks[0],vb=masks[1],
 surface_a=geom[0]-origin[1],surface_b=geom[1]-origin[1],full_surface_a=geom[0],full_surface_b=geom[1],
 new_mask_surface_a=ns[0],new_mask_surface_b=ns[1],origin0=origin,old_origin0=oldorigin,
 previous_origin0=previous_origin,points=fullpoints-origin,previous_points=previous_points-origin,
 original_points=old['original_points']+previous_origin-origin,manual_xy=manual-origin,
 manual_truth=old['manual_truth'],supplied_dx=supplied_dx,supplied_dy=supplied_dy,DX=DX,DT=DT,max_depth=maxdepth)
manifest=dict(input_frames=records,origin_zero_based=origin.tolist(),end_exclusive=end.tolist(),grid_nodes=len(fullpoints),
 previous_grid_nodes=len(previous_points),new_grid_nodes=len(fullpoints)-len(previous_points),grid_horizontal_step_pixels=8,
 grid_full_x_min=float(fullpoints[:,0].min()),grid_full_x_max=float(fullpoints[:,0].max()),max_depth_pixels=maxdepth,
 full_image_width_pixels=2048,analysis_depth_cm_if_SI=354*DX*100,full_image_width_cm_if_SI=2048*DX*100,
 DX=DX,DT=DT,physical_units_confirmed=False,previous_roi_raw_and_masks_exact=True,
 surface_note='Same full geometric surface as previous analysis; provided surfacePIVImg equals this trace+12px and marks already-applied glare mask.',
 preservation_note='Previous2184localpatches and3748automaticcandidate records are retained; newimage-normalization context and neighbors may affect only addedpatches and blendedboundaryvalues.',
 coordinate_note='Fullimage zero-based xright ydown; oldrawcrop starts[299,339]. No synthetic pixels; original overlap pixels are retained from earlier suppliedTIFFs.')
(O/'input_manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
