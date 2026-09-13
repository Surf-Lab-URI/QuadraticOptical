"""Prepare fresh image-only top-centimetre inputs. Never read supplied velocities."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import hashlib,json,argparse
import numpy as np
import h5py
from PIL import Image
from scipy.ndimage import gaussian_filter

ROOT=Path(__file__).resolve().parents[2]
BASE=Path('historical-user-files/Downloads')
ALLOWLIST=['compVel/DX','compVel/DT','imSurfa/surfacePIVImg','imSurfb/surfacePIVImg']

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def normalize(raw,mask):
    v=mask.astype(float)
    sm=gaussian_filter(raw*v,.65)/np.maximum(gaussian_filter(v,.65),1e-5)
    mean=gaussian_filter(sm*v,7)/np.maximum(gaussian_filter(v,7),1e-5)
    high=sm-mean
    rms=np.sqrt(gaussian_filter(high*high*v,9)/np.maximum(gaussian_filter(v,9),1e-5)+25)
    return np.clip(high/rms,-3,4)

def prepare(pair):
    folder=ROOT/'work/image_only_cm'/str(pair)
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'inputs.npz').exists():raise RuntimeError('Fresh preparation refuses to overwrite inputs')
    base=BASE/('ExpLCL_1_03_'+str(pair))
    mat=Path(str(base)+'_PIV.mat')
    accessed=[];metadata={};metadata_hashes={}
    with h5py.File(mat,'r') as f:
        for key in ALLOWLIST:
            a=f[key][...];accessed.append(key);metadata[key]=a
            metadata_hashes[key]=hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
    dx=float(metadata['compVel/DX'].ravel()[0]);dt=float(metadata['compVel/DT'].ravel()[0])
    assert dx>0 and dt>0
    requested=.01/dx;fitting=requested+64;detector=fitting+25
    arrays={};records=[]
    for fr in 'ab':
        imagepath=Path(str(base)+'_img'+fr.upper()+'.tif')
        original=np.asarray(Image.open(imagepath))
        assert original.shape==(2048,2048) and original.dtype==np.uint8
        assert (original>0).any(axis=0).all()
        first=(original>0).argmax(axis=0)
        last=2047-(original[::-1]>0).argmax(axis=0)
        provided=metadata['imSurf'+fr+'/surfacePIVImg'].ravel()
        assert provided.shape==(2048,)
        assert np.array_equal(first,np.round(provided)) and np.all(last==2047)
        surface=provided-12.
        availability=np.arange(2048)[:,None]>=first[None,:]
        valid=availability&(np.arange(2048)[:,None]>=surface[None,:]+10)
        raw=original.astype(float)
        arrays.update({fr.upper():normalize(raw,valid),'raw'+fr.upper():raw,
            'availability_'+fr:availability,'v'+fr:valid,'surface_'+fr:surface,
            'first_retained_row_'+fr:first,'last_retained_row_'+fr:last})
        records.append(dict(frame=fr,path=str(imagepath),sha256=sha(imagepath),
            shape=list(original.shape),dtype=str(original.dtype),raw_pixel_values_unchanged=True,
            first_retained_range=[int(first.min()),int(first.max())],
            first_retained_equals_rounded_surfacePIVImg=True,
            available_pixels=int(availability.sum())))
    xx,yy=np.meshgrid(np.arange(7,2048,8.),np.arange(7,2048,8.))
    points=np.c_[xx.ravel(),yy.ravel()]
    depth=points[:,1]-np.interp(points[:,0],np.arange(2048),arrays['surface_a'])
    keep=(depth>=12)&(depth<=fitting)
    points=points[keep];depth=depth[keep]
    assert np.all(arrays['va'][points[:,1].astype(int),points[:,0].astype(int)])
    arrays.update(points=points,origin0=np.array([0,0]),DX=np.array(dx),DT=np.array(dt),
        requested_max_depth_px=np.array(requested),fitting_max_depth_px=np.array(fitting),
        detector_max_depth_px=np.array(detector),source_depth=depth,
        image_only=np.array(True),supplied_velocity_used=np.array(False),
        surface_geometry_inferred=np.array(True),surface_trace_offset_pixels=np.array(12.),
        physical_units_confirmed=np.array(False),pair_number=np.array(pair))
    np.savez_compressed(folder/'inputs.npz',**arrays)
    manifest=dict(pair=pair,image_only=True,supplied_velocity_used=False,
        previous_prediction_or_track_files_read=False,mat_path=str(mat),
        exact_mat_dataset_allowlist=ALLOWLIST,actual_mat_dataset_reads=accessed,
        mat_dataset_sha256=metadata_hashes,image_records=records,
        inputs_sha256=sha(folder/'inputs.npz'),input_keys=sorted(arrays),
        DX=dx,DT=dt,physical_units_confirmed=False,
        units_note='DX assumed metres/pixel and DT seconds; file datasets contain no explicit unit labels.',
        surface_geometry_inferred=True,surface_trace_offset_pixels=12.,
        surface_convention='surfacePIVImg minus 12 pixels; inferred export convention, not independently recovered from masked TIFFs.',
        actual_image_boundary_verified=True,grid_spacing_pixels=8,grid_phase_xy=[7,7],
        grid_nodes=len(points),requested_nodes=int(np.sum(depth<=requested)),
        requested_max_depth_m=.01,requested_max_depth_px=requested,
        fitting_max_depth_px=fitting,detector_max_depth_px=detector,
        support_halo_note='64 pixel fitting halo and further 25 pixel particle-detection halo support top-centimetre estimates; no deeper results are reported.',
        input_audit_pass=True)
    (folder/'input_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:manifest[k] for k in ['pair','grid_nodes','requested_nodes','DX','DT','inputs_sha256']}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('pairs',nargs='*',type=int,default=[80,100])
    for pair in p.parse_args().pairs:prepare(pair)
