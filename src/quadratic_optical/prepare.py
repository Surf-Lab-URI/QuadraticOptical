"""Prepare image-only inputs; comparison velocities are never read here."""
from pathlib import Path
import hashlib,json
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from .matio import read_mat_fields

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

CONTRAST_FLOOR=25.


def contrast_floor(intensity_scale):
    """Variance floor matched to the intensity conversion.

    The bare 25 is a variance in 0-255 units, a standard-deviation floor of five
    brightness levels, and it only means that when one converted unit is one
    8-bit count. Scaling it by the conversion makes normalize invariant under a
    linear rescale of intensity: high scales with the conversion, high squared
    with its square, and the floor with its square, so high/rms is unchanged.
    Without this a 12-bit image mapped into 0-255 has its real contrast divided
    by about sixteen while the floor stays fixed, and the normalization goes
    flat with nothing raising an error.
    """
    scale=float(intensity_scale)
    if not np.isfinite(scale) or scale<=0:raise ValueError('intensity_scale must be finite and positive.')
    return CONTRAST_FLOOR*scale*scale


def normalize(raw,valid,floor=CONTRAST_FLOOR):
    v=valid.astype(float)
    sm=gaussian_filter(raw*v,.65)/np.maximum(gaussian_filter(v,.65),1e-5)
    high=sm-gaussian_filter(sm*v,7)/np.maximum(gaussian_filter(v,7),1e-5)
    rms=np.sqrt(gaussian_filter(high*high*v,9)/np.maximum(gaussian_filter(v,9),1e-5)+floor)
    return np.clip(high/rms,-3,4)

def first_nonzero(a):
    if not (a!=0).any(axis=0).all():raise ValueError('nonzero_boundary needs at least one retained nonzero pixel in every column. Supply an explicit surface/availability instead.')
    return (a!=0).argmax(axis=0)

def atomic_npz(path,arrays):
    temp=path.with_suffix('.tmp')
    with temp.open('wb') as f:np.savez_compressed(f,**arrays)
    temp.replace(path)

def prepare(pair,directory,config):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    keys=[]
    if config['dx_m_per_px'] is None:keys.append('compVel.DX')
    if config['dt_s'] is None:keys.append('compVel.DT')
    surface=dict(config['surface']);mode=surface['mode']
    if mode=='auto':mode='sidecar' if pair.surface_file else 'piv_mat' if pair.piv_mat else 'missing'
    if mode=='missing':raise ValueError('No surface geometry. Supply NAME_surface.npz, a PIV metadata file, or an explicit surface mode in the config.')
    if mode=='piv_mat':keys+=['imSurfa.surfacePIVImg','imSurfb.surfacePIVImg']
    if keys and pair.piv_mat is None:raise ValueError('Calibration/surface metadata are missing. Set dx_m_per_px and dt_s and provide surface geometry.')
    metadata=read_mat_fields(pair.piv_mat,keys) if keys else {}
    dx=float(config['dx_m_per_px'] if config['dx_m_per_px'] is not None else np.asarray(metadata['compVel.DX']).squeeze())
    dt=float(config['dt_s'] if config['dt_s'] is not None else np.asarray(metadata['compVel.DT']).squeeze())
    if not np.isfinite([dx,dt]).all() or dx<=0 or dt<=0:raise ValueError('DX and DT must be positive finite values in m/pixel and seconds.')
    if config['depth_m'] < 20*dx:
        raise ValueError('depth_m must reach at least 20 pixels below the surface for the common-domain integration band.')
    images={};records=[];boundaries={}
    for fr,path in [('A',pair.image_a),('B',pair.image_b)]:
        with Image.open(path) as im:
            if getattr(im,'n_frames',1)!=1:raise ValueError('Multi-page images are ambiguous; export separate A/B frames: '+str(path))
            a=np.asarray(im)
        if a.ndim!=2:raise ValueError('Images must be grayscale 2-D arrays: '+str(path))
        if mode=='nonzero_boundary' or config['availability']=='nonzero_boundary':
            # Retained-image geometry comes from the original pixels, before
            # a nonzero intensity offset can turn discarded zero rows nonzero.
            boundaries[fr]=first_nonzero(a)
        if a.dtype!=np.uint8 and config['intensity_scale']==1 and config['intensity_offset']==0:
            raise ValueError('Particle thresholds are calibrated to 8-bit intensity. Set an explicit intensity_scale (and optional offset) for '+str(a.dtype)+'.')
        a=a.astype(float)*config['intensity_scale']+config['intensity_offset']
        if not np.isfinite(a).all() or a.min()<0 or a.max()>255+1e-9:raise ValueError('Scaled intensities must lie within [0,255]. Set the intensity conversion explicitly.')
        images[fr]=a
        records.append({'frame':fr,'file_name':Path(path).name,'sha256':sha(path),'shape':list(a.shape)})
    if images['A'].shape!=images['B'].shape:raise ValueError('A/B image dimensions differ.')
    height,width=images['A'].shape
    if min(height,width)<48:raise ValueError('Images must be at least 48 pixels on each axis for these patch sizes.')
    if mode=='sidecar':
        if pair.surface_file is None:raise ValueError('surface.mode=sidecar requires NAME_surface.npz.')
        with np.load(pair.surface_file,allow_pickle=False) as f:
            surf={fr:f['surface_'+fr.lower()].astype(float).ravel() for fr in 'AB'}
            masks={}
            for fr in 'AB':
                key='availability_'+fr.lower()
                if key in f:
                    value=np.asarray(f[key])
                    if value.dtype.kind not in 'bifu' or not np.isfinite(value).all() or not np.isin(value,[0,1]).all():
                        raise ValueError('Sidecar availability must contain finite binary values: True/1=retained, False/0=missing.')
                    masks[fr]=value.astype(bool)
        # Sidecars always contain zero-based image row coordinates.
        base=0;offset=float(surface.get('offset_px',0))
    else:
        masks={};base=int(surface.get('index_base',1));offset=float(surface.get('offset_px',0))
        if mode=='piv_mat':surf={fr:np.asarray(metadata['imSurf'+fr.lower()+'.surfacePIVImg']).ravel().astype(float) for fr in 'AB'}
        elif mode=='constant':
            if 'row_a' not in surface:raise ValueError('surface.mode=constant requires row_a (and optional row_b).')
            surf={fr:np.full(width,float(surface.get('row_'+fr.lower(),surface['row_a']))) for fr in 'AB'}
        elif mode=='nonzero_boundary':surf={fr:boundaries[fr].astype(float) for fr in 'AB'};base=0
        else:raise ValueError('Unsupported surface mode: '+mode)
    arrays={}
    yy=np.arange(height)[:,None]
    exclusion=float(config['surface_exclusion_px']);floor=contrast_floor(config['intensity_scale'])
    for fr in 'AB':
        s=surf[fr]-base+offset
        if s.shape!=(width,) or not np.isfinite(s).all():raise ValueError('Each surface must contain one finite row coordinate per image column.')
        if np.any(s<-.5) or np.any(s>height-.5):raise ValueError('Surface coordinates lie outside the image. Check index_base/offset_px.')
        avail=masks.get(fr,np.ones((height,width),bool))
        if avail.shape!=(height,width):raise ValueError('Availability mask must match image dimensions.')
        if config['availability']=='nonzero_boundary':avail=avail&(yy>=boundaries[fr][None,:])
        valid=avail&(yy>=s[None,:]+exclusion)
        arrays.update({fr:normalize(images[fr],valid,floor),'raw'+fr:images[fr],
            'availability_'+fr.lower():avail,'v'+fr.lower():valid,'surface_'+fr.lower():s})
    phase=int(config['grid_phase_px']);spacing=int(config['grid_spacing_px'])
    xx,yy=np.meshgrid(np.arange(phase,width,spacing,dtype=float),np.arange(phase,height,spacing,dtype=float))
    points=np.c_[xx.ravel(),yy.ravel()]
    depth=points[:,1]-np.interp(points[:,0],np.arange(width),arrays['surface_a'])
    requested=config['depth_m']/dx;fitting=requested+64;detector=fitting+25
    use=(depth>=12)&(depth<=fitting)&arrays['va'][points[:,1].astype(int),points[:,0].astype(int)]
    points=points[use];depth=depth[use]
    if len(points)<16 or not np.any(depth<=requested):raise ValueError('Insufficient source grid points in the requested depth band. Check geometry/calibration/depth.')
    arrays.update(points=points,source_depth=depth,origin0=np.array([0,0]),DX=np.array(dx),DT=np.array(dt),
        requested_max_depth_px=np.array(requested),fitting_max_depth_px=np.array(fitting),detector_max_depth_px=np.array(detector),
        image_only=np.array(True),supplied_velocity_used=np.array(False),physical_units_confirmed=np.array(True),
        surface_geometry_inferred=np.array(mode=='nonzero_boundary' or offset!=0),surface_trace_offset_px=np.array(offset),
        surface_exclusion_px=np.array(exclusion),contrast_floor=np.array(floor),intensity_scale=np.array(float(config['intensity_scale'])),
        pair_number=np.array(pair.pair_number if pair.pair_number is not None else -1))
    plan={'schema':1,'pair_name':pair.name,'experiment':pair.experiment,'pair_number':pair.pair_number,
        'image_records':records,'config':config,'surface_mode_resolved':mode,
        'mat_dataset_reads':keys,'metadata_values_sha256':{k:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k,v in metadata.items()},
        'surface_sidecar_sha256':sha(pair.surface_file) if mode=='sidecar' else None,
        'implementation_sha256':sha(__file__),'DX':dx,'DT':dt,'image_only':True,'supplied_velocity_used':False}
    # Worker count affects execution only; PIV quality belongs solely to the
    # held-out comparison. Retain the full initial config in the manifest while
    # signing every scientific preparation/integration setting.
    scientific_config={key:value for key,value in config.items() if key not in ['workers','piv_quality']}
    signed_plan=dict(plan,config=scientific_config)
    signature=hashlib.sha256(json.dumps(signed_plan,sort_keys=True).encode()).hexdigest()
    path=directory/'inputs.npz';manifest=directory/'input_manifest.json'
    if path.exists() or manifest.exists():
        if not path.exists() or not manifest.exists():raise ValueError('Incomplete input preparation exists. Use a new output directory.')
        old=json.loads(manifest.read_text())
        if old.get('preparation_signature')!=signature or old.get('inputs_sha256')!=sha(path):
            raise ValueError('Existing results belong to different inputs/settings. Choose a new output directory; original results are preserved.')
        return old
    atomic_npz(path,arrays);plan.update(preparation_signature=signature,inputs_sha256=sha(path),grid_nodes=len(points))
    manifest.write_text(json.dumps(plan,indent=2)+'\n')
    return plan
