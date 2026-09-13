"""Prepare full observed-water domains for new pairs 80 and 100.

No old image crop or velocity field is inserted. Surface geometry uses the
explicitly inferred prior export convention; actual TIFF availability is verified.
Dense supplied fields are initializers. Native PIV arrays remain separate.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import argparse,gc,hashlib,json
import numpy as np
import h5py
from PIL import Image
from scipy.ndimage import gaussian_filter

ROOT=Path(__file__).resolve().parents[2]
BASE=Path.home()/'Downloads'
OUT=ROOT/'work/pairs'

def normalize(raw,mask):
    v=mask.astype(float)
    sm=gaussian_filter(raw*v,.65)/np.maximum(gaussian_filter(v,.65),1e-5)
    mean=gaussian_filter(sm*v,7)/np.maximum(gaussian_filter(v,7),1e-5)
    high=sm-mean
    rms=np.sqrt(gaussian_filter(high*high*v,9)/np.maximum(gaussian_filter(v,9),1e-5)+25)
    return np.clip(high/rms,-3,4)

def describe(a):
    a=np.asarray(a);finite=np.isfinite(a);v=a[finite]
    return dict(shape=list(a.shape),dtype=str(a.dtype),finite=int(finite.sum()),
                minimum=float(v.min()) if v.size else None,
                median=float(np.median(v)) if v.size else None,
                maximum=float(v.max()) if v.size else None)

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def native_lookup(x,y,values,points):
    # The prediction grid coincides exactly with every other native node.
    # Exact lookup avoids 0*NaN interpolation contamination at an observed node.
    ix=np.searchsorted(x,points[:,0]);iy=np.searchsorted(y,points[:,1])
    inside=(ix<len(x))&(iy<len(y))
    exact=np.zeros(len(points),bool)
    exact[inside]=(x[ix[inside]]==points[inside,0])&(y[iy[inside]]==points[inside,1])
    out=np.full(len(points),np.nan)
    out[exact]=values[iy[exact],ix[exact]]
    return out,exact

def prepare(pair):
    folder=OUT/str(pair);folder.mkdir(parents=True,exist_ok=True)
    base=BASE/f'ExpLCL_1_03_{pair}'
    mat=Path(str(base)+'_PIV.mat')
    with h5py.File(mat,'r') as f:
        top=[k for k in f.keys() if k!='#refs#']
        assert set(top)=={'compVel','imSurfa','imSurfb'}
        traces={fr:{k:f['imSurf'+fr+'/'+k][:].ravel() for k in
                    ['surfacePIVImg','surfaceSurfImgScaled','surface_raw']} for fr in 'ab'}
        dx=float(f['compVel/DX'][0,0]);dt=float(f['compVel/DT'][0,0])
        gs=float(f['compVel/GS'][0,0]);iw=float(f['compVel/IW'][0,0])
        sx=f['compVel/delta_x1'][:].T.copy()
        sy=-f['compVel/delta_z1'][:].T.copy()
        cx=f['compVel/xPIV'][:].ravel()-1
        cy=f['compVel/zPIV'][:].ravel()-1
        cdx=f['compVel/delta_x'][:].T.copy()
        cdy=-f['compVel/delta_z'][:].T.copy()
        ccor=f['compVel/dcor'][:].T.copy()
        bad={fr:float(f['imSurf'+fr+'/badFrameBool'][0,0]) for fr in 'ab'}
    assert np.array_equal(cx,np.arange(3,2044,4))
    assert np.array_equal(cy,np.arange(3,2044,4))
    records=[];arrays={};geometric=[]
    for fr in 'ab':
        imagepath=Path(str(base)+'_img'+fr.upper()+'.tif')
        original=np.asarray(Image.open(imagepath))
        assert original.shape==(2048,2048) and original.dtype==np.uint8
        columns=(original>0).any(axis=0)
        assert columns.all()
        first=(original>0).argmax(axis=0)
        last=2047-(original[::-1]>0).argmax(axis=0)
        provided=traces[fr]['surfacePIVImg']
        assert np.array_equal(first,np.round(provided))
        assert np.all(last==2047)
        assert np.allclose(traces[fr]['surfaceSurfImgScaled'][654:2702]-1429,
                           provided,rtol=0,atol=1e-10)
        # No explicit unshifted surface_orig exists in these two MAT files.
        # This offset is inherited as an inferred convention, not newly measured.
        surface=provided-12.0
        availability=np.arange(2048)[:,None]>=first[None,:]
        valid=availability&(np.arange(2048)[:,None]>=surface[None,:]+10)
        raw=original.astype(float)
        arrays.update({f'raw{fr.upper()}':raw,fr.upper():normalize(raw,valid),
                       'availability_'+fr:availability,'v'+fr:valid,
                       'surface_'+fr:surface,'full_surface_'+fr:surface,
                       'new_mask_surface_'+fr:provided,
                       'first_retained_row_'+fr:first,
                       'last_retained_row_'+fr:last})
        geometric.append(surface)
        records.append(dict(frame=fr.upper(),path=str(imagepath),sha256=sha(imagepath),
            shape=list(original.shape),dtype=str(original.dtype),
            all_columns_observed=True,first_retained_row_range=[int(first.min()),int(first.max())],
            last_retained_row_range=[int(last.min()),int(last.max())],
            first_retained_equals_rounded_surfacePIVImg=True,
            raw_pixel_values_unchanged=True,original_crop_restored=False,
            available_pixel_count=int(availability.sum()),primary_mask_pixel_count=int(valid.sum()),
            nonzero_pixel_count=int(np.count_nonzero(original)),
            badFrameBool=bad[fr],surfacePIVImg_range=[float(provided.min()),float(provided.max())]))
    xx,yy=np.meshgrid(np.arange(7,2048,8.),np.arange(7,2048,8.))
    points=np.c_[xx.ravel(),yy.ravel()]
    depth=points[:,1]-np.interp(points[:,0],np.arange(2048),geometric[0])
    points=points[depth>=12]
    depth=points[:,1]-np.interp(points[:,0],np.arange(2048),geometric[0])
    atx,exact=native_lookup(cx,cy,cdx,points)
    aty,exacty=native_lookup(cx,cy,cdy,points)
    atcor,exactcor=native_lookup(cx,cy,ccor,points)
    assert np.array_equal(exact,exacty) and np.array_equal(exact,exactcor)
    nx,ny=np.meshgrid(cx.astype(int),cy.astype(int))
    native_visible=arrays['va'][ny,nx]
    native_vector_finite=np.isfinite(cdx)&np.isfinite(cdy)
    native_correlation_finite=np.isfinite(ccor)
    arrays.update(points=points,origin0=np.array([0,0]),supplied_dx=sx,supplied_dy=sy,
        classical_x=cx,classical_y=cy,classical_dx=cdx,classical_dy=cdy,
        classical_dcor=ccor,classical_native_source_visible=native_visible,
        classical_at_points=np.c_[atx,aty],classical_dcor_at_points=atcor,
        classical_exact_node_at_points=exact,DX=np.array(dx),DT=np.array(dt),
        GS=np.array(gs),IW=np.array(iw),surface_geometry_inferred=np.array(True),
        surface_trace_offset_pixels=np.array(12.),pair_number=np.array(pair),
        source_depth=depth,minimum_source_depth=np.array(12.))
    assert np.all(arrays['va'][points[:,1].astype(int),points[:,0].astype(int)])
    manifest=dict(pair=pair,mat_path=str(mat),mat_sha256=sha(mat),mat_top_variables=top,
        image_records=records,origin0=[0,0],shape=[2048,2048],grid_spacing_pixels=8,
        grid_phase_xy=[7,7],grid_nodes=len(points),grid_x_range=[float(points[:,0].min()),float(points[:,0].max())],
        grid_y_range=[float(points[:,1].min()),float(points[:,1].max())],
        source_depth_range=[float(depth.min()),float(depth.max())],
        minimum_source_depth=12,maximum_source_depth_cap=None,
        full_observed_vertical_water_domain=True,no_old_image_or_flow_inserted=True,
        surface_geometry_inferred=True,explicit_unshifted_surface_present=False,
        surface_convention='surface_a/b = surfacePIVImg -12, transferred from verified pair123 exporter convention; not independently determined from these masked images.',
        surface_export_layout_check='surfaceSurfImgScaled[654:2702]-1429 equals surfacePIVImg exactly in both frames.',
        actual_availability_verified=True,
        primary_mask='availability & (image_y >= inferred_geometric_surface +10)',
        DX=dx,DT=dt,GS=gs,IW=iw,physical_units_confirmed=False,
        units_note='DX and DT numeric values match earlier pair; metres/pixel and seconds remain assumed rather than explicit file units.',
        native_PIV_axes='xPIV-1,zPIV-1 =3,7,...,2043; HDF5 arrays transposed to [y,x]',
        image_displacement_sign='dx=delta_x, dy=-delta_z',
        dense_PIV_role='delta_x1 and -delta_z1 are initializer fields only; never use these interpolated/extrapolated arrays as native comparison observations.',
        classical_sampling='Exact native lookup at coincident prediction grid nodes; NaN outside native axes. Avoids 0*NaN contamination by generic linear interpolation.',
        native_vector_finite_count=int(native_vector_finite.sum()),
        native_correlation_finite_count=int(native_correlation_finite.sum()),
        native_visible_vector_count=int(np.sum(native_visible&native_vector_finite)),
        native_visible_vector_with_correlation_count=int(np.sum(native_visible&native_vector_finite&native_correlation_finite)),
        native_finite_vectors_in_unavailable_or_excluded_source=int(np.sum(native_vector_finite&~native_visible)),
        native_vector_at_points_count=int(np.sum(np.isfinite(atx)&np.isfinite(aty))),
        native_vector_with_correlation_at_points_count=int(np.sum(np.isfinite(atx)&np.isfinite(aty)&np.isfinite(atcor))),
        manual_reference_present=False,
        normalization=dict(smooth_sigma=.65,mean_sigma=7,rms_sigma=9,rms_variance_floor=25,denominator_floor=1e-5,clip=[-3,4]),
        field_summaries={k:describe(v) for k,v in
                         [('classical_dx',cdx),('classical_dy',cdy),('classical_dcor',ccor),
                          ('supplied_dx',sx),('supplied_dy',sy)]})
    np.savez_compressed(folder/'inputs.npz',**arrays)
    (folder/'input_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({k:manifest[k] for k in ['pair','grid_nodes','grid_x_range','grid_y_range',
        'source_depth_range','surface_geometry_inferred','native_vector_at_points_count',
        'native_vector_with_correlation_at_points_count','native_finite_vectors_in_unavailable_or_excluded_source']}),
        flush=True)
    del arrays,raw,original,sx,sy,cdx,cdy,ccor,points
    gc.collect()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('pairs',nargs='*',type=int,default=[80,100])
    parser.add_argument('--input-dir',type=Path,default=BASE)
    args=parser.parse_args();BASE=args.input_dir
    for pair in args.pairs:prepare(pair)
