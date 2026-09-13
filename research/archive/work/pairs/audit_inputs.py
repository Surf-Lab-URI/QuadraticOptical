"""Independent read-only audit of pair inputs against the supplied source files."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json
import numpy as np
import h5py
from PIL import Image
from scipy.ndimage import gaussian_filter,map_coordinates

ROOT=Path(__file__).resolve().parents[2]
BASE=Path('historical-user-files/Downloads')

def run(pair):
    folder=ROOT/'work/pairs'/str(pair)
    report={'pair':pair,'failures':[],'frames':{}}
    with np.load(folder/'inputs.npz',allow_pickle=False) as z,h5py.File(BASE/f'ExpLCL_1_03_{pair}_PIV.mat') as m:
        q=z['points'];depth=z['source_depth'];sa=z['surface_a']
        for f in 'ab':
            raw=np.asarray(Image.open(BASE/f'ExpLCL_1_03_{pair}_img{f.upper()}.tif'))
            saved=z['raw'+f.upper()];avail=z['availability_'+f];valid=z['v'+f]
            trace=m['imSurf'+f+'/surfacePIVImg'][:].ravel()
            first=np.argmax(raw!=0,axis=0)
            last=2047-np.argmax(raw[::-1]!=0,axis=0)
            independent_av=np.arange(raw.shape[0])[:,None]>=first
            expected_geom=trace-12
            r={'raw_array_exactly_equal_to_original_TIFF':bool(np.array_equal(saved,raw)),
               'all_above_first_retained_pixels_zero':bool(np.all(raw[~independent_av]==0)),
               'zero_pixels_within_retained_region':int(np.count_nonzero(raw[independent_av]==0)),
               'availability_exactly_matches_first_retained_row':bool(np.array_equal(avail,independent_av)),
               'primary_mask_exactly_equals_availability':bool(np.array_equal(valid,avail)),
               'primary_mask_formula_matches':bool(np.array_equal(valid,independent_av&(np.arange(2048)[:,None]>=expected_geom+10))),
               'first_retained_equals_round_supplied_trace':bool(np.array_equal(first,np.round(trace))),
               'all_columns_retained_through_last_image_row':bool(np.all(last==2047)),
               'inferred_geometric_surface_is_exactly_supplied_trace_minus_12':bool(np.array_equal(z['surface_'+f],expected_geom)),
               'same_exporter_scaled_trace_relation':bool(np.array_equal(m['imSurf'+f+'/surfaceSurfImgScaled'][:].ravel()[654:2702]-1429,trace)),
               'first_retained_row_range':[int(first.min()),int(first.max())],
               'available_pixels':int(avail.sum()),
               'normalized_image_finite':bool(np.isfinite(z[f.upper()]).all())}
            report['frames'][f.upper()]=r
            report['failures'] += [f'{f}:{k}' for k,v in r.items() if isinstance(v,bool) and not v]
        cx=m['compVel/xPIV'][:].ravel()-1;cy=m['compVel/zPIV'][:].ravel()-1
        direct_ix=((q[:,0]-3)/4).astype(int);direct_iy=((q[:,1]-3)/4).astype(int)
        inside=(q[:,0]>=cx[0])&(q[:,0]<=cx[-1])&(q[:,1]>=cy[0])&(q[:,1]<=cy[-1])
        expected=np.full((len(q),2),np.nan);expected_cor=np.full(len(q),np.nan)
        # HDF5 dimensions here are [MATLAB x, MATLAB y]. Avoid the preparer's lookup helper.
        for j,(name,sign) in enumerate([('delta_x',1),('delta_z',-1)]):
            source=m['compVel/'+name][:]
            expected[inside,j]=sign*source[direct_ix[inside],direct_iy[inside]]
        expected_cor[inside]=m['compVel/dcor'][:][direct_ix[inside],direct_iy[inside]]
        nx,ny=np.meshgrid(cx.astype(int),cy.astype(int))
        native_vec=np.isfinite(z['classical_dx'])&np.isfinite(z['classical_dy'])
        native_valid=z['va'][ny,nx]
        native_dcor=np.isfinite(z['classical_dcor'])
        grid_expected=np.array([(x,y) for y in range(7,2048,8) for x in range(7,2048,8) if y>=sa[x]+12],float)
        report['coordinates']={
            'origin_is_zero':bool(np.array_equal(z['origin0'],[0,0])),
            'grid_exactly_all_8px_nodes_below_minimum_source_depth_no_depth_cap':bool(np.array_equal(q,grid_expected)),
            'all_grid_nodes_source_visible':bool(np.all(z['va'][q[:,1].astype(int),q[:,0].astype(int)])),
            'recorded_depth_exact':bool(np.array_equal(depth,q[:,1]-sa[q[:,0].astype(int)])),
            'node_count':len(q),'x_range':[float(q[:,0].min()),float(q[:,0].max())],
            'y_range':[float(q[:,1].min()),float(q[:,1].max())],
            'depth_range':[float(depth.min()),float(depth.max())],
            'native_axes_exact_3_mod_4':bool(np.array_equal(cx,np.arange(3,2044,4)) and np.array_equal(cy,np.arange(3,2044,4))),
            'native_vector_lookup_exact_including_NaNs':bool(np.all((expected==z['classical_at_points'])|(np.isnan(expected)&np.isnan(z['classical_at_points'])))),
            'native_dcor_lookup_exact_including_NaNs':bool(np.all((expected_cor==z['classical_dcor_at_points'])|(np.isnan(expected_cor)&np.isnan(z['classical_dcor_at_points'])))),
            'outside_native_domain_all_NaN':bool(np.isnan(z['classical_at_points'][~inside]).all()),
            'native_domain_grid_nodes':int(inside.sum()),'outside_native_domain_grid_nodes':int((~inside).sum()),
            'dense_dx_source_exact':bool(np.array_equal(z['supplied_dx'],m['compVel/delta_x1'][:].T)),
            'dense_dy_sign_and_transpose_exact':bool(np.array_equal(z['supplied_dy'],-m['compVel/delta_z1'][:].T)),
            'native_source_visibility_exact':bool(np.array_equal(native_valid,z['classical_native_source_visible'])),
            'native_finite_vectors':int(native_vec.sum()),
            'native_finite_vectors_source_unavailable':int((native_vec&~native_valid).sum()),
            'native_source_available_finite_vectors':int((native_vec&native_valid).sum()),
            'native_source_available_finite_vectors_with_finite_dcor':int((native_vec&native_valid&native_dcor).sum()),
            'native_finite_vectors_at_grid':int(np.isfinite(expected).all(axis=1).sum()),
            'native_finite_vectors_with_finite_dcor_at_grid':int((np.isfinite(expected).all(axis=1)&np.isfinite(expected_cor)).sum())}
        report['failures'] += ['coordinates:'+k for k,v in report['coordinates'].items() if isinstance(v,bool) and not v]
        # Fixed image-only check of displacement coordinate orientation; no targets or tuning.
        patch_xy=np.array([(x,y) for y in [400,440,600,1000,1500,1900] for x in [160,480,800,1120,1440,1760,1920]])
        offsets=np.arange(-8,9);ox,oy=np.meshgrid(offsets,offsets)
        smoothA=gaussian_filter(z['rawA'],.5);smoothB=gaussian_filter(z['rawB'],.5)
        va=z['va'];vb=z['vb'];dx=z['supplied_dx'];dy=z['supplied_dy']
        checks={}
        for label,transpose,vertical_sign in [('transposed_image_down',True,1),('transposed_wrong_y_sign',True,-1),('untransposed_image_down',False,1)]:
            scores=[]
            for x,y in patch_xy:
                disp=np.array([dx[y,x],dy[y,x]]) if transpose else np.array([dx[x,y],dy[x,y]])
                disp[1]*=vertical_sign
                x0=x+ox;y0=y+oy;x1=x0+disp[0];y1=y0+disp[1]
                mask=va[y0,x0]&(map_coordinates(vb.astype(float),[y1,x1],order=1,mode='constant',cval=0)>.99)
                if mask.sum()<200:continue
                aa=smoothA[y0,x0][mask];bb=map_coordinates(smoothB,[y1,x1],order=1,mode='constant',cval=0)[mask]
                scores.append(float(np.corrcoef(aa,bb)[0,1]))
            checks[label]={'patch_count':len(scores),'mean_NCC':float(np.mean(scores)),'median_NCC':float(np.median(scores))}
        report['image_only_orientation_check']=checks
        report['calibration']={'DX':float(z['DX']),'DT':float(z['DT']),'GS':float(z['GS']),'IW':float(z['IW']),
            'units_explicit_in_file':False,'metres_per_pixel_and_seconds_are_assumptions':True}
        report['surface_geometry_note']='The 12px subtraction is inherited from registered pair123; these two masked images independently verify retained-data boundaries but cannot independently establish the missing physical interface.'
        report['comparison_note']='Native finite vectors can occur above the retained-data boundary. Comparison must use native samples and its own source/target visibility; dense delta_x1/delta_z1 arrays are initializer fields only.'
        report['no_manual_reference_in_MAT']=set(k for k in m if k!='#refs#')=={'compVel','imSurfa','imSurfb'}
    report['passed']=len(report['failures'])==0
    (folder/'input_audit.json').write_text(json.dumps(report,indent=2))
    return report

if __name__=='__main__':
    results=[run(pair) for pair in [80,100]]
    (ROOT/'work/pairs/input_audit.json').write_text(json.dumps(results,indent=2))
    for r in results:print(json.dumps({'pair':r['pair'],'passed':r['passed'],'failures':r['failures'],'coordinates':r['coordinates'],'orientation':r['image_only_orientation_check']}),flush=True)
