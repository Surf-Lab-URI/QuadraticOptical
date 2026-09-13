"""Independent read-only verification of full-width image/coordinate preparation."""
from pathlib import Path
import hashlib,json
import numpy as np
from PIL import Image
import h5py
from scipy.spatial import cKDTree

O=Path(__file__).resolve().parent;W=O.parent
i=np.load(O/'inputs.npz');prev=np.load(W/'large_frame/inputs.npz');meta=np.load(W/'metadata.npz')
origin=i['origin0'];h,w=i['A'].shape
assert np.array_equal(origin,[0,260]) and (h,w)==(580,2048)
yy,xx=np.indices((h,w));Y=yy+origin[1];X=xx+origin[0]
restored=(X>=299)&(X<800)&(Y>=339)&(Y<840)
out={'input_sha256':hashlib.sha256((O/'inputs.npz').read_bytes()).hexdigest(),'frames':{}}
with h5py.File('data/ExpLCL_1_03_123_PIV.mat','r') as f:
    for frame in 'AB':
        s=frame.lower()
        tif=np.asarray(Image.open('data/ExpLCL_1_03_123_img'+frame+'.tif'))
        original=np.asarray(Image.open('data/ExpLCL_1_03-123_img'+frame+'.tif'))
        surface=meta['surf'+s+'_orig'].ravel()-2
        mask=f['imSurf'+s+'/surfacePIVImg'][:].ravel()
        first=(tif>0).argmax(axis=0)
        av=(Y>=first[None,:])|restored
        valid=av&(Y>=surface[None,:]+10)
        raw=i['raw'+frame];prevroi=raw[40:580,240:860]
        tests={
            'full_surface_equals_original_metadata_minus2':bool(np.array_equal(i['full_surface_'+s],surface)),
            'crop_surface_equals_full_minus_origin':bool(np.array_equal(i['surface_'+s],surface-origin[1])),
            'full_mask_equals_trace_plus12':bool(np.allclose(mask,surface+12,rtol=0,atol=1e-10)),
            'full_mask_equals_input':bool(np.array_equal(i['new_mask_surface_'+s],mask)),
            'first_observed_row_matches_rounded_mask_all_columns':bool(np.array_equal(first,np.round(mask))),
            'outside_original_crop_raw_exact_TIFF':bool(np.array_equal(raw[~restored],tif[260:840,:][~restored])),
            'restored_original_crop_exact_original_TIFF':bool(np.array_equal(raw[79:580,299:800],original)),
            'availability_exact':bool(np.array_equal(i['availability_'+s],av)),
            'fit_validity_exact':bool(np.array_equal(i['v'+s],valid)),
            'unavailable_pixels_zero':bool(np.all(raw[~av]==0)),
            'no_valid_pixels_are_unavailable':bool(np.all(~i['v'+s]|i['availability_'+s])),
            'previous_raw_exact':bool(np.array_equal(prevroi,prev['raw'+frame])),
            'previous_validity_exact':bool(np.array_equal(i['v'+s][40:580,240:860],prev['v'+s])),
            'normalized_image_finite':bool(np.isfinite(i[frame]).all()),
        }
        delta_key='delta_x1' if frame=='A' else 'delta_z1'
        expected=f['compVel/'+delta_key][:].T[260:840,:]*(1 if frame=='A' else -1)
        tests['supplied_PIV_component_mapping_exact']=bool(np.array_equal(i['supplied_dx' if frame=='A' else 'supplied_dy'],expected))
        out['frames'][frame]={'tests':tests,'passed':all(tests.values()),
             'valid_pixels':int(valid.sum()),'unavailable_pixels':int((~av).sum()),
             'valid_pixels_per_column_minmax':[int(valid.sum(axis=0).min()),int(valid.sum(axis=0).max())]}
    out['calibration']={'DX':float(i['DX']),'DT':float(i['DT']),
        'unchanged_from_previous':bool(float(i['DX'])==float(prev['DX']) and float(i['DT'])==float(prev['DT'])),
        'matches_supplied_MAT':bool(float(i['DX'])==float(f['compVel/DX'][0,0]) and float(i['DT'])==float(f['compVel/DT'][0,0])),
        'physical_units_confirmed':False,'full_width_cm_if_SI':2048*float(i['DX'])*100,
        'maximum_depth_cm_if_SI':float(i['max_depth'])*float(i['DX'])*100}
q=i['points']+origin;dep=q[:,1]-np.interp(q[:,0],np.arange(2048),i['full_surface_a'])
dist,j=cKDTree(q).query(prev['points']+prev['origin0'])
manual=i['manual_xy']+origin;oldmanual=meta['p'][:,0].T-1+[299,339]
ct={
    'all_grid_x_columns_present':bool(np.array_equal(np.unique(q[:,0]),np.arange(7,2048,8.))),
    'points_inside_TIFF':bool(np.all((q>=0)&(q<2048))),
    'depth_extent_unchanged':bool(np.all((dep>=12)&(dep<=354)) and float(i['max_depth'])==354),
    'all_previous_nodes_present':bool(np.max(dist)==0),
    'previous_points_transform_exact':bool(np.array_equal(i['previous_points']+origin,prev['points']+prev['origin0'])),
    'all_200_manual_sources_exact':bool(np.array_equal(manual,oldmanual)),
    'all_200_manual_displacements_exact':bool(np.array_equal(i['manual_truth'],meta['p'][:,1].T-meta['p'][:,0].T)),
}
out['coordinates']={'tests':ct,'grid_count':len(q),'horizontal_columns':len(np.unique(q[:,0])),
    'grid_x_minmax':q[:,0][[np.argmin(q[:,0]),np.argmax(q[:,0])]].tolist(),
    'grid_depth_minmax':[float(dep.min()),float(dep.max())],
    'maximum_previous_node_mapping_error':float(dist.max()),
    'previous_node_count':len(j),'previous_nodes_by_index':j.tolist(),
    'image_convention':'Zero-based x right, y down; displacement is B minus A.',
    'physical_convention':'If SI calibration is assumed: U=dx*DX/DT, Wup=-dy*DX/DT; dU/dX=G00/DT.'}
out['passed']=all(t['passed'] for t in out['frames'].values()) and all(ct.values()) and out['calibration']['unchanged_from_previous'] and out['calibration']['matches_supplied_MAT']
(O/'independent_input_audit.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='coordinates'},indent=2));print({k:v for k,v in out['coordinates'].items() if k!='previous_nodes_by_index'})
