"""Read-only metadata/coordinate audit of the enlarged observation."""
from pathlib import Path
import h5py,numpy as np,json
from PIL import Image
from scipy.ndimage import map_coordinates
O=Path('work/large_frame_audit');O.mkdir(exist_ok=True)
newpath=Path('data/ExpLCL_1_03_123_PIV.mat');oldpath=Path('data/ExpLCL_1_03-123.mat')
old=np.load('work/metadata.npz');manual=old['p'][:,0].T-1+[299,339];truth=old['p'][:,1].T-old['p'][:,0].T
assert np.allclose(manual,old['p_orig'][:,0].T-2)
res={'new_mat_path':str(newpath),'bytes':newpath.stat().st_size,'format':'MATLAB7.3/HDF5','top_level_variables':[],'manual_pairs_in_new_mat':False,'scalar_units_explicitly_present':False,'variable_inventory':{},'image_audit':{},'surface_relations':{},'coordinate_conventions':{},'supplied_computed_reference':{}}
small={};pred=[]
with h5py.File(newpath,'r') as f:
 res['top_level_variables']=list(f.keys())
 for group in f:
  if group.startswith('#'):continue
  for key,o in f[group].items():
   v=o[()];num=v[np.isfinite(v)]
   res['variable_inventory'][group+'/'+key]={'hdf_shape':list(v.shape),'dtype':str(v.dtype),'finite_count':int(len(num)),'total_count':int(v.size),'min':float(np.min(num)),'max':float(np.max(num)),'median':float(np.median(num)),'attrs':{k:str(vv) for k,vv in o.attrs.items()}}
 scalar={k:float(f['compVel/'+k][()][0,0]) for k in ['DX','DT','GS','IW']};res['scalars']=scalar
 for a in ['a','b']:
  surface=f['imSurf'+a+'/surfacePIVImg'][()].ravel();sc=f['imSurf'+a+'/surfaceSurfImgScaled'][()].ravel();raw=f['imSurf'+a+'/surface_raw'][()].ravel();saved=old['surf'+a+'_orig'].ravel();crop=old['surf'+a].ravel();geom=saved-2
  assert np.array_equal(surface,saved+10);assert np.array_equal(crop,saved[299:800]-340);assert np.array_equal(sc[654:2702]-1429,surface)
  small['geometric_surface_'+a+'_full0']=geom;small['provided_surfacePIVImg_'+a+'_stored']=surface;small['provided_mask_first_row_'+a]=np.rint(surface).astype(int)
  res['surface_relations'][a]={'new_surface_minus_old_full_saved':10,'old_crop_equals_old_full_slice_299_800_minus':340,'original_pipeline_global_geometric_surface':'old_full_saved−2 = new_surfacePIVImg−12','scaled_surface_slice0':[654,2702],'scaled_surface_vertical_subtraction':1429,'depression_apex_global0':[int(np.argmax(geom)),float(np.max(geom))],'geometric_surface_range0':[float(np.min(geom)),float(np.max(geom))],'difference_using_old_full_minus1_instead':1}
  im=np.array(Image.open('data/ExpLCL_1_03_123_img'+a.upper()+'.tif'));imold=np.array(Image.open('data/ExpLCL_1_03-123_img'+a.upper()+'.tif'));imcrop=im[339:840,299:800];first=np.argmax(im>0,axis=0);diff=imold!=imcrop;yy,_=np.indices(imold.shape)
  assert np.array_equal(first,np.rint(surface).astype(int));assert np.sum(diff&(imcrop!=0))==0
  res['image_audit'][a]={'shape_yx':list(im.shape),'dtype':str(im.dtype),'already_zero_masked':True,'first_nonzero_row_equals_rounded_stored_PIV_surface_all_columns':True,'original_crop_origin0_xy':[299,339],'crop_difference_count':int(diff.sum()),'differences_new_zero_count':int(np.sum(diff&(imcrop==0))),'differences_new_nonzero_count':int(np.sum(diff&(imcrop!=0))),'difference_ycrop_range':[int(np.min(yy[diff])),int(np.max(yy[diff]))],'unchanged_everywhere_after_crop_row100':bool(np.array_equal(imold[100:],imcrop[100:])),'first_retained_relative_original_geometric_surface':'round(geometric_surface)+12','original_altmask_first_row_relative_same_geometry':'round(geometric_surface)+10'}
 for axis in ['x','z']:
  fine=f['compVel/delta_'+axis+'1'][()].T
  vals=map_coordinates(fine,[manual[:,1],manual[:,0]],order=1);pred.append(vals if axis=='x' else -vals)
 small['xPIV_stored']=f['compVel/xPIV'][()].ravel();small['zPIV_stored']=f['compVel/zPIV'][()].ravel()
 DX=scalar['DX'];DT=scalar['DT'];res['SI_assumption_conversions']={'assumption':'DX ismetres/pixel andDTisseconds; noexplicitunitsattribute','metres_per_pixel':DX,'seconds_per_pair':DT,'mm_per_pixel':DX*1000,'mps_per_pixel_per_pair':DX/DT,'inverse_seconds_per_displacement_gradient':1/DT,'pixels_per_cm':.01/DX,'pixels_per_2cm':.02/DX,'old330px_report_width_cm':330*DX*100,'old150px_report_depth_cm':150*DX*100,'full2048pixel_width_cm':2048*DX*100}
res['coordinate_conventions']={'hdf2D_arrays':'transpose HDF5arrays to get MATLAB[y,x]imageaxes','xPIV_zPIV_stored':'4,8,...,2044pixels; no physical origin provided','xPIV_zPIV_zero_based_conventional':'stored−1, if originalMATLABvalues denote pixelcenters; no sourcecode proveshalf-pixel placement','original_crop_to_full0':'(xcrop0,ycrop0)+(299,339)','manual_full0':'p_orig−2 = (p−1)+(299,339)','surface_full0':'old_surfa_orig−2,old_surfb_orig−2 preserves originalpipeline andregisteredcrop','positive_horizontal':'right','positive_image_vertical':'down','supplied_delta_z':'evidence supports upward-positive: useimage_dy=−delta_z1','physical_origin':'notprovided; chooseexplicitimageorwave-referencedorigin'}
pred=np.array(pred).T;err=np.linalg.norm(pred-truth,axis=1);dep=old['p'][:,0].T[:,1]-1-np.interp(old['p'][:,0].T[:,0]-1,np.arange(501),old['surfa'].ravel()-1)
for lab,sel in [('all200',np.ones(200,bool)),('near40',dep<40),('deep80plus',dep>=80)]:
 v=err[sel];res['supplied_computed_reference'][lab]={'n':int(sel.sum()),'mean_disagreement_px':float(np.mean(v)),'median_disagreement_px':float(np.median(v)),'p90_disagreement_px':float(np.percentile(v,90))}
res['supplied_computed_reference']['interpretation']='Computedreference,notmanualtruth. Densefields finite even inmasked/airregions; origin/unit/signinferencesstateexplicitly.'
small.update(manual_source_full0=manual,manual_target_full0=manual+truth,manual_displacement_image_px=truth,supplied_PIV_dense_at_manual_image_px=pred,DX=DX,DT=DT,GS=scalar['GS'],IW=scalar['IW'])
np.savez_compressed(O/'coordinate_metadata.npz',**small)
(O/'mat_audit.json').write_text(json.dumps(res,indent=2));print(json.dumps({k:v for k,v in res.items() if k!='variable_inventory'},indent=2))
