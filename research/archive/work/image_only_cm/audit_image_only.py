"""Independent provenance, image, domain and integration-bookkeeping audit.

Reads only the fresh image_only_cm pair directory, original TIFF pixels and
four explicitly allowed scalar/geometric MAT datasets. No supplied computed
velocity dataset or earlier prediction/track artifact is opened.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import h5py
from PIL import Image
from scipy.ndimage import map_coordinates

HERE=Path(__file__).resolve().parent
BASE=Path('historical-user-files/Downloads')
ALLOW=['compVel/DX','compVel/DT','imSurfa/surfacePIVImg','imSurfb/surfacePIVImg']

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def read(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def read_inputs(path):
    # Inspect names before loading any payload; even an accidentally substituted
    # archive cannot cause this audit to read forbidden velocity arrays.
    with np.load(path,allow_pickle=False) as z:
        bad=[k for k in z.files if k!='supplied_velocity_used' and
             (k.startswith(('classical','supplied_','manual','previous_','old_','prior_','tracks')) or
              k in ['disp','displacement','delta_x','delta_z','delta_x1','delta_z1','velocity','flow'])]
        if bad:raise ValueError('Refusing forbidden input payload keys: '+', '.join(bad))
        return {k:z[k] for k in z.files}

def same(a,b,tol=1e-12):
    a=np.asarray(a);b=np.asarray(b)
    if a.shape!=b.shape:return False
    if a.dtype.kind=='b' and b.dtype.kind=='b':return bool(np.array_equal(a,b))
    finite=np.isfinite(a)&np.isfinite(b)
    return bool(np.array_equal(np.isfinite(a),np.isfinite(b)) and np.array_equal(np.isnan(a),np.isnan(b)) and np.all(np.abs(a[finite]-b[finite])<=tol))

def visible(mask,q,strict=False):
    good=np.isfinite(q).all(axis=1)&(q>=0).all(axis=1)&(q<=np.array(mask.shape[::-1])-1).all(axis=1)
    if strict:good&=(q>=1).all(axis=1)&(q<np.array(mask.shape[::-1])-2).all(axis=1)
    out=np.zeros(len(q),bool)
    out[good]=map_coordinates(mask.astype(float),q[good,::-1].T,order=1,mode='constant',cval=0)>.99
    return out

def profile_audit(directory,inp):
    z=read(directory/'integration_profile.npz');DX=float(inp['DX']);DT=float(inp['DT'])
    x=z['horizontal_sample_x_px'];h=z['depth_m'];u=z['sample_primary_u_assumed_m_per_s'];ok=z['sample_accepted'];v=z['sample_variant_u_assumed_m_per_s']
    edges=x[::2];length=np.diff(edges)*DX;nd=len(h);ni=len(length)
    assert u.shape==ok.shape==(nd,len(x)) and v.shape==(3,nd,len(x))
    sy=np.interp(x,np.arange(len(inp['surface_a'])),inp['surface_a'])
    qy=sy[None]+h[:,None]/DX
    qx=np.broadcast_to(x,qy.shape);q=np.stack([qx,qy],axis=2).reshape(-1,2)
    d=z['sample_displacement_px'];target=q+d.reshape(-1,2)
    allowed=visible(inp['va'],q)&visible(inp['vb'],target,True)&(np.repeat(h/DX,len(x))>=12)&(np.repeat(h/DX,len(x))<=.01/DX+1e-9)
    valid=np.zeros((nd,ni),bool);parts=np.full((nd,ni),np.nan)
    covered=np.full(nd,np.nan);width=np.zeros(nd);mean=np.full(nd,np.nan);full=np.full(nd,np.nan)
    for row in range(nd):
        for col in range(ni):
            ids=[2*col,2*col+1,2*col+2]
            if np.all(ok[row,ids]) and np.isfinite(u[row,ids]).all():
                valid[row,col]=True;parts[row,col]=length[col]*(u[row,ids[0]]+4*u[row,ids[1]]+u[row,ids[2]])/6
        width[row]=sum(length[valid[row]])
        if width[row]>0:covered[row]=sum(parts[row,valid[row]]);mean[row]=covered[row]/width[row]
        if valid[row].all():full[row]=covered[row]
    band=(h>=20*DX-1e-15)&(h<=.01+1e-15)
    common=valid[band].all(axis=0);cw=float(sum(length[common]));cq=np.full(nd,np.nan)
    for row in range(nd):
        if cw>0 and valid[row,common].all():cq[row]=sum(parts[row,common])
    vq=np.full((3,nd),np.nan);vcq=vq.copy()
    for variant in range(3):
        for row in range(nd):
            for use,out in [(valid[row],vq),(common,vcq)]:
                if not use.any() or not valid[row,use].all():continue
                total=0.;available=True
                for col in np.flatnonzero(use):
                    a=v[variant,row,2*col:2*col+3]
                    if not np.isfinite(a).all():available=False;break
                    total+=length[col]*(a[0]+4*a[1]+a[2])/6
                if available:out[variant,row]=total
    checks=dict(local_surface_query_formula=same(qy,z['sample_query_y_px']),
        horizontal_u_unit_formula=same(d[:,:,0]*DX/DT,u),
        all_accepted_samples_observed_and_in_requested_depth=bool(np.all(allowed[ok.ravel()])),
        requested_depth_zero_to_one_cm=bool(h[0]==0 and h[-1]==.01),
        full_pixel_edge_width=bool(x[0]==-.5 and x[-1]==inp['va'].shape[1]-.5 and same(z['target_full_width_m'],inp['va'].shape[1]*DX)),
        interval_accepted_exact=same(valid,z['interval_accepted']),
        horizontal_Simpson_contributions_no_arclength=same(parts,z['interval_primary_integral_assumed_m2_per_s']),
        covered_integral_no_gap_fill=same(covered,z['covered_segment_integral_assumed_m2_per_s']),
        covered_width_exact=same(width,z['covered_width_m']),
        coverage_fraction_exact=same(width/(inp['va'].shape[1]*DX),z['coverage_fraction']),
        covered_mean_exact=same(mean,z['covered_width_mean_u_assumed_m_per_s']),
        full_width_missing_if_any_gap=same(full,z['full_width_integral_assumed_m2_per_s']),
        common_band_declared=same(z['common_depth_band_requested_m'],[20*DX,.01]),
        common_interval_mask_exact=same(common,z['common_domain_interval_mask']),
        common_width_constant=same(cw,z['common_domain_width_m']),
        common_integral_exact=same(cq,z['common_domain_integral_assumed_m2_per_s']),
        all_requested_depth_intersection_exact=same(valid.all(axis=0),z['all_requested_depth_common_interval_mask']),
        same_primary_domain_for_all_variants=same(vq,z['variant_covered_integral_assumed_m2_per_s']),
        same_common_domain_for_all_variants=same(vcq,z['variant_common_integral_assumed_m2_per_s']),
        no_accepted_geometric_surface_samples=bool(not ok[0].any()))
    for label,primary,variants in [('covered',covered,vq),('common',cq,vcq)]:
        vv=np.r_[primary[None],variants];finite=np.isfinite(vv).all(axis=0)
        lo=np.full(nd,np.nan);hi=lo.copy();lo[finite]=vv[:,finite].min(axis=0);hi[finite]=vv[:,finite].max(axis=0)
        checks[label+'_sensitivity_min_exact']=same(lo,z[label+'_model_sensitivity_min_assumed_m2_per_s'])
        checks[label+'_sensitivity_max_exact']=same(hi,z[label+'_model_sensitivity_max_assumed_m2_per_s'])
    return dict(passed=all(checks.values()),checks=checks,depth_count=nd,interval_count=ni,
        common_width_m=cw,full_width_available_depth_count=int(np.isfinite(full).sum()),
        note='Independent loop quadrature and support bookkeeping; no velocity field was refitted or supplied PIV read.')

def audit(pair,inputs_only=False):
    directory=HERE/str(pair);inp=read_inputs(directory/'inputs.npz')
    manifest=json.loads((directory/'input_manifest.json').read_text());checks={}
    with h5py.File(BASE/('ExpLCL_1_03_%d_PIV.mat'%pair),'r') as f:
        metadata={key:f[key][...] for key in ALLOW}
    hashes={key:hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest() for key,value in metadata.items()}
    input_hash=sha(directory/'inputs.npz')
    checks['preparer_dataset_allowlist_exact']=manifest['actual_mat_dataset_reads']==ALLOW and manifest['exact_mat_dataset_allowlist']==ALLOW
    checks['metadata_hashes_match_original_allowed_datasets']=hashes==manifest['mat_dataset_sha256']
    checks['input_hash_matches_manifest']=input_hash==manifest['inputs_sha256']
    forbidden=[k for k in inp if k!='supplied_velocity_used' and (k.startswith(('classical','supplied_','manual','previous_','old_','prior_','tracks')) or k in ['disp','displacement','delta_x','delta_z','delta_x1','delta_z1','velocity','flow'])]
    checks['no_velocity_payload_keys_in_inputs']=not forbidden
    checks['image_only_provenance']=bool(inp['image_only']) and not bool(inp['supplied_velocity_used'])
    DX=float(inp['DX']);DT=float(inp['DT'])
    checks['calibration_from_allowed_scalars']=DX==float(metadata['compVel/DX'].ravel()[0]) and DT==float(metadata['compVel/DT'].ravel()[0])
    for fr in 'ab':
        raw=np.asarray(Image.open(BASE/('ExpLCL_1_03_%d_img%s.tif'%(pair,fr.upper()))))
        trace=metadata['imSurf'+fr+'/surfacePIVImg'].ravel();first=np.argmax(raw>0,axis=0)
        availability=np.arange(raw.shape[0])[:,None]>=first
        valid=availability&(np.arange(raw.shape[0])[:,None]>=trace-12+10)
        checks[fr+'_raw_TIFF_exact']=same(raw,inp['raw'+fr.upper()],0)
        checks[fr+'_trace_offset_exact']=same(trace-12,inp['surface_'+fr],0)
        checks[fr+'_availability_exact']=same(availability,inp['availability_'+fr])
        checks[fr+'_valid_mask_exact']=same(valid,inp['v'+fr])
        checks[fr+'_zero_only_above_retained_region']=bool(np.all(raw[~availability]==0) and np.all(raw[availability]>0))
        checks[fr+'_retained_boundary_equals_rounded_trace']=same(first,np.round(trace),0)
    q=inp['points'];depth=q[:,1]-inp['surface_a'][q[:,0].astype(int)]
    xx,yy=np.meshgrid(np.arange(7,2048,8.),np.arange(7,2048,8.));candidate=np.c_[xx.ravel(),yy.ravel()]
    dep=candidate[:,1]-inp['surface_a'][candidate[:,0].astype(int)]
    expected=candidate[(dep>=12)&(dep<=.01/DX+64)]
    checks['fitting_grid_exact_including_halo']=same(q,expected,0)
    checks['requested_depth_exact']=float(inp['requested_max_depth_px'])==.01/DX
    checks['fitting_and_detection_halos_exact']=float(inp['fitting_max_depth_px'])==.01/DX+64 and float(inp['detector_max_depth_px'])==.01/DX+89
    checks['geometry_and_units_explicitly_inferred']=bool(inp['surface_geometry_inferred']) and not bool(inp['physical_units_confirmed'])
    result_checks={};profile=None
    if not inputs_only:
        z=read(directory/'results.npz');rq=z['query'];rd=rq[:,1]-np.interp(rq[:,0],np.arange(len(inp['surface_a'])),inp['surface_a'])
        result_checks['regular_query_exact_requested_subset']=same(rq,q[depth<=.01/DX+1e-9],0)
        result_checks['result_depth_formula_exact']=same(rd,z['depth'])
        result_checks['accepted_domain_within_top_centimetre']=bool(np.all((rd[z['accepted']]>=12)&(rd[z['accepted']]<=.01/DX+1e-9)))
        result_checks['source_visibility_exact']=same(visible(inp['va'],rq),z['source_visible'])
        result_checks['target_visibility_exact']=same(visible(inp['vb'],rq+z['disp'],True),z['target_visible'])
        result_checks['mask_flags_applied']=same(z['accepted'],z['evidence_pass']&z['source_visible']&z['target_visible'])
        result_checks['result_image_only_provenance']=bool(z['image_only']) and not bool(z['supplied_velocity_used'])
        track_hash=sha(directory/'ptv_tracks.npz')
        result_checks['result_matches_current_input_and_tracks']=str(z['input_sha256'].item())==input_hash and str(z['tracks_sha256'].item())==track_hash
        for name in ['main','affine','large_window','margin14','reverse']:
            with np.load(directory/(name+'.npz'),allow_pickle=False) as f:
                result_checks[name+'_complete_and_fresh']=bool(f['complete']) and bool(f['image_only']) and not bool(f['supplied_velocity_used']) and str(f['input_sha256'].item())==input_hash and str(f['tracks_sha256'].item())==track_hash
        profile=profile_audit(directory,inp)
    report=dict(pair=pair,passed=all(checks.values()) and all(result_checks.values()) and (profile is None or profile['passed']),
        inputs_only=inputs_only,input_checks=checks,result_checks=result_checks,integration_audit=profile,
        original_mat_datasets_read=ALLOW,supplied_computed_velocity_datasets_read=[],previous_results_or_tracks_read=[],
        input_sha256=input_hash,script_sha256=sha(Path(__file__)),grid_nodes=len(q),requested_grid_nodes=int(np.sum(depth<=.01/DX)),
        note='Static dependency review plus direct fresh-payload/source verification; only scalar calibration and geometric traces are read from original MAT files.')
    path=directory/('independent_inputs_audit.json' if inputs_only else 'independent_image_only_audit.json')
    path.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
    if not report['passed']:raise SystemExit(1)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pair',type=int,choices=[80,100],required=True);p.add_argument('--inputs-only',action='store_true');a=p.parse_args();audit(a.pair,a.inputs_only)
