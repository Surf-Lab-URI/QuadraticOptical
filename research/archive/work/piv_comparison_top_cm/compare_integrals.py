"""Held-out native PIV/image-only integrals on exactly matched domains.

Reads completed image-only profiles and native PIV solely for comparison.
No fitting, smoothing, missing-node filling or image-only artifact modification.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import argparse,csv,json,time
import numpy as np
from native_piv import NativePIV,sha

HERE=Path(__file__).resolve().parent
IMAGE_ONLY=HERE.parent/'image_only_cm'

def read(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def equal(a,b,tol=1e-12):
    a=np.asarray(a);b=np.asarray(b)
    if a.shape!=b.shape:return False
    if a.dtype.kind=='b' and b.dtype.kind=='b':return bool(np.array_equal(a,b))
    f=np.isfinite(a)&np.isfinite(b)
    return bool(np.array_equal(np.isfinite(a),np.isfinite(b)) and np.array_equal(np.isnan(a),np.isnan(b)) and np.all(np.abs(a[f]-b[f])<=tol))

def intervals(nodes):
    return nodes[:,:-2:2]&nodes[:,1::2]&nodes[:,2::2]

def simpson_values(u,width):
    return width[None]*(u[:,:-2:2]+4*u[:,1::2]+u[:,2::2])/6

def supported_sum(values,mask):
    total=np.sum(np.where(mask,values,0.),axis=1)
    total[~mask.any(axis=1)]=np.nan
    return total

def mean_on_width(total,width):
    out=np.full(total.shape,np.nan);good=width>0
    out[good]=total[good]/width[good]
    return out

def gaps(mask,width):
    components=np.sum(mask&~np.c_[np.zeros(len(mask),bool),mask[:,:-1]],axis=1)
    longest=np.zeros(len(mask))
    for j,row in enumerate(mask):
        current=0.
        for available,span in zip(row,width):
            current=0. if available else current+span
            longest[j]=max(longest[j],current)
    return components,longest

def domain_stats(u,mask,width,full_width):
    total=supported_sum(simpson_values(u,width),mask)
    covered=np.sum(width[None]*mask,axis=1)
    ncomp,longest=gaps(mask,width)
    return dict(integral_m2_per_s=total,width_m=covered,coverage_fraction=covered/full_width,
                mean_u_m_per_s=mean_on_width(total,covered),supported_component_count=ncomp,
                longest_gap_m=longest)

def pair_stats(of_u,piv_u,mask,width,full_width):
    of=domain_stats(of_u,mask,width,full_width);piv=domain_stats(piv_u,mask,width,full_width)
    difference=of_u-piv_u
    mae=mean_on_width(supported_sum(simpson_values(np.abs(difference),width),mask),of['width_m'])
    mse=mean_on_width(supported_sum(simpson_values(difference*difference,width),mask),of['width_m'])
    used=np.zeros(of_u.shape,bool)
    used[:,:-2:2]|=mask;used[:,1::2]|=mask;used[:,2::2]|=mask
    maximum=np.full(len(mask),np.nan)
    for j in range(len(mask)):
        if used[j].any():maximum[j]=np.max(np.abs(difference[j,used[j]]))
    return dict(of=of,piv=piv,integral_difference_of_minus_piv_m2_per_s=of['integral_m2_per_s']-piv['integral_m2_per_s'],
        mean_difference_m_per_s=of['mean_u_m_per_s']-piv['mean_u_m_per_s'],
        mean_absolute_difference_m_per_s=mae,rms_difference_m_per_s=np.sqrt(mse),
        max_absolute_difference_m_per_s=maximum,contributing_sample_count=used.sum(axis=1))

def put_pair(out,name,stats):
    for side in ['of','piv']:
        for key in ['integral_m2_per_s','mean_u_m_per_s']:out[side+'_'+name+'_'+key]=stats[side][key]
    for key in ['width_m','coverage_fraction','supported_component_count','longest_gap_m']:
        out[name+'_'+key]=stats['of'][key]
    for key,value in stats.items():
        if key not in ['of','piv']:out[name+'_'+key]=value

def finite_summary(a):
    a=np.asarray(a);v=a[np.isfinite(a)]
    return dict(n=int(v.size),minimum=float(v.min()),median=float(np.median(v)),maximum=float(v.max())) if v.size else dict(n=0,minimum=None,median=None,maximum=None)

def compare(pair):
    started=time.time();directory=IMAGE_ONLY/str(pair)
    immutable=[directory/(name+'.npz') for name in ['inputs','results','main','affine','large_window','margin14','reverse','ptv_tracks','integration_profile']]
    immutable += [directory/'integration_metadata.json']
    before={str(path):sha(path) for path in immutable}
    profile=read(directory/'integration_profile.npz')
    native=NativePIV(pair)
    DX=native.DX;DT=native.DT
    if float(profile['DX'])!=DX or float(profile['DT'])!=DT:raise ValueError('Calibration mismatch.')
    h=profile['depth_m'];x=profile['horizontal_sample_x_px'];y=profile['sample_query_y_px']
    of_u=profile['sample_primary_u_assumed_m_per_s'];of_nodes=profile['sample_accepted']&np.isfinite(of_u)
    of_mask=intervals(of_nodes);width=profile['horizontal_interval_width_m'];L=float(profile['target_full_width_m'])
    if not equal(of_mask,profile['interval_accepted']):raise ValueError('Frozen image-only interval mask does not match its samples.')
    query=np.stack([np.broadcast_to(x,y.shape),y],axis=2).reshape(-1,2)
    sample=native.sample(query);piv_d=sample['disp'].reshape(len(h),len(x),2)
    piv_u=piv_d[:,:,0]*DX/DT;piv_nodes=sample['available'].reshape(of_u.shape)&np.isfinite(piv_u)
    piv_mask=intervals(piv_nodes)
    piv_dcor_nodes=sample['available_with_dcor'].reshape(of_u.shape)
    piv_target_nodes=sample['available_with_target'].reshape(of_u.shape)
    matched=of_mask&piv_mask
    paired=pair_stats(of_u,piv_u,matched,width,L)
    band_bounds=profile['common_depth_band_requested_m']
    band=(h>=band_bounds[0]-1e-15)&(h<=band_bounds[1]+1e-15)
    if not band.any():raise ValueError('Declared common depth band is empty.')
    common=matched[band].all(axis=0);common_width=float(width[common].sum())
    common_available=(common_width>0)&matched[:,common].all(axis=1)
    fixed_mask=np.broadcast_to(common,matched.shape)&common_available[:,None]
    fixed=pair_stats(of_u,piv_u,fixed_mask,width,L)

    old_common=profile['common_domain_interval_mask'];old_width=float(profile['common_domain_width_m'])
    of_old_available=profile['common_domain_available_at_depth']
    piv_on_old_available=(old_width>0)&of_old_available&piv_mask[:,old_common].all(axis=1)
    old_mask=np.broadcast_to(old_common,matched.shape)&piv_on_old_available[:,None]
    on_old=domain_stats(piv_u,old_mask,width,L)
    piv_old_fraction=(width[None]*piv_mask*old_common[None]).sum(axis=1)/old_width if old_width>0 else np.full(len(h),np.nan)
    own=domain_stats(piv_u,piv_mask,width,L)
    original=domain_stats(of_u,of_mask,width,L)
    if not equal(original['integral_m2_per_s'],profile['covered_segment_integral_assumed_m2_per_s']):raise ValueError('Frozen OF integral changed under identical quadrature.')
    if not equal(original['width_m'],profile['covered_width_m']):raise ValueError('Frozen OF width changed.')

    out=dict(pair=np.array(pair),depth_m=h,horizontal_sample_x_px=x,sample_query_y_px=y,
        horizontal_interval_bounds_px=profile['horizontal_interval_bounds_px'],horizontal_interval_width_m=width,
        target_full_width_m=np.array(L),DX=np.array(DX),DT=np.array(DT),physical_units_confirmed=np.array(False),
        of_sample_u_m_per_s=of_u,of_sample_accepted=profile['sample_accepted'],of_original_interval_mask=of_mask,
        piv_sample_displacement_px=piv_d,piv_sample_u_m_per_s=piv_u,piv_sample_available=piv_nodes,
        piv_sample_dcor=sample['dcor'].reshape(of_u.shape),piv_sample_all_contributors_finite_dcor=piv_dcor_nodes,
        piv_sample_available_with_target=piv_target_nodes,piv_own_interval_mask=piv_mask,matched_interval_mask=matched,
        common_interval_mask=common,common_depth_band_m=np.array([h[band].min(),h[band].max()]),
        common_requested_depth_band_m=band_bounds,common_available_at_depth=common_available,
        common_observed_fraction_at_depth=((width[None]*matched*common[None]).sum(axis=1)/common_width if common_width>0 else np.full(len(h),np.nan)),
        all_requested_depth_joint_common_width_m=np.array(width[matched.all(axis=0)].sum()),
        of_original_common_interval_mask=old_common,of_original_common_width_m=np.array(old_width),
        of_original_common_coverage_fraction=np.array(old_width/L),
        of_original_common_integral_m2_per_s=profile['common_domain_integral_assumed_m2_per_s'],
        of_original_common_mean_u_m_per_s=profile['common_domain_mean_u_assumed_m_per_s'],
        of_original_common_available=of_old_available,piv_on_of_original_common_available=piv_on_old_available,
        piv_on_of_original_common_integral_m2_per_s=on_old['integral_m2_per_s'],
        piv_on_of_original_common_mean_u_m_per_s=on_old['mean_u_m_per_s'],piv_fraction_of_of_original_common_width=piv_old_fraction,
        piv_on_of_original_common_difference_of_minus_piv_m2_per_s=np.where(piv_on_old_available,profile['common_domain_integral_assumed_m2_per_s']-on_old['integral_m2_per_s'],np.nan))
    put_pair(out,'matched',paired);put_pair(out,'common',fixed)
    # Common-domain width is a scalar constant, even at unsupported shallower
    # depths. Keep evaluated width separate to avoid ambiguous missing-data plots.
    out['common_evaluated_width_m']=out.pop('common_width_m')
    out['common_evaluated_coverage_fraction']=out.pop('common_coverage_fraction')
    out['common_width_m']=np.array(common_width);out['common_coverage_fraction']=np.array(common_width/L)
    for key,value in own.items():out['piv_own_'+key]=value
    for key,value in original.items():out['of_original_'+key]=value
    # Exact plot aliases preserve the earlier agreed schema.
    out['of_original_covered_integral_m2_per_s']=original['integral_m2_per_s']
    out['of_original_covered_mean_u_m_per_s']=original['mean_u_m_per_s']

    # Availability-only sensitivity; no arbitrary correlation threshold and no
    # changes to primary curves. Both methods still use identical intervals.
    for label,node in [('finite_dcor',piv_dcor_nodes),('source_target',piv_target_nodes)]:
        mask=matched&intervals(node);out[label+'_matched_interval_mask']=mask
        stats=pair_stats(of_u,piv_u,mask,width,L);put_pair(out,label+'_matched',stats)
        out[label+'_retained_fraction_of_matched_width']=np.divide(stats['of']['width_m'],paired['of']['width_m'],out=np.full(len(h),np.nan),where=paired['of']['width_m']>0)
    after={str(path):sha(path) for path in immutable}
    unchanged=before==after
    if not unchanged:raise RuntimeError('Image-only field artifact changed during held-out comparison.')

    base=HERE/('pair_%d_integral_comparison'%pair)
    np.savez_compressed(str(base)+'.npz',**out)
    summary=dict(pair=pair,passed=True,image_only_fields_unchanged=unchanged,
        immutable_image_only_hashes_before=before,immutable_image_only_hashes_after=after,
        comparison_script_sha256=sha(Path(__file__)),native_helper_sha256=sha(HERE/'native_piv.py'),
        native_mat_path=str(native.mat_path),native_mat_sha256=sha(native.mat_path),
        native_dataset_reads=['compVel/xPIV','compVel/zPIV','compVel/delta_x','compVel/delta_z','compVel/dcor','compVel/DX','compVel/DT','compVel/IW','compVel/GS'],
        dense_PIV_fields_read=False,refitting_performed=False,PIV_used_only_after_image_only_results_frozen=True,
        source_profile=str(directory/'integration_profile.npz'),depth_count=len(h),horizontal_samples=len(x),interval_count=len(width),
        target_width_m=L,DX=DX,DT=DT,physical_units_confirmed=False,
        primary_PIV_validity='Finite supplied native dx and -dz; every positive-weight bilinear corner source-underwater and in actual retained A image; query source also visible and underwater. No extrapolation, dcor threshold or target filter.',
        native_note='Native refers to supplied saved PIV grid; its values may include the source application postprocessing. No dense delta_x1/delta_z1 field is sampled.',
        integration_note='The frozen 2px endpoint/midpoint Simpson intervals are used unchanged. Q integrates u over horizontal width, without an arc-length factor. Missing intervals are excluded explicitly, never zero-filled or bridged.',
        comparison_note='PIV was held out of the image-only estimation. Both methods observe the same images; this comparison is not independent ground truth.',
        sign_of_difference='Image-only minus PIV',
        variable_joint_coverage_fraction=finite_summary(out['matched_coverage_fraction']),
        native_own_coverage_fraction=finite_summary(out['piv_own_coverage_fraction']),
        joint_common_depth_band_m=out['common_depth_band_m'].tolist(),joint_common_width_m=common_width,
        joint_common_fraction=common_width/L,joint_common_interval_count=int(common.sum()),joint_common_profile_available=bool(common_width>0),
        of_original_common_width_m=old_width,of_original_common_fraction=old_width/L,
        PIV_fully_supported_on_original_OF_common_depth_count=int(piv_on_old_available.sum()),
        original_OF_common_feasibility_note='PIV integral on the original OF fixed domain is NaN unless that entire same domain is supported at the depth. Its partial support is recorded only as a fraction.',
        matched_Q_difference_m2_per_s=finite_summary(out['matched_integral_difference_of_minus_piv_m2_per_s']),
        common_Q_difference_m2_per_s=finite_summary(out['common_integral_difference_of_minus_piv_m2_per_s']),
        matched_u_rms_difference_m_per_s=finite_summary(out['matched_rms_difference_m_per_s']),
        sample_PIV_available=int(piv_nodes.sum()),sample_primary_PIV_without_finite_dcor=int((piv_nodes&~piv_dcor_nodes).sum()),
        finite_dcor_sensitivity='Separate same-support curves restricted to all contributing native correlations finite; no score cutoff, not a calibrated confidence interval.',
        source_target_sensitivity='Separate same-support curves additionally requiring actual target endpoint visibility at native contributors and query.',
        elapsed_seconds=time.time()-started)
    Path(str(base)+'.json').write_text(json.dumps(summary,indent=2))
    keys=['matched_width_m','matched_coverage_fraction','of_matched_integral_m2_per_s','piv_matched_integral_m2_per_s',
          'of_matched_mean_u_m_per_s','piv_matched_mean_u_m_per_s','matched_integral_difference_of_minus_piv_m2_per_s',
          'matched_mean_absolute_difference_m_per_s','matched_rms_difference_m_per_s','matched_supported_component_count','matched_longest_gap_m',
          'of_common_integral_m2_per_s','piv_common_integral_m2_per_s','of_common_mean_u_m_per_s','piv_common_mean_u_m_per_s','common_available_at_depth',
          'of_original_covered_integral_m2_per_s','of_original_coverage_fraction','of_original_common_integral_m2_per_s',
          'piv_on_of_original_common_integral_m2_per_s','piv_on_of_original_common_available','piv_fraction_of_of_original_common_width',
          'piv_own_integral_m2_per_s','piv_own_coverage_fraction','piv_own_mean_u_m_per_s',
          'of_finite_dcor_matched_integral_m2_per_s','piv_finite_dcor_matched_integral_m2_per_s','finite_dcor_matched_coverage_fraction']
    with Path(str(base)+'.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['pair','depth_m','depth_cm','common_width_m','common_coverage_fraction','of_original_common_width_m']+keys)
        writer.writeheader()
        for j in range(len(h)):
            row=dict(pair=pair,depth_m=float(h[j]),depth_cm=float(h[j]*100),common_width_m=common_width,common_coverage_fraction=common_width/L,of_original_common_width_m=old_width)
            row.update({key:out[key][j].item() for key in keys});writer.writerow(row)
    print(json.dumps({key:summary[key] for key in ['pair','image_only_fields_unchanged','joint_common_width_m','joint_common_fraction','PIV_fully_supported_on_original_OF_common_depth_count','variable_joint_coverage_fraction','elapsed_seconds']},indent=2),flush=True)
    return out,summary

def self_test():
    width=np.array([2.,2.,2.,2.]);of=np.ones((3,9))*3;piv=np.ones((3,9))*2
    mask=np.array([[1,1,1,1],[1,0,0,1],[0,0,0,0]],bool)
    z=pair_stats(of,piv,mask,width,8.)
    tests=dict(identical_domains=equal(z['of']['width_m'],z['piv']['width_m']),
        known_integrals=equal(z['of']['integral_m2_per_s'],[24,12,np.nan]) and equal(z['piv']['integral_m2_per_s'],[16,8,np.nan]),
        gap_not_bridged=equal(z['of']['width_m'],[8,4,0]),
        constant_difference_RMS=equal(z['rms_difference_m_per_s'],[1,1,np.nan]),
        constant_difference_MAE=equal(z['mean_absolute_difference_m_per_s'],[1,1,np.nan]),
        disconnected_components=equal(z['of']['supported_component_count'],[1,2,0]),
        longest_gaps=equal(z['of']['longest_gap_m'],[0,4,8]))
    report=dict(passed=all(tests.values()),tests=tests,synthetic_only=True)
    (HERE/'integral_comparison_self_test.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pairs',nargs='+',type=int,choices=[80,100],default=[80,100]);p.add_argument('--self-test',action='store_true');a=p.parse_args()
    if a.self_test:self_test()
    else:
        for pair in a.pairs:compare(pair)
