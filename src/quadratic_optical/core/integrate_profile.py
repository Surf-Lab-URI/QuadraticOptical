"""Surface-relative horizontal integration of NEW image-only velocity fields.

No prior results, supplied PIV velocity arrays or manual endpoints are read.
Missing intervals remain missing. Model envelopes are sensitivity, not CIs.
"""
import os
import json
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np

VARIANT_NAMES=['affine_radius13','quadratic_radius19','quadratic_margin14']

def horizontal_samples(width,interval_px=2.):
    if width<=0 or interval_px<=0:raise ValueError('Positive image width and interval required.')
    edges=np.r_[np.arange(-.5,width-.5,interval_px),width-.5]
    x=np.empty(2*len(edges)-1)
    x[::2]=edges;x[1::2]=(edges[:-1]+edges[1:])/2
    return x

def sum_supported(values,valid):
    """Integral of the selected segments; an empty observed domain is NaN."""
    out=np.sum(np.where(valid,values,0.),axis=-1)
    return np.where(np.any(valid,axis=-1),out,np.nan)

def sensitivity(primary,variants):
    all_finite=np.isfinite(primary)&np.isfinite(variants).all(axis=0)
    all_values=np.concatenate([primary[None],variants],axis=0)
    lower=np.full(primary.shape,np.nan);upper=lower.copy();delta=lower.copy()
    lower[all_finite]=np.min(all_values[:,all_finite],axis=0)
    upper[all_finite]=np.max(all_values[:,all_finite],axis=0)
    delta[all_finite]=np.max(np.abs(variants[:,all_finite]-primary[None,all_finite]),axis=0)
    return lower,upper,delta

def integrate_arrays(sample_x_px,depth_m,primary_u,accepted,variant_u,DX,
                     common_depth_min_m,common_depth_max_m):
    """Integrate array samples without interpolation, gap bridging or filling.

    x alternates left endpoint, midpoint, right endpoint, midpoint, ... .
    primary_u and accepted have shape (depth, x), variant_u (variant, depth, x).
    Velocities are already physical u in m/s under the assumed calibration.
    """
    x=np.asarray(sample_x_px,float);h=np.asarray(depth_m,float)
    u=np.asarray(primary_u,float);ok=np.asarray(accepted,bool);v=np.asarray(variant_u,float)
    if x.ndim!=1 or len(x)<3 or len(x)%2!=1:raise ValueError('Alternating endpoint/midpoint x array required.')
    if not np.all(np.diff(x)>0):raise ValueError('Horizontal coordinates must increase strictly.')
    if not np.allclose(x[1::2],(x[:-2:2]+x[2::2])/2,rtol=0,atol=1e-12):raise ValueError('Midpoints disagree with their intervals.')
    if u.shape!=(len(h),len(x)) or ok.shape!=u.shape or v.ndim!=3 or v.shape[1:]!=u.shape:raise ValueError('Sample-array shape mismatch.')
    if not np.all(np.diff(h)>0):raise ValueError('Depths must increase strictly.')
    if DX<=0:raise ValueError('Positive DX required.')
    edges=x[::2];width=np.diff(edges)*DX;L=float((edges[-1]-edges[0])*DX)
    valid_nodes=ok&np.isfinite(u)
    valid=valid_nodes[:,:-2:2]&valid_nodes[:,1::2]&valid_nodes[:,2::2]
    integral=width[None]*(u[:,:-2:2]+4*u[:,1::2]+u[:,2::2])/6
    integral=np.where(valid,integral,np.nan)
    covered=sum_supported(integral,valid)
    covered_width=np.sum(width[None]*valid,axis=1)
    fraction=covered_width/L
    mean=np.full(len(h),np.nan);nonempty=covered_width>0
    mean[nonempty]=covered[nonempty]/covered_width[nonempty]
    full=np.where(valid.all(axis=1),covered,np.nan)

    band=(h>=common_depth_min_m-1e-15)&(h<=common_depth_max_m+1e-15)
    if not band.any():raise ValueError('The declared common-depth band has no sampled depths.')
    common=np.all(valid[band],axis=0)
    all_depth_common=np.all(valid,axis=0)
    common_width=float(np.sum(width[common]))
    common_available=np.all(valid[:,common],axis=1)&(common_width>0)
    common_q=np.full(len(h),np.nan)
    if common_width>0:
        common_q[common_available]=np.sum(integral[common_available][:,common],axis=1)
    common_mean=common_q/common_width if common_width>0 else np.full(len(h),np.nan)

    # Each variant uses exactly the primary's accepted interval set. A missing
    # required variant sample invalidates its whole sum, not merely that segment.
    vfinite=np.isfinite(v)
    vvalid=vfinite[:,:,:-2:2]&vfinite[:,:,1::2]&vfinite[:,:,2::2]
    vint=width[None,None]*(v[:,:,:-2:2]+4*v[:,:,1::2]+v[:,:,2::2])/6
    vq=np.sum(np.where(valid[None],vint,0.),axis=2)
    vavailable=np.all(~valid[None]|vvalid,axis=2)&nonempty[None]
    vq[~vavailable]=np.nan
    vmean=np.full(vq.shape,np.nan);vmean[:,nonempty]=vq[:,nonempty]/covered_width[None,nonempty]
    vcq=np.full(vq.shape,np.nan)
    if common_width>0:
        vcavailable=np.all(vvalid[:,:,common],axis=2)&common_available[None]
        vsum=np.sum(vint[:,:,common],axis=2)
        vcq[vcavailable]=vsum[vcavailable]
    vcmean=vcq/common_width if common_width>0 else np.full(vq.shape,np.nan)
    lo,hi,delta=sensitivity(covered,vq);clo,chi,cdelta=sensitivity(common_q,vcq)

    components=np.sum(valid&~np.c_[np.zeros(len(h),bool),valid[:,:-1]],axis=1)
    longest_missing=[]
    for row in valid:
        longest=0.;current=0.
        for good,span in zip(row,width):
            current=0. if good else current+span
            longest=max(longest,current)
        longest_missing.append(longest)

    # Nested comparison: join pairs of fine intervals into coarse ones. Both
    # estimates use the SAME domain, requiring all five fine sample locations.
    n_pair=len(width)//2;coarse_valid=valid[:,:2*n_pair:2]&valid[:,1:2*n_pair:2]
    pair_width=width[:2*n_pair:2]+width[1:2*n_pair:2]
    qfine=integral[:,:2*n_pair:2]+integral[:,1:2*n_pair:2]
    # Generalized comparison only where two adjacent intervals have equal width.
    equal_width=np.isclose(width[:2*n_pair:2],width[1:2*n_pair:2],rtol=0,atol=1e-15)
    coarse_valid &= equal_width[None]
    qcoarse=pair_width[None]*(u[:,:4*n_pair:4]+4*u[:,2:4*n_pair:4]+u[:,4:4*n_pair+1:4])/6
    fine_same=sum_supported(qfine,coarse_valid);coarse_same=sum_supported(qcoarse,coarse_valid)

    return dict(depth_m=h,horizontal_sample_x_px=x,horizontal_interval_edges_px=edges,
        horizontal_interval_bounds_px=np.c_[edges[:-1],edges[1:]],horizontal_interval_width_m=width,
        target_full_width_m=np.array(L),sample_primary_u_assumed_m_per_s=u,sample_accepted=ok,
        sample_variant_u_assumed_m_per_s=v,interval_accepted=valid,
        interval_primary_integral_assumed_m2_per_s=integral,
        covered_segment_integral_assumed_m2_per_s=covered,covered_width_m=covered_width,
        coverage_fraction=fraction,covered_width_mean_u_assumed_m_per_s=mean,
        full_width_integral_assumed_m2_per_s=full,
        supported_component_count=components,longest_missing_run_m=np.array(longest_missing),
        common_depth_band_requested_m=np.array([common_depth_min_m,common_depth_max_m]),
        common_depth_band_actual_m=np.array([h[band].min(),h[band].max()]),common_depth_band_indices=np.flatnonzero(band),
        common_domain_interval_mask=common,common_domain_width_m=np.array(common_width),
        common_domain_fraction=np.array(common_width/L),common_domain_available_at_depth=common_available,
        common_domain_integral_assumed_m2_per_s=common_q,common_domain_mean_u_assumed_m_per_s=common_mean,
        all_requested_depth_common_interval_mask=all_depth_common,
        all_requested_depth_common_width_m=np.array(np.sum(width[all_depth_common])),
        variant_covered_integral_assumed_m2_per_s=vq,variant_covered_mean_u_assumed_m_per_s=vmean,
        variant_common_integral_assumed_m2_per_s=vcq,variant_common_mean_u_assumed_m_per_s=vcmean,
        covered_model_sensitivity_min_assumed_m2_per_s=lo,covered_model_sensitivity_max_assumed_m2_per_s=hi,
        covered_model_sensitivity_max_abs_delta_assumed_m2_per_s=delta,
        common_model_sensitivity_min_assumed_m2_per_s=clo,common_model_sensitivity_max_assumed_m2_per_s=chi,
        common_model_sensitivity_max_abs_delta_assumed_m2_per_s=cdelta,
        quadrature_comparison_interval_mask=coarse_valid,
        quadrature_comparison_width_m=np.sum(pair_width[None]*coarse_valid,axis=1),
        quadrature_fine_same_domain_assumed_m2_per_s=fine_same,
        quadrature_coarse_same_domain_assumed_m2_per_s=coarse_same,
        quadrature_coarse_minus_fine_assumed_m2_per_s=coarse_same-fine_same)

def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def atomic_npz(path,arrays):
    temp=path.with_name(path.name+'.tmp-'+str(os.getpid()))
    with temp.open('wb') as f:np.savez_compressed(f,**arrays)
    os.replace(str(temp),str(path))

def run(directory, requested_depth_m=.01, interval_px=2., depth_step_m=.0001, depth_batch=8,
        acceptance=None):
    """Evaluate and integrate horizontal paths below the local free surface."""
    if not np.isfinite([requested_depth_m, interval_px, depth_step_m]).all() or min(requested_depth_m, interval_px, depth_step_m) <= 0 or depth_batch < 1:
        raise ValueError('Positive finite depth/sampling and positive depth_batch are required.')
    started=time.time();directory=Path(directory).resolve();pair=directory.name
    # Only the new image-only evaluator is imported; it does not use prior fields.
    from .finalize_fields import ConservativeEvaluator
    # Integrate the same field that was reported: a profile screened by the
    # default rule would not match the results.npz it claims to summarise.
    if acceptance is None:
        summary_path=Path(directory)/'summary.json'
        if summary_path.exists():
            acceptance=json.loads(summary_path.read_text()).get('acceptance_profile')
    evaluator=ConservativeEvaluator(directory, requested_depth_m=requested_depth_m,
                                    acceptance=acceptance)
    inp=evaluator.inputs;DX=float(evaluator.DX);DT=float(evaluator.DT)
    if DX<=0 or DT<=0:raise ValueError('Positive calibration values required.')
    height,width=inp['va'].shape;surface=np.asarray(inp['surface_a'],float)
    if surface.shape!=(width,):raise ValueError('Surface must cover the image width.')
    if not np.array_equal(inp['origin0'],[0,0]):raise ValueError('Full-frame zero-origin geometry required.')
    maximum=float(requested_depth_m);common_min=20*DX
    if maximum < common_min:
        raise ValueError('Requested depth must include the conservative 20-pixel common-domain band.')
    depth=np.unique(np.r_[np.arange(0,maximum,depth_step_m),maximum,common_min])
    depth=depth[(depth>=0)&(depth<=maximum)]
    x=horizontal_samples(width,interval_px);sx=np.interp(x,np.arange(width),surface)
    uu=np.full((len(depth),len(x)),np.nan);accepted=np.zeros(uu.shape,bool)
    variants=np.full((3,)+uu.shape,np.nan);displacement=np.full(uu.shape+(2,),np.nan)
    for start in range(0,len(depth),depth_batch):
        stop=min(start+depth_batch,len(depth));dh=depth[start:stop]
        qx=np.broadcast_to(x,(len(dh),len(x)));qy=sx[None]+dh[:,None]/DX
        query=np.stack([qx,qy],axis=2).reshape(-1,2)
        result=evaluator.evaluate_conservative(query)
        d=result['disp'].reshape(len(dh),len(x),2)
        alt=result['alternative_displacements'].reshape(3,len(dh),len(x),2)
        displacement[start:stop]=d;uu[start:stop]=d[:,:,0]*DX/DT
        accepted[start:stop]=result['accepted'].reshape(len(dh),len(x))
        variants[:,start:stop]=alt[:,:,:,0]*DX/DT
        print('pair',pair,'surface-following depths',stop,'/',len(depth),'seconds',round(time.time()-started,1),flush=True)
    arrays=integrate_arrays(x,depth,uu,accepted,variants,DX,common_min,maximum)
    arrays.update(sample_displacement_px=displacement,sample_surface_y_px=sx,
        sample_query_y_px=sx[None]+depth[:,None]/DX,variant_names=np.array(VARIANT_NAMES),
        DX=np.array(DX),DT=np.array(DT),physical_units_confirmed=np.array(bool(inp.get('physical_units_confirmed', False))),
        surface_geometry_inferred=np.array(bool(inp.get('surface_geometry_inferred', False))),
        surface_trace_offset_px=np.array(float(inp.get('surface_trace_offset_px', np.nan))),
        image_shape=np.array([height,width]),pair=np.array(pair),
        interval_px=np.array(interval_px),depth_step_m=np.array(depth_step_m))
    atomic_npz(directory/'integration_profile.npz',arrays)
    hashes={name:file_hash(directory/(name+'.npz')) for name in ['inputs','main','affine','large_window','margin14','reverse','ptv_tracks']}
    metadata=dict(pair=pair,source_directory=str(directory),source_hashes=hashes,
        integration_script_sha256=file_hash(Path(__file__)),fresh_image_only_fields=True,
        earlier_velocity_fields_read=False,supplied_computed_velocity_arrays_read=False,manual_endpoints_read=False,
        formula='Q(h)=DX*integral u(x,s_A(x)+h/DX) dx; u=DX/DT*d_x; horizontal measure only, no arc-length factor.',
        coordinate_note='Zero-based image x right, y down; h increases vertically downward from the local inferred frame-A surface.',
        target_pixel_edge_interval=[-.5,width-.5],target_width_m=float(arrays['target_full_width_m']),
        depth_range_m=[float(depth.min()),float(depth.max())],depth_count=len(depth),horizontal_sample_count=len(x),
        horizontal_interval_count=(len(x)-1)//2,simpson_interval_px=interval_px,
        support_rule='Every endpoint and midpoint passes the new image-only conservative vector screen and has finite u. No gap bridging or zero-filled velocity.',
        covered_quantity_note='Covered integral, covered width, width fraction and covered-width mean are distinct. Empty covered domains have NaN integral/mean and zero width.',
        full_width_quantity_note='Full-width integral is NaN whenever any target interval is unsupported; no conditional extrapolation or hidden renormalization is produced.',
        all_depth_common_width_m=float(arrays['all_requested_depth_common_width_m']),
        common_depth_band_requested_m=arrays['common_depth_band_requested_m'].tolist(),
        common_depth_band_actual_m=arrays['common_depth_band_actual_m'].tolist(),
        common_domain_width_m=float(arrays['common_domain_width_m']),common_domain_fraction=float(arrays['common_domain_fraction']),
        common_domain_note='One fixed, potentially disconnected horizontal intersection across every declared-band depth; shallow unsupported depths remain unavailable.',
        model_sensitivity_note='Primary plus affine13, quadratic19 and margin14 extrema on identical primary support; model sensitivity, not confidence intervals or independent errors.',
        quadrature_note='Nested fine/coarse Simpson comparison uses identical unions of fully supported coarse intervals. It measures quadrature sensitivity separately from coverage.',
        units_note='SI values assume supplied DX metres/pixel and DT seconds; Q is m^2/s, multiply by1e4 for cm^2/s. Calibration units are not independently certified.',
        physical_units_confirmed=bool(inp.get('physical_units_confirmed', False)),
        surface_geometry_inferred=bool(inp.get('surface_geometry_inferred', False)),
        surface_offset_pixels=float(inp.get('surface_trace_offset_px', np.nan)),
        masked_near_surface_note='Image-masked or conservatively unsupported near-surface samples remain missing; h=0 is not filled.',
        full_width_supported_depth_count=int(np.isfinite(arrays['full_width_integral_assumed_m2_per_s']).sum()),
        elapsed_seconds=time.time()-started)
    (directory/'integration_metadata.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps({k:metadata[k] for k in ['pair','depth_count','horizontal_sample_count','all_depth_common_width_m','common_domain_width_m','common_domain_fraction','full_width_supported_depth_count','elapsed_seconds']}),flush=True)
    return arrays,metadata

def self_test():
    x=horizontal_samples(8,2);h=np.array([0.,.1,.2]);u=np.full((3,len(x)),2.);ok=np.ones(u.shape,bool)
    v=np.array([u-.1,u+.1,u]);r=integrate_arrays(x,h,u,ok,v,1.,.1,.2);checks={}
    checks['constant_integral_and_mean']=bool(np.allclose(r['covered_segment_integral_assumed_m2_per_s'],16)&np.allclose(r['covered_width_mean_u_assumed_m_per_s'],2))
    checks['constant_full_width']=bool(np.allclose(r['full_width_integral_assumed_m2_per_s'],16))
    checks['constant_sensitivity']=bool(np.allclose(r['covered_model_sensitivity_min_assumed_m2_per_s'],15.2)&np.allclose(r['covered_model_sensitivity_max_assumed_m2_per_s'],16.8))
    linear=np.broadcast_to(x+.5,u.shape).copy();lr=integrate_arrays(x,h,linear,ok,np.array([linear]*3),1.,.1,.2)
    checks['linear_integral_exact']=bool(np.allclose(lr['covered_segment_integral_assumed_m2_per_s'],32))
    signed=np.broadcast_to(x-3.5,u.shape).copy();sr=integrate_arrays(x,h,signed,ok,np.array([signed]*3),1.,.1,.2)
    checks['signed_cancellation_retained']=bool(np.allclose(sr['covered_segment_integral_assumed_m2_per_s'],0))
    gaps=ok.copy();gaps[0]=False;gaps[1,4]=False;gaps[2,2]=False
    gr=integrate_arrays(x,h,u,gaps,v,1.,.1,.2)
    checks['missing_intervals_not_bridged']=bool(np.array_equal(gr['interval_accepted'][1],[1,0,0,1]) and np.allclose(gr['covered_width_m'],[0,4,4]))
    checks['empty_integral_nan_not_zero']=bool(np.isnan(gr['covered_segment_integral_assumed_m2_per_s'][0]) and np.isnan(gr['covered_width_mean_u_assumed_m_per_s'][0]))
    checks['partial_domain_not_full_width']=bool(np.isnan(gr['full_width_integral_assumed_m2_per_s']).all())
    checks['fixed_common_domain_exact']=bool(np.array_equal(gr['common_domain_interval_mask'],[0,0,0,1]) and gr['common_domain_width_m']==2 and np.allclose(gr['common_domain_integral_assumed_m2_per_s'][1:],4) and np.isnan(gr['common_domain_integral_assumed_m2_per_s'][0]))
    checks['all_depth_common_empty']=bool(gr['all_requested_depth_common_width_m']==0)
    zero=np.zeros(u.shape);zr=integrate_arrays(x,h,zero,ok,np.array([zero]*3),1.,.1,.2)
    checks['observed_zero_is_zero']=bool(np.array_equal(zr['covered_segment_integral_assumed_m2_per_s'],np.zeros(3)))
    bad=v.copy();bad[0,1,0]=np.nan;br=integrate_arrays(x,h,u,gaps,bad,1.,.1,.2)
    checks['required_variant_nan_invalidates_sensitivity']=bool(np.isnan(br['variant_covered_integral_assumed_m2_per_s'][0,1]) and np.isnan(br['covered_model_sensitivity_min_assumed_m2_per_s'][1]))
    bad=v.copy();bad[0,1,4]=np.nan;br=integrate_arrays(x,h,u,gaps,bad,1.,.1,.2)
    checks['missing_variant_outside_primary_support_ignored']=bool(np.isfinite(br['variant_covered_integral_assumed_m2_per_s'][0,1]))
    cubic=np.broadcast_to((x+.5)**3,u.shape).copy();cr=integrate_arrays(x,h,cubic,ok,np.array([cubic]*3),1.,.1,.2)
    checks['nested_simpson_same_domain_exact_for_cubic']=bool(np.allclose(cr['quadrature_coarse_minus_fine_assumed_m2_per_s'],0) and np.allclose(cr['quadrature_comparison_width_m'],8))
    report=dict(passed=all(checks.values()),checks=checks,synthetic_only=True,prior_or_supplied_velocity_data_read=False)
    return report

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    p.add_argument('--requested-depth-m', type=float, default=.01)
    p.add_argument('--interval-px', type=float, default=2.)
    p.add_argument('--depth-step-m', type=float, default=.0001)
    p.add_argument('--depth-batch', type=int, default=8)
    run(**vars(p.parse_args()))
