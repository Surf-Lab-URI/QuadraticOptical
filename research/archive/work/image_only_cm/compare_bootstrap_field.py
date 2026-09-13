"""Fresh image-only ±64/±48 initialization sensitivity on identical query sets.

Only the alternative main model is fitted; no alternative reverse validation is
claimed. Integral differences use the primary accepted intervals without change.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import argparse,json,time
import numpy as np
from finalize_fields import ConservativeEvaluator,FastLocalField
from fit_fields import read_npz,file_hash,verify_image_only,atomic_npz,atomic_json
from integrate_profile import integrate_arrays

HERE=Path(__file__).resolve().parent


def stats(values):
    a=np.asarray(values);a=a[np.isfinite(a)]
    if not len(a):return {'finite_count':0}
    return dict(finite_count=int(len(a)),mean=float(a.mean()),median=float(np.median(a)),
        p95=float(np.percentile(a,95)),p99=float(np.percentile(a,99)),maximum=float(a.max()),
        above_point01=int(np.sum(a>.01)),above_point08=int(np.sum(a>.08)),
        above_point1=int(np.sum(a>.1)),above1=int(np.sum(a>1)),above1point5=int(np.sum(a>1.5)))


def compare(query,primary,alternative):
    d,g,ncc,share=alternative.evaluate(query,diagnostics=True,include_share=True)
    finite=np.isfinite(d).all(axis=1)&np.isfinite(g).all(axis=(1,2))
    dd=np.linalg.norm(d-primary['disp'],axis=1)
    gd=np.abs(g-primary['gradient'])
    with np.errstate(invalid='ignore'):
        vector_extra=finite&(dd<=1.5)
        accepted4=primary['accepted']&vector_extra
        gx4=primary['gradient_accepted_xx']&vector_extra&(gd[:,0,0]<=.08)
        gy4=primary['gradient_accepted_yy']&vector_extra&(gd[:,1,1]<=.08)
        alt_quality=finite&(ncc>=.6)&(share>=.95)
    return dict(alt_displacement=d,alt_gradient=g,alt_ncc=ncc,alt_good_patch_share=share,
        alt_finite=finite,displacement_difference_px=dd,absolute_gradient_difference=gd,
        accepted_with_fourth_alternative=accepted4,gradient_xx_with_fourth_alternative=gx4,
        gradient_yy_with_fourth_alternative=gy4,alt_ncc_and_patch_quality_pass=alt_quality,
        accepted_with_fourth_and_alt_quality=accepted4&alt_quality)


def summarize(q,primary,c):
    accepted=primary['accepted'];near=primary['depth']<40
    d=c['displacement_difference_px'];g=c['absolute_gradient_difference']
    worst=np.flatnonzero(accepted&np.isfinite(d));worst=worst[np.argsort(d[worst])[::-1]][:15]
    alt_ncc_pass=np.zeros(len(q),bool)
    alt_ncc_finite=np.isfinite(c['alt_ncc'])
    alt_ncc_pass[alt_ncc_finite]=c['alt_ncc'][alt_ncc_finite]>=.6
    return dict(query_count=len(q),primary_accepted_count=int(accepted.sum()),
        raw_displacement_difference=stats(d),accepted_displacement_difference=stats(d[accepted]),
        near_surface_accepted_displacement_difference=stats(d[accepted&near]),
        accepted_gradient_xx_difference=stats(g[primary['gradient_accepted_xx'],0,0]),
        accepted_gradient_yy_difference=stats(g[primary['gradient_accepted_yy'],1,1]),
        gradient_all_components_difference_on_vector_accepted=stats(g[accepted].reshape(-1)),
        accepted_nonfinite_alternative=int(np.sum(accepted&~c['alt_finite'])),
        accepted_alt_ncc_below_point6=int(np.sum(accepted&~alt_ncc_pass)),
        accepted_alt_good_patch_share_below_point95=int(np.sum(accepted&~(c['alt_good_patch_share']>=.95))),
        primary_gradient_xx_count=int(primary['gradient_accepted_xx'].sum()),
        primary_gradient_yy_count=int(primary['gradient_accepted_yy'].sum()),
        fourth_alternative_vector_count=int(c['accepted_with_fourth_alternative'].sum()),
        fourth_alternative_gradient_xx_count=int(c['gradient_xx_with_fourth_alternative'].sum()),
        fourth_alternative_gradient_yy_count=int(c['gradient_yy_with_fourth_alternative'].sum()),
        stricter_fourth_plus_alt_ncc_share_vector_count=int(c['accepted_with_fourth_and_alt_quality'].sum()),
        greatest_accepted_differences=[dict(index0=int(i),point=q[i].tolist(),depth=float(primary['depth'][i]),
            displacement_difference_px=float(d[i]),gradient_xx_difference=float(g[i,0,0]),
            gradient_yy_difference=float(g[i,1,1]),alt_ncc=float(c['alt_ncc'][i]),
            alt_good_patch_share=float(c['alt_good_patch_share'][i])) for i in worst])


def run(pair):
    started=time.time();directory=HERE/str(pair);alt_dir=directory/'bootstrap48'
    sources={k:directory/(k+'.npz') for k in ['inputs','main','affine','large_window','margin14','reverse','ptv_tracks','results','integration_profile']}
    sources.update(alt_inputs=alt_dir/'inputs.npz',alt_tracks=alt_dir/'ptv_tracks.npz',alt_main=alt_dir/'main.npz')
    hashes={k:file_hash(p) for k,p in sources.items()}
    assert hashes['inputs']==hashes['alt_inputs']
    primary=ConservativeEvaluator(directory);alt=FastLocalField(sources['alt_main'])
    verify_image_only(alt.data,'Bootstrap48 main')
    assert bool(alt.data['complete'])
    assert str(alt.data['input_sha256'])==hashes['inputs']
    assert str(alt.data['tracks_sha256'])==hashes['alt_tracks']
    assert np.array_equal(alt.points,primary.models['main'].points)
    a=read_npz(sources['results']);p=read_npz(sources['integration_profile'])
    grid=compare(a['query'],a,alt)
    report=dict(pair=pair,status='passed',image_only=True,supplied_velocity_used=False,
        gradient_difference_units='Pixel displacement per pixel coordinate (pixel/pixel); divide by DT for velocity-gradient differences in 1/s.',
        previous_velocity_fields_or_tracks_used=False,source_sha256=hashes,
        grid=summarize(a['query'],a,grid))
    arrays={'grid_query':a['query'],'grid_primary_displacement':a['disp'],'grid_primary_gradient':a['gradient'],
        'grid_primary_accepted':a['accepted'],'grid_primary_gradient_xx_accepted':a['gradient_accepted_xx'],
        'grid_primary_gradient_yy_accepted':a['gradient_accepted_yy']}
    arrays.update({'grid_'+k:v for k,v in grid.items()})
    x=p['horizontal_sample_x_px'];y=p['sample_query_y_px'];nh,nx=y.shape
    q=np.stack([np.broadcast_to(x,y.shape),y],axis=2).reshape(-1,2)
    accum={};pa={};reproduction_error=0.
    primary_keys=['disp','gradient','depth','accepted','gradient_accepted_xx','gradient_accepted_yy']
    for start in range(0,len(q),8*nx):
        stop=min(start+8*nx,len(q));r=primary.evaluate_conservative(q[start:stop]);c=compare(q[start:stop],r,alt)
        for k in primary_keys:pa.setdefault(k,[]).append(r[k])
        for k,v in c.items():accum.setdefault(k,[]).append(v)
        print('Bootstrap48 pair',pair,'integration queries',stop,'/',len(q),'seconds',round(time.time()-started,1),flush=True)
    pa={k:np.concatenate(v,axis=0) for k,v in pa.items()};c={k:np.concatenate(v,axis=0) for k,v in accum.items()}
    assert np.array_equal(pa['accepted'].reshape(nh,nx),p['sample_accepted'])
    saved_d=p['sample_displacement_px'].reshape(-1,2)
    assert np.array_equal(np.isfinite(saved_d),np.isfinite(pa['disp']))
    reproduction_error=float(np.nanmax(np.abs(saved_d-pa['disp'])))
    assert reproduction_error<1e-12
    report['integration_queries']=summarize(q,pa,c)
    report['primary_integration_displacement_reproduction_max_px']=reproduction_error
    arrays['integration_query']=q.reshape(nh,nx,2)
    arrays.update({'integration_primary_'+k:v.reshape((nh,nx)+v.shape[1:]) for k,v in pa.items()})
    arrays.update({'integration_'+k:v.reshape((nh,nx)+v.shape[1:]) for k,v in c.items()})
    alt_u=c['alt_displacement'][:,0].reshape(nh,nx)*float(p['DX'])/float(p['DT'])
    out=integrate_arrays(x,p['depth_m'],p['sample_primary_u_assumed_m_per_s'],p['sample_accepted'],
        alt_u[None],float(p['DX']),*p['common_depth_band_requested_m'])
    for k in ['interval_accepted','common_domain_interval_mask','common_domain_available_at_depth']:
        assert np.array_equal(out[k],p[k]),k
    cq=out['variant_covered_integral_assumed_m2_per_s'][0]
    common=out['variant_common_integral_assumed_m2_per_s'][0]
    dcq=cq-p['covered_segment_integral_assumed_m2_per_s']
    dcommon=common-p['common_domain_integral_assumed_m2_per_s']
    assert np.all(np.isfinite(cq[np.isfinite(p['covered_segment_integral_assumed_m2_per_s'])]))
    assert np.all(np.isfinite(common[np.isfinite(p['common_domain_integral_assumed_m2_per_s'])]))
    newok=c['accepted_with_fourth_alternative'].reshape(nh,nx)
    newinterval=newok[:,:-2:2]&newok[:,1::2]&newok[:,2::2]
    width=p['horizontal_interval_width_m']
    lost_width=np.sum((p['interval_accepted']&~newinterval)*width[None],axis=1)
    arrays.update(depth_m=p['depth_m'],horizontal_sample_x_px=x,
        primary_interval_accepted=p['interval_accepted'],primary_common_domain_interval_mask=p['common_domain_interval_mask'],
        primary_covered_width_m=p['covered_width_m'],primary_common_domain_width_m=p['common_domain_width_m'],
        primary_covered_integral_assumed_m2_per_s=p['covered_segment_integral_assumed_m2_per_s'],
        primary_common_integral_assumed_m2_per_s=p['common_domain_integral_assumed_m2_per_s'],
        alt_covered_integral_same_primary_domain_assumed_m2_per_s=cq,
        alt_common_integral_same_primary_domain_assumed_m2_per_s=common,
        covered_integral_delta_assumed_m2_per_s=dcq,common_integral_delta_assumed_m2_per_s=dcommon,
        interval_accepted_with_fourth_alternative=newinterval,
        lost_covered_width_with_fourth_alternative_m=lost_width)
    report['integrals']=dict(primary_intervals_used_unchanged=True,
        primary_nonempty_covered_depth_count=int(np.isfinite(p['covered_segment_integral_assumed_m2_per_s']).sum()),
        primary_available_common_depth_count=int(np.isfinite(p['common_domain_integral_assumed_m2_per_s']).sum()),
        alternative_missing_required_sample_depth_count=int(np.sum(np.isfinite(p['covered_segment_integral_assumed_m2_per_s'])&~np.isfinite(cq))),
        covered_abs_delta_assumed_m2_per_s=stats(np.abs(dcq)),common_abs_delta_assumed_m2_per_s=stats(np.abs(dcommon)),
        max_primary_covered_abs_integral_assumed_m2_per_s=float(np.nanmax(np.abs(p['covered_segment_integral_assumed_m2_per_s']))),
        max_primary_common_abs_integral_assumed_m2_per_s=float(np.nanmax(np.abs(p['common_domain_integral_assumed_m2_per_s']))),
        fourth_alternative_total_interval_losses=int(np.sum(p['interval_accepted']&~newinterval)),
        fourth_alternative_max_covered_width_loss_m=float(lost_width.max()),
        fourth_alternative_depths_losing_intervals=int(np.sum(lost_width>0)))
    report['interpretation']=[
        'This compares image-only bootstrap search ranges, not independent truth or uncertainty intervals.',
        'Both tracking branches and full-domain seed/main fits are fresh; only current image-derived artifacts are read.',
        'The alternative has no separate reverse/model-variant fits; alternative main diagnostic checks do not establish a separately accepted complete field.',
        'Fourth-alternative counts apply exactly the existing finite/displacement/diagonal-gradient sensitivity gates to the additional main field.',
        'Alt NCC>=0.6 and good-patch share>=0.95 are additional diagnostics, distinguished from the existing alternative-spread gates.',
        'Integral changes use identical primary intervals, with no missing-sample removal, filling, bridging or renormalization.',
        'SI quantities are conditional on DX in metres/pixel and DT in seconds.']
    assert hashes=={k:file_hash(p) for k,p in sources.items()}
    report['source_files_unchanged']=True;report['elapsed_seconds']=time.time()-started
    arrays.update(image_only=np.array(True),supplied_velocity_used=np.array(False),physical_units_confirmed=np.array(False),
        DX=p['DX'],DT=p['DT'],pair=np.array(pair),input_sha256=np.array(hashes['inputs']))
    atomic_npz(directory/'bootstrap_field_comparison.npz',**arrays)
    atomic_json(directory/'bootstrap_field_comparison.json',report)
    print(json.dumps(report,indent=2),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--pair',choices=['80','100'],required=True)
    run(parser.parse_args().pair)
