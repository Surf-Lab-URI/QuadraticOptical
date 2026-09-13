"""Independent bounded checks of bootstrap sensitivity arrays and sums."""
from pathlib import Path
import argparse,json
import numpy as np
from finalize_fields import FastLocalField
from fit_fields import read_npz,atomic_json

HERE=Path(__file__).resolve().parent


def run(pair):
    base=HERE/str(pair);z=read_npz(base/'bootstrap_field_comparison.npz')
    p=read_npz(base/'integration_profile.npz')
    assert np.array_equal(z['primary_interval_accepted'],p['interval_accepted'])
    assert np.array_equal(z['primary_common_domain_interval_mask'],p['common_domain_interval_mask'])
    u=z['integration_alt_displacement'][:,:,0]*float(z['DX'])/float(z['DT'])
    width=p['horizontal_interval_width_m'];sums=[];common=[]
    for i,row in enumerate(p['interval_accepted']):
        ids=np.flatnonzero(row)
        terms=np.array([width[j]*(u[i,2*j]+4*u[i,2*j+1]+u[i,2*j+2])/6 for j in ids])
        assert np.isfinite(terms).all()
        sums.append(terms.sum() if len(terms) else np.nan)
        ids=np.flatnonzero(p['common_domain_interval_mask'])
        terms=np.array([width[j]*(u[i,2*j]+4*u[i,2*j+1]+u[i,2*j+2])/6 for j in ids])
        common.append(terms.sum() if p['common_domain_available_at_depth'][i] else np.nan)
    covered_error=float(np.nanmax(np.abs(np.array(sums)-z['alt_covered_integral_same_primary_domain_assumed_m2_per_s'])))
    common_error=float(np.nanmax(np.abs(np.array(common)-z['alt_common_integral_same_primary_domain_assumed_m2_per_s'])))
    assert covered_error<1e-16 and common_error<1e-16
    ok=z['grid_primary_accepted'];delta=z['grid_displacement_difference_px']
    ids=np.flatnonzero(ok);ids=ids[np.argsort(delta[ids])[-20:]]
    q=z['grid_query'][ids];field=FastLocalField(base/'bootstrap48/main.npz')
    d,g=field.evaluate(q);fd=np.zeros_like(g);h=1e-4
    for axis in range(2):
        offset=np.zeros(2);offset[axis]=h
        fd[:,:,axis]=(field.evaluate(q+offset)[0]-field.evaluate(q-offset)[0])/(2*h)
    gradient_error=float(np.max(np.abs(fd-g)));assert gradient_error<1e-6
    for domain in ['grid','integration']:
        orig=z[domain+'_primary_accepted'];extra=z[domain+'_accepted_with_fourth_alternative']
        assert np.all(~extra|orig)
        assert np.array_equal(orig,extra)
        for component in ['xx','yy']:
            before=z[domain+'_primary_gradient_'+component+'_accepted'] if domain=='grid' else z[domain+'_primary_gradient_accepted_'+component]
            after=z[domain+'_gradient_'+component+'_with_fourth_alternative']
            assert np.array_equal(before,after)
    report=dict(pair=pair,status='passed',primary_domains_unchanged=True,
        independently_recomputed_covered_integral_max_abs_error_m2_per_s=covered_error,
        independently_recomputed_common_integral_max_abs_error_m2_per_s=common_error,
        derivative_check_points=int(len(q)),finite_difference_step_px=h,
        analytic_vs_finite_difference_gradient_max_abs_error_pixel_per_pixel=gradient_error,
        fourth_alternative_vector_and_accepted_diagonal_gradient_masks_unchanged=True,
        no_prediction_or_primary_results_modified=True)
    atomic_json(base/'bootstrap_field_independent_numeric_audit.json',report)
    print(json.dumps(report,indent=2));return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pair',choices=['80','100'],required=True)
    run(p.parse_args().pair)
